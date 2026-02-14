//! Model download system with progress reporting and atomic installation.
//!
//! Downloads model files from `HuggingFace` with SHA-256 verification,
//! atomic installation (rename-over), and progress callbacks.
//!
//! Gated behind the `download` feature flag to keep the core crate network-free.

use std::fmt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};
use tracing::{debug, info, warn};

use asupersync::http::h1::{ClientError, HttpClient, HttpClientConfig, RedirectPolicy};
use frankensearch_core::error::{SearchError, SearchResult};

use crate::model_manifest::{ModelFile, ModelLifecycle, ModelManifest};

// ─── Configuration ──────────────────────────────────────────────────────────

/// Configuration for model downloads.
#[derive(Debug, Clone)]
pub struct DownloadConfig {
    /// Maximum retries per file on transient failure.
    pub max_retries: u32,
    /// Base delay for exponential backoff between retries.
    pub retry_base_delay: Duration,
    /// User-Agent header value.
    pub user_agent: String,
    /// Maximum redirects to follow.
    pub max_redirects: u32,
}

impl Default for DownloadConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            retry_base_delay: Duration::from_secs(1),
            user_agent: format!("frankensearch/{}", env!("CARGO_PKG_VERSION")),
            max_redirects: 5,
        }
    }
}

// ─── Progress ───────────────────────────────────────────────────────────────

/// Progress information for an in-flight model download.
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    /// Name of the file currently being downloaded.
    pub file_name: String,
    /// Bytes downloaded so far (current file).
    pub bytes_downloaded: u64,
    /// Total bytes expected (current file), if known.
    pub total_bytes: Option<u64>,
    /// Number of files completed so far.
    pub files_completed: usize,
    /// Total number of files to download.
    pub files_total: usize,
    /// Estimated download speed in bytes per second.
    pub speed_bytes_per_sec: f64,
    /// Estimated time remaining in seconds, if calculable.
    pub eta_seconds: Option<f64>,
}

impl fmt::Display for DownloadProgress {
    #[allow(clippy::cast_precision_loss)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pct = self
            .total_bytes
            .filter(|&t| t > 0)
            .map(|t| self.bytes_downloaded as f64 / t as f64 * 100.0);

        if let Some(pct) = pct {
            write!(
                f,
                "[{}/{}] {} {:.0}% ({}/{})",
                self.files_completed + 1,
                self.files_total,
                self.file_name,
                pct,
                format_bytes(self.bytes_downloaded),
                format_bytes(self.total_bytes.unwrap_or(0)),
            )
        } else {
            write!(
                f,
                "[{}/{}] {} {}",
                self.files_completed + 1,
                self.files_total,
                self.file_name,
                format_bytes(self.bytes_downloaded),
            )
        }
    }
}

// ─── Downloader ─────────────────────────────────────────────────────────────

/// Downloads model files from `HuggingFace` with verification and progress reporting.
pub struct ModelDownloader {
    config: DownloadConfig,
    client: HttpClient,
}

impl ModelDownloader {
    /// Create a new downloader with the given configuration.
    #[must_use]
    pub fn new(config: DownloadConfig) -> Self {
        let client_config = HttpClientConfig {
            redirect_policy: RedirectPolicy::Limited(config.max_redirects),
            user_agent: Some(config.user_agent.clone()),
            ..HttpClientConfig::default()
        };
        Self {
            config,
            client: HttpClient::with_config(client_config),
        }
    }

    /// Create a downloader with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(DownloadConfig::default())
    }

    /// Download all files for a model manifest into a staging directory.
    ///
    /// The staging directory is created as `{dest_dir}/.download/` and files
    /// are placed there during download. After all files are verified, the
    /// caller should use [`ModelManifest::promote_verified_installation`] to
    /// atomically install the model.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` on network failure, hash mismatch, or I/O error.
    pub async fn download_model(
        &self,
        manifest: &ModelManifest,
        dest_dir: &Path,
        lifecycle: &mut ModelLifecycle,
        on_progress: impl Fn(&DownloadProgress) + Send + Sync,
    ) -> SearchResult<PathBuf> {
        let staging_dir = dest_dir.join(".download");
        std::fs::create_dir_all(&staging_dir).map_err(SearchError::from)?;

        let total_bytes = manifest.total_size_bytes();
        lifecycle.begin_download(total_bytes.max(1))?;

        let files_total = manifest.files.len();
        let mut cumulative_bytes: u64 = 0;

        for (idx, file) in manifest.files.iter().enumerate() {
            let url = huggingface_url(&manifest.repo, &manifest.revision, &file.name);
            let file_dest = staging_dir.join(&file.name);

            // Create parent directories for nested paths (e.g., "onnx/model.onnx").
            if let Some(parent) = file_dest.parent() {
                std::fs::create_dir_all(parent).map_err(SearchError::from)?;
            }

            info!(
                file = %file.name,
                size = file.size,
                url = %url,
                "downloading model file"
            );

            self.download_file_with_retry(&url, &file_dest, file, idx, files_total, &on_progress)
                .await?;

            cumulative_bytes = cumulative_bytes.saturating_add(file.size);
            lifecycle.update_download_progress(cumulative_bytes)?;
        }

        // Verify all files.
        lifecycle.begin_verification()?;
        info!(model = %manifest.id, "verifying downloaded files");
        match manifest.verify_dir(&staging_dir) {
            Ok(()) => {
                lifecycle.mark_ready();
                info!(model = %manifest.id, "model download complete and verified");
                Ok(staging_dir)
            }
            Err(e) => {
                lifecycle.fail_verification(e.to_string());
                Err(e)
            }
        }
    }

    /// Download a single file with retry logic.
    async fn download_file_with_retry(
        &self,
        url: &str,
        dest: &Path,
        file: &ModelFile,
        file_idx: usize,
        files_total: usize,
        on_progress: &(impl Fn(&DownloadProgress) + Send + Sync),
    ) -> SearchResult<()> {
        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                let delay = self.config.retry_base_delay * 2_u32.saturating_pow(attempt - 1);
                warn!(
                    file = %file.name,
                    attempt,
                    delay_ms = delay.as_millis(),
                    "retrying download after failure"
                );
                std::thread::sleep(delay);
            }

            match self
                .download_single_file(url, dest, file, file_idx, files_total, on_progress)
                .await
            {
                Ok(()) => return Ok(()),
                Err(e) => {
                    warn!(
                        file = %file.name,
                        attempt,
                        error = %e,
                        "download attempt failed"
                    );
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| SearchError::ModelLoadFailed {
            path: dest.to_path_buf(),
            source: "download failed after all retries".into(),
        }))
    }

    /// Download a single file (one attempt).
    #[allow(clippy::cast_precision_loss)]
    async fn download_single_file(
        &self,
        url: &str,
        dest: &Path,
        file: &ModelFile,
        file_idx: usize,
        files_total: usize,
        on_progress: &(impl Fn(&DownloadProgress) + Send + Sync),
    ) -> SearchResult<()> {
        let start = Instant::now();

        // Report start.
        on_progress(&DownloadProgress {
            file_name: file.name.clone(),
            bytes_downloaded: 0,
            total_bytes: if file.size > 0 { Some(file.size) } else { None },
            files_completed: file_idx,
            files_total,
            speed_bytes_per_sec: 0.0,
            eta_seconds: None,
        });

        // Download into memory using the high-level HTTP client.
        let response = self
            .client
            .get(url)
            .await
            .map_err(|e| client_error_to_search(e, url))?;

        // Check HTTP status.
        if response.status < 200 || response.status >= 300 {
            return Err(SearchError::ModelLoadFailed {
                path: dest.to_path_buf(),
                source: format!("HTTP {} {} for {url}", response.status, response.reason).into(),
            });
        }

        let body = &response.body;
        let elapsed = start.elapsed();
        let speed = if elapsed.as_secs_f64() > 0.0 {
            body.len() as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        // Report completion.
        on_progress(&DownloadProgress {
            file_name: file.name.clone(),
            bytes_downloaded: body.len() as u64,
            total_bytes: Some(body.len() as u64),
            files_completed: file_idx,
            files_total,
            speed_bytes_per_sec: speed,
            eta_seconds: Some(0.0),
        });

        debug!(
            file = %file.name,
            bytes = body.len(),
            elapsed_ms = elapsed.as_millis(),
            speed_mbps = format_args!("{:.1}", speed / 1_048_576.0),
            "file downloaded"
        );

        // Verify SHA-256 in memory before writing (if checksum is not placeholder).
        if file.has_verified_checksum() {
            let actual_hash = sha256_hex(body);
            if actual_hash != file.sha256 {
                return Err(SearchError::HashMismatch {
                    path: dest.to_path_buf(),
                    expected: format!("sha256={},size={}", file.sha256, file.size),
                    actual: format!("sha256={actual_hash},size={}", body.len()),
                });
            }

            if body.len() as u64 != file.size {
                return Err(SearchError::HashMismatch {
                    path: dest.to_path_buf(),
                    expected: format!("size={}", file.size),
                    actual: format!("size={}", body.len()),
                });
            }
        }

        // Write atomically: write to .tmp then rename.
        let tmp_path = dest.with_extension("tmp");
        std::fs::write(&tmp_path, body).map_err(SearchError::from)?;
        std::fs::rename(&tmp_path, dest).map_err(SearchError::from)?;

        info!(
            file = %file.name,
            bytes = body.len(),
            elapsed_ms = elapsed.as_millis(),
            "file saved"
        );

        Ok(())
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Build a `HuggingFace` CDN URL for a model file.
fn huggingface_url(repo: &str, revision: &str, file_name: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/{revision}/{file_name}")
}

/// Compute lowercase hex SHA-256 of a byte slice.
fn sha256_hex(data: &[u8]) -> String {
    let hash = Sha256::digest(data);
    let mut out = String::with_capacity(64);
    for byte in &hash {
        use std::fmt::Write;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

/// Format bytes as a human-readable string.
#[allow(clippy::cast_precision_loss)]
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * 1024;
    const GB: u64 = 1024 * 1024 * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Convert asupersync `ClientError` to `SearchError`.
fn client_error_to_search(error: ClientError, url: &str) -> SearchError {
    SearchError::ModelLoadFailed {
        path: PathBuf::from(url),
        source: Box::new(error),
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn huggingface_url_format() {
        let url = huggingface_url(
            "sentence-transformers/all-MiniLM-L6-v2",
            "abc123",
            "onnx/model.onnx",
        );
        assert_eq!(
            url,
            "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/abc123/onnx/model.onnx"
        );
    }

    #[test]
    fn sha256_hex_known_value() {
        let hash = sha256_hex(b"hello world");
        assert_eq!(
            hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn format_bytes_units() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1_048_576), "1.0 MB");
        assert_eq!(format_bytes(1_073_741_824), "1.0 GB");
    }

    #[test]
    fn download_config_defaults() {
        let config = DownloadConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.retry_base_delay, Duration::from_secs(1));
        assert_eq!(config.max_redirects, 5);
        assert!(config.user_agent.starts_with("frankensearch/"));
    }

    #[test]
    fn download_progress_display_with_total() {
        let progress = DownloadProgress {
            file_name: "model.onnx".to_owned(),
            bytes_downloaded: 524_288,
            total_bytes: Some(1_048_576),
            files_completed: 0,
            files_total: 3,
            speed_bytes_per_sec: 1_048_576.0,
            eta_seconds: Some(0.5),
        };
        let display = progress.to_string();
        assert!(display.contains("[1/3]"));
        assert!(display.contains("model.onnx"));
        assert!(display.contains("50%"));
    }

    #[test]
    fn download_progress_display_without_total() {
        let progress = DownloadProgress {
            file_name: "config.json".to_owned(),
            bytes_downloaded: 1024,
            total_bytes: None,
            files_completed: 2,
            files_total: 3,
            speed_bytes_per_sec: 0.0,
            eta_seconds: None,
        };
        let display = progress.to_string();
        assert!(display.contains("[3/3]"));
        assert!(display.contains("config.json"));
        assert!(display.contains("1.0 KB"));
    }

    #[test]
    fn client_error_converts_to_search_error() {
        let err = client_error_to_search(
            ClientError::InvalidUrl("bad".to_owned()),
            "https://example.com",
        );
        assert!(matches!(err, SearchError::ModelLoadFailed { .. }));
    }
}
