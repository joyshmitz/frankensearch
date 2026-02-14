use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use frankensearch_core::{SearchError, SearchResult};
use fsqlite_core::raptorq_integration::{DecodeFailureReason, SymbolCodec};
use serde::Serialize;
use tracing::{debug, info, warn};

use crate::codec::{CodecFacade, DecodedPayload};
use crate::config::DurabilityConfig;
use crate::metrics::{DurabilityMetrics, DurabilityMetricsSnapshot};
use crate::repair_trailer::{
    RepairSymbol, RepairTrailerHeader, deserialize_repair_trailer, serialize_repair_trailer,
};

/// Result produced after writing a durability sidecar.
#[derive(Debug, Clone)]
pub struct FileProtectionResult {
    pub sidecar_path: PathBuf,
    pub source_len: u64,
    pub source_crc32: u32,
    /// Number of source symbols the file was split into.
    pub k_source: u32,
    pub repair_symbol_count: u32,
}

/// Verification status for a payload+sidecar pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileVerifyResult {
    pub healthy: bool,
    pub expected_crc32: u32,
    pub actual_crc32: u32,
    pub expected_len: u64,
    pub actual_len: u64,
}

/// Repair outcome for a file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileRepairOutcome {
    NotNeeded,
    Repaired {
        bytes_written: usize,
        symbols_used: u32,
    },
    Unrecoverable {
        reason: DecodeFailureReason,
        symbols_received: u32,
        k_required: u32,
    },
}

/// Health status for a single file after verify-and-repair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileHealth {
    /// File integrity confirmed; no action needed.
    Intact,
    /// Corruption was detected and successfully repaired.
    Repaired {
        /// Number of bytes written during repair.
        bytes_written: usize,
        /// Wall-clock time for the repair operation.
        repair_time: Duration,
    },
    /// Corruption was detected but repair failed.
    Unrecoverable {
        /// Explanation of why repair failed.
        reason: String,
    },
    /// No `.fec` sidecar exists for this file.
    Unprotected,
}

/// Result of a single-file verify-and-repair pipeline.
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    /// Path to the checked file.
    pub path: PathBuf,
    /// Health status after check (and optional repair).
    pub status: FileHealth,
}

/// Report produced after protecting all files in a directory.
#[derive(Debug, Clone)]
pub struct DirectoryProtectionReport {
    /// Number of files newly protected.
    pub files_protected: usize,
    /// Number of files already protected (skipped).
    pub files_already_protected: usize,
    /// Total source bytes across newly protected files.
    pub total_source_bytes: u64,
    /// Total repair sidecar bytes generated.
    pub total_repair_bytes: u64,
    /// Wall-clock time for the protection pass.
    pub elapsed: Duration,
}

/// Report produced after verifying all files in a directory.
#[derive(Debug, Clone)]
pub struct DirectoryHealthReport {
    /// Per-file health check results.
    pub results: Vec<HealthCheckResult>,
    /// Number of intact files.
    pub intact_count: usize,
    /// Number of repaired files.
    pub repaired_count: usize,
    /// Number of unrecoverable files.
    pub unrecoverable_count: usize,
    /// Number of unprotected files (no sidecar).
    pub unprotected_count: usize,
    /// Wall-clock time for the full check.
    pub elapsed: Duration,
}

/// JSONL repair event record, appended to the repair log file.
#[derive(Debug, Serialize)]
struct RepairEvent {
    timestamp: String,
    path: String,
    corrupted: bool,
    repair_succeeded: bool,
    bytes_written: usize,
    source_crc32_expected: u32,
    source_crc32_after: u32,
    repair_time_ms: u64,
}

/// Abstract durability provider with no-op defaults.
///
/// When the `durability` feature is disabled at compile time, consumers can
/// use [`NoopDurability`] which satisfies this trait with zero overhead.
pub trait DurabilityProvider: Send + Sync {
    /// Protect a file by generating a `.fec` sidecar.
    fn protect(&self, path: &Path) -> SearchResult<FileProtectionResult> {
        let _ = path;
        Ok(FileProtectionResult {
            sidecar_path: PathBuf::new(),
            source_len: 0,
            source_crc32: 0,
            k_source: 0,
            repair_symbol_count: 0,
        })
    }

    /// Verify a file's integrity using its sidecar.
    fn verify(&self, path: &Path) -> SearchResult<FileVerifyResult> {
        let _ = path;
        Ok(FileVerifyResult {
            healthy: true,
            expected_crc32: 0,
            actual_crc32: 0,
            expected_len: 0,
            actual_len: 0,
        })
    }

    /// Attempt to repair a corrupted file.
    fn repair(&self, path: &Path) -> SearchResult<FileRepairOutcome> {
        let _ = path;
        Err(SearchError::DurabilityDisabled)
    }

    /// Verify and optionally repair a single file.
    fn check_health(&self, path: &Path) -> SearchResult<HealthCheckResult> {
        let _ = path;
        Ok(HealthCheckResult {
            path: PathBuf::new(),
            status: FileHealth::Unprotected,
        })
    }

    /// Protect all protectable files in a directory.
    fn protect_directory(&self, dir: &Path) -> SearchResult<DirectoryProtectionReport> {
        let _ = dir;
        Ok(DirectoryProtectionReport {
            files_protected: 0,
            files_already_protected: 0,
            total_source_bytes: 0,
            total_repair_bytes: 0,
            elapsed: Duration::ZERO,
        })
    }

    /// Verify (and auto-repair) all protected files in a directory.
    fn verify_directory(&self, dir: &Path) -> SearchResult<DirectoryHealthReport> {
        let _ = dir;
        Ok(DirectoryHealthReport {
            results: Vec::new(),
            intact_count: 0,
            repaired_count: 0,
            unrecoverable_count: 0,
            unprotected_count: 0,
            elapsed: Duration::ZERO,
        })
    }

    /// Get a metrics snapshot.
    fn metrics_snapshot(&self) -> DurabilityMetricsSnapshot;
}

/// No-op durability provider for when the feature is disabled.
#[derive(Debug, Default)]
pub struct NoopDurability;

impl DurabilityProvider for NoopDurability {
    fn metrics_snapshot(&self) -> DurabilityMetricsSnapshot {
        DurabilityMetricsSnapshot {
            encoded_bytes_total: 0,
            source_symbols_total: 0,
            repair_symbols_total: 0,
            decoded_bytes_total: 0,
            decode_symbols_used_total: 0,
            decode_symbols_received_total: 0,
            decode_k_required_total: 0,
            encode_ops: 0,
            decode_ops: 0,
            decode_failures: 0,
            decode_failures_recoverable: 0,
            decode_failures_unrecoverable: 0,
            encode_latency_us_total: 0,
            decode_latency_us_total: 0,
            repair_attempts: 0,
            repair_successes: 0,
            repair_failures: 0,
        }
    }
}

/// Configuration for the repair pipeline.
#[derive(Debug, Clone)]
pub struct RepairPipelineConfig {
    /// Whether to verify indices on load.
    pub verify_on_open: bool,
    /// Whether to generate `.fec` after index write.
    pub protect_on_write: bool,
    /// Whether to attempt repair when corruption detected.
    pub auto_repair: bool,
    /// Optional directory for JSONL repair event logs.
    pub repair_log_dir: Option<PathBuf>,
    /// Maximum repair log entries before rotation.
    pub max_repair_log_entries: usize,
}

impl Default for RepairPipelineConfig {
    fn default() -> Self {
        Self {
            verify_on_open: true,
            protect_on_write: true,
            auto_repair: true,
            repair_log_dir: None,
            max_repair_log_entries: 1000,
        }
    }
}

/// File-level protect/verify/repair orchestrator.
#[derive(Debug, Clone)]
pub struct FileProtector {
    codec: CodecFacade,
    metrics: Arc<DurabilityMetrics>,
    pipeline_config: RepairPipelineConfig,
}

impl FileProtector {
    pub fn new(codec: Arc<dyn SymbolCodec>, config: DurabilityConfig) -> SearchResult<Self> {
        let metrics = Arc::new(DurabilityMetrics::default());
        Self::new_with_metrics(codec, config, metrics)
    }

    /// Create a `FileProtector` sharing an externally-owned metrics instance.
    pub fn new_with_metrics(
        codec: Arc<dyn SymbolCodec>,
        config: DurabilityConfig,
        metrics: Arc<DurabilityMetrics>,
    ) -> SearchResult<Self> {
        let codec = CodecFacade::new(codec, config, Arc::clone(&metrics))?;
        Ok(Self {
            codec,
            metrics,
            pipeline_config: RepairPipelineConfig::default(),
        })
    }

    /// Create a `FileProtector` with full pipeline configuration.
    pub fn new_with_pipeline_config(
        codec: Arc<dyn SymbolCodec>,
        config: DurabilityConfig,
        metrics: Arc<DurabilityMetrics>,
        pipeline_config: RepairPipelineConfig,
    ) -> SearchResult<Self> {
        let codec = CodecFacade::new(codec, config, Arc::clone(&metrics))?;
        Ok(Self {
            codec,
            metrics,
            pipeline_config,
        })
    }

    /// Access the pipeline configuration.
    pub fn pipeline_config(&self) -> &RepairPipelineConfig {
        &self.pipeline_config
    }

    pub fn metrics_snapshot(&self) -> DurabilityMetricsSnapshot {
        self.metrics.snapshot()
    }

    pub fn sidecar_path(path: &Path) -> PathBuf {
        PathBuf::from(format!("{}.fec", path.display()))
    }

    pub fn protect_file(&self, path: &Path) -> SearchResult<FileProtectionResult> {
        let source_bytes = fs::read(path)?;
        let encoded = self.codec.encode(&source_bytes)?;

        let repair_symbol_count = u32::try_from(encoded.repair_symbols.len()).map_err(|_| {
            SearchError::InvalidConfig {
                field: "repair_symbol_count".to_owned(),
                value: encoded.repair_symbols.len().to_string(),
                reason: "repair symbol count exceeds u32".to_owned(),
            }
        })?;
        let header = RepairTrailerHeader {
            symbol_size: encoded.symbol_size,
            k_source: encoded.k_source,
            source_len: encoded.source_len,
            source_crc32: encoded.source_crc32,
            repair_symbol_count,
        };

        let repair_symbols: Vec<RepairSymbol> = encoded
            .repair_symbols
            .into_iter()
            .map(|(esi, data)| RepairSymbol { esi, data })
            .collect();
        let trailer = serialize_repair_trailer(&header, &repair_symbols)?;

        let sidecar_path = Self::sidecar_path(path);
        fs::write(&sidecar_path, trailer)?;

        info!(
            path = %path.display(),
            sidecar = %sidecar_path.display(),
            repair_symbols = repair_symbol_count,
            "durability sidecar written"
        );

        Ok(FileProtectionResult {
            sidecar_path,
            source_len: header.source_len,
            source_crc32: header.source_crc32,
            k_source: header.k_source,
            repair_symbol_count: header.repair_symbol_count,
        })
    }

    pub fn verify_file(&self, path: &Path, sidecar_path: &Path) -> SearchResult<FileVerifyResult> {
        let source_bytes = fs::read(path)?;
        let trailer_bytes = fs::read(sidecar_path)?;
        let (header, _) = deserialize_repair_trailer(&trailer_bytes)?;

        let actual_crc32 = crc32fast::hash(&source_bytes);
        let actual_len = saturating_u64(source_bytes.len());
        let healthy = actual_crc32 == header.source_crc32 && actual_len == header.source_len;

        Ok(FileVerifyResult {
            healthy,
            expected_crc32: header.source_crc32,
            actual_crc32,
            expected_len: header.source_len,
            actual_len,
        })
    }

    #[allow(clippy::too_many_lines)]
    pub fn repair_file(&self, path: &Path, sidecar_path: &Path) -> SearchResult<FileRepairOutcome> {
        self.metrics.record_repair_attempt();

        let mut source_bytes = Vec::new();
        match self.verify_file(path, sidecar_path) {
            Ok(verify) => {
                if verify.healthy {
                    return Ok(FileRepairOutcome::NotNeeded);
                }
                source_bytes = fs::read(path)?;
            }
            Err(SearchError::Io(error)) if error.kind() == ErrorKind::NotFound => {
                // Missing source file is recoverable if we have sufficient sidecar symbols.
            }
            Err(error) => return Err(error),
        }

        let trailer_bytes = fs::read(sidecar_path)?;
        let (header, trailer_symbols) = deserialize_repair_trailer(&trailer_bytes)?;

        let repair_symbols: Vec<(u32, Vec<u8>)> = trailer_symbols
            .into_iter()
            .map(|symbol| (symbol.esi, symbol.data))
            .collect();

        let mut symbols =
            source_symbols_from_bytes(&source_bytes, header.symbol_size, header.k_source)?;
        symbols.extend(repair_symbols.iter().cloned());

        match self.codec.decode(&symbols, header.k_source)? {
            DecodedPayload::Success {
                data, symbols_used, ..
            } => {
                let data = normalize_recovered_data(data, &header)?;
                let recovered_crc32 = crc32fast::hash(&data);
                if recovered_crc32 != header.source_crc32 && !source_bytes.is_empty() {
                    warn!(
                        path = %path.display(),
                        expected_crc32 = header.source_crc32,
                        recovered_crc32,
                        "decoded payload failed crc verification; retrying with repair symbols only"
                    );

                    match self.codec.decode(&repair_symbols, header.k_source)? {
                        DecodedPayload::Success {
                            data, symbols_used, ..
                        } => {
                            let data = normalize_recovered_data(data, &header)?;
                            let recovered_crc32 = crc32fast::hash(&data);
                            if recovered_crc32 == header.source_crc32 {
                                fs::write(path, &data)?;
                                self.metrics.record_repair_success();
                                info!(
                                    path = %path.display(),
                                    bytes_written = data.len(),
                                    symbols_used,
                                    "durability repair completed (repair symbols only)"
                                );
                                return Ok(FileRepairOutcome::Repaired {
                                    bytes_written: data.len(),
                                    symbols_used,
                                });
                            }

                            self.metrics.record_repair_failure();
                            warn!(
                                path = %path.display(),
                                expected_crc32 = header.source_crc32,
                                recovered_crc32,
                                "repair-only decode failed crc verification"
                            );
                            return Ok(FileRepairOutcome::Unrecoverable {
                                reason: DecodeFailureReason::SymbolSizeMismatch,
                                symbols_received: u32::try_from(repair_symbols.len())
                                    .unwrap_or(u32::MAX),
                                k_required: header.k_source,
                            });
                        }
                        DecodedPayload::Failure {
                            reason,
                            symbols_received,
                            k_required,
                            ..
                        } => {
                            self.metrics.record_repair_failure();
                            warn!(
                                path = %path.display(),
                                ?reason,
                                symbols_received,
                                k_required,
                                "repair-only decode failed"
                            );
                            return Ok(FileRepairOutcome::Unrecoverable {
                                reason,
                                symbols_received,
                                k_required,
                            });
                        }
                    }
                }

                if recovered_crc32 != header.source_crc32 {
                    self.metrics.record_repair_failure();
                    warn!(
                        path = %path.display(),
                        expected_crc32 = header.source_crc32,
                        recovered_crc32,
                        "decoded payload failed crc verification"
                    );
                    return Ok(FileRepairOutcome::Unrecoverable {
                        reason: DecodeFailureReason::SymbolSizeMismatch,
                        symbols_received: u32::try_from(symbols.len()).unwrap_or(u32::MAX),
                        k_required: header.k_source,
                    });
                }

                fs::write(path, &data)?;
                self.metrics.record_repair_success();
                info!(
                    path = %path.display(),
                    bytes_written = data.len(),
                    symbols_used,
                    "durability repair completed"
                );
                Ok(FileRepairOutcome::Repaired {
                    bytes_written: data.len(),
                    symbols_used,
                })
            }
            DecodedPayload::Failure {
                reason,
                symbols_received,
                k_required,
                ..
            } => {
                self.metrics.record_repair_failure();
                warn!(
                    path = %path.display(),
                    ?reason,
                    symbols_received,
                    k_required,
                    "durability repair could not recover file"
                );
                Ok(FileRepairOutcome::Unrecoverable {
                    reason,
                    symbols_received,
                    k_required,
                })
            }
        }
    }

    /// Single-file verify-and-repair pipeline with backup-before-repair.
    ///
    /// Steps:
    /// 1. Check for `.fec` sidecar — if missing, return `Unprotected`.
    /// 2. Verify file integrity via CRC.
    /// 3. If corrupted and `auto_repair` is enabled, back up the corrupted file,
    ///    attempt repair, verify the result, and clean up or restore the backup.
    /// 4. Log the repair event if `repair_log_dir` is configured.
    pub fn verify_and_repair_file(&self, path: &Path) -> SearchResult<HealthCheckResult> {
        let sidecar = Self::sidecar_path(path);
        if !sidecar.exists() {
            return Ok(HealthCheckResult {
                path: path.to_path_buf(),
                status: FileHealth::Unprotected,
            });
        }

        match self.verify_file(path, &sidecar) {
            Ok(verify) if verify.healthy => {
                return Ok(HealthCheckResult {
                    path: path.to_path_buf(),
                    status: FileHealth::Intact,
                });
            }
            Ok(verify) => {
                debug!(
                    path = %path.display(),
                    expected_crc32 = verify.expected_crc32,
                    actual_crc32 = verify.actual_crc32,
                    "corruption detected"
                );
            }
            Err(SearchError::Io(ref e)) if e.kind() == ErrorKind::NotFound => {
                // Source file missing entirely — still try repair if auto_repair enabled.
                debug!(
                    path = %path.display(),
                    "source file missing, will attempt repair from sidecar"
                );
            }
            Err(e) => return Err(e),
        }

        if !self.pipeline_config.auto_repair {
            return Ok(HealthCheckResult {
                path: path.to_path_buf(),
                status: FileHealth::Unrecoverable {
                    reason: "auto_repair is disabled".to_owned(),
                },
            });
        }

        // Backup-before-repair: rename corrupted file to .corrupt.{timestamp}
        let timestamp = unix_timestamp_secs();
        let backup_path = PathBuf::from(format!("{}.corrupt.{timestamp}", path.display()));
        let had_source = path.exists();
        if had_source {
            fs::rename(path, &backup_path).map_err(|e| {
                warn!(
                    path = %path.display(),
                    backup = %backup_path.display(),
                    error = %e,
                    "failed to create backup before repair"
                );
                e
            })?;
        }

        let repair_start = Instant::now();
        let outcome = self.repair_file(path, &sidecar);
        let repair_time = repair_start.elapsed();

        match outcome {
            Ok(FileRepairOutcome::Repaired { bytes_written, .. }) => {
                // Verify the repaired file passes integrity check.
                let post_verify = self.verify_file(path, &sidecar);
                match post_verify {
                    Ok(v) if v.healthy => {
                        // Success — clean up backup.
                        if had_source {
                            let _ = fs::remove_file(&backup_path);
                        }
                        self.log_repair_event(
                            path,
                            true,
                            bytes_written,
                            v.expected_crc32,
                            v.actual_crc32,
                            repair_time,
                        );
                        Ok(HealthCheckResult {
                            path: path.to_path_buf(),
                            status: FileHealth::Repaired {
                                bytes_written,
                                repair_time,
                            },
                        })
                    }
                    _ => {
                        // Repaired file failed verification — restore backup.
                        warn!(
                            path = %path.display(),
                            "repaired file failed post-repair verification, restoring backup"
                        );
                        if had_source {
                            let _ = fs::remove_file(path);
                            let _ = fs::rename(&backup_path, path);
                        }
                        self.log_repair_event(path, false, 0, 0, 0, repair_time);
                        Ok(HealthCheckResult {
                            path: path.to_path_buf(),
                            status: FileHealth::Unrecoverable {
                                reason: "repaired file failed post-repair verification".to_owned(),
                            },
                        })
                    }
                }
            }
            Ok(FileRepairOutcome::NotNeeded) => {
                // Race condition: file was fine when repair ran.
                if had_source {
                    let _ = fs::rename(&backup_path, path);
                }
                Ok(HealthCheckResult {
                    path: path.to_path_buf(),
                    status: FileHealth::Intact,
                })
            }
            Ok(FileRepairOutcome::Unrecoverable { reason, .. }) => {
                // Restore backup — repair failed.
                if had_source {
                    let _ = fs::rename(&backup_path, path);
                }
                self.log_repair_event(path, false, 0, 0, 0, repair_time);
                Ok(HealthCheckResult {
                    path: path.to_path_buf(),
                    status: FileHealth::Unrecoverable {
                        reason: format!("{reason:?}"),
                    },
                })
            }
            Err(e) => {
                // Restore backup on error.
                if had_source {
                    let _ = fs::rename(&backup_path, path);
                }
                Err(e)
            }
        }
    }

    /// Protect all protectable files in a directory.
    ///
    /// Scans for files without a corresponding `.fec` sidecar and generates
    /// protection for them. Skips `.fec` files themselves and hidden files.
    pub fn protect_directory(&self, dir: &Path) -> SearchResult<DirectoryProtectionReport> {
        let start = Instant::now();
        let mut files_protected = 0_usize;
        let mut files_already_protected = 0_usize;
        let mut total_source_bytes = 0_u64;
        let mut total_repair_bytes = 0_u64;

        let entries = fs::read_dir(dir)?;
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let name = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n.to_owned(),
                None => continue,
            };
            // Skip sidecar files and hidden files.
            if name.ends_with(".fec") || name.starts_with('.') || name.contains(".corrupt.") {
                continue;
            }

            let sidecar = Self::sidecar_path(&path);
            if sidecar.exists() {
                files_already_protected += 1;
                continue;
            }

            match self.protect_file(&path) {
                Ok(result) => {
                    total_source_bytes += result.source_len;
                    let repair_size = fs::metadata(&result.sidecar_path)
                        .map(|m| m.len())
                        .unwrap_or(0);
                    total_repair_bytes += repair_size;
                    files_protected += 1;
                }
                Err(e) => {
                    warn!(
                        path = %path.display(),
                        error = %e,
                        "failed to protect file in directory scan"
                    );
                }
            }
        }

        let elapsed = start.elapsed();
        info!(
            dir = %dir.display(),
            files_protected,
            files_already_protected,
            total_source_bytes,
            total_repair_bytes,
            elapsed_ms = elapsed.as_millis(),
            "directory protection pass complete"
        );

        Ok(DirectoryProtectionReport {
            files_protected,
            files_already_protected,
            total_source_bytes,
            total_repair_bytes,
            elapsed,
        })
    }

    /// Verify (and auto-repair) all protected files in a directory.
    pub fn verify_directory(&self, dir: &Path) -> SearchResult<DirectoryHealthReport> {
        let start = Instant::now();
        let mut results = Vec::new();
        let mut intact_count = 0_usize;
        let mut repaired_count = 0_usize;
        let mut unrecoverable_count = 0_usize;
        let mut unprotected_count = 0_usize;

        let entries = fs::read_dir(dir)?;
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let name = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n.to_owned(),
                None => continue,
            };
            // Skip sidecar files, hidden files, and backup files.
            if name.ends_with(".fec") || name.starts_with('.') || name.contains(".corrupt.") {
                continue;
            }

            let result = self.verify_and_repair_file(&path)?;
            match &result.status {
                FileHealth::Intact => intact_count += 1,
                FileHealth::Repaired { .. } => repaired_count += 1,
                FileHealth::Unrecoverable { .. } => unrecoverable_count += 1,
                FileHealth::Unprotected => unprotected_count += 1,
            }
            results.push(result);
        }

        let elapsed = start.elapsed();
        info!(
            dir = %dir.display(),
            intact = intact_count,
            repaired = repaired_count,
            unrecoverable = unrecoverable_count,
            unprotected = unprotected_count,
            elapsed_ms = elapsed.as_millis(),
            "directory health check complete"
        );

        Ok(DirectoryHealthReport {
            results,
            intact_count,
            repaired_count,
            unrecoverable_count,
            unprotected_count,
            elapsed,
        })
    }

    /// Protect all existing unprotected files in a directory.
    ///
    /// This handles the migration case where durability is enabled on a
    /// system with pre-existing unprotected indices. Without this, all
    /// existing indices emit warnings on every open.
    pub fn protect_all_existing(&self, dir: &Path) -> SearchResult<DirectoryProtectionReport> {
        self.protect_directory(dir)
    }

    /// Log a repair event to the configured JSONL log directory.
    fn log_repair_event(
        &self,
        path: &Path,
        repair_succeeded: bool,
        bytes_written: usize,
        expected_crc32: u32,
        actual_crc32: u32,
        repair_time: Duration,
    ) {
        let log_dir = match &self.pipeline_config.repair_log_dir {
            Some(dir) => dir,
            None => return,
        };

        let event = RepairEvent {
            timestamp: iso8601_now(),
            path: path.display().to_string(),
            corrupted: true,
            repair_succeeded,
            bytes_written,
            source_crc32_expected: expected_crc32,
            source_crc32_after: actual_crc32,
            repair_time_ms: repair_time.as_millis() as u64,
        };

        let log_path = log_dir.join("repair-events.jsonl");

        if let Ok(json) = serde_json::to_string(&event) {
            // Rotate if needed.
            if let Ok(true) = should_rotate(&log_path, self.pipeline_config.max_repair_log_entries)
            {
                let rotated = log_dir.join("repair-events.1.jsonl");
                let _ = fs::rename(&log_path, &rotated);
            }

            let line = format!("{json}\n");
            if let Err(e) = append_to_file(&log_path, line.as_bytes()) {
                warn!(
                    log_path = %log_path.display(),
                    error = %e,
                    "failed to write repair event log"
                );
            }
        }
    }
}

impl DurabilityProvider for FileProtector {
    fn protect(&self, path: &Path) -> SearchResult<FileProtectionResult> {
        self.protect_file(path)
    }

    fn verify(&self, path: &Path) -> SearchResult<FileVerifyResult> {
        let sidecar = Self::sidecar_path(path);
        self.verify_file(path, &sidecar)
    }

    fn repair(&self, path: &Path) -> SearchResult<FileRepairOutcome> {
        let sidecar = Self::sidecar_path(path);
        self.repair_file(path, &sidecar)
    }

    fn check_health(&self, path: &Path) -> SearchResult<HealthCheckResult> {
        self.verify_and_repair_file(path)
    }

    fn protect_directory(&self, dir: &Path) -> SearchResult<DirectoryProtectionReport> {
        FileProtector::protect_directory(self, dir)
    }

    fn verify_directory(&self, dir: &Path) -> SearchResult<DirectoryHealthReport> {
        FileProtector::verify_directory(self, dir)
    }

    fn metrics_snapshot(&self) -> DurabilityMetricsSnapshot {
        self.metrics.snapshot()
    }
}

fn normalize_recovered_data(
    mut data: Vec<u8>,
    header: &RepairTrailerHeader,
) -> SearchResult<Vec<u8>> {
    let expected_len =
        usize::try_from(header.source_len).map_err(|_| SearchError::InvalidConfig {
            field: "source_len".to_owned(),
            value: header.source_len.to_string(),
            reason: "cannot convert source_len to usize".to_owned(),
        })?;
    if data.len() > expected_len {
        data.truncate(expected_len);
    }
    Ok(data)
}

fn source_symbols_from_bytes(
    bytes: &[u8],
    symbol_size: u32,
    k_source: u32,
) -> SearchResult<Vec<(u32, Vec<u8>)>> {
    let symbol_size_usize =
        usize::try_from(symbol_size).map_err(|_| SearchError::InvalidConfig {
            field: "symbol_size".to_owned(),
            value: symbol_size.to_string(),
            reason: "cannot convert symbol_size to usize".to_owned(),
        })?;

    let mut out = Vec::new();
    for esi in 0..k_source {
        let esi_usize = usize::try_from(esi).map_err(|_| SearchError::InvalidConfig {
            field: "esi".to_owned(),
            value: esi.to_string(),
            reason: "cannot convert symbol index to usize".to_owned(),
        })?;
        let start =
            esi_usize
                .checked_mul(symbol_size_usize)
                .ok_or_else(|| SearchError::InvalidConfig {
                    field: "start_offset".to_owned(),
                    value: format!("{esi_usize}*{symbol_size_usize}"),
                    reason: "source symbol offset overflow".to_owned(),
                })?;
        if start >= bytes.len() {
            continue;
        }

        let end = start.saturating_add(symbol_size_usize).min(bytes.len());
        let mut symbol = bytes[start..end].to_vec();
        if symbol.len() < symbol_size_usize {
            symbol.resize(symbol_size_usize, 0);
        }
        out.push((esi, symbol));
    }

    Ok(out)
}

fn saturating_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn unix_timestamp_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn iso8601_now() -> String {
    let secs = unix_timestamp_secs();
    // Simple ISO 8601 format without external chrono dependency.
    format!("{secs}")
}

fn should_rotate(log_path: &Path, max_entries: usize) -> std::io::Result<bool> {
    if !log_path.exists() {
        return Ok(false);
    }
    let contents = fs::read_to_string(log_path)?;
    let line_count = contents.lines().count();
    Ok(line_count >= max_entries)
}

fn append_to_file(path: &Path, data: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    file.write_all(data)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    use fsqlite_core::raptorq_integration::{CodecDecodeResult, CodecEncodeResult, SymbolCodec};

    use super::{
        DurabilityProvider, FileHealth, FileProtector, FileRepairOutcome, NoopDurability,
        RepairPipelineConfig,
    };
    use crate::config::DurabilityConfig;

    #[derive(Debug)]
    struct MockRepairCodec;

    impl SymbolCodec for MockRepairCodec {
        fn encode(
            &self,
            source_data: &[u8],
            symbol_size: u32,
            _repair_overhead: f64,
        ) -> fsqlite_error::Result<CodecEncodeResult> {
            let symbol_size_usize = usize::try_from(symbol_size).unwrap_or(1);
            let mut source_symbols = Vec::new();
            let mut repair_symbols = Vec::new();

            let mut esi: u32 = 0;
            for chunk in source_data.chunks(symbol_size_usize) {
                let mut data = chunk.to_vec();
                if data.len() < symbol_size_usize {
                    data.resize(symbol_size_usize, 0);
                }
                source_symbols.push((esi, data.clone()));
                repair_symbols.push((esi + 1_000_000, data));
                esi = esi.saturating_add(1);
            }

            Ok(CodecEncodeResult {
                source_symbols,
                repair_symbols,
                k_source: esi,
            })
        }

        fn decode(
            &self,
            symbols: &[(u32, Vec<u8>)],
            k_source: u32,
            _symbol_size: u32,
        ) -> fsqlite_error::Result<CodecDecodeResult> {
            let mut reconstructed = Vec::new();
            for source_esi in 0..k_source {
                let primary = symbols
                    .iter()
                    .find(|(esi, _)| *esi == source_esi)
                    .map(|(_, data)| data.clone());
                let fallback = symbols
                    .iter()
                    .find(|(esi, _)| *esi == source_esi + 1_000_000)
                    .map(|(_, data)| data.clone());

                match primary.or(fallback) {
                    Some(data) => reconstructed.extend_from_slice(&data),
                    None => {
                        return Ok(CodecDecodeResult::Failure {
                            reason: fsqlite_core::raptorq_integration::DecodeFailureReason::InsufficientSymbols,
                            symbols_received: u32::try_from(symbols.len()).unwrap_or(u32::MAX),
                            k_required: k_source,
                        });
                    }
                }
            }

            Ok(CodecDecodeResult::Success {
                data: reconstructed,
                symbols_used: k_source,
                peeled_count: k_source,
                inactivated_count: 0,
            })
        }
    }

    #[test]
    fn protect_verify_and_repair_file_roundtrip() {
        let config = DurabilityConfig {
            symbol_size: 256,
            // Overhead must be >= 100% so repair symbols cover all source symbols
            // when the entire source file is lost.
            repair_overhead: 2.0,
            ..DurabilityConfig::default()
        };
        let protector = FileProtector::new(Arc::new(MockRepairCodec), config).expect("protector");

        let path = temp_path("durability-roundtrip");
        let payload = vec![42_u8; 700];
        std::fs::write(&path, &payload).expect("write payload");

        let protected = protector.protect_file(&path).expect("protect");
        let verify = protector
            .verify_file(&path, &protected.sidecar_path)
            .expect("verify");
        assert!(verify.healthy);

        // Simulate catastrophic data loss; repair should restore from sidecar symbols.
        std::fs::write(&path, []).expect("wipe file");
        let repaired = protector
            .repair_file(&path, &protected.sidecar_path)
            .expect("repair");
        assert!(matches!(repaired, FileRepairOutcome::Repaired { .. }));

        let restored = std::fs::read(&path).expect("read restored");
        assert_eq!(restored, payload);

        let snapshot = protector.metrics_snapshot();
        assert_eq!(snapshot.repair_attempts, 1);
        assert_eq!(snapshot.repair_successes, 1);
    }

    #[test]
    fn repair_restores_deleted_file_from_sidecar() {
        let config = DurabilityConfig {
            symbol_size: 256,
            repair_overhead: 2.0,
            ..DurabilityConfig::default()
        };
        let protector = FileProtector::new(Arc::new(MockRepairCodec), config).expect("protector");

        let path = temp_path("durability-missing-file");
        let payload = b"recover-me-from-sidecar".to_vec();
        std::fs::write(&path, &payload).expect("write payload");
        let protected = protector.protect_file(&path).expect("protect");

        std::fs::remove_file(&path).expect("remove payload file");
        let repaired = protector
            .repair_file(&path, &protected.sidecar_path)
            .expect("repair missing file");
        assert!(matches!(repaired, FileRepairOutcome::Repaired { .. }));

        let restored = std::fs::read(&path).expect("read restored file");
        assert_eq!(restored, payload);
    }

    fn temp_path(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "frankensearch-durability-{prefix}-{}-{nanos}.bin",
            std::process::id()
        ))
    }

    fn temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "frankensearch-durability-dir-{prefix}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn test_config() -> DurabilityConfig {
        DurabilityConfig {
            symbol_size: 256,
            repair_overhead: 2.0,
            ..DurabilityConfig::default()
        }
    }

    // --- DurabilityProvider trait tests ---

    #[test]
    fn noop_durability_returns_defaults() {
        let noop = NoopDurability;
        let result = noop.protect(std::path::Path::new("/nonexistent")).unwrap();
        assert_eq!(result.source_len, 0);

        let verify = noop.verify(std::path::Path::new("/nonexistent")).unwrap();
        assert!(verify.healthy);

        let repair = noop.repair(std::path::Path::new("/nonexistent"));
        assert!(repair.is_err());

        let snap = noop.metrics_snapshot();
        assert_eq!(snap.encode_ops, 0);
    }

    // --- HealthCheckResult / verify_and_repair_file tests ---

    #[test]
    fn verify_and_repair_intact_file() {
        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");

        let path = temp_path("health-intact");
        std::fs::write(&path, vec![42_u8; 500]).expect("write");
        protector.protect_file(&path).expect("protect");

        let result = protector.verify_and_repair_file(&path).expect("check");
        assert!(matches!(result.status, FileHealth::Intact));
    }

    #[test]
    fn verify_and_repair_unprotected_file() {
        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");

        let path = temp_path("health-unprotected");
        std::fs::write(&path, vec![42_u8; 500]).expect("write");

        let result = protector.verify_and_repair_file(&path).expect("check");
        assert!(matches!(result.status, FileHealth::Unprotected));
    }

    #[test]
    fn verify_and_repair_corrupted_file_with_backup() {
        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");

        let path = temp_path("health-corrupt");
        let payload = vec![42_u8; 500];
        std::fs::write(&path, &payload).expect("write");
        protector.protect_file(&path).expect("protect");

        // Corrupt the file.
        let mut corrupted = payload.clone();
        corrupted[0] ^= 0xFF;
        std::fs::write(&path, &corrupted).expect("corrupt");

        let result = protector.verify_and_repair_file(&path).expect("check");
        assert!(
            matches!(result.status, FileHealth::Repaired { .. }),
            "expected Repaired, got {:?}",
            result.status
        );

        // Verify the file was restored.
        let restored = std::fs::read(&path).expect("read");
        assert_eq!(restored, payload);

        // Verify no backup file remains (successful repair cleans up).
        let dir = path.parent().unwrap();
        let backup_exists = std::fs::read_dir(dir).unwrap().flatten().any(|e| {
            e.file_name()
                .to_str()
                .map(|n| n.contains(".corrupt."))
                .unwrap_or(false)
                && e.path()
                    .to_str()
                    .map(|p| p.contains("health-corrupt"))
                    .unwrap_or(false)
        });
        assert!(
            !backup_exists,
            "backup should be cleaned up after successful repair"
        );
    }

    #[test]
    fn verify_and_repair_auto_repair_disabled() {
        let metrics = Arc::new(crate::metrics::DurabilityMetrics::default());
        let pipeline_config = RepairPipelineConfig {
            auto_repair: false,
            ..RepairPipelineConfig::default()
        };
        let protector = FileProtector::new_with_pipeline_config(
            Arc::new(MockRepairCodec),
            test_config(),
            metrics,
            pipeline_config,
        )
        .expect("protector");

        let path = temp_path("health-no-repair");
        let payload = vec![42_u8; 500];
        std::fs::write(&path, &payload).expect("write");
        protector.protect_file(&path).expect("protect");

        // Corrupt the file.
        let mut corrupted = payload;
        corrupted[0] ^= 0xFF;
        std::fs::write(&path, &corrupted).expect("corrupt");

        let result = protector.verify_and_repair_file(&path).expect("check");
        assert!(
            matches!(result.status, FileHealth::Unrecoverable { .. }),
            "expected Unrecoverable when auto_repair disabled, got {:?}",
            result.status
        );
    }

    // --- Directory-level operation tests ---

    #[test]
    fn protect_directory_generates_sidecars() {
        let dir = temp_dir("protect-dir");
        std::fs::write(dir.join("file1.dat"), vec![1_u8; 300]).expect("write");
        std::fs::write(dir.join("file2.dat"), vec![2_u8; 400]).expect("write");

        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");
        let report = protector.protect_directory(&dir).expect("protect dir");

        assert_eq!(report.files_protected, 2);
        assert_eq!(report.files_already_protected, 0);
        assert!(report.total_source_bytes > 0);
        assert!(report.total_repair_bytes > 0);

        // Second pass should skip.
        let report2 = protector
            .protect_directory(&dir)
            .expect("protect dir again");
        assert_eq!(report2.files_protected, 0);
        assert_eq!(report2.files_already_protected, 2);
    }

    #[test]
    fn verify_directory_detects_corruption() {
        let dir = temp_dir("verify-dir");
        let payload = vec![42_u8; 500];
        std::fs::write(dir.join("good.dat"), &payload).expect("write");
        std::fs::write(dir.join("bad.dat"), &payload).expect("write");

        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");
        protector.protect_directory(&dir).expect("protect dir");

        // Corrupt one file.
        let mut corrupted = payload;
        corrupted[0] ^= 0xFF;
        std::fs::write(dir.join("bad.dat"), &corrupted).expect("corrupt");

        let report = protector.verify_directory(&dir).expect("verify dir");
        assert_eq!(report.intact_count, 1);
        assert!(
            report.repaired_count >= 1,
            "expected at least 1 repaired, got {}",
            report.repaired_count
        );
        assert_eq!(report.unrecoverable_count, 0);
    }

    // --- Repair event logging tests ---

    #[test]
    fn repair_event_is_logged_to_jsonl() {
        let log_dir = temp_dir("repair-log");
        let metrics = Arc::new(crate::metrics::DurabilityMetrics::default());
        let pipeline_config = RepairPipelineConfig {
            repair_log_dir: Some(log_dir.clone()),
            ..RepairPipelineConfig::default()
        };
        let protector = FileProtector::new_with_pipeline_config(
            Arc::new(MockRepairCodec),
            test_config(),
            metrics,
            pipeline_config,
        )
        .expect("protector");

        let path = temp_path("repair-logged");
        let payload = vec![42_u8; 500];
        std::fs::write(&path, &payload).expect("write");
        protector.protect_file(&path).expect("protect");

        // Corrupt and repair.
        let mut corrupted = payload;
        corrupted[0] ^= 0xFF;
        std::fs::write(&path, &corrupted).expect("corrupt");
        let result = protector.verify_and_repair_file(&path).expect("repair");
        assert!(matches!(result.status, FileHealth::Repaired { .. }));

        // Check log file.
        let log_path = log_dir.join("repair-events.jsonl");
        assert!(log_path.exists(), "repair event log should exist");
        let contents = std::fs::read_to_string(&log_path).expect("read log");
        assert!(!contents.is_empty(), "log should not be empty");
        assert!(
            contents.contains("repair_succeeded"),
            "log should contain event data"
        );
    }

    // --- DurabilityProvider trait impl tests ---

    #[test]
    fn file_protector_implements_durability_provider() {
        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");

        let path = temp_path("provider-impl");
        std::fs::write(&path, vec![42_u8; 500]).expect("write");

        // Use through trait.
        let provider: &dyn DurabilityProvider = &protector;
        let _protection = provider.protect(&path).expect("protect via trait");

        let verify = provider.verify(&path).expect("verify via trait");
        assert!(verify.healthy);

        let health = provider.check_health(&path).expect("health via trait");
        assert!(matches!(health.status, FileHealth::Intact));
    }

    // --- protect_all_existing migration test ---

    #[test]
    fn protect_all_existing_migration() {
        let dir = temp_dir("migrate");
        std::fs::write(dir.join("old_index.fsvi"), vec![1_u8; 300]).expect("write");
        std::fs::write(dir.join("old_index.tantivy"), vec![2_u8; 400]).expect("write");

        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");
        let report = protector.protect_all_existing(&dir).expect("migrate");
        assert_eq!(report.files_protected, 2);

        // Verify both have sidecars now.
        assert!(dir.join("old_index.fsvi.fec").exists());
        assert!(dir.join("old_index.tantivy.fec").exists());
    }

    // --- Corruption simulation tests ---

    #[test]
    fn detect_single_bit_flip() {
        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");

        let path = temp_path("single-bit-flip");
        let payload = vec![42_u8; 500];
        std::fs::write(&path, &payload).expect("write");
        let result = protector.protect_file(&path).expect("protect");

        // Flip a single bit.
        let mut corrupted = payload.clone();
        corrupted[100] ^= 0x01;
        std::fs::write(&path, &corrupted).expect("corrupt");

        let verify = protector
            .verify_file(&path, &result.sidecar_path)
            .expect("verify");
        assert!(!verify.healthy, "single bit flip should be detected");
    }

    #[test]
    fn repair_single_bit_flip() {
        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");

        let path = temp_path("repair-bit-flip");
        let payload = vec![42_u8; 500];
        std::fs::write(&path, &payload).expect("write");
        protector.protect_file(&path).expect("protect");

        // Flip a single bit and repair via pipeline.
        let mut corrupted = payload.clone();
        corrupted[100] ^= 0x01;
        std::fs::write(&path, &corrupted).expect("corrupt");

        let result = protector.verify_and_repair_file(&path).expect("repair");
        assert!(
            matches!(result.status, FileHealth::Repaired { .. }),
            "single bit flip should be repaired, got {:?}",
            result.status
        );
        let restored = std::fs::read(&path).expect("read");
        assert_eq!(restored, payload);
    }

    #[test]
    fn detect_zeroed_block() {
        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");

        let path = temp_path("zeroed-block");
        // Use data with non-zero content so zeroing is detectable.
        let payload: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        std::fs::write(&path, &payload).expect("write");
        let result = protector.protect_file(&path).expect("protect");

        // Zero out a 256-byte block (one symbol).
        let mut corrupted = payload.clone();
        corrupted[256..512].fill(0);
        std::fs::write(&path, &corrupted).expect("corrupt");

        let verify = protector
            .verify_file(&path, &result.sidecar_path)
            .expect("verify");
        assert!(!verify.healthy, "zeroed block should be detected");
    }

    #[test]
    fn repair_zeroed_block() {
        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");

        let path = temp_path("repair-zeroed");
        let payload: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        std::fs::write(&path, &payload).expect("write");
        protector.protect_file(&path).expect("protect");

        // Zero out a 256-byte block.
        let mut corrupted = payload.clone();
        corrupted[256..512].fill(0);
        std::fs::write(&path, &corrupted).expect("corrupt");

        let result = protector.verify_and_repair_file(&path).expect("repair");
        assert!(
            matches!(result.status, FileHealth::Repaired { .. }),
            "zeroed block should be repaired, got {:?}",
            result.status
        );
        let restored = std::fs::read(&path).expect("read");
        assert_eq!(restored, payload);
    }

    #[test]
    fn detect_appended_data() {
        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");

        let path = temp_path("appended-data");
        let payload = vec![42_u8; 500];
        std::fs::write(&path, &payload).expect("write");
        let result = protector.protect_file(&path).expect("protect");

        // Append extra bytes.
        let mut extended = payload;
        extended.extend_from_slice(&[0xFF; 100]);
        std::fs::write(&path, &extended).expect("extend");

        let verify = protector
            .verify_file(&path, &result.sidecar_path)
            .expect("verify");
        assert!(!verify.healthy, "appended data should change CRC");
        assert_ne!(verify.expected_len, verify.actual_len);
    }

    #[test]
    fn repair_multiple_non_adjacent_corruptions() {
        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");

        let path = temp_path("multi-corrupt");
        let payload: Vec<u8> = (0..2048).map(|i| ((i * 7) % 256) as u8).collect();
        std::fs::write(&path, &payload).expect("write");
        protector.protect_file(&path).expect("protect");

        // Corrupt 3 non-adjacent 32-byte regions.
        let mut corrupted = payload.clone();
        for byte in &mut corrupted[0..32] {
            *byte ^= 0xFF;
        }
        for byte in &mut corrupted[512..544] {
            *byte ^= 0xFF;
        }
        for byte in &mut corrupted[1024..1056] {
            *byte ^= 0xFF;
        }
        std::fs::write(&path, &corrupted).expect("corrupt");

        let result = protector.verify_and_repair_file(&path).expect("repair");
        assert!(
            matches!(result.status, FileHealth::Repaired { .. }),
            "multiple non-adjacent corruptions should be repaired, got {:?}",
            result.status
        );
        let restored = std::fs::read(&path).expect("read");
        assert_eq!(restored, payload);
    }

    #[test]
    fn small_file_protect_and_repair() {
        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");

        // File smaller than one symbol (256 bytes).
        let path = temp_path("tiny-file");
        let payload = vec![7_u8; 50];
        std::fs::write(&path, &payload).expect("write");
        protector.protect_file(&path).expect("protect");

        // Delete and repair.
        std::fs::remove_file(&path).expect("delete");
        let result = protector.verify_and_repair_file(&path).expect("repair");
        assert!(
            matches!(result.status, FileHealth::Repaired { .. }),
            "small file should be repaired, got {:?}",
            result.status
        );
        let restored = std::fs::read(&path).expect("read");
        assert_eq!(restored, payload);
    }

    #[test]
    fn directory_skips_hidden_and_backup_files() {
        let dir = temp_dir("skip-hidden");
        std::fs::write(dir.join("normal.dat"), vec![1_u8; 300]).expect("write normal");
        std::fs::write(dir.join(".hidden"), vec![2_u8; 300]).expect("write hidden");
        std::fs::write(dir.join("old.dat.corrupt.12345"), vec![3_u8; 300]).expect("write backup");

        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");
        let report = protector.protect_directory(&dir).expect("protect");
        assert_eq!(
            report.files_protected, 1,
            "only normal.dat should be protected"
        );
    }

    #[test]
    fn empty_file_protect_and_verify() {
        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");

        let path = temp_path("empty-file");
        std::fs::write(&path, &[]).expect("write empty");

        // Empty file should still be protectable (0 source symbols).
        let result = protector.protect_file(&path).expect("protect");
        assert_eq!(result.source_len, 0);

        let verify = protector
            .verify_file(&path, &result.sidecar_path)
            .expect("verify");
        assert!(verify.healthy);
    }
}
