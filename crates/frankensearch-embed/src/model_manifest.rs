//! Model manifest definitions and verification helpers.
//!
//! This module is intentionally synchronous and runtime-agnostic:
//! it performs filesystem and hashing work only, and leaves transport/network
//! to higher-level download orchestration.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::{OnceLock, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use frankensearch_core::error::{SearchError, SearchResult};

/// Environment variable for explicit model-download consent.
pub const DOWNLOAD_CONSENT_ENV: &str = "FRANKENSEARCH_ALLOW_DOWNLOAD";

/// Placeholder checksum used until a model file is downloaded and verified.
pub const PLACEHOLDER_VERIFY_AFTER_DOWNLOAD: &str = "PLACEHOLDER_VERIFY_AFTER_DOWNLOAD";

/// Placeholder revision used by built-in manifests until pinned revisions are filled in.
pub const PLACEHOLDER_PINNED_REVISION: &str = "UNPINNED_VERIFY_AFTER_DOWNLOAD";

const HASH_BUFFER_SIZE: usize = 8 * 1024;

/// One file required by a model manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelFile {
    /// Relative path inside the model directory.
    pub name: String,
    /// Expected lowercase SHA256 hex digest.
    pub sha256: String,
    /// Expected size in bytes.
    pub size: u64,
}

impl ModelFile {
    /// Returns true when the file still uses the placeholder checksum.
    #[must_use]
    pub fn uses_placeholder_checksum(&self) -> bool {
        self.sha256 == PLACEHOLDER_VERIFY_AFTER_DOWNLOAD
    }

    /// Returns true when checksum and size are usable for production verification.
    #[must_use]
    pub fn has_verified_checksum(&self) -> bool {
        self.size > 0 && is_valid_sha256_hex(&self.sha256) && !self.uses_placeholder_checksum()
    }
}

/// Manifest for one downloadable model bundle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Stable model identifier.
    pub id: String,
    /// `HuggingFace` repository slug.
    pub repo: String,
    /// Pinned revision (commit SHA).
    pub revision: String,
    /// Required files for this model.
    pub files: Vec<ModelFile>,
    /// SPDX-style license identifier.
    pub license: String,
}

impl ModelManifest {
    /// Built-in manifest for MiniLM-L6-v2 (quality tier).
    #[must_use]
    pub fn minilm_v2() -> Self {
        Self {
            id: "all-minilm-l6-v2".to_owned(),
            repo: "sentence-transformers/all-MiniLM-L6-v2".to_owned(),
            revision: PLACEHOLDER_PINNED_REVISION.to_owned(),
            files: vec![
                ModelFile {
                    name: "onnx/model.onnx".to_owned(),
                    sha256: PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned(),
                    size: 0,
                },
                ModelFile {
                    name: "tokenizer.json".to_owned(),
                    sha256: PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned(),
                    size: 0,
                },
                ModelFile {
                    name: "config.json".to_owned(),
                    sha256: PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned(),
                    size: 0,
                },
                ModelFile {
                    name: "special_tokens_map.json".to_owned(),
                    sha256: PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned(),
                    size: 0,
                },
                ModelFile {
                    name: "tokenizer_config.json".to_owned(),
                    sha256: PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned(),
                    size: 0,
                },
            ],
            license: "Apache-2.0".to_owned(),
        }
    }

    /// Built-in manifest for potion-128M style `Model2Vec` assets (fast tier).
    #[must_use]
    pub fn potion_128m() -> Self {
        Self {
            id: "potion-multilingual-128m".to_owned(),
            repo: "minishlab/potion-multilingual-128M".to_owned(),
            revision: PLACEHOLDER_PINNED_REVISION.to_owned(),
            files: vec![
                ModelFile {
                    name: "tokenizer.json".to_owned(),
                    sha256: PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned(),
                    size: 0,
                },
                ModelFile {
                    name: "model.safetensors".to_owned(),
                    sha256: PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned(),
                    size: 0,
                },
            ],
            license: "Apache-2.0".to_owned(),
        }
    }

    /// Parse a manifest from JSON and validate basic structure.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if JSON parsing or validation fails.
    pub fn from_json_str(raw: &str) -> SearchResult<Self> {
        let manifest =
            serde_json::from_str::<Self>(raw).map_err(|source| SearchError::InvalidConfig {
                field: "manifest_json".to_owned(),
                value: truncate_for_error(raw),
                reason: format!("failed to parse manifest JSON: {source}"),
            })?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Serialize this manifest to pretty JSON.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if serialization fails.
    pub fn to_pretty_json(&self) -> SearchResult<String> {
        serde_json::to_string_pretty(self).map_err(|source| SearchError::InvalidConfig {
            field: "manifest_json".to_owned(),
            value: self.id.clone(),
            reason: format!("failed to serialize manifest: {source}"),
        })
    }

    /// Returns true when all files have non-placeholder checksums and non-zero sizes.
    #[must_use]
    pub fn has_verified_checksums(&self) -> bool {
        !self.files.is_empty() && self.files.iter().all(ModelFile::has_verified_checksum)
    }

    /// Returns true when revision appears pinned (not empty and not floating aliases).
    #[must_use]
    pub fn has_pinned_revision(&self) -> bool {
        let revision = self.revision.trim();
        !(revision.is_empty()
            || revision.eq_ignore_ascii_case("main")
            || revision.eq_ignore_ascii_case("master")
            || revision.eq_ignore_ascii_case("latest")
            || revision.eq_ignore_ascii_case("head")
            || revision == PLACEHOLDER_PINNED_REVISION)
    }

    /// Returns true when this manifest is ready for production-grade verification.
    #[must_use]
    pub fn is_production_ready(&self) -> bool {
        self.has_verified_checksums() && self.has_pinned_revision()
    }

    /// Sum of expected bytes for all files.
    #[must_use]
    pub fn total_size_bytes(&self) -> u64 {
        self.files.iter().map(|file| file.size).sum()
    }

    /// Validate manifest fields for shape and checksum format.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` for malformed fields.
    pub fn validate(&self) -> SearchResult<()> {
        if self.id.trim().is_empty() {
            return Err(invalid_manifest_field("id", &self.id, "must not be empty"));
        }
        if self.repo.trim().is_empty() {
            return Err(invalid_manifest_field(
                "repo",
                &self.repo,
                "must not be empty",
            ));
        }
        if self.revision.trim().is_empty() {
            return Err(invalid_manifest_field(
                "revision",
                &self.revision,
                "must not be empty",
            ));
        }
        if self.license.trim().is_empty() {
            return Err(invalid_manifest_field(
                "license",
                &self.license,
                "must not be empty",
            ));
        }

        for file in &self.files {
            if file.name.trim().is_empty() {
                return Err(invalid_manifest_field(
                    "files[].name",
                    &file.name,
                    "must not be empty",
                ));
            }
            if file.uses_placeholder_checksum() {
                continue;
            }
            if !is_valid_sha256_hex(&file.sha256) {
                return Err(invalid_manifest_field(
                    "files[].sha256",
                    &file.sha256,
                    "must be lowercase 64-char SHA256 hex or placeholder",
                ));
            }
            if file.size == 0 {
                return Err(invalid_manifest_field(
                    "files[].size",
                    "0",
                    "must be > 0 when checksum is non-placeholder",
                ));
            }
        }

        Ok(())
    }

    /// Enforce checksum policy; placeholder checksums are rejected in release mode.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if a release policy violation is detected.
    pub fn validate_checksum_policy(&self) -> SearchResult<()> {
        self.validate_checksum_policy_for(cfg!(not(debug_assertions)))
    }

    /// Enforce checksum policy with explicit release-mode toggle (useful for tests).
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if release-mode requires concrete checksums.
    pub fn validate_checksum_policy_for(&self, release_mode: bool) -> SearchResult<()> {
        if release_mode && self.files.iter().any(ModelFile::uses_placeholder_checksum) {
            return Err(invalid_manifest_field(
                "files[].sha256",
                PLACEHOLDER_VERIFY_AFTER_DOWNLOAD,
                "placeholder checksums are forbidden in release mode",
            ));
        }
        Ok(())
    }

    /// Verify all manifest files in `model_dir` using streaming SHA256 checks.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` when any file is missing or hash/size verification fails.
    pub fn verify_dir(&self, model_dir: &Path) -> SearchResult<()> {
        for file in &self.files {
            let path = model_dir.join(&file.name);
            verify_file_sha256(&path, &file.sha256, file.size)?;
        }
        Ok(())
    }

    /// Promote a staged model directory to final destination atomically after verification.
    ///
    /// Returns the backup path when an existing install was moved out of the way.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if verification or filesystem rename operations fail.
    pub fn promote_verified_installation(
        &self,
        staged_dir: &Path,
        destination_dir: &Path,
    ) -> SearchResult<Option<PathBuf>> {
        self.verify_dir(staged_dir)?;
        promote_atomically(staged_dir, destination_dir)
    }

    /// Return `UpdateAvailable` when installed revision differs from pinned revision.
    #[must_use]
    pub fn detect_update_state(&self, installed_revision: &str) -> Option<ModelState> {
        if !self.has_pinned_revision() {
            return None;
        }
        let current = installed_revision.trim();
        if current == self.revision {
            return None;
        }
        Some(ModelState::UpdateAvailable {
            current_revision: if current.is_empty() {
                "unknown".to_owned()
            } else {
                current.to_owned()
            },
            latest_revision: self.revision.clone(),
        })
    }

    /// Register this manifest in the in-process registry.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if validation fails or registry lock is poisoned.
    pub fn register(self) -> SearchResult<()> {
        self.validate()?;
        manifest_registry()
            .write()
            .map_err(|_| manifest_registry_lock_error("write"))?
            .insert(self.id.clone(), self);
        Ok(())
    }

    /// Look up a registered manifest by id.
    #[must_use]
    pub fn lookup(id: &str) -> Option<Self> {
        let guard = manifest_registry().read().ok()?;
        guard.get(id).cloned()
    }

    /// Return all registered manifests in deterministic id order.
    #[must_use]
    pub fn registered() -> Vec<Self> {
        manifest_registry()
            .read()
            .map(|guard| guard.values().cloned().collect())
            .unwrap_or_default()
    }
}

/// Model manifest catalog for bulk load/validation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelManifestCatalog {
    /// Manifests contained in this catalog.
    #[serde(default)]
    pub models: Vec<ModelManifest>,
}

impl ModelManifestCatalog {
    /// Parse a catalog from JSON.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if parsing fails.
    pub fn from_json_str(raw: &str) -> SearchResult<Self> {
        serde_json::from_str::<Self>(raw).map_err(|source| SearchError::InvalidConfig {
            field: "manifest_catalog_json".to_owned(),
            value: truncate_for_error(raw),
            reason: format!("failed to parse manifest catalog JSON: {source}"),
        })
    }

    /// Validate every manifest in the catalog.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if any contained manifest is invalid.
    pub fn validate(&self) -> SearchResult<()> {
        for model in &self.models {
            model.validate()?;
        }
        Ok(())
    }
}

/// Runtime state of model availability and lifecycle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelState {
    NotInstalled,
    NeedsConsent,
    Downloading {
        progress_pct: u8,
        bytes_downloaded: u64,
        total_bytes: u64,
    },
    Verifying,
    Ready,
    Disabled {
        reason: String,
    },
    VerificationFailed {
        reason: String,
    },
    UpdateAvailable {
        current_revision: String,
        latest_revision: String,
    },
    Cancelled,
}

/// Where a consent decision came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsentSource {
    Programmatic,
    Environment,
    Interactive,
    ConfigFile,
}

/// Resolved consent decision for model downloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DownloadConsent {
    /// Whether downloads are allowed.
    pub granted: bool,
    /// Origin of the consent signal.
    pub source: Option<ConsentSource>,
}

impl DownloadConsent {
    /// Explicitly granted consent.
    #[must_use]
    pub const fn granted(source: ConsentSource) -> Self {
        Self {
            granted: true,
            source: Some(source),
        }
    }

    /// Explicitly denied consent.
    #[must_use]
    pub const fn denied(source: Option<ConsentSource>) -> Self {
        Self {
            granted: false,
            source,
        }
    }
}

/// Resolve download consent using priority:
/// programmatic > environment > interactive > config.
#[must_use]
pub fn resolve_download_consent(
    programmatic: Option<bool>,
    interactive: Option<bool>,
    config_file: Option<bool>,
) -> DownloadConsent {
    let env_value = std::env::var(DOWNLOAD_CONSENT_ENV).ok();
    resolve_download_consent_with_env(programmatic, env_value.as_deref(), interactive, config_file)
}

fn resolve_download_consent_with_env(
    programmatic: Option<bool>,
    env_value: Option<&str>,
    interactive: Option<bool>,
    config_file: Option<bool>,
) -> DownloadConsent {
    if let Some(granted) = programmatic {
        return DownloadConsent {
            granted,
            source: Some(ConsentSource::Programmatic),
        };
    }

    if let Some(raw) = env_value
        && let Some(granted) = parse_bool_flag(raw)
    {
        return DownloadConsent {
            granted,
            source: Some(ConsentSource::Environment),
        };
    }

    if let Some(granted) = interactive {
        return DownloadConsent {
            granted,
            source: Some(ConsentSource::Interactive),
        };
    }

    if let Some(granted) = config_file {
        return DownloadConsent {
            granted,
            source: Some(ConsentSource::ConfigFile),
        };
    }

    DownloadConsent::denied(None)
}

/// Stateful lifecycle helper for model installation progress.
#[derive(Debug, Clone)]
pub struct ModelLifecycle {
    manifest: ModelManifest,
    state: ModelState,
    consent: DownloadConsent,
}

impl ModelLifecycle {
    /// Create lifecycle state for a manifest.
    #[must_use]
    pub const fn new(manifest: ModelManifest, consent: DownloadConsent) -> Self {
        let state = if consent.granted {
            ModelState::NotInstalled
        } else {
            ModelState::NeedsConsent
        };
        Self {
            manifest,
            state,
            consent,
        }
    }

    /// Current lifecycle state.
    #[must_use]
    pub const fn state(&self) -> &ModelState {
        &self.state
    }

    /// Underlying manifest for this lifecycle.
    #[must_use]
    pub const fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    /// Mark consent as granted (e.g., after explicit user approval).
    pub fn approve_consent(&mut self, source: ConsentSource) {
        self.consent = DownloadConsent::granted(source);
        if matches!(self.state, ModelState::NeedsConsent) {
            self.state = ModelState::NotInstalled;
        }
    }

    /// Start the download state.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` on invalid transition or zero total bytes.
    pub fn begin_download(&mut self, total_bytes: u64) -> SearchResult<()> {
        if !self.consent.granted {
            self.state = ModelState::NeedsConsent;
            return Err(SearchError::EmbedderUnavailable {
                model: self.manifest.id.clone(),
                reason: "download consent required".to_owned(),
            });
        }
        if total_bytes == 0 {
            return Err(SearchError::InvalidConfig {
                field: "total_bytes".to_owned(),
                value: "0".to_owned(),
                reason: "must be greater than zero".to_owned(),
            });
        }

        match self.state {
            ModelState::NotInstalled
            | ModelState::Cancelled
            | ModelState::VerificationFailed { .. } => {
                self.state = ModelState::Downloading {
                    progress_pct: 0,
                    bytes_downloaded: 0,
                    total_bytes,
                };
                Ok(())
            }
            _ => Err(invalid_state_transition(
                &self.state,
                "begin_download",
                "expected NotInstalled/Cancelled/VerificationFailed",
            )),
        }
    }

    /// Update bytes downloaded and recompute bounded percent.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if not currently downloading.
    pub fn update_download_progress(&mut self, bytes_downloaded: u64) -> SearchResult<()> {
        let (progress_pct, total_bytes, bounded_bytes) = match self.state {
            ModelState::Downloading { total_bytes, .. } => {
                let bounded = bytes_downloaded.min(total_bytes);
                let pct_u64 = bounded.saturating_mul(100) / total_bytes;
                #[allow(clippy::cast_possible_truncation)]
                let pct = pct_u64 as u8;
                (pct.min(100), total_bytes, bounded)
            }
            _ => {
                return Err(invalid_state_transition(
                    &self.state,
                    "update_download_progress",
                    "expected Downloading",
                ));
            }
        };

        self.state = ModelState::Downloading {
            progress_pct,
            bytes_downloaded: bounded_bytes,
            total_bytes,
        };
        Ok(())
    }

    /// Move from downloading to verifying.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if not currently downloading.
    pub fn begin_verification(&mut self) -> SearchResult<()> {
        if matches!(self.state, ModelState::Downloading { .. }) {
            self.state = ModelState::Verifying;
            return Ok(());
        }
        Err(invalid_state_transition(
            &self.state,
            "begin_verification",
            "expected Downloading",
        ))
    }

    /// Mark install ready.
    pub fn mark_ready(&mut self) {
        self.state = ModelState::Ready;
    }

    /// Mark install verification failed.
    pub fn fail_verification(&mut self, reason: impl Into<String>) {
        self.state = ModelState::VerificationFailed {
            reason: reason.into(),
        };
    }

    /// Mark model disabled.
    pub fn disable(&mut self, reason: impl Into<String>) {
        self.state = ModelState::Disabled {
            reason: reason.into(),
        };
    }

    /// Mark update available.
    pub fn mark_update_available(
        &mut self,
        current_revision: impl Into<String>,
        latest_revision: impl Into<String>,
    ) {
        self.state = ModelState::UpdateAvailable {
            current_revision: current_revision.into(),
            latest_revision: latest_revision.into(),
        };
    }

    /// Cancel current operation.
    pub fn cancel(&mut self) {
        self.state = ModelState::Cancelled;
    }

    /// Recover from cancelled state so a new download can start.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if current state is not `Cancelled`.
    pub fn recover_after_cancel(&mut self) -> SearchResult<()> {
        if !matches!(self.state, ModelState::Cancelled) {
            return Err(invalid_state_transition(
                &self.state,
                "recover_after_cancel",
                "expected Cancelled",
            ));
        }
        self.state = if self.consent.granted {
            ModelState::NotInstalled
        } else {
            ModelState::NeedsConsent
        };
        Ok(())
    }
}

/// Verify file size + SHA256 using streaming read.
///
/// # Errors
///
/// Returns `SearchError` when file is missing, unreadable, or hash/size mismatch occurs.
pub fn verify_file_sha256(
    path: &Path,
    expected_sha256: &str,
    expected_size: u64,
) -> SearchResult<()> {
    if expected_sha256 == PLACEHOLDER_VERIFY_AFTER_DOWNLOAD {
        return Err(SearchError::InvalidConfig {
            field: "sha256".to_owned(),
            value: expected_sha256.to_owned(),
            reason: "placeholder checksum cannot be verified".to_owned(),
        });
    }
    if !is_valid_sha256_hex(expected_sha256) {
        return Err(SearchError::InvalidConfig {
            field: "sha256".to_owned(),
            value: expected_sha256.to_owned(),
            reason: "expected lowercase 64-char SHA256 hex".to_owned(),
        });
    }
    if expected_size == 0 {
        return Err(SearchError::InvalidConfig {
            field: "size".to_owned(),
            value: "0".to_owned(),
            reason: "expected size must be greater than zero".to_owned(),
        });
    }

    if !path.exists() {
        return Err(SearchError::ModelNotFound {
            name: format!("missing model file: {}", path.display()),
        });
    }

    let metadata = fs::metadata(path).map_err(|source| SearchError::ModelLoadFailed {
        path: path.to_path_buf(),
        source: Box::new(source),
    })?;
    if !metadata.is_file() {
        return Err(SearchError::ModelLoadFailed {
            path: path.to_path_buf(),
            source: "expected a regular file".into(),
        });
    }

    let file = File::open(path).map_err(|source| SearchError::ModelLoadFailed {
        path: path.to_path_buf(),
        source: Box::new(source),
    })?;
    let mut reader = BufReader::new(file);
    let mut buffer = [0_u8; HASH_BUFFER_SIZE];
    let mut hasher = Sha256::new();
    let mut bytes_read = 0_u64;

    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|source| SearchError::ModelLoadFailed {
                path: path.to_path_buf(),
                source: Box::new(source),
            })?;
        if read == 0 {
            break;
        }
        let read_u64 = u64::try_from(read).map_err(|_| SearchError::InvalidConfig {
            field: "read_size".to_owned(),
            value: read.to_string(),
            reason: "read size does not fit u64".to_owned(),
        })?;
        bytes_read = bytes_read.saturating_add(read_u64);
        hasher.update(&buffer[..read]);
    }

    let actual_sha256 = to_hex_lowercase(&hasher.finalize());
    let expected_lower = expected_sha256.to_ascii_lowercase();
    if bytes_read != expected_size || actual_sha256 != expected_lower {
        return Err(SearchError::HashMismatch {
            path: path.to_path_buf(),
            expected: format!("sha256={expected_lower},size={expected_size}"),
            actual: format!("sha256={actual_sha256},size={bytes_read}"),
        });
    }

    Ok(())
}

fn promote_atomically(staged_dir: &Path, destination_dir: &Path) -> SearchResult<Option<PathBuf>> {
    let destination_parent =
        destination_dir
            .parent()
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "destination_dir".to_owned(),
                value: destination_dir.display().to_string(),
                reason: "destination must have a parent directory".to_owned(),
            })?;
    fs::create_dir_all(destination_parent).map_err(SearchError::from)?;

    let stage_name = destination_dir.file_name().map_or_else(
        || "model".to_owned(),
        |part| part.to_string_lossy().into_owned(),
    );
    let stage_target = destination_parent.join(format!(".{stage_name}.installing"));
    fs::rename(staged_dir, &stage_target).map_err(SearchError::from)?;

    let backup_path = if destination_dir.exists() {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_secs());
        let backup = destination_parent.join(format!("{stage_name}.backup.{timestamp}"));
        fs::rename(destination_dir, &backup).map_err(SearchError::from)?;
        Some(backup)
    } else {
        None
    };

    fs::rename(&stage_target, destination_dir).map_err(SearchError::from)?;
    Ok(backup_path)
}

fn manifest_registry() -> &'static RwLock<BTreeMap<String, ModelManifest>> {
    static REGISTRY: OnceLock<RwLock<BTreeMap<String, ModelManifest>>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let mut data = BTreeMap::new();
        let minilm = ModelManifest::minilm_v2();
        data.insert(minilm.id.clone(), minilm);
        let potion = ModelManifest::potion_128m();
        data.insert(potion.id.clone(), potion);
        RwLock::new(data)
    })
}

fn manifest_registry_lock_error(action: &str) -> SearchError {
    SearchError::SubsystemError {
        subsystem: "model_manifest",
        source: std::io::Error::other(format!("manifest registry {action} lock poisoned")).into(),
    }
}

fn invalid_manifest_field(field: &str, value: &str, reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: field.to_owned(),
        value: value.to_owned(),
        reason: reason.to_owned(),
    }
}

fn invalid_state_transition(state: &ModelState, operation: &str, reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: "model_state".to_owned(),
        value: format!("{state:?}"),
        reason: format!("invalid transition for {operation}: {reason}"),
    }
}

fn truncate_for_error(value: &str) -> String {
    const MAX: usize = 120;
    let mut chars = value.chars();
    let truncated: String = chars.by_ref().take(MAX).collect();
    if chars.next().is_none() {
        return truncated;
    }
    let mut out = truncated;
    out.push_str("...");
    out
}

fn parse_bool_flag(raw: &str) -> Option<bool> {
    let value = raw.trim();
    if value == "1"
        || value.eq_ignore_ascii_case("true")
        || value.eq_ignore_ascii_case("yes")
        || value.eq_ignore_ascii_case("on")
    {
        return Some(true);
    }
    if value == "0"
        || value.eq_ignore_ascii_case("false")
        || value.eq_ignore_ascii_case("no")
        || value.eq_ignore_ascii_case("off")
    {
        return Some(false);
    }
    None
}

fn is_valid_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn to_hex_lowercase(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp_file(path: &Path, bytes: &[u8]) {
        let mut file = File::create(path).unwrap();
        file.write_all(bytes).unwrap();
        file.flush().unwrap();
    }

    #[test]
    fn invalid_manifest_json_returns_clear_error() {
        let err = ModelManifest::from_json_str("{not-valid-json]").unwrap_err();
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
        assert!(err.to_string().contains("manifest JSON"));
    }

    #[test]
    fn verify_file_sha256_success_wrong_hash_and_truncated() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("model.bin");
        let bytes = b"model-bytes";
        write_temp_file(&path, bytes);

        let expected_hash = to_hex_lowercase(&Sha256::digest(bytes));
        let expected_size = u64::try_from(bytes.len()).unwrap();
        verify_file_sha256(&path, &expected_hash, expected_size).unwrap();

        let wrong_hash = "0000000000000000000000000000000000000000000000000000000000000000";
        let err = verify_file_sha256(&path, wrong_hash, expected_size).unwrap_err();
        assert!(matches!(err, SearchError::HashMismatch { .. }));

        let err = verify_file_sha256(&path, &expected_hash, expected_size + 1).unwrap_err();
        assert!(matches!(err, SearchError::HashMismatch { .. }));
    }

    #[test]
    fn lifecycle_state_machine_success_path() {
        let manifest = ModelManifest::potion_128m();
        let mut lifecycle = ModelLifecycle::new(
            manifest,
            DownloadConsent::granted(ConsentSource::Programmatic),
        );

        assert_eq!(lifecycle.state(), &ModelState::NotInstalled);

        lifecycle.begin_download(100).unwrap();
        lifecycle.update_download_progress(40).unwrap();
        lifecycle.begin_verification().unwrap();
        lifecycle.mark_ready();

        assert_eq!(lifecycle.state(), &ModelState::Ready);
    }

    #[test]
    fn lifecycle_state_machine_failure_path() {
        let manifest = ModelManifest::potion_128m();
        let mut lifecycle = ModelLifecycle::new(
            manifest,
            DownloadConsent::granted(ConsentSource::Programmatic),
        );

        lifecycle.begin_download(100).unwrap();
        lifecycle.fail_verification("checksum mismatch");
        assert!(matches!(
            lifecycle.state(),
            ModelState::VerificationFailed { .. }
        ));
    }

    #[test]
    fn download_progress_percent_is_bounded_to_100() {
        let manifest = ModelManifest::minilm_v2();
        let mut lifecycle = ModelLifecycle::new(
            manifest,
            DownloadConsent::granted(ConsentSource::Programmatic),
        );
        lifecycle.begin_download(10).unwrap();
        lifecycle.update_download_progress(10_000).unwrap();

        let progress_pct = match lifecycle.state() {
            ModelState::Downloading { progress_pct, .. } => *progress_pct,
            _ => 0,
        };
        assert!(progress_pct <= 100);
        assert_eq!(progress_pct, 100);
    }

    #[test]
    fn placeholder_checksums_are_rejected_in_release_policy_mode() {
        let manifest = ModelManifest::minilm_v2();
        let err = manifest.validate_checksum_policy_for(true).unwrap_err();
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[test]
    fn cancelled_state_can_recover() {
        let manifest = ModelManifest::potion_128m();
        let mut lifecycle = ModelLifecycle::new(
            manifest,
            DownloadConsent::granted(ConsentSource::Programmatic),
        );
        lifecycle.begin_download(10).unwrap();
        lifecycle.cancel();
        lifecycle.recover_after_cancel().unwrap();
        assert_eq!(lifecycle.state(), &ModelState::NotInstalled);
    }

    #[test]
    fn empty_manifest_catalog_is_valid() {
        let catalog = ModelManifestCatalog::from_json_str(r#"{"models":[]}"#).unwrap();
        assert!(catalog.models.is_empty());
        catalog.validate().unwrap();
    }

    #[test]
    fn unreadable_model_file_returns_clear_error() {
        let temp = tempfile::tempdir().unwrap();
        let model_root = temp.path();
        let bogus_path = model_root.join("tokenizer.json");
        fs::create_dir_all(&bogus_path).unwrap();

        let manifest = ModelManifest {
            id: "test".to_owned(),
            repo: "owner/repo".to_owned(),
            revision: "abcdef1".to_owned(),
            files: vec![ModelFile {
                name: "tokenizer.json".to_owned(),
                sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    .to_owned(),
                size: 1,
            }],
            license: "MIT".to_owned(),
        };

        let err = manifest.verify_dir(model_root).unwrap_err();
        assert!(matches!(err, SearchError::ModelLoadFailed { .. }));
        assert!(err.to_string().contains("regular file"));
    }

    #[test]
    fn can_register_and_lookup_custom_manifest() {
        let unique_id = format!(
            "custom-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let manifest = ModelManifest {
            id: unique_id.clone(),
            repo: "acme/custom".to_owned(),
            revision: "0123456789abcdef0123456789abcdef01234567".to_owned(),
            files: vec![ModelFile {
                name: "weights.bin".to_owned(),
                sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    .to_owned(),
                size: 42,
            }],
            license: "MIT".to_owned(),
        };

        manifest.clone().register().unwrap();
        let loaded = ModelManifest::lookup(&unique_id).unwrap();
        assert_eq!(loaded, manifest);
    }

    #[test]
    fn resolve_download_consent_priority_order() {
        let consent =
            resolve_download_consent_with_env(Some(false), Some("1"), Some(true), Some(true));
        assert_eq!(consent.source, Some(ConsentSource::Programmatic));
        assert!(!consent.granted);

        let consent = resolve_download_consent_with_env(None, Some("1"), Some(false), Some(true));
        assert_eq!(consent.source, Some(ConsentSource::Environment));
        assert!(consent.granted);

        let consent = resolve_download_consent_with_env(None, None, Some(false), Some(true));
        assert_eq!(consent.source, Some(ConsentSource::Interactive));
        assert!(!consent.granted);
    }
}
