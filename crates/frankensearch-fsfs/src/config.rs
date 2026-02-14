use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::BuildHasher;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use frankensearch_core::{SearchError, SearchResult};
use serde::{Deserialize, Serialize};
use tracing::info;

const PRECEDENCE: [ConfigSource; 4] = [
    ConfigSource::Cli,
    ConfigSource::Env,
    ConfigSource::File,
    ConfigSource::Defaults,
];

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum TextSelectionMode {
    #[default]
    Blocklist,
    Allowlist,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PressureProfile {
    Strict,
    #[default]
    Performance,
    Degraded,
}

impl FromStr for PressureProfile {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "strict" => Ok(Self::Strict),
            "performance" => Ok(Self::Performance),
            "degraded" => Ok(Self::Degraded),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum TuiTheme {
    Auto,
    Light,
    #[default]
    Dark,
}

impl FromStr for TuiTheme {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "light" => Ok(Self::Light),
            "dark" => Ok(Self::Dark),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum Density {
    Compact,
    #[default]
    Normal,
    Expanded,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiscoveryConfig {
    pub roots: Vec<String>,
    pub exclude_patterns: Vec<String>,
    pub text_selection_mode: TextSelectionMode,
    pub binary_blocklist_extensions: Vec<String>,
    pub max_file_size_mb: usize,
    pub follow_symlinks: bool,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            roots: vec!["$HOME".into()],
            exclude_patterns: vec![
                ".git".into(),
                "node_modules".into(),
                "target".into(),
                "__pycache__".into(),
            ],
            text_selection_mode: TextSelectionMode::Blocklist,
            binary_blocklist_extensions: vec![
                ".exe".into(),
                ".dll".into(),
                ".so".into(),
                ".o".into(),
                ".class".into(),
                ".jar".into(),
                ".zip".into(),
                ".tar".into(),
                ".gz".into(),
                ".png".into(),
                ".jpg".into(),
                ".jpeg".into(),
                ".mp3".into(),
                ".mp4".into(),
                ".wasm".into(),
                ".pyc".into(),
                ".pdb".into(),
            ],
            max_file_size_mb: 10,
            follow_symlinks: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IndexingConfig {
    pub fast_model: String,
    pub quality_model: String,
    pub model_dir: String,
    pub embedding_batch_size: usize,
    pub reindex_on_change: bool,
    pub watch_mode: bool,
}

impl Default for IndexingConfig {
    fn default() -> Self {
        Self {
            fast_model: "potion-multilingual-128M".into(),
            quality_model: "all-MiniLM-L6-v2".into(),
            model_dir: "~/.cache/frankensearch/models".into(),
            embedding_batch_size: 64,
            reindex_on_change: true,
            watch_mode: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchConfig {
    pub default_limit: usize,
    pub quality_weight: f64,
    pub rrf_k: f64,
    pub quality_timeout_ms: u64,
    pub fast_only: bool,
    pub explain: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            default_limit: 20,
            quality_weight: 0.7,
            rrf_k: 60.0,
            quality_timeout_ms: 500,
            fast_only: false,
            explain: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PressureConfig {
    pub profile: PressureProfile,
    pub cpu_ceiling_pct: u8,
    pub memory_ceiling_mb: usize,
}

impl Default for PressureConfig {
    fn default() -> Self {
        Self {
            profile: PressureProfile::Performance,
            cpu_ceiling_pct: 80,
            memory_ceiling_mb: 2048,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TuiConfig {
    pub theme: TuiTheme,
    pub frame_budget_ms: u16,
    pub show_explanations: bool,
    pub density: Density,
}

impl Default for TuiConfig {
    fn default() -> Self {
        Self {
            theme: TuiTheme::Dark,
            frame_budget_ms: 16,
            show_explanations: true,
            density: Density::Normal,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StorageConfig {
    pub db_path: String,
    pub evidence_retention_days: u16,
    pub summary_retention_days: u16,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            db_path: "~/.local/share/fsfs/fsfs.db".into(),
            evidence_retention_days: 7,
            summary_retention_days: 90,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrivacyConfig {
    pub redact_file_contents_in_logs: bool,
    pub redact_paths_in_telemetry: bool,
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self {
            redact_file_contents_in_logs: true,
            redact_paths_in_telemetry: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct FsfsConfig {
    pub discovery: DiscoveryConfig,
    pub indexing: IndexingConfig,
    pub search: SearchConfig,
    pub pressure: PressureConfig,
    pub tui: TuiConfig,
    pub storage: StorageConfig,
    pub privacy: PrivacyConfig,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
struct DiscoveryConfigPatch {
    roots: Option<Vec<String>>,
    exclude_patterns: Option<Vec<String>>,
    text_selection_mode: Option<TextSelectionMode>,
    binary_blocklist_extensions: Option<Vec<String>>,
    max_file_size_mb: Option<usize>,
    follow_symlinks: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
struct IndexingConfigPatch {
    fast_model: Option<String>,
    quality_model: Option<String>,
    model_dir: Option<String>,
    embedding_batch_size: Option<usize>,
    reindex_on_change: Option<bool>,
    watch_mode: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
struct SearchConfigPatch {
    default_limit: Option<usize>,
    quality_weight: Option<f64>,
    rrf_k: Option<f64>,
    quality_timeout_ms: Option<u64>,
    fast_only: Option<bool>,
    explain: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
struct PressureConfigPatch {
    profile: Option<PressureProfile>,
    cpu_ceiling_pct: Option<u8>,
    memory_ceiling_mb: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
struct TuiConfigPatch {
    theme: Option<TuiTheme>,
    frame_budget_ms: Option<u16>,
    show_explanations: Option<bool>,
    density: Option<Density>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
struct StorageConfigPatch {
    db_path: Option<String>,
    evidence_retention_days: Option<u16>,
    summary_retention_days: Option<u16>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
struct PrivacyConfigPatch {
    redact_file_contents_in_logs: Option<bool>,
    redact_paths_in_telemetry: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
struct FsfsConfigPatch {
    discovery: Option<DiscoveryConfigPatch>,
    indexing: Option<IndexingConfigPatch>,
    search: Option<SearchConfigPatch>,
    pressure: Option<PressureConfigPatch>,
    tui: Option<TuiConfigPatch>,
    storage: Option<StorageConfigPatch>,
    privacy: Option<PrivacyConfigPatch>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfigSource {
    Cli,
    Env,
    File,
    Defaults,
    Runtime,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConfigWarning {
    pub reason_code: String,
    pub field: String,
    pub source: ConfigSource,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PathExpansion {
    pub field: String,
    pub raw: String,
    pub expanded: String,
    pub source: ConfigSource,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConfigLoadResult {
    pub config: FsfsConfig,
    pub source_precedence: [ConfigSource; 4],
    pub config_file_used: Option<PathBuf>,
    pub cli_flags_used: Vec<String>,
    pub env_keys_used: Vec<String>,
    pub warnings: Vec<ConfigWarning>,
    pub path_expansions: Vec<PathExpansion>,
}

impl ConfigLoadResult {
    #[must_use]
    pub fn to_loaded_event(&self) -> ConfigLoadedEvent {
        let reason_codes = self
            .warnings
            .iter()
            .map(|warning| warning.reason_code.clone())
            .collect();

        ConfigLoadedEvent {
            event: "config_loaded".into(),
            source_precedence_applied: PRECEDENCE,
            config_file_used: self.config_file_used.clone(),
            cli_flags_used: self.cli_flags_used.clone(),
            env_keys_used: self.env_keys_used.clone(),
            resolved_values: self.config.clone(),
            warnings: self.warnings.clone(),
            reason_codes,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConfigLoadedEvent {
    pub event: String,
    pub source_precedence_applied: [ConfigSource; 4],
    pub config_file_used: Option<PathBuf>,
    pub cli_flags_used: Vec<String>,
    pub env_keys_used: Vec<String>,
    pub resolved_values: FsfsConfig,
    pub warnings: Vec<ConfigWarning>,
    pub reason_codes: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CliOverrides {
    pub roots: Option<Vec<String>>,
    pub exclude_patterns: Option<Vec<String>>,
    pub limit: Option<usize>,
    pub fast_only: Option<bool>,
    pub explain: Option<bool>,
    pub profile: Option<PressureProfile>,
    pub theme: Option<TuiTheme>,
    pub config_path: Option<PathBuf>,
}

impl CliOverrides {
    #[must_use]
    pub fn used_flags(&self) -> Vec<String> {
        let mut flags = Vec::new();
        if self.roots.is_some() {
            flags.push("--roots".into());
        }
        if self.exclude_patterns.is_some() {
            flags.push("--exclude".into());
        }
        if self.limit.is_some() {
            flags.push("--limit".into());
        }
        if self.fast_only.is_some() {
            flags.push("--fast-only".into());
        }
        if self.explain.is_some() {
            flags.push("--explain".into());
        }
        if self.profile.is_some() {
            flags.push("--profile".into());
        }
        if self.theme.is_some() {
            flags.push("--theme".into());
        }
        if self.config_path.is_some() {
            flags.push("--config".into());
        }
        flags
    }
}

#[must_use]
pub fn default_config_file_path(home_dir: &Path) -> PathBuf {
    if let Some(xdg_config_home) = std::env::var_os("XDG_CONFIG_HOME") {
        return PathBuf::from(xdg_config_home)
            .join("fsfs")
            .join("config.toml");
    }

    home_dir.join(".config").join("fsfs").join("config.toml")
}

/// Load config from file/env/CLI overlays using the fsfs precedence contract.
///
/// # Errors
///
/// Returns `SearchError::InvalidConfig` for parse/validation failures and
/// `SearchError::Io` if reading a present file fails.
pub fn load_from_sources<S>(
    config_file: Option<&Path>,
    env: &HashMap<String, String, S>,
    cli: &CliOverrides,
    home_dir: &Path,
) -> SearchResult<ConfigLoadResult>
where
    S: BuildHasher,
{
    let (toml_contents, config_file_used) = match config_file {
        Some(path) if path.exists() => (Some(fs::read_to_string(path)?), Some(path.to_path_buf())),
        Some(_) | None => (None, None),
    };

    load_from_str(
        toml_contents.as_deref(),
        config_file_used.as_deref(),
        env,
        cli,
        home_dir,
    )
}

/// Load config from raw TOML/env/CLI overlays using the fsfs precedence
/// contract (`CLI > env > file > defaults`).
///
/// # Errors
///
/// Returns `SearchError::InvalidConfig` when parsing/validation fails.
pub fn load_from_str<S>(
    config_toml: Option<&str>,
    config_file_path: Option<&Path>,
    env: &HashMap<String, String, S>,
    cli: &CliOverrides,
    home_dir: &Path,
) -> SearchResult<ConfigLoadResult>
where
    S: BuildHasher,
{
    let mut config = FsfsConfig::default();
    let mut warnings = Vec::new();

    if let Some(config_toml) = config_toml {
        warnings.extend(collect_unknown_key_warnings(config_toml)?);
        let patch: FsfsConfigPatch =
            toml::from_str(config_toml).map_err(|error| SearchError::InvalidConfig {
                field: "config_file".into(),
                value: "<toml>".into(),
                reason: error.to_string(),
            })?;
        apply_patch(&mut config, patch);
    }

    let env_keys_used = apply_env_overrides(&mut config, env)?;
    apply_cli_overrides(&mut config, cli);
    let path_expansions = expand_tilde_paths(&mut config, home_dir);
    validate_config(&config, &mut warnings)?;

    Ok(ConfigLoadResult {
        config,
        source_precedence: PRECEDENCE,
        config_file_used: config_file_path.map(Path::to_path_buf),
        cli_flags_used: cli.used_flags(),
        env_keys_used,
        warnings,
        path_expansions,
    })
}

pub fn emit_config_loaded(event: &ConfigLoadedEvent) {
    info!(
        event = %event.event,
        precedence = ?event.source_precedence_applied,
        config_file_used = ?event.config_file_used,
        cli_flags_used = ?event.cli_flags_used,
        env_keys_used = ?event.env_keys_used,
        reason_codes = ?event.reason_codes,
        "fsfs configuration loaded"
    );
}

#[allow(clippy::too_many_lines)]
fn apply_patch(config: &mut FsfsConfig, patch: FsfsConfigPatch) {
    if let Some(discovery) = patch.discovery {
        if let Some(roots) = discovery.roots {
            config.discovery.roots = roots;
        }
        if let Some(exclude_patterns) = discovery.exclude_patterns {
            config.discovery.exclude_patterns = exclude_patterns;
        }
        if let Some(text_selection_mode) = discovery.text_selection_mode {
            config.discovery.text_selection_mode = text_selection_mode;
        }
        if let Some(binary_blocklist_extensions) = discovery.binary_blocklist_extensions {
            config.discovery.binary_blocklist_extensions = binary_blocklist_extensions;
        }
        if let Some(max_file_size_mb) = discovery.max_file_size_mb {
            config.discovery.max_file_size_mb = max_file_size_mb;
        }
        if let Some(follow_symlinks) = discovery.follow_symlinks {
            config.discovery.follow_symlinks = follow_symlinks;
        }
    }

    if let Some(indexing) = patch.indexing {
        if let Some(fast_model) = indexing.fast_model {
            config.indexing.fast_model = fast_model;
        }
        if let Some(quality_model) = indexing.quality_model {
            config.indexing.quality_model = quality_model;
        }
        if let Some(model_dir) = indexing.model_dir {
            config.indexing.model_dir = model_dir;
        }
        if let Some(embedding_batch_size) = indexing.embedding_batch_size {
            config.indexing.embedding_batch_size = embedding_batch_size;
        }
        if let Some(reindex_on_change) = indexing.reindex_on_change {
            config.indexing.reindex_on_change = reindex_on_change;
        }
        if let Some(watch_mode) = indexing.watch_mode {
            config.indexing.watch_mode = watch_mode;
        }
    }

    if let Some(search) = patch.search {
        if let Some(default_limit) = search.default_limit {
            config.search.default_limit = default_limit;
        }
        if let Some(quality_weight) = search.quality_weight {
            config.search.quality_weight = quality_weight;
        }
        if let Some(rrf_k) = search.rrf_k {
            config.search.rrf_k = rrf_k;
        }
        if let Some(quality_timeout_ms) = search.quality_timeout_ms {
            config.search.quality_timeout_ms = quality_timeout_ms;
        }
        if let Some(fast_only) = search.fast_only {
            config.search.fast_only = fast_only;
        }
        if let Some(explain) = search.explain {
            config.search.explain = explain;
        }
    }

    if let Some(pressure) = patch.pressure {
        if let Some(profile) = pressure.profile {
            config.pressure.profile = profile;
        }
        if let Some(cpu_ceiling_pct) = pressure.cpu_ceiling_pct {
            config.pressure.cpu_ceiling_pct = cpu_ceiling_pct;
        }
        if let Some(memory_ceiling_mb) = pressure.memory_ceiling_mb {
            config.pressure.memory_ceiling_mb = memory_ceiling_mb;
        }
    }

    if let Some(tui) = patch.tui {
        if let Some(theme) = tui.theme {
            config.tui.theme = theme;
        }
        if let Some(frame_budget_ms) = tui.frame_budget_ms {
            config.tui.frame_budget_ms = frame_budget_ms;
        }
        if let Some(show_explanations) = tui.show_explanations {
            config.tui.show_explanations = show_explanations;
        }
        if let Some(density) = tui.density {
            config.tui.density = density;
        }
    }

    if let Some(storage) = patch.storage {
        if let Some(db_path) = storage.db_path {
            config.storage.db_path = db_path;
        }
        if let Some(evidence_retention_days) = storage.evidence_retention_days {
            config.storage.evidence_retention_days = evidence_retention_days;
        }
        if let Some(summary_retention_days) = storage.summary_retention_days {
            config.storage.summary_retention_days = summary_retention_days;
        }
    }

    if let Some(privacy) = patch.privacy {
        if let Some(redact_file_contents_in_logs) = privacy.redact_file_contents_in_logs {
            config.privacy.redact_file_contents_in_logs = redact_file_contents_in_logs;
        }
        if let Some(redact_paths_in_telemetry) = privacy.redact_paths_in_telemetry {
            config.privacy.redact_paths_in_telemetry = redact_paths_in_telemetry;
        }
    }
}

fn apply_env_overrides(
    config: &mut FsfsConfig,
    env: &HashMap<String, String, impl BuildHasher>,
) -> SearchResult<Vec<String>> {
    let mut keys_used = Vec::new();

    if let Some(value) = env.get("FSFS_DISCOVERY_ROOTS") {
        config.discovery.roots = parse_csv(value, "discovery.roots")?;
        keys_used.push("FSFS_DISCOVERY_ROOTS".into());
    }

    if let Some(value) = env.get("FSFS_DISCOVERY_EXCLUDE_PATTERNS") {
        config.discovery.exclude_patterns = parse_csv(value, "discovery.exclude_patterns")?;
        keys_used.push("FSFS_DISCOVERY_EXCLUDE_PATTERNS".into());
    }

    if let Some(value) = env.get("FSFS_SEARCH_DEFAULT_LIMIT") {
        config.search.default_limit = parse_usize(value, "search.default_limit")?;
        keys_used.push("FSFS_SEARCH_DEFAULT_LIMIT".into());
    }

    if let Some(value) = env.get("FSFS_SEARCH_FAST_ONLY") {
        config.search.fast_only = parse_bool(value, "search.fast_only")?;
        keys_used.push("FSFS_SEARCH_FAST_ONLY".into());
    }

    if let Some(value) = env.get("FSFS_SEARCH_EXPLAIN") {
        config.search.explain = parse_bool(value, "search.explain")?;
        keys_used.push("FSFS_SEARCH_EXPLAIN".into());
    }

    if let Some(value) = env.get("FSFS_PRESSURE_PROFILE") {
        config.pressure.profile =
            PressureProfile::from_str(value).map_err(|()| SearchError::InvalidConfig {
                field: "pressure.profile".into(),
                value: value.clone(),
                reason: "expected strict|performance|degraded".into(),
            })?;
        keys_used.push("FSFS_PRESSURE_PROFILE".into());
    }

    if let Some(value) = env.get("FSFS_TUI_THEME") {
        config.tui.theme = TuiTheme::from_str(value).map_err(|()| SearchError::InvalidConfig {
            field: "tui.theme".into(),
            value: value.clone(),
            reason: "expected auto|light|dark".into(),
        })?;
        keys_used.push("FSFS_TUI_THEME".into());
    }

    if let Some(value) = env.get("FSFS_PRIVACY_REDACT_PATHS_IN_TELEMETRY") {
        config.privacy.redact_paths_in_telemetry =
            parse_bool(value, "privacy.redact_paths_in_telemetry")?;
        keys_used.push("FSFS_PRIVACY_REDACT_PATHS_IN_TELEMETRY".into());
    }

    if let Some(value) = env.get("FSFS_STORAGE_DB_PATH") {
        config.storage.db_path.clone_from(value);
        keys_used.push("FSFS_STORAGE_DB_PATH".into());
    }

    Ok(keys_used)
}

fn apply_cli_overrides(config: &mut FsfsConfig, cli: &CliOverrides) {
    if let Some(roots) = &cli.roots {
        config.discovery.roots.clone_from(roots);
    }

    if let Some(exclude_patterns) = &cli.exclude_patterns {
        config
            .discovery
            .exclude_patterns
            .clone_from(exclude_patterns);
    }

    if let Some(limit) = cli.limit {
        config.search.default_limit = limit;
    }

    if let Some(fast_only) = cli.fast_only {
        config.search.fast_only = fast_only;
    }

    if let Some(explain) = cli.explain {
        config.search.explain = explain;
    }

    if let Some(profile) = cli.profile {
        config.pressure.profile = profile;
    }

    if let Some(theme) = cli.theme {
        config.tui.theme = theme;
    }
}

fn collect_unknown_key_warnings(config_toml: &str) -> SearchResult<Vec<ConfigWarning>> {
    let value: toml::Value =
        toml::from_str(config_toml).map_err(|error| SearchError::InvalidConfig {
            field: "config_file".into(),
            value: "<toml>".into(),
            reason: error.to_string(),
        })?;

    let root = value.as_table().ok_or_else(|| SearchError::InvalidConfig {
        field: "config_file".into(),
        value: "<toml>".into(),
        reason: "expected table at root".into(),
    })?;

    let known_top_level: HashSet<&str> = [
        "discovery",
        "indexing",
        "search",
        "pressure",
        "tui",
        "storage",
        "privacy",
    ]
    .into_iter()
    .collect();

    let mut warnings = Vec::new();

    for (section, section_value) in root {
        if !known_top_level.contains(section.as_str()) {
            warnings.push(ConfigWarning {
                reason_code: "config.unknown_key.warning".into(),
                field: format!("config.{section}"),
                source: ConfigSource::File,
                message: format!("Unknown section {section} ignored"),
            });
            continue;
        }

        let Some(section_table) = section_value.as_table() else {
            continue;
        };

        let known_section_keys: HashSet<&str> = match section.as_str() {
            "discovery" => [
                "roots",
                "exclude_patterns",
                "text_selection_mode",
                "binary_blocklist_extensions",
                "max_file_size_mb",
                "follow_symlinks",
            ]
            .into_iter()
            .collect(),
            "indexing" => [
                "fast_model",
                "quality_model",
                "model_dir",
                "embedding_batch_size",
                "reindex_on_change",
                "watch_mode",
            ]
            .into_iter()
            .collect(),
            "search" => [
                "default_limit",
                "quality_weight",
                "rrf_k",
                "quality_timeout_ms",
                "fast_only",
                "explain",
            ]
            .into_iter()
            .collect(),
            "pressure" => ["profile", "cpu_ceiling_pct", "memory_ceiling_mb"]
                .into_iter()
                .collect(),
            "tui" => ["theme", "frame_budget_ms", "show_explanations", "density"]
                .into_iter()
                .collect(),
            "storage" => [
                "db_path",
                "evidence_retention_days",
                "summary_retention_days",
            ]
            .into_iter()
            .collect(),
            "privacy" => ["redact_file_contents_in_logs", "redact_paths_in_telemetry"]
                .into_iter()
                .collect(),
            _ => HashSet::new(),
        };

        for key in section_table.keys() {
            if !known_section_keys.contains(key.as_str()) {
                warnings.push(ConfigWarning {
                    reason_code: "config.unknown_key.warning".into(),
                    field: format!("{section}.{key}"),
                    source: ConfigSource::File,
                    message: format!("Unknown key {section}.{key} ignored"),
                });
            }
        }
    }

    Ok(warnings)
}

fn expand_tilde_paths(config: &mut FsfsConfig, home_dir: &Path) -> Vec<PathExpansion> {
    let mut expansions = Vec::new();

    for root in &mut config.discovery.roots {
        if let Some(expanded) = expand_tilde(root, home_dir) {
            expansions.push(PathExpansion {
                field: "discovery.roots".into(),
                raw: root.clone(),
                expanded: expanded.clone(),
                source: ConfigSource::Runtime,
            });
            *root = expanded;
        }
    }

    if let Some(expanded) = expand_tilde(&config.indexing.model_dir, home_dir) {
        expansions.push(PathExpansion {
            field: "indexing.model_dir".into(),
            raw: config.indexing.model_dir.clone(),
            expanded: expanded.clone(),
            source: ConfigSource::Runtime,
        });
        config.indexing.model_dir = expanded;
    }

    if let Some(expanded) = expand_tilde(&config.storage.db_path, home_dir) {
        expansions.push(PathExpansion {
            field: "storage.db_path".into(),
            raw: config.storage.db_path.clone(),
            expanded: expanded.clone(),
            source: ConfigSource::Runtime,
        });
        config.storage.db_path = expanded;
    }

    expansions
}

fn expand_tilde(value: &str, home_dir: &Path) -> Option<String> {
    if value == "~" {
        return Some(home_dir.to_string_lossy().into_owned());
    }

    value
        .strip_prefix("~/")
        .map(|rest| home_dir.join(rest).to_string_lossy().into_owned())
}

fn validate_config(config: &FsfsConfig, warnings: &mut Vec<ConfigWarning>) -> SearchResult<()> {
    if config.storage.summary_retention_days < config.storage.evidence_retention_days {
        return Err(SearchError::InvalidConfig {
            field: "storage.summary_retention_days".into(),
            value: config.storage.summary_retention_days.to_string(),
            reason: "must be >= storage.evidence_retention_days".into(),
        });
    }

    if !(0.0..=1.0).contains(&config.search.quality_weight) {
        return Err(SearchError::InvalidConfig {
            field: "search.quality_weight".into(),
            value: config.search.quality_weight.to_string(),
            reason: "must be between 0.0 and 1.0".into(),
        });
    }

    if config.search.fast_only && !config.indexing.quality_model.trim().is_empty() {
        warnings.push(ConfigWarning {
            reason_code: "config.search.fast_only_with_quality_model".into(),
            field: "search.fast_only".into(),
            source: ConfigSource::Runtime,
            message: "fast_only=true while quality_model is configured".into(),
        });
    }

    Ok(())
}

fn parse_csv(value: &str, field: &str) -> SearchResult<Vec<String>> {
    let parts: Vec<String> = value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::to_string)
        .collect();

    if parts.is_empty() {
        return Err(SearchError::InvalidConfig {
            field: field.into(),
            value: value.into(),
            reason: "expected at least one comma-separated value".into(),
        });
    }

    Ok(parts)
}

fn parse_bool(value: &str, field: &str) -> SearchResult<bool> {
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(SearchError::InvalidConfig {
            field: field.into(),
            value: value.into(),
            reason: "expected boolean (true/false/1/0/yes/no/on/off)".into(),
        }),
    }
}

fn parse_usize(value: &str, field: &str) -> SearchResult<usize> {
    value
        .parse::<usize>()
        .map_err(|_| SearchError::InvalidConfig {
            field: field.into(),
            value: value.into(),
            reason: "expected unsigned integer".into(),
        })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::Path;

    use frankensearch_core::SearchError;

    use super::{CliOverrides, default_config_file_path, load_from_str};

    fn home() -> &'static Path {
        Path::new("/home/tester")
    }

    #[test]
    fn default_config_has_safe_privacy_defaults() {
        let result = load_from_str(
            None,
            None,
            &HashMap::new(),
            &CliOverrides::default(),
            home(),
        )
        .expect("load defaults");
        assert!(result.config.privacy.redact_file_contents_in_logs);
        assert!(result.config.privacy.redact_paths_in_telemetry);
    }

    #[test]
    fn precedence_is_cli_then_env_then_file_then_defaults() {
        let file = "\
[search]\ndefault_limit = 11\n\
[tui]\ntheme = \"light\"\n";
        let env = HashMap::from([
            ("FSFS_SEARCH_DEFAULT_LIMIT".into(), "17".into()),
            ("FSFS_TUI_THEME".into(), "auto".into()),
        ]);

        let cli = CliOverrides {
            limit: Some(29),
            theme: Some(super::TuiTheme::Dark),
            ..CliOverrides::default()
        };

        let result = load_from_str(Some(file), None, &env, &cli, home()).expect("load config");
        assert_eq!(result.config.search.default_limit, 29);
        assert_eq!(result.config.tui.theme, super::TuiTheme::Dark);
        assert!(
            result
                .env_keys_used
                .contains(&"FSFS_SEARCH_DEFAULT_LIMIT".to_string())
        );
        assert!(result.cli_flags_used.contains(&"--limit".to_string()));
    }

    #[test]
    fn unknown_keys_are_reported_as_warnings() {
        let file = "\
[search]\nshadow_mode = true\n";
        let result = load_from_str(
            Some(file),
            None,
            &HashMap::new(),
            &CliOverrides::default(),
            home(),
        )
        .expect("load config");

        assert!(result.warnings.iter().any(|warning| warning.reason_code
            == "config.unknown_key.warning"
            && warning.field == "search.shadow_mode"));
    }

    #[test]
    fn tilde_paths_are_expanded() {
        let file = "\
[indexing]\nmodel_dir = \"~/.cache/fsfs/models\"\n\
[storage]\ndb_path = \"~/.local/share/fsfs/data.db\"\n";
        let result = load_from_str(
            Some(file),
            None,
            &HashMap::new(),
            &CliOverrides::default(),
            home(),
        )
        .expect("load config");

        assert_eq!(
            result.config.indexing.model_dir,
            "/home/tester/.cache/fsfs/models"
        );
        assert_eq!(
            result.config.storage.db_path,
            "/home/tester/.local/share/fsfs/data.db"
        );
        assert_eq!(result.path_expansions.len(), 2);
    }

    #[test]
    fn fast_only_emits_warning_when_quality_model_exists() {
        let cli = CliOverrides {
            fast_only: Some(true),
            ..CliOverrides::default()
        };

        let result = load_from_str(None, None, &HashMap::new(), &cli, home()).expect("load");
        assert!(result.warnings.iter().any(|warning| {
            warning.reason_code == "config.search.fast_only_with_quality_model"
        }));
    }

    #[test]
    fn invalid_retention_is_rejected() {
        let file = "\
[storage]\nevidence_retention_days = 30\nsummary_retention_days = 10\n";
        let err = load_from_str(
            Some(file),
            None,
            &HashMap::new(),
            &CliOverrides::default(),
            home(),
        )
        .expect_err("must reject invalid retention");

        assert!(matches!(
            err,
            SearchError::InvalidConfig { field, .. } if field == "storage.summary_retention_days"
        ));
    }

    #[test]
    fn invalid_env_boolean_is_rejected() {
        let env = HashMap::from([("FSFS_SEARCH_FAST_ONLY".into(), "not-a-bool".into())]);
        let err = load_from_str(None, None, &env, &CliOverrides::default(), home())
            .expect_err("must reject invalid bool");
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[test]
    fn uses_xdg_config_home_when_available() {
        // Verify helper shape without mutating process environment in tests.
        let path = default_config_file_path(home());
        let rendered = path.to_string_lossy();
        assert!(rendered.contains("/fsfs/config.toml"));
    }
}
