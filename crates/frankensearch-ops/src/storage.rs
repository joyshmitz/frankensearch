//! FrankenSQLite-backed storage bootstrap for ops telemetry.
//!
//! This module provides the schema contract for the control-plane database
//! (`frankensearch-ops.db`) and a small connection wrapper that applies
//! pragmas, runs migrations, and validates migration checksums.

use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use frankensearch_core::{SearchError, SearchResult};
use fsqlite::{Connection, Row};
use fsqlite_types::value::SqliteValue;
use serde::{Deserialize, Serialize};

/// Current schema version for the ops telemetry database.
pub const OPS_SCHEMA_VERSION: i64 = 1;

#[allow(clippy::needless_raw_string_hashes)]
const OPS_SCHEMA_MIGRATIONS_TABLE_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS ops_schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at_ms INTEGER NOT NULL,
    checksum TEXT NOT NULL,
    reversible INTEGER NOT NULL CHECK (reversible IN (0, 1))
);
"#;

const OPS_SCHEMA_V1_NAME: &str = "ops_telemetry_storage_v1";
const OPS_SCHEMA_V1_CHECKSUM: &str = "ops-schema-v1-20260214";

#[allow(clippy::needless_raw_string_hashes)]
const OPS_SCHEMA_V1_STATEMENTS: &[&str] = &[
    r#"
CREATE TABLE IF NOT EXISTS projects (
    project_key TEXT PRIMARY KEY,
    display_name TEXT,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL
);
"#,
    r#"
CREATE TABLE IF NOT EXISTS instances (
    instance_id TEXT PRIMARY KEY,
    project_key TEXT NOT NULL REFERENCES projects(project_key) ON DELETE CASCADE,
    host_name TEXT,
    pid INTEGER,
    version TEXT,
    first_seen_ms INTEGER NOT NULL,
    last_heartbeat_ms INTEGER NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('started', 'healthy', 'degraded', 'stale', 'stopped'))
);
"#,
    r#"
CREATE TABLE IF NOT EXISTS search_events (
    event_id TEXT PRIMARY KEY,
    project_key TEXT NOT NULL REFERENCES projects(project_key) ON DELETE CASCADE,
    instance_id TEXT NOT NULL REFERENCES instances(instance_id) ON DELETE CASCADE,
    correlation_id TEXT NOT NULL,
    query_hash TEXT,
    query_class TEXT,
    phase TEXT NOT NULL CHECK (phase IN ('initial', 'refined', 'failed')),
    latency_us INTEGER NOT NULL,
    result_count INTEGER,
    memory_bytes INTEGER,
    ts_ms INTEGER NOT NULL
);
"#,
    r#"
CREATE TABLE IF NOT EXISTS search_summaries (
    project_key TEXT NOT NULL REFERENCES projects(project_key) ON DELETE CASCADE,
    instance_id TEXT NOT NULL REFERENCES instances(instance_id) ON DELETE CASCADE,
    window TEXT NOT NULL CHECK (window IN ('1m', '15m', '1h', '6h', '24h', '3d', '1w')),
    window_start_ms INTEGER NOT NULL,
    search_count INTEGER NOT NULL,
    p50_latency_us INTEGER,
    p95_latency_us INTEGER,
    p99_latency_us INTEGER,
    avg_result_count REAL,
    PRIMARY KEY (project_key, instance_id, window, window_start_ms)
);
"#,
    r#"
CREATE TABLE IF NOT EXISTS embedding_job_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    project_key TEXT NOT NULL REFERENCES projects(project_key) ON DELETE CASCADE,
    instance_id TEXT NOT NULL REFERENCES instances(instance_id) ON DELETE CASCADE,
    embedder_id TEXT NOT NULL,
    pending_jobs INTEGER NOT NULL,
    processing_jobs INTEGER NOT NULL,
    completed_jobs INTEGER NOT NULL,
    failed_jobs INTEGER NOT NULL,
    retried_jobs INTEGER NOT NULL,
    batch_latency_us INTEGER,
    ts_ms INTEGER NOT NULL,
    UNIQUE (project_key, instance_id, embedder_id, ts_ms)
);
"#,
    r#"
CREATE TABLE IF NOT EXISTS index_inventory_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    project_key TEXT NOT NULL REFERENCES projects(project_key) ON DELETE CASCADE,
    instance_id TEXT NOT NULL REFERENCES instances(instance_id) ON DELETE CASCADE,
    index_name TEXT NOT NULL,
    index_type TEXT NOT NULL,
    record_count INTEGER NOT NULL,
    file_size_bytes INTEGER,
    file_hash TEXT,
    is_stale INTEGER NOT NULL CHECK (is_stale IN (0, 1)),
    ts_ms INTEGER NOT NULL,
    UNIQUE (project_key, instance_id, index_name, ts_ms)
);
"#,
    r#"
CREATE TABLE IF NOT EXISTS resource_samples (
    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_key TEXT NOT NULL REFERENCES projects(project_key) ON DELETE CASCADE,
    instance_id TEXT NOT NULL REFERENCES instances(instance_id) ON DELETE CASCADE,
    cpu_pct REAL,
    rss_bytes INTEGER,
    io_read_bytes INTEGER,
    io_write_bytes INTEGER,
    queue_depth INTEGER,
    ts_ms INTEGER NOT NULL,
    UNIQUE (project_key, instance_id, ts_ms)
);
"#,
    r#"
CREATE TABLE IF NOT EXISTS alerts_timeline (
    alert_id TEXT PRIMARY KEY,
    project_key TEXT NOT NULL REFERENCES projects(project_key) ON DELETE CASCADE,
    instance_id TEXT,
    category TEXT NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('info', 'warn', 'error', 'critical')),
    reason_code TEXT NOT NULL,
    summary TEXT,
    state TEXT NOT NULL CHECK (state IN ('open', 'acknowledged', 'resolved')),
    opened_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    resolved_at_ms INTEGER
);
"#,
    r#"
CREATE TABLE IF NOT EXISTS evidence_links (
    link_id TEXT PRIMARY KEY,
    project_key TEXT NOT NULL REFERENCES projects(project_key) ON DELETE CASCADE,
    alert_id TEXT NOT NULL REFERENCES alerts_timeline(alert_id) ON DELETE CASCADE,
    evidence_type TEXT NOT NULL,
    evidence_uri TEXT NOT NULL,
    evidence_hash TEXT,
    created_at_ms INTEGER NOT NULL,
    UNIQUE (alert_id, evidence_uri)
);
"#,
    "CREATE INDEX IF NOT EXISTS ix_inst_pk_hb ON instances(project_key, last_heartbeat_ms DESC);",
    "CREATE INDEX IF NOT EXISTS ix_se_pk_ts ON search_events(project_key, ts_ms DESC);",
    "CREATE INDEX IF NOT EXISTS ix_se_ii_ts ON search_events(instance_id, ts_ms DESC);",
    "CREATE INDEX IF NOT EXISTS ix_se_corr ON search_events(project_key, correlation_id);",
    "CREATE INDEX IF NOT EXISTS ix_ss_pk_w ON search_summaries(project_key, window, window_start_ms DESC);",
    "CREATE INDEX IF NOT EXISTS ix_ejs_pk ON embedding_job_snapshots(project_key, ts_ms DESC);",
    "CREATE INDEX IF NOT EXISTS ix_iis_pk ON index_inventory_snapshots(project_key, ts_ms DESC);",
    "CREATE INDEX IF NOT EXISTS ix_rs_pk ON resource_samples(project_key, ts_ms DESC);",
    "CREATE INDEX IF NOT EXISTS ix_at_pk ON alerts_timeline(project_key, opened_at_ms DESC);",
    "CREATE INDEX IF NOT EXISTS ix_at_open ON alerts_timeline(project_key, state, severity, updated_at_ms DESC) WHERE state != 'resolved';",
    "CREATE INDEX IF NOT EXISTS ix_el_aid ON evidence_links(alert_id, created_at_ms DESC);",
];

struct OpsMigration {
    version: i64,
    name: &'static str,
    checksum: &'static str,
    reversible: bool,
    statements: &'static [&'static str],
}

const OPS_MIGRATIONS: &[OpsMigration] = &[OpsMigration {
    version: 1,
    name: OPS_SCHEMA_V1_NAME,
    checksum: OPS_SCHEMA_V1_CHECKSUM,
    reversible: true,
    statements: OPS_SCHEMA_V1_STATEMENTS,
}];

/// Configuration for the ops telemetry storage connection.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpsStorageConfig {
    /// Path to the ops telemetry database.
    pub db_path: PathBuf,
    /// Enable `WAL` journaling mode when true.
    pub wal_mode: bool,
    /// `SQLite` busy timeout in milliseconds.
    pub busy_timeout_ms: u64,
    /// `SQLite` cache size in pages.
    pub cache_size_pages: i32,
}

impl OpsStorageConfig {
    /// In-memory configuration useful for unit tests.
    #[must_use]
    pub fn in_memory() -> Self {
        Self {
            db_path: PathBuf::from(":memory:"),
            ..Self::default()
        }
    }
}

impl Default for OpsStorageConfig {
    fn default() -> Self {
        Self {
            db_path: PathBuf::from("frankensearch-ops.db"),
            wal_mode: true,
            busy_timeout_ms: 5_000,
            cache_size_pages: 2_000,
        }
    }
}

/// Search phase classification persisted to `search_events.phase`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchEventPhase {
    Initial,
    Refined,
    Failed,
}

impl SearchEventPhase {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Initial => "initial",
            Self::Refined => "refined",
            Self::Failed => "failed",
        }
    }
}

/// Idempotent write payload for `search_events`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchEventRecord {
    pub event_id: String,
    pub project_key: String,
    pub instance_id: String,
    pub correlation_id: String,
    pub query_hash: Option<String>,
    pub query_class: Option<String>,
    pub phase: SearchEventPhase,
    pub latency_us: u64,
    pub result_count: Option<u64>,
    pub memory_bytes: Option<u64>,
    pub ts_ms: i64,
}

impl SearchEventRecord {
    fn validate(&self) -> SearchResult<()> {
        ensure_non_empty(&self.event_id, "event_id")?;
        ensure_non_empty(&self.project_key, "project_key")?;
        ensure_non_empty(&self.instance_id, "instance_id")?;
        ensure_non_empty(&self.correlation_id, "correlation_id")?;
        if self.ts_ms < 0 {
            return Err(SearchError::InvalidConfig {
                field: "ts_ms".to_owned(),
                value: self.ts_ms.to_string(),
                reason: "must be >= 0".to_owned(),
            });
        }
        let _ = u64_to_i64(self.latency_us, "latency_us")?;
        if let Some(result_count) = self.result_count {
            let _ = u64_to_i64(result_count, "result_count")?;
        }
        if let Some(memory_bytes) = self.memory_bytes {
            let _ = u64_to_i64(memory_bytes, "memory_bytes")?;
        }
        Ok(())
    }
}

/// Upsert payload for `resource_samples`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceSampleRecord {
    pub project_key: String,
    pub instance_id: String,
    pub cpu_pct: Option<f64>,
    pub rss_bytes: Option<u64>,
    pub io_read_bytes: Option<u64>,
    pub io_write_bytes: Option<u64>,
    pub queue_depth: Option<u64>,
    pub ts_ms: i64,
}

impl ResourceSampleRecord {
    fn validate(&self) -> SearchResult<()> {
        ensure_non_empty(&self.project_key, "project_key")?;
        ensure_non_empty(&self.instance_id, "instance_id")?;
        if self.ts_ms < 0 {
            return Err(SearchError::InvalidConfig {
                field: "ts_ms".to_owned(),
                value: self.ts_ms.to_string(),
                reason: "must be >= 0".to_owned(),
            });
        }
        if let Some(rss_bytes) = self.rss_bytes {
            let _ = u64_to_i64(rss_bytes, "rss_bytes")?;
        }
        if let Some(io_read_bytes) = self.io_read_bytes {
            let _ = u64_to_i64(io_read_bytes, "io_read_bytes")?;
        }
        if let Some(io_write_bytes) = self.io_write_bytes {
            let _ = u64_to_i64(io_write_bytes, "io_write_bytes")?;
        }
        if let Some(queue_depth) = self.queue_depth {
            let _ = u64_to_i64(queue_depth, "queue_depth")?;
        }
        Ok(())
    }
}

/// Per-call ingestion accounting for search event batches.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpsIngestBatchResult {
    pub requested: usize,
    pub inserted: usize,
    pub deduplicated: usize,
    pub failed: usize,
    pub queue_depth_before: usize,
    pub queue_depth_after: usize,
    pub write_latency_us: u64,
}

/// Aggregate ingestion metrics for observability.
#[derive(Debug, Default)]
pub struct OpsIngestionMetrics {
    total_batches: AtomicU64,
    total_inserted: AtomicU64,
    total_deduplicated: AtomicU64,
    total_failed_records: AtomicU64,
    total_backpressured_batches: AtomicU64,
    total_write_latency_us: AtomicU64,
    pending_events: AtomicUsize,
    high_watermark_pending_events: AtomicUsize,
}

/// Snapshot of aggregate ingestion counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpsIngestionMetricsSnapshot {
    pub total_batches: u64,
    pub total_inserted: u64,
    pub total_deduplicated: u64,
    pub total_failed_records: u64,
    pub total_backpressured_batches: u64,
    pub total_write_latency_us: u64,
    pub pending_events: usize,
    pub high_watermark_pending_events: usize,
}

impl OpsIngestionMetrics {
    #[must_use]
    pub fn snapshot(&self) -> OpsIngestionMetricsSnapshot {
        OpsIngestionMetricsSnapshot {
            total_batches: self.total_batches.load(Ordering::Relaxed),
            total_inserted: self.total_inserted.load(Ordering::Relaxed),
            total_deduplicated: self.total_deduplicated.load(Ordering::Relaxed),
            total_failed_records: self.total_failed_records.load(Ordering::Relaxed),
            total_backpressured_batches: self.total_backpressured_batches.load(Ordering::Relaxed),
            total_write_latency_us: self.total_write_latency_us.load(Ordering::Relaxed),
            pending_events: self.pending_events.load(Ordering::Relaxed),
            high_watermark_pending_events: self
                .high_watermark_pending_events
                .load(Ordering::Relaxed),
        }
    }

    fn update_high_watermark(&self, candidate: usize) {
        let mut current = self.high_watermark_pending_events.load(Ordering::Relaxed);
        while candidate > current {
            match self.high_watermark_pending_events.compare_exchange_weak(
                current,
                candidate,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(observed) => current = observed,
            }
        }
    }
}

/// Connection wrapper for ops telemetry storage.
pub struct OpsStorage {
    conn: Connection,
    config: OpsStorageConfig,
    ingestion_metrics: Arc<OpsIngestionMetrics>,
}

impl std::fmt::Debug for OpsStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpsStorage")
            .field("path", &self.config.db_path)
            .field("wal_mode", &self.config.wal_mode)
            .field("busy_timeout_ms", &self.config.busy_timeout_ms)
            .field("cache_size_pages", &self.config.cache_size_pages)
            .field(
                "pending_ingest_events",
                &self.ingestion_metrics.snapshot().pending_events,
            )
            .finish_non_exhaustive()
    }
}

impl OpsStorage {
    /// Open storage and bootstrap schema if needed.
    ///
    /// # Errors
    ///
    /// Returns an error if the database connection cannot be opened,
    /// pragmas fail to apply, or schema bootstrap/migration fails.
    pub fn open(config: OpsStorageConfig) -> SearchResult<Self> {
        tracing::debug!(
            target: "frankensearch.ops.storage",
            path = %config.db_path.display(),
            wal_mode = config.wal_mode,
            busy_timeout_ms = config.busy_timeout_ms,
            cache_size_pages = config.cache_size_pages,
            "opening ops storage connection"
        );

        let conn =
            Connection::open(config.db_path.to_string_lossy().to_string()).map_err(ops_error)?;
        let storage = Self {
            conn,
            config,
            ingestion_metrics: Arc::new(OpsIngestionMetrics::default()),
        };
        storage.apply_pragmas()?;
        bootstrap(storage.connection())?;
        Ok(storage)
    }

    /// Open in-memory storage and bootstrap schema.
    ///
    /// # Errors
    ///
    /// Returns an error if in-memory database bootstrap fails.
    pub fn open_in_memory() -> SearchResult<Self> {
        Self::open(OpsStorageConfig::in_memory())
    }

    /// Underlying database connection.
    #[must_use]
    pub const fn connection(&self) -> &Connection {
        &self.conn
    }

    /// Runtime configuration used by this storage handle.
    #[must_use]
    pub const fn config(&self) -> &OpsStorageConfig {
        &self.config
    }

    /// Current schema version.
    ///
    /// # Errors
    ///
    /// Returns an error if schema metadata cannot be read.
    pub fn current_schema_version(&self) -> SearchResult<i64> {
        current_version(self.connection())
    }

    /// Current ingestion metrics snapshot used by dashboards and tests.
    #[must_use]
    pub fn ingestion_metrics(&self) -> OpsIngestionMetricsSnapshot {
        self.ingestion_metrics.snapshot()
    }

    /// Insert one search event with idempotent semantics.
    ///
    /// This is equivalent to calling [`Self::ingest_search_events_batch`] with
    /// a single payload.
    ///
    /// # Errors
    ///
    /// Returns an error when validation fails, backpressure rejects the write,
    /// or database I/O fails.
    pub fn ingest_search_event(
        &self,
        event: &SearchEventRecord,
        backpressure_threshold: usize,
    ) -> SearchResult<OpsIngestBatchResult> {
        self.ingest_search_events_batch(std::slice::from_ref(event), backpressure_threshold)
    }

    /// Insert a batch of search events atomically.
    ///
    /// `event_id` is treated as an idempotency key. Duplicate IDs are counted
    /// as deduplicated records instead of failures.
    ///
    /// # Errors
    ///
    /// Returns an error if backpressure is active, payload validation fails, or
    /// database operations fail. On error, the full batch is rolled back.
    pub fn ingest_search_events_batch(
        &self,
        events: &[SearchEventRecord],
        backpressure_threshold: usize,
    ) -> SearchResult<OpsIngestBatchResult> {
        if events.is_empty() {
            return Ok(OpsIngestBatchResult::default());
        }
        if backpressure_threshold == 0 {
            return Err(SearchError::InvalidConfig {
                field: "backpressure_threshold".to_owned(),
                value: "0".to_owned(),
                reason: "must be > 0".to_owned(),
            });
        }

        let requested = events.len();
        let queue_depth_before = self
            .ingestion_metrics
            .pending_events
            .fetch_add(requested, Ordering::Relaxed);
        let queue_depth_with_reservation = queue_depth_before.saturating_add(requested);
        self.ingestion_metrics
            .update_high_watermark(queue_depth_with_reservation);

        if queue_depth_with_reservation > backpressure_threshold {
            self.ingestion_metrics
                .pending_events
                .fetch_sub(requested, Ordering::Relaxed);
            self.ingestion_metrics
                .total_backpressured_batches
                .fetch_add(1, Ordering::Relaxed);
            return Err(SearchError::QueueFull {
                pending: queue_depth_with_reservation,
                capacity: backpressure_threshold,
            });
        }

        let started = Instant::now();
        let ingest_result = self.with_transaction(|conn| {
            let mut inserted = 0_usize;
            let mut deduplicated = 0_usize;

            for event in events {
                event.validate()?;
                if insert_search_event_row(conn, event)? {
                    inserted = inserted.saturating_add(1);
                } else {
                    deduplicated = deduplicated.saturating_add(1);
                }
            }
            Ok((inserted, deduplicated))
        });

        let write_latency_us = duration_as_u64(started.elapsed().as_micros());
        let queue_depth_after = self
            .ingestion_metrics
            .pending_events
            .fetch_sub(requested, Ordering::Relaxed)
            .saturating_sub(requested);

        self.ingestion_metrics
            .total_batches
            .fetch_add(1, Ordering::Relaxed);
        self.ingestion_metrics
            .total_write_latency_us
            .fetch_add(write_latency_us, Ordering::Relaxed);

        match ingest_result {
            Ok((inserted, deduplicated)) => {
                self.ingestion_metrics
                    .total_inserted
                    .fetch_add(usize_to_u64(inserted), Ordering::Relaxed);
                self.ingestion_metrics
                    .total_deduplicated
                    .fetch_add(usize_to_u64(deduplicated), Ordering::Relaxed);

                Ok(OpsIngestBatchResult {
                    requested,
                    inserted,
                    deduplicated,
                    failed: 0,
                    queue_depth_before,
                    queue_depth_after,
                    write_latency_us,
                })
            }
            Err(error) => {
                self.ingestion_metrics
                    .total_failed_records
                    .fetch_add(usize_to_u64(requested), Ordering::Relaxed);
                Err(error)
            }
        }
    }

    /// Upsert a resource sample keyed by `(project_key, instance_id, ts_ms)`.
    ///
    /// # Errors
    ///
    /// Returns an error when validation fails or the database write fails.
    pub fn upsert_resource_sample(&self, sample: &ResourceSampleRecord) -> SearchResult<()> {
        sample.validate()?;
        let conn = self.connection();

        // Manual upsert: FrankenSQLite does not yet support
        // ON CONFLICT(...) DO UPDATE, so we check existence first.
        let key_params = [
            SqliteValue::Text(sample.project_key.clone()),
            SqliteValue::Text(sample.instance_id.clone()),
            SqliteValue::Integer(sample.ts_ms),
        ];
        let existing = conn
            .query_with_params(
                "SELECT sample_id FROM resource_samples \
                 WHERE project_key = ?1 AND instance_id = ?2 AND ts_ms = ?3;",
                &key_params,
            )
            .map_err(ops_error)?;

        if existing.is_empty() {
            let params = [
                SqliteValue::Text(sample.project_key.clone()),
                SqliteValue::Text(sample.instance_id.clone()),
                optional_f64(sample.cpu_pct),
                optional_u64(sample.rss_bytes, "rss_bytes")?,
                optional_u64(sample.io_read_bytes, "io_read_bytes")?,
                optional_u64(sample.io_write_bytes, "io_write_bytes")?,
                optional_u64(sample.queue_depth, "queue_depth")?,
                SqliteValue::Integer(sample.ts_ms),
            ];
            conn.execute_with_params(
                "INSERT INTO resource_samples(\
                    project_key, instance_id, cpu_pct, rss_bytes, io_read_bytes, io_write_bytes, \
                    queue_depth, ts_ms\
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8);",
                &params,
            )
            .map_err(ops_error)?;
        } else {
            let params = [
                optional_f64(sample.cpu_pct),
                optional_u64(sample.rss_bytes, "rss_bytes")?,
                optional_u64(sample.io_read_bytes, "io_read_bytes")?,
                optional_u64(sample.io_write_bytes, "io_write_bytes")?,
                optional_u64(sample.queue_depth, "queue_depth")?,
                SqliteValue::Text(sample.project_key.clone()),
                SqliteValue::Text(sample.instance_id.clone()),
                SqliteValue::Integer(sample.ts_ms),
            ];
            conn.execute_with_params(
                "UPDATE resource_samples SET \
                    cpu_pct = ?1, rss_bytes = ?2, io_read_bytes = ?3, \
                    io_write_bytes = ?4, queue_depth = ?5 \
                 WHERE project_key = ?6 AND instance_id = ?7 AND ts_ms = ?8;",
                &params,
            )
            .map_err(ops_error)?;
        }
        Ok(())
    }

    fn apply_pragmas(&self) -> SearchResult<()> {
        self.conn
            .execute("PRAGMA foreign_keys=ON;")
            .map_err(ops_error)?;
        if self.config.wal_mode {
            self.conn
                .execute("PRAGMA journal_mode=WAL;")
                .map_err(ops_error)?;
        } else if let Err(error) = self.conn.execute("PRAGMA journal_mode=DELETE;") {
            tracing::warn!(
                target: "frankensearch.ops.storage",
                ?error,
                "journal_mode=DELETE was not accepted; falling back to WAL"
            );
            self.conn
                .execute("PRAGMA journal_mode=WAL;")
                .map_err(ops_error)?;
        }

        self.conn
            .execute(&format!(
                "PRAGMA busy_timeout={};",
                self.config.busy_timeout_ms
            ))
            .map_err(ops_error)?;
        self.conn
            .execute(&format!(
                "PRAGMA cache_size={};",
                self.config.cache_size_pages
            ))
            .map_err(ops_error)?;

        Ok(())
    }

    fn with_transaction<T, F>(&self, operation: F) -> SearchResult<T>
    where
        F: FnOnce(&Connection) -> SearchResult<T>,
    {
        self.connection().execute("BEGIN;").map_err(ops_error)?;
        let result = operation(self.connection());
        match result {
            Ok(value) => {
                self.connection().execute("COMMIT;").map_err(ops_error)?;
                Ok(value)
            }
            Err(error) => {
                let _ignored = self.connection().execute("ROLLBACK;");
                Err(error)
            }
        }
    }
}

/// Bootstrap ops schema to the latest supported version.
///
/// # Errors
///
/// Returns an error if migration metadata cannot be read, any migration fails,
/// checksums do not match, or an unsupported schema version is detected.
pub fn bootstrap(conn: &Connection) -> SearchResult<()> {
    conn.execute(OPS_SCHEMA_MIGRATIONS_TABLE_SQL)
        .map_err(ops_error)?;

    let mut version = current_version_optional(conn)?.unwrap_or(0);
    if version > OPS_SCHEMA_VERSION {
        return Err(SearchError::SubsystemError {
            subsystem: "ops-storage",
            source: Box::new(io::Error::other(format!(
                "ops schema version {version} is newer than supported {OPS_SCHEMA_VERSION}"
            ))),
        });
    }

    for migration in OPS_MIGRATIONS {
        if migration.version <= version {
            continue;
        }
        apply_migration(conn, migration)?;
        version = migration.version;
    }

    validate_migration_checksums(conn)?;
    Ok(())
}

/// Read the latest applied schema version.
///
/// # Errors
///
/// Returns an error if migration metadata cannot be queried or no versions
/// have been recorded.
pub fn current_version(conn: &Connection) -> SearchResult<i64> {
    current_version_optional(conn)?.ok_or_else(|| SearchError::SubsystemError {
        subsystem: "ops-storage",
        source: Box::new(io::Error::other(
            "ops_schema_migrations table has no version rows",
        )),
    })
}

fn apply_migration(conn: &Connection, migration: &OpsMigration) -> SearchResult<()> {
    tracing::debug!(
        target: "frankensearch.ops.storage",
        migration_version = migration.version,
        migration_name = migration.name,
        "applying ops storage migration"
    );

    // Execute DDL statements individually in autocommit mode so that each
    // CREATE TABLE / CREATE INDEX is committed separately.  This prevents
    // the sqlite_master btree page from overflowing when the accumulated
    // DDL text exceeds the 4 KiB page size.
    for statement in migration.statements {
        conn.execute(statement).map_err(ops_error)?;
    }

    // Record the migration metadata.
    let params = [
        SqliteValue::Integer(migration.version),
        SqliteValue::Text(migration.name.to_owned()),
        SqliteValue::Integer(unix_timestamp_ms()?),
        SqliteValue::Text(migration.checksum.to_owned()),
        SqliteValue::Integer(i64::from(migration.reversible)),
    ];
    conn.execute_with_params(
        "INSERT INTO ops_schema_migrations(version, name, applied_at_ms, checksum, reversible) \
         VALUES (?1, ?2, ?3, ?4, ?5);",
        &params,
    )
    .map_err(ops_error)?;
    Ok(())
}

fn validate_migration_checksums(conn: &Connection) -> SearchResult<()> {
    let rows = conn
        .query("SELECT version, checksum FROM ops_schema_migrations ORDER BY version ASC;")
        .map_err(ops_error)?;
    for row in &rows {
        let version = row_i64(row, 0, "ops_schema_migrations.version")?;
        let checksum = row_text(row, 1, "ops_schema_migrations.checksum")?;
        let Some(expected) = expected_checksum(version) else {
            return Err(SearchError::SubsystemError {
                subsystem: "ops-storage",
                source: Box::new(io::Error::other(format!(
                    "unknown ops migration version {version} found in ops_schema_migrations"
                ))),
            });
        };
        if checksum != expected {
            return Err(SearchError::SubsystemError {
                subsystem: "ops-storage",
                source: Box::new(io::Error::other(format!(
                    "checksum mismatch for ops migration {version}: expected {expected}, found \
                     {checksum}"
                ))),
            });
        }
    }
    Ok(())
}

fn current_version_optional(conn: &Connection) -> SearchResult<Option<i64>> {
    let rows = conn
        .query("SELECT version FROM ops_schema_migrations ORDER BY version DESC LIMIT 1;")
        .map_err(ops_error)?;
    let Some(row) = rows.first() else {
        return Ok(None);
    };
    row_i64(row, 0, "ops_schema_migrations.version").map(Some)
}

fn expected_checksum(version: i64) -> Option<&'static str> {
    OPS_MIGRATIONS
        .iter()
        .find(|migration| migration.version == version)
        .map(|migration| migration.checksum)
}

fn row_i64(row: &Row, index: usize, field: &str) -> SearchResult<i64> {
    match row.get(index) {
        Some(SqliteValue::Integer(value)) => Ok(*value),
        Some(other) => Err(SearchError::SubsystemError {
            subsystem: "ops-storage",
            source: Box::new(io::Error::other(format!(
                "unexpected type for {field}: {other:?}"
            ))),
        }),
        None => Err(SearchError::SubsystemError {
            subsystem: "ops-storage",
            source: Box::new(io::Error::other(format!("missing column for {field}"))),
        }),
    }
}

fn row_text<'a>(row: &'a Row, index: usize, field: &str) -> SearchResult<&'a str> {
    match row.get(index) {
        Some(SqliteValue::Text(value)) => Ok(value.as_str()),
        Some(other) => Err(SearchError::SubsystemError {
            subsystem: "ops-storage",
            source: Box::new(io::Error::other(format!(
                "unexpected type for {field}: {other:?}"
            ))),
        }),
        None => Err(SearchError::SubsystemError {
            subsystem: "ops-storage",
            source: Box::new(io::Error::other(format!("missing column for {field}"))),
        }),
    }
}

fn unix_timestamp_ms() -> SearchResult<i64> {
    let since_epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(ops_error)?;
    i64::try_from(since_epoch.as_millis()).map_err(ops_error)
}

fn insert_search_event_row(conn: &Connection, event: &SearchEventRecord) -> SearchResult<bool> {
    // Manual dedup: FrankenSQLite does not yet handle INSERT OR IGNORE
    // correctly on PK conflicts, so we check existence first.
    let existing = conn
        .query_with_params(
            "SELECT 1 FROM search_events WHERE event_id = ?1;",
            &[SqliteValue::Text(event.event_id.clone())],
        )
        .map_err(ops_error)?;
    if !existing.is_empty() {
        return Ok(false);
    }

    let params = [
        SqliteValue::Text(event.event_id.clone()),
        SqliteValue::Text(event.project_key.clone()),
        SqliteValue::Text(event.instance_id.clone()),
        SqliteValue::Text(event.correlation_id.clone()),
        optional_text(event.query_hash.as_deref()),
        optional_text(event.query_class.as_deref()),
        SqliteValue::Text(event.phase.as_str().to_owned()),
        SqliteValue::Integer(u64_to_i64(event.latency_us, "latency_us")?),
        optional_u64(event.result_count, "result_count")?,
        optional_u64(event.memory_bytes, "memory_bytes")?,
        SqliteValue::Integer(event.ts_ms),
    ];

    conn.execute_with_params(
        "INSERT INTO search_events(\
            event_id, project_key, instance_id, correlation_id, query_hash, query_class, \
            phase, latency_us, result_count, memory_bytes, ts_ms\
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11);",
        &params,
    )
    .map_err(ops_error)?;
    Ok(true)
}

fn ensure_non_empty(value: &str, field: &str) -> SearchResult<()> {
    if value.trim().is_empty() {
        return Err(SearchError::InvalidConfig {
            field: field.to_owned(),
            value: value.to_owned(),
            reason: "must be non-empty".to_owned(),
        });
    }
    Ok(())
}

fn optional_text(value: Option<&str>) -> SqliteValue {
    value.map_or(SqliteValue::Null, |text| SqliteValue::Text(text.to_owned()))
}

fn optional_u64(value: Option<u64>, field: &str) -> SearchResult<SqliteValue> {
    value.map_or(Ok(SqliteValue::Null), |number| {
        u64_to_i64(number, field).map(SqliteValue::Integer)
    })
}

fn optional_f64(value: Option<f64>) -> SqliteValue {
    value.map_or(SqliteValue::Null, SqliteValue::Float)
}

fn u64_to_i64(value: u64, field: &str) -> SearchResult<i64> {
    i64::try_from(value).map_err(|_| SearchError::InvalidConfig {
        field: field.to_owned(),
        value: value.to_string(),
        reason: "must fit into signed 64-bit integer".to_owned(),
    })
}

fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn duration_as_u64(value: u128) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn ops_error<E>(source: E) -> SearchError
where
    E: std::error::Error + Send + Sync + 'static,
{
    SearchError::SubsystemError {
        subsystem: "ops-storage",
        source: Box::new(source),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        OPS_SCHEMA_MIGRATIONS_TABLE_SQL, OPS_SCHEMA_VERSION, OpsStorage, ResourceSampleRecord,
        SearchEventPhase, SearchEventRecord, bootstrap, current_version, ops_error,
    };
    use frankensearch_core::SearchError;
    use fsqlite::Connection;
    use fsqlite_types::value::SqliteValue;

    fn table_exists(conn: &Connection, table_name: &str) -> bool {
        let params = [SqliteValue::Text(table_name.to_owned())];
        let rows = conn
            .query_with_params(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?1 LIMIT 1;",
                &params,
            )
            .map_err(ops_error)
            .expect("sqlite_master table query should succeed");
        !rows.is_empty()
    }

    fn index_exists(conn: &Connection, index_name: &str) -> bool {
        let params = [SqliteValue::Text(index_name.to_owned())];
        let rows = conn
            .query_with_params(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND name = ?1 LIMIT 1;",
                &params,
            )
            .map_err(ops_error)
            .expect("sqlite_master index query should succeed");
        !rows.is_empty()
    }

    fn migration_row_count(conn: &Connection) -> i64 {
        let rows = conn
            .query("SELECT COUNT(*) FROM ops_schema_migrations;")
            .map_err(ops_error)
            .expect("count query should succeed");
        let Some(row) = rows.first() else {
            return 0;
        };
        match row.get(0) {
            Some(SqliteValue::Integer(value)) => *value,
            other => panic!("unexpected row type for count: {other:?}"),
        }
    }

    fn search_event_count(conn: &Connection) -> i64 {
        let rows = conn
            .query("SELECT COUNT(*) FROM search_events;")
            .map_err(ops_error)
            .expect("count query should succeed");
        let Some(row) = rows.first() else {
            return 0;
        };
        match row.get(0) {
            Some(SqliteValue::Integer(value)) => *value,
            other => panic!("unexpected row type for count: {other:?}"),
        }
    }

    fn latest_resource_queue_depth(conn: &Connection) -> i64 {
        let rows = conn
            .query(
                "SELECT queue_depth FROM resource_samples \
                 ORDER BY ts_ms DESC LIMIT 1;",
            )
            .map_err(ops_error)
            .expect("queue depth query should succeed");
        let Some(row) = rows.first() else {
            panic!("expected one resource sample row");
        };
        match row.get(0) {
            Some(SqliteValue::Integer(value)) => *value,
            other => panic!("unexpected row type for queue_depth: {other:?}"),
        }
    }

    fn sample_search_event(event_id: &str, ts_ms: i64) -> SearchEventRecord {
        SearchEventRecord {
            event_id: event_id.to_owned(),
            project_key: "project-a".to_owned(),
            instance_id: "instance-a".to_owned(),
            correlation_id: "corr-a".to_owned(),
            query_hash: Some("hash-a".to_owned()),
            query_class: Some("nl".to_owned()),
            phase: SearchEventPhase::Initial,
            latency_us: 1_200,
            result_count: Some(7),
            memory_bytes: Some(8_192),
            ts_ms,
        }
    }

    fn seed_project_and_instance(conn: &Connection) {
        conn.execute(
            "INSERT INTO projects(project_key, display_name, created_at_ms, updated_at_ms) \
             VALUES ('project-a', 'Project A', 1, 1);",
        )
        .expect("project row should insert");
        conn.execute(
            "INSERT INTO instances(\
                instance_id, project_key, host_name, pid, version, first_seen_ms, \
                last_heartbeat_ms, state\
             ) VALUES (\
                'instance-a', 'project-a', 'host-a', 123, '0.1.0', 1, 1, 'healthy'\
             );",
        )
        .expect("instance row should insert");
    }

    #[test]
    fn bootstrap_creates_v1_schema_tables_and_indexes() {
        let conn = Connection::open(":memory:".to_owned()).expect("in-memory connection");

        bootstrap(&conn).expect("bootstrap should succeed");
        assert_eq!(
            current_version(&conn).expect("schema version should be present"),
            OPS_SCHEMA_VERSION
        );

        for table in [
            "projects",
            "instances",
            "search_events",
            "search_summaries",
            "embedding_job_snapshots",
            "index_inventory_snapshots",
            "resource_samples",
            "alerts_timeline",
            "evidence_links",
            "ops_schema_migrations",
        ] {
            assert!(
                table_exists(&conn, table),
                "expected table {table} to exist"
            );
        }

        for index in [
            "ix_inst_pk_hb",
            "ix_se_pk_ts",
            "ix_se_ii_ts",
            "ix_se_corr",
            "ix_ss_pk_w",
            "ix_ejs_pk",
            "ix_iis_pk",
            "ix_rs_pk",
            "ix_at_pk",
            "ix_at_open",
            "ix_el_aid",
        ] {
            assert!(
                index_exists(&conn, index),
                "expected index {index} to exist"
            );
        }
    }

    #[test]
    fn bootstrap_is_idempotent() {
        let conn = Connection::open(":memory:".to_owned()).expect("in-memory connection");

        bootstrap(&conn).expect("first bootstrap should succeed");
        bootstrap(&conn).expect("second bootstrap should succeed");
        bootstrap(&conn).expect("third bootstrap should succeed");
        assert_eq!(
            current_version(&conn).expect("schema version should be present"),
            OPS_SCHEMA_VERSION
        );
        assert_eq!(
            migration_row_count(&conn),
            1,
            "schema should record a single applied migration"
        );
    }

    #[test]
    fn bootstrap_rejects_newer_schema_versions() {
        let conn = Connection::open(":memory:".to_owned()).expect("in-memory connection");
        conn.execute(OPS_SCHEMA_MIGRATIONS_TABLE_SQL)
            .expect("migrations table creation should succeed");
        conn.execute(
            "INSERT INTO ops_schema_migrations(version, name, applied_at_ms, checksum, reversible) \
             VALUES (99, 'future', 0, 'future-checksum', 0);",
        )
        .expect("future migration row should insert");

        let error = bootstrap(&conn).expect_err("newer versions should be rejected");
        let message = error.to_string();
        assert!(
            message.contains("ops schema version 99 is newer than supported"),
            "unexpected error message: {message}"
        );
    }

    #[test]
    fn bootstrap_detects_checksum_mismatch() {
        let conn = Connection::open(":memory:".to_owned()).expect("in-memory connection");
        conn.execute(OPS_SCHEMA_MIGRATIONS_TABLE_SQL)
            .expect("migrations table creation should succeed");
        conn.execute(
            "INSERT INTO ops_schema_migrations(version, name, applied_at_ms, checksum, reversible) \
             VALUES (1, 'ops_telemetry_storage_v1', 0, 'bad-checksum', 1);",
        )
        .expect("mismatch migration row should insert");

        let error = bootstrap(&conn).expect_err("checksum mismatch should fail");
        let message = error.to_string();
        assert!(
            message.contains("checksum mismatch"),
            "unexpected error message: {message}"
        );
    }

    #[test]
    fn open_in_memory_bootstraps_schema() {
        let storage = OpsStorage::open_in_memory().expect("in-memory ops storage should open");
        assert_eq!(
            storage
                .current_schema_version()
                .expect("schema version should load"),
            OPS_SCHEMA_VERSION
        );
    }

    #[test]
    fn ingest_search_events_batch_is_idempotent_and_tracks_metrics() {
        let storage = OpsStorage::open_in_memory().expect("in-memory ops storage should open");
        seed_project_and_instance(storage.connection());
        let event = sample_search_event("event-1", 42);

        let first = storage
            .ingest_search_events_batch(std::slice::from_ref(&event), 64)
            .expect("first ingest should succeed");
        assert_eq!(first.inserted, 1);
        assert_eq!(first.deduplicated, 0);
        assert_eq!(first.failed, 0);
        assert_eq!(first.queue_depth_after, 0);

        let second = storage
            .ingest_search_events_batch(&[event], 64)
            .expect("second ingest should succeed");
        assert_eq!(second.inserted, 0);
        assert_eq!(second.deduplicated, 1);
        assert_eq!(search_event_count(storage.connection()), 1);

        let metrics = storage.ingestion_metrics();
        assert_eq!(metrics.total_batches, 2);
        assert_eq!(metrics.total_inserted, 1);
        assert_eq!(metrics.total_deduplicated, 1);
        assert_eq!(metrics.total_failed_records, 0);
        assert_eq!(metrics.total_backpressured_batches, 0);
    }

    #[test]
    fn ingest_search_events_batch_rejects_when_backpressured() {
        let storage = OpsStorage::open_in_memory().expect("in-memory ops storage should open");
        seed_project_and_instance(storage.connection());
        let event_a = sample_search_event("event-a", 1);
        let event_b = sample_search_event("event-b", 2);

        let error = storage
            .ingest_search_events_batch(&[event_a, event_b], 1)
            .expect_err("batch should be rejected by backpressure threshold");
        assert!(
            matches!(
                error,
                SearchError::QueueFull {
                    pending: 2,
                    capacity: 1
                }
            ),
            "unexpected backpressure error: {error}"
        );
        assert_eq!(search_event_count(storage.connection()), 0);

        let metrics = storage.ingestion_metrics();
        assert_eq!(metrics.total_backpressured_batches, 1);
        assert_eq!(metrics.pending_events, 0);
        assert_eq!(metrics.total_failed_records, 0);
    }

    #[test]
    fn ingest_search_events_batch_rolls_back_on_validation_error() {
        let storage = OpsStorage::open_in_memory().expect("in-memory ops storage should open");
        seed_project_and_instance(storage.connection());

        let valid = sample_search_event("event-valid", 7);
        let invalid = SearchEventRecord {
            event_id: String::new(),
            ..sample_search_event("event-invalid", 8)
        };

        let error = storage
            .ingest_search_events_batch(&[valid, invalid], 64)
            .expect_err("validation failure should abort full batch");
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "event_id"),
            "unexpected validation error: {error}"
        );
        assert_eq!(search_event_count(storage.connection()), 0);

        let metrics = storage.ingestion_metrics();
        assert_eq!(metrics.total_batches, 1);
        assert_eq!(metrics.total_failed_records, 2);
        assert_eq!(metrics.total_inserted, 0);
        assert_eq!(metrics.total_deduplicated, 0);
    }

    #[test]
    fn upsert_resource_sample_replaces_existing_queue_depth() {
        let storage = OpsStorage::open_in_memory().expect("in-memory ops storage should open");
        seed_project_and_instance(storage.connection());

        let first = ResourceSampleRecord {
            project_key: "project-a".to_owned(),
            instance_id: "instance-a".to_owned(),
            cpu_pct: Some(8.0),
            rss_bytes: Some(1_024),
            io_read_bytes: Some(256),
            io_write_bytes: Some(64),
            queue_depth: Some(3),
            ts_ms: 123,
        };
        storage
            .upsert_resource_sample(&first)
            .expect("first resource sample upsert should succeed");

        let second = ResourceSampleRecord {
            queue_depth: Some(9),
            cpu_pct: Some(9.5),
            ..first
        };
        storage
            .upsert_resource_sample(&second)
            .expect("second resource sample upsert should succeed");

        assert_eq!(latest_resource_queue_depth(storage.connection()), 9);
    }

    #[test]
    #[ignore = "FrankenSQLite does not yet enforce CHECK constraints"]
    fn search_summaries_window_check_rejects_invalid_values() {
        let conn = Connection::open(":memory:".to_owned()).expect("in-memory connection");
        bootstrap(&conn).expect("bootstrap should succeed");
        seed_project_and_instance(&conn);

        let params = [
            SqliteValue::Text("project-a".to_owned()),
            SqliteValue::Text("instance-a".to_owned()),
            SqliteValue::Text("2h".to_owned()),
            SqliteValue::Integer(0),
            SqliteValue::Integer(10),
            SqliteValue::Integer(100),
            SqliteValue::Integer(200),
            SqliteValue::Integer(300),
            SqliteValue::Float(4.2),
        ];
        let result = conn.execute_with_params(
            "INSERT INTO search_summaries(\
                project_key, instance_id, window, window_start_ms, search_count, \
                p50_latency_us, p95_latency_us, p99_latency_us, avg_result_count\
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9);",
            &params,
        );
        assert!(
            result.is_err(),
            "invalid window label should fail CHECK constraint"
        );
    }

    #[test]
    fn search_summaries_accepts_all_supported_windows() {
        let conn = Connection::open(":memory:".to_owned()).expect("in-memory connection");
        bootstrap(&conn).expect("bootstrap should succeed");
        seed_project_and_instance(&conn);

        for (index, window) in ["1m", "15m", "1h", "6h", "24h", "3d", "1w"]
            .iter()
            .enumerate()
        {
            let params = [
                SqliteValue::Text("project-a".to_owned()),
                SqliteValue::Text("instance-a".to_owned()),
                SqliteValue::Text((*window).to_owned()),
                SqliteValue::Integer(i64::try_from(index).expect("index fits in i64")),
                SqliteValue::Integer(10),
                SqliteValue::Integer(100),
                SqliteValue::Integer(200),
                SqliteValue::Integer(300),
                SqliteValue::Float(5.0),
            ];
            conn.execute_with_params(
                "INSERT INTO search_summaries(\
                    project_key, instance_id, window, window_start_ms, search_count, \
                    p50_latency_us, p95_latency_us, p99_latency_us, avg_result_count\
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9);",
                &params,
            )
            .expect("supported window should insert");
        }
    }

    #[test]
    #[ignore = "FrankenSQLite does not yet enforce UNIQUE constraints on non-PK columns"]
    fn evidence_links_unique_constraint_prevents_duplicate_alert_uri_pairs() {
        let conn = Connection::open(":memory:".to_owned()).expect("in-memory connection");
        bootstrap(&conn).expect("bootstrap should succeed");
        seed_project_and_instance(&conn);
        conn.execute(
            "INSERT INTO alerts_timeline(\
                alert_id, project_key, instance_id, category, severity, reason_code, summary, \
                state, opened_at_ms, updated_at_ms\
             ) VALUES (\
                'alert-1', 'project-a', 'instance-a', 'latency', 'warn', 'latency.spike', \
                'spike', 'open', 1, 1\
             );",
        )
        .expect("alert row should insert");

        let first = [
            SqliteValue::Text("link-1".to_owned()),
            SqliteValue::Text("project-a".to_owned()),
            SqliteValue::Text("alert-1".to_owned()),
            SqliteValue::Text("jsonl".to_owned()),
            SqliteValue::Text("file:///tmp/evidence.jsonl".to_owned()),
            SqliteValue::Text("hash-1".to_owned()),
            SqliteValue::Integer(1),
        ];
        conn.execute_with_params(
            "INSERT INTO evidence_links(\
                link_id, project_key, alert_id, evidence_type, evidence_uri, \
                evidence_hash, created_at_ms\
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7);",
            &first,
        )
        .expect("first evidence link should insert");

        let duplicate_pair = [
            SqliteValue::Text("link-2".to_owned()),
            SqliteValue::Text("project-a".to_owned()),
            SqliteValue::Text("alert-1".to_owned()),
            SqliteValue::Text("jsonl".to_owned()),
            SqliteValue::Text("file:///tmp/evidence.jsonl".to_owned()),
            SqliteValue::Text("hash-2".to_owned()),
            SqliteValue::Integer(2),
        ];
        let duplicate_result = conn.execute_with_params(
            "INSERT INTO evidence_links(\
                link_id, project_key, alert_id, evidence_type, evidence_uri, \
                evidence_hash, created_at_ms\
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7);",
            &duplicate_pair,
        );
        assert!(
            duplicate_result.is_err(),
            "duplicate alert/evidence_uri pair should violate UNIQUE constraint"
        );
    }
}
