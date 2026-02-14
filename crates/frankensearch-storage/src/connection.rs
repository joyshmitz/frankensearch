use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::path::PathBuf;

use frankensearch_core::{SearchError, SearchResult};
use fsqlite::Connection;
use serde::{Deserialize, Serialize};

use crate::metrics::{StorageMetrics, StorageMetricsSnapshot};
use crate::schema;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StorageConfig {
    pub db_path: PathBuf,
    pub wal_mode: bool,
    pub busy_timeout_ms: u64,
    pub raptorq_repair_symbols: u32,
    pub cache_size_pages: i32,
    pub concurrent_transactions: bool,
}

impl StorageConfig {
    #[must_use]
    pub fn in_memory() -> Self {
        Self {
            db_path: PathBuf::from(":memory:"),
            ..Self::default()
        }
    }

    fn begin_sql(&self) -> &'static str {
        if self.concurrent_transactions {
            "BEGIN CONCURRENT;"
        } else {
            "BEGIN;"
        }
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            db_path: PathBuf::from("storage.sqlite3"),
            wal_mode: true,
            busy_timeout_ms: 5_000,
            raptorq_repair_symbols: 2,
            cache_size_pages: 2_000,
            concurrent_transactions: true,
        }
    }
}

pub struct Storage {
    conn: Connection,
    config: StorageConfig,
    metrics: StorageMetrics,
}

impl std::fmt::Debug for Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Storage")
            .field("path", &self.config.db_path)
            .field("wal_mode", &self.config.wal_mode)
            .field("busy_timeout_ms", &self.config.busy_timeout_ms)
            .finish_non_exhaustive()
    }
}

impl Storage {
    pub fn open(config: StorageConfig) -> SearchResult<Self> {
        tracing::debug!(
            target: "frankensearch.storage",
            path = %config.db_path.display(),
            wal_mode = config.wal_mode,
            busy_timeout_ms = config.busy_timeout_ms,
            cache_size_pages = config.cache_size_pages,
            concurrent_transactions = config.concurrent_transactions,
            "opening storage connection"
        );

        let path = config.db_path.to_string_lossy().to_string();
        let conn = Connection::open(path).map_err(map_storage_error)?;

        let storage = Self {
            conn,
            config,
            metrics: StorageMetrics::default(),
        };

        storage.metrics.record_open();
        storage.apply_pragmas()?;
        schema::bootstrap(storage.connection())?;
        storage.metrics.record_schema_bootstrap();

        if let Ok(version) = schema::current_version(storage.connection()) {
            tracing::debug!(
                target: "frankensearch.storage",
                schema_version = version,
                "storage bootstrap complete"
            );
        }

        Ok(storage)
    }

    pub fn open_in_memory() -> SearchResult<Self> {
        Self::open(StorageConfig::in_memory())
    }

    #[must_use]
    pub fn connection(&self) -> &Connection {
        &self.conn
    }

    #[must_use]
    pub fn config(&self) -> &StorageConfig {
        &self.config
    }

    #[must_use]
    pub fn metrics_snapshot(&self) -> StorageMetricsSnapshot {
        self.metrics.snapshot()
    }

    pub fn transaction<F, T>(&self, f: F) -> SearchResult<T>
    where
        F: FnOnce(&Connection) -> SearchResult<T>,
    {
        tracing::trace!(
            target: "frankensearch.storage",
            begin_sql = self.config.begin_sql(),
            "starting storage transaction"
        );

        self.conn
            .execute(self.config.begin_sql())
            .map_err(map_storage_error)?;

        let outcome = catch_unwind(AssertUnwindSafe(|| f(&self.conn)));

        match outcome {
            Ok(Ok(value)) => {
                self.conn.execute("COMMIT;").map_err(|commit_err| {
                    let _ = self.conn.execute("ROLLBACK;");
                    map_storage_error(commit_err)
                })?;
                self.metrics.record_commit();
                tracing::trace!(target: "frankensearch.storage", "storage transaction committed");
                Ok(value)
            }
            Ok(Err(err)) => {
                let _ = self.conn.execute("ROLLBACK;");
                self.metrics.record_rollback();
                tracing::debug!(
                    target: "frankensearch.storage",
                    ?err,
                    "storage transaction rolled back due to closure error"
                );
                Err(err)
            }
            Err(payload) => {
                let _ = self.conn.execute("ROLLBACK;");
                self.metrics.record_rollback();
                tracing::error!(
                    target: "frankensearch.storage",
                    "storage transaction rolled back after panic"
                );
                resume_unwind(payload);
            }
        }
    }

    fn apply_pragmas(&self) -> SearchResult<()> {
        tracing::trace!(
            target: "frankensearch.storage",
            wal_mode = self.config.wal_mode,
            busy_timeout_ms = self.config.busy_timeout_ms,
            cache_size_pages = self.config.cache_size_pages,
            raptorq_repair_symbols = self.config.raptorq_repair_symbols,
            "applying storage pragmas"
        );

        self.conn
            .execute("PRAGMA foreign_keys=ON;")
            .map_err(map_storage_error)?;

        if self.config.wal_mode {
            self.conn
                .execute("PRAGMA journal_mode=WAL;")
                .map_err(map_storage_error)?;
        } else if let Err(error) = self.conn.execute("PRAGMA journal_mode=DELETE;") {
            tracing::warn!(
                target: "frankensearch.storage",
                ?error,
                "journal_mode=DELETE was not accepted by backend; falling back to WAL"
            );
            self.conn
                .execute("PRAGMA journal_mode=WAL;")
                .map_err(map_storage_error)?;
        }

        self.conn
            .execute(&format!(
                "PRAGMA busy_timeout={};",
                self.config.busy_timeout_ms
            ))
            .map_err(map_storage_error)?;

        self.conn
            .execute(&format!(
                "PRAGMA cache_size={};",
                self.config.cache_size_pages
            ))
            .map_err(map_storage_error)?;

        if let Err(error) = self.conn.execute(&format!(
            "PRAGMA raptorq_repair_symbols={};",
            self.config.raptorq_repair_symbols
        )) {
            tracing::warn!(
                target: "frankensearch.storage",
                ?error,
                symbols = self.config.raptorq_repair_symbols,
                "raptorq pragma was not accepted by backend"
            );
        }

        Ok(())
    }
}

pub(crate) fn map_storage_error<E>(source: E) -> SearchError
where
    E: std::error::Error + Send + Sync + 'static,
{
    SearchError::SubsystemError {
        subsystem: "storage",
        source: Box::new(source),
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{self, AssertUnwindSafe};
    use std::path::PathBuf;
    use std::process;
    use std::time::{SystemTime, UNIX_EPOCH};

    use frankensearch_core::{SearchError, SearchResult};
    use fsqlite_types::value::SqliteValue;
    use serde_json::json;

    use crate::document::{DocumentRecord, count_documents, upsert_document};
    use crate::schema::{self, SCHEMA_VERSION};

    use super::{Storage, StorageConfig};

    struct TempDbPath {
        path: PathBuf,
    }

    impl TempDbPath {
        fn new(tag: &str) -> Self {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "frankensearch-storage-{tag}-{}-{nanos}.sqlite3",
                process::id()
            ));
            Self { path }
        }

        fn config(&self) -> StorageConfig {
            StorageConfig {
                db_path: self.path.clone(),
                ..StorageConfig::default()
            }
        }
    }

    impl Drop for TempDbPath {
        fn drop(&mut self) {
            for suffix in ["", "-wal", "-shm"] {
                let candidate = if suffix.is_empty() {
                    self.path.clone()
                } else {
                    PathBuf::from(format!("{}{}", self.path.display(), suffix))
                };
                let _ = std::fs::remove_file(candidate);
            }
        }
    }

    fn sample_document(id: &str) -> DocumentRecord {
        let mut doc = DocumentRecord::new(
            id,
            "Rust ownership keeps heap usage explicit during indexing.",
            [0x2a; 32],
            128,
            1_739_499_200,
            1_739_499_200,
        );
        doc.source_path = Some("tests://fixture".to_owned());
        doc.metadata = Some(json!({"fixture": true}));
        doc
    }

    fn insert_document_minimal(conn: &fsqlite::Connection, id: &str) -> SearchResult<()> {
        let params = [
            SqliteValue::Text(id.to_owned()),
            SqliteValue::Text("fixture-content".to_owned()),
            SqliteValue::Blob(vec![0x33; 32]),
            SqliteValue::Integer(64),
            SqliteValue::Integer(1_739_499_200),
            SqliteValue::Integer(1_739_499_200),
        ];
        conn.execute_with_params(
            "INSERT INTO documents (\
                doc_id, content_preview, content_hash, content_length, created_at, updated_at\
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6);",
            &params,
        )
        .map_err(super::map_storage_error)?;
        Ok(())
    }

    fn pragma_i64(conn: &fsqlite::Connection, name: &str) -> SearchResult<i64> {
        let row = conn
            .query_row(&format!("PRAGMA {name};"))
            .map_err(super::map_storage_error)?;
        match row.get(0) {
            Some(SqliteValue::Integer(value)) => Ok(*value),
            Some(SqliteValue::Text(value)) => {
                value
                    .parse::<i64>()
                    .map_err(|parse_error| SearchError::SubsystemError {
                        subsystem: "storage",
                        source: Box::new(parse_error),
                    })
            }
            Some(other) => Err(SearchError::SubsystemError {
                subsystem: "storage",
                source: Box::new(std::io::Error::other(format!(
                    "unexpected pragma type for {name}: {:?}",
                    other
                ))),
            }),
            None => Err(SearchError::SubsystemError {
                subsystem: "storage",
                source: Box::new(std::io::Error::other(format!(
                    "missing pragma value for {name}"
                ))),
            }),
        }
    }

    fn pragma_text(conn: &fsqlite::Connection, name: &str) -> SearchResult<String> {
        let row = conn
            .query_row(&format!("PRAGMA {name};"))
            .map_err(super::map_storage_error)?;
        match row.get(0) {
            Some(SqliteValue::Text(value)) => Ok(value.clone()),
            Some(SqliteValue::Integer(value)) => Ok(value.to_string()),
            Some(other) => Err(SearchError::SubsystemError {
                subsystem: "storage",
                source: Box::new(std::io::Error::other(format!(
                    "unexpected pragma type for {name}: {:?}",
                    other
                ))),
            }),
            None => Err(SearchError::SubsystemError {
                subsystem: "storage",
                source: Box::new(std::io::Error::other(format!(
                    "missing pragma value for {name}"
                ))),
            }),
        }
    }

    #[test]
    fn open_in_memory_bootstraps_schema() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let version = schema::current_version(storage.connection()).expect("schema version row");
        assert_eq!(version, SCHEMA_VERSION);

        let metrics = storage.metrics_snapshot();
        assert_eq!(metrics.opens, 1);
        assert_eq!(metrics.schema_bootstraps, 1);
    }

    #[test]
    fn open_applies_configured_pragmas_deterministically() {
        let tmp = TempDbPath::new("pragmas");
        let storage = Storage::open(StorageConfig {
            db_path: tmp.path.clone(),
            wal_mode: true,
            busy_timeout_ms: 1_234,
            cache_size_pages: 321,
            concurrent_transactions: false,
            ..StorageConfig::default()
        })
        .expect("storage should open with configured pragmas");

        assert_eq!(
            pragma_text(storage.connection(), "journal_mode")
                .expect("journal_mode pragma should be queryable")
                .to_ascii_lowercase(),
            "wal"
        );
        assert_eq!(
            pragma_i64(storage.connection(), "busy_timeout")
                .expect("busy_timeout pragma should be queryable"),
            1_234
        );
        assert_eq!(
            pragma_i64(storage.connection(), "cache_size")
                .expect("cache_size pragma should be queryable"),
            321
        );
    }

    #[test]
    fn schema_bootstrap_is_idempotent() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        schema::bootstrap(storage.connection()).expect("second bootstrap should succeed");
        schema::bootstrap(storage.connection()).expect("third bootstrap should succeed");

        let version = schema::current_version(storage.connection()).expect("schema version row");
        assert_eq!(version, SCHEMA_VERSION);
    }

    #[test]
    fn transaction_rolls_back_on_error() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let doc = sample_document("doc-error");

        let result: SearchResult<()> = storage.transaction(|conn| {
            upsert_document(conn, &doc)?;
            Err(SearchError::InvalidConfig {
                field: "test".to_owned(),
                value: "forced".to_owned(),
                reason: "force rollback".to_owned(),
            })
        });

        assert!(result.is_err(), "transaction should return original error");
        assert_eq!(
            count_documents(storage.connection()).expect("count should work"),
            0,
            "document insert should have been rolled back"
        );

        let metrics = storage.metrics_snapshot();
        assert_eq!(metrics.tx_commits, 0);
        assert_eq!(metrics.tx_rollbacks, 1);
    }

    #[test]
    fn transaction_rolls_back_on_panic_and_connection_stays_usable() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let doc = sample_document("doc-panic");

        let panic_result = panic::catch_unwind(AssertUnwindSafe(|| {
            let _: SearchResult<()> = storage.transaction(|conn| {
                upsert_document(conn, &doc).expect("insert should succeed before panic");
                panic!("forced panic");
            });
        }));

        assert!(panic_result.is_err(), "panic should propagate to caller");
        assert_eq!(
            count_documents(storage.connection()).expect("count should work"),
            0,
            "panic path should rollback transaction"
        );
        assert_eq!(
            schema::current_version(storage.connection()).expect("connection should remain usable"),
            SCHEMA_VERSION
        );
    }

    #[test]
    fn commit_persists_after_reopen() {
        let tmp = TempDbPath::new("visibility");
        let config = tmp.config();

        let storage_a = Storage::open(config.clone()).expect("writer storage should open");
        let doc = sample_document("doc-visible");

        storage_a
            .transaction(|conn| {
                insert_document_minimal(conn, &doc.doc_id)?;
                assert_eq!(
                    count_documents(conn)?,
                    1,
                    "write should be visible inside writer transaction"
                );
                Ok(())
            })
            .expect("transaction should commit");

        drop(storage_a);

        let storage_c = Storage::open(config).expect("post-commit storage should open");
        assert_eq!(
            count_documents(storage_c.connection()).expect("count after commit"),
            1,
            "committed write should be visible to newly opened connection"
        );
    }

    #[test]
    fn rollback_is_not_persisted_after_reopen() {
        let tmp = TempDbPath::new("rollback");
        let config = tmp.config();

        let storage_a = Storage::open(config).expect("writer storage should open");
        let doc_id = "doc-rollback";

        let tx_result: SearchResult<()> = storage_a.transaction(|conn| {
            insert_document_minimal(conn, doc_id)?;
            Err(SearchError::InvalidConfig {
                field: "test".to_owned(),
                value: "rollback".to_owned(),
                reason: "forced rollback".to_owned(),
            })
        });
        assert!(tx_result.is_err(), "transaction should rollback");
        assert_eq!(
            count_documents(storage_a.connection()).expect("count after rollback"),
            0,
            "rolled back write should not persist in active connection state"
        );
    }

    #[test]
    fn dropped_connection_allows_reopen_and_write() {
        let tmp = TempDbPath::new("drop");
        let config = tmp.config();

        let storage_a = Storage::open(config.clone()).expect("first storage should open");
        drop(storage_a);

        let storage_b = Storage::open(config).expect("second storage should open");

        storage_b
            .transaction(|conn| {
                insert_document_minimal(conn, "doc-after-drop")?;
                Ok(())
            })
            .expect("second connection should remain fully usable after first drop");

        assert_eq!(
            count_documents(storage_b.connection()).expect("count after write"),
            1
        );
    }
}
