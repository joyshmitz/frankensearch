use std::io;

use frankensearch_core::{SearchError, SearchResult};
use fsqlite::{Connection, Row};
use fsqlite_types::value::SqliteValue;

pub const SCHEMA_VERSION: i64 = 5;

struct Migration {
    version: i64,
    statements: &'static [&'static str],
}

const MIGRATIONS: &[Migration] = &[
    Migration {
        version: 1,
        statements: &[
            "CREATE TABLE IF NOT EXISTS documents (\
                id TEXT PRIMARY KEY,\
                title TEXT,\
                content TEXT NOT NULL,\
                created_at INTEGER NOT NULL,\
                doc_type TEXT,\
                source TEXT,\
                metadata_json TEXT,\
                content_hash TEXT NOT NULL,\
                updated_at INTEGER NOT NULL\
            );",
            "CREATE INDEX IF NOT EXISTS idx_documents_updated_at ON documents(updated_at DESC);",
            "CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);",
            "CREATE TABLE IF NOT EXISTS embedding_jobs (\
                doc_id TEXT PRIMARY KEY,\
                status TEXT NOT NULL,\
                attempts INTEGER NOT NULL DEFAULT 0,\
                queued_at INTEGER NOT NULL,\
                started_at INTEGER,\
                finished_at INTEGER,\
                last_error TEXT,\
                FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE\
            );",
            "CREATE INDEX IF NOT EXISTS idx_embedding_jobs_status_queued ON embedding_jobs(status, queued_at);",
            "CREATE TABLE IF NOT EXISTS content_hashes (\
                content_hash TEXT PRIMARY KEY,\
                first_doc_id TEXT NOT NULL,\
                seen_count INTEGER NOT NULL DEFAULT 1,\
                first_seen_at INTEGER NOT NULL,\
                last_seen_at INTEGER NOT NULL\
            );",
        ],
    },
    Migration {
        version: 2,
        statements: &[
            "CREATE INDEX IF NOT EXISTS idx_embedding_jobs_status_started ON embedding_jobs(status, started_at);",
        ],
    },
    Migration {
        version: 3,
        statements: &[
            "DROP INDEX IF EXISTS idx_embedding_jobs_status_queued;",
            "DROP INDEX IF EXISTS idx_embedding_jobs_status_started;",
            "DROP TABLE IF EXISTS embedding_jobs;",
            "DROP INDEX IF EXISTS idx_documents_updated_at;",
            "DROP INDEX IF EXISTS idx_documents_content_hash;",
            "DROP TABLE IF EXISTS documents;",
            "CREATE TABLE IF NOT EXISTS documents (\
                doc_id TEXT PRIMARY KEY,\
                source_path TEXT,\
                content_preview TEXT NOT NULL,\
                content_hash BLOB NOT NULL,\
                content_length INTEGER NOT NULL,\
                created_at INTEGER NOT NULL,\
                updated_at INTEGER NOT NULL,\
                metadata_json TEXT\
            );",
            "CREATE TABLE IF NOT EXISTS embedding_jobs (\
                doc_id TEXT PRIMARY KEY,\
                status TEXT NOT NULL,\
                attempts INTEGER NOT NULL DEFAULT 0,\
                queued_at INTEGER NOT NULL,\
                started_at INTEGER,\
                finished_at INTEGER,\
                last_error TEXT,\
                FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE\
            );",
            "CREATE INDEX IF NOT EXISTS idx_embedding_jobs_status_queued ON embedding_jobs(status, queued_at);",
            "CREATE INDEX IF NOT EXISTS idx_embedding_jobs_status_started ON embedding_jobs(status, started_at);",
            "CREATE TABLE IF NOT EXISTS embedding_status (\
                doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,\
                embedder_id TEXT NOT NULL,\
                embedder_revision TEXT,\
                status TEXT NOT NULL DEFAULT 'pending',\
                embedded_at INTEGER,\
                error_message TEXT,\
                retry_count INTEGER NOT NULL DEFAULT 0,\
                PRIMARY KEY(doc_id, embedder_id)\
            );",
            "CREATE INDEX IF NOT EXISTS idx_embedding_status_pending ON embedding_status(status, doc_id) WHERE status = 'pending';",
            "CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);",
            "CREATE INDEX IF NOT EXISTS idx_documents_updated_at ON documents(updated_at DESC);",
            "CREATE TABLE IF NOT EXISTS content_hashes (\
                content_hash TEXT PRIMARY KEY,\
                first_doc_id TEXT NOT NULL,\
                seen_count INTEGER NOT NULL DEFAULT 1,\
                first_seen_at INTEGER NOT NULL,\
                last_seen_at INTEGER NOT NULL\
            );",
        ],
    },
    Migration {
        version: 4,
        statements: &[
            "DROP INDEX IF EXISTS idx_embedding_jobs_status_queued;",
            "DROP INDEX IF EXISTS idx_embedding_jobs_status_started;",
            "DROP TABLE IF EXISTS embedding_jobs;",
            "CREATE TABLE IF NOT EXISTS embedding_jobs (\
                job_id INTEGER PRIMARY KEY AUTOINCREMENT,\
                doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,\
                embedder_id TEXT NOT NULL,\
                priority INTEGER NOT NULL DEFAULT 0,\
                submitted_at INTEGER NOT NULL,\
                started_at INTEGER,\
                completed_at INTEGER,\
                status TEXT NOT NULL DEFAULT 'pending',\
                retry_count INTEGER NOT NULL DEFAULT 0,\
                max_retries INTEGER NOT NULL DEFAULT 3,\
                error_message TEXT,\
                content_hash BLOB,\
                worker_id TEXT,\
                UNIQUE(doc_id, embedder_id, status)\
            );",
            "CREATE INDEX IF NOT EXISTS idx_jobs_pending ON embedding_jobs(status, priority DESC, submitted_at ASC) WHERE status = 'pending';",
            "CREATE INDEX IF NOT EXISTS idx_jobs_processing ON embedding_jobs(status, started_at) WHERE status = 'processing';",
        ],
    },
    Migration {
        version: 5,
        statements: &[
            "CREATE TABLE IF NOT EXISTS index_metadata (\
                index_name TEXT PRIMARY KEY,\
                index_type TEXT NOT NULL,\
                embedder_id TEXT NOT NULL,\
                embedder_revision TEXT,\
                dimension INTEGER NOT NULL,\
                record_count INTEGER NOT NULL DEFAULT 0,\
                file_path TEXT,\
                file_size_bytes INTEGER,\
                file_hash TEXT,\
                schema_version INTEGER,\
                built_at INTEGER,\
                build_duration_ms INTEGER,\
                source_doc_count INTEGER NOT NULL DEFAULT 0,\
                config_json TEXT,\
                fec_path TEXT,\
                fec_size_bytes INTEGER,\
                last_verified_at INTEGER,\
                last_repair_at INTEGER,\
                repair_count INTEGER NOT NULL DEFAULT 0,\
                mean_norm REAL,\
                variance REAL\
            );",
            "CREATE TABLE IF NOT EXISTS index_build_history (\
                build_id INTEGER PRIMARY KEY AUTOINCREMENT,\
                index_name TEXT NOT NULL REFERENCES index_metadata(index_name) ON DELETE CASCADE,\
                built_at INTEGER NOT NULL,\
                build_duration_ms INTEGER NOT NULL,\
                record_count INTEGER NOT NULL,\
                source_doc_count INTEGER NOT NULL,\
                trigger TEXT NOT NULL,\
                config_json TEXT,\
                notes TEXT,\
                mean_norm REAL,\
                variance REAL\
            );",
            "CREATE INDEX IF NOT EXISTS idx_build_history_index ON index_build_history(index_name, built_at DESC);",
        ],
    },
];

pub fn bootstrap(conn: &Connection) -> SearchResult<()> {
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);")
        .map_err(storage_error)?;

    let mut version = current_version_optional(conn)?.unwrap_or(0);
    if version > SCHEMA_VERSION {
        return Err(SearchError::SubsystemError {
            subsystem: "storage",
            source: Box::new(io::Error::other(format!(
                "schema version {version} is newer than supported {SCHEMA_VERSION}"
            ))),
        });
    }

    for migration in MIGRATIONS {
        if migration.version <= version {
            continue;
        }

        tracing::debug!(
            target: "frankensearch.storage",
            from_version = version,
            to_version = migration.version,
            "applying storage schema migration"
        );

        for statement in migration.statements {
            conn.execute(statement).map_err(storage_error)?;
        }

        let params = [SqliteValue::Integer(migration.version)];
        conn.execute_with_params(
            "INSERT OR IGNORE INTO schema_version(version) VALUES (?1);",
            &params,
        )
        .map_err(storage_error)?;
        version = migration.version;
    }

    tracing::debug!(
        target: "frankensearch.storage",
        schema_version = version,
        "storage schema bootstrap complete"
    );

    Ok(())
}

pub fn current_version(conn: &Connection) -> SearchResult<i64> {
    current_version_optional(conn)?.ok_or_else(|| SearchError::SubsystemError {
        subsystem: "storage",
        source: Box::new(io::Error::other("schema_version table has no rows")),
    })
}

fn current_version_optional(conn: &Connection) -> SearchResult<Option<i64>> {
    let rows = conn
        .query("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1;")
        .map_err(storage_error)?;
    let Some(row) = rows.first() else {
        return Ok(None);
    };
    row_i64(row, 0, "schema_version.version").map(Some)
}

fn row_i64(row: &Row, index: usize, field: &str) -> SearchResult<i64> {
    match row.get(index) {
        Some(SqliteValue::Integer(value)) => Ok(*value),
        Some(other) => Err(SearchError::SubsystemError {
            subsystem: "storage",
            source: Box::new(io::Error::other(format!(
                "unexpected type for {field}: {:?}",
                other
            ))),
        }),
        None => Err(SearchError::SubsystemError {
            subsystem: "storage",
            source: Box::new(io::Error::other(format!("missing column for {field}"))),
        }),
    }
}

fn storage_error<E>(source: E) -> SearchError
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
    use super::{
        MIGRATIONS, SCHEMA_VERSION, bootstrap, current_version, current_version_optional,
        storage_error,
    };
    use fsqlite::Connection;
    use fsqlite_types::value::SqliteValue;

    fn index_exists(conn: &Connection, index_name: &str) -> bool {
        let params = [SqliteValue::Text(index_name.to_owned())];
        let rows = conn
            .query_with_params(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND name = ?1 LIMIT 1;",
                &params,
            )
            .map_err(storage_error)
            .expect("sqlite_master query should succeed");
        !rows.is_empty()
    }

    #[test]
    fn bootstrap_sets_latest_version_for_fresh_database() {
        let conn = Connection::open(":memory:".to_owned()).expect("in-memory connection");

        bootstrap(&conn).expect("bootstrap should succeed");
        assert_eq!(
            current_version(&conn).expect("schema version should exist"),
            SCHEMA_VERSION
        );
        assert!(
            index_exists(&conn, "idx_embedding_status_pending"),
            "latest schema should include pending-status index"
        );
        assert!(
            index_exists(&conn, "idx_jobs_pending"),
            "latest schema should include queue pending index"
        );
        assert!(
            index_exists(&conn, "idx_jobs_processing"),
            "latest schema should include queue processing index"
        );
    }

    #[test]
    fn bootstrap_upgrades_v1_database_to_latest() {
        let conn = Connection::open(":memory:".to_owned()).expect("in-memory connection");

        conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);")
            .expect("schema_version should be creatable");
        for statement in MIGRATIONS[0].statements {
            conn.execute(statement)
                .expect("v1 schema statement should execute");
        }
        conn.execute("INSERT INTO schema_version(version) VALUES (1);")
            .expect("v1 marker row should insert");

        assert_eq!(
            current_version_optional(&conn).expect("version should read"),
            Some(1)
        );
        assert!(
            !index_exists(&conn, "idx_embedding_status_pending"),
            "v3 index should not exist before bootstrap upgrade"
        );

        bootstrap(&conn).expect("bootstrap should upgrade schema");
        assert_eq!(
            current_version(&conn).expect("schema version should exist"),
            SCHEMA_VERSION
        );
        assert!(
            index_exists(&conn, "idx_embedding_status_pending"),
            "v3 migration should create pending-status index"
        );
        assert!(
            index_exists(&conn, "idx_jobs_pending"),
            "v4 migration should create queue pending index"
        );
        assert!(
            index_exists(&conn, "idx_jobs_processing"),
            "v4 migration should create queue processing index"
        );
    }

    #[test]
    fn bootstrap_is_idempotent_at_latest_version() {
        let conn = Connection::open(":memory:".to_owned()).expect("in-memory connection");

        bootstrap(&conn).expect("first bootstrap should succeed");
        bootstrap(&conn).expect("second bootstrap should succeed");
        bootstrap(&conn).expect("third bootstrap should succeed");

        assert_eq!(
            current_version(&conn).expect("schema version should exist"),
            SCHEMA_VERSION
        );
    }
}
