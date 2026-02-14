# FrankenSQLite Fleet Telemetry Schema Design (bd-2yu.4.1)

## Scope
Design a normalized, query-optimized telemetry and timeline schema for fleet-level ops data in a dedicated FrankenSQLite database.

- Database file: `{data_dir}/frankensearch-ops.db`
- Storage mode: append-heavy events + periodic summary compaction
- Primary access pattern: per-project, time-window dashboards and incident timelines

This is intentionally separated from `{data_dir}/frankensearch.db` (document/search metadata) to avoid WAL contention between search writes and ops ingestion.

## Design Goals
1. Capture all required telemetry/timeline entities.
2. Enforce deterministic deduplication for retry-safe ingestion.
3. Optimize for project-scoped time-window queries.
4. Keep migration history explicit and reversible for development/testing.

## Core Entities
### Projects and Instances
- `projects`: tenant/root namespace for dashboards and policy.
- `instances`: runtime processes emitting telemetry and control-plane events.

### Search and Embedding Telemetry
- `search_events`: raw search-phase events.
- `search_summaries`: windowed aggregates (`1m`, `15m`, `1h`, `6h`, `24h`).
- `embedding_job_snapshots`: queue depth/progress and throughput snapshots.

### Index and Resource Telemetry
- `index_inventory_snapshots`: index health/size/hash snapshots.
- `resource_samples`: CPU/memory/IO samples.

### Timeline and Evidence
- `alerts_timeline`: normalized incident/degradation timeline.
- `evidence_links`: links alerts to evidence artifacts (JSONL, files, diagnostics).

## DDL (v1)
```sql
CREATE TABLE projects (
    project_key TEXT PRIMARY KEY,
    display_name TEXT,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL
);

CREATE TABLE instances (
    instance_id TEXT PRIMARY KEY,
    project_key TEXT NOT NULL REFERENCES projects(project_key) ON DELETE CASCADE,
    host_name TEXT,
    pid INTEGER,
    version TEXT,
    first_seen_ms INTEGER NOT NULL,
    last_heartbeat_ms INTEGER NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('started','healthy','degraded','stale','stopped'))
);

CREATE TABLE search_events (
    event_id TEXT PRIMARY KEY,
    project_key TEXT NOT NULL REFERENCES projects(project_key) ON DELETE CASCADE,
    instance_id TEXT NOT NULL REFERENCES instances(instance_id) ON DELETE CASCADE,
    correlation_id TEXT NOT NULL,
    query_hash TEXT,
    query_class TEXT,
    phase TEXT NOT NULL CHECK (phase IN ('initial','refined','failed')),
    latency_us INTEGER NOT NULL,
    result_count INTEGER,
    memory_bytes INTEGER,
    ts_ms INTEGER NOT NULL
);

CREATE TABLE search_summaries (
    project_key TEXT NOT NULL REFERENCES projects(project_key) ON DELETE CASCADE,
    instance_id TEXT NOT NULL REFERENCES instances(instance_id) ON DELETE CASCADE,
    window TEXT NOT NULL CHECK (window IN ('1m','15m','1h','6h','24h')),
    window_start_ms INTEGER NOT NULL,
    search_count INTEGER NOT NULL,
    p50_latency_us INTEGER,
    p95_latency_us INTEGER,
    p99_latency_us INTEGER,
    avg_result_count REAL,
    PRIMARY KEY (project_key, instance_id, window, window_start_ms)
);

CREATE TABLE embedding_job_snapshots (
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

CREATE TABLE index_inventory_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    project_key TEXT NOT NULL REFERENCES projects(project_key) ON DELETE CASCADE,
    instance_id TEXT NOT NULL REFERENCES instances(instance_id) ON DELETE CASCADE,
    index_name TEXT NOT NULL,
    index_type TEXT NOT NULL,
    record_count INTEGER NOT NULL,
    file_size_bytes INTEGER,
    file_hash TEXT,
    is_stale INTEGER NOT NULL CHECK (is_stale IN (0,1)),
    ts_ms INTEGER NOT NULL,
    UNIQUE (project_key, instance_id, index_name, ts_ms)
);

CREATE TABLE resource_samples (
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

CREATE TABLE alerts_timeline (
    alert_id TEXT PRIMARY KEY,
    project_key TEXT NOT NULL REFERENCES projects(project_key) ON DELETE CASCADE,
    instance_id TEXT,
    category TEXT NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('info','warn','error','critical')),
    reason_code TEXT NOT NULL,
    summary TEXT,
    state TEXT NOT NULL CHECK (state IN ('open','acknowledged','resolved')),
    opened_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    resolved_at_ms INTEGER
);

CREATE TABLE evidence_links (
    link_id TEXT PRIMARY KEY,
    project_key TEXT NOT NULL REFERENCES projects(project_key) ON DELETE CASCADE,
    alert_id TEXT NOT NULL REFERENCES alerts_timeline(alert_id) ON DELETE CASCADE,
    evidence_type TEXT NOT NULL,
    evidence_uri TEXT NOT NULL,
    evidence_hash TEXT,
    created_at_ms INTEGER NOT NULL,
    UNIQUE (alert_id, evidence_uri)
);
```

## Query Index Plan
```sql
CREATE INDEX idx_instances_project_heartbeat
    ON instances(project_key, last_heartbeat_ms DESC);

CREATE INDEX idx_search_events_project_time
    ON search_events(project_key, ts_ms DESC);

CREATE INDEX idx_search_events_instance_time
    ON search_events(instance_id, ts_ms DESC);

CREATE INDEX idx_search_events_corr
    ON search_events(project_key, correlation_id);

CREATE INDEX idx_embedding_snapshots_project_time
    ON embedding_job_snapshots(project_key, ts_ms DESC);

CREATE INDEX idx_index_inventory_project_time
    ON index_inventory_snapshots(project_key, ts_ms DESC);

CREATE INDEX idx_resource_samples_project_time
    ON resource_samples(project_key, ts_ms DESC);

CREATE INDEX idx_alerts_project_time
    ON alerts_timeline(project_key, opened_at_ms DESC);

CREATE INDEX idx_alerts_open
    ON alerts_timeline(project_key, state, severity, updated_at_ms DESC)
    WHERE state != 'resolved';

CREATE INDEX idx_evidence_alert
    ON evidence_links(alert_id, created_at_ms DESC);
```

## Deduplication and Integrity
- Raw events are idempotent via primary keys (`event_id`, `snapshot_id`, `alert_id`, `link_id`).
- Snapshot writers use composite `UNIQUE` constraints on `(project_key, instance_id, ..., ts_ms)`.
- Retry-safe ingestion uses `INSERT OR IGNORE` for immutable events and `INSERT ... ON CONFLICT DO UPDATE` for summary rows.
- All timeline and telemetry rows are project-scoped with explicit FKs to prevent orphaned records.

## Migration Strategy
### Schema Tracking
```sql
CREATE TABLE IF NOT EXISTS ops_schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at_ms INTEGER NOT NULL,
    checksum TEXT NOT NULL,
    reversible INTEGER NOT NULL CHECK (reversible IN (0,1))
);
```

### Rules
- Each migration version is monotonic and idempotent.
- Development/testing: every migration must include explicit `down` SQL in migration artifacts.
- Production: only forward migrations are auto-applied; rollback is manual and requires operator confirmation.
- Migration checksums are validated before apply.

## Dashboard Query Patterns
### Per-project search latency window
```sql
SELECT window_start_ms, p50_latency_us, p95_latency_us, p99_latency_us, search_count
FROM search_summaries
WHERE project_key = ?1 AND instance_id = ?2 AND window = ?3
  AND window_start_ms BETWEEN ?4 AND ?5
ORDER BY window_start_ms ASC;
```

### Live open alerts
```sql
SELECT alert_id, severity, category, reason_code, summary, opened_at_ms, updated_at_ms
FROM alerts_timeline
WHERE project_key = ?1 AND state != 'resolved'
ORDER BY updated_at_ms DESC
LIMIT ?2;
```

### Resource timeline
```sql
SELECT ts_ms, cpu_pct, rss_bytes, io_read_bytes, io_write_bytes, queue_depth
FROM resource_samples
WHERE project_key = ?1 AND instance_id = ?2
  AND ts_ms BETWEEN ?3 AND ?4
ORDER BY ts_ms ASC;
```

## Retention Policy
- Raw events/timeline snapshots: default 7 days.
- Summaries and evidence links: default 90 days.
- Retention cleanup runs hourly in bounded batches by table.
- Retention parameters come from OpsConfig (not TwoTierConfig).

## Integration Boundaries
- This schema is for ops/control-plane data only.
- Search/document metadata remains in the primary storage database.
- Future ingestion writer bead (`bd-2yu.4.2`) should write through this contract using idempotent upserts.
