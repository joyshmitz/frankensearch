use std::fmt;
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use frankensearch_core::{SearchError, SearchResult};
use fsqlite::{Connection, Row};
use fsqlite_types::value::SqliteValue;
use serde::{Deserialize, Serialize};

use crate::connection::{Storage, map_storage_error};

const SUBSYSTEM: &str = "storage";
const HASH_EMBEDDER_PREFIX: &str = "fnv1a-";
const MAX_BACKOFF_EXPONENT: u32 = 20;
const MAX_RETRY_DELAY_MS: u64 = 30_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobStatus {
    Pending,
    Processing,
    Completed,
    Failed,
    Skipped,
}

impl JobStatus {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Processing => "processing",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Skipped => "skipped",
        }
    }

    fn from_str(value: &str) -> Option<Self> {
        match value {
            "pending" => Some(Self::Pending),
            "processing" => Some(Self::Processing),
            "completed" => Some(Self::Completed),
            "failed" => Some(Self::Failed),
            "skipped" => Some(Self::Skipped),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueueErrorKind {
    NotFound,
    Conflict,
    Validation,
}

impl QueueErrorKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::NotFound => "not_found",
            Self::Conflict => "conflict",
            Self::Validation => "validation",
        }
    }
}

#[derive(Debug)]
struct QueueError {
    kind: QueueErrorKind,
    message: String,
}

impl fmt::Display for QueueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind.as_str(), self.message)
    }
}

impl std::error::Error for QueueError {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnqueueRequest {
    pub doc_id: String,
    pub embedder_id: String,
    pub content_hash: [u8; 32],
    pub priority: i32,
}

impl EnqueueRequest {
    #[must_use]
    pub fn new(
        doc_id: impl Into<String>,
        embedder_id: impl Into<String>,
        content_hash: [u8; 32],
        priority: i32,
    ) -> Self {
        Self {
            doc_id: doc_id.into(),
            embedder_id: embedder_id.into(),
            content_hash,
            priority,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchEnqueueResult {
    pub inserted: u64,
    pub replaced: u64,
    pub deduplicated: u64,
    pub skipped_hash_embedder: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimedJob {
    pub job_id: i64,
    pub doc_id: String,
    pub embedder_id: String,
    pub priority: i32,
    pub retry_count: u32,
    pub max_retries: u32,
    pub submitted_at: i64,
    pub content_hash: Option<[u8; 32]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailResult {
    Retried {
        retry_count: u32,
        delay_ms: u64,
        next_attempt_at_ms: i64,
    },
    TerminalFailed {
        retry_count: u32,
    },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueueDepth {
    pub pending: usize,
    pub ready_pending: usize,
    pub processing: usize,
    pub completed: usize,
    pub failed: usize,
    pub skipped: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobQueueConfig {
    pub batch_size: usize,
    pub visibility_timeout_ms: u64,
    pub max_retries: u32,
    pub retry_base_delay_ms: u64,
    pub stale_job_threshold_ms: u64,
    pub backpressure_threshold: usize,
}

impl Default for JobQueueConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            visibility_timeout_ms: 30_000,
            max_retries: 3,
            retry_base_delay_ms: 100,
            stale_job_threshold_ms: 300_000,
            backpressure_threshold: 10_000,
        }
    }
}

#[derive(Debug, Default)]
pub struct JobQueueMetrics {
    pub total_enqueued: AtomicU64,
    pub total_completed: AtomicU64,
    pub total_failed: AtomicU64,
    pub total_skipped: AtomicU64,
    pub total_retried: AtomicU64,
    pub total_deduplicated: AtomicU64,
    pub total_batches_processed: AtomicU64,
    pub total_embed_time_us: AtomicU64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobQueueMetricsSnapshot {
    pub total_enqueued: u64,
    pub total_completed: u64,
    pub total_failed: u64,
    pub total_skipped: u64,
    pub total_retried: u64,
    pub total_deduplicated: u64,
    pub total_batches_processed: u64,
    pub total_embed_time_us: u64,
}

impl JobQueueMetrics {
    #[must_use]
    pub fn snapshot(&self) -> JobQueueMetricsSnapshot {
        JobQueueMetricsSnapshot {
            total_enqueued: self.total_enqueued.load(Ordering::Relaxed),
            total_completed: self.total_completed.load(Ordering::Relaxed),
            total_failed: self.total_failed.load(Ordering::Relaxed),
            total_skipped: self.total_skipped.load(Ordering::Relaxed),
            total_retried: self.total_retried.load(Ordering::Relaxed),
            total_deduplicated: self.total_deduplicated.load(Ordering::Relaxed),
            total_batches_processed: self.total_batches_processed.load(Ordering::Relaxed),
            total_embed_time_us: self.total_embed_time_us.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PersistentJobQueue {
    storage: Arc<Storage>,
    config: JobQueueConfig,
    metrics: Arc<JobQueueMetrics>,
}

impl PersistentJobQueue {
    #[must_use]
    pub fn new(storage: Arc<Storage>, config: JobQueueConfig) -> Self {
        Self {
            storage,
            config,
            metrics: Arc::new(JobQueueMetrics::default()),
        }
    }

    #[must_use]
    pub fn with_metrics(
        storage: Arc<Storage>,
        config: JobQueueConfig,
        metrics: Arc<JobQueueMetrics>,
    ) -> Self {
        Self {
            storage,
            config,
            metrics,
        }
    }

    #[must_use]
    pub const fn config(&self) -> &JobQueueConfig {
        &self.config
    }

    #[must_use]
    pub fn metrics(&self) -> &JobQueueMetrics {
        self.metrics.as_ref()
    }

    pub fn enqueue(
        &self,
        doc_id: &str,
        embedder_id: &str,
        content_hash: &[u8; 32],
        priority: i32,
    ) -> SearchResult<bool> {
        let request = EnqueueRequest::new(doc_id, embedder_id, *content_hash, priority);
        let submitted_at = unix_timestamp_ms()?;
        let outcome = self.storage.transaction(|conn| {
            enqueue_inner(conn, &request, submitted_at, self.config.max_retries)
        })?;
        self.record_enqueue_outcome(outcome);

        tracing::debug!(
            target: "frankensearch.storage",
            op = "queue.enqueue",
            doc_id,
            embedder_id,
            outcome = ?outcome,
            "embedding job enqueue completed"
        );

        Ok(matches!(
            outcome,
            EnqueueOutcome::Inserted | EnqueueOutcome::Replaced
        ))
    }

    pub fn enqueue_batch(&self, jobs: &[EnqueueRequest]) -> SearchResult<BatchEnqueueResult> {
        if jobs.is_empty() {
            return Ok(BatchEnqueueResult::default());
        }

        let submitted_base = unix_timestamp_ms()?;
        let max_retries = self.config.max_retries;
        let summary = self.storage.transaction(|conn| {
            let mut summary = BatchEnqueueResult::default();
            for (index, job) in jobs.iter().enumerate() {
                let submitted_at = submitted_base.saturating_add(usize_to_i64(index)?);
                let outcome = enqueue_inner(conn, job, submitted_at, max_retries)?;
                summary.record(outcome);
            }
            Ok(summary)
        })?;

        if summary.inserted > 0 || summary.replaced > 0 {
            self.metrics
                .total_enqueued
                .fetch_add(summary.inserted + summary.replaced, Ordering::Relaxed);
        }
        if summary.deduplicated > 0 || summary.skipped_hash_embedder > 0 {
            self.metrics.total_deduplicated.fetch_add(
                summary.deduplicated + summary.skipped_hash_embedder,
                Ordering::Relaxed,
            );
        }

        tracing::debug!(
            target: "frankensearch.storage",
            op = "queue.enqueue_batch",
            requested = jobs.len(),
            inserted = summary.inserted,
            replaced = summary.replaced,
            deduplicated = summary.deduplicated,
            skipped_hash_embedder = summary.skipped_hash_embedder,
            "embedding job batch enqueue completed"
        );

        Ok(summary)
    }

    pub fn claim_batch(&self, worker_id: &str, batch_size: usize) -> SearchResult<Vec<ClaimedJob>> {
        ensure_non_empty(worker_id, "worker_id")?;
        if batch_size == 0 {
            return Ok(Vec::new());
        }

        let start = Instant::now();
        let batch_limit = batch_size.min(self.config.batch_size);
        let now_ms = unix_timestamp_ms()?;
        let limit = usize_to_i64(batch_limit)?;
        let claimed = self.storage.transaction(|conn| {
            let claim_params = [SqliteValue::Integer(now_ms), SqliteValue::Integer(limit)];
            let candidates = conn
                .query_with_params(
                    "SELECT job_id, doc_id, embedder_id, priority, retry_count, max_retries, content_hash, submitted_at \
                     FROM embedding_jobs \
                     WHERE status = 'pending' \
                       AND submitted_at <= ?1 \
                       AND NOT EXISTS ( \
                           SELECT 1 FROM embedding_jobs active \
                           WHERE active.doc_id = embedding_jobs.doc_id \
                             AND active.embedder_id = embedding_jobs.embedder_id \
                             AND active.status = 'processing' \
                       ) \
                     ORDER BY priority DESC, submitted_at ASC \
                     LIMIT ?2;",
                    &claim_params,
                )
                .map_err(map_storage_error)?;

            let mut claimed = Vec::with_capacity(candidates.len());
            for row in &candidates {
                let job_id = row_i64(row, 0, "embedding_jobs.job_id")?;
                let update_params = [
                    SqliteValue::Text(JobStatus::Processing.as_str().to_owned()),
                    SqliteValue::Integer(now_ms),
                    SqliteValue::Text(worker_id.to_owned()),
                    SqliteValue::Integer(job_id),
                ];
                let updated = conn
                    .execute_with_params(
                        "UPDATE embedding_jobs \
                         SET status = ?1, started_at = ?2, worker_id = ?3, error_message = NULL \
                         WHERE job_id = ?4 AND status = 'pending';",
                        &update_params,
                    )
                    .map_err(map_storage_error)?;
                if updated != 1 {
                    continue;
                }

                claimed.push(ClaimedJob {
                    job_id,
                    doc_id: row_text(row, 1, "embedding_jobs.doc_id")?.to_owned(),
                    embedder_id: row_text(row, 2, "embedding_jobs.embedder_id")?.to_owned(),
                    priority: row_i32(row, 3, "embedding_jobs.priority")?,
                    retry_count: row_u32(row, 4, "embedding_jobs.retry_count")?,
                    max_retries: row_u32(row, 5, "embedding_jobs.max_retries")?,
                    content_hash: row_optional_blob_32(row, 6, "embedding_jobs.content_hash")?,
                    submitted_at: row_i64(row, 7, "embedding_jobs.submitted_at")?,
                });
            }
            Ok(claimed)
        })?;

        if !claimed.is_empty() {
            self.metrics
                .total_batches_processed
                .fetch_add(1, Ordering::Relaxed);
        }
        let elapsed_us = duration_as_u64(start.elapsed().as_micros());
        tracing::debug!(
            target: "frankensearch.storage",
            op = "queue.claim_batch",
            worker_id,
            requested = batch_size,
            effective_batch_size = batch_limit,
            claimed = claimed.len(),
            claim_latency_us = elapsed_us,
            "embedding job claim completed"
        );

        Ok(claimed)
    }

    pub fn complete(&self, job_id: i64) -> SearchResult<()> {
        let now_ms = unix_timestamp_ms()?;
        let started_at = self.storage.transaction(|conn| {
            let Some(state) = load_job_state(conn, job_id)? else {
                return Err(not_found_error("embedding_jobs", &job_id.to_string()));
            };
            if state.status != JobStatus::Processing {
                return Err(conflict_error(format!(
                    "job {job_id} is not processing (status={})",
                    state.status.as_str()
                )));
            }

            let params = [
                SqliteValue::Text(JobStatus::Completed.as_str().to_owned()),
                SqliteValue::Integer(now_ms),
                SqliteValue::Integer(job_id),
            ];
            let updated = conn
                .execute_with_params(
                    "UPDATE embedding_jobs \
                     SET status = ?1, completed_at = ?2, worker_id = NULL, error_message = NULL \
                     WHERE job_id = ?3 AND status = 'processing';",
                    &params,
                )
                .map_err(map_storage_error)?;
            if updated != 1 {
                return Err(conflict_error(format!(
                    "job {job_id} changed status during completion"
                )));
            }
            Ok(state.started_at)
        })?;

        self.metrics.total_completed.fetch_add(1, Ordering::Relaxed);
        if let Some(started_at_ms) = started_at {
            let elapsed_ms = now_ms.saturating_sub(started_at_ms);
            if let Ok(elapsed_ms_u64) = u64::try_from(elapsed_ms) {
                self.metrics
                    .total_embed_time_us
                    .fetch_add(elapsed_ms_u64.saturating_mul(1_000), Ordering::Relaxed);
            }
        }

        tracing::debug!(
            target: "frankensearch.storage",
            op = "queue.complete",
            job_id,
            "embedding job marked completed"
        );
        Ok(())
    }

    pub fn fail(&self, job_id: i64, error: &str) -> SearchResult<FailResult> {
        ensure_non_empty(error, "error")?;

        let now_ms = unix_timestamp_ms()?;
        let retry_base_delay_ms = self.config.retry_base_delay_ms;
        let result = self.storage.transaction(|conn| {
            let Some(state) = load_job_state(conn, job_id)? else {
                return Err(not_found_error("embedding_jobs", &job_id.to_string()));
            };
            if state.status != JobStatus::Processing {
                return Err(conflict_error(format!(
                    "job {job_id} is not processing (status={})",
                    state.status.as_str()
                )));
            }

            let retry_count = state.retry_count.saturating_add(1);
            if retry_count > state.max_retries {
                let params = [
                    SqliteValue::Text(JobStatus::Failed.as_str().to_owned()),
                    SqliteValue::Integer(i64::from(retry_count)),
                    SqliteValue::Integer(now_ms),
                    SqliteValue::Text(error.to_owned()),
                    SqliteValue::Integer(job_id),
                ];
                let updated = conn
                    .execute_with_params(
                        "UPDATE embedding_jobs \
                         SET status = ?1, retry_count = ?2, completed_at = ?3, error_message = ?4, worker_id = NULL \
                         WHERE job_id = ?5 AND status = 'processing';",
                        &params,
                    )
                    .map_err(map_storage_error)?;
                if updated != 1 {
                    return Err(conflict_error(format!(
                        "job {job_id} changed status during fail/terminal transition"
                    )));
                }
                return Ok(FailResult::TerminalFailed { retry_count });
            }

            let delay_ms =
                compute_retry_delay_ms(retry_base_delay_ms, retry_count.saturating_sub(1));
            let next_attempt_at_ms = now_ms.saturating_add(i64::try_from(delay_ms).unwrap_or(i64::MAX));
            let params = [
                SqliteValue::Text(JobStatus::Pending.as_str().to_owned()),
                SqliteValue::Integer(i64::from(retry_count)),
                SqliteValue::Integer(next_attempt_at_ms),
                SqliteValue::Text(error.to_owned()),
                SqliteValue::Integer(job_id),
            ];
            let updated = conn
                .execute_with_params(
                    "UPDATE embedding_jobs \
                     SET status = ?1, retry_count = ?2, submitted_at = ?3, started_at = NULL, completed_at = NULL, \
                         error_message = ?4, worker_id = NULL \
                     WHERE job_id = ?5 AND status = 'processing';",
                    &params,
                )
                .map_err(map_storage_error)?;
            if updated != 1 {
                return Err(conflict_error(format!(
                    "job {job_id} changed status during fail/retry transition"
                )));
            }
            Ok(FailResult::Retried {
                retry_count,
                delay_ms,
                next_attempt_at_ms,
            })
        })?;

        match result {
            FailResult::Retried { .. } => {
                self.metrics.total_retried.fetch_add(1, Ordering::Relaxed);
            }
            FailResult::TerminalFailed { .. } => {
                self.metrics.total_failed.fetch_add(1, Ordering::Relaxed);
            }
        }

        tracing::debug!(
            target: "frankensearch.storage",
            op = "queue.fail",
            job_id,
            ?result,
            "embedding job failure transition completed"
        );

        Ok(result)
    }

    pub fn skip(&self, job_id: i64, reason: &str) -> SearchResult<()> {
        ensure_non_empty(reason, "reason")?;
        let now_ms = unix_timestamp_ms()?;
        self.storage.transaction(|conn| {
            let Some(state) = load_job_state(conn, job_id)? else {
                return Err(not_found_error("embedding_jobs", &job_id.to_string()));
            };
            if !matches!(state.status, JobStatus::Pending | JobStatus::Processing) {
                return Err(conflict_error(format!(
                    "job {job_id} cannot be skipped from status {}",
                    state.status.as_str()
                )));
            }

            let params = [
                SqliteValue::Text(JobStatus::Skipped.as_str().to_owned()),
                SqliteValue::Integer(now_ms),
                SqliteValue::Text(reason.to_owned()),
                SqliteValue::Integer(job_id),
            ];
            let updated = conn
                .execute_with_params(
                    "UPDATE embedding_jobs \
                     SET status = ?1, completed_at = ?2, worker_id = NULL, error_message = ?3 \
                     WHERE job_id = ?4;",
                    &params,
                )
                .map_err(map_storage_error)?;
            if updated != 1 {
                return Err(conflict_error(format!(
                    "job {job_id} changed status during skip transition"
                )));
            }
            Ok(())
        })?;

        self.metrics.total_skipped.fetch_add(1, Ordering::Relaxed);
        tracing::debug!(
            target: "frankensearch.storage",
            op = "queue.skip",
            job_id,
            "embedding job marked skipped"
        );
        Ok(())
    }

    pub fn reclaim_stale_jobs(&self) -> SearchResult<usize> {
        let now_ms = unix_timestamp_ms()?;
        let reclaim_after_ms = self
            .config
            .visibility_timeout_ms
            .max(self.config.stale_job_threshold_ms);
        let cutoff = now_ms.saturating_sub(i64::try_from(reclaim_after_ms).unwrap_or(i64::MAX));
        let params = [
            SqliteValue::Text(JobStatus::Pending.as_str().to_owned()),
            SqliteValue::Integer(now_ms),
            SqliteValue::Text("reclaimed stale lease".to_owned()),
            SqliteValue::Integer(cutoff),
        ];
        let reclaimed = self
            .storage
            .connection()
            .execute_with_params(
                "UPDATE embedding_jobs \
                 SET status = ?1, submitted_at = ?2, started_at = NULL, worker_id = NULL, error_message = ?3 \
                 WHERE status = 'processing' AND (started_at IS NULL OR started_at <= ?4);",
                &params,
            )
            .map_err(map_storage_error)?;

        if reclaimed > 0 {
            self.metrics
                .total_retried
                .fetch_add(usize_to_u64(reclaimed), Ordering::Relaxed);
            tracing::warn!(
                target: "frankensearch.storage",
                op = "queue.reclaim_stale_jobs",
                reclaimed,
                cutoff_ms = cutoff,
                "reclaimed stale embedding jobs"
            );
        } else {
            tracing::trace!(
                target: "frankensearch.storage",
                op = "queue.reclaim_stale_jobs",
                reclaimed = 0,
                "no stale embedding jobs found"
            );
        }

        Ok(reclaimed)
    }

    pub fn is_backpressured(&self) -> SearchResult<bool> {
        let depth = self.queue_depth()?;
        Ok(depth.pending > self.config.backpressure_threshold)
    }

    pub fn queue_depth(&self) -> SearchResult<QueueDepth> {
        let mut depth = QueueDepth::default();
        let rows = self
            .storage
            .connection()
            .query("SELECT status, COUNT(*) FROM embedding_jobs GROUP BY status;")
            .map_err(map_storage_error)?;
        for row in &rows {
            let status = row_text(row, 0, "embedding_jobs.status")?;
            let count = i64_to_usize(row_i64(row, 1, "embedding_jobs.count")?)?;
            match JobStatus::from_str(status) {
                Some(JobStatus::Pending) => depth.pending = count,
                Some(JobStatus::Processing) => depth.processing = count,
                Some(JobStatus::Completed) => depth.completed = count,
                Some(JobStatus::Failed) => depth.failed = count,
                Some(JobStatus::Skipped) => depth.skipped = count,
                None => {
                    return Err(queue_error(
                        QueueErrorKind::Validation,
                        format!("unknown queue status value: {status:?}"),
                    ));
                }
            }
        }

        let now_ms = unix_timestamp_ms()?;
        let ready_params = [SqliteValue::Integer(now_ms)];
        let ready_rows = self
            .storage
            .connection()
            .query_with_params(
                "SELECT COUNT(*) FROM embedding_jobs WHERE status = 'pending' AND submitted_at <= ?1;",
                &ready_params,
            )
            .map_err(map_storage_error)?;
        if let Some(row) = ready_rows.first() {
            depth.ready_pending = i64_to_usize(row_i64(row, 0, "embedding_jobs.ready_pending")?)?;
        }

        Ok(depth)
    }

    fn record_enqueue_outcome(&self, outcome: EnqueueOutcome) {
        match outcome {
            EnqueueOutcome::Inserted | EnqueueOutcome::Replaced => {
                self.metrics.total_enqueued.fetch_add(1, Ordering::Relaxed);
            }
            EnqueueOutcome::Deduplicated | EnqueueOutcome::HashEmbedderSkipped => {
                self.metrics
                    .total_deduplicated
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

impl BatchEnqueueResult {
    fn record(&mut self, outcome: EnqueueOutcome) {
        match outcome {
            EnqueueOutcome::Inserted => self.inserted += 1,
            EnqueueOutcome::Replaced => self.replaced += 1,
            EnqueueOutcome::Deduplicated => self.deduplicated += 1,
            EnqueueOutcome::HashEmbedderSkipped => self.skipped_hash_embedder += 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EnqueueOutcome {
    Inserted,
    Replaced,
    Deduplicated,
    HashEmbedderSkipped,
}

#[derive(Debug, Clone, Copy)]
struct JobState {
    status: JobStatus,
    retry_count: u32,
    max_retries: u32,
    started_at: Option<i64>,
}

fn enqueue_inner(
    conn: &Connection,
    request: &EnqueueRequest,
    submitted_at: i64,
    max_retries: u32,
) -> SearchResult<EnqueueOutcome> {
    ensure_non_empty(&request.doc_id, "doc_id")?;
    ensure_non_empty(&request.embedder_id, "embedder_id")?;

    if !document_exists(conn, &request.doc_id)? {
        return Err(not_found_error("documents", &request.doc_id));
    }

    if is_hash_embedder(&request.embedder_id) {
        return Ok(EnqueueOutcome::HashEmbedderSkipped);
    }

    let active_params = [
        SqliteValue::Text(request.doc_id.clone()),
        SqliteValue::Text(request.embedder_id.clone()),
    ];
    let active_rows = conn
        .query_with_params(
            "SELECT job_id, content_hash \
             FROM embedding_jobs \
             WHERE doc_id = ?1 AND embedder_id = ?2 AND status IN ('pending', 'processing');",
            &active_params,
        )
        .map_err(map_storage_error)?;

    let mut has_active_job = false;
    for row in &active_rows {
        has_active_job = true;
        let existing_hash = row_optional_blob_32(row, 1, "embedding_jobs.content_hash")?;
        if existing_hash.as_ref() == Some(&request.content_hash) {
            return Ok(EnqueueOutcome::Deduplicated);
        }
    }

    if has_active_job {
        conn.execute_with_params(
            "DELETE FROM embedding_jobs \
             WHERE doc_id = ?1 AND embedder_id = ?2 AND status IN ('pending', 'processing');",
            &active_params,
        )
        .map_err(map_storage_error)?;
    }

    let insert_params = [
        SqliteValue::Text(request.doc_id.clone()),
        SqliteValue::Text(request.embedder_id.clone()),
        SqliteValue::Integer(i64::from(request.priority)),
        SqliteValue::Integer(submitted_at),
        SqliteValue::Integer(i64::from(max_retries)),
        SqliteValue::Blob(request.content_hash.to_vec()),
    ];
    conn.execute_with_params(
        "INSERT INTO embedding_jobs (\
            doc_id, embedder_id, priority, submitted_at, status, retry_count, max_retries, content_hash\
         ) VALUES (?1, ?2, ?3, ?4, 'pending', 0, ?5, ?6);",
        &insert_params,
    )
    .map_err(map_storage_error)?;

    Ok(if has_active_job {
        EnqueueOutcome::Replaced
    } else {
        EnqueueOutcome::Inserted
    })
}

fn load_job_state(conn: &Connection, job_id: i64) -> SearchResult<Option<JobState>> {
    let params = [SqliteValue::Integer(job_id)];
    let rows = conn
        .query_with_params(
            "SELECT status, retry_count, max_retries, started_at \
             FROM embedding_jobs \
             WHERE job_id = ?1 \
             LIMIT 1;",
            &params,
        )
        .map_err(map_storage_error)?;
    let Some(row) = rows.first() else {
        return Ok(None);
    };

    let status_value = row_text(row, 0, "embedding_jobs.status")?;
    let status = JobStatus::from_str(status_value).ok_or_else(|| {
        queue_error(
            QueueErrorKind::Validation,
            format!("unknown queue status value: {status_value:?}"),
        )
    })?;

    Ok(Some(JobState {
        status,
        retry_count: row_u32(row, 1, "embedding_jobs.retry_count")?,
        max_retries: row_u32(row, 2, "embedding_jobs.max_retries")?,
        started_at: row_optional_i64(row, 3)?,
    }))
}

fn document_exists(conn: &Connection, doc_id: &str) -> SearchResult<bool> {
    let params = [SqliteValue::Text(doc_id.to_owned())];
    let rows = conn
        .query_with_params(
            "SELECT doc_id FROM documents WHERE doc_id = ?1 LIMIT 1;",
            &params,
        )
        .map_err(map_storage_error)?;
    Ok(!rows.is_empty())
}

fn is_hash_embedder(embedder_id: &str) -> bool {
    embedder_id.starts_with(HASH_EMBEDDER_PREFIX) || embedder_id == "hash/fnv1a"
}

fn compute_retry_delay_ms(base_delay_ms: u64, exponent: u32) -> u64 {
    let shift = exponent.min(MAX_BACKOFF_EXPONENT);
    let factor = 1_u64.checked_shl(shift).unwrap_or(u64::MAX);
    base_delay_ms.saturating_mul(factor).min(MAX_RETRY_DELAY_MS)
}

fn ensure_non_empty(value: &str, field: &str) -> SearchResult<()> {
    if value.trim().is_empty() {
        return Err(queue_error(
            QueueErrorKind::Validation,
            format!("{field} must not be empty"),
        ));
    }
    Ok(())
}

fn not_found_error(entity: &str, key: &str) -> SearchError {
    queue_error(
        QueueErrorKind::NotFound,
        format!("{entity} record not found for key {key:?}"),
    )
}

fn conflict_error(message: String) -> SearchError {
    queue_error(QueueErrorKind::Conflict, message)
}

fn queue_error(kind: QueueErrorKind, message: String) -> SearchError {
    SearchError::SubsystemError {
        subsystem: SUBSYSTEM,
        source: Box::new(QueueError { kind, message }),
    }
}

fn row_text<'a>(row: &'a Row, index: usize, field: &str) -> SearchResult<&'a str> {
    match row.get(index) {
        Some(SqliteValue::Text(value)) => Ok(value),
        Some(other) => Err(SearchError::SubsystemError {
            subsystem: SUBSYSTEM,
            source: Box::new(io::Error::other(format!(
                "unexpected type for {field}: {:?}",
                other
            ))),
        }),
        None => Err(SearchError::SubsystemError {
            subsystem: SUBSYSTEM,
            source: Box::new(io::Error::other(format!("missing column for {field}"))),
        }),
    }
}

fn row_i64(row: &Row, index: usize, field: &str) -> SearchResult<i64> {
    match row.get(index) {
        Some(SqliteValue::Integer(value)) => Ok(*value),
        Some(other) => Err(SearchError::SubsystemError {
            subsystem: SUBSYSTEM,
            source: Box::new(io::Error::other(format!(
                "unexpected type for {field}: {:?}",
                other
            ))),
        }),
        None => Err(SearchError::SubsystemError {
            subsystem: SUBSYSTEM,
            source: Box::new(io::Error::other(format!("missing column for {field}"))),
        }),
    }
}

fn row_i32(row: &Row, index: usize, field: &str) -> SearchResult<i32> {
    let value = row_i64(row, index, field)?;
    i32::try_from(value).map_err(|_| {
        queue_error(
            QueueErrorKind::Validation,
            format!("{field} value {value} does not fit into i32"),
        )
    })
}

fn row_u32(row: &Row, index: usize, field: &str) -> SearchResult<u32> {
    let value = row_i64(row, index, field)?;
    u32::try_from(value).map_err(|_| {
        queue_error(
            QueueErrorKind::Validation,
            format!("{field} value {value} does not fit into u32"),
        )
    })
}

fn row_optional_i64(row: &Row, index: usize) -> SearchResult<Option<i64>> {
    match row.get(index) {
        Some(SqliteValue::Integer(value)) => Ok(Some(*value)),
        Some(SqliteValue::Null) | None => Ok(None),
        Some(other) => Err(SearchError::SubsystemError {
            subsystem: SUBSYSTEM,
            source: Box::new(io::Error::other(format!(
                "unexpected optional i64 type: {:?}",
                other
            ))),
        }),
    }
}

fn row_optional_blob_32(row: &Row, index: usize, field: &str) -> SearchResult<Option<[u8; 32]>> {
    match row.get(index) {
        Some(SqliteValue::Blob(value)) => {
            if value.len() != 32 {
                return Err(queue_error(
                    QueueErrorKind::Validation,
                    format!("{field} expected 32-byte hash, found {} bytes", value.len()),
                ));
            }
            let mut hash = [0_u8; 32];
            hash.copy_from_slice(value);
            Ok(Some(hash))
        }
        Some(SqliteValue::Null) | None => Ok(None),
        Some(other) => Err(SearchError::SubsystemError {
            subsystem: SUBSYSTEM,
            source: Box::new(io::Error::other(format!(
                "unexpected optional blob type for {field}: {:?}",
                other
            ))),
        }),
    }
}

fn usize_to_i64(value: usize) -> SearchResult<i64> {
    i64::try_from(value).map_err(|_| {
        queue_error(
            QueueErrorKind::Validation,
            format!("value {value} does not fit into i64"),
        )
    })
}

fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn i64_to_usize(value: i64) -> SearchResult<usize> {
    usize::try_from(value).map_err(|_| {
        queue_error(
            QueueErrorKind::Validation,
            format!("value {value} does not fit into usize"),
        )
    })
}

fn duration_as_u64(value: u128) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn unix_timestamp_ms() -> SearchResult<i64> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(map_storage_error)?;
    i64::try_from(duration.as_millis()).map_err(|_| {
        queue_error(
            QueueErrorKind::Validation,
            "system timestamp overflowed i64 milliseconds".to_owned(),
        )
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arc_with_non_send_sync)]

    use std::collections::HashSet;
    use std::path::PathBuf;
    use std::process;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    use fsqlite_types::value::SqliteValue;

    use crate::connection::{Storage, StorageConfig};
    use crate::document::DocumentRecord;

    use super::{
        ClaimedJob, EnqueueRequest, FailResult, JobQueueConfig, JobStatus, PersistentJobQueue,
        QueueDepth, unix_timestamp_ms,
    };

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
                "frankensearch-job-queue-{tag}-{}-{nanos}.sqlite3",
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

    fn queue_fixture(config: JobQueueConfig) -> (PersistentJobQueue, Arc<Storage>) {
        let storage = Arc::new(Storage::open_in_memory().expect("in-memory storage should open"));
        let queue = PersistentJobQueue::new(Arc::clone(&storage), config);
        (queue, storage)
    }

    fn insert_document(storage: &Storage, doc_id: &str, hash_seed: u8) {
        let mut hash = [0_u8; 32];
        hash.fill(hash_seed);
        let doc = DocumentRecord {
            doc_id: doc_id.to_owned(),
            source_path: Some(format!("tests://{doc_id}")),
            content_preview: format!("content for {doc_id}"),
            content_hash: hash,
            content_length: 32,
            created_at: 1_739_499_200,
            updated_at: 1_739_499_200,
            metadata: None,
        };
        storage
            .upsert_document(&doc)
            .expect("document insert should succeed");
    }

    fn status_counts(storage: &Storage) -> QueueDepth {
        let rows = storage
            .connection()
            .query("SELECT status, COUNT(*) FROM embedding_jobs GROUP BY status;")
            .expect("status query should succeed");
        let mut depth = QueueDepth::default();
        for row in &rows {
            let status = row
                .get(0)
                .and_then(|value| match value {
                    SqliteValue::Text(text) => Some(text.clone()),
                    _ => None,
                })
                .expect("status column should be text");
            let raw_count = row
                .get(1)
                .and_then(|value| match value {
                    SqliteValue::Integer(count) => Some(*count),
                    _ => None,
                })
                .expect("count column should be integer");
            let count = usize::try_from(raw_count).expect("count value should fit into usize");
            match JobStatus::from_str(&status).expect("queue status should be known") {
                JobStatus::Pending => depth.pending = count,
                JobStatus::Processing => depth.processing = count,
                JobStatus::Completed => depth.completed = count,
                JobStatus::Failed => depth.failed = count,
                JobStatus::Skipped => depth.skipped = count,
            }
        }
        depth
    }

    fn claim_single(queue: &PersistentJobQueue, worker_id: &str) -> ClaimedJob {
        let claimed = queue
            .claim_batch(worker_id, 1)
            .expect("claim should succeed");
        assert_eq!(claimed.len(), 1, "exactly one job should be claimed");
        claimed
            .into_iter()
            .next()
            .expect("claim result should contain a job")
    }

    #[test]
    fn enqueue_deduplicates_same_job() {
        let (queue, storage) = queue_fixture(JobQueueConfig::default());
        insert_document(storage.as_ref(), "doc-1", 1);

        let hash = [9_u8; 32];
        assert!(
            queue
                .enqueue("doc-1", "all-MiniLM-L6-v2", &hash, 7)
                .expect("initial enqueue should succeed")
        );
        assert!(
            !queue
                .enqueue("doc-1", "all-MiniLM-L6-v2", &hash, 7)
                .expect("duplicate enqueue should succeed")
        );

        let depth = queue.queue_depth().expect("queue depth should load");
        assert_eq!(depth.pending, 1);
        assert_eq!(depth.processing, 0);

        let metrics = queue.metrics().snapshot();
        assert_eq!(metrics.total_enqueued, 1);
        assert_eq!(metrics.total_deduplicated, 1);
    }

    #[test]
    fn enqueue_replaces_active_job_when_hash_changes() {
        let (queue, storage) = queue_fixture(JobQueueConfig::default());
        insert_document(storage.as_ref(), "doc-2", 2);

        let hash_a = [1_u8; 32];
        let hash_b = [2_u8; 32];
        assert!(
            queue
                .enqueue("doc-2", "all-MiniLM-L6-v2", &hash_a, 1)
                .expect("first enqueue should succeed")
        );
        assert!(
            queue
                .enqueue("doc-2", "all-MiniLM-L6-v2", &hash_b, 1)
                .expect("replacement enqueue should succeed")
        );

        let depth = status_counts(storage.as_ref());
        assert_eq!(
            depth.pending, 1,
            "only replacement pending job should remain"
        );
        assert_eq!(depth.processing, 0);

        let params = [SqliteValue::Text("doc-2".to_owned())];
        let rows = storage
            .connection()
            .query_with_params(
                "SELECT content_hash FROM embedding_jobs WHERE doc_id = ?1 AND status = 'pending' LIMIT 1;",
                &params,
            )
            .expect("pending row query should succeed");
        assert_eq!(rows.len(), 1);
        let pending_hash = rows[0]
            .get(0)
            .and_then(|value| match value {
                SqliteValue::Blob(bytes) => Some(bytes.clone()),
                _ => None,
            })
            .expect("pending hash should be blob");
        assert_eq!(pending_hash, hash_b.to_vec());
    }

    #[test]
    fn hash_embedder_jobs_are_skipped() {
        let (queue, storage) = queue_fixture(JobQueueConfig::default());
        insert_document(storage.as_ref(), "doc-3", 3);

        let hash = [3_u8; 32];
        assert!(
            !queue
                .enqueue("doc-3", "fnv1a-384", &hash, 0)
                .expect("hash enqueue should succeed")
        );

        let depth = queue.queue_depth().expect("queue depth should load");
        assert_eq!(depth.pending, 0);
        assert_eq!(depth.processing, 0);
        assert_eq!(queue.metrics().snapshot().total_deduplicated, 1);
    }

    #[test]
    fn claim_batch_assigns_disjoint_jobs() {
        let (queue, storage) = queue_fixture(JobQueueConfig {
            batch_size: 4,
            ..JobQueueConfig::default()
        });

        insert_document(storage.as_ref(), "doc-a", 4);
        insert_document(storage.as_ref(), "doc-b", 5);
        insert_document(storage.as_ref(), "doc-c", 6);

        let hash_a = [4_u8; 32];
        let hash_b = [5_u8; 32];
        let hash_c = [6_u8; 32];
        queue
            .enqueue("doc-a", "all-MiniLM-L6-v2", &hash_a, 0)
            .expect("enqueue a");
        queue
            .enqueue("doc-b", "all-MiniLM-L6-v2", &hash_b, 10)
            .expect("enqueue b");
        queue
            .enqueue("doc-c", "all-MiniLM-L6-v2", &hash_c, 5)
            .expect("enqueue c");

        let first = queue
            .claim_batch("worker-a", 2)
            .expect("first claim should succeed");
        assert_eq!(first.len(), 2);
        assert!(first[0].priority >= first[1].priority);

        let second = queue
            .claim_batch("worker-b", 2)
            .expect("second claim should succeed");
        assert_eq!(second.len(), 1);

        let mut seen = HashSet::new();
        for job in first.iter().chain(second.iter()) {
            assert!(
                seen.insert(job.job_id),
                "duplicate claim for job {}",
                job.job_id
            );
        }

        let depth = queue.queue_depth().expect("queue depth should load");
        assert_eq!(depth.pending, 0);
        assert_eq!(depth.processing, 3);
    }

    #[test]
    fn fail_transitions_retry_then_terminal_failure() {
        let (queue, storage) = queue_fixture(JobQueueConfig {
            max_retries: 1,
            retry_base_delay_ms: 0,
            ..JobQueueConfig::default()
        });
        insert_document(storage.as_ref(), "doc-fail", 8);

        let hash = [8_u8; 32];
        queue
            .enqueue("doc-fail", "all-MiniLM-L6-v2", &hash, 0)
            .expect("enqueue should succeed");
        let first_claim = claim_single(&queue, "worker-f1");

        let first_fail = queue
            .fail(first_claim.job_id, "transient failure")
            .expect("first fail should succeed");
        assert!(matches!(
            first_fail,
            FailResult::Retried { retry_count: 1, .. }
        ));

        let second_claim = claim_single(&queue, "worker-f2");
        assert_eq!(second_claim.job_id, first_claim.job_id);

        let second_fail = queue
            .fail(second_claim.job_id, "permanent failure")
            .expect("second fail should succeed");
        assert!(matches!(
            second_fail,
            FailResult::TerminalFailed { retry_count: 2 }
        ));

        let depth = queue.queue_depth().expect("queue depth should load");
        assert_eq!(depth.failed, 1);
        assert_eq!(depth.pending, 0);
        assert_eq!(depth.processing, 0);

        let metrics = queue.metrics().snapshot();
        assert_eq!(metrics.total_retried, 1);
        assert_eq!(metrics.total_failed, 1);
    }

    #[test]
    fn reclaim_stale_jobs_restores_processing_work() {
        let (queue, storage) = queue_fixture(JobQueueConfig {
            visibility_timeout_ms: 10,
            stale_job_threshold_ms: 10,
            ..JobQueueConfig::default()
        });
        insert_document(storage.as_ref(), "doc-stale", 9);

        let hash = [9_u8; 32];
        queue
            .enqueue("doc-stale", "all-MiniLM-L6-v2", &hash, 0)
            .expect("enqueue should succeed");
        let claim = claim_single(&queue, "worker-stale");

        let stale_started_at = unix_timestamp_ms()
            .expect("timestamp should resolve")
            .saturating_sub(1_000);
        let params = [
            SqliteValue::Integer(stale_started_at),
            SqliteValue::Integer(claim.job_id),
        ];
        storage
            .connection()
            .execute_with_params(
                "UPDATE embedding_jobs SET started_at = ?1 WHERE job_id = ?2;",
                &params,
            )
            .expect("stale timestamp update should succeed");

        let reclaimed = queue
            .reclaim_stale_jobs()
            .expect("stale reclaim should succeed");
        assert_eq!(reclaimed, 1);

        let depth = queue.queue_depth().expect("queue depth should load");
        assert_eq!(depth.pending, 1);
        assert_eq!(depth.processing, 0);

        let reclaimed_claim = claim_single(&queue, "worker-restored");
        assert_eq!(reclaimed_claim.job_id, claim.job_id);
    }

    #[test]
    fn restart_recovery_preserves_and_reclaims_jobs() {
        let tmp = TempDbPath::new("restart");
        let queue_config = JobQueueConfig {
            visibility_timeout_ms: 10,
            stale_job_threshold_ms: 10,
            ..JobQueueConfig::default()
        };

        let storage_a =
            Arc::new(Storage::open(tmp.config()).expect("initial storage open should succeed"));
        insert_document(storage_a.as_ref(), "doc-restart", 10);
        let queue_a = PersistentJobQueue::new(Arc::clone(&storage_a), queue_config);
        let hash = [10_u8; 32];
        queue_a
            .enqueue("doc-restart", "all-MiniLM-L6-v2", &hash, 0)
            .expect("enqueue should succeed");
        let claim = claim_single(&queue_a, "worker-before-restart");

        let stale_started_at = unix_timestamp_ms()
            .expect("timestamp should resolve")
            .saturating_sub(5_000);
        let params = [
            SqliteValue::Integer(stale_started_at),
            SqliteValue::Integer(claim.job_id),
        ];
        storage_a
            .connection()
            .execute_with_params(
                "UPDATE embedding_jobs SET started_at = ?1 WHERE job_id = ?2;",
                &params,
            )
            .expect("stale timestamp update should succeed");
        drop(queue_a);
        drop(storage_a);

        let storage_b =
            Arc::new(Storage::open(tmp.config()).expect("reopened storage should succeed"));
        let queue_b = PersistentJobQueue::new(Arc::clone(&storage_b), queue_config);

        let before_reclaim = queue_b.queue_depth().expect("queue depth should load");
        assert_eq!(before_reclaim.processing, 1);
        assert_eq!(before_reclaim.pending, 0);

        let reclaimed = queue_b
            .reclaim_stale_jobs()
            .expect("stale reclaim should succeed");
        assert_eq!(reclaimed, 1);

        let recovered = claim_single(&queue_b, "worker-after-restart");
        assert_eq!(recovered.job_id, claim.job_id);
    }

    #[test]
    fn enqueue_batch_is_atomic_on_partial_failure() {
        let (queue, storage) = queue_fixture(JobQueueConfig::default());
        insert_document(storage.as_ref(), "doc-ok", 11);

        let jobs = vec![
            EnqueueRequest::new("doc-ok", "all-MiniLM-L6-v2", [11_u8; 32], 0),
            EnqueueRequest::new("doc-missing", "all-MiniLM-L6-v2", [12_u8; 32], 0),
        ];
        let err = queue
            .enqueue_batch(&jobs)
            .expect_err("batch enqueue should fail when one row is invalid");
        assert!(
            err.to_string().contains("not_found"),
            "error should classify as not_found: {err}"
        );

        let depth = queue.queue_depth().expect("queue depth should load");
        assert_eq!(
            depth.pending, 0,
            "transaction should rollback partial batch insert"
        );
    }

    #[test]
    fn enqueue_batch_reports_insert_replace_dedup_and_hash_skip() {
        let (queue, storage) = queue_fixture(JobQueueConfig::default());
        insert_document(storage.as_ref(), "doc-batch-1", 13);
        insert_document(storage.as_ref(), "doc-batch-2", 14);

        let jobs = vec![
            EnqueueRequest::new("doc-batch-1", "all-MiniLM-L6-v2", [13_u8; 32], 0),
            EnqueueRequest::new("doc-batch-1", "all-MiniLM-L6-v2", [13_u8; 32], 0),
            EnqueueRequest::new("doc-batch-1", "all-MiniLM-L6-v2", [15_u8; 32], 0),
            EnqueueRequest::new("doc-batch-2", "fnv1a-384", [14_u8; 32], 0),
        ];
        let summary = queue
            .enqueue_batch(&jobs)
            .expect("batch enqueue should succeed");
        assert_eq!(summary.inserted, 1);
        assert_eq!(summary.replaced, 1);
        assert_eq!(summary.deduplicated, 1);
        assert_eq!(summary.skipped_hash_embedder, 1);

        let depth = queue.queue_depth().expect("queue depth should load");
        assert_eq!(
            depth.pending, 1,
            "only one pending semantic job should remain"
        );

        let metrics = queue.metrics().snapshot();
        assert_eq!(metrics.total_enqueued, 2);
        assert_eq!(metrics.total_deduplicated, 2);
    }
}
