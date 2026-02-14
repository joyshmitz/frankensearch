use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

#[derive(Debug, Default)]
pub struct StorageMetrics {
    opens: AtomicU64,
    schema_bootstraps: AtomicU64,
    tx_commits: AtomicU64,
    tx_rollbacks: AtomicU64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageMetricsSnapshot {
    pub opens: u64,
    pub schema_bootstraps: u64,
    pub tx_commits: u64,
    pub tx_rollbacks: u64,
}

impl StorageMetrics {
    pub fn record_open(&self) {
        self.opens.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_schema_bootstrap(&self) {
        self.schema_bootstraps.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_commit(&self) {
        self.tx_commits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_rollback(&self) {
        self.tx_rollbacks.fetch_add(1, Ordering::Relaxed);
    }

    #[must_use]
    pub fn snapshot(&self) -> StorageMetricsSnapshot {
        StorageMetricsSnapshot {
            opens: self.opens.load(Ordering::Relaxed),
            schema_bootstraps: self.schema_bootstraps.load(Ordering::Relaxed),
            tx_commits: self.tx_commits.load(Ordering::Relaxed),
            tx_rollbacks: self.tx_rollbacks.load(Ordering::Relaxed),
        }
    }
}
