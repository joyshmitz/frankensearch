//! Shared application state for async/sync bridge.
//!
//! The [`AppState`] holds fleet status, metrics, and connection info.
//! Background async tasks write updates; the synchronous render loop reads.
//! Thread safety is provided by the consumer's runtime (asupersync `RwLock`
//! when integrated; `std::sync::RwLock` for standalone testing).

use std::collections::HashMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};

// ─── Instance Info ───────────────────────────────────────────────────────────

/// Discovered frankensearch instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceInfo {
    /// Unique instance identifier.
    pub id: String,
    /// Host project name (e.g., "cass", "xf", "agent-mail").
    pub project: String,
    /// Process ID on the host machine.
    pub pid: Option<u32>,
    /// Whether the instance is currently healthy.
    pub healthy: bool,
    /// Number of indexed documents.
    pub doc_count: u64,
    /// Number of pending embedding jobs.
    pub pending_jobs: u64,
}

// ─── Metrics Snapshot ────────────────────────────────────────────────────────

/// Resource metrics snapshot for an instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    /// CPU usage percentage (0.0 - 100.0).
    pub cpu_percent: f64,
    /// Memory usage in bytes.
    pub memory_bytes: u64,
    /// Disk I/O bytes read since last snapshot.
    pub io_read_bytes: u64,
    /// Disk I/O bytes written since last snapshot.
    pub io_write_bytes: u64,
}

/// Search performance metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchMetrics {
    /// Total searches in the current window.
    pub total_searches: u64,
    /// Average search latency in microseconds.
    pub avg_latency_us: u64,
    /// P95 search latency in microseconds.
    pub p95_latency_us: u64,
    /// Number of searches that used refinement.
    pub refined_count: u64,
}

// ─── Fleet Snapshot ──────────────────────────────────────────────────────────

/// Complete fleet snapshot for rendering.
#[derive(Debug, Clone, Default)]
pub struct FleetSnapshot {
    /// All discovered instances.
    pub instances: Vec<InstanceInfo>,
    /// Per-instance resource metrics (keyed by instance ID).
    pub resources: HashMap<String, ResourceMetrics>,
    /// Per-instance search metrics (keyed by instance ID).
    pub search_metrics: HashMap<String, SearchMetrics>,
}

impl FleetSnapshot {
    /// Number of discovered instances.
    #[must_use]
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    /// Number of healthy instances.
    #[must_use]
    pub fn healthy_count(&self) -> usize {
        self.instances.iter().filter(|i| i.healthy).count()
    }

    /// Total documents across all instances.
    #[must_use]
    pub fn total_docs(&self) -> u64 {
        self.instances.iter().map(|i| i.doc_count).sum()
    }

    /// Total pending jobs across all instances.
    #[must_use]
    pub fn total_pending_jobs(&self) -> u64 {
        self.instances.iter().map(|i| i.pending_jobs).sum()
    }
}

// ─── App State ───────────────────────────────────────────────────────────────

/// Shared application state read by the render loop.
///
/// Background async tasks update this via `update_fleet()`.
/// The render loop reads it via `fleet()`.
#[derive(Debug, Clone)]
pub struct AppState {
    /// Latest fleet snapshot.
    fleet: FleetSnapshot,
    /// When the fleet was last updated.
    last_update: Option<Instant>,
    /// Connection status message.
    connection_status: String,
}

impl AppState {
    /// Create a new empty app state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            fleet: FleetSnapshot::default(),
            last_update: None,
            connection_status: "Discovering instances...".to_string(),
        }
    }

    /// Update the fleet snapshot.
    pub fn update_fleet(&mut self, snapshot: FleetSnapshot) {
        let count = snapshot.instance_count();
        let healthy = snapshot.healthy_count();
        self.connection_status = format!("{count} instances, {healthy} healthy");
        self.fleet = snapshot;
        self.last_update = Some(Instant::now());
    }

    /// Get the current fleet snapshot.
    #[must_use]
    pub fn fleet(&self) -> &FleetSnapshot {
        &self.fleet
    }

    /// Get the connection status string.
    #[must_use]
    pub fn connection_status(&self) -> &str {
        &self.connection_status
    }

    /// When the fleet was last updated.
    #[must_use]
    pub fn last_update(&self) -> Option<Instant> {
        self.last_update
    }

    /// Whether we have received at least one fleet update.
    #[must_use]
    pub fn has_data(&self) -> bool {
        self.last_update.is_some()
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_snapshot() -> FleetSnapshot {
        FleetSnapshot {
            instances: vec![
                InstanceInfo {
                    id: "inst-1".to_string(),
                    project: "cass".to_string(),
                    pid: Some(1234),
                    healthy: true,
                    doc_count: 5000,
                    pending_jobs: 10,
                },
                InstanceInfo {
                    id: "inst-2".to_string(),
                    project: "xf".to_string(),
                    pid: Some(5678),
                    healthy: false,
                    doc_count: 3000,
                    pending_jobs: 200,
                },
            ],
            resources: HashMap::new(),
            search_metrics: HashMap::new(),
        }
    }

    #[test]
    fn app_state_initial() {
        let state = AppState::new();
        assert!(!state.has_data());
        assert_eq!(state.fleet().instance_count(), 0);
    }

    #[test]
    fn app_state_update_fleet() {
        let mut state = AppState::new();
        state.update_fleet(sample_snapshot());
        assert!(state.has_data());
        assert_eq!(state.fleet().instance_count(), 2);
        assert_eq!(state.fleet().healthy_count(), 1);
    }

    #[test]
    fn fleet_snapshot_aggregates() {
        let snap = sample_snapshot();
        assert_eq!(snap.total_docs(), 8000);
        assert_eq!(snap.total_pending_jobs(), 210);
    }

    #[test]
    fn connection_status_updates() {
        let mut state = AppState::new();
        assert!(state.connection_status().contains("Discovering"));
        state.update_fleet(sample_snapshot());
        assert!(state.connection_status().contains("2 instances"));
        assert!(state.connection_status().contains("1 healthy"));
    }

    #[test]
    fn instance_info_serde_roundtrip() {
        let info = InstanceInfo {
            id: "test".to_string(),
            project: "proj".to_string(),
            pid: Some(42),
            healthy: true,
            doc_count: 100,
            pending_jobs: 5,
        };
        let json = serde_json::to_string(&info).unwrap();
        let decoded: InstanceInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, info.id);
        assert_eq!(decoded.doc_count, info.doc_count);
    }
}
