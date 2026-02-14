//! Shared application state for async/sync bridge.
//!
//! The [`AppState`] holds fleet status, metrics, and connection info.
//! Background async tasks write updates; the synchronous render loop reads.
//! Thread safety is provided by the consumer's runtime (asupersync `RwLock`
//! when integrated; `std::sync::RwLock` for standalone testing).

use std::collections::HashMap;
use std::fmt;
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

// ─── Control-Plane Health ───────────────────────────────────────────────────

/// Severity level for control-plane health.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlPlaneHealth {
    /// All monitored signals are within healthy thresholds.
    Healthy,
    /// One or more signals are degraded but not critical.
    Degraded,
    /// One or more signals are at a critical threshold.
    Critical,
}

impl ControlPlaneHealth {
    /// Short status badge for status bar chrome.
    #[must_use]
    pub const fn badge(self) -> &'static str {
        match self {
            Self::Healthy => "CP:OK",
            Self::Degraded => "CP:WARN",
            Self::Critical => "CP:CRIT",
        }
    }
}

impl fmt::Display for ControlPlaneHealth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Healthy => write!(f, "healthy"),
            Self::Degraded => write!(f, "degraded"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

/// Internal self-monitoring metrics for the ops control plane.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlPlaneMetrics {
    /// Events currently backlogged in ingestion.
    pub ingestion_lag_events: u64,
    /// Approximate persisted bytes used by control-plane storage.
    pub storage_bytes: u64,
    /// Storage soft limit for warnings.
    pub storage_limit_bytes: u64,
    /// Average frame time budget in milliseconds.
    pub frame_time_ms: f64,
    /// Discovery scan latency in milliseconds.
    pub discovery_latency_ms: u64,
    /// Event processing throughput (events/sec).
    pub event_throughput_eps: f64,
    /// Resident set size of control-plane process.
    pub rss_bytes: u64,
    /// RSS soft limit for warnings.
    pub rss_limit_bytes: u64,
    /// Dead-letter events pending triage.
    pub dead_letter_events: u64,
}

impl Default for ControlPlaneMetrics {
    fn default() -> Self {
        Self {
            ingestion_lag_events: 0,
            storage_bytes: 0,
            storage_limit_bytes: 1,
            frame_time_ms: 16.0,
            discovery_latency_ms: 0,
            event_throughput_eps: 0.0,
            rss_bytes: 0,
            rss_limit_bytes: 1,
            dead_letter_events: 0,
        }
    }
}

impl ControlPlaneMetrics {
    const LAG_WARN_EVENTS: u64 = 1_000;
    const LAG_CRIT_EVENTS: u64 = 10_000;
    const DISCOVERY_WARN_MS: u64 = 2_000;
    const DISCOVERY_CRIT_MS: u64 = 5_000;
    const DEAD_LETTER_WARN: u64 = 1;
    const DEAD_LETTER_CRIT: u64 = 20;
    const STORAGE_WARN_RATIO: f64 = 0.80;
    const STORAGE_CRIT_RATIO: f64 = 0.95;
    const RSS_WARN_RATIO: f64 = 0.80;
    const RSS_CRIT_RATIO: f64 = 0.95;
    const FPS_WARN: f64 = 30.0;
    const FPS_CRIT: f64 = 15.0;

    fn ratio_as_f64(numer: u64, denom: u64) -> f64 {
        if denom == 0 {
            return 0.0;
        }
        let scaled = numer.saturating_mul(10_000).saturating_div(denom);
        let scaled_u32 = u32::try_from(scaled).unwrap_or(u32::MAX);
        f64::from(scaled_u32) / 10_000.0
    }

    /// Storage utilization ratio in `[0.0, +inf)`.
    #[must_use]
    pub fn storage_utilization(&self) -> f64 {
        Self::ratio_as_f64(self.storage_bytes, self.storage_limit_bytes)
    }

    /// RSS utilization ratio in `[0.0, +inf)`.
    #[must_use]
    pub fn rss_utilization(&self) -> f64 {
        Self::ratio_as_f64(self.rss_bytes, self.rss_limit_bytes)
    }

    /// Approximate renderer frame rate.
    #[must_use]
    pub fn estimated_fps(&self) -> f64 {
        if self.frame_time_ms <= 0.0 {
            0.0
        } else {
            1000.0 / self.frame_time_ms
        }
    }

    /// Compute aggregate control-plane health.
    #[must_use]
    pub fn health(&self) -> ControlPlaneHealth {
        let storage_ratio = self.storage_utilization();
        let rss_ratio = self.rss_utilization();
        let fps = self.estimated_fps();
        let lag = self.ingestion_lag_events;
        let dead = self.dead_letter_events;
        let discovery = self.discovery_latency_ms;

        if lag >= Self::LAG_CRIT_EVENTS
            || storage_ratio >= Self::STORAGE_CRIT_RATIO
            || rss_ratio >= Self::RSS_CRIT_RATIO
            || fps <= Self::FPS_CRIT
            || discovery >= Self::DISCOVERY_CRIT_MS
            || dead >= Self::DEAD_LETTER_CRIT
            || (lag >= 5_000 && self.event_throughput_eps < 0.5)
        {
            return ControlPlaneHealth::Critical;
        }

        if lag >= Self::LAG_WARN_EVENTS
            || storage_ratio >= Self::STORAGE_WARN_RATIO
            || rss_ratio >= Self::RSS_WARN_RATIO
            || fps <= Self::FPS_WARN
            || discovery >= Self::DISCOVERY_WARN_MS
            || dead >= Self::DEAD_LETTER_WARN
            || (lag > 0 && self.event_throughput_eps < 1.0)
        {
            return ControlPlaneHealth::Degraded;
        }

        ControlPlaneHealth::Healthy
    }

    /// Deterministic multi-line report for diagnostics and operator overlays.
    #[must_use]
    pub fn self_check_report(&self) -> String {
        format!(
            "health: {}\ningestion_lag_events: {}\nstorage_utilization: {:.1}% ({}/{})\nframe_rate_fps: {:.1}\ndiscovery_latency_ms: {}\nevent_throughput_eps: {:.2}\nrss_utilization: {:.1}% ({}/{})\ndead_letter_events: {}",
            self.health(),
            self.ingestion_lag_events,
            self.storage_utilization() * 100.0,
            self.storage_bytes,
            self.storage_limit_bytes,
            self.estimated_fps(),
            self.discovery_latency_ms,
            self.event_throughput_eps,
            self.rss_utilization() * 100.0,
            self.rss_bytes,
            self.rss_limit_bytes,
            self.dead_letter_events
        )
    }
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
    /// Internal self-monitoring metrics for this control plane.
    control_plane: ControlPlaneMetrics,
    /// Aggregate health derived from `control_plane`.
    control_plane_health: ControlPlaneHealth,
}

impl AppState {
    /// Create a new empty app state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            fleet: FleetSnapshot::default(),
            last_update: None,
            connection_status: "Discovering instances...".to_string(),
            control_plane: ControlPlaneMetrics::default(),
            control_plane_health: ControlPlaneHealth::Healthy,
        }
    }

    /// Update the fleet snapshot.
    pub fn update_fleet(&mut self, snapshot: FleetSnapshot) {
        self.fleet = snapshot;
        self.last_update = Some(Instant::now());
        self.refresh_connection_status();
    }

    /// Update internal control-plane metrics.
    pub fn update_control_plane(&mut self, metrics: ControlPlaneMetrics) {
        self.control_plane_health = metrics.health();
        self.control_plane = metrics;
    }

    fn refresh_connection_status(&mut self) {
        let count = self.fleet.instance_count();
        let healthy = self.fleet.healthy_count();
        self.connection_status = format!("{count} instances, {healthy} healthy");
    }

    /// Get the current fleet snapshot.
    #[must_use]
    pub const fn fleet(&self) -> &FleetSnapshot {
        &self.fleet
    }

    /// Get the connection status string.
    #[must_use]
    pub fn connection_status(&self) -> &str {
        &self.connection_status
    }

    /// Get control-plane self-monitoring metrics.
    #[must_use]
    pub const fn control_plane_metrics(&self) -> &ControlPlaneMetrics {
        &self.control_plane
    }

    /// Get current aggregate control-plane health.
    #[must_use]
    pub const fn control_plane_health(&self) -> ControlPlaneHealth {
        self.control_plane_health
    }

    /// Produce a deterministic self-check report for overlays/logging.
    #[must_use]
    pub fn self_check_report(&self) -> String {
        self.control_plane.self_check_report()
    }

    /// When the fleet was last updated.
    #[must_use]
    pub const fn last_update(&self) -> Option<Instant> {
        self.last_update
    }

    /// Whether we have received at least one fleet update.
    #[must_use]
    pub const fn has_data(&self) -> bool {
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
    fn control_plane_health_transitions() {
        let mut state = AppState::new();
        let healthy = ControlPlaneMetrics {
            event_throughput_eps: 10.0,
            ..ControlPlaneMetrics::default()
        };
        state.update_control_plane(healthy.clone());
        assert_eq!(state.control_plane_health(), ControlPlaneHealth::Healthy);

        let degraded = ControlPlaneMetrics {
            ingestion_lag_events: 2_500,
            ..healthy
        };
        state.update_control_plane(degraded.clone());
        assert_eq!(state.control_plane_health(), ControlPlaneHealth::Degraded);

        let critical = ControlPlaneMetrics {
            ingestion_lag_events: 12_000,
            ..degraded
        };
        state.update_control_plane(critical);
        assert_eq!(state.control_plane_health(), ControlPlaneHealth::Critical);
    }

    #[test]
    fn self_check_report_contains_core_fields() {
        let mut state = AppState::new();
        state.update_control_plane(ControlPlaneMetrics {
            ingestion_lag_events: 42,
            storage_bytes: 800,
            storage_limit_bytes: 1000,
            frame_time_ms: 20.0,
            discovery_latency_ms: 25,
            event_throughput_eps: 12.5,
            rss_bytes: 256,
            rss_limit_bytes: 1024,
            dead_letter_events: 0,
        });

        let report = state.self_check_report();
        assert!(report.contains("ingestion_lag_events: 42"));
        assert!(report.contains("storage_utilization"));
        assert!(report.contains("frame_rate_fps"));
        assert!(report.contains("dead_letter_events: 0"));
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
