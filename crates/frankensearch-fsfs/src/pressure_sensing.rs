//! Host pressure sensing with `sysinfo`-backed OS signal collection.
//!
//! This module provides a 4-state control model for runtime resource pressure:
//!
//! - **Normal** — all signals within comfortable bounds.
//! - **Constrained** — one or more signals near configured ceilings.
//! - **Degraded** — signals exceed primary ceilings; shed optional work.
//! - **Emergency** — critical resource exhaustion; shed everything non-essential.
//!
//! Key mechanisms:
//! - EWMA smoothing (configurable alpha, default 0.3) to dampen noise.
//! - Hysteresis via separate enter/exit thresholds per state boundary.
//! - Anti-flap guard requiring N consecutive readings before state transition.

use serde::{Deserialize, Serialize};
use sysinfo::System;
use tracing::{debug, info, warn};

// ─── Control State ──────────────────────────────────────────────────────────

/// Runtime control state derived from smoothed host signals.
///
/// Distinct from [`crate::config::PressureProfile`], which is a static config
/// preset; `ControlState` is the live runtime classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlState {
    Normal,
    Constrained,
    Degraded,
    Emergency,
}

impl ControlState {
    /// Numeric severity for ordering comparisons.
    #[must_use]
    pub const fn severity(self) -> u8 {
        match self {
            Self::Normal => 0,
            Self::Constrained => 1,
            Self::Degraded => 2,
            Self::Emergency => 3,
        }
    }
}

impl std::fmt::Display for ControlState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Normal => "normal",
            Self::Constrained => "constrained",
            Self::Degraded => "degraded",
            Self::Emergency => "emergency",
        };
        f.write_str(s)
    }
}

// ─── Raw Sample ─────────────────────────────────────────────────────────────

/// Raw sensor reading from the OS.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PressureSample {
    /// CPU utilization as a percentage (0.0 to 100.0+).
    pub cpu_pct: f64,
    /// Resident set size in bytes.
    pub rss_bytes: u64,
    /// Cumulative I/O read bytes since process start.
    pub io_read_bytes: u64,
    /// Cumulative I/O write bytes since process start.
    pub io_write_bytes: u64,
    /// 1-minute load average, if available.
    pub load_avg_1m: Option<f64>,
}

// ─── Smoothed Readings ──────────────────────────────────────────────────────

/// EWMA-smoothed sensor values used for state evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SmoothedReadings {
    pub cpu_pct: f64,
    pub memory_bytes: f64,
    pub io_bytes_per_sec: f64,
    pub load_avg: f64,
    pub samples_processed: u64,
}

impl SmoothedReadings {
    const fn zero() -> Self {
        Self {
            cpu_pct: 0.0,
            memory_bytes: 0.0,
            io_bytes_per_sec: 0.0,
            load_avg: 0.0,
            samples_processed: 0,
        }
    }

    /// Apply one raw sample via EWMA. First sample initializes directly.
    #[allow(clippy::cast_precision_loss)]
    fn update(&mut self, sample: &PressureSample, alpha: f64, prev_io_total: Option<u64>) {
        let cpu = clamp_sensor(sample.cpu_pct);
        let mem = sample.rss_bytes as f64;
        let load = clamp_sensor(sample.load_avg_1m.unwrap_or(0.0));

        // IO rate: delta bytes since last sample. On first sample, 0.
        let io_total = sample.io_read_bytes.saturating_add(sample.io_write_bytes);
        let io_rate = prev_io_total.map_or(0.0, |prev| io_total.saturating_sub(prev) as f64);

        if self.samples_processed == 0 {
            // First sample: initialize directly, no smoothing.
            self.cpu_pct = cpu;
            self.memory_bytes = mem;
            self.io_bytes_per_sec = io_rate;
            self.load_avg = load;
        } else {
            self.cpu_pct = ewma(self.cpu_pct, cpu, alpha);
            self.memory_bytes = ewma(self.memory_bytes, mem, alpha);
            self.io_bytes_per_sec = ewma(self.io_bytes_per_sec, io_rate, alpha);
            self.load_avg = ewma(self.load_avg, load, alpha);
        }
        self.samples_processed += 1;
    }
}

// ─── Thresholds ─────────────────────────────────────────────────────────────

/// Enter/exit threshold pair implementing hysteresis.
///
/// `enter` is the value that must be exceeded to enter a worse state;
/// `exit` is the value that must be dropped below to return to a better state.
/// Invariant: `exit <= enter`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThresholdPair {
    pub enter: f64,
    pub exit: f64,
}

/// Thresholds for each state boundary, covering CPU and memory dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PressureThresholds {
    pub constrained_cpu: ThresholdPair,
    pub constrained_memory_mb: ThresholdPair,
    pub degraded_cpu: ThresholdPair,
    pub degraded_memory_mb: ThresholdPair,
    pub emergency_cpu: ThresholdPair,
    pub emergency_memory_mb: ThresholdPair,
}

impl Default for PressureThresholds {
    fn default() -> Self {
        Self {
            constrained_cpu: ThresholdPair {
                enter: 60.0,
                exit: 50.0,
            },
            constrained_memory_mb: ThresholdPair {
                enter: 1024.0,
                exit: 896.0,
            },
            degraded_cpu: ThresholdPair {
                enter: 80.0,
                exit: 70.0,
            },
            degraded_memory_mb: ThresholdPair {
                enter: 1536.0,
                exit: 1280.0,
            },
            emergency_cpu: ThresholdPair {
                enter: 95.0,
                exit: 85.0,
            },
            emergency_memory_mb: ThresholdPair {
                enter: 1920.0,
                exit: 1664.0,
            },
        }
    }
}

// ─── Transition ─────────────────────────────────────────────────────────────

/// Emitted when the control state changes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PressureTransition {
    pub from: ControlState,
    pub to: ControlState,
    pub reason: String,
    pub smoothed: SmoothedReadings,
}

// ─── Pressure Sensor (State Machine) ────────────────────────────────────────

/// Stateful pressure sensor implementing EWMA + hysteresis + anti-flap.
#[derive(Debug, Clone)]
pub struct PressureSensor {
    state: ControlState,
    smoothed: SmoothedReadings,
    alpha: f64,
    anti_flap_threshold: u32,
    consecutive_toward: u32,
    pending_state: Option<ControlState>,
    thresholds: PressureThresholds,
    transition_count: u64,
    prev_io_total: Option<u64>,
}

impl PressureSensor {
    /// Create a sensor with explicit configuration.
    #[must_use]
    pub fn new(thresholds: PressureThresholds, alpha: f64, anti_flap: u32) -> Self {
        Self {
            state: ControlState::Normal,
            smoothed: SmoothedReadings::zero(),
            alpha: alpha.clamp(0.0, 1.0),
            anti_flap_threshold: anti_flap.max(1),
            consecutive_toward: 0,
            pending_state: None,
            thresholds,
            transition_count: 0,
            prev_io_total: None,
        }
    }

    /// Create a sensor with default thresholds (`alpha=0.3`, `anti_flap=3`).
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(PressureThresholds::default(), 0.3, 3)
    }

    /// Ingest one raw sample and possibly produce a state transition.
    ///
    /// Returns `Some(PressureTransition)` when the state actually changes,
    /// `None` when it remains stable or the anti-flap guard holds.
    #[must_use]
    pub fn process_sample(&mut self, sample: &PressureSample) -> Option<PressureTransition> {
        // Update EWMA smoothed readings.
        self.smoothed.update(sample, self.alpha, self.prev_io_total);
        self.prev_io_total = Some(sample.io_read_bytes.saturating_add(sample.io_write_bytes));

        debug!(
            target: "frankensearch.fsfs.pressure",
            cpu_pct = sample.cpu_pct,
            rss_bytes = sample.rss_bytes,
            smoothed_cpu = self.smoothed.cpu_pct,
            smoothed_mem = self.smoothed.memory_bytes,
            state = %self.state,
            "pressure sample ingested"
        );

        // Determine target state using hysteresis.
        let target = self.target_state();

        if target == self.state {
            // Stable — reset pending.
            self.pending_state = None;
            self.consecutive_toward = 0;
            return None;
        }

        // Accumulate consecutive readings toward the target state.
        if self.pending_state == Some(target) {
            self.consecutive_toward += 1;
        } else {
            self.pending_state = Some(target);
            self.consecutive_toward = 1;
        }

        // Anti-flap: only transition after N consecutive readings.
        if self.consecutive_toward < self.anti_flap_threshold {
            return None;
        }

        // Execute transition.
        let from = self.state;
        self.state = target;
        self.pending_state = None;
        self.consecutive_toward = 0;
        self.transition_count += 1;

        let reason = format!(
            "cpu={:.1}% mem={:.0}B io={:.0}B/s load={:.2}",
            self.smoothed.cpu_pct,
            self.smoothed.memory_bytes,
            self.smoothed.io_bytes_per_sec,
            self.smoothed.load_avg,
        );

        let transition = PressureTransition {
            from,
            to: target,
            reason: reason.clone(),
            smoothed: self.smoothed,
        };

        // Emit tracing events at appropriate severity.
        match target {
            ControlState::Emergency => {
                warn!(
                    target: "frankensearch.fsfs.pressure",
                    from = %from,
                    to = %target,
                    reason = %reason,
                    "pressure state EMERGENCY"
                );
            }
            _ => {
                info!(
                    target: "frankensearch.fsfs.pressure",
                    from = %from,
                    to = %target,
                    reason = %reason,
                    "pressure state transition"
                );
            }
        }

        Some(transition)
    }

    /// Current control state.
    #[must_use]
    pub const fn current_state(&self) -> ControlState {
        self.state
    }

    /// Current smoothed readings.
    #[must_use]
    pub const fn smoothed_readings(&self) -> &SmoothedReadings {
        &self.smoothed
    }

    /// Total number of state transitions since creation.
    #[must_use]
    pub const fn transition_count(&self) -> u64 {
        self.transition_count
    }

    /// Evaluate the target state using hysteresis-aware thresholds.
    ///
    /// For escalation (entering a worse state), the `enter` threshold applies.
    /// For de-escalation (returning to a better state), the `exit` threshold
    /// applies. Values between `exit` and `enter` keep the current state.
    fn target_state(&self) -> ControlState {
        let cpu = self.smoothed.cpu_pct;
        let mem_mb = self.smoothed.memory_bytes / (1024.0 * 1024.0);
        let t = &self.thresholds;

        // Check from highest severity downward.
        // Emergency boundary.
        if self.state.severity() >= ControlState::Emergency.severity() {
            // Currently emergency: need to drop below exit to leave.
            if cpu < t.emergency_cpu.exit && mem_mb < t.emergency_memory_mb.exit {
                // Fall through to check degraded.
            } else {
                return ControlState::Emergency;
            }
        } else {
            // Not currently emergency: need to exceed enter to escalate.
            if cpu >= t.emergency_cpu.enter || mem_mb >= t.emergency_memory_mb.enter {
                return ControlState::Emergency;
            }
        }

        // Degraded boundary.
        if self.state.severity() >= ControlState::Degraded.severity() {
            if cpu < t.degraded_cpu.exit && mem_mb < t.degraded_memory_mb.exit {
                // Fall through to check constrained.
            } else {
                return ControlState::Degraded;
            }
        } else if cpu >= t.degraded_cpu.enter || mem_mb >= t.degraded_memory_mb.enter {
            return ControlState::Degraded;
        }

        // Constrained boundary.
        if self.state.severity() >= ControlState::Constrained.severity() {
            if cpu < t.constrained_cpu.exit && mem_mb < t.constrained_memory_mb.exit {
                return ControlState::Normal;
            }
            return ControlState::Constrained;
        }

        if cpu >= t.constrained_cpu.enter || mem_mb >= t.constrained_memory_mb.enter {
            return ControlState::Constrained;
        }

        ControlState::Normal
    }
}

// ─── Host Sampler ───────────────────────────────────────────────────────────

/// OS signal collector backed by the `sysinfo` crate.
///
/// On Linux, also reads `/proc/self/io` for cumulative I/O counters.
pub struct HostSampler {
    system: System,
    prev_io: Option<(u64, u64)>,
}

impl std::fmt::Debug for HostSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostSampler")
            .field("prev_io", &self.prev_io)
            .finish_non_exhaustive()
    }
}

impl Default for HostSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl HostSampler {
    /// Create a new sampler. Performs initial sysinfo refresh.
    #[must_use]
    pub fn new() -> Self {
        let mut system = System::new();
        system.refresh_cpu_all();
        system.refresh_memory();
        Self {
            system,
            prev_io: None,
        }
    }

    /// Collect one pressure sample from the OS.
    #[allow(clippy::cast_precision_loss)]
    pub fn sample(&mut self) -> PressureSample {
        self.system.refresh_cpu_all();
        self.system.refresh_memory();

        let cpu_pct = f64::from(self.system.global_cpu_usage());

        // RSS: sysinfo gives us per-process info via Pid lookup.
        let pid = sysinfo::get_current_pid().ok();
        let rss_bytes = pid
            .and_then(|p| {
                self.system
                    .refresh_processes(sysinfo::ProcessesToUpdate::Some(&[p]), true);
                self.system.process(p).map(sysinfo::Process::memory)
            })
            .unwrap_or(0);

        // I/O counters from /proc/self/io on Linux.
        let (io_read, io_write) = read_proc_self_io().unwrap_or((0, 0));

        let load_avg_1m = {
            let load = System::load_average();
            if load.one >= 0.0 {
                Some(load.one)
            } else {
                None
            }
        };

        PressureSample {
            cpu_pct,
            rss_bytes,
            io_read_bytes: io_read,
            io_write_bytes: io_write,
            load_avg_1m,
        }
    }
}

/// Read `/proc/self/io` for cumulative read/write byte counters.
///
/// Returns `None` on non-Linux or when the file is unreadable.
fn read_proc_self_io() -> Option<(u64, u64)> {
    #[cfg(target_os = "linux")]
    {
        let contents = std::fs::read_to_string("/proc/self/io").ok()?;
        let mut read_bytes = None;
        let mut write_bytes = None;
        for line in contents.lines() {
            let mut parts = line.split_whitespace();
            let key = parts.next().unwrap_or_default();
            let val = parts.next().unwrap_or_default();
            if key == "read_bytes:" {
                read_bytes = val.parse::<u64>().ok();
            } else if key == "write_bytes:" {
                write_bytes = val.parse::<u64>().ok();
            }
        }
        Some((read_bytes?, write_bytes?))
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// EWMA: `alpha * new + (1 - alpha) * old`.
fn ewma(old: f64, new: f64, alpha: f64) -> f64 {
    alpha.mul_add(new, (1.0 - alpha) * old)
}

/// Clamp sensor value: NaN/negative => 0, finite otherwise.
fn clamp_sensor(value: f64) -> f64 {
    if !value.is_finite() || value < 0.0 {
        0.0
    } else {
        value
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sample(
        cpu: f64,
        rss_mb: u64,
        io_read: u64,
        io_write: u64,
        load: f64,
    ) -> PressureSample {
        PressureSample {
            cpu_pct: cpu,
            rss_bytes: rss_mb * 1024 * 1024,
            io_read_bytes: io_read,
            io_write_bytes: io_write,
            load_avg_1m: Some(load),
        }
    }

    fn idle_sample() -> PressureSample {
        make_sample(5.0, 100, 0, 0, 0.5)
    }

    // ── ControlState ──

    #[test]
    fn control_state_display_and_serde() {
        let states = [
            (ControlState::Normal, "normal"),
            (ControlState::Constrained, "constrained"),
            (ControlState::Degraded, "degraded"),
            (ControlState::Emergency, "emergency"),
        ];
        for (state, expected) in states {
            assert_eq!(format!("{state}"), expected);
            let json = serde_json::to_string(&state).unwrap();
            let parsed: ControlState = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, state);
        }
    }

    #[test]
    fn control_state_severity_ordering() {
        assert!(ControlState::Normal.severity() < ControlState::Constrained.severity());
        assert!(ControlState::Constrained.severity() < ControlState::Degraded.severity());
        assert!(ControlState::Degraded.severity() < ControlState::Emergency.severity());
    }

    // ── EWMA ──

    #[test]
    fn ewma_first_sample_initializes_directly() {
        let mut sensor = PressureSensor::with_defaults();
        let sample = make_sample(75.0, 512, 0, 0, 1.0);
        let _ = sensor.process_sample(&sample);
        assert!(
            (sensor.smoothed_readings().cpu_pct - 75.0).abs() < f64::EPSILON,
            "first sample should initialize directly, got {}",
            sensor.smoothed_readings().cpu_pct
        );
    }

    #[test]
    fn ewma_smoothing_converges() {
        let mut sensor = PressureSensor::new(PressureThresholds::default(), 0.3, 1);
        // First sample at 100.
        let _ = sensor.process_sample(&make_sample(100.0, 100, 0, 0, 0.0));
        // Second sample at 0 => smoothed = 0.3*0 + 0.7*100 = 70.
        let _ = sensor.process_sample(&make_sample(0.0, 100, 0, 0, 0.0));
        assert!(
            (sensor.smoothed_readings().cpu_pct - 70.0).abs() < 0.001,
            "expected ~70.0, got {}",
            sensor.smoothed_readings().cpu_pct
        );
        // Third sample at 0 => smoothed = 0.3*0 + 0.7*70 = 49.
        let _ = sensor.process_sample(&make_sample(0.0, 100, 0, 0, 0.0));
        assert!(
            (sensor.smoothed_readings().cpu_pct - 49.0).abs() < 0.001,
            "expected ~49.0, got {}",
            sensor.smoothed_readings().cpu_pct
        );
    }

    // ── State Transitions ──

    #[test]
    fn normal_to_constrained_transition() {
        // anti_flap=1 so transition happens immediately.
        let mut sensor = PressureSensor::new(PressureThresholds::default(), 1.0, 1);
        // CPU 65% exceeds constrained enter=60%.
        let result = sensor.process_sample(&make_sample(65.0, 100, 0, 0, 0.0));
        assert!(result.is_some());
        let t = result.unwrap();
        assert_eq!(t.from, ControlState::Normal);
        assert_eq!(t.to, ControlState::Constrained);
        assert_eq!(sensor.current_state(), ControlState::Constrained);
    }

    #[test]
    fn anti_flap_requires_consecutive_readings() {
        let mut sensor = PressureSensor::new(PressureThresholds::default(), 1.0, 3);
        let hot = make_sample(65.0, 100, 0, 0, 0.0);

        // First two readings: no transition.
        assert!(sensor.process_sample(&hot).is_none());
        assert!(sensor.process_sample(&hot).is_none());
        assert_eq!(sensor.current_state(), ControlState::Normal);

        // Third reading: transition fires.
        let result = sensor.process_sample(&hot);
        assert!(result.is_some());
        assert_eq!(sensor.current_state(), ControlState::Constrained);
    }

    #[test]
    fn anti_flap_resets_on_different_target() {
        let mut sensor = PressureSensor::new(PressureThresholds::default(), 1.0, 3);
        let constrained = make_sample(65.0, 100, 0, 0, 0.0);
        let degraded = make_sample(85.0, 100, 0, 0, 0.0);

        // Two readings toward constrained.
        assert!(sensor.process_sample(&constrained).is_none());
        assert!(sensor.process_sample(&constrained).is_none());

        // Switch to degraded — counter resets.
        assert!(sensor.process_sample(&degraded).is_none());
        assert!(sensor.process_sample(&degraded).is_none());

        // Third degraded reading triggers transition.
        let result = sensor.process_sample(&degraded);
        assert!(result.is_some());
        assert_eq!(sensor.current_state(), ControlState::Degraded);
    }

    #[test]
    fn hysteresis_prevents_oscillation() {
        let mut sensor = PressureSensor::new(PressureThresholds::default(), 1.0, 1);

        // Escalate to constrained (enter=60, exit=50).
        let _ = sensor.process_sample(&make_sample(65.0, 100, 0, 0, 0.0));
        assert_eq!(sensor.current_state(), ControlState::Constrained);

        // CPU at 55% — above exit (50) but below enter (60).
        // Should stay constrained due to hysteresis.
        let result = sensor.process_sample(&make_sample(55.0, 100, 0, 0, 0.0));
        assert!(result.is_none());
        assert_eq!(sensor.current_state(), ControlState::Constrained);

        // CPU at 45% — below exit (50). Should de-escalate.
        let result = sensor.process_sample(&make_sample(45.0, 100, 0, 0, 0.0));
        assert!(result.is_some());
        assert_eq!(sensor.current_state(), ControlState::Normal);
    }

    #[test]
    fn full_escalation_normal_to_emergency() {
        let mut sensor = PressureSensor::new(PressureThresholds::default(), 1.0, 1);

        // Normal -> Constrained (enter=60).
        let _ = sensor.process_sample(&make_sample(65.0, 100, 0, 0, 0.0));
        assert_eq!(sensor.current_state(), ControlState::Constrained);

        // Constrained -> Degraded (enter=80).
        let _ = sensor.process_sample(&make_sample(85.0, 100, 0, 0, 0.0));
        assert_eq!(sensor.current_state(), ControlState::Degraded);

        // Degraded -> Emergency (enter=95).
        let _ = sensor.process_sample(&make_sample(96.0, 100, 0, 0, 0.0));
        assert_eq!(sensor.current_state(), ControlState::Emergency);
    }

    #[test]
    fn de_escalation_emergency_to_normal() {
        let mut sensor = PressureSensor::new(PressureThresholds::default(), 1.0, 1);

        // Escalate to emergency.
        let _ = sensor.process_sample(&make_sample(96.0, 100, 0, 0, 0.0));
        assert_eq!(sensor.current_state(), ControlState::Emergency);

        // Drop below emergency exit (85) but above degraded exit (70).
        let _ = sensor.process_sample(&make_sample(75.0, 100, 0, 0, 0.0));
        assert_eq!(sensor.current_state(), ControlState::Degraded);

        // Drop below degraded exit (70) but above constrained exit (50).
        let _ = sensor.process_sample(&make_sample(55.0, 100, 0, 0, 0.0));
        assert_eq!(sensor.current_state(), ControlState::Constrained);

        // Drop below constrained exit (50).
        let _ = sensor.process_sample(&make_sample(40.0, 100, 0, 0, 0.0));
        assert_eq!(sensor.current_state(), ControlState::Normal);
    }

    #[test]
    fn threshold_defaults_are_sensible() {
        // Default thresholds should not trigger on an idle system.
        let mut sensor = PressureSensor::with_defaults();
        for _ in 0..10 {
            let result = sensor.process_sample(&idle_sample());
            assert!(result.is_none());
        }
        assert_eq!(sensor.current_state(), ControlState::Normal);
    }

    #[test]
    fn host_sampler_returns_valid_sample() {
        let mut sampler = HostSampler::new();
        let sample = sampler.sample();
        // CPU should be non-negative (may be 0 on first call).
        assert!(sample.cpu_pct >= 0.0);
        // RSS should be > 0 for our running process.
        // Note: on some CI environments sysinfo may not find our process.
        // We just verify it doesn't panic.
    }

    #[test]
    fn pressure_sensor_with_defaults() {
        let sensor = PressureSensor::with_defaults();
        assert_eq!(sensor.current_state(), ControlState::Normal);
        assert_eq!(sensor.transition_count(), 0);
        assert_eq!(sensor.smoothed_readings().samples_processed, 0);
    }

    #[test]
    fn transition_count_increments() {
        let mut sensor = PressureSensor::new(PressureThresholds::default(), 1.0, 1);
        assert_eq!(sensor.transition_count(), 0);

        let _ = sensor.process_sample(&make_sample(65.0, 100, 0, 0, 0.0));
        assert_eq!(sensor.transition_count(), 1);

        let _ = sensor.process_sample(&make_sample(85.0, 100, 0, 0, 0.0));
        assert_eq!(sensor.transition_count(), 2);
    }

    #[test]
    fn zero_readings_stay_normal() {
        let mut sensor = PressureSensor::with_defaults();
        let zero = PressureSample {
            cpu_pct: 0.0,
            rss_bytes: 0,
            io_read_bytes: 0,
            io_write_bytes: 0,
            load_avg_1m: Some(0.0),
        };
        for _ in 0..5 {
            assert!(sensor.process_sample(&zero).is_none());
        }
        assert_eq!(sensor.current_state(), ControlState::Normal);
    }

    #[test]
    fn nan_readings_handled_gracefully() {
        let mut sensor = PressureSensor::with_defaults();
        let nan_sample = PressureSample {
            cpu_pct: f64::NAN,
            rss_bytes: 0,
            io_read_bytes: 0,
            io_write_bytes: 0,
            load_avg_1m: Some(f64::NAN),
        };
        // NaN should be clamped to 0, not cause panics or transitions.
        for _ in 0..5 {
            let result = sensor.process_sample(&nan_sample);
            assert!(result.is_none());
        }
        assert_eq!(sensor.current_state(), ControlState::Normal);
        assert!(sensor.smoothed_readings().cpu_pct.abs() < f64::EPSILON);
        assert!(sensor.smoothed_readings().load_avg.abs() < f64::EPSILON);
    }

    #[test]
    fn memory_threshold_triggers_transition() {
        // Verify memory-based escalation works independently of CPU.
        let thresholds = PressureThresholds {
            constrained_memory_mb: ThresholdPair {
                enter: 500.0,
                exit: 400.0,
            },
            ..PressureThresholds::default()
        };
        let mut sensor = PressureSensor::new(thresholds, 1.0, 1);

        // CPU low, memory high (600 MB > enter 500 MB).
        let sample = make_sample(10.0, 600, 0, 0, 0.0);
        let result = sensor.process_sample(&sample);
        assert!(result.is_some());
        assert_eq!(sensor.current_state(), ControlState::Constrained);
    }
}
