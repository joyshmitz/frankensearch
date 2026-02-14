//! Deterministic multi-instance telemetry simulator for ops testing.
//!
//! This module generates reproducible fleet telemetry across multiple host
//! projects and workload profiles. It is designed to feed integration, e2e,
//! and performance-style tests without requiring live runtimes.
#![allow(clippy::missing_const_for_fn)]

use std::collections::BTreeSet;
use std::time::Duration;

use frankensearch_core::{
    PhaseMetrics, QueryClass, RankChanges, ScoreSource, ScoredResult, SearchError, SearchPhase,
    SearchResult, SkipReason,
};
use serde::{Deserialize, Serialize};

use crate::storage::{
    OpsIngestionMetricsSnapshot, ResourceSampleRecord, SearchEventPhase, SearchEventRecord,
};
use crate::{
    DiscoveredInstance, DiscoverySignalKind, DiscoveryStatus, OpsStorage, SloMaterializationConfig,
    SloMaterializationResult, SloScope,
};

/// Workload archetypes used by the simulator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkloadProfile {
    /// Stable traffic with low variance.
    Steady,
    /// Periodic query storms with elevated latency.
    Burst,
    /// Oscillating embedding backlog pressure.
    EmbeddingWave,
    /// Frequent restart-like behavior with degraded periods.
    Restarting,
}

/// One host-project simulation lane.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SimulatedProject {
    /// Telemetry project key used in storage records.
    pub project_key: String,
    /// Host name prefix for generated instances.
    pub host_name: String,
    /// Number of instances to generate for this project.
    pub instance_count: usize,
    /// Workload profile to emulate.
    pub workload: WorkloadProfile,
}

impl SimulatedProject {
    fn validate(&self) -> SearchResult<()> {
        if self.project_key.trim().is_empty() {
            return Err(SearchError::InvalidConfig {
                field: "project_key".to_owned(),
                value: self.project_key.clone(),
                reason: "must not be empty".to_owned(),
            });
        }
        if self.host_name.trim().is_empty() {
            return Err(SearchError::InvalidConfig {
                field: "host_name".to_owned(),
                value: self.host_name.clone(),
                reason: "must not be empty".to_owned(),
            });
        }
        if self.instance_count == 0 {
            return Err(SearchError::InvalidConfig {
                field: "instance_count".to_owned(),
                value: "0".to_owned(),
                reason: "must be > 0".to_owned(),
            });
        }
        Ok(())
    }
}

/// Deterministic simulator configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelemetrySimulatorConfig {
    /// Seed controlling deterministic pseudo-random generation.
    pub seed: u64,
    /// Start timestamp in unix milliseconds.
    pub start_ms: u64,
    /// Tick spacing in milliseconds.
    pub tick_interval_ms: u64,
    /// Number of ticks to generate.
    pub ticks: usize,
    /// Project lanes included in this simulation.
    pub projects: Vec<SimulatedProject>,
}

impl Default for TelemetrySimulatorConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            start_ms: 1_734_503_200_000,
            tick_interval_ms: 1_000,
            ticks: 24,
            projects: vec![
                SimulatedProject {
                    project_key: "cass".to_owned(),
                    host_name: "cass-devbox".to_owned(),
                    instance_count: 2,
                    workload: WorkloadProfile::Steady,
                },
                SimulatedProject {
                    project_key: "xf".to_owned(),
                    host_name: "xf-worker".to_owned(),
                    instance_count: 2,
                    workload: WorkloadProfile::Burst,
                },
                SimulatedProject {
                    project_key: "mcp_agent_mail_rust".to_owned(),
                    host_name: "mail-runner".to_owned(),
                    instance_count: 1,
                    workload: WorkloadProfile::EmbeddingWave,
                },
                SimulatedProject {
                    project_key: "frankenterm".to_owned(),
                    host_name: "terminal-host".to_owned(),
                    instance_count: 1,
                    workload: WorkloadProfile::Restarting,
                },
            ],
        }
    }
}

impl TelemetrySimulatorConfig {
    fn validate(&self) -> SearchResult<()> {
        if self.tick_interval_ms == 0 {
            return Err(SearchError::InvalidConfig {
                field: "tick_interval_ms".to_owned(),
                value: "0".to_owned(),
                reason: "must be > 0".to_owned(),
            });
        }
        if self.ticks == 0 {
            return Err(SearchError::InvalidConfig {
                field: "ticks".to_owned(),
                value: "0".to_owned(),
                reason: "must be > 0".to_owned(),
            });
        }
        if self.projects.is_empty() {
            return Err(SearchError::InvalidConfig {
                field: "projects".to_owned(),
                value: "[]".to_owned(),
                reason: "must include at least one project".to_owned(),
            });
        }
        for project in &self.projects {
            project.validate()?;
        }
        Ok(())
    }
}

/// Search event plus simulator-only diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SimulatedSearchEvent {
    /// Persistable record consumed by `OpsStorage`.
    pub record: SearchEventRecord,
    /// Query text used for class modeling.
    pub query: String,
    /// Classified query class from `frankensearch-core`.
    pub query_class: QueryClass,
    /// Search phase label emitted by simulated flow.
    pub phase_label: String,
    /// Optional skip reason metadata for degraded scenarios.
    pub skip_reason: Option<SkipReason>,
    /// Optional failure kind classification.
    pub failure_kind: Option<String>,
}

/// One generated simulation tick.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimulationBatch {
    /// Zero-based tick index.
    pub tick_index: usize,
    /// Tick timestamp in unix milliseconds.
    pub now_ms: u64,
    /// Discovery-state view at this tick.
    pub discovered_instances: Vec<DiscoveredInstance>,
    /// Search telemetry events generated for this tick.
    pub search_events: Vec<SimulatedSearchEvent>,
    /// Resource samples generated for this tick.
    pub resource_samples: Vec<ResourceSampleRecord>,
}

/// Full deterministic simulation output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimulationRun {
    /// Seed used to generate this run.
    pub seed: u64,
    /// Config snapshot used to generate this run.
    pub config: TelemetrySimulatorConfig,
    /// Generated tick batches.
    pub batches: Vec<SimulationBatch>,
}

impl SimulationRun {
    /// Total generated search events.
    #[must_use]
    pub fn total_search_events(&self) -> usize {
        self.batches
            .iter()
            .map(|batch| batch.search_events.len())
            .sum()
    }

    /// Total generated resource samples.
    #[must_use]
    pub fn total_resource_samples(&self) -> usize {
        self.batches
            .iter()
            .map(|batch| batch.resource_samples.len())
            .sum()
    }

    /// Unique `(project, instance)` pairs covered by this run.
    #[must_use]
    pub fn instance_pairs(&self) -> BTreeSet<(String, String)> {
        let mut pairs = BTreeSet::new();
        for batch in &self.batches {
            for sample in &batch.resource_samples {
                pairs.insert((sample.project_key.clone(), sample.instance_id.clone()));
            }
        }
        pairs
    }

    /// Deterministic signature for reproducibility assertions.
    #[must_use]
    pub fn signature(&self) -> u64 {
        const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

        let mut hash = OFFSET;
        for batch in &self.batches {
            hash = fnv_hash_u64(hash, usize_to_u64(batch.tick_index));
            hash = fnv_hash_u64(hash, batch.now_ms);
            for event in &batch.search_events {
                hash = fnv_hash_str(hash, &event.record.event_id);
                hash = fnv_hash_str(hash, &event.record.project_key);
                hash = fnv_hash_str(hash, &event.record.instance_id);
                hash = fnv_hash_str(hash, event.query_class.to_string().as_str());
                hash = fnv_hash_u64(hash, event.record.latency_us);
                hash = fnv_hash_str(hash, &event.phase_label);
                if let Some(reason) = &event.skip_reason {
                    hash = fnv_hash_str(hash, &format!("{reason:?}"));
                }
                if let Some(kind) = &event.failure_kind {
                    hash = fnv_hash_str(hash, kind);
                }
            }
            for sample in &batch.resource_samples {
                hash = fnv_hash_str(hash, &sample.project_key);
                hash = fnv_hash_str(hash, &sample.instance_id);
                hash = fnv_hash_u64(hash, u64::try_from(sample.ts_ms).unwrap_or_default());
                hash = fnv_hash_u64(hash, sample.rss_bytes.unwrap_or_default());
            }
        }
        hash
    }
}

/// Report produced by the e2e simulation entrypoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct E2eSimulationReport {
    /// Signature of the simulation run.
    pub run_signature: u64,
    /// Number of search events ingested.
    pub events_ingested: u64,
    /// Number of resource samples upserted.
    pub resource_samples_upserted: u64,
    /// Number of summary rows refreshed across ticks.
    pub summaries_materialized: usize,
    /// Materialization counters from the final pass.
    pub final_slo_result: SloMaterializationResult,
    /// Current count of open anomalies across the fleet scope.
    pub open_anomalies: usize,
    /// Final ingestion metrics snapshot from storage.
    pub ingestion_metrics: OpsIngestionMetricsSnapshot,
}

/// Report produced by the performance simulation entrypoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerfSimulationReport {
    /// Signature of the simulation run.
    pub run_signature: u64,
    /// Number of search events ingested.
    pub events_ingested: u64,
    /// Simulated wall-clock duration represented by the run.
    pub simulated_duration_ms: u64,
    /// Effective events/second from deterministic simulated time.
    pub events_per_second: f64,
    /// P95 latency across generated search events.
    pub p95_event_latency_us: u64,
    /// Mean write latency per batch as observed by `OpsStorage`.
    pub avg_write_latency_us: f64,
    /// Number of backpressured batches seen while ingesting.
    pub backpressured_batches: u64,
}

/// Deterministic telemetry simulator orchestrator.
#[derive(Debug, Clone)]
pub struct TelemetrySimulator {
    config: TelemetrySimulatorConfig,
}

impl TelemetrySimulator {
    /// Construct a simulator from config.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if configuration bounds are invalid.
    pub fn new(config: TelemetrySimulatorConfig) -> SearchResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Config snapshot.
    #[must_use]
    pub const fn config(&self) -> &TelemetrySimulatorConfig {
        &self.config
    }

    /// Generate deterministic telemetry batches for the configured fleet.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if config validation fails.
    #[allow(clippy::too_many_lines)]
    pub fn generate(&self) -> SearchResult<SimulationRun> {
        self.config.validate()?;
        let mut rng = DeterministicRng::new(self.config.seed);
        let instance_specs = expand_instance_specs(&self.config);

        let mut batches = Vec::with_capacity(self.config.ticks);
        for tick_index in 0..self.config.ticks {
            let now_ms = self.config.start_ms.saturating_add(
                usize_to_u64(tick_index).saturating_mul(self.config.tick_interval_ms),
            );

            let mut discovered_instances = Vec::with_capacity(instance_specs.len());
            let mut search_events = Vec::new();
            let mut resource_samples = Vec::with_capacity(instance_specs.len());

            for spec in &instance_specs {
                let restart_tick = spec.workload == WorkloadProfile::Restarting
                    && tick_index > 0
                    && tick_index % 5 == 0;
                let stale_tick = spec.workload == WorkloadProfile::Restarting
                    && tick_index > 0
                    && tick_index % 7 == 0;

                let pid = spec
                    .base_pid
                    .saturating_add(u32::try_from(tick_index / 5).unwrap_or(u32::MAX));

                let instance_status = if stale_tick {
                    DiscoveryStatus::Stale
                } else {
                    DiscoveryStatus::Active
                };
                let last_seen_ms = if stale_tick {
                    now_ms.saturating_sub(self.config.tick_interval_ms.saturating_mul(2))
                } else {
                    now_ms
                };

                discovered_instances.push(DiscoveredInstance {
                    instance_id: spec.instance_id.clone(),
                    project_key_hint: Some(spec.project_key.clone()),
                    host_name: Some(spec.host_name.clone()),
                    pid: Some(pid),
                    version: Some("0.1.0".to_owned()),
                    first_seen_ms: self.config.start_ms,
                    last_seen_ms,
                    status: instance_status,
                    sources: vec![
                        DiscoverySignalKind::Process,
                        DiscoverySignalKind::Heartbeat,
                        DiscoverySignalKind::ControlEndpoint,
                    ],
                    identity_keys: vec![
                        format!("instance:{}:{}", spec.project_key, spec.instance_index),
                        format!("hostpid:{}:{pid}", spec.host_name),
                    ],
                });

                let target_events =
                    event_count_for_tick(spec.workload, tick_index, restart_tick, &mut rng);
                let mut emitted_events = 0_u64;
                for event_index in 0..target_events {
                    let query = query_for_event(
                        spec.workload,
                        tick_index,
                        event_index,
                        &spec.instance_id,
                        &mut rng,
                    );
                    let query_class = QueryClass::classify(&query);
                    let phase = synthesize_phase(
                        spec.workload,
                        query_class,
                        restart_tick,
                        tick_index,
                        event_index,
                        &spec.instance_id,
                        &mut rng,
                    );
                    let skip_reason = derive_skip_reason(
                        spec.workload,
                        query_class,
                        tick_index,
                        restart_tick,
                        &mut rng,
                    );

                    let phase_label = phase_label(&phase).to_owned();
                    let failure_kind = failure_kind_for_phase(&phase).map(str::to_owned);
                    let (record_phase, latency_us, result_count) = phase_to_record_fields(&phase);

                    let correlation_idx = event_index / 2;
                    let event_id = format!(
                        "evt:{}:{}:{tick_index}:{event_index}:{phase_label}",
                        spec.project_key, spec.instance_index
                    );
                    let correlation_id =
                        format!("corr:{}:{tick_index}:{correlation_idx}", spec.instance_id);

                    search_events.push(SimulatedSearchEvent {
                        record: SearchEventRecord {
                            event_id,
                            project_key: spec.project_key.clone(),
                            instance_id: spec.instance_id.clone(),
                            correlation_id,
                            query_hash: Some(format!(
                                "{:016x}",
                                stable_u64_hash(
                                    format!("{query}:{tick_index}:{event_index}").as_str()
                                )
                            )),
                            query_class: Some(query_class.to_string()),
                            phase: record_phase,
                            latency_us,
                            result_count,
                            memory_bytes: Some(memory_bytes_for_event(
                                spec.workload,
                                query_class,
                                &mut rng,
                            )),
                            ts_ms: u64_to_i64(now_ms)?,
                        },
                        query,
                        query_class,
                        phase_label,
                        skip_reason,
                        failure_kind,
                    });
                    emitted_events = emitted_events.saturating_add(1);
                }

                resource_samples.push(ResourceSampleRecord {
                    project_key: spec.project_key.clone(),
                    instance_id: spec.instance_id.clone(),
                    cpu_pct: Some(cpu_pct_for_tick(
                        spec.workload,
                        emitted_events,
                        tick_index,
                        &mut rng,
                    )),
                    rss_bytes: Some(rss_for_tick(spec.workload, emitted_events, &mut rng)),
                    io_read_bytes: Some(io_read_for_tick(spec.workload, emitted_events, &mut rng)),
                    io_write_bytes: Some(io_write_for_tick(
                        spec.workload,
                        emitted_events,
                        &mut rng,
                    )),
                    queue_depth: Some(queue_depth_for_tick(
                        spec.workload,
                        emitted_events,
                        tick_index,
                        &mut rng,
                    )),
                    ts_ms: u64_to_i64(now_ms)?,
                });
            }

            discovered_instances.sort_by(|left, right| left.instance_id.cmp(&right.instance_id));
            search_events.sort_by(|left, right| left.record.event_id.cmp(&right.record.event_id));
            resource_samples.sort_by(|left, right| left.instance_id.cmp(&right.instance_id));

            batches.push(SimulationBatch {
                tick_index,
                now_ms,
                discovered_instances,
                search_events,
                resource_samples,
            });
        }

        Ok(SimulationRun {
            seed: self.config.seed,
            config: self.config.clone(),
            batches,
        })
    }

    /// Run the deterministic e2e-oriented simulator entrypoint against storage.
    ///
    /// This entrypoint ingests generated telemetry, refreshes search summaries,
    /// and materializes SLO/anomaly rows at each tick.
    ///
    /// # Errors
    ///
    /// Returns a `SearchError` when generation, ingest, or storage queries fail.
    pub fn run_e2e_entrypoint(
        &self,
        storage: &OpsStorage,
        backpressure_threshold: usize,
    ) -> SearchResult<E2eSimulationReport> {
        let run = self.generate()?;
        let replay = replay_run_into_storage(storage, &run, backpressure_threshold)?;
        let open_anomalies = storage
            .query_open_anomalies_for_scope(SloScope::Fleet, "__fleet__", 256)?
            .len();
        Ok(E2eSimulationReport {
            run_signature: run.signature(),
            events_ingested: replay.events_ingested,
            resource_samples_upserted: replay.resource_samples_upserted,
            summaries_materialized: replay.summaries_materialized,
            final_slo_result: replay.final_slo_result,
            open_anomalies,
            ingestion_metrics: storage.ingestion_metrics(),
        })
    }

    /// Run the deterministic performance-oriented simulator entrypoint.
    ///
    /// This entrypoint measures deterministic throughput over simulated time
    /// and reports ingestion statistics derived from `OpsStorage`.
    ///
    /// # Errors
    ///
    /// Returns a `SearchError` when generation or storage operations fail.
    pub fn run_performance_entrypoint(
        &self,
        storage: &OpsStorage,
        backpressure_threshold: usize,
    ) -> SearchResult<PerfSimulationReport> {
        let run = self.generate()?;
        let replay = replay_run_into_storage(storage, &run, backpressure_threshold)?;
        let simulated_duration_ms = self
            .config
            .tick_interval_ms
            .saturating_mul(usize_to_u64(self.config.ticks));
        let events_per_second = ratio_per_second(replay.events_ingested, simulated_duration_ms);
        let metrics = storage.ingestion_metrics();
        let avg_write_latency_us =
            average_u64(metrics.total_write_latency_us, metrics.total_batches);
        let p95_event_latency_us = p95_from_run(&run);

        Ok(PerfSimulationReport {
            run_signature: run.signature(),
            events_ingested: replay.events_ingested,
            simulated_duration_ms,
            events_per_second,
            p95_event_latency_us,
            avg_write_latency_us,
            backpressured_batches: metrics.total_backpressured_batches,
        })
    }
}

#[derive(Debug, Clone)]
struct InstanceSpec {
    project_key: String,
    host_name: String,
    workload: WorkloadProfile,
    instance_index: usize,
    instance_id: String,
    base_pid: u32,
}

fn expand_instance_specs(config: &TelemetrySimulatorConfig) -> Vec<InstanceSpec> {
    let mut specs = Vec::new();
    for project in &config.projects {
        for idx in 0..project.instance_count {
            let instance_index = idx.saturating_add(1);
            specs.push(InstanceSpec {
                project_key: project.project_key.clone(),
                host_name: format!("{}-{instance_index}", project.host_name),
                workload: project.workload,
                instance_index,
                instance_id: format!("{}-{instance_index:02}", project.project_key),
                base_pid: stable_pid(&project.project_key, instance_index),
            });
        }
    }
    specs.sort_by(|left, right| left.instance_id.cmp(&right.instance_id));
    specs
}

fn stable_pid(project_key: &str, instance_index: usize) -> u32 {
    let seed = stable_u64_hash(&format!("{project_key}:{instance_index}"));
    let offset = u32::try_from(seed % 30_000).unwrap_or(0);
    10_000_u32.saturating_add(offset)
}

fn query_for_event(
    workload: WorkloadProfile,
    tick_index: usize,
    event_index: u64,
    instance_id: &str,
    rng: &mut DeterministicRng,
) -> String {
    let pool: &[&str] = match workload {
        WorkloadProfile::Steady => &[
            "rust ownership boundaries",
            "hybrid search ranking",
            "tantivy query parser",
            "index compaction schedule",
        ],
        WorkloadProfile::Burst => &[
            "bd-2yu.7.2",
            "src/screens/live_stream.rs",
            "latency p95 spike",
            "vector queue depth",
        ],
        WorkloadProfile::EmbeddingWave => &[
            "how does two tier refinement work",
            "why are quality embeds lagging behind",
            "explain memory pressure from embedding backlog",
            "what changed in model loading pipeline",
        ],
        WorkloadProfile::Restarting => &[
            "service restart detected",
            "why did this instance go stale",
            "recover from control plane timeout",
            "",
        ],
    };
    let idx = usize::try_from(rng.next_bounded(usize_to_u64(pool.len()))).unwrap_or(0);
    let base = pool.get(idx).copied().unwrap_or("query");
    format!("{base} {instance_id} t{tick_index} e{event_index}")
}

fn event_count_for_tick(
    workload: WorkloadProfile,
    tick_index: usize,
    restart_tick: bool,
    rng: &mut DeterministicRng,
) -> u64 {
    let baseline = match workload {
        WorkloadProfile::Steady => 3_u64,
        WorkloadProfile::Burst => 5_u64,
        WorkloadProfile::EmbeddingWave => 4_u64,
        WorkloadProfile::Restarting => 2_u64,
    };
    let burst_bonus = if workload == WorkloadProfile::Burst && tick_index % 4 == 0 {
        8
    } else {
        0
    };
    let wave_bonus = if workload == WorkloadProfile::EmbeddingWave && tick_index % 6 < 3 {
        5
    } else {
        0
    };
    let restart_penalty = if restart_tick { 2 } else { 0 };
    let jitter = rng.next_bounded(3);
    baseline
        .saturating_add(burst_bonus)
        .saturating_add(wave_bonus)
        .saturating_add(jitter)
        .saturating_sub(restart_penalty)
        .max(1)
}

fn synthesize_phase(
    workload: WorkloadProfile,
    query_class: QueryClass,
    restart_tick: bool,
    tick_index: usize,
    event_index: u64,
    instance_id: &str,
    rng: &mut DeterministicRng,
) -> SearchPhase {
    let roll = rng.next_bounded(100);
    let fail_cutoff = match workload {
        WorkloadProfile::Steady => 8_u64,
        WorkloadProfile::Burst => 20_u64,
        WorkloadProfile::EmbeddingWave => 28_u64,
        WorkloadProfile::Restarting => {
            if restart_tick {
                60
            } else {
                22
            }
        }
    };
    let initial_cutoff = match workload {
        WorkloadProfile::Steady => 30_u64,
        WorkloadProfile::Burst => 55_u64,
        WorkloadProfile::EmbeddingWave => 45_u64,
        WorkloadProfile::Restarting => 70_u64,
    };

    let result_count = result_count_for_query_class(query_class);
    let latency_base_us = match workload {
        WorkloadProfile::Steady => 900_u64,
        WorkloadProfile::Burst => 4_500_u64,
        WorkloadProfile::EmbeddingWave => 8_000_u64,
        WorkloadProfile::Restarting => 12_000_u64,
    };

    if roll < fail_cutoff {
        let timeout_ms = 120_u64.saturating_add(rng.next_bounded(120));
        return SearchPhase::RefinementFailed {
            initial_results: build_results(
                instance_id,
                tick_index,
                event_index,
                result_count,
                ScoreSource::SemanticFast,
            ),
            error: SearchError::SearchTimeout {
                elapsed_ms: timeout_ms,
                budget_ms: 100,
            },
            latency: Duration::from_micros(latency_base_us.saturating_mul(30)),
        };
    }

    if query_class == QueryClass::Empty || roll < initial_cutoff {
        return SearchPhase::Initial {
            results: build_results(
                instance_id,
                tick_index,
                event_index,
                result_count,
                ScoreSource::SemanticFast,
            ),
            latency: Duration::from_micros(latency_base_us.saturating_add(rng.next_bounded(1_000))),
            metrics: PhaseMetrics {
                embedder_id: "potion-128m".to_owned(),
                vectors_searched: 1_024,
                lexical_candidates: 32,
                fused_count: result_count,
            },
        };
    }

    SearchPhase::Refined {
        results: build_results(
            instance_id,
            tick_index,
            event_index,
            result_count.saturating_add(1),
            ScoreSource::Hybrid,
        ),
        latency: Duration::from_micros(
            latency_base_us
                .saturating_mul(16)
                .saturating_add(rng.next_bounded(12_000)),
        ),
        metrics: PhaseMetrics {
            embedder_id: "minilm-l6-v2".to_owned(),
            vectors_searched: 4_096,
            lexical_candidates: 96,
            fused_count: result_count.saturating_add(1),
        },
        rank_changes: RankChanges {
            promoted: 1,
            demoted: 1,
            stable: result_count.saturating_sub(1),
        },
    }
}

fn build_results(
    instance_id: &str,
    tick_index: usize,
    event_index: u64,
    count: usize,
    source: ScoreSource,
) -> Vec<ScoredResult> {
    let mut results = Vec::with_capacity(count);
    for rank in 0..count {
        let rank_u32 = u32::try_from(rank).unwrap_or(u32::MAX);
        let rank_score = 100_u32.saturating_sub(rank_u32.saturating_mul(7));
        results.push(ScoredResult {
            doc_id: format!("{instance_id}:doc:{tick_index}:{event_index}:{rank}"),
            score: f32::from(u16::try_from(rank_score).unwrap_or(u16::MAX)) / 100.0,
            source,
            fast_score: Some(0.8),
            quality_score: Some(0.9),
            lexical_score: Some(4.2),
            rerank_score: None,
            metadata: None,
        });
    }
    results
}

const fn result_count_for_query_class(class: QueryClass) -> usize {
    match class {
        QueryClass::Empty => 0,
        QueryClass::Identifier => 2,
        QueryClass::ShortKeyword => 3,
        QueryClass::NaturalLanguage => 4,
    }
}

const fn phase_label(phase: &SearchPhase) -> &'static str {
    match phase {
        SearchPhase::Initial { .. } => "initial",
        SearchPhase::Refined { .. } => "refined",
        SearchPhase::RefinementFailed { .. } => "failed",
    }
}

const fn failure_kind_for_phase(phase: &SearchPhase) -> Option<&'static str> {
    match phase {
        SearchPhase::RefinementFailed { error, .. } => Some(search_error_kind(error)),
        SearchPhase::Initial { .. } | SearchPhase::Refined { .. } => None,
    }
}

const fn search_error_kind(error: &SearchError) -> &'static str {
    match error {
        SearchError::EmbedderUnavailable { .. } => "embedder_unavailable",
        SearchError::EmbeddingFailed { .. } => "embedding_failed",
        SearchError::ModelNotFound { .. } => "model_not_found",
        SearchError::ModelLoadFailed { .. } => "model_load_failed",
        SearchError::IndexCorrupted { .. } => "index_corrupted",
        SearchError::IndexVersionMismatch { .. } => "index_version_mismatch",
        SearchError::DimensionMismatch { .. } => "dimension_mismatch",
        SearchError::IndexNotFound { .. } => "index_not_found",
        SearchError::QueryParseError { .. } => "query_parse_error",
        SearchError::SearchTimeout { .. } => "search_timeout",
        SearchError::FederatedInsufficientResponses { .. } => "federated_insufficient_responses",
        SearchError::RerankerUnavailable { .. } => "reranker_unavailable",
        SearchError::RerankFailed { .. } => "rerank_failed",
        SearchError::Io(_) => "io",
        SearchError::InvalidConfig { .. } => "invalid_config",
        SearchError::HashMismatch { .. } => "hash_mismatch",
        SearchError::Cancelled { .. } => "cancelled",
        SearchError::QueueFull { .. } => "queue_full",
        SearchError::SubsystemError { .. } => "subsystem_error",
        SearchError::DurabilityDisabled => "durability_disabled",
    }
}

fn phase_to_record_fields(phase: &SearchPhase) -> (SearchEventPhase, u64, Option<u64>) {
    match phase {
        SearchPhase::Initial {
            latency, results, ..
        } => (
            SearchEventPhase::Initial,
            duration_to_us(*latency),
            Some(usize_to_u64(results.len())),
        ),
        SearchPhase::Refined {
            latency, results, ..
        } => (
            SearchEventPhase::Refined,
            duration_to_us(*latency),
            Some(usize_to_u64(results.len())),
        ),
        SearchPhase::RefinementFailed { latency, .. } => {
            (SearchEventPhase::Failed, duration_to_us(*latency), None)
        }
    }
}

fn derive_skip_reason(
    workload: WorkloadProfile,
    query_class: QueryClass,
    tick_index: usize,
    restart_tick: bool,
    rng: &mut DeterministicRng,
) -> Option<SkipReason> {
    if query_class == QueryClass::Empty {
        return Some(SkipReason::EmptyCommit);
    }
    if restart_tick && workload == WorkloadProfile::Restarting && rng.next_bounded(4) == 0 {
        let expected = usize_to_u64(tick_index);
        return Some(SkipReason::OutOfOrder {
            expected,
            got: expected.saturating_add(1),
        });
    }
    if workload == WorkloadProfile::EmbeddingWave && rng.next_bounded(10) == 0 {
        return Some(SkipReason::AlreadyApplied);
    }
    None
}

fn memory_bytes_for_event(
    workload: WorkloadProfile,
    query_class: QueryClass,
    rng: &mut DeterministicRng,
) -> u64 {
    let base = match workload {
        WorkloadProfile::Steady => 32_u64 * 1024 * 1024,
        WorkloadProfile::Burst => 96_u64 * 1024 * 1024,
        WorkloadProfile::EmbeddingWave => 180_u64 * 1024 * 1024,
        WorkloadProfile::Restarting => 64_u64 * 1024 * 1024,
    };
    let class_bonus = match query_class {
        QueryClass::Empty => 0_u64,
        QueryClass::Identifier => 4_u64 * 1024 * 1024,
        QueryClass::ShortKeyword => 8_u64 * 1024 * 1024,
        QueryClass::NaturalLanguage => 16_u64 * 1024 * 1024,
    };
    let jitter = rng.next_bounded(8_u64 * 1024 * 1024);
    base.saturating_add(class_bonus).saturating_add(jitter)
}

fn cpu_pct_for_tick(
    workload: WorkloadProfile,
    events: u64,
    tick_index: usize,
    rng: &mut DeterministicRng,
) -> f64 {
    let baseline = match workload {
        WorkloadProfile::Steady => 18.0,
        WorkloadProfile::Burst => 52.0,
        WorkloadProfile::EmbeddingWave => 64.0,
        WorkloadProfile::Restarting => 27.0,
    };
    let events_component = u32_to_f64(u32::try_from(events).unwrap_or(u32::MAX)) * 1.4;
    let wave_component = if workload == WorkloadProfile::EmbeddingWave {
        let phase = tick_index % 6;
        u32_to_f64(u32::try_from(phase).unwrap_or_default()) * 2.0
    } else {
        0.0
    };
    let jitter = u32_to_f64(u32::try_from(rng.next_bounded(300)).unwrap_or_default()) / 20.0;
    let raw = baseline + events_component + wave_component + jitter;
    raw.clamp(0.0, 100.0)
}

fn rss_for_tick(workload: WorkloadProfile, events: u64, rng: &mut DeterministicRng) -> u64 {
    let baseline = match workload {
        WorkloadProfile::Steady => 192_u64 * 1024 * 1024,
        WorkloadProfile::Burst => 256_u64 * 1024 * 1024,
        WorkloadProfile::EmbeddingWave => 512_u64 * 1024 * 1024,
        WorkloadProfile::Restarting => 160_u64 * 1024 * 1024,
    };
    baseline
        .saturating_add(events.saturating_mul(2 * 1024 * 1024))
        .saturating_add(rng.next_bounded(24_u64 * 1024 * 1024))
}

fn io_read_for_tick(workload: WorkloadProfile, events: u64, rng: &mut DeterministicRng) -> u64 {
    let base = match workload {
        WorkloadProfile::Steady => 512_u64 * 1024,
        WorkloadProfile::Burst => 3_u64 * 1024 * 1024,
        WorkloadProfile::EmbeddingWave => 5_u64 * 1024 * 1024,
        WorkloadProfile::Restarting => 1024_u64 * 1024,
    };
    base.saturating_add(events.saturating_mul(180 * 1024))
        .saturating_add(rng.next_bounded(512_u64 * 1024))
}

fn io_write_for_tick(workload: WorkloadProfile, events: u64, rng: &mut DeterministicRng) -> u64 {
    let base = match workload {
        WorkloadProfile::Steady => 256_u64 * 1024,
        WorkloadProfile::Burst => 2_u64 * 1024 * 1024,
        WorkloadProfile::EmbeddingWave => 4_u64 * 1024 * 1024,
        WorkloadProfile::Restarting => 768_u64 * 1024,
    };
    base.saturating_add(events.saturating_mul(140 * 1024))
        .saturating_add(rng.next_bounded(512_u64 * 1024))
}

fn queue_depth_for_tick(
    workload: WorkloadProfile,
    events: u64,
    tick_index: usize,
    rng: &mut DeterministicRng,
) -> u64 {
    let base = match workload {
        WorkloadProfile::Steady => 2_u64,
        WorkloadProfile::Burst => 20_u64,
        WorkloadProfile::EmbeddingWave => 60_u64,
        WorkloadProfile::Restarting => 8_u64,
    };
    let wave = if workload == WorkloadProfile::EmbeddingWave {
        usize_to_u64(tick_index % 5).saturating_mul(12)
    } else {
        0
    };
    base.saturating_add(events / 2)
        .saturating_add(wave)
        .saturating_add(rng.next_bounded(8))
}

#[derive(Debug, Clone, Copy, Default)]
struct ReplayStats {
    events_ingested: u64,
    resource_samples_upserted: u64,
    summaries_materialized: usize,
    final_slo_result: SloMaterializationResult,
}

fn replay_run_into_storage(
    storage: &OpsStorage,
    run: &SimulationRun,
    backpressure_threshold: usize,
) -> SearchResult<ReplayStats> {
    let mut stats = ReplayStats::default();

    for batch in &run.batches {
        let records: Vec<SearchEventRecord> = batch
            .search_events
            .iter()
            .map(|event| event.record.clone())
            .collect();
        let ingest = storage.ingest_search_events_batch(&records, backpressure_threshold)?;
        stats.events_ingested = stats
            .events_ingested
            .saturating_add(usize_to_u64(ingest.inserted));

        for sample in &batch.resource_samples {
            storage.upsert_resource_sample(sample)?;
            stats.resource_samples_upserted = stats.resource_samples_upserted.saturating_add(1);
        }

        let mut pairs: BTreeSet<(String, String)> = BTreeSet::new();
        for sample in &batch.resource_samples {
            pairs.insert((sample.project_key.clone(), sample.instance_id.clone()));
        }
        let now_ms = u64_to_i64(batch.now_ms)?;
        for (project_key, instance_id) in &pairs {
            let snapshots =
                storage.refresh_search_summaries_for_instance(project_key, instance_id, now_ms)?;
            stats.summaries_materialized =
                stats.summaries_materialized.saturating_add(snapshots.len());
        }

        stats.final_slo_result = storage
            .materialize_slo_rollups_and_anomalies(now_ms, SloMaterializationConfig::default())?;
    }

    Ok(stats)
}

fn p95_from_run(run: &SimulationRun) -> u64 {
    let mut latencies: Vec<u64> = run
        .batches
        .iter()
        .flat_map(|batch| {
            batch
                .search_events
                .iter()
                .map(|event| event.record.latency_us)
        })
        .collect();
    if latencies.is_empty() {
        return 0;
    }
    latencies.sort_unstable();
    let len = latencies.len();
    let idx = ((len.saturating_mul(95)).saturating_sub(1)) / 100;
    latencies.get(idx).copied().unwrap_or_default()
}

#[derive(Debug, Clone, Copy)]
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    const MULTIPLIER: u64 = 6_364_136_223_846_793_005;
    const INCREMENT: u64 = 1_442_695_040_888_963_407;

    const fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        };
        Self { state }
    }

    const fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(Self::MULTIPLIER)
            .wrapping_add(Self::INCREMENT);
        self.state
    }

    const fn next_bounded(&mut self, upper_exclusive: u64) -> u64 {
        if upper_exclusive == 0 {
            return 0;
        }
        self.next_u64() % upper_exclusive
    }
}

fn stable_u64_hash(seed: &str) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    let mut hash = OFFSET;
    for byte in seed.bytes() {
        hash = fnv_hash_u64(hash, u64::from(byte));
    }
    hash
}

fn fnv_hash_str(mut hash: u64, value: &str) -> u64 {
    for byte in value.bytes() {
        hash = fnv_hash_u64(hash, u64::from(byte));
    }
    hash
}

fn fnv_hash_u64(mut hash: u64, value: u64) -> u64 {
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    for byte in value.to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn u64_to_i64(value: u64) -> SearchResult<i64> {
    i64::try_from(value).map_err(|_| SearchError::InvalidConfig {
        field: "timestamp".to_owned(),
        value: value.to_string(),
        reason: "must fit into i64".to_owned(),
    })
}

fn duration_to_us(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

fn ratio_per_second(events: u64, duration_ms: u64) -> f64 {
    if duration_ms == 0 {
        return 0.0;
    }
    let events_u32 = u32::try_from(events).unwrap_or(u32::MAX);
    let duration_u32 = u32::try_from(duration_ms).unwrap_or(u32::MAX);
    (u32_to_f64(events_u32) * 1000.0) / u32_to_f64(duration_u32)
}

fn average_u64(total: u64, count: u64) -> f64 {
    if count == 0 {
        return 0.0;
    }
    let total_u32 = u32::try_from(total).unwrap_or(u32::MAX);
    let count_u32 = u32::try_from(count).unwrap_or(u32::MAX);
    u32_to_f64(total_u32) / u32_to_f64(count_u32)
}

fn u32_to_f64(value: u32) -> f64 {
    f64::from(value)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::storage::SummaryWindow;
    use crate::{OpsStorage, SloHealth};

    #[test]
    fn simulator_emulates_multiple_projects_and_profiles() {
        let config = TelemetrySimulatorConfig {
            seed: 7,
            start_ms: 1_734_503_200_000,
            tick_interval_ms: 1_000,
            ticks: 8,
            projects: vec![
                SimulatedProject {
                    project_key: "cass".to_owned(),
                    host_name: "cass-host".to_owned(),
                    instance_count: 2,
                    workload: WorkloadProfile::Steady,
                },
                SimulatedProject {
                    project_key: "xf".to_owned(),
                    host_name: "xf-host".to_owned(),
                    instance_count: 2,
                    workload: WorkloadProfile::Burst,
                },
                SimulatedProject {
                    project_key: "mail".to_owned(),
                    host_name: "mail-host".to_owned(),
                    instance_count: 1,
                    workload: WorkloadProfile::Restarting,
                },
            ],
        };
        let simulator = TelemetrySimulator::new(config).expect("config should be valid");
        let run = simulator.generate().expect("generation should succeed");

        assert_eq!(run.batches.len(), 8);
        assert_eq!(run.instance_pairs().len(), 5);
        assert!(run.total_search_events() > 50);
        assert_eq!(run.total_resource_samples(), 8 * 5);

        let phase_counts: BTreeMap<String, usize> = run
            .batches
            .iter()
            .flat_map(|batch| batch.search_events.iter())
            .fold(BTreeMap::new(), |mut acc, event| {
                *acc.entry(event.phase_label.clone()).or_insert(0) += 1;
                acc
            });
        assert!(
            phase_counts.get("failed").copied().unwrap_or_default() > 0,
            "mixed workloads should include refinement failures"
        );
        assert!(
            phase_counts.get("refined").copied().unwrap_or_default() > 0,
            "mixed workloads should include refined events"
        );
    }

    #[test]
    fn simulator_seed_is_reproducible() {
        let base = TelemetrySimulatorConfig::default();
        let simulator_a =
            TelemetrySimulator::new(base.clone()).expect("default config should validate");
        let simulator_b =
            TelemetrySimulator::new(base.clone()).expect("default config should validate");
        let run_a = simulator_a.generate().expect("generation should succeed");
        let run_b = simulator_b.generate().expect("generation should succeed");
        assert_eq!(run_a.signature(), run_b.signature());
        assert_eq!(run_a.total_search_events(), run_b.total_search_events());

        let mut different_seed = base;
        different_seed.seed = 43;
        let run_c = TelemetrySimulator::new(different_seed)
            .expect("config should validate")
            .generate()
            .expect("generation should succeed");
        assert_ne!(run_a.signature(), run_c.signature());
    }

    #[test]
    fn simulator_e2e_entrypoint_materializes_storage_views() {
        let config = TelemetrySimulatorConfig {
            ticks: 6,
            ..TelemetrySimulatorConfig::default()
        };
        let simulator = TelemetrySimulator::new(config).expect("config should validate");
        let storage = OpsStorage::open_in_memory().expect("in-memory storage should open");

        let report = simulator
            .run_e2e_entrypoint(&storage, 8_192)
            .expect("e2e replay should succeed");
        assert!(report.events_ingested > 0);
        assert!(report.resource_samples_upserted > 0);
        assert!(report.summaries_materialized > 0);
        assert!(
            report.final_slo_result.rollups_upserted > 0,
            "materialization should emit rollups"
        );
        assert_eq!(
            report.ingestion_metrics.total_inserted, report.events_ingested,
            "all generated events should be inserted"
        );

        let one_minute = storage
            .latest_search_summary("cass", "cass-01", SummaryWindow::OneMinute)
            .expect("summary query should succeed");
        assert!(one_minute.is_some());

        let fleet_rollups = storage
            .query_slo_rollups_for_scope(SloScope::Fleet, "__fleet__", 16)
            .expect("fleet rollups should query");
        assert!(
            fleet_rollups
                .iter()
                .any(|row| row.health != SloHealth::NoData || row.total_requests > 0),
            "simulated traffic should materialize non-empty fleet context"
        );
    }

    #[test]
    fn simulator_performance_entrypoint_reports_deterministic_throughput() {
        let config = TelemetrySimulatorConfig {
            ticks: 10,
            projects: vec![
                SimulatedProject {
                    project_key: "xf".to_owned(),
                    host_name: "xf-perf".to_owned(),
                    instance_count: 3,
                    workload: WorkloadProfile::Burst,
                },
                SimulatedProject {
                    project_key: "mail".to_owned(),
                    host_name: "mail-perf".to_owned(),
                    instance_count: 2,
                    workload: WorkloadProfile::EmbeddingWave,
                },
            ],
            ..TelemetrySimulatorConfig::default()
        };
        let simulator = TelemetrySimulator::new(config).expect("config should validate");
        let storage = OpsStorage::open_in_memory().expect("in-memory storage should open");

        let report = simulator
            .run_performance_entrypoint(&storage, 16_384)
            .expect("perf replay should succeed");

        assert!(report.events_ingested > 0);
        assert!(report.events_per_second > 0.0);
        assert!(report.p95_event_latency_us > 0);
        assert_eq!(report.backpressured_batches, 0);
        assert!(report.avg_write_latency_us >= 0.0);
    }
}
