use std::collections::BTreeSet;

use frankensearch_core::{SearchError, SearchResult};
use frankensearch_ops::storage::SummaryWindow;
use frankensearch_ops::{
    OpsStorage, SimulatedProject, SimulationRun, SloMaterializationConfig,
    SloMaterializationResult, SloScope, TelemetrySimulator, TelemetrySimulatorConfig,
    WorkloadProfile,
};

fn pipeline_config(seed: u64) -> TelemetrySimulatorConfig {
    TelemetrySimulatorConfig {
        seed,
        start_ms: 1_734_503_200_000,
        tick_interval_ms: 1_000,
        ticks: 6,
        projects: vec![
            SimulatedProject {
                project_key: "cass".to_owned(),
                host_name: "cass-itest".to_owned(),
                instance_count: 2,
                workload: WorkloadProfile::Steady,
            },
            SimulatedProject {
                project_key: "xf".to_owned(),
                host_name: "xf-itest".to_owned(),
                instance_count: 2,
                workload: WorkloadProfile::Burst,
            },
            SimulatedProject {
                project_key: "mail".to_owned(),
                host_name: "mail-itest".to_owned(),
                instance_count: 1,
                workload: WorkloadProfile::EmbeddingWave,
            },
        ],
    }
}

fn apply_pipeline(
    storage: &OpsStorage,
    run: &SimulationRun,
    backpressure_threshold: usize,
) -> SearchResult<SloMaterializationResult> {
    let mut last_result = SloMaterializationResult::default();
    for batch in &run.batches {
        let records: Vec<_> = batch
            .search_events
            .iter()
            .map(|event| event.record.clone())
            .collect();
        storage.ingest_search_events_batch(&records, backpressure_threshold)?;

        for sample in &batch.resource_samples {
            storage.upsert_resource_sample(sample)?;
        }

        let now_ms = i64::try_from(batch.now_ms).map_err(|_| SearchError::InvalidConfig {
            field: "now_ms".to_owned(),
            value: batch.now_ms.to_string(),
            reason: "must fit into i64".to_owned(),
        })?;
        let mut pairs = BTreeSet::new();
        for sample in &batch.resource_samples {
            pairs.insert((sample.project_key.clone(), sample.instance_id.clone()));
        }
        for (project_key, instance_id) in &pairs {
            let _ =
                storage.refresh_search_summaries_for_instance(project_key, instance_id, now_ms)?;
        }

        last_result = storage
            .materialize_slo_rollups_and_anomalies(now_ms, SloMaterializationConfig::default())?;
    }
    Ok(last_result)
}

#[test]
fn pipeline_ingest_to_aggregation_materializes_expected_views() {
    let simulator = TelemetrySimulator::new(pipeline_config(21)).expect("config should validate");
    let run = simulator.generate().expect("generation should succeed");
    let storage = OpsStorage::open_in_memory().expect("storage should open");

    let materialization = apply_pipeline(&storage, &run, 8_192).expect("pipeline should succeed");
    assert!(
        materialization.rollups_upserted > 0,
        "pipeline should produce SLO rollups"
    );

    let metrics = storage.ingestion_metrics();
    assert_eq!(
        metrics.total_inserted,
        u64::try_from(run.total_search_events()).expect("event count should fit into u64")
    );

    for (project_key, instance_id) in run.instance_pairs() {
        let one_minute = storage
            .latest_search_summary(&project_key, &instance_id, SummaryWindow::OneMinute)
            .expect("summary query should succeed");
        assert!(
            one_minute.is_some(),
            "expected search summary for {project_key}/{instance_id}"
        );

        let trend = storage
            .query_resource_trend(
                &project_key,
                &instance_id,
                SummaryWindow::OneHour,
                i64::try_from(run.config.start_ms + 30_000).expect("timestamp should fit i64"),
                128,
            )
            .expect("resource trend query should succeed");
        assert!(
            !trend.is_empty(),
            "expected resource trend rows for {project_key}/{instance_id}"
        );
    }

    let fleet_rollups = storage
        .query_slo_rollups_for_scope(SloScope::Fleet, "__fleet__", 16)
        .expect("fleet rollup query should succeed");
    assert!(
        !fleet_rollups.is_empty(),
        "expected fleet rollups after materialization"
    );
}

#[test]
fn pipeline_replay_is_deterministic_for_same_seed() {
    let config = pipeline_config(55);
    let run_a = TelemetrySimulator::new(config.clone())
        .expect("config should validate")
        .generate()
        .expect("generation should succeed");
    let run_b = TelemetrySimulator::new(config)
        .expect("config should validate")
        .generate()
        .expect("generation should succeed");
    assert_eq!(run_a.signature(), run_b.signature());

    let storage_a = OpsStorage::open_in_memory().expect("storage should open");
    let storage_b = OpsStorage::open_in_memory().expect("storage should open");
    let _ = apply_pipeline(&storage_a, &run_a, 8_192).expect("pipeline A should succeed");
    let _ = apply_pipeline(&storage_b, &run_b, 8_192).expect("pipeline B should succeed");

    let metrics_a = storage_a.ingestion_metrics();
    let metrics_b = storage_b.ingestion_metrics();
    assert_eq!(metrics_a.total_batches, metrics_b.total_batches);
    assert_eq!(metrics_a.total_inserted, metrics_b.total_inserted);
    assert_eq!(metrics_a.total_deduplicated, metrics_b.total_deduplicated);
    assert_eq!(
        metrics_a.total_failed_records,
        metrics_b.total_failed_records
    );
    assert_eq!(
        metrics_a.total_backpressured_batches,
        metrics_b.total_backpressured_batches
    );
    assert_eq!(metrics_a.pending_events, metrics_b.pending_events);
    assert_eq!(
        metrics_a.high_watermark_pending_events,
        metrics_b.high_watermark_pending_events
    );

    let fleet_a = storage_a
        .query_slo_rollups_for_scope(SloScope::Fleet, "__fleet__", 8)
        .expect("fleet rollup query A should succeed");
    let fleet_b = storage_b
        .query_slo_rollups_for_scope(SloScope::Fleet, "__fleet__", 8)
        .expect("fleet rollup query B should succeed");

    let summary_a: Vec<_> = fleet_a
        .iter()
        .map(|row| {
            (
                row.window,
                row.window_start_ms,
                row.total_requests,
                row.reason_code.clone(),
            )
        })
        .collect();
    let summary_b: Vec<_> = fleet_b
        .iter()
        .map(|row| {
            (
                row.window,
                row.window_start_ms,
                row.total_requests,
                row.reason_code.clone(),
            )
        })
        .collect();
    assert_eq!(summary_a, summary_b);
}

#[test]
fn pipeline_recovers_after_backpressure_rejection() {
    let simulator = TelemetrySimulator::new(pipeline_config(99)).expect("config should validate");
    let run = simulator.generate().expect("generation should succeed");
    let storage = OpsStorage::open_in_memory().expect("storage should open");

    let first_batch = run.batches.first().expect("run should include first batch");
    let first_records: Vec<_> = first_batch
        .search_events
        .iter()
        .map(|event| event.record.clone())
        .collect();
    let err = storage
        .ingest_search_events_batch(&first_records, 1)
        .expect_err("threshold=1 should trigger backpressure for first batch");
    assert!(
        matches!(err, SearchError::QueueFull { .. }),
        "expected QueueFull, got {err:?}"
    );

    let _ = apply_pipeline(&storage, &run, 8_192).expect("pipeline should recover and succeed");

    let metrics = storage.ingestion_metrics();
    assert!(
        metrics.total_backpressured_batches >= 1,
        "expected backpressure counter to record initial rejection"
    );
    assert!(metrics.total_inserted > 0);
}
