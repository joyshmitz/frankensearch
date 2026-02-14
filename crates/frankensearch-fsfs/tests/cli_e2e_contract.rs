use frankensearch_core::{E2eOutcome, ExitStatus};
use frankensearch_fsfs::{
    CLI_E2E_SCHEMA_VERSION, CliE2eArtifactBundle, CliE2eRunConfig, CliE2eScenarioKind,
    build_default_cli_e2e_bundles, default_cli_e2e_scenarios, replay_command_for_scenario,
};

fn scenario_by_kind(kind: CliE2eScenarioKind) -> frankensearch_fsfs::CliE2eScenario {
    default_cli_e2e_scenarios()
        .into_iter()
        .find(|scenario| scenario.kind == kind)
        .expect("scenario should exist")
}

#[test]
fn scenario_cli_index_baseline() {
    let scenario = scenario_by_kind(CliE2eScenarioKind::Index);
    let bundle = CliE2eArtifactBundle::build(&CliE2eRunConfig::default(), &scenario, ExitStatus::Pass);
    bundle.validate().expect("bundle must validate");
    assert_eq!(bundle.schema_version, CLI_E2E_SCHEMA_VERSION);
    assert_eq!(bundle.scenario.kind, CliE2eScenarioKind::Index);
    assert_eq!(bundle.scenario.args.first().map(String::as_str), Some("index"));
}

#[test]
fn scenario_cli_search_stream() {
    let scenario = scenario_by_kind(CliE2eScenarioKind::Search);
    let bundle = CliE2eArtifactBundle::build(&CliE2eRunConfig::default(), &scenario, ExitStatus::Pass);
    bundle.validate().expect("bundle must validate");
    assert_eq!(bundle.scenario.kind, CliE2eScenarioKind::Search);
    assert!(bundle.scenario.args.contains(&"--stream".to_owned()));
    assert!(bundle.events.iter().any(|event| {
        event
            .body
            .reason_code
            .as_deref()
            .is_some_and(|code| code.starts_with("e2e.cli."))
    }));
}

#[test]
fn scenario_cli_explain_hit() {
    let scenario = scenario_by_kind(CliE2eScenarioKind::Explain);
    let bundle = CliE2eArtifactBundle::build(&CliE2eRunConfig::default(), &scenario, ExitStatus::Pass);
    bundle.validate().expect("bundle must validate");
    assert_eq!(bundle.scenario.kind, CliE2eScenarioKind::Explain);
    assert!(bundle.scenario.args.contains(&"toon".to_owned()));
}

#[test]
fn scenario_cli_degrade_path() {
    let scenario = scenario_by_kind(CliE2eScenarioKind::Degrade);
    let bundle = CliE2eArtifactBundle::build(&CliE2eRunConfig::default(), &scenario, ExitStatus::Fail);
    bundle.validate().expect("bundle must validate");
    assert_eq!(bundle.scenario.kind, CliE2eScenarioKind::Degrade);
    assert!(bundle
        .events
        .iter()
        .any(|event| event.body.outcome == Some(E2eOutcome::Fail)));
    assert!(bundle
        .manifest
        .body
        .artifacts
        .iter()
        .any(|artifact| artifact.file == "replay_command.txt"));
    assert!(bundle
        .replay_command
        .contains("--exact scenario_cli_degrade_path"));
}

#[test]
fn default_bundle_set_covers_all_cli_flows() {
    let bundles = build_default_cli_e2e_bundles(&CliE2eRunConfig::default());
    assert_eq!(bundles.len(), 4);
    for bundle in bundles {
        bundle.validate().expect("bundle must validate");
        assert_eq!(bundle.schema_version, CLI_E2E_SCHEMA_VERSION);
    }
}

#[test]
fn replay_guidance_points_at_exact_scenario_test() {
    let scenario = scenario_by_kind(CliE2eScenarioKind::Search);
    let replay = replay_command_for_scenario(&scenario);
    assert!(replay.contains("cargo test -p frankensearch-fsfs --test cli_e2e_contract"));
    assert!(replay.contains("--exact scenario_cli_search_stream"));
}
