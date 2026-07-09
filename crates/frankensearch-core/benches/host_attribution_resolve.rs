use std::collections::BTreeSet;
use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use frankensearch_core::host_adapter::{HostProjectAttribution, resolve_host_project_attribution};

#[derive(Debug, Clone, Copy)]
struct Workload {
    identity: Option<&'static str>,
    telemetry: Option<&'static str>,
    host: Option<&'static str>,
}

const WORKLOADS: &[Workload] = &[
    Workload {
        identity: Some("coding-agent-session-search"),
        telemetry: Some("cass"),
        host: Some("cass-devbox"),
    },
    Workload {
        identity: Some("xf"),
        telemetry: Some("xf-worker"),
        host: Some("xf-node-02"),
    },
    Workload {
        identity: Some("mcp_agent_mail_rust"),
        telemetry: Some("agent-mail"),
        host: Some("mail-runner-42"),
    },
    Workload {
        identity: Some("frankenterm"),
        telemetry: Some("term"),
        host: Some("frankenterm-pane"),
    },
    Workload {
        identity: None,
        telemetry: Some("agent-mail"),
        host: Some("mail-host"),
    },
    Workload {
        identity: None,
        telemetry: Some("custom-app"),
        host: Some("mystery-box"),
    },
    Workload {
        identity: Some("xf"),
        telemetry: Some("cass-host"),
        host: Some("coding-agent-session-search"),
    },
    Workload {
        identity: Some(""),
        telemetry: Some("   "),
        host: Some("unknown"),
    },
];

#[derive(Debug, Clone, Copy)]
struct Candidate {
    project: &'static str,
    weight: u8,
    reason: &'static str,
}

fn legacy_resolve(
    identity_host_project: Option<&str>,
    telemetry_project_key: Option<&str>,
    host_name_hint: Option<&str>,
) -> HostProjectAttribution {
    let mut candidates: Vec<Candidate> = Vec::new();

    if let Some(hint) = identity_host_project {
        for project in legacy_canonical_projects_from_hint(hint) {
            candidates.push(Candidate {
                project,
                weight: 4,
                reason: "adapter_identity",
            });
        }
    }

    if let Some(hint) = telemetry_project_key {
        for project in legacy_canonical_projects_from_hint(hint) {
            candidates.push(Candidate {
                project,
                weight: 3,
                reason: "telemetry_project_key",
            });
        }
    }

    if let Some(hint) = host_name_hint {
        for project in legacy_canonical_projects_from_hint(hint) {
            candidates.push(Candidate {
                project,
                weight: 1,
                reason: "host_name",
            });
        }
    }

    if candidates.is_empty() {
        return HostProjectAttribution::unknown("attribution.unknown");
    }

    let unique_projects: BTreeSet<&str> = candidates
        .iter()
        .map(|candidate| candidate.project)
        .collect();
    let collision = unique_projects.len() > 1;

    let Some(winner) = candidates.into_iter().max_by(|left, right| {
        left.weight
            .cmp(&right.weight)
            .then_with(|| right.project.cmp(left.project))
            .then_with(|| right.reason.cmp(left.reason))
    }) else {
        return HostProjectAttribution::unknown("attribution.unknown");
    };

    let mut confidence_score: u8 = match winner.weight {
        4 => 95,
        3 => 85,
        1 => 60,
        _ => 50,
    };
    if collision {
        confidence_score = confidence_score.saturating_sub(25);
    }

    let reason_code = if collision {
        "attribution.collision".to_owned()
    } else {
        format!("attribution.{}", winner.reason)
    };

    HostProjectAttribution {
        resolved_project_key: winner.project.to_owned(),
        confidence_score,
        reason_code,
        collision,
    }
}

fn legacy_canonical_projects_from_hint(hint: &str) -> Vec<&'static str> {
    const CANONICAL_ALIASES: &[(&str, &[&str])] = &[
        (
            "coding_agent_session_search",
            &[
                "coding_agent_session_search",
                "codingagentsessionsearch",
                "cass",
            ],
        ),
        ("xf", &["xf"]),
        (
            "mcp_agent_mail_rust",
            &[
                "mcp_agent_mail_rust",
                "mcpagentmailrust",
                "mcpagentmail",
                "agent_mail",
                "agentmail",
                "amail",
            ],
        ),
        ("frankenterm", &["frankenterm"]),
    ];

    let normalized = legacy_normalize_project_hint(hint);
    if normalized.is_empty() {
        return Vec::new();
    }
    let ordered_parts: Vec<&str> = normalized
        .split('_')
        .filter(|part| !part.is_empty())
        .collect();
    let parts: BTreeSet<&str> = ordered_parts.iter().copied().collect();

    let mut matches = Vec::new();
    for (canonical, aliases) in CANONICAL_ALIASES {
        if aliases
            .iter()
            .any(|alias| legacy_alias_matches_hint(&normalized, &ordered_parts, &parts, alias))
        {
            matches.push(*canonical);
        }
    }

    matches.sort_unstable();
    matches.dedup();
    matches
}

fn legacy_alias_matches_hint(
    normalized_hint: &str,
    ordered_hint_parts: &[&str],
    hint_parts: &BTreeSet<&str>,
    alias: &str,
) -> bool {
    if normalized_hint == alias || hint_parts.contains(alias) {
        return true;
    }

    let alias_parts: Vec<&str> = alias.split('_').filter(|part| !part.is_empty()).collect();
    alias_parts.len() > 1
        && alias_parts.len() <= ordered_hint_parts.len()
        && ordered_hint_parts
            .windows(alias_parts.len())
            .any(|window| window == alias_parts.as_slice())
}

fn legacy_normalize_project_hint(hint: &str) -> String {
    let mut normalized = String::with_capacity(hint.len());
    let mut pending_separator = false;

    for ch in hint.chars() {
        if ch.is_ascii_alphanumeric() {
            if pending_separator && !normalized.is_empty() {
                normalized.push('_');
            }
            normalized.push(ch.to_ascii_lowercase());
            pending_separator = false;
        } else {
            pending_separator = true;
        }
    }

    normalized
}

fn run_legacy(iterations: usize) -> u64 {
    let mut acc = 0_u64;
    for workload in WORKLOADS.iter().cycle().take(iterations) {
        let attribution = legacy_resolve(workload.identity, workload.telemetry, workload.host);
        acc = accumulate(&attribution, acc);
    }
    acc
}

fn run_current(iterations: usize) -> u64 {
    let mut acc = 0_u64;
    for workload in WORKLOADS.iter().cycle().take(iterations) {
        let attribution =
            resolve_host_project_attribution(workload.identity, workload.telemetry, workload.host);
        acc = accumulate(&attribution, acc);
    }
    acc
}

fn accumulate(attribution: &HostProjectAttribution, acc: u64) -> u64 {
    let mut next = acc.wrapping_add(u64::from(attribution.confidence_score));
    next ^= u64::try_from(attribution.resolved_project_key.len()).unwrap_or(0);
    next ^= u64::try_from(attribution.reason_code.len()).unwrap_or(0) << 8;
    if attribution.collision {
        next ^= 1 << 31;
    }
    next
}

fn bench_host_attribution(c: &mut Criterion) {
    for workload in WORKLOADS {
        assert_eq!(
            legacy_resolve(workload.identity, workload.telemetry, workload.host),
            resolve_host_project_attribution(workload.identity, workload.telemetry, workload.host)
        );
    }

    let mut group = c.benchmark_group("host_attribution_resolve");
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(350));
    group.sample_size(12);

    for iterations in [64_usize, 1_024, 16_384] {
        group.throughput(Throughput::Elements(
            u64::try_from(iterations).unwrap_or(u64::MAX),
        ));
        group.bench_with_input(
            BenchmarkId::new("legacy_ORIG", iterations),
            &iterations,
            |bench, iterations| bench.iter(|| black_box(run_legacy(black_box(*iterations)))),
        );
        group.bench_with_input(
            BenchmarkId::new("current", iterations),
            &iterations,
            |bench, iterations| bench.iter(|| black_box(run_current(black_box(*iterations)))),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_host_attribution);
criterion_main!(benches);
