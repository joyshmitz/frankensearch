//! Candidate-list code-structure sidecar scoring benchmark.
//!
//! Legacy: callers loop `CodeStructureSidecar::score_query(query, doc_id)`,
//! which prepares query tokens and normalized query text once per candidate.
//! Candidate path: `prior_signals_for_candidates` can prepare that query state
//! once and reuse it across all candidates. The output map must match.

use std::collections::HashMap;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_fsfs::code_structure_sidecar::CodeStructureSidecar;
use frankensearch_fsfs::query_execution::{FusedCandidate, RankingPriorSignals};

fn rust_doc(i: usize) -> String {
    format!(
        r"
        use crate::auth::token_store::TokenStore{i};
        use std::collections::HashMap;

        pub struct UserSessionManager{i} {{
            cache: HashMap<String, String>,
        }}

        pub enum AuthResult{i} {{
            Accepted,
            Rejected,
        }}

        pub fn authenticate_user_session_{i}(token: &str) -> bool {{
            validate_bearer_token_{i}(token)
        }}

        fn validate_bearer_token_{i}(token: &str) -> bool {{
            !token.is_empty()
        }}

        mod session_cookie_jar_{i};
        "
    )
}

fn markdown_doc(i: usize) -> String {
    format!(
        "# Authentication Guide {i}\n\n\
         ## Session Token Rotation {i}\n\n\
         This document describes bearer token refresh and login session handling.\n"
    )
}

fn build_fixture(n: usize) -> (CodeStructureSidecar, Vec<FusedCandidate>) {
    let mut sidecar = CodeStructureSidecar::new();
    let mut candidates = Vec::with_capacity(n);
    for i in 0..n {
        let doc_id = format!("doc-{i:05}");
        let path = if i % 8 == 0 {
            format!("docs/auth/session_guide_{i}.md")
        } else {
            format!("src/auth/session_handler_{i}.rs")
        };
        let content = if i % 8 == 0 {
            markdown_doc(i)
        } else {
            rust_doc(i)
        };
        sidecar.insert_document(doc_id.clone(), path, content);
        candidates.push(FusedCandidate {
            doc_id,
            fused_score: 1.0 / (i + 1) as f64,
            prior_boost: 0.0,
            lexical_rank: Some(i),
            semantic_rank: None,
            lexical_score: None,
            semantic_score: None,
            in_both_sources: false,
        });
    }
    (sidecar, candidates)
}

fn legacy_prior_signals(
    sidecar: &CodeStructureSidecar,
    query: &str,
    candidates: &[FusedCandidate],
) -> HashMap<String, RankingPriorSignals> {
    let mut signals = HashMap::new();
    for candidate in candidates {
        let evidence = sidecar.score_query(query, &candidate.doc_id);
        if evidence.score > 0.0 {
            signals.insert(
                candidate.doc_id.clone(),
                RankingPriorSignals::default().with_code_structure(Some(evidence.score)),
            );
        }
    }
    signals
}

fn bench_sidecar_candidate_score(c: &mut Criterion) {
    let query = "authenticate user session token";
    let mut group = c.benchmark_group("sidecar_candidate_score");
    for &n in &[32_usize, 128, 512] {
        let (sidecar, candidates) = build_fixture(n);
        let legacy = legacy_prior_signals(&sidecar, query, &candidates);
        let candidate = sidecar.prior_signals_for_candidates(query, &candidates);
        assert_eq!(legacy, candidate);

        group.bench_with_input(
            BenchmarkId::new("legacy_score_query_loop", n),
            &n,
            |b, _| {
                b.iter(|| {
                    black_box(legacy_prior_signals(
                        black_box(&sidecar),
                        black_box(query),
                        black_box(&candidates),
                    ))
                });
            },
        );
        group.bench_with_input(BenchmarkId::new("prior_signals", n), &n, |b, _| {
            b.iter(|| {
                black_box(
                    sidecar.prior_signals_for_candidates(black_box(query), black_box(&candidates)),
                )
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_sidecar_candidate_score);
criterion_main!(benches);
