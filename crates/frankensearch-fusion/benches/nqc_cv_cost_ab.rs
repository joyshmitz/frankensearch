//! Enabled-path NQC allocation A/B. The original sync-searcher wiring collected every lexical
//! score into a temporary `Vec<f32>` solely to call `nqc_cv`; the candidate reduces the same
//! scores directly from the `ScoredResult` slice. Both call the shipped implementations, assert
//! bit identity before timing, and use alternating rounds plus an A/A null floor.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- \
//!   cargo bench -p frankensearch-fusion --profile release --features bench-internals \
//!   --bench nqc_cv_cost_ab
//! ```

use std::hint::black_box;

use frankensearch_core::bench_support::{PairedRatio, paired_median_ratio};
use frankensearch_core::{ScoreSource, ScoredResult};
use frankensearch_fusion::NqcDenseWeight;
use frankensearch_fusion::sync_searcher::{bench_nqc_cv_collect, bench_nqc_cv_iter};

fn make_hits(n: usize) -> Vec<ScoredResult> {
    (0..n)
        .map(|i| {
            let score = 20.0 / (1.0 + i as f32) + (i % 7) as f32 * 0.1;
            ScoredResult {
                doc_id: format!("doc-{i:06}").into(),
                score,
                source: ScoreSource::Lexical,
                index: None,
                fast_score: None,
                quality_score: None,
                lexical_score: Some(score),
                rerank_score: None,
                explanation: None,
                metadata: None,
            }
        })
        .collect()
}

fn verdict(lever: &PairedRatio, null: &PairedRatio) -> &'static str {
    if lever.decidable_against(null) {
        if lever.median < 1.0 {
            "DECIDABLE WIN"
        } else {
            "DECIDABLE REGRESSION"
        }
    } else {
        "INSIDE NULL FLOOR (not decidable)"
    }
}

fn make_query_scores(query_count: usize, scores_per_query: usize) -> Vec<Vec<f32>> {
    (0..query_count)
        .map(|query| {
            (0..scores_per_query)
                .map(|rank| {
                    20.0 / (1.0 + rank as f32) + ((query * 17 + rank * 7) % 23) as f32 * 0.01
                })
                .collect()
        })
        .collect()
}

fn query_slices(queries: &[Vec<f32>]) -> impl Iterator<Item = &[f32]> {
    queries.iter().map(Vec::as_slice)
}

fn assert_same_weight(original: &NqcDenseWeight, candidate: &NqcDenseWeight) {
    assert_eq!(original.len(), candidate.len());
    for cv in [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, f32::INFINITY] {
        assert_eq!(
            original.percentile(cv).to_bits(),
            candidate.percentile(cv).to_bits(),
            "percentile changed for cv={cv}"
        );
        assert_eq!(
            original.dense_weight(cv, 0.5, 0.1).to_bits(),
            candidate.dense_weight(cv, 0.5, 0.1).to_bits(),
            "dense weight changed for cv={cv}"
        );
    }
}

fn bench_sample_builder() {
    for (query_count, inner) in [(32_usize, 512_u32), (256, 64), (4_096, 4)] {
        let queries = make_query_scores(query_count, 100);
        let original = NqcDenseWeight::bench_from_query_scores_collect(query_slices(&queries));
        let candidate = NqcDenseWeight::bench_from_query_scores_iter(query_slices(&queries));
        assert_same_weight(&original, &candidate);

        let run_orig = || {
            black_box(NqcDenseWeight::bench_from_query_scores_collect(black_box(
                query_slices(&queries),
            )));
        };
        let run_cand = || {
            black_box(NqcDenseWeight::bench_from_query_scores_iter(black_box(
                query_slices(&queries),
            )));
        };
        let null = paired_median_ratio(41, inner, run_orig, run_orig);
        let lever = paired_median_ratio(41, inner, run_orig, run_cand);
        eprintln!(
            "[null]  nqc_sample_builder/q{query_count}: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
            null.median, null.p5, null.p95, null.rounds
        );
        eprintln!(
            "[lever] nqc_sample_builder/q{query_count}: cand/ORIG median {:.4} p5 {:.4} p95 {:.4} -> {}",
            lever.median,
            lever.p5,
            lever.p95,
            verdict(&lever, &null)
        );
    }
}

fn main() {
    bench_sample_builder();

    let inner = std::env::var("NQC_ALLOC_AB_INNER")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2_048);

    for n in [20_usize, 100, 1_000] {
        let hits = make_hits(n);
        assert_eq!(
            bench_nqc_cv_collect(&hits).to_bits(),
            bench_nqc_cv_iter(&hits).to_bits(),
            "collect and iterator NQC differ for n={n}"
        );

        let run_orig = || {
            black_box(bench_nqc_cv_collect(black_box(&hits)));
        };
        let run_cand = || {
            black_box(bench_nqc_cv_iter(black_box(&hits)));
        };
        let null = paired_median_ratio(41, inner, run_orig, run_orig);
        let lever = paired_median_ratio(41, inner, run_orig, run_cand);
        eprintln!(
            "[null]  nqc_alloc/n{n}: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
            null.median, null.p5, null.p95, null.rounds
        );
        eprintln!(
            "[lever] nqc_alloc/n{n}: cand/ORIG median {:.4} p5 {:.4} p95 {:.4} -> {}",
            lever.median,
            lever.p5,
            lever.p95,
            verdict(&lever, &null)
        );
    }
}
