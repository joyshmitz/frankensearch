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

fn main() {
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
