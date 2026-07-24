//! Latency A/B: merge-structured pool-min-max fusion vs the value-map ORIGINAL.
//!
//! `pool_minmax_fuse` (LEGACY ORIGINAL) accumulates every doc into one `N`-entry
//! `HashMap<&str, FusedHitScratch>`, then `into_values()` in **random** hash order and sorts
//! from scratch — an `O(N log N)` sort on random input for the `limit_all` shape (window ≥ N,
//! e.g. returning a full ranked feed to a reranker). `pool_minmax_fuse_merge` keeps only a small
//! `&str → (rank, score)` lexical contribution map and walks the already-score-sorted `semantic`
//! slice once in order, so `results` is **near-sorted** and the final sort runs near-`O(N)`
//! (pdqsort is adaptive). This is the same primitive that made `rrf_fuse_with_graph_merge` beat
//! the RRF value-map (`4aeb66b`, 1.31–1.46× on limit_all); it had never been ported to the
//! pool-min-max quality operator (`a9e53b4`, +0.0038 nDCG@10 over RRF).
//!
//! Two shapes: `limit_all` (window ≥ N — where the merge pays) and `top10` (select_nth dominates
//! — a regression guard: the merge must not lose the small-window path). Bit-identity is gated
//! before timing.
//!
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-copper \
//!   rch exec -- cargo bench -p frankensearch-fusion --profile release --bench pool_minmax_merge_ab
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_core::{FusedHit, FusionStrategy, ScoreSource, ScoredResult, VectorHit};
use frankensearch_fusion::bench_support::paired_median_ratio;
use frankensearch_fusion::rrf::rrf_fuse_with_graph_merge_unique;
use frankensearch_fusion::{RrfConfig, fuse_by_strategy, pool_minmax_fuse, pool_minmax_fuse_merge};

fn lexical(doc: &str, score: f32) -> ScoredResult {
    ScoredResult {
        doc_id: doc.into(),
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
}

fn semantic(doc: &str, score: f32, index: u32) -> VectorHit {
    VectorHit {
        index,
        score,
        doc_id: doc.into(),
    }
}

/// Two pools of `pool` candidates with ~50% doc overlap (shared docs in the middle), heavy-tailed
/// scores like real BM25 / cosine pools. Semantic is score-sorted descending (as a vector index
/// returns), so the merge path's near-sorted assumption is exercised.
fn build(pool: usize) -> (Vec<ScoredResult>, Vec<VectorHit>) {
    let lex: Vec<ScoredResult> = (0..pool)
        .map(|i| lexical(&format!("doc{i:06}"), (pool - i) as f32 * 0.5 + 1.0))
        .collect();
    let half = pool / 2;
    let sem: Vec<VectorHit> = (half..pool + half)
        .map(|i| {
            semantic(
                &format!("doc{i:06}"),
                1.0 / ((i - half) as f32 + 1.0),
                i as u32,
            )
        })
        .collect();
    (lex, sem)
}

fn assert_bit_identical(
    lex: &[ScoredResult],
    sem: &[VectorHit],
    limit: usize,
    cfg: &RrfConfig,
    tag: &str,
) {
    let a: Vec<FusedHit> = pool_minmax_fuse(lex, sem, limit, 0, cfg);
    let b: Vec<FusedHit> = pool_minmax_fuse_merge(lex, sem, limit, 0, cfg);
    assert_eq!(a.len(), b.len(), "{tag}: length differs");
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert_eq!(x.doc_id, y.doc_id, "{tag} row {i}: doc_id");
        assert_eq!(
            x.rrf_score.to_bits(),
            y.rrf_score.to_bits(),
            "{tag} row {i}: fused score ({} vs {})",
            x.rrf_score,
            y.rrf_score
        );
        assert_eq!(
            x.semantic_index, y.semantic_index,
            "{tag} row {i}: semantic_index"
        );
        assert_eq!(
            x.in_both_sources, y.in_both_sources,
            "{tag} row {i}: in_both"
        );
    }
}

/// `FusedHit` has no `PartialEq`, so compare the fields that define a ranking: order, doc, and the
/// fused score to the bit.
fn assert_fused_identical(a: &[FusedHit], b: &[FusedHit], tag: &str) {
    assert_eq!(a.len(), b.len(), "{tag}: length differs");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.doc_id, y.doc_id, "{tag} row {i}: doc_id");
        assert_eq!(
            x.rrf_score.to_bits(),
            y.rrf_score.to_bits(),
            "{tag} row {i}: fused score ({} vs {})",
            x.rrf_score,
            y.rrf_score
        );
    }
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("pool_minmax_merge");
    g.sample_size(50);
    g.warm_up_time(Duration::from_millis(400));
    g.measurement_time(Duration::from_millis(2000));
    let cfg = RrfConfig::default();

    for &pool in &[50usize, 100, 1000] {
        let (lex, sem) = build(pool);
        let all = 3 * pool; // window ≥ unique N (~1.5·pool) → full sort (limit_all shape)

        assert_bit_identical(&lex, &sem, all, &cfg, &format!("limit_all/{pool}"));
        assert_bit_identical(&lex, &sem, 10, &cfg, &format!("top10/{pool}"));

        // ── DECIDABILITY: A/A null control vs the limit_all merge lever (bd-zgq6) ──────────
        //
        // The criterion `limit_all_ORIG`/`limit_all_merge` arms below run as separate benchmarks,
        // minutes apart, so drift between them is NOT cancelled — its A/A null on this fleet is
        // ~±12%, and the claimed merge win (1.15–1.32×) partly sits inside that. This alternating
        // -round sampler (shared harness, promoted to core) collapses the floor and gives a
        // decidable verdict. Gate on the median vs the null p5..p95 spread.
        let null = paired_median_ratio(
            41,
            8,
            || {
                black_box(pool_minmax_fuse(
                    black_box(&lex),
                    black_box(&sem),
                    all,
                    0,
                    &cfg,
                ));
            },
            || {
                black_box(pool_minmax_fuse(
                    black_box(&lex),
                    black_box(&sem),
                    all,
                    0,
                    &cfg,
                ));
            },
        );
        // Candidate = merge / ORIG, so <1.0 means the merge structure is faster.
        let lever = paired_median_ratio(
            41,
            8,
            || {
                black_box(pool_minmax_fuse(
                    black_box(&lex),
                    black_box(&sem),
                    all,
                    0,
                    &cfg,
                ));
            },
            || {
                black_box(pool_minmax_fuse_merge(
                    black_box(&lex),
                    black_box(&sem),
                    all,
                    0,
                    &cfg,
                ));
            },
        );
        eprintln!(
            "[null]  limit_all/{pool}: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
            null.median, null.p5, null.p95, null.rounds
        );
        eprintln!(
            "[lever] limit_all/{pool}: merge/ORIG median {:.4} p5 {:.4} p95 {:.4} -> {}",
            lever.median,
            lever.p5,
            lever.p95,
            if lever.decidable_against(&null) {
                "DECIDABLE"
            } else {
                "INSIDE NULL FLOOR (not decidable)"
            }
        );

        // limit_all — the shape the merge structure targets.
        g.bench_with_input(BenchmarkId::new("limit_all_ORIG", pool), &(), |b, ()| {
            b.iter(|| {
                black_box(pool_minmax_fuse(
                    black_box(&lex),
                    black_box(&sem),
                    all,
                    0,
                    &cfg,
                ))
            });
        });
        g.bench_with_input(BenchmarkId::new("limit_all_merge", pool), &(), |b, ()| {
            b.iter(|| {
                black_box(pool_minmax_fuse_merge(
                    black_box(&lex),
                    black_box(&sem),
                    all,
                    0,
                    &cfg,
                ))
            });
        });
        // top10 — regression guard (select_nth dominates; merge must not lose here).
        g.bench_with_input(BenchmarkId::new("top10_ORIG", pool), &(), |b, ()| {
            b.iter(|| {
                black_box(pool_minmax_fuse(
                    black_box(&lex),
                    black_box(&sem),
                    10,
                    0,
                    &cfg,
                ))
            });
        });
        g.bench_with_input(BenchmarkId::new("top10_merge", pool), &(), |b, ()| {
            b.iter(|| {
                black_box(pool_minmax_fuse_merge(
                    black_box(&lex),
                    black_box(&sem),
                    10,
                    0,
                    &cfg,
                ))
            });
        });

        // ── Searcher-wiring dispatch overhead (bd-2ocs) ──────────────────────────────────────
        //
        // Both searchers now call `fuse_by_strategy` instead of the RRF entry point directly.
        // The DEFAULT strategy must be latency-neutral, not merely byte-identical: this pair is
        // the ORIGINAL (direct call, what shipped before the wiring) vs the dispatched call.
        // Any gap is the cost of one `match` on a `Copy` enum, and must vanish under inlining.
        assert_fused_identical(
            &fuse_by_strategy(FusionStrategy::Rrf, &lex, &sem, &[], 0.0, all, 0, &cfg),
            &rrf_fuse_with_graph_merge_unique(&lex, &sem, &[], 0.0, all, 0, &cfg),
            &format!("dispatch(Rrf) vs direct RRF at pool {pool}"),
        );
        g.bench_with_input(
            BenchmarkId::new("dispatch_ORIG_direct_rrf", pool),
            &(),
            |b, ()| {
                b.iter(|| {
                    black_box(rrf_fuse_with_graph_merge_unique(
                        black_box(&lex),
                        black_box(&sem),
                        &[],
                        0.0,
                        all,
                        0,
                        &cfg,
                    ))
                });
            },
        );
        g.bench_with_input(
            BenchmarkId::new("dispatch_via_strategy_rrf", pool),
            &(),
            |b, ()| {
                b.iter(|| {
                    black_box(fuse_by_strategy(
                        black_box(FusionStrategy::Rrf),
                        black_box(&lex),
                        black_box(&sem),
                        &[],
                        0.0,
                        all,
                        0,
                        &cfg,
                    ))
                });
            },
        );
        // ORIG measured last too, to bracket criterion's arm-ordering bias.
        g.bench_with_input(
            BenchmarkId::new("dispatch_ORIG_direct_rrf2", pool),
            &(),
            |b, ()| {
                b.iter(|| {
                    black_box(rrf_fuse_with_graph_merge_unique(
                        black_box(&lex),
                        black_box(&sem),
                        &[],
                        0.0,
                        all,
                        0,
                        &cfg,
                    ))
                });
            },
        );

        // ── DECIDABILITY (bd-zgq6): top10 regression-guard and dispatch-overhead pairs ──
        //
        // Same alternating-round paired sampler + A/A null control as the limit_all block
        // above; one null+lever pair per remaining comparison.
        let top10_orig = || {
            black_box(pool_minmax_fuse(
                black_box(&lex),
                black_box(&sem),
                10,
                0,
                &cfg,
            ));
        };
        let top10_merge = || {
            black_box(pool_minmax_fuse_merge(
                black_box(&lex),
                black_box(&sem),
                10,
                0,
                &cfg,
            ));
        };
        let null = paired_median_ratio(41, 8, top10_orig, top10_orig);
        let lever = paired_median_ratio(41, 8, top10_orig, top10_merge);
        eprintln!(
            "[null]  top10/{pool}: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
            null.median, null.p5, null.p95, null.rounds
        );
        eprintln!(
            "[lever] top10/{pool}: merge/ORIG median {:.4} p5 {:.4} p95 {:.4} -> {}",
            lever.median,
            lever.p5,
            lever.p95,
            if lever.decidable_against(&null) {
                "DECIDABLE"
            } else {
                "INSIDE NULL FLOOR (not decidable)"
            }
        );
        let dispatch_direct = || {
            black_box(rrf_fuse_with_graph_merge_unique(
                black_box(&lex),
                black_box(&sem),
                &[],
                0.0,
                all,
                0,
                &cfg,
            ));
        };
        let dispatch_strategy = || {
            black_box(fuse_by_strategy(
                black_box(FusionStrategy::Rrf),
                black_box(&lex),
                black_box(&sem),
                &[],
                0.0,
                all,
                0,
                &cfg,
            ));
        };
        let null = paired_median_ratio(41, 8, dispatch_direct, dispatch_direct);
        let lever = paired_median_ratio(41, 8, dispatch_direct, dispatch_strategy);
        eprintln!(
            "[null]  dispatch/{pool}: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
            null.median, null.p5, null.p95, null.rounds
        );
        eprintln!(
            "[lever] dispatch/{pool}: strategy/direct median {:.4} p5 {:.4} p95 {:.4} -> {}",
            lever.median,
            lever.p5,
            lever.p95,
            if lever.decidable_against(&null) {
                "DECIDABLE"
            } else {
                "INSIDE NULL FLOOR (not decidable)"
            }
        );
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
