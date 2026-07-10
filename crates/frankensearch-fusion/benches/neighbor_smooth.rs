//! Latency A/B: graph score-diffusion (kNN neighbor smoothing) vs the no-smooth ORIGINAL.
//!
//! [`neighbor_smooth`] is a label-propagation pass that rescues below-threshold relevants
//! sitting in confident clusters — measured **+0.0052…+0.0114 hybrid nDCG@10 on recall-bound
//! BEIR corpora** (docs/NEGATIVE_EVIDENCE.md; pool-restricted deployable form). This bench
//! confirms it is nearly free atop an existing k-NN neighbor graph: the LEGACY ORIGINAL is
//! the pipeline with no smoothing (identity, α=0 — a plain pool clone); the candidate is the
//! full diffusion pass (α=0.3, M=10) and its reciprocal-kNN refinement. Cost is O(pool · M).
//!
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-fusion --profile release --bench neighbor_smooth
//! ```

use std::hint::black_box;
use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_core::{DocumentGraph, EdgeType, VectorHit};
use frankensearch_fusion::bench_support::paired_median_ratio;
use frankensearch_fusion::{SmoothConfig, neighbor_smooth, neighbor_smooth_ranked};

/// Calls per timed region in the interleaved paired sampler — amortizes the `Instant::now()` pair.
const PAIR_BATCH: u32 = 16;

/// A pool of `pool` candidates with heavy-tailed cosine scores, and an M-nearest `Similar`
/// graph, all in-pool so the kernel does real averaging work.
///
/// Neighbours are **scattered** (`i*37 + j*101 mod pool`), not the `m` following docs. That is
/// load bearing for the `smooth_ranked` arm: with a forward chain, every doc's neighbours score
/// strictly below it, so diffusing a convex decreasing sequence preserves its order and the pool
/// comes out ALREADY SORTED — `sort_unstable_by` then short-circuits on its sortedness check and
/// the arm times a detection pass over dead code instead of a sort. A scattered graph lets a
/// low-ranked doc borrow from a high-scoring cluster and actually overtake its neighbours, which
/// is the promotion smoothing exists to produce. The reachability gate in `bench` asserts this.
fn build(pool: usize, m: usize) -> (Vec<VectorHit>, DocumentGraph) {
    let hits: Vec<VectorHit> = (0..pool)
        .map(|i| VectorHit {
            index: i as u32,
            score: 1.0 / ((i as f32) + 1.0), // heavy-tailed like real cosine pools
            doc_id: format!("doc{i:06}").into(),
        })
        .collect();
    let mut graph = DocumentGraph::new();
    for i in 0..pool {
        for j in 1..=m {
            let nbr = (i * 37 + j * 101) % pool;
            if nbr == i {
                continue;
            }
            graph.add_edge(
                format!("doc{i:06}"),
                format!("doc{nbr:06}"),
                EdgeType::Similar,
                1.0 - (j as f32) * 0.01,
            );
            // add the reverse edge so mutual-kNN has real reciprocal work to do
            graph.add_edge(
                format!("doc{nbr:06}"),
                format!("doc{i:06}"),
                EdgeType::Similar,
                1.0 - (j as f32) * 0.01,
            );
        }
    }
    (hits, graph)
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("neighbor_smooth");
    g.sample_size(50);
    g.warm_up_time(Duration::from_millis(400));
    g.measurement_time(Duration::from_millis(2000));

    let identity = SmoothConfig {
        alpha: 0.0,
        m: 10,
        mutual: false,
    };
    let smooth = SmoothConfig {
        alpha: 0.3,
        m: 10,
        mutual: false,
    };
    let mutual = SmoothConfig {
        alpha: 0.3,
        m: 10,
        mutual: true,
    };

    for &pool in &[50usize, 100, 1000] {
        let (hits, graph) = build(pool, 10);
        // Sanity: smoothing changes the scores (real work happens).
        let s = neighbor_smooth(&hits, &graph, &smooth);
        assert_eq!(s.len(), pool);
        assert!((s[10].score - hits[10].score).abs() > 1e-9);

        // REACHABILITY GATE for the `smooth_ranked` arm. Changing scores is not enough: if
        // diffusion left the pool in sorted order, `sort_unstable_by` would short-circuit and the
        // arm would time a detection pass over dead code, making the "re-sort is nearly free"
        // claim an artifact. Assert the sort genuinely permutes the pool, and report by how much.
        let ranked = neighbor_smooth_ranked(&hits, &graph, &smooth);
        let displaced = s
            .iter()
            .zip(&ranked)
            .filter(|(a, b)| a.doc_id != b.doc_id)
            .count();
        assert!(
            displaced > 0,
            "pool {pool}: smoothing left the pool already sorted — the sort arm would measure nothing"
        );
        eprintln!("[reachability] pool {pool}: {displaced}/{pool} hits displaced by the re-sort");

        // ── DECIDABILITY: alternating-round paired sampler + A/A null control ──────────────
        //
        // The criterion `paired_*` arms below CANNOT decide this lever: criterion runs the two arms
        // as separate benchmarks minutes apart, so worker drift between them is not cancelled. Its
        // A/A null measured 1.1265x at pool 50 — a 12.65% floor, larger than the effect. The sampler
        // here runs both arms in ONE routine in alternating rounds and takes the median per-round
        // ratio, collapsing the floor. Gate on the median against the null spread, not on cv.
        let null = paired_median_ratio(
            41,
            8,
            || {
                black_box(neighbor_smooth(
                    black_box(&hits),
                    black_box(&graph),
                    &smooth,
                ));
            },
            || {
                black_box(neighbor_smooth(
                    black_box(&hits),
                    black_box(&graph),
                    &smooth,
                ));
            },
        );
        let lever = paired_median_ratio(
            41,
            8,
            || {
                black_box(neighbor_smooth(
                    black_box(&hits),
                    black_box(&graph),
                    &smooth,
                ));
            },
            || {
                black_box(neighbor_smooth_ranked(
                    black_box(&hits),
                    black_box(&graph),
                    &smooth,
                ));
            },
        );
        eprintln!(
            "[null]  pool {pool}: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
            null.median, null.p5, null.p95, null.rounds
        );
        eprintln!(
            "[lever] pool {pool}: re-sort median {:.4} p5 {:.4} p95 {:.4} -> {}",
            lever.median,
            lever.p5,
            lever.p95,
            if lever.decidable_against(&null) {
                "DECIDABLE"
            } else {
                "INSIDE NULL FLOOR (not decidable)"
            }
        );

        g.bench_with_input(BenchmarkId::new("identity", pool), &(), |b, ()| {
            b.iter(|| {
                black_box(neighbor_smooth(
                    black_box(&hits),
                    black_box(&graph),
                    &identity,
                ))
            });
        });
        g.bench_with_input(BenchmarkId::new("smooth", pool), &(), |b, ()| {
            b.iter(|| {
                black_box(neighbor_smooth(
                    black_box(&hits),
                    black_box(&graph),
                    &smooth,
                ))
            });
        });
        g.bench_with_input(BenchmarkId::new("mutual", pool), &(), |b, ()| {
            b.iter(|| {
                black_box(neighbor_smooth(
                    black_box(&hits),
                    black_box(&graph),
                    &mutual,
                ))
            });
        });

        // ── Searcher wiring (bd-kdjr): the re-sort that makes smoothing usable by rank fusion ──
        //
        // `smooth` (ORIG) is the bare kernel; `smooth_ranked` is what the searcher actually calls.
        // The delta is the deterministic descending re-sort, which is a CORRECTNESS requirement:
        // fusion assigns ranks by position, so without it a promoted doc keeps its old rank.
        //
        // The default-off path is not benched because it does not exist as work: the searcher's
        // `match` yields `fast_hits` by move when smoothing is disabled — no call, no clone.
        assert_eq!(
            neighbor_smooth_ranked(&hits, &graph, &identity),
            hits,
            "identity must be byte-identical passthrough at pool {pool}"
        );
        // INTERLEAVED PAIRED SAMPLER. Criterion group members run *sequentially*, so registering
        // ORIG and CAND side by side (even ORIG-first-and-last) does not cancel worker or thermal
        // drift — it only exposes it. Here each arm's measured routine runs BOTH implementations
        // every iteration and times only its own, so the two benchmarks perform identical total
        // work and see identical machine state; drift hits both equally and cancels in the ratio.
        // Calls are batched so the two `Instant::now()` reads amortize to <2% of the timed region.
        //
        // NULL CONTROL (A/A): the SAME arm registered twice with the identical interleaved
        // structure, measuring this harness's noise floor. Read it BEFORE believing any lever
        // ratio below — an effect smaller than the floor is indistinguishable from noise, and a
        // WIN or REJECT resting on such an effect is meaningless. The re-sort cost measured here
        // is only ~7–11%, so the floor is load bearing, not ceremony.
        g.bench_with_input(BenchmarkId::new("paired_null_a", pool), &(), |b, ()| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    for _ in 0..PAIR_BATCH {
                        black_box(neighbor_smooth(
                            black_box(&hits),
                            black_box(&graph),
                            &smooth,
                        ));
                    }
                    let t = Instant::now();
                    for _ in 0..PAIR_BATCH {
                        black_box(neighbor_smooth(
                            black_box(&hits),
                            black_box(&graph),
                            &smooth,
                        ));
                    }
                    total += t.elapsed();
                }
                total
            });
        });
        g.bench_with_input(BenchmarkId::new("paired_null_b", pool), &(), |b, ()| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    for _ in 0..PAIR_BATCH {
                        black_box(neighbor_smooth(
                            black_box(&hits),
                            black_box(&graph),
                            &smooth,
                        ));
                    }
                    let t = Instant::now();
                    for _ in 0..PAIR_BATCH {
                        black_box(neighbor_smooth(
                            black_box(&hits),
                            black_box(&graph),
                            &smooth,
                        ));
                    }
                    total += t.elapsed();
                }
                total
            });
        });

        g.bench_with_input(BenchmarkId::new("paired_smooth", pool), &(), |b, ()| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    for _ in 0..PAIR_BATCH {
                        black_box(neighbor_smooth_ranked(
                            black_box(&hits),
                            black_box(&graph),
                            &smooth,
                        ));
                    }
                    let t = Instant::now();
                    for _ in 0..PAIR_BATCH {
                        black_box(neighbor_smooth(
                            black_box(&hits),
                            black_box(&graph),
                            &smooth,
                        ));
                    }
                    total += t.elapsed();
                }
                total
            });
        });
        g.bench_with_input(
            BenchmarkId::new("paired_smooth_ranked", pool),
            &(),
            |b, ()| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        for _ in 0..PAIR_BATCH {
                            black_box(neighbor_smooth(
                                black_box(&hits),
                                black_box(&graph),
                                &smooth,
                            ));
                        }
                        let t = Instant::now();
                        for _ in 0..PAIR_BATCH {
                            black_box(neighbor_smooth_ranked(
                                black_box(&hits),
                                black_box(&graph),
                                &smooth,
                            ));
                        }
                        total += t.elapsed();
                    }
                    total
                });
            },
        );
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
