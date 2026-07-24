//! Latency A/B: query-hubness dense-score correction vs the no-correction ORIGINAL.
//!
//! [`apply_hubness_penalty`] demotes high-hub docs before fusion — measured **all-positive
//! +0.0033 mean hybrid nDCG@10 (β=0.2, leakage-free) across 4 BEIR corpora**, with genuine
//! dense-tier gains on stance/citation corpora (`docs/NEGATIVE_EVIDENCE.md`). The query-time
//! kernel is a trivial O(pool) subtract; the LEGACY ORIGINAL is the pipeline with no correction
//! (identity, β=0 — a plain pool clone). We also bench the offline `compute_query_hubness`
//! builder (amortized: recomputed periodically from the query log, not per query).
//!
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-fusion --profile release --bench hubness_penalty
//! ```

use std::hint::black_box;
use std::time::{Duration, Instant};

/// Calls per timed region in the interleaved paired sampler — amortizes the `Instant::now()` pair.
const PAIR_BATCH: u32 = 16;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_core::VectorHit;
use frankensearch_fusion::{HubnessConfig, apply_hubness_penalty, compute_query_hubness};

fn build_pool(pool: usize) -> (Vec<VectorHit>, Vec<f32>) {
    let hits: Vec<VectorHit> = (0..pool)
        .map(|i| VectorHit {
            index: u32::try_from(i).expect("benchmark index fits u32"),
            score: 1.0 / ((i as f32) + 1.0),
            doc_id: format!("doc{i:06}").into(),
        })
        .collect();
    // per-doc hubness in [0,1], heavy near the front (hubs)
    let hubness: Vec<f32> = (0..pool)
        .map(|i| 0.9 - (i as f32 / pool as f32) * 0.6)
        .collect();
    (hits, hubness)
}

/// Deterministic normalized vectors (no RNG in benches).
fn vecs(n: usize, dim: usize, seed: u32) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| {
            let i_u32 = u32::try_from(i).expect("benchmark vector index fits u32");
            let mut v: Vec<f32> = (0..dim)
                .map(|j| {
                    let j_u32 = u32::try_from(j).expect("benchmark dimension index fits u32");
                    let mixed = i_u32
                        .wrapping_mul(2_654_435_761)
                        .wrapping_add(j_u32.wrapping_mul(40_503))
                        .wrapping_add(seed);
                    ((mixed % 997) as f32) / 997.0 - 0.5
                })
                .collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            for x in &mut v {
                *x /= norm;
            }
            v
        })
        .collect()
}

#[allow(
    clippy::significant_drop_tightening,
    reason = "Criterion benchmark groups intentionally span their complete arm sets"
)]
fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("hubness_penalty");
    g.sample_size(50);
    g.warm_up_time(Duration::from_millis(400));
    g.measurement_time(Duration::from_millis(2000));

    let identity = HubnessConfig { beta: 0.0, kq: 10 };
    let correct = HubnessConfig { beta: 0.2, kq: 10 };

    for &pool in &[50usize, 100, 1000] {
        let (hits, hub) = build_pool(pool);
        let s = apply_hubness_penalty(&hits, &hub, &correct);
        assert_eq!(s.len(), pool);
        assert!((s[10].score - hits[10].score).abs() > 1e-9);

        // REACHABILITY GATE for the `correct_ranked` arm. If the corrected pool were already in
        // sorted order, `sort_unstable_by` would short-circuit and the arm would time a detection
        // pass over dead code rather than a real sort. Assert the penalty genuinely permutes the
        // pool, and report how far, so the measured re-sort cost is attributable to actual work.
        let mut ranked = s.clone();
        ranked.sort_unstable_by(VectorHit::cmp_rank);
        let displaced = s
            .iter()
            .zip(&ranked)
            .filter(|(a, b)| a.doc_id != b.doc_id)
            .count();
        assert!(
            displaced > 0,
            "pool {pool}: hubness left the pool already sorted — the sort arm would measure nothing"
        );
        eprintln!("[reachability] pool {pool}: {displaced}/{pool} hits displaced by the re-sort");

        g.bench_with_input(BenchmarkId::new("identity", pool), &(), |b, ()| {
            b.iter(|| {
                black_box(apply_hubness_penalty(
                    black_box(&hits),
                    black_box(&hub),
                    &identity,
                ))
            });
        });
        g.bench_with_input(BenchmarkId::new("correct", pool), &(), |b, ()| {
            b.iter(|| {
                black_box(apply_hubness_penalty(
                    black_box(&hits),
                    black_box(&hub),
                    &correct,
                ))
            });
        });

        // ── Searcher wiring (bd-kdjr): the re-sort that makes the demotion reach fusion ──
        //
        // `correct` (ORIG) is the bare O(pool) subtract; the searcher must additionally re-sort,
        // because rank-based fusion assigns ranks by POSITION — without it a demoted hub keeps its
        // rank and the correction is silently discarded. `correct2` re-measures ORIG last to
        // bracket criterion's ordering bias. The identity arm is not the searcher's default path:
        // when `beta == 0.0` the searcher never calls the kernel at all (the pool moves through).
        //
        // The input is score-sorted and hubness is heaviest at the front, so the penalty perturbs
        // a sorted pool — the same near-sorted shape the real Phase-1 pool has.
        // INTERLEAVED PAIRED SAMPLER — see `neighbor_smooth.rs` for the rationale. Criterion group
        // members run sequentially, so an ORIG-first-and-last bracket exposes drift rather than
        // cancelling it. Each arm below runs BOTH implementations per iteration and times only its
        // own, batched so the `Instant::now()` pair amortizes.
        let ranked = |hits: &[VectorHit], hub: &[f32]| {
            let mut out = apply_hubness_penalty(hits, hub, &correct);
            out.sort_unstable_by(VectorHit::cmp_rank);
            out
        };
        // NULL CONTROL (A/A): the same arm twice, identical interleaved structure. It measures the
        // harness noise floor, which must be read before any lever ratio below is believed.
        g.bench_with_input(BenchmarkId::new("paired_null_a", pool), &(), |b, ()| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    for _ in 0..PAIR_BATCH {
                        black_box(apply_hubness_penalty(
                            black_box(&hits),
                            black_box(&hub),
                            &correct,
                        ));
                    }
                    let t = Instant::now();
                    for _ in 0..PAIR_BATCH {
                        black_box(apply_hubness_penalty(
                            black_box(&hits),
                            black_box(&hub),
                            &correct,
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
                        black_box(apply_hubness_penalty(
                            black_box(&hits),
                            black_box(&hub),
                            &correct,
                        ));
                    }
                    let t = Instant::now();
                    for _ in 0..PAIR_BATCH {
                        black_box(apply_hubness_penalty(
                            black_box(&hits),
                            black_box(&hub),
                            &correct,
                        ));
                    }
                    total += t.elapsed();
                }
                total
            });
        });

        g.bench_with_input(BenchmarkId::new("paired_correct", pool), &(), |b, ()| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    for _ in 0..PAIR_BATCH {
                        black_box(ranked(black_box(&hits), black_box(&hub)));
                    }
                    let t = Instant::now();
                    for _ in 0..PAIR_BATCH {
                        black_box(apply_hubness_penalty(
                            black_box(&hits),
                            black_box(&hub),
                            &correct,
                        ));
                    }
                    total += t.elapsed();
                }
                total
            });
        });
        g.bench_with_input(
            BenchmarkId::new("paired_correct_ranked", pool),
            &(),
            |b, ()| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        for _ in 0..PAIR_BATCH {
                            black_box(apply_hubness_penalty(
                                black_box(&hits),
                                black_box(&hub),
                                &correct,
                            ));
                        }
                        let t = Instant::now();
                        for _ in 0..PAIR_BATCH {
                            black_box(ranked(black_box(&hits), black_box(&hub)));
                        }
                        total += t.elapsed();
                    }
                    total
                });
            },
        );
    }
    g.finish();

    // Offline builder cost (amortized): per-doc mean-cos to kq nearest of a query sample.
    let mut gb = c.benchmark_group("hubness_build");
    gb.sample_size(20);
    gb.warm_up_time(Duration::from_millis(300));
    gb.measurement_time(Duration::from_millis(2000));
    let dim = 384;
    for &(docs, queries) in &[(2000usize, 200usize), (5000, 500usize)] {
        let dv = vecs(docs, dim, 1);
        let qv = vecs(queries, dim, 7);
        let dref: Vec<&[f32]> = dv.iter().map(Vec::as_slice).collect();
        let qref: Vec<&[f32]> = qv.iter().map(Vec::as_slice).collect();
        gb.bench_with_input(
            BenchmarkId::new("build", format!("{docs}x{queries}")),
            &(),
            |b, ()| {
                b.iter(|| {
                    black_box(compute_query_hubness(
                        black_box(&dref),
                        black_box(&qref),
                        10,
                    ))
                });
            },
        );
    }
    gb.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
