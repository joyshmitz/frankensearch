//! Latency A/B for the `compute_query_hubness` inner kernel and outer loop.
//!
//! [`compute_query_hubness`] builds the per-doc query-hubness table `r_d` — for every doc, the
//! mean cosine to its `kq` nearest *sample queries* — an **O(docs · queries · dim)** batch that the
//! `PERF_LEDGER` clocks at ~109 ms / 696 ms @ 2000×200 / 5000×500 docs×queries. Two independent
//! levers, measured here in one binary on one worker so the arms share a build and a machine:
//!
//! 1. **Inner dot (LANDED: `simd`).** The ORIGINAL was a scalar `iter().zip().map(|(x,y)| x*y).sum()`:
//!    one f32 accumulator, a serial add chain LLVM cannot reassociate without fast-math,
//!    latency-bound at ~1 add / 4–5 cyc no matter how wide the multiplies vectorize. Two candidates:
//!    - `multiacc` — eight independent f32 accumulators, relying on LLVM's SLP vectorizer.
//!      **REJECTED: only 2.53–2.59×**, i.e. 4.4× slower than simply reusing the kernel below.
//!    - `simd` — [`frankensearch_index::dot_product_f32_f32`], the **already-shipped** hand-written
//!      AVX2 kernel (4×`f32x8` = 32 lanes of accumulator ILP, `wide` fallback off-x86). fusion
//!      already depends on frankensearch-index, so this arm is pure reuse, not new code.
//!      **KEPT: 11.34× on the dot, 10.89×/11.28× on the builder.** `multiacc` is retained as a
//!      timed arm so the rejection stays reproducible rather than merely asserted.
//! 2. **Outer loop (LANDED: `simd_par`).** Each doc's `r_d` is independent, so the
//!    `doc_vecs.iter().map(..)` is a rayon `par_iter()` above `PARALLEL_THRESHOLD` dot products of
//!    total work. An indexed parallel iterator preserves order and no element's arithmetic changes,
//!    so this is **bit-identical**, not merely ULP-equal — the `simd_par` vs `simd` and `simd_par`
//!    vs `shipped` gates below assert exactly that. Measured **5.85× on top of `simd`**
//!    (2.0565 ms → 351.66 µs @ 1000×100×384). The `shipped` arm times the real
//!    [`compute_query_hubness`] entry point, which takes the rayon branch at both shapes here.
//!
//! Parity gates run before any timing:
//! - `simd` mirror is **bit-identical** to the shipped [`compute_query_hubness`] (mirror faithful —
//!   this is the gate that catches the bench drifting from what `hubness::dot` actually calls).
//! - every candidate is within `max Δr_d < 1e-4` of the ORIGINAL scalar (selection of the kq
//!   nearest queries unchanged; `r_d` is an approximate demotion statistic scaled by β≈0.2).
//! - `simd_par` is **bit-identical** to `simd` (rayon changes scheduling, not arithmetic).
//!
//! ORIG is measured **first and last** (`ORIG_scalar2`) to bracket Criterion's ordering bias.
//!
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/hub-dot \
//!   rch exec -- cargo bench -p frankensearch-fusion --bench hubness_dot_ab
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_fusion::bench_support::paired_median_ratio;
use frankensearch_fusion::compute_query_hubness;
use frankensearch_index::dot_product_f32_f32;
use rayon::prelude::*;

/// ORIG: the pre-optimization scalar dot — one f32 accumulator, a serial add chain.
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Candidate 1 — mirror of the currently-committed `dot`: eight independent f32 accumulators.
fn dot_multiacc(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let a = &a[..len];
    let b = &b[..len];
    let mut acc = [0.0f32; 8];
    let chunks = len / 8;
    for c in 0..chunks {
        let i = c * 8;
        acc[0] += a[i] * b[i];
        acc[1] += a[i + 1] * b[i + 1];
        acc[2] += a[i + 2] * b[i + 2];
        acc[3] += a[i + 3] * b[i + 3];
        acc[4] += a[i + 4] * b[i + 4];
        acc[5] += a[i + 5] * b[i + 5];
        acc[6] += a[i + 6] * b[i + 6];
        acc[7] += a[i + 7] * b[i + 7];
    }
    let mut sum = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for i in (chunks * 8)..len {
        sum += a[i] * b[i];
    }
    sum
}

/// Candidate 2 — the shipped AVX2 kernel from frankensearch-index. Slices to the common length so
/// the dimension-mismatch error is unreachable (the ORIGINAL `zip` truncated the same way).
fn dot_simd(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    dot_product_f32_f32(&a[..len], &b[..len]).unwrap_or(0.0)
}

/// Serial driver mirroring `compute_query_hubness`, parameterized by the dot kernel.
fn hubness_with(
    doc_vecs: &[&[f32]],
    query_sample: &[&[f32]],
    kq: usize,
    dotf: fn(&[f32], &[f32]) -> f32,
) -> Vec<f32> {
    if query_sample.is_empty() || kq == 0 {
        return vec![0.0; doc_vecs.len()];
    }
    let k = kq.min(query_sample.len());
    doc_vecs
        .iter()
        .map(|d| reduce_doc(d, query_sample, k, dotf))
        .collect()
}

/// Rayon driver: identical per-doc arithmetic, order preserved by the indexed parallel iterator.
fn hubness_par(
    doc_vecs: &[&[f32]],
    query_sample: &[&[f32]],
    kq: usize,
    dotf: fn(&[f32], &[f32]) -> f32,
) -> Vec<f32> {
    if query_sample.is_empty() || kq == 0 {
        return vec![0.0; doc_vecs.len()];
    }
    let k = kq.min(query_sample.len());
    doc_vecs
        .par_iter()
        .map(|d| reduce_doc(d, query_sample, k, dotf))
        .collect()
}

/// One doc's `r_d`: mean cosine to its `k` nearest sample queries.
fn reduce_doc(
    d: &[f32],
    query_sample: &[&[f32]],
    k: usize,
    dotf: fn(&[f32], &[f32]) -> f32,
) -> f32 {
    let mut sims: Vec<f32> = query_sample.iter().map(|q| dotf(d, q)).collect();
    let n = sims.len();
    let (_, pivot, top) = sims.select_nth_unstable_by(n - k, |a, b| a.total_cmp(b));
    let sum: f32 = *pivot + top.iter().sum::<f32>();
    sum / k as f32
}

/// Deterministic L2-normalized f32 vectors (xorshift, no `rand` dep). `dim`-wide, `n` of them.
fn gen_vecs(n: usize, dim: usize, mut state: u64) -> Vec<Vec<f32>> {
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 40) as f32 / (1u32 << 23) as f32) - 1.0
    };
    (0..n)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| next()).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for x in &mut v {
                *x /= norm;
            }
            v
        })
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn assert_bit_identical(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}: doc {i} ({x} vs {y})");
    }
}

fn bench(c: &mut Criterion) {
    // ── Micro: the dot kernel in isolation on one realistic 384-dim pair ──────────────────────
    {
        let lhs = gen_vecs(1, 384, 0x1234_5678_9abc_def0).pop().unwrap();
        let rhs = gen_vecs(1, 384, 0x0fed_cba9_8765_4321).pop().unwrap();
        let want = dot_scalar(&lhs, &rhs);
        for (name, got) in [
            ("multiacc", dot_multiacc(&lhs, &rhs)),
            ("simd", dot_simd(&lhs, &rhs)),
        ] {
            assert!(
                (want - got).abs() < 1e-4,
                "dot micro parity {name}: scalar {want} vs {got}"
            );
        }

        let mut group = c.benchmark_group("hubness_dot_micro");
        group.sample_size(100);
        group.warm_up_time(Duration::from_millis(300));
        group.measurement_time(Duration::from_millis(1500));
        group.bench_function("ORIG_scalar", |bch| {
            bch.iter(|| black_box(dot_scalar(black_box(&lhs), black_box(&rhs))));
        });
        group.bench_function("multiacc", |bch| {
            bch.iter(|| black_box(dot_multiacc(black_box(&lhs), black_box(&rhs))));
        });
        group.bench_function("simd", |bch| {
            bch.iter(|| black_box(dot_simd(black_box(&lhs), black_box(&rhs))));
        });
        group.bench_function("ORIG_scalar2", |bch| {
            bch.iter(|| black_box(dot_scalar(black_box(&lhs), black_box(&rhs))));
        });
        group.finish();

        // ── DECIDABILITY: alternating-round paired sampler + A/A null control ──
        //
        // The criterion arms above run as separate benchmarks minutes apart, so worker
        // drift between them is not cancelled. The paired sampler runs both arms in ONE
        // routine in alternating rounds; gate on the median against the A/A null's
        // observed spread. Base is the ORIGINAL scalar dot; one null+lever pair per
        // candidate kernel.
        let scalar = || {
            black_box(dot_scalar(black_box(&lhs), black_box(&rhs)));
        };
        let multiacc = || {
            black_box(dot_multiacc(black_box(&lhs), black_box(&rhs)));
        };
        let null = paired_median_ratio(41, 8, scalar, scalar);
        let lever = paired_median_ratio(41, 8, scalar, multiacc);
        eprintln!(
            "[null]  hubness_dot_micro d384: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
            null.median, null.p5, null.p95, null.rounds
        );
        eprintln!(
            "[lever] hubness_dot_micro d384: multiacc/scalar median {:.4} p5 {:.4} p95 {:.4} -> {}",
            lever.median,
            lever.p5,
            lever.p95,
            if lever.decidable_against(&null) {
                "DECIDABLE"
            } else {
                "INSIDE NULL FLOOR (not decidable)"
            }
        );
        let scalar = || {
            black_box(dot_scalar(black_box(&lhs), black_box(&rhs)));
        };
        let simd = || {
            black_box(dot_simd(black_box(&lhs), black_box(&rhs)));
        };
        let null = paired_median_ratio(41, 8, scalar, scalar);
        let lever = paired_median_ratio(41, 8, scalar, simd);
        eprintln!(
            "[null]  hubness_dot_micro d384: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
            null.median, null.p5, null.p95, null.rounds
        );
        eprintln!(
            "[lever] hubness_dot_micro d384: simd/scalar median {:.4} p5 {:.4} p95 {:.4} -> {}",
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

    // ── Macro: the full O(docs·queries·dim) hubness build ─────────────────────────────────────
    let kq = 10;
    for &(docs, queries, dim) in &[(1000usize, 100usize, 384usize), (500, 200, 384)] {
        let dv = gen_vecs(docs, dim, 0xdead_beef_0000_0001);
        let qv = gen_vecs(queries, dim, 0xf00d_face_0000_0002);
        let doc_vecs: Vec<&[f32]> = dv.iter().map(Vec::as_slice).collect();
        let query_sample: Vec<&[f32]> = qv.iter().map(Vec::as_slice).collect();

        let r_scalar = hubness_with(&doc_vecs, &query_sample, kq, dot_scalar);
        let r_multi = hubness_with(&doc_vecs, &query_sample, kq, dot_multiacc);
        let r_simd = hubness_with(&doc_vecs, &query_sample, kq, dot_simd);
        let r_simd_par = hubness_par(&doc_vecs, &query_sample, kq, dot_simd);
        let r_ship = compute_query_hubness(&doc_vecs, &query_sample, kq);

        // `hubness::dot` delegates to the same AVX2 kernel as `dot_simd`, and above
        // PARALLEL_THRESHOLD dot products the shipped builder takes its rayon branch — so the
        // shipped result must be bit-identical to BOTH mirrors. These gates catch the bench
        // drifting from src, and pin rayon to scheduling-only effects.
        assert_bit_identical(&r_simd, &r_ship, "simd mirror vs shipped");
        assert_bit_identical(&r_simd_par, &r_ship, "simd_par mirror vs shipped");
        // rayon reorders scheduling, never arithmetic.
        assert_bit_identical(&r_simd_par, &r_simd, "simd_par vs simd");
        // Every reassociated candidate leaves the kq-nearest selection intact.
        for (name, cand) in [("multiacc", &r_multi), ("simd", &r_simd)] {
            let d = max_abs_diff(&r_scalar, cand);
            assert!(
                d < 1e-4,
                "hubness {docs}x{queries}x{dim} {name}: max Δr_d {d} ≥ 1e-4"
            );
        }

        let mut g = c.benchmark_group("hubness_build");
        // The rayon arms are ~380 µs, short enough that fork/join jitter dominates a criterion
        // sample built from few iterations. Criterion ramps iterations per sample to fill
        // `measurement_time`, so a longer window (not a larger `sample_size`, which *shrinks*
        // per-sample iterations) is what drives cv_pct down. At 3 s the `shipped` arm measured
        // cv 7.3%; 12 s puts it under the 5% keep-gate.
        g.sample_size(30);
        g.warm_up_time(Duration::from_millis(2000));
        g.measurement_time(Duration::from_millis(12_000));
        let id = format!("{docs}x{queries}x{dim}");
        g.bench_with_input(BenchmarkId::new("ORIG_scalar", &id), &(), |bch, ()| {
            bch.iter(|| {
                black_box(hubness_with(
                    black_box(&doc_vecs),
                    black_box(&query_sample),
                    kq,
                    dot_scalar,
                ))
            });
        });
        g.bench_with_input(BenchmarkId::new("multiacc", &id), &(), |bch, ()| {
            bch.iter(|| {
                black_box(hubness_with(
                    black_box(&doc_vecs),
                    black_box(&query_sample),
                    kq,
                    dot_multiacc,
                ))
            });
        });
        g.bench_with_input(BenchmarkId::new("simd", &id), &(), |bch, ()| {
            bch.iter(|| {
                black_box(hubness_with(
                    black_box(&doc_vecs),
                    black_box(&query_sample),
                    kq,
                    dot_simd,
                ))
            });
        });
        g.bench_with_input(BenchmarkId::new("simd_par", &id), &(), |bch, ()| {
            bch.iter(|| {
                black_box(hubness_par(
                    black_box(&doc_vecs),
                    black_box(&query_sample),
                    kq,
                    dot_simd,
                ))
            });
        });
        // The shipped public fn. Both shapes here exceed PARALLEL_THRESHOLD dot products, so this
        // arm exercises its rayon branch and should track `simd_par`. Timing the real entry point
        // — not just a mirror — is what the ledger ratio is quoted from.
        g.bench_with_input(BenchmarkId::new("shipped", &id), &(), |bch, ()| {
            bch.iter(|| {
                black_box(compute_query_hubness(
                    black_box(&doc_vecs),
                    black_box(&query_sample),
                    kq,
                ))
            });
        });
        g.bench_with_input(BenchmarkId::new("ORIG_scalar2", &id), &(), |bch, ()| {
            bch.iter(|| {
                black_box(hubness_with(
                    black_box(&doc_vecs),
                    black_box(&query_sample),
                    kq,
                    dot_scalar,
                ))
            });
        });
        g.finish();

        // ── DECIDABILITY: alternating-round paired sampler + A/A null control ──
        //
        // The criterion arms above run as separate benchmarks minutes apart, so worker
        // drift between them is not cancelled. The paired sampler runs both arms in ONE
        // routine in alternating rounds; gate on the median against the A/A null's
        // observed spread. One null+lever pair per comparison: the inner-dot levers
        // (multiacc, simd vs scalar), the outer-loop lever (simd_par vs simd), and the
        // shipped entry point vs its simd_par mirror.
        let scalar_build = || {
            black_box(hubness_with(
                black_box(&doc_vecs),
                black_box(&query_sample),
                kq,
                dot_scalar,
            ));
        };
        let multiacc_build = || {
            black_box(hubness_with(
                black_box(&doc_vecs),
                black_box(&query_sample),
                kq,
                dot_multiacc,
            ));
        };
        let null = paired_median_ratio(41, 8, scalar_build, scalar_build);
        let lever = paired_median_ratio(41, 8, scalar_build, multiacc_build);
        eprintln!(
            "[null]  hubness_build {id}: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
            null.median, null.p5, null.p95, null.rounds
        );
        eprintln!(
            "[lever] hubness_build {id}: multiacc/scalar median {:.4} p5 {:.4} p95 {:.4} -> {}",
            lever.median,
            lever.p5,
            lever.p95,
            if lever.decidable_against(&null) {
                "DECIDABLE"
            } else {
                "INSIDE NULL FLOOR (not decidable)"
            }
        );
        let scalar_build = || {
            black_box(hubness_with(
                black_box(&doc_vecs),
                black_box(&query_sample),
                kq,
                dot_scalar,
            ));
        };
        let simd_build = || {
            black_box(hubness_with(
                black_box(&doc_vecs),
                black_box(&query_sample),
                kq,
                dot_simd,
            ));
        };
        let null = paired_median_ratio(41, 8, scalar_build, scalar_build);
        let lever = paired_median_ratio(41, 8, scalar_build, simd_build);
        eprintln!(
            "[null]  hubness_build {id}: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
            null.median, null.p5, null.p95, null.rounds
        );
        eprintln!(
            "[lever] hubness_build {id}: simd/scalar median {:.4} p5 {:.4} p95 {:.4} -> {}",
            lever.median,
            lever.p5,
            lever.p95,
            if lever.decidable_against(&null) {
                "DECIDABLE"
            } else {
                "INSIDE NULL FLOOR (not decidable)"
            }
        );
        let simd_build = || {
            black_box(hubness_with(
                black_box(&doc_vecs),
                black_box(&query_sample),
                kq,
                dot_simd,
            ));
        };
        let simd_par_build = || {
            black_box(hubness_par(
                black_box(&doc_vecs),
                black_box(&query_sample),
                kq,
                dot_simd,
            ));
        };
        let null = paired_median_ratio(41, 8, simd_build, simd_build);
        let lever = paired_median_ratio(41, 8, simd_build, simd_par_build);
        eprintln!(
            "[null]  hubness_build {id}: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
            null.median, null.p5, null.p95, null.rounds
        );
        eprintln!(
            "[lever] hubness_build {id}: simd_par/simd median {:.4} p5 {:.4} p95 {:.4} -> {}",
            lever.median,
            lever.p5,
            lever.p95,
            if lever.decidable_against(&null) {
                "DECIDABLE"
            } else {
                "INSIDE NULL FLOOR (not decidable)"
            }
        );
        let simd_par_build = || {
            black_box(hubness_par(
                black_box(&doc_vecs),
                black_box(&query_sample),
                kq,
                dot_simd,
            ));
        };
        let shipped_build = || {
            black_box(compute_query_hubness(
                black_box(&doc_vecs),
                black_box(&query_sample),
                kq,
            ));
        };
        let null = paired_median_ratio(41, 8, simd_par_build, simd_par_build);
        let lever = paired_median_ratio(41, 8, simd_par_build, shipped_build);
        eprintln!(
            "[null]  hubness_build {id}: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
            null.median, null.p5, null.p95, null.rounds
        );
        eprintln!(
            "[lever] hubness_build {id}: shipped/simd_par median {:.4} p5 {:.4} p95 {:.4} -> {}",
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
}

criterion_group!(benches, bench);
criterion_main!(benches);
