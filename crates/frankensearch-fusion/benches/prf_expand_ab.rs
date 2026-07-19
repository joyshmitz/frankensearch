//! Paired A/B for `prf::prf_expand`'s interpolation buffer.
//!
//! `prf_expand` (pseudo-relevance-feedback query expansion) builds a weighted
//! feedback `centroid`, then interpolates `expanded = alpha*original + beta*centroid`
//! and L2-normalizes. The shipping code allocated a **second** `dims`-length vector
//! for `expanded` and zero-initialized it — then immediately overwrote every element
//! — a wasted allocation and zero-init. Folding the interpolation back into the
//! already-owned `centroid` buffer (read-then-write per index) drops both. This
//! bench mirrors the two variants over a realistic fixture (dim 384, 8 feedback
//! embeddings) and asserts a bit-identical `Vec<f32>` before timing:
//!
//! - `two_buffer` : allocate + zero-init a second `expanded` vector (shipping path).
//! - `in_place`   : interpolate into `centroid` and return it (the new path).
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/<lane> \
//!   rch exec -- cargo bench -p frankensearch-fusion --bench prf_expand_ab
//! ```

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_fusion::bench_support::paired_median_ratio;

const DIM: usize = 384;

fn next(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

fn unit_f32(bits: u64) -> f32 {
    ((bits >> 40) as f32) / 16_777_216.0 * 2.0 - 1.0
}

fn make_vec(state: &mut u64) -> Vec<f32> {
    (0..DIM).map(|_| unit_f32(next(state))).collect()
}

/// Shipping path: second `expanded` buffer (alloc + zero-init, then overwrite).
fn two_buffer(original: &[f32], feedback: &[(Vec<f32>, f64)], alpha: f64) -> Option<Vec<f32>> {
    if feedback.is_empty() {
        return None;
    }
    let dims = original.len();
    let alpha = if alpha.is_finite() {
        alpha.clamp(0.5, 1.0)
    } else {
        0.8
    };
    let beta = 1.0 - alpha;
    let total_weight: f64 = feedback.iter().map(|(_, w)| w.max(0.0)).sum();
    if total_weight < f64::EPSILON {
        return None;
    }
    let mut centroid = vec![0.0_f32; dims];
    for (emb, weight) in feedback {
        let w = (weight.max(0.0) / total_weight) as f32;
        let len = emb.len().min(dims);
        for j in 0..len {
            centroid[j] = emb[j].mul_add(w, centroid[j]);
        }
    }
    let alpha_f32 = alpha as f32;
    let beta_f32 = beta as f32;
    let mut expanded = vec![0.0_f32; dims];
    for i in 0..dims {
        expanded[i] = alpha_f32 * original[i] + beta_f32 * centroid[i];
    }
    let norm_sq: f32 = expanded.iter().map(|x| x * x).sum();
    if !norm_sq.is_finite() || norm_sq < f32::EPSILON {
        return None;
    }
    let inv_norm = 1.0 / norm_sq.sqrt();
    for v in &mut expanded {
        *v *= inv_norm;
    }
    Some(expanded)
}

/// New path: interpolate into `centroid` in place (no second buffer).
fn in_place(original: &[f32], feedback: &[(Vec<f32>, f64)], alpha: f64) -> Option<Vec<f32>> {
    if feedback.is_empty() {
        return None;
    }
    let dims = original.len();
    let alpha = if alpha.is_finite() {
        alpha.clamp(0.5, 1.0)
    } else {
        0.8
    };
    let beta = 1.0 - alpha;
    let total_weight: f64 = feedback.iter().map(|(_, w)| w.max(0.0)).sum();
    if total_weight < f64::EPSILON {
        return None;
    }
    let mut centroid = vec![0.0_f32; dims];
    for (emb, weight) in feedback {
        let w = (weight.max(0.0) / total_weight) as f32;
        let len = emb.len().min(dims);
        for j in 0..len {
            centroid[j] = emb[j].mul_add(w, centroid[j]);
        }
    }
    let alpha_f32 = alpha as f32;
    let beta_f32 = beta as f32;
    for i in 0..dims {
        centroid[i] = alpha_f32 * original[i] + beta_f32 * centroid[i];
    }
    let norm_sq: f32 = centroid.iter().map(|x| x * x).sum();
    if !norm_sq.is_finite() || norm_sq < f32::EPSILON {
        return None;
    }
    let inv_norm = 1.0 / norm_sq.sqrt();
    for v in &mut centroid {
        *v *= inv_norm;
    }
    Some(centroid)
}

fn bench(c: &mut Criterion) {
    let mut state = 0x1234_5678_9abc_def0_u64;
    let original = make_vec(&mut state);

    let mut group = c.benchmark_group("prf_expand");
    for &n in &[3usize, 8, 20] {
        let feedback: Vec<(Vec<f32>, f64)> = (0..n)
            .map(|k| (make_vec(&mut state), 1.0 / (k as f64 + 1.0)))
            .collect();
        let alpha = 0.8;

        // Bit-identical parity gate.
        let a = two_buffer(&original, &feedback, alpha).expect("some");
        let b = in_place(&original, &feedback, alpha).expect("some");
        assert_eq!(a.len(), b.len(), "len parity (n={n})");
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.to_bits(), y.to_bits(), "score parity (n={n})");
        }

        group.bench_with_input(BenchmarkId::new("two_buffer", n), &feedback, |bch, fb| {
            bch.iter(|| black_box(two_buffer(black_box(&original), fb, alpha)));
        });
        group.bench_with_input(BenchmarkId::new("in_place", n), &feedback, |bch, fb| {
            bch.iter(|| black_box(in_place(black_box(&original), fb, alpha)));
        });

        // ── DECIDABILITY: alternating-round paired sampler + A/A null control ──
        //
        // The criterion arms above CANNOT decide this lever: criterion runs them as
        // separate benchmarks minutes apart, so worker drift between them is not
        // cancelled. The paired sampler runs both arms in ONE routine in alternating
        // rounds and takes the median per-round ratio; gate on the median against the
        // A/A null's observed spread, not on cv.
        let two_buf = || {
            black_box(two_buffer(
                black_box(&original),
                black_box(&feedback),
                alpha,
            ));
        };
        let in_pl = || {
            black_box(in_place(black_box(&original), black_box(&feedback), alpha));
        };
        let null = paired_median_ratio(41, 8, two_buf, two_buf);
        let lever = paired_median_ratio(41, 8, two_buf, in_pl);
        eprintln!(
            "[null]  prf_expand n {n}: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
            null.median, null.p5, null.p95, null.rounds
        );
        eprintln!(
            "[lever] prf_expand n {n}: in_place median {:.4} p5 {:.4} p95 {:.4} -> {}",
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
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
