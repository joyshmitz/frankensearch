//! FFN-GELU instruction-level-parallelism A/B for the native reranker.
//!
//! `fast_gelu_inplace` (native.rs) is a MEASURED ~10-14% of the cross-encoder
//! forward — a wide `[total, 1536]` elementwise exact-GELU over every FFN layer.
//! Its inner loop processes ONE `f32x8` lane group per iteration: load 8 → the full
//! `gelu_vec8` dependency chain (`z → |z| → t=1/(1+c·|z|) → 5-term Horner erf poly →
//! exp(-z²) → copysign → 0.5·x·(1+erf)`) → store 8. That chain is LATENCY-bound
//! (each op waits on the prior; the `exp` alone is a long polynomial), and the
//! reciprocal + exp have high latency but pipelined throughput — so a single group
//! in flight leaves most of the FMA/EXP units idle.
//!
//! Consecutive groups are FULLY INDEPENDENT (GELU is a pure elementwise map, no
//! cross-lane reduction), so issuing N groups' `gelu_vec8` back-to-back lets the CPU
//! overlap their latency chains → converts latency-bound → throughput-bound. This is
//! the same ILP lever that won 1.45× on the f16 dot (single- → 4-accumulator), but
//! here it is EXACT AND BYTE-IDENTICAL: no reduction is reassociated, each 8-element
//! group gets the identical `gelu_vec8` at the identical position (parity asserts
//! max-delta 0), unlike the softmax max-reduce (which was reassociation + turned out
//! exp-bound-on-the-reduce; this is the exp/poly THROUGHPUT, never before tested).
//!
//! Arms: `base` (= shipped, 1 group/iter) vs `ilp2`, `ilp4` (2/4 independent groups
//! interleaved per iter). Swept over buffer sizes mapping to FFN GELU call widths
//! (multiples of INTER=1536: 1, 8, 64, 512 FFN rows).
//!
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-rerank --features native \
//!     --profile release --bench gelu_ilp
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use wide::f32x8;

const INTER: usize = 1536; // FFN intermediate width (4 * H, H=384)

#[inline]
fn f32x8_from_slice(slice: &[f32]) -> f32x8 {
    let mut buf = [0.0f32; 8];
    buf.copy_from_slice(&slice[..8]);
    f32x8::new(buf)
}

/// Faithful copy of native.rs `gelu_vec8` (A–S 7.1.26 erf), identical constants.
#[inline]
fn gelu_vec8(x: f32x8) -> f32x8 {
    const C: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let one = f32x8::splat(1.0);
    let z = x * f32x8::splat(C);
    let az = z.abs();
    let t = one / (one + f32x8::splat(0.327_591_1) * az);
    let a1 = f32x8::splat(0.254_829_6);
    let a2 = f32x8::splat(-0.284_496_73);
    let a3 = f32x8::splat(1.421_413_7);
    let a4 = f32x8::splat(-1.453_152);
    let a5 = f32x8::splat(1.061_405_4);
    let poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
    let erf_abs = one - poly * (-(z * z)).exp();
    let erf = erf_abs.copysign(z);
    f32x8::splat(0.5) * x * (one + erf)
}

#[inline]
fn gelu_scalar(x: f32) -> f32 {
    const C: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let z = x * C;
    let az = z.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * az);
    let poly = t
        * (0.254_829_6
            + t * (-0.284_496_73 + t * (1.421_413_7 + t * (-1.453_152 + t * 1.061_405_4))));
    let erf = (1.0 - poly * (-(z * z)).exp()).copysign(z);
    0.5 * x * (1.0 + erf)
}

/// SHIPPED: one f32x8 group per iteration (latency-bound per-group chain).
fn gelu_base(chunk: &mut [f32]) {
    let n = chunk.len();
    let mut i = 0;
    while i + 8 <= n {
        let g = gelu_vec8(f32x8_from_slice(&chunk[i..i + 8]));
        chunk[i..i + 8].copy_from_slice(&g.to_array());
        i += 8;
    }
    while i < n {
        chunk[i] = gelu_scalar(chunk[i]);
        i += 1;
    }
}

/// CANDIDATE: 2 independent groups interleaved per iteration.
fn gelu_ilp2(chunk: &mut [f32]) {
    let n = chunk.len();
    let mut i = 0;
    while i + 16 <= n {
        let v0 = f32x8_from_slice(&chunk[i..i + 8]);
        let v1 = f32x8_from_slice(&chunk[i + 8..i + 16]);
        let g0 = gelu_vec8(v0);
        let g1 = gelu_vec8(v1);
        chunk[i..i + 8].copy_from_slice(&g0.to_array());
        chunk[i + 8..i + 16].copy_from_slice(&g1.to_array());
        i += 16;
    }
    while i + 8 <= n {
        let g = gelu_vec8(f32x8_from_slice(&chunk[i..i + 8]));
        chunk[i..i + 8].copy_from_slice(&g.to_array());
        i += 8;
    }
    while i < n {
        chunk[i] = gelu_scalar(chunk[i]);
        i += 1;
    }
}

/// CANDIDATE: 4 independent groups interleaved per iteration.
fn gelu_ilp4(chunk: &mut [f32]) {
    let n = chunk.len();
    let mut i = 0;
    while i + 32 <= n {
        let v0 = f32x8_from_slice(&chunk[i..i + 8]);
        let v1 = f32x8_from_slice(&chunk[i + 8..i + 16]);
        let v2 = f32x8_from_slice(&chunk[i + 16..i + 24]);
        let v3 = f32x8_from_slice(&chunk[i + 24..i + 32]);
        let g0 = gelu_vec8(v0);
        let g1 = gelu_vec8(v1);
        let g2 = gelu_vec8(v2);
        let g3 = gelu_vec8(v3);
        chunk[i..i + 8].copy_from_slice(&g0.to_array());
        chunk[i + 8..i + 16].copy_from_slice(&g1.to_array());
        chunk[i + 16..i + 24].copy_from_slice(&g2.to_array());
        chunk[i + 24..i + 32].copy_from_slice(&g3.to_array());
        i += 32;
    }
    while i + 8 <= n {
        let g = gelu_vec8(f32x8_from_slice(&chunk[i..i + 8]));
        chunk[i..i + 8].copy_from_slice(&g.to_array());
        i += 8;
    }
    while i < n {
        chunk[i] = gelu_scalar(chunk[i]);
        i += 1;
    }
}

fn fixture(n: usize) -> Vec<f32> {
    // GELU input = post-int8-linear activations: roughly N(0, ~2), covering the
    // erf transition and both tails so exp/poly cost is representative.
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(((i.wrapping_mul(2_654_435_761) % 4096) as f32 / 4096.0 - 0.5) * 8.0);
    }
    out
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("gelu_ilp");
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_millis(1000));

    // FFN GELU widths: 1, 8, 64, 512 FFN rows (× INTER=1536). All multiples of 32.
    for &rows in &[1usize, 8, 64, 512] {
        let n = rows * INTER;
        let base_fix = fixture(n);

        // Parity: ILP variants are byte-identical to base (same op, same position).
        let mut a = base_fix.clone();
        let mut b2 = base_fix.clone();
        let mut b4 = base_fix.clone();
        gelu_base(&mut a);
        gelu_ilp2(&mut b2);
        gelu_ilp4(&mut b4);
        let d2 = a
            .iter()
            .zip(&b2)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let d4 = a
            .iter()
            .zip(&b4)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            d2 == 0.0,
            "ilp2 diverged from base by {d2} (rows={rows}) — must be byte-identical"
        );
        assert!(
            d4 == 0.0,
            "ilp4 diverged from base by {d4} (rows={rows}) — must be byte-identical"
        );

        // No per-iter reset: GELU is a pure elementwise map doing data-independent
        // WORK (poly + exp evaluated regardless of value) and its output stays finite,
        // so re-running on the evolving buffer measures the kernel without an in-loop copy.
        group.bench_with_input(BenchmarkId::new("base", rows), &base_fix, |bn, base| {
            let mut buf = base.clone();
            bn.iter(|| {
                gelu_base(black_box(&mut buf));
                black_box(&buf);
            });
        });
        group.bench_with_input(BenchmarkId::new("ilp2", rows), &base_fix, |bn, base| {
            let mut buf = base.clone();
            bn.iter(|| {
                gelu_ilp2(black_box(&mut buf));
                black_box(&buf);
            });
        });
        group.bench_with_input(BenchmarkId::new("ilp4", rows), &base_fix, |bn, base| {
            let mut buf = base.clone();
            bn.iter(|| {
                gelu_ilp4(black_box(&mut buf));
                black_box(&buf);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
