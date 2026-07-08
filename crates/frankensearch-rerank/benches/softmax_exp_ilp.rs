//! Attention-softmax EXP-throughput ILP A/B for the native reranker.
//!
//! `softmax_row_fused` (native.rs) is the DOMINANT growing frame of the
//! cross-encoder forward — its own profiling note: "the attention softmax is the
//! dominant growing f32 frame — ~24% of the per-doc forward wall-clock at seq 512
//! (12·S² exp calls)". Its exp+sum loop processes ONE f32x8 lane group per
//! iteration: load 8 → `((x-max)·scale).exp()` → store 8 → `sum_v += e`. The `exp`
//! is a long-latency polynomial (pipelined throughput, high latency); with a single
//! group in flight the exp port is starved — exactly the latency-bound shape the
//! GELU 4-group ILP win (aa11627, ~4-5%) exploited.
//!
//! This is NOT the rejected softmax lever (75f0f8f = the scalar MAX-reduce, which is
//! negligible; that rejection's own conclusion was "softmax is EXP-BOUND"). Here we
//! attack the exp THROUGHPUT: issue 4 independent `exp` per iteration so the core
//! overlaps their latency.
//!
//! Two candidates:
//! * `ilp4_seq` computes e0..e3 (independent exps overlap), then `sum_v += e0;
//!   += e1; += e2; += e3` in the base's exact order with a single accumulator. It is
//!   byte-identical to `base`; only the exp scheduling changes, and parity asserts
//!   max-delta 0.
//! * `ilp4_multi` uses 4 independent sum accumulators, breaking the loop-carried add
//!   chain, then combines them. It is reassociated (~1e-7, below the softmax's own
//!   validated ~1e-6 exp tolerance), ranking-invariant, and not bit-identical. This
//!   tests whether the serial sum chain matters.
//!
//! Shape = full-attention softmax (NH·n rows of width n), swept over n = `s_len`.
//!
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-rerank --features native \
//!     --profile release --bench softmax_exp_ilp
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use wide::f32x8;

const NH: usize = 12;
const SCALE: f32 = 0.176_776_69; // 1 / sqrt(32)

#[inline]
fn f32x8_from_slice(slice: &[f32]) -> f32x8 {
    let mut buf = [0.0f32; 8];
    buf.copy_from_slice(&slice[..8]);
    f32x8::new(buf)
}

/// SHIPPED: one f32x8 group per iter, single in-order `sum_v` (native.rs verbatim).
fn softmax_base(row: &mut [f32], scale: f32) {
    let n = row.len();
    let mut max_raw = f32::NEG_INFINITY;
    for &x in row.iter() {
        max_raw = max_raw.max(x);
    }
    let max_v = f32x8::splat(max_raw);
    let scale_v = f32x8::splat(scale);
    let mut sum_v = f32x8::splat(0.0);
    let mut i = 0;
    while i + 8 <= n {
        let e = ((f32x8_from_slice(&row[i..i + 8]) - max_v) * scale_v).exp();
        row[i..i + 8].copy_from_slice(&e.to_array());
        sum_v += e;
        i += 8;
    }
    let mut sum: f32 = sum_v.to_array().iter().sum();
    while i < n {
        let e = ((row[i] - max_raw) * scale).exp();
        row[i] = e;
        sum += e;
        i += 1;
    }
    let inv = 1.0 / sum;
    for x in row.iter_mut() {
        *x *= inv;
    }
}

/// CANDIDATE A (BYTE-IDENTICAL): 4 independent exps/iter, single in-order `sum_v`.
fn softmax_ilp4_seq(row: &mut [f32], scale: f32) {
    let n = row.len();
    let mut max_raw = f32::NEG_INFINITY;
    for &x in row.iter() {
        max_raw = max_raw.max(x);
    }
    let max_v = f32x8::splat(max_raw);
    let scale_v = f32x8::splat(scale);
    let mut sum_v = f32x8::splat(0.0);
    let mut i = 0;
    while i + 32 <= n {
        let e0 = ((f32x8_from_slice(&row[i..i + 8]) - max_v) * scale_v).exp();
        let e1 = ((f32x8_from_slice(&row[i + 8..i + 16]) - max_v) * scale_v).exp();
        let e2 = ((f32x8_from_slice(&row[i + 16..i + 24]) - max_v) * scale_v).exp();
        let e3 = ((f32x8_from_slice(&row[i + 24..i + 32]) - max_v) * scale_v).exp();
        row[i..i + 8].copy_from_slice(&e0.to_array());
        row[i + 8..i + 16].copy_from_slice(&e1.to_array());
        row[i + 16..i + 24].copy_from_slice(&e2.to_array());
        row[i + 24..i + 32].copy_from_slice(&e3.to_array());
        // Same order + single accumulator as `base` → bit-identical sum_v.
        sum_v += e0;
        sum_v += e1;
        sum_v += e2;
        sum_v += e3;
        i += 32;
    }
    while i + 8 <= n {
        let e = ((f32x8_from_slice(&row[i..i + 8]) - max_v) * scale_v).exp();
        row[i..i + 8].copy_from_slice(&e.to_array());
        sum_v += e;
        i += 8;
    }
    let mut sum: f32 = sum_v.to_array().iter().sum();
    while i < n {
        let e = ((row[i] - max_raw) * scale).exp();
        row[i] = e;
        sum += e;
        i += 1;
    }
    let inv = 1.0 / sum;
    for x in row.iter_mut() {
        *x *= inv;
    }
}

/// CANDIDATE B (reassociated): 4 independent exps AND 4 independent sum accumulators.
fn softmax_ilp4_multi(row: &mut [f32], scale: f32) {
    let n = row.len();
    let mut max_raw = f32::NEG_INFINITY;
    for &x in row.iter() {
        max_raw = max_raw.max(x);
    }
    let max_v = f32x8::splat(max_raw);
    let scale_v = f32x8::splat(scale);
    let mut s0 = f32x8::splat(0.0);
    let mut s1 = f32x8::splat(0.0);
    let mut s2 = f32x8::splat(0.0);
    let mut s3 = f32x8::splat(0.0);
    let mut i = 0;
    while i + 32 <= n {
        let e0 = ((f32x8_from_slice(&row[i..i + 8]) - max_v) * scale_v).exp();
        let e1 = ((f32x8_from_slice(&row[i + 8..i + 16]) - max_v) * scale_v).exp();
        let e2 = ((f32x8_from_slice(&row[i + 16..i + 24]) - max_v) * scale_v).exp();
        let e3 = ((f32x8_from_slice(&row[i + 24..i + 32]) - max_v) * scale_v).exp();
        row[i..i + 8].copy_from_slice(&e0.to_array());
        row[i + 8..i + 16].copy_from_slice(&e1.to_array());
        row[i + 16..i + 24].copy_from_slice(&e2.to_array());
        row[i + 24..i + 32].copy_from_slice(&e3.to_array());
        s0 += e0;
        s1 += e1;
        s2 += e2;
        s3 += e3;
        i += 32;
    }
    let mut sum_v = (s0 + s1) + (s2 + s3);
    while i + 8 <= n {
        let e = ((f32x8_from_slice(&row[i..i + 8]) - max_v) * scale_v).exp();
        row[i..i + 8].copy_from_slice(&e.to_array());
        sum_v += e;
        i += 8;
    }
    let mut sum: f32 = sum_v.to_array().iter().sum();
    while i < n {
        let e = ((row[i] - max_raw) * scale).exp();
        row[i] = e;
        sum += e;
        i += 1;
    }
    let inv = 1.0 / sum;
    for x in row.iter_mut() {
        *x *= inv;
    }
}

fn softmax_rows(data: &mut [f32], n: usize, scale: f32, kind: u8) {
    for row in data.chunks_exact_mut(n) {
        match kind {
            0 => softmax_base(row, scale),
            1 => softmax_ilp4_seq(row, scale),
            _ => softmax_ilp4_multi(row, scale),
        }
    }
}

fn fixture(rows: usize, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * n);
    for i in 0..rows * n {
        out.push(((i.wrapping_mul(31) % 257) as f32 - 128.0) * 0.02);
    }
    out
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax_exp_ilp");
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_millis(1000));

    for &n in &[64usize, 128, 256, 512] {
        let rows = NH * n; // full-attention softmax shape (12·S² exps)
        let base_fix = fixture(rows, n);

        // Parity: ilp4_seq is byte-identical (untouched reduction); ilp4_multi is
        // reassociated (must stay within the softmax's validated ~1e-6 tolerance).
        let mut a = base_fix.clone();
        let mut bseq = base_fix.clone();
        let mut bmul = base_fix.clone();
        softmax_rows(&mut a, n, SCALE, 0);
        softmax_rows(&mut bseq, n, SCALE, 1);
        softmax_rows(&mut bmul, n, SCALE, 2);
        let dseq = a
            .iter()
            .zip(&bseq)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let dmul = a
            .iter()
            .zip(&bmul)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            dseq == 0.0,
            "ilp4_seq diverged from base by {dseq} (n={n}) — must be byte-identical"
        );
        assert!(
            dmul < 1e-6,
            "ilp4_multi diverged from base by {dmul} (n={n}) — exceeds softmax ~1e-6 tolerance"
        );

        // No per-iter reset: softmax does data-independent WORK and stays finite.
        for (kind, name) in [(0u8, "base"), (1u8, "ilp4_seq"), (2u8, "ilp4_multi")] {
            group.bench_with_input(BenchmarkId::new(name, n), &base_fix, |bn, base| {
                let mut buf = base.clone();
                bn.iter(|| {
                    softmax_rows(black_box(&mut buf), n, SCALE, kind);
                    black_box(&buf);
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
