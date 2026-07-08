//! Attention softmax SIMD max-reduce A/B for the native reranker.
//!
//! `softmax_row_fused` (native.rs) does the exp+sum pass in SIMD but the initial
//! max-reduce `for &x in row { max = max.max(x) }` is SCALAR — LLVM can't
//! auto-vectorize an f32 max-reduction (it needs reassociation it won't do without
//! fast-math). The max pass is a full row-length scalar loop (~20% of a softmax row,
//! comparable to the SIMD exp's effective cost), and softmax runs in every attention
//! (CLS + the earlier full m×n layers) × top-50 docs. Replace the scalar max-reduce
//! with an f32x8 `max` accumulate + horizontal fold (once per row, not per element).
//!
//! EXACT: max is associative/commutative for finite scores (attention logits are
//! finite), so the SIMD reduction is bit-identical to the scalar one (parity asserts
//! max-delta 0). The rest of the softmax is unchanged.
//!
//! Arms: `scalar_max` (= shipped) vs `simd_max`. Shape = full-attention softmax
//! (NH·n rows of width n), swept over n; this is where the max-reduce cost is largest.
//!
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-rerank --features native \
//!     --profile release --bench softmax_simdmax
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use wide::f32x8;

const NH: usize = 12;
const SCALE: f32 = 0.176_776_69;

#[inline]
fn f32x8_from_slice(slice: &[f32]) -> f32x8 {
    let mut buf = [0.0f32; 8];
    buf.copy_from_slice(&slice[..8]);
    f32x8::new(buf)
}

/// SHIPPED: scalar max-reduce + SIMD exp/sum + normalize.
fn softmax_row_scalar_max(row: &mut [f32], scale: f32) {
    let mut max_raw = f32::NEG_INFINITY;
    for &x in row.iter() {
        max_raw = max_raw.max(x);
    }
    let max_v = f32x8::splat(max_raw);
    let scale_v = f32x8::splat(scale);
    let mut sum_v = f32x8::splat(0.0);
    let mut i = 0;
    while i + 8 <= row.len() {
        let e = ((f32x8_from_slice(&row[i..i + 8]) - max_v) * scale_v).exp();
        row[i..i + 8].copy_from_slice(&e.to_array());
        sum_v += e;
        i += 8;
    }
    let mut sum = sum_v.reduce_add();
    while i < row.len() {
        let e = ((row[i] - max_raw) * scale).exp();
        row[i] = e;
        sum += e;
        i += 1;
    }
    let inv = 1.0 / sum;
    for x in row {
        *x *= inv;
    }
}

/// CANDIDATE: f32x8 max-reduce (one horizontal fold per row) + same exp/sum/normalize.
fn softmax_row_simd_max(row: &mut [f32], scale: f32) {
    let mut max_v8 = f32x8::splat(f32::NEG_INFINITY);
    let mut i = 0;
    while i + 8 <= row.len() {
        max_v8 = max_v8.max(f32x8_from_slice(&row[i..i + 8]));
        i += 8;
    }
    let mut max_raw = max_v8
        .to_array()
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    while i < row.len() {
        max_raw = max_raw.max(row[i]);
        i += 1;
    }
    let max_v = f32x8::splat(max_raw);
    let scale_v = f32x8::splat(scale);
    let mut sum_v = f32x8::splat(0.0);
    let mut i = 0;
    while i + 8 <= row.len() {
        let e = ((f32x8_from_slice(&row[i..i + 8]) - max_v) * scale_v).exp();
        row[i..i + 8].copy_from_slice(&e.to_array());
        sum_v += e;
        i += 8;
    }
    let mut sum = sum_v.reduce_add();
    while i < row.len() {
        let e = ((row[i] - max_raw) * scale).exp();
        row[i] = e;
        sum += e;
        i += 1;
    }
    let inv = 1.0 / sum;
    for x in row {
        *x *= inv;
    }
}

fn softmax_rows(data: &mut [f32], n: usize, scale: f32, simd: bool) {
    for row in data.chunks_exact_mut(n) {
        if simd {
            softmax_row_simd_max(row, scale);
        } else {
            softmax_row_scalar_max(row, scale);
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
    let mut group = c.benchmark_group("softmax_simdmax");
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_millis(800));

    for &n in &[64usize, 128, 256, 512] {
        let rows = NH * n; // full-attention softmax shape
        let base = fixture(rows, n);

        // parity: SIMD max-reduce is bit-identical to scalar for finite inputs.
        let mut a = base.clone();
        let mut b = base.clone();
        softmax_rows(&mut a, n, SCALE, false);
        softmax_rows(&mut b, n, SCALE, true);
        let md = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            md == 0.0,
            "simd max-reduce diverged from scalar by {md} (n={n}) — must be bit-identical"
        );

        // No per-iter reset: softmax does data-independent WORK (max/exp/sum/normalize
        // loops are the same regardless of values) and its output stays finite, so
        // re-running on the evolving buffer measures the kernel without a copy in-loop.
        group.bench_with_input(BenchmarkId::new("scalar_max", n), &base, |bn, base| {
            let mut buf = base.clone();
            bn.iter(|| {
                softmax_rows(black_box(&mut buf), n, SCALE, false);
                black_box(&buf);
            });
        });
        group.bench_with_input(BenchmarkId::new("simd_max", n), &base, |bn, base| {
            let mut buf = base.clone();
            bn.iter(|| {
                softmax_rows(black_box(&mut buf), n, SCALE, true);
                black_box(&buf);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
