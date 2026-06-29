//! Latency-bound-reduction probe: scalar serial `f32::max` reduction vs an
//! f32x8 SIMD lanewise-max + horizontal reduce.
//!
//! This is the shape of the attention-softmax max pass in the native reranker
//! (`softmax_row_fused`: `for &x in row { max_raw = max_raw.max(x) }`), which
//! `f32::max`'s NaN semantics typically block LLVM from auto-vectorizing — so it
//! runs as one serial `maxss` dependency chain (~3-4 cycle latency/elem), the
//! same latency-bound shape the f16 dot kernel had. SIMD max over 8 independent
//! lanes + one horizontal reduce should break that chain. Bit-identical for
//! finite/-inf inputs (max is order-independent; `maxps`/`f32::max` both return
//! the larger operand and pass through -inf).
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-index --bench max_reduction
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use wide::f32x8;

fn xorshift(s: &mut u64) -> f32 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    #[allow(clippy::cast_precision_loss)]
    let u = (*s >> 40) as f32 / f64::from(1_u32 << 24) as f32;
    u.mul_add(20.0, -10.0)
}

fn build_row(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| xorshift(&mut s)).collect()
}

#[inline(never)]
fn max_scalar(row: &[f32]) -> f32 {
    let mut m = f32::NEG_INFINITY;
    for &x in row {
        m = m.max(x);
    }
    m
}

#[inline(never)]
fn max_simd(row: &[f32]) -> f32 {
    let mut acc = f32x8::splat(f32::NEG_INFINITY);
    let mut chunks = row.chunks_exact(8);
    for c in chunks.by_ref() {
        let v: &[f32; 8] = c.try_into().expect("chunks_exact(8)");
        acc = acc.max(f32x8::from(*v));
    }
    let mut m = acc.to_array().into_iter().fold(f32::NEG_INFINITY, f32::max);
    for &x in chunks.remainder() {
        m = m.max(x);
    }
    m
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("max_reduction");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    for &n in &[128_usize, 512, 2048] {
        let row = build_row(n, 0x1234 ^ (n as u64));
        // Equivalence (finite inputs ⇒ identical max).
        assert_eq!(max_scalar(&row), max_simd(&row), "simd max must equal scalar at n={n}");

        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |bch, _| {
            bch.iter(|| black_box(max_scalar(black_box(&row))));
        });
        group.bench_with_input(BenchmarkId::new("simd", n), &n, |bch, _| {
            bch.iter(|| black_box(max_simd(black_box(&row))));
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
