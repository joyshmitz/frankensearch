//! A/B for the lazy int8 ADC slab build (`quantize_f16_slab_to_i8`): the f16→i8
//! quantization is decode-bound (`f16::to_f32` software, twice — max-abs then
//! quantize) + a per-element `round`, all of which the AVX2+F16C kernel crushes.
//!
//! - `dispatch` : `quantize_f16_slab_to_i8` (runtime AVX2+F16C when available).
//! - `generic`  : `quantize_f16_slab_to_i8_generic` (portable scalar fallback).
//!
//! Both return identical `Vec<i8>` (asserted). This is an index-build / cold-start
//! cost (the slab is `OnceLock`-cached, so it is amortized for a static index but
//! recurs on rebuild/refresh). Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/<lane> \
//!   rch exec -- cargo bench -p frankensearch-index --bench quantize_slab
//! ```

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_index::{
    pack_f16_slab_to_4bit, pack_f16_slab_to_4bit_generic, quantize_f16_slab_to_i8,
    quantize_f16_slab_to_i8_generic,
};
use half::f16;

const DIM: usize = 384;

fn make_slab(n: usize) -> Vec<f16> {
    let mut state = 0x1234_5678_9abc_def0_u64;
    (0..n * DIM)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            #[allow(clippy::cast_precision_loss)]
            let x = ((state >> 40) as f32 / (1_u64 << 23) as f32) - 1.0;
            f16::from_f32(x)
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("quantize_i8_slab");
    g.sample_size(20);
    for &n in &[10_000_usize, 50_000] {
        let slab = make_slab(n);
        assert_eq!(
            quantize_f16_slab_to_i8(&slab),
            quantize_f16_slab_to_i8_generic(&slab),
            "dispatch and generic must match"
        );
        g.bench_function(BenchmarkId::new("generic", n), |b| {
            b.iter(|| black_box(quantize_f16_slab_to_i8_generic(black_box(&slab))));
        });
        g.bench_function(BenchmarkId::new("dispatch", n), |b| {
            b.iter(|| black_box(quantize_f16_slab_to_i8(black_box(&slab))));
        });
    }
    g.finish();

    // 4-bit slab pack — the wired-default two-pass pass-1 storage build.
    let mut g4 = c.benchmark_group("pack_4bit_slab");
    g4.sample_size(20);
    for &n in &[10_000_usize, 50_000] {
        let slab = make_slab(n);
        assert_eq!(
            pack_f16_slab_to_4bit(&slab, DIM),
            pack_f16_slab_to_4bit_generic(&slab, DIM),
            "4bit dispatch and generic must match"
        );
        g4.bench_function(BenchmarkId::new("generic", n), |b| {
            b.iter(|| black_box(pack_f16_slab_to_4bit_generic(black_box(&slab), DIM)));
        });
        g4.bench_function(BenchmarkId::new("dispatch", n), |b| {
            b.iter(|| black_box(pack_f16_slab_to_4bit(black_box(&slab), DIM)));
        });
    }
    g4.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
