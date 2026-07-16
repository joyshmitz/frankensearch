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

// bd-yt8m: this bench hand-writes an AVX2 intrinsic reduction as a same-binary proxy for the
// `-C target-cpu=x86-64-v3` codegen of the `wide::f32x8` site. The workspace denies unsafe;
// opt in locally (the intrinsics are the whole point of the measurement).
#![allow(unsafe_code)]

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_core::bench_support::paired_median_ratio;
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

// ── bd-yt8m: x86-64-v3 codegen proxy for the `wide::f32x8` reduction site ─────
//
// The workspace builds at the generic `x86-64` baseline (SSE2), so `max_simd`'s
// `wide::f32x8` lowers to TWO 128-bit `maxps` per 8 lanes. A `-C target-cpu=x86-64-v3`
// build would lower the same `f32x8` to ONE 256-bit `vmaxps`. Since `wide` picks its
// width by compile-time `cfg(target_feature)`, that codegen can't be toggled per-fn — but
// a hand `#[target_feature(enable = "avx2")]` 256-bit reduction emits exactly the
// instructions v3 would, INSIDE this baseline binary. So `max_simd` (SSE2) vs `max_avx2`
// (AVX2-256) is a same-binary A/B that quantifies the v3 upside for this site with one
// build — no cross-build required. Same chunking/horizontal-reduce shape as `max_simd`,
// so it is bit-identical for finite inputs (asserted).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn max_avx2_impl(row: &[f32]) -> f32 {
    use core::arch::x86_64::{_mm256_loadu_ps, _mm256_max_ps, _mm256_set1_ps, _mm256_storeu_ps};
    let mut acc = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut chunks = row.chunks_exact(8);
    for c in chunks.by_ref() {
        // SAFETY: `chunks_exact(8)` guarantees 8 readable f32s at `c.as_ptr()`.
        let v = unsafe { _mm256_loadu_ps(c.as_ptr()) };
        acc = _mm256_max_ps(acc, v);
    }
    let mut tmp = [0.0_f32; 8];
    // SAFETY: `tmp` holds 8 f32s.
    unsafe { _mm256_storeu_ps(tmp.as_mut_ptr(), acc) };
    let mut m = tmp.into_iter().fold(f32::NEG_INFINITY, f32::max);
    for &x in chunks.remainder() {
        m = m.max(x);
    }
    m
}

#[inline(never)]
fn max_avx2(row: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by runtime AVX2 detection.
            return unsafe { max_avx2_impl(row) };
        }
    }
    max_simd(row)
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("max_reduction");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    for &n in &[128_usize, 512, 2048] {
        let row = build_row(n, 0x1234 ^ (n as u64));
        // Equivalence (finite inputs ⇒ identical max).
        assert_eq!(
            max_scalar(&row),
            max_simd(&row),
            "simd max must equal scalar at n={n}"
        );
        assert_eq!(
            max_avx2(&row),
            max_simd(&row),
            "avx2 max must equal wide-simd at n={n}"
        );

        // bd-yt8m: drift-cancelled A/B of the wide::f32x8 reduction as-shipped (SSE2 in this
        // baseline binary) vs the AVX2-256 codegen a `-C target-cpu=x86-64-v3` build would emit.
        // `<1` ⇒ AVX2 (v3) faster; INSIDE NULL FLOOR ⇒ v3 buys nothing at this site.
        let run_simd = || {
            black_box(max_simd(black_box(&row)));
        };
        let run_avx2 = || {
            black_box(max_avx2(black_box(&row)));
        };
        let null = paired_median_ratio(41, 64, run_simd, run_simd);
        let lever = paired_median_ratio(41, 64, run_simd, run_avx2);
        eprintln!(
            "[v3-null  n={n}] simd/simd median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
            null.median, null.p5, null.p95, null.rounds
        );
        eprintln!(
            "[v3-lever n={n}] avx2/simd median {:.4} p5 {:.4} p95 {:.4} -> {} (<1 ⇒ v3 faster)",
            lever.median,
            lever.p5,
            lever.p95,
            if lever.decidable_against(&null) {
                "DECIDABLE"
            } else {
                "INSIDE NULL FLOOR"
            }
        );

        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |bch, _| {
            bch.iter(|| black_box(max_scalar(black_box(&row))));
        });
        group.bench_with_input(BenchmarkId::new("simd", n), &n, |bch, _| {
            bch.iter(|| black_box(max_simd(black_box(&row))));
        });
        group.bench_with_input(BenchmarkId::new("avx2", n), &n, |bch, _| {
            bch.iter(|| black_box(max_avx2(black_box(&row))));
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
