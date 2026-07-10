//! f16-slab → int8 quantization: scalar per-element `f16::to_f32()` vs SIMD widen.
//!
//! `search.rs::quantize_f16_bytes_to_i8` lazily builds (once, cached in `int8_slab`)
//! the int8 quantization of the whole F16 main-vector region for the int8 two-pass
//! scan. It decodes every f16 with a SCALAR `f16::from_le_bytes(..).to_f32()` — the
//! exact decode-bound bottleneck the f16-dot arc already fixed with a branchless SIMD
//! widen (`simd.rs::widen8_f16_bytes`, Giesen magic-multiply, bit-exact), but this
//! sibling quantize path was MISSED and stayed scalar (a sibling-path-consistency gap).
//!
//! Structurally-different primitive (SIMD data-layout / branchless widen): decode 8
//! f16 lanes per instruction group. The scale+round+clamp stays SCALAR (identical to
//! the shipped code), so the output int8 is BIT-IDENTICAL (the SIMD widen is bit-exact
//! to `f16::to_f32()` for finite/subnormal/zero — verified exhaustively in simd.rs;
//! the max-abs reduction is order-independent). Parity asserts equality ∀ input.
//!
//! Arms: `scalar` (= shipped) vs `simd_widen`. Swept over vector count (dim=384).
//!
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-index --profile release --bench f16_slab_quantize
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_index::quantize_f16_le_bytes_to_i8;
use half::f16;
use wide::{f32x8, u16x8, u32x8};

const DIM: usize = 384;

// ── Giesen branchless f16→f32 widen (copied verbatim from simd.rs) ───────────
const F16_WIDEN_MAGIC: f32 = f32::from_bits(0x7780_0000);

#[inline(always)]
fn widen8_f16_lanes(h: u32x8) -> f32x8 {
    let sign = (h & u32x8::splat(0x0000_8000)) << 16_u32;
    let exp_mant = (h & u32x8::splat(0x0000_7fff)) << 13_u32;
    let scaled = bytemuck::cast::<u32x8, f32x8>(exp_mant) * f32x8::splat(F16_WIDEN_MAGIC);
    let scaled_bits = bytemuck::cast::<f32x8, u32x8>(scaled);
    let he = h & u32x8::splat(0x0000_7c00);
    let carry = (he + u32x8::splat(0x0000_0400)) & u32x8::splat(0x0000_8000);
    let infnan_mask = (carry >> 15_u32) * u32x8::splat(0xff << 23);
    bytemuck::cast::<u32x8, f32x8>((scaled_bits | infnan_mask) | sign)
}

#[inline(always)]
fn widen8_f16_bytes(b: &[u8; 16]) -> f32x8 {
    let lanes = bytemuck::cast::<[u8; 16], u16x8>(*b);
    widen8_f16_lanes(u32x8::from(lanes))
}

// ── SHIPPED scalar quantize (verbatim from search.rs) ────────────────────────
fn quantize_scalar(bytes: &[u8]) -> Vec<i8> {
    let mut max_abs = 0.0_f32;
    for chunk in bytes.chunks_exact(2) {
        let value = f16::from_le_bytes([chunk[0], chunk[1]]).to_f32().abs();
        if value > max_abs {
            max_abs = value;
        }
    }
    if max_abs <= 0.0 {
        return vec![0; bytes.len() / 2];
    }
    let scale = 127.0 / max_abs;
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            let value = f16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
            (value * scale).round().clamp(-127.0, 127.0) as i8
        })
        .collect()
}

// ── CANDIDATE: SIMD-widen decode (bottleneck), scalar scale/round/clamp (identical) ──
fn quantize_simd(bytes: &[u8]) -> Vec<i8> {
    let n = bytes.len();
    // Pass 1: max-abs. SIMD abs+max (order-independent → identical max value).
    let mut maxv = f32x8::splat(0.0);
    let mut i = 0;
    while i + 16 <= n {
        let v = widen8_f16_bytes(bytes[i..i + 16].try_into().unwrap());
        maxv = maxv.max(v.abs());
        i += 16;
    }
    let mut max_abs = maxv.to_array().into_iter().fold(0.0_f32, f32::max);
    while i + 2 <= n {
        let value = f16::from_le_bytes([bytes[i], bytes[i + 1]]).to_f32().abs();
        if value > max_abs {
            max_abs = value;
        }
        i += 2;
    }
    if max_abs <= 0.0 {
        return vec![0; n / 2];
    }
    let scale = 127.0 / max_abs;
    // Pass 2: SIMD decode 8 f16 → f32, then SCALAR scale/round/clamp (bit-identical).
    let mut out = Vec::with_capacity(n / 2);
    let mut i = 0;
    while i + 16 <= n {
        let v = widen8_f16_bytes(bytes[i..i + 16].try_into().unwrap());
        for value in v.to_array() {
            out.push((value * scale).round().clamp(-127.0, 127.0) as i8);
        }
        i += 16;
    }
    while i + 2 <= n {
        let value = f16::from_le_bytes([bytes[i], bytes[i + 1]]).to_f32();
        out.push((value * scale).round().clamp(-127.0, 127.0) as i8);
        i += 2;
    }
    out
}

fn slab_fixture(vectors: usize) -> Vec<u8> {
    // Realistic L2-normalized-ish f16 embeddings: small finite values.
    let mut out = Vec::with_capacity(vectors * DIM * 2);
    let mut s = 0x2545_f491_4f6c_dd1d_u64;
    for _ in 0..vectors * DIM {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let x = ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.12;
        out.extend_from_slice(&f16::from_f32(x).to_le_bytes());
    }
    out
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("f16_slab_quantize");
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_millis(1000));

    for &vectors in &[1_000usize, 10_000, 50_000] {
        let slab = slab_fixture(vectors);
        // Parity: SIMD widen is bit-exact + scalar round unchanged → identical i8.
        let a = quantize_scalar(&slab);
        let b = quantize_simd(&slab);
        assert!(
            a == b,
            "simd quantize diverged from scalar (vectors={vectors}) — must be bit-identical"
        );
        let d = quantize_f16_le_bytes_to_i8(&slab);
        assert!(
            a == d,
            "dispatch quantize diverged from scalar (vectors={vectors}) — must be bit-identical"
        );

        group.throughput(criterion::Throughput::Elements((vectors * DIM) as u64));
        group.bench_with_input(BenchmarkId::new("scalar", vectors), &slab, |bn, slab| {
            bn.iter(|| black_box(quantize_scalar(black_box(slab))));
        });
        group.bench_with_input(
            BenchmarkId::new("simd_widen", vectors),
            &slab,
            |bn, slab| {
                bn.iter(|| black_box(quantize_simd(black_box(slab))));
            },
        );
        group.bench_with_input(BenchmarkId::new("dispatch", vectors), &slab, |bn, slab| {
            bn.iter(|| black_box(quantize_f16_le_bytes_to_i8(black_box(slab))));
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
