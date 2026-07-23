//! Paired A/B for `VectorIndex::vector_at_f32`'s F16 decode.
//!
//! Materializing an f16-stored vector to `Vec<f32>` (PRF feedback lookup per query,
//! index refresh/build) decoded each f16 with a **scalar** `half::f16::to_f32` loop —
//! even though the f16 dot kernels already widen 8 f16 per 16-byte block through the
//! `wide` magic-factor `widen8_f16_bytes` (bit-identical to the scalar decode). This
//! bench mirrors both decoders over a realistic f16 slab and asserts a bit-identical
//! `Vec<f32>` before timing:
//!
//! - `scalar` : `chunks_exact(2)` → `f16::from_le_bytes(..).to_f32()` (shipping path).
//! - `widen`  : `chunks_exact(16)` → SIMD widen 8 lanes + scalar tail (the new path).
//!
//! The `widen` reimplementation is byte-for-byte the crate's internal
//! `widen8_f16_bytes` (which is `pub(crate)`); the parity gate proves it matches the
//! scalar reference, so the measurement is faithful to the shipped change.
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/<lane> \
//!   rch exec -- cargo bench -p frankensearch-index --bench vector_decode_f16_ab
//! ```

#![allow(clippy::chunks_exact_to_as_chunks, clippy::inline_always)]

use std::hint::black_box;

use bytemuck::cast;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use half::f16;
use wide::{f32x8, u16x8, u32x8};

const F16_WIDEN_MAGIC: f32 = f32::from_bits(0x7780_0000);

/// Exact copy of `simd::widen8_f16_lanes` (Giesen magic-factor widen).
#[inline(always)]
fn widen8_f16_lanes(h: u32x8) -> f32x8 {
    let sign = (h & u32x8::splat(0x0000_8000)) << 16_u32;
    let exp_mant = (h & u32x8::splat(0x0000_7fff)) << 13_u32;
    let scaled = cast::<u32x8, f32x8>(exp_mant) * f32x8::splat(F16_WIDEN_MAGIC);
    let scaled_bits = cast::<f32x8, u32x8>(scaled);
    let he = h & u32x8::splat(0x0000_7c00);
    let carry = (he + u32x8::splat(0x0000_0400)) & u32x8::splat(0x0000_8000);
    let infnan_mask = (carry >> 15_u32) * u32x8::splat(0xff << 23);
    cast::<u32x8, f32x8>((scaled_bits | infnan_mask) | sign)
}

/// Exact copy of `simd::widen8_f16_bytes` (little-endian path).
#[inline(always)]
fn widen8_f16_bytes(b: &[u8; 16]) -> f32x8 {
    let lanes = cast::<[u8; 16], u16x8>(*b);
    widen8_f16_lanes(u32x8::from(lanes))
}

/// Shipping path: scalar `half::to_f32` per f16.
fn decode_scalar(bytes: &[u8], dim: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(dim);
    for chunk in bytes.chunks_exact(2) {
        out.push(f16::from_le_bytes([chunk[0], chunk[1]]).to_f32());
    }
    out
}

/// New path: SIMD widen 8 lanes per block + scalar tail.
fn decode_widen(bytes: &[u8], dim: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(dim);
    let mut blocks = bytes.chunks_exact(16);
    for block in &mut blocks {
        let arr: &[u8; 16] = block.try_into().expect("16-byte block");
        out.extend_from_slice(&widen8_f16_bytes(arr).to_array());
    }
    for chunk in blocks.remainder().chunks_exact(2) {
        out.push(f16::from_le_bytes([chunk[0], chunk[1]]).to_f32());
    }
    out
}

fn next(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

/// A realistic f16 vector slab (`dim` little-endian halves in roughly [-1, 1)).
fn make_slab(dim: usize) -> Vec<u8> {
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut slab = Vec::with_capacity(dim * 2);
    for _ in 0..dim {
        let v = ((next(&mut state) >> 40) as f32) / 16_777_216.0 * 2.0 - 1.0;
        slab.extend_from_slice(&f16::from_f32(v).to_le_bytes());
    }
    slab
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_decode_f16");
    for &dim in &[256usize, 384, 768] {
        let slab = make_slab(dim);

        // Bit-identical parity gate.
        let a = decode_scalar(&slab, dim);
        let b = decode_widen(&slab, dim);
        assert_eq!(a.len(), b.len(), "len parity (dim={dim})");
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.to_bits(), y.to_bits(), "decode parity (dim={dim})");
        }

        group.bench_with_input(BenchmarkId::new("scalar", dim), &slab, |bch, slab| {
            bch.iter(|| black_box(decode_scalar(black_box(slab), dim)));
        });
        group.bench_with_input(BenchmarkId::new("widen", dim), &slab, |bch, slab| {
            bch.iter(|| black_box(decode_widen(black_box(slab), dim)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
