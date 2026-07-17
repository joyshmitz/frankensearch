//! Paired A/B for `InMemoryVectorIndex::vector_at_f32`'s f16 decode.
//!
//! The fully-resident index's `vector_at_f32` materializes an f16 vector slice to
//! `Vec<f32>` (HNSW graph build, fingerprinting) and decoded each f16 with a scalar
//! `half::f16::to_f32` loop — the in-memory twin of the FSVI `vector_at_f32` that
//! now widens 8 f16 per block via `simd::widen8_f16_slice` (the `wide` magic-factor
//! widen, bit-identical to `to_f32`). This bench mirrors both decoders over an f16
//! slice (the `widen` arm is a byte-for-byte copy of `widen8_f16_slice`) and asserts
//! a bit-identical `Vec<f32>` before timing:
//!
//! - `scalar` : `iter().map(|v| v.to_f32()).collect()` (shipping path).
//! - `widen`  : `chunks_exact(8)` → SIMD widen + scalar tail (the new path).
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/<lane> \
//!   rch exec -- cargo bench -p frankensearch-index --bench inmem_vector_decode_f16_ab
//! ```

use std::hint::black_box;

use bytemuck::cast;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use half::f16;
use wide::{f32x8, u32x8};

const F16_WIDEN_MAGIC: f32 = f32::from_bits(0x7780_0000);

/// Exact copy of `simd::widen8_f16_lanes`.
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

/// Exact copy of `simd::widen8_f16_slice`.
#[inline(always)]
fn widen8_f16_slice(s: &[f16; 8]) -> f32x8 {
    let lanes: [u32; 8] = [
        u32::from(s[0].to_bits()),
        u32::from(s[1].to_bits()),
        u32::from(s[2].to_bits()),
        u32::from(s[3].to_bits()),
        u32::from(s[4].to_bits()),
        u32::from(s[5].to_bits()),
        u32::from(s[6].to_bits()),
        u32::from(s[7].to_bits()),
    ];
    widen8_f16_lanes(cast::<[u32; 8], u32x8>(lanes))
}

/// Shipping path: scalar `to_f32` per f16.
fn decode_scalar(stored: &[f16]) -> Vec<f32> {
    stored.iter().map(|v| v.to_f32()).collect()
}

/// New path: SIMD widen 8 f16 per block + scalar tail.
fn decode_widen(stored: &[f16]) -> Vec<f32> {
    let mut out = Vec::with_capacity(stored.len());
    let mut blocks = stored.chunks_exact(8);
    for block in &mut blocks {
        let arr: &[f16; 8] = block.try_into().expect("8-f16 block");
        out.extend_from_slice(&widen8_f16_slice(arr).to_array());
    }
    for v in blocks.remainder() {
        out.push(v.to_f32());
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

fn make_slab(dim: usize) -> Vec<f16> {
    let mut state = 0x1234_5678_9abc_def0_u64;
    (0..dim)
        .map(|_| {
            let v = ((next(&mut state) >> 40) as f32) / 16_777_216.0 * 2.0 - 1.0;
            f16::from_f32(v)
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("inmem_vector_decode_f16");
    for &dim in &[256usize, 384, 768] {
        let slab = make_slab(dim);

        // Bit-identical parity gate.
        let a = decode_scalar(&slab);
        let b = decode_widen(&slab);
        assert_eq!(a.len(), b.len(), "len parity (dim={dim})");
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.to_bits(), y.to_bits(), "decode parity (dim={dim})");
        }

        group.bench_with_input(BenchmarkId::new("scalar", dim), &slab, |bch, slab| {
            bch.iter(|| black_box(decode_scalar(black_box(slab))));
        });
        group.bench_with_input(BenchmarkId::new("widen", dim), &slab, |bch, slab| {
            bch.iter(|| black_box(decode_widen(black_box(slab))));
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
