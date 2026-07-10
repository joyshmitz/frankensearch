//! f16-slab → signed-4-bit packing: scalar per-element `f16::to_f32()` vs SIMD widen.
//!
//! Sibling of `f16_slab_quantize` (int8). `search.rs::pack_4bit_f16_bytes` builds the
//! packed 4-bit slab for the 4-bit two-pass scan, decoding every f16 with a SCALAR
//! `f16::from_le_bytes(..).to_f32()` — the same decode-bound gap the f16-DOT arc fixed
//! with the branchless SIMD widen (`simd.rs::widen8_f16_bytes`, Giesen, bit-exact).
//!
//! SIMD-widen the decode (8 f16/group), keep the max-abs reduction + nibble packing
//! scalar → BIT-IDENTICAL 4-bit slab (parity asserts equality ∀ input).
//!
//! Arms: `scalar` (= shipped) vs `simd_widen`. Swept over vector count (dim=384).
//!
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-index --profile release --bench f16_slab_pack4bit
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
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

#[inline]
fn nibble_of(value: f32, scale: f32) -> u8 {
    let q = (value * scale).round().clamp(-7.0, 7.0) as i8;
    (q as u8) & 0x0F
}

// ── SHIPPED scalar pack (verbatim from search.rs, pre-change) ────────────────
fn pack_scalar(bytes: &[u8], dim: usize) -> Vec<u8> {
    let mut max_abs = 0.0_f32;
    for chunk in bytes.chunks_exact(2) {
        let value = f16::from_le_bytes([chunk[0], chunk[1]]).to_f32().abs();
        if value > max_abs {
            max_abs = value;
        }
    }
    let scale = if max_abs > 1e-9 { 7.0 / max_abs } else { 0.0 };
    let count = bytes.len() / (dim * 2);
    let bytes_per_vector = dim.div_ceil(2);
    let mut slab = vec![0_u8; count * bytes_per_vector];
    for v in 0..count {
        let base = v * dim * 2;
        let out = v * bytes_per_vector;
        for d in 0..dim {
            let value = f16::from_le_bytes([bytes[base + d * 2], bytes[base + d * 2 + 1]]).to_f32();
            let nib = nibble_of(value, scale);
            if d % 2 == 0 {
                slab[out + d / 2] |= nib;
            } else {
                slab[out + d / 2] |= nib << 4;
            }
        }
    }
    slab
}

// ── CANDIDATE: SIMD-widen decode, scalar max-abs + nibble pack (bit-identical) ──
fn pack_simd(bytes: &[u8], dim: usize) -> Vec<u8> {
    let n = bytes.len();
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
    let scale = if max_abs > 1e-9 { 7.0 / max_abs } else { 0.0 };
    let count = bytes.len() / (dim * 2);
    let bytes_per_vector = dim.div_ceil(2);
    let mut slab = vec![0_u8; count * bytes_per_vector];
    for v in 0..count {
        let base = v * dim * 2;
        let out = v * bytes_per_vector;
        let mut d = 0;
        while d + 8 <= dim {
            let off = base + d * 2;
            let vv = widen8_f16_bytes(bytes[off..off + 16].try_into().unwrap());
            for (j, value) in vv.to_array().into_iter().enumerate() {
                let dd = d + j;
                let nib = nibble_of(value, scale);
                if dd % 2 == 0 {
                    slab[out + dd / 2] |= nib;
                } else {
                    slab[out + dd / 2] |= nib << 4;
                }
            }
            d += 8;
        }
        while d < dim {
            let value = f16::from_le_bytes([bytes[base + d * 2], bytes[base + d * 2 + 1]]).to_f32();
            let nib = nibble_of(value, scale);
            if d % 2 == 0 {
                slab[out + d / 2] |= nib;
            } else {
                slab[out + d / 2] |= nib << 4;
            }
            d += 1;
        }
    }
    slab
}

fn slab_fixture(vectors: usize) -> Vec<u8> {
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
    let mut group = c.benchmark_group("f16_slab_pack4bit");
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_millis(1000));

    for &vectors in &[1_000usize, 10_000, 50_000] {
        let slab = slab_fixture(vectors);
        let a = pack_scalar(&slab, DIM);
        let b = pack_simd(&slab, DIM);
        assert!(
            a == b,
            "simd pack diverged from scalar (vectors={vectors}) — must be bit-identical"
        );

        group.throughput(criterion::Throughput::Elements((vectors * DIM) as u64));
        group.bench_with_input(BenchmarkId::new("scalar", vectors), &slab, |bn, slab| {
            bn.iter(|| black_box(pack_scalar(black_box(slab), DIM)));
        });
        group.bench_with_input(
            BenchmarkId::new("simd_widen", vectors),
            &slab,
            |bn, slab| {
                bn.iter(|| black_box(pack_simd(black_box(slab), DIM)));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
