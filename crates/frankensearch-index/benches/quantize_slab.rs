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

#![allow(clippy::chunks_exact_to_as_chunks, clippy::significant_drop_tightening)]

use std::hint::black_box;
use std::io::Write;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_index::{
    encode_f32_to_f16_extend, encode_f32_to_f16_extend_generic, pack_f16_slab_to_4bit,
    pack_f16_slab_to_4bit_generic, quantize_f16_slab_to_i8, quantize_f16_slab_to_i8_generic,
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

    // f32→f16 encode — the per-element conversion in every index build.
    let mut ge = c.benchmark_group("encode_f32_to_f16");
    ge.sample_size(20);
    for &n in &[10_000_usize, 50_000] {
        let src: Vec<f32> = make_slab(n).iter().map(|h| h.to_f32()).collect();
        let mut a = Vec::new();
        encode_f32_to_f16_extend(&src, &mut a);
        let mut b = Vec::new();
        encode_f32_to_f16_extend_generic(&src, &mut b);
        assert_eq!(
            a.iter().map(|h| h.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|h| h.to_bits()).collect::<Vec<_>>(),
            "encode dispatch and generic must match"
        );
        // Reuse one buffer (clear keeps capacity) so we measure the encode, not a
        // fresh per-iter allocation's page faults — matching the real build, which
        // appends to one pre-reserved flat Vec.
        ge.bench_function(BenchmarkId::new("generic", n), |bn| {
            let mut out = Vec::with_capacity(src.len());
            bn.iter(|| {
                out.clear();
                encode_f32_to_f16_extend_generic(black_box(&src), &mut out);
                black_box(&out);
            });
        });
        ge.bench_function(BenchmarkId::new("dispatch", n), |bn| {
            let mut out = Vec::with_capacity(src.len());
            bn.iter(|| {
                out.clear();
                encode_f32_to_f16_extend(black_box(&src), &mut out);
                black_box(&out);
            });
        });
    }
    ge.finish();

    // write_vector_slab F16 arm: per-element `from_f32` + 2-byte `write_all`
    // (`per_element`) vs the F16C encode + one `write_all` per record (`batched`).
    let mut gw = c.benchmark_group("write_f16_slab");
    gw.sample_size(20);
    for &n in &[10_000_usize, 50_000] {
        let flat: Vec<f32> = make_slab(n).iter().map(|h| h.to_f32()).collect();
        // Verify both arms emit identical bytes.
        let mut old_bytes = Vec::new();
        for chunk in flat.chunks_exact(DIM) {
            for &v in chunk {
                old_bytes.extend_from_slice(&f16::from_f32(v).to_le_bytes());
            }
        }
        let mut new_bytes = Vec::new();
        let mut scratch: Vec<f16> = Vec::with_capacity(DIM);
        for chunk in flat.chunks_exact(DIM) {
            scratch.clear();
            encode_f32_to_f16_extend(chunk, &mut scratch);
            for &h in &scratch {
                new_bytes.extend_from_slice(&h.to_le_bytes());
            }
        }
        assert_eq!(old_bytes, new_bytes, "write arms must emit identical bytes");

        gw.bench_function(BenchmarkId::new("per_element", n), |bn| {
            let mut out = Vec::with_capacity(flat.len() * 2);
            bn.iter(|| {
                out.clear();
                for chunk in flat.chunks_exact(DIM) {
                    for &v in chunk {
                        out.write_all(&f16::from_f32(v).to_le_bytes()).unwrap();
                    }
                }
                black_box(&out);
            });
        });
        gw.bench_function(BenchmarkId::new("batched", n), |bn| {
            let mut out = Vec::with_capacity(flat.len() * 2);
            let mut scratch: Vec<f16> = Vec::with_capacity(DIM);
            bn.iter(|| {
                out.clear();
                for chunk in flat.chunks_exact(DIM) {
                    scratch.clear();
                    encode_f32_to_f16_extend(chunk, &mut scratch);
                    // SAFETY: little-endian target; `half::f16` is `repr(transparent)`
                    // over `u16`, so native bytes equal the LE on-disk encoding.
                    #[allow(unsafe_code)]
                    let bytes = unsafe {
                        core::slice::from_raw_parts(
                            scratch.as_ptr().cast::<u8>(),
                            scratch.len() * 2,
                        )
                    };
                    out.write_all(bytes).unwrap();
                }
                black_box(&out);
            });
        });
    }
    gw.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
