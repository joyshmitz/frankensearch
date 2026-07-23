//! Batched multi-query vector-scan execution model vs per-query sequential.
//!
//! Realistic workload: a high-QPS server scores a *batch* of `B` concurrent
//! query vectors against the full corpus. The production path scans the corpus
//! once **per query** (`dot_product_f16_bytes_f32` over the mmap'd f16 slab),
//! so `B` queries stream the entire corpus `B` times. When the corpus exceeds
//! last-level cache (100k × 384 × 2 B = 73 MB ≫ L3), each pass is **memory-
//! bandwidth-bound** and re-fetches every vector from RAM.
//!
//! The batched model interchanges the loops: for each corpus vector, score it
//! against **all `B` queries while it is hot in L1**, so the corpus is streamed
//! from RAM exactly once and each loaded vector is reused `B` times. Identical
//! arithmetic (`B·N` dots), identical results — only the memory access pattern
//! differs. This is the fundamental amortization Lucene/Tantivy/Meili do not do
//! (they execute queries independently).
//!
//! Both arms run head-to-head in one process so the ratio is host-independent.
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-index --bench batched_query_scan
//! ```

#![allow(clippy::cast_possible_truncation)]

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_index::{dot_product_f16_bytes_f32, dot_product_f32_f32};
use half::f16;

const DIM: usize = 384;

/// AVX2+F16C decode of `dst.len()` packed little-endian f16 values to f32.
/// `vcvtph2ps` is the exact same hardware decode the f16 dot kernel uses, so the
/// widened f32 values are bit-identical; we just do it **once** per corpus vector
/// instead of once per (corpus vector, query) pair. `DIM` is a multiple of 8, so
/// the scalar tail never runs in this bench.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "f16c"
))]
#[inline]
fn decode_f16_bytes_to_f32(src: &[u8], dst: &mut [f32]) {
    use std::arch::x86_64::{__m128i, _mm_loadu_si128, _mm256_cvtph_ps, _mm256_storeu_ps};
    let n = dst.len();
    let mut i = 0;
    // SAFETY: cfg-gated to avx2+f16c; reads 16 B and writes 8 f32 per step within
    // bounds (i + 8 <= n); pointers are from valid slices.
    unsafe {
        while i + 8 <= n {
            let bits = _mm_loadu_si128(src.as_ptr().add(i * 2).cast::<__m128i>());
            let f = _mm256_cvtph_ps(bits);
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), f);
            i += 8;
        }
    }
    while i < n {
        dst[i] = f16::from_le_bytes([src[i * 2], src[i * 2 + 1]]).to_f32();
        i += 1;
    }
}

#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "f16c"
)))]
#[inline]
fn decode_f16_bytes_to_f32(src: &[u8], dst: &mut [f32]) {
    for (i, d) in dst.iter_mut().enumerate() {
        *d = f16::from_le_bytes([src[i * 2], src[i * 2 + 1]]).to_f32();
    }
}

fn xorshift(s: &mut u64) -> f32 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    #[allow(clippy::cast_precision_loss)]
    let u = (*s >> 40) as f32 / f64::from(1_u32 << 24) as f32;
    u.mul_add(2.0, -1.0)
}

/// Build a contiguous f16 corpus slab of `n` vectors (the production storage).
fn build_corpus_bytes(n: usize, dim: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut bytes = Vec::with_capacity(n * dim * 2);
    for _ in 0..n {
        for _ in 0..dim {
            let h = f16::from_f32(xorshift(&mut s));
            bytes.extend_from_slice(&h.to_le_bytes());
        }
    }
    bytes
}

fn build_queries(b: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
    (0..b)
        .map(|_| (0..dim).map(|_| xorshift(&mut s)).collect())
        .collect()
}

/// Per-query sequential scan: the corpus is streamed once **per query**.
/// Returns the running max per query (a cheap top-1 proxy that forces every dot
/// to be computed and prevents dead-code elimination).
#[inline(never)]
fn scan_sequential(corpus: &[u8], n: usize, stride: usize, queries: &[Vec<f32>]) -> Vec<f32> {
    let mut maxes = vec![f32::NEG_INFINITY; queries.len()];
    for (qi, q) in queries.iter().enumerate() {
        let mut m = f32::NEG_INFINITY;
        for c in 0..n {
            let v = &corpus[c * stride..c * stride + stride];
            let d = dot_product_f16_bytes_f32(v, q).unwrap();
            if d > m {
                m = d;
            }
        }
        maxes[qi] = m;
    }
    maxes
}

/// Batched interleaved scan: the corpus is streamed **once**; each loaded vector
/// is scored against all `B` queries while hot in L1. Identical work + results.
#[inline(never)]
fn scan_batched(corpus: &[u8], n: usize, stride: usize, queries: &[Vec<f32>]) -> Vec<f32> {
    let mut maxes = vec![f32::NEG_INFINITY; queries.len()];
    for c in 0..n {
        let v = &corpus[c * stride..c * stride + stride];
        for (qi, q) in queries.iter().enumerate() {
            let d = dot_product_f16_bytes_f32(v, q).unwrap();
            if d > maxes[qi] {
                maxes[qi] = d;
            }
        }
    }
    maxes
}

/// Decode-once batched kernel: decode each f16 corpus vector to f32 **once**,
/// then score it against all `B` queries from the hot f32 scratch (in L1). This
/// amortizes both the RAM fetch AND the `vcvtph2ps` decode across the batch —
/// `N` decodes total instead of the sequential path's `B·N`.
#[inline(never)]
fn scan_batched_decode_once(
    corpus: &[u8],
    n: usize,
    stride: usize,
    queries: &[Vec<f32>],
) -> Vec<f32> {
    let mut maxes = vec![f32::NEG_INFINITY; queries.len()];
    let mut scratch = vec![0.0_f32; DIM];
    for c in 0..n {
        let v = &corpus[c * stride..c * stride + stride];
        decode_f16_bytes_to_f32(v, &mut scratch);
        for (qi, q) in queries.iter().enumerate() {
            let d = dot_product_f32_f32(&scratch, q).unwrap();
            if d > maxes[qi] {
                maxes[qi] = d;
            }
        }
    }
    maxes
}

fn bench(c: &mut Criterion) {
    let stride = DIM * 2;
    let mut group = c.benchmark_group("batched_query_scan");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(20);

    // 100k vectors (73 MB f16 slab) — well beyond L3, the memory-bound regime
    // where the per-query model re-streams the corpus from RAM B times.
    for &n in &[100_000_usize] {
        let corpus = build_corpus_bytes(n, DIM, 0x1234_5678);
        for &b in &[4_usize, 16, 64] {
            let queries = build_queries(b, DIM, 0xABCD);

            // Equivalence guards. Loop interchange is bit-identical (same kernel);
            // decode-once routes through dot_product_f32_f32 on the (exactly)
            // widened f16 values — equal up to f32 accumulator lane-grouping, so
            // checked within a tight relative epsilon.
            let seq = scan_sequential(&corpus, n, stride, &queries);
            let bat = scan_batched(&corpus, n, stride, &queries);
            assert_eq!(seq, bat, "batched scan must equal sequential scan");
            let dec = scan_batched_decode_once(&corpus, n, stride, &queries);
            for (a, d) in seq.iter().zip(&dec) {
                assert!(
                    (a - d).abs() <= 1e-3 * a.abs().max(1.0),
                    "decode-once result {d} diverged from sequential {a}"
                );
            }

            group.bench_with_input(
                BenchmarkId::new("sequential", format!("n{n}_b{b}")),
                &b,
                |bch, _| {
                    bch.iter(|| {
                        black_box(scan_sequential(&corpus, n, stride, black_box(&queries)))
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new("batched", format!("n{n}_b{b}")),
                &b,
                |bch, _| {
                    bch.iter(|| black_box(scan_batched(&corpus, n, stride, black_box(&queries))));
                },
            );
            group.bench_with_input(
                BenchmarkId::new("batched_decode_once", format!("n{n}_b{b}")),
                &b,
                |bch, _| {
                    bch.iter(|| {
                        black_box(scan_batched_decode_once(
                            &corpus,
                            n,
                            stride,
                            black_box(&queries),
                        ))
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
