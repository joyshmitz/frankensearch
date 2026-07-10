//! ILP probe for the int8·int8 dot kernel: 2 accumulators (the shipped
//! `dot_i8_i8_avx2`) vs 4 accumulators. Integer adds associate, so any
//! accumulator count is bit-identical — this only asks whether the 2-acc kernel
//! is latency-bound on its `vpaddd` chains (like the f16 dot was) or already
//! throughput-bound on `vpmovsxbw`/`vpmaddwd`.
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-index --bench i8_dot_ilp
//! ```

#![allow(unsafe_code)]

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

const DIM: usize = 384;
const N: usize = 4096;

fn xorshift(s: &mut u64) -> i8 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    (((*s >> 40) % 255) as i8).wrapping_sub(127)
}

fn build_corpus(n: usize, dim: usize, seed: u64) -> Vec<i8> {
    let mut s = seed;
    (0..n * dim).map(|_| xorshift(&mut s)).collect()
}

/// 2-accumulator AVX2 i8·i8 dot — faithful copy of the shipped `dot_i8_i8_avx2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_2acc(stored: &[i8], query: &[i8]) -> i32 {
    use core::arch::x86_64::{
        __m256i, _mm_add_epi32, _mm_cvtsi128_si32, _mm_shuffle_epi32, _mm_unpackhi_epi64,
        _mm256_add_epi32, _mm256_castsi256_si128, _mm256_cvtepi8_epi16, _mm256_extracti128_si256,
        _mm256_loadu_si256, _mm256_madd_epi16, _mm256_setzero_si256,
    };
    let n = stored.len().min(query.len());
    unsafe {
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut i = 0;
        while i + 32 <= n {
            let s = _mm256_loadu_si256(stored.as_ptr().add(i).cast::<__m256i>());
            let q = _mm256_loadu_si256(query.as_ptr().add(i).cast::<__m256i>());
            let s_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(s));
            let q_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q));
            let s_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(s));
            let q_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(q));
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(s_lo, q_lo));
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(s_hi, q_hi));
            i += 32;
        }
        let acc = _mm256_add_epi32(acc0, acc1);
        let sum128 = _mm_add_epi32(
            _mm256_castsi256_si128(acc),
            _mm256_extracti128_si256::<1>(acc),
        );
        let sum64 = _mm_add_epi32(sum128, _mm_unpackhi_epi64(sum128, sum128));
        let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32::<0b01>(sum64));
        let mut sum = _mm_cvtsi128_si32(sum32);
        while i < n {
            sum += i32::from(stored[i]) * i32::from(query[i]);
            i += 1;
        }
        sum
    }
}

/// 4-accumulator AVX2 i8·i8 dot — unrolls two 32-byte chunks per iteration into
/// four independent `vpaddd` chains. Bit-identical (integer sum).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_4acc(stored: &[i8], query: &[i8]) -> i32 {
    use core::arch::x86_64::{
        __m256i, _mm_add_epi32, _mm_cvtsi128_si32, _mm_shuffle_epi32, _mm_unpackhi_epi64,
        _mm256_add_epi32, _mm256_castsi256_si128, _mm256_cvtepi8_epi16, _mm256_extracti128_si256,
        _mm256_loadu_si256, _mm256_madd_epi16, _mm256_setzero_si256,
    };
    let n = stored.len().min(query.len());
    macro_rules! madd_at {
        ($off:expr) => {{
            let s = _mm256_loadu_si256(stored.as_ptr().add($off).cast::<__m256i>());
            let q = _mm256_loadu_si256(query.as_ptr().add($off).cast::<__m256i>());
            let s_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(s));
            let q_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q));
            let s_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(s));
            let q_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(q));
            (_mm256_madd_epi16(s_lo, q_lo), _mm256_madd_epi16(s_hi, q_hi))
        }};
    }
    unsafe {
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut acc2 = _mm256_setzero_si256();
        let mut acc3 = _mm256_setzero_si256();
        let mut i = 0;
        while i + 64 <= n {
            let (a, b) = madd_at!(i);
            let (c, d) = madd_at!(i + 32);
            acc0 = _mm256_add_epi32(acc0, a);
            acc1 = _mm256_add_epi32(acc1, b);
            acc2 = _mm256_add_epi32(acc2, c);
            acc3 = _mm256_add_epi32(acc3, d);
            i += 64;
        }
        while i + 32 <= n {
            let (a, b) = madd_at!(i);
            acc0 = _mm256_add_epi32(acc0, a);
            acc1 = _mm256_add_epi32(acc1, b);
            i += 32;
        }
        let acc = _mm256_add_epi32(_mm256_add_epi32(acc0, acc1), _mm256_add_epi32(acc2, acc3));
        let sum128 = _mm_add_epi32(
            _mm256_castsi256_si128(acc),
            _mm256_extracti128_si256::<1>(acc),
        );
        let sum64 = _mm_add_epi32(sum128, _mm_unpackhi_epi64(sum128, sum128));
        let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32::<0b01>(sum64));
        let mut sum = _mm_cvtsi128_si32(sum32);
        while i < n {
            sum += i32::from(stored[i]) * i32::from(query[i]);
            i += 1;
        }
        sum
    }
}

fn bench(c: &mut Criterion) {
    #[cfg(target_arch = "x86_64")]
    if !std::is_x86_feature_detected!("avx2") {
        eprintln!("i8_dot_ilp: avx2 not available; skipping");
        return;
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        eprintln!("i8_dot_ilp: non-x86_64; skipping");
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        let corpus = build_corpus(N, DIM, 0x9E37);
        let query: Vec<i8> = {
            let mut s = 0xD1CE_u64;
            (0..DIM).map(|_| xorshift(&mut s)).collect()
        };
        for i in 0..N {
            let v = &corpus[i * DIM..i * DIM + DIM];
            let a = unsafe { dot_i8_2acc(v, &query) };
            let b = unsafe { dot_i8_4acc(v, &query) };
            assert_eq!(a, b, "4-acc must equal 2-acc (integer) at {i}");
        }

        let mut group = c.benchmark_group("i8_dot_ilp");
        group.warm_up_time(Duration::from_millis(500));
        group.measurement_time(Duration::from_secs(3));
        group.bench_with_input(BenchmarkId::new("acc2", N), &N, |bch, _| {
            bch.iter(|| {
                let mut acc = 0_i32;
                for i in 0..N {
                    let v = &corpus[i * DIM..i * DIM + DIM];
                    acc = acc.wrapping_add(unsafe { dot_i8_2acc(black_box(v), black_box(&query)) });
                }
                black_box(acc)
            });
        });
        group.bench_with_input(BenchmarkId::new("acc4", N), &N, |bch, _| {
            bch.iter(|| {
                let mut acc = 0_i32;
                for i in 0..N {
                    let v = &corpus[i * DIM..i * DIM + DIM];
                    acc = acc.wrapping_add(unsafe { dot_i8_4acc(black_box(v), black_box(&query)) });
                }
                black_box(acc)
            });
        });
        group.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
