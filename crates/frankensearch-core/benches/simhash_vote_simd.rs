//! SimHash vote-accumulation A/B: L1 table-lookup vs tableless AVX2 bit-expansion.
//!
//! `fingerprint::apply_hash_votes` accumulates 64 ±1 votes per shingle window into
//! `[i32;64]` bit-weights. The shipped form does 8 `VOTE_TABLE[byte]` L1 loads per
//! window (a load-dependency chain: extract byte → load 32 B row → add). This runs
//! per 3-token window × every document at index time (re-embedding decision, 130k×).
//!
//! Structurally-different primitive (SWAR/SIMD, no table): expand each byte's 8 bits
//! directly to a ±1 `i32x8` with `broadcast(byte) & [1,2,4,…,128]`, `cmpeq` → mask,
//! `-(2·mask+1)` → ±1, then accumulate. Removes the 8 table loads/window (no L1 load
//! port pressure, no gather-like dependency) in favour of ALU. BIT-IDENTICAL: the
//! ±1 landing in `bit_weights[8j+k]` is the same value, so the output hash is
//! byte-for-byte equal (parity asserts equality ∀ input).
//!
//! Arms: `table_i32` (= shipped) vs `simd_expand`. Swept over token count (doc size).
//!
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-core --profile release --bench simhash_vote_simd
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

const SHINGLE_SIZE: usize = 3;
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

static VOTE_TABLE: [[i32; 8]; 256] = build_vote_table();

const fn build_vote_table() -> [[i32; 8]; 256] {
    let mut table = [[0_i32; 8]; 256];
    let mut byte = 0;
    while byte < 256 {
        let mut k = 0;
        while k < 8 {
            table[byte][k] = if ((byte >> k) & 1) == 0 { -1 } else { 1 };
            k += 1;
        }
        byte += 1;
    }
    table
}

fn hash_token_window(window: &[&str]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for token in window {
        for byte in token.bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash ^= u64::from(b' ');
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// SHIPPED: 8 table loads + 8-wide adds.
#[inline]
fn apply_votes_table(hash: u64, bit_weights: &mut [i32; 64]) {
    for j in 0..8 {
        let byte = ((hash >> (8 * j)) & 0xFF) as usize;
        let votes = &VOTE_TABLE[byte];
        let base = 8 * j;
        for k in 0..8 {
            bit_weights[base + k] += votes[k];
        }
    }
}

/// CANDIDATE: tableless AVX2 bit→±1 expansion (no L1 table load).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn apply_votes_simd_avx2(hash: u64, bit_weights: &mut [i32; 64]) {
    use std::arch::x86_64::{
        _mm256_add_epi32, _mm256_and_si256, _mm256_cmpeq_epi32, _mm256_loadu_si256,
        _mm256_set1_epi32, _mm256_set_epi32, _mm256_setzero_si256, _mm256_storeu_si256,
        _mm256_sub_epi32,
    };
    // lane k holds 1<<k (lane 0 = least-significant bit), matching VOTE_TABLE[b][k]=bit k.
    let bitsel = _mm256_set_epi32(128, 64, 32, 16, 8, 4, 2, 1);
    let ones = _mm256_set1_epi32(1);
    let zero = _mm256_setzero_si256();
    let ptr = bit_weights.as_mut_ptr();
    for j in 0..8 {
        let byte = ((hash >> (8 * j)) & 0xFF) as i32;
        let b = _mm256_set1_epi32(byte);
        let m = _mm256_cmpeq_epi32(_mm256_and_si256(b, bitsel), bitsel); // -1 set, 0 clear
        // vote = -(2*m + 1): set -> +1, clear -> -1
        let vote = _mm256_sub_epi32(zero, _mm256_add_epi32(_mm256_add_epi32(m, m), ones));
        let dst = ptr.add(8 * j).cast();
        let cur = _mm256_loadu_si256(dst);
        _mm256_storeu_si256(dst, _mm256_add_epi32(cur, vote));
    }
}

#[inline]
fn apply_votes_simd(hash: u64, bit_weights: &mut [i32; 64]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by runtime avx2 detection.
            #[allow(unsafe_code)]
            unsafe {
                apply_votes_simd_avx2(hash, bit_weights);
            }
            return;
        }
    }
    apply_votes_table(hash, bit_weights);
}

fn hash_from_weights(bit_weights: &[i32; 64]) -> u64 {
    let mut h = 0_u64;
    for (bit, w) in bit_weights.iter().enumerate() {
        if *w > 0 {
            h |= 1_u64 << bit;
        }
    }
    h
}

fn simhash(text: &str, simd: bool) -> u64 {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    if tokens.is_empty() {
        return 0;
    }
    let mut bw = [0_i32; 64];
    let apply = |h: u64, bw: &mut [i32; 64]| {
        if simd {
            apply_votes_simd(h, bw);
        } else {
            apply_votes_table(h, bw);
        }
    };
    if tokens.len() < SHINGLE_SIZE {
        for t in &tokens {
            apply(hash_token_window(&[t]), &mut bw);
        }
    } else {
        for w in tokens.windows(SHINGLE_SIZE) {
            apply(hash_token_window(w), &mut bw);
        }
    }
    hash_from_weights(&bw)
}

fn doc(tokens: usize) -> String {
    let words = [
        "retry", "backoff", "async", "await", "vector", "search", "embedding", "rerank",
        "token", "hash", "shingle", "fusion", "lexical", "index", "query", "score",
    ];
    let mut s = String::new();
    let mut r = 0x2545_f491_4f6c_dd1d_u64;
    for i in 0..tokens {
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        if i > 0 {
            s.push(' ');
        }
        s.push_str(words[(r as usize) % words.len()]);
    }
    s
}

fn apply_all(hashes: &[u64], simd: bool) -> u64 {
    let mut bw = [0_i32; 64];
    if simd {
        for &h in hashes {
            apply_votes_simd(h, &mut bw);
        }
    } else {
        for &h in hashes {
            apply_votes_table(h, &mut bw);
        }
    }
    hash_from_weights(&bw)
}

fn bench(c: &mut Criterion) {
    // Full simhash (real impact — includes the FNV window hashing, shared by both arms).
    let mut group = c.benchmark_group("simhash_vote_simd");
    group.sample_size(40);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_millis(1000));
    for &t in &[64usize, 256, 1024] {
        let text = doc(t);
        assert_eq!(
            simhash(&text, false),
            simhash(&text, true),
            "simd simhash diverged from table (t={t}) — must be bit-identical"
        );
        group.bench_with_input(BenchmarkId::new("table_i32", t), &text, |b, text| {
            b.iter(|| black_box(simhash(black_box(text), false)));
        });
        group.bench_with_input(BenchmarkId::new("simd_expand", t), &text, |b, text| {
            b.iter(|| black_box(simhash(black_box(text), true)));
        });
    }
    group.finish();

    // Isolated vote accumulation (the primitive's true speedup — no FNV dilution).
    let mut vg = c.benchmark_group("simhash_votes_only");
    vg.sample_size(40);
    vg.warm_up_time(Duration::from_millis(300));
    vg.measurement_time(Duration::from_millis(1000));
    for &n in &[256usize, 1024] {
        let mut r = 0x9e37_79b9_7f4a_7c15_u64;
        let hashes: Vec<u64> = (0..n)
            .map(|_| {
                r ^= r << 13;
                r ^= r >> 7;
                r ^= r << 17;
                r
            })
            .collect();
        assert_eq!(apply_all(&hashes, false), apply_all(&hashes, true));
        vg.bench_with_input(BenchmarkId::new("table_i32", n), &hashes, |b, h| {
            b.iter(|| black_box(apply_all(black_box(h), false)));
        });
        vg.bench_with_input(BenchmarkId::new("simd_expand", n), &hashes, |b, h| {
            b.iter(|| black_box(apply_all(black_box(h), true)));
        });
    }
    vg.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
