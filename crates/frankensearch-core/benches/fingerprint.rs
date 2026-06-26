//! SimHash vote-accumulation benchmark.
//!
//! `Fingerprint::compute` builds a 64-bit SimHash over 3-token shingles for
//! incremental re-embedding decisions (runs per document). The dominant inner
//! work is `apply_hash_votes`: for each window's hash, accumulate ±1 into 64 bit
//! counters. The committed code did `if (bit set) { +1 } else { -1 }` — a
//! data-dependent branch on effectively-random hash bits (~50% misprediction);
//! the new form is branchless (`2*b - 1`). This bench is the head-to-head over
//! the full per-document shingle sweep (`old` = branch, `new` = branchless).
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
//!   rch exec -- cargo bench -p frankensearch-core --bench fingerprint
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

const SHINGLE_SIZE: usize = 3;
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

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

fn apply_votes_old(hash: u64, bit_weights: &mut [i32; 64]) {
    for (bit, weight) in bit_weights.iter_mut().enumerate() {
        if hash & (1_u64 << bit) == 0 {
            *weight -= 1;
        } else {
            *weight += 1;
        }
    }
}

fn apply_votes_new(hash: u64, bit_weights: &mut [i32; 64]) {
    for (bit, weight) in bit_weights.iter_mut().enumerate() {
        let vote = 2 * ((hash >> bit) & 1) as i32 - 1;
        *weight += vote;
    }
}

fn simhash(tokens: &[&str], branchless: bool) -> u64 {
    let mut bit_weights = [0_i32; 64];
    for window in tokens.windows(SHINGLE_SIZE) {
        let h = hash_token_window(window);
        if branchless {
            apply_votes_new(h, &mut bit_weights);
        } else {
            apply_votes_old(h, &mut bit_weights);
        }
    }
    let mut semantic_hash = 0_u64;
    for (bit, weight) in bit_weights.iter().enumerate() {
        if *weight > 0 {
            semantic_hash |= 1_u64 << bit;
        }
    }
    semantic_hash
}

fn bench_fingerprint(c: &mut Criterion) {
    // ~300-token document (typical chunk fed to the fingerprinter).
    let text = "the quick brown fox jumps over the lazy dog while the engineer \
                refactors a retry backoff loop and the parser tokenizes every \
                identifier in the source file before the index writer commits "
        .repeat(15);
    let tokens: Vec<&str> = text.split_whitespace().collect();

    // Sanity: both produce the same SimHash.
    debug_assert_eq!(simhash(&tokens, false), simhash(&tokens, true));

    let mut g = c.benchmark_group("simhash_votes");
    g.bench_function("old", |b| {
        b.iter(|| black_box(simhash(black_box(&tokens), false)));
    });
    g.bench_function("new", |b| {
        b.iter(|| black_box(simhash(black_box(&tokens), true)));
    });
    g.finish();
}

criterion_group!(benches, bench_fingerprint);
criterion_main!(benches);
