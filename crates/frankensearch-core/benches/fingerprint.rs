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

static VOTE_TABLE: [[i32; 8]; 256] = build_vote_table();
const fn build_vote_table() -> [[i32; 8]; 256] {
    let mut table = [[0_i32; 8]; 256];
    let mut byte = 0;
    while byte < 256 {
        let mut k = 0;
        while k < 8 {
            table[byte][k] = 2 * ((byte >> k) & 1) as i32 - 1;
            k += 1;
        }
        byte += 1;
    }
    table
}
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

// mode: 0 = branch, 1 = branchless (current main), 2 = table-driven
fn simhash(tokens: &[&str], mode: u8) -> u64 {
    let mut bit_weights = [0_i32; 64];
    for window in tokens.windows(SHINGLE_SIZE) {
        let h = hash_token_window(window);
        match mode {
            0 => apply_votes_old(h, &mut bit_weights),
            1 => apply_votes_new(h, &mut bit_weights),
            _ => apply_votes_table(h, &mut bit_weights),
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

    // Sanity: all three modes produce the same SimHash.
    debug_assert_eq!(simhash(&tokens, 0), simhash(&tokens, 1));
    debug_assert_eq!(simhash(&tokens, 1), simhash(&tokens, 2));

    // The kept `branchless` (current main) vs the candidate `table` (byte-indexed).
    let mut g = c.benchmark_group("simhash_votes");
    g.bench_function("branchless", |b| {
        b.iter(|| black_box(simhash(black_box(&tokens), 1)));
    });
    g.bench_function("table", |b| {
        b.iter(|| black_box(simhash(black_box(&tokens), 2)));
    });
    g.finish();
}

criterion_group!(benches, bench_fingerprint);
criterion_main!(benches);
