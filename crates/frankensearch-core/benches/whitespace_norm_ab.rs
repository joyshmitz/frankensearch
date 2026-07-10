//! Index-time A/B: byte-fast `normalize_whitespace` vs the char-by-char original.
//!
//! `normalize_whitespace` runs on **every document** during canonicalization (collapse whitespace
//! runs to a single space, trim). The original decoded every char via `text.chars()`, ran the
//! Unicode `is_whitespace()` per char, and re-encoded each kept char. The shipped version scans
//! bytes with an ASCII fast-path (cheap byte whitespace test, no decode), decoding only non-ASCII
//! lead bytes. Byte-for-byte identical (`normalize_whitespace_matches_slow`), so canonicalization —
//! hence the tokens/embeddings that feed indexing and ranking — is unchanged.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- cargo bench -p frankensearch-core --features bench-internals --bench whitespace_norm_ab
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_core::bench_support::paired_median_ratio;
use frankensearch_core::canonicalize::{
    normalize_whitespace_fast_bench, normalize_whitespace_slow,
};

/// A realistic ~4 KiB document body (English prose with the usual whitespace runs, tabs, newlines,
/// and a little Unicode) — what canonicalization actually sees post markdown-strip.
fn document() -> String {
    let chunks = [
        "the quick brown fox   jumps over the lazy dog. ",
        "search index vector\tembedding rerank tokenizer\n\n",
        "document content preview  bd-q3fy  ID_42   ",
        "camelCase snake_case_name   café déjà vu\n",
    ];
    let mut s = String::with_capacity(4096);
    let mut r = 0x9e37_79b9_7f4a_7c15_u64;
    while s.len() < 4000 {
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        let idx = usize::try_from(r % chunks.len() as u64).unwrap_or(0);
        s.push_str(chunks[idx]);
    }
    s
}

fn bench(c: &mut Criterion) {
    let text = document();
    assert_eq!(
        normalize_whitespace_fast_bench(&text),
        normalize_whitespace_slow(&text),
    );

    let run_fast = || {
        black_box(normalize_whitespace_fast_bench(black_box(&text)));
    };
    let run_slow = || {
        black_box(normalize_whitespace_slow(black_box(&text)));
    };

    // NULL (slow vs slow) then lever (slow=ORIG vs fast). Ratio = fast/ORIG, <1.0 = byte-fast wins.
    let null = paired_median_ratio(41, 4, run_slow, run_slow);
    let lever = paired_median_ratio(41, 4, run_slow, run_fast);
    eprintln!(
        "[null]  ws_norm: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
        null.median, null.p5, null.p95, null.rounds
    );
    eprintln!(
        "[lever] ws_norm: fast/ORIG median {:.4} p5 {:.4} p95 {:.4} -> {}",
        lever.median,
        lever.p5,
        lever.p95,
        if lever.decidable_against(&null) {
            if lever.median < 1.0 {
                "DECIDABLE WIN (byte fast-path faster)"
            } else {
                "DECIDABLE REGRESSION"
            }
        } else {
            "INSIDE NULL FLOOR (not decidable)"
        }
    );

    let mut g = c.benchmark_group("whitespace_norm");
    g.sample_size(30);
    g.bench_function("char_decode", |b| b.iter(run_slow));
    g.bench_function("byte_fast", |b| b.iter(run_fast));
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
