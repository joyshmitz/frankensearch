//! Ingest A/B: ASCII-fast `char_count` vs `text.chars().count()`.
//!
//! `DocumentFingerprint::compute` runs per document at ingest and calls `char_count(text)` for its
//! dedup delta metadata — previously `text.chars().count()`, a *second* full-text UTF-8 decode on
//! top of the one `semantic_simhash_text` already does. The shipped version returns `text.len()` for
//! ASCII (a SIMD `is_ascii` byte scan), decoding only non-ASCII. Identical count
//! (`char_count_matches_slow`), so dedup decisions are unchanged.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- cargo bench -p frankensearch-core --features bench-internals --bench char_count_ab
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_core::bench_support::paired_median_ratio;
use frankensearch_core::fingerprint::{char_count_fast_bench, char_count_slow};

/// A ~4 KiB all-ASCII document — the case the fast-path targets (English/code). The `is_ascii` fast
/// path only fires when the *whole* text is ASCII; a doc with any non-ASCII char falls back to the
/// full decode (correctness covered by `char_count_matches_slow`), so this measures the fast path.
fn document() -> String {
    let words = [
        "the",
        "quick",
        "brown",
        "fox",
        "search",
        "index",
        "document",
        "content",
        "tokenizer",
        "vector",
        "identifier",
    ];
    let mut s = String::with_capacity(4096);
    let mut r = 0x9e37_79b9_7f4a_7c15_u64;
    while s.len() < 4000 {
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        let idx = usize::try_from(r % words.len() as u64).unwrap_or(0);
        s.push_str(words[idx]);
        s.push(' ');
    }
    s
}

fn bench(c: &mut Criterion) {
    let text = document();
    assert_eq!(char_count_fast_bench(&text), char_count_slow(&text));

    // A batch per timed region (per-doc ingest workload).
    let run_fast = || {
        let mut acc = 0usize;
        for _ in 0..256 {
            acc = acc.wrapping_add(char_count_fast_bench(black_box(&text)));
        }
        black_box(acc);
    };
    let run_slow = || {
        let mut acc = 0usize;
        for _ in 0..256 {
            acc = acc.wrapping_add(char_count_slow(black_box(&text)));
        }
        black_box(acc);
    };

    // NULL (slow vs slow) then lever (slow=ORIG vs fast). Ratio = fast/ORIG, <1.0 = ASCII-fast wins.
    let null = paired_median_ratio(41, 4, run_slow, run_slow);
    let lever = paired_median_ratio(41, 4, run_slow, run_fast);
    eprintln!(
        "[null]  char_count: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
        null.median, null.p5, null.p95, null.rounds
    );
    eprintln!(
        "[lever] char_count: fast/ORIG median {:.4} p5 {:.4} p95 {:.4} -> {}",
        lever.median,
        lever.p5,
        lever.p95,
        if lever.decidable_against(&null) {
            if lever.median < 1.0 {
                "DECIDABLE WIN (ASCII byte scan faster)"
            } else {
                "DECIDABLE REGRESSION"
            }
        } else {
            "INSIDE NULL FLOOR (not decidable)"
        }
    );

    let mut g = c.benchmark_group("char_count");
    g.sample_size(30);
    g.bench_function("chars_count", |b| b.iter(run_slow));
    g.bench_function("is_ascii_len", |b| b.iter(run_fast));
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
