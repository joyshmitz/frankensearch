//! Canonicalize A/B: ASCII-fast `truncate_to_chars` vs the `char_indices` forward scan.
//!
//! `truncate_to_chars` caps canonicalized doc/query text at `max_length` chars (default 2000). For
//! text longer than the cap, the original forward-scanned `char_indices` (`O(max_chars)` decodes) to
//! find the cut. The shipped version, when the first `max_chars` bytes are ASCII, cuts at byte
//! `max_chars` directly (a SIMD `is_ascii` scan, no decode). Byte-for-byte identical
//! (`truncate_to_chars_matches_slow`), so canonicalization — hence indexing/ranking — is unchanged.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- cargo bench -p frankensearch-core --features bench-internals --bench truncate_chars_ab
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_core::bench_support::paired_median_ratio;
use frankensearch_core::canonicalize::{truncate_to_chars_fast_bench, truncate_to_chars_slow};

const MAX_CHARS: usize = 2000; // production `DefaultCanonicalizer::max_length`

/// A ~16 KiB ASCII document (well over the cap, so both arms hit the truncation scan — the case the
/// fast-path targets; short text early-returns identically in both).
fn big_doc() -> String {
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
    ];
    let mut s = String::with_capacity(16 * 1024);
    let mut r = 0x2545_f491_4f6c_dd1d_u64;
    while s.len() < 16 * 1024 {
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
    let text = big_doc();
    assert_eq!(
        truncate_to_chars_fast_bench(&text, MAX_CHARS),
        truncate_to_chars_slow(&text, MAX_CHARS),
    );

    let run_fast = || {
        black_box(truncate_to_chars_fast_bench(black_box(&text), MAX_CHARS));
    };
    let run_slow = || {
        black_box(truncate_to_chars_slow(black_box(&text), MAX_CHARS));
    };

    // NULL (slow vs slow) then lever (slow=ORIG vs fast). Ratio = fast/ORIG, <1.0 = ASCII-fast wins.
    let null = paired_median_ratio(41, 4, run_slow, run_slow);
    let lever = paired_median_ratio(41, 4, run_slow, run_fast);
    eprintln!(
        "[null]  truncate: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
        null.median, null.p5, null.p95, null.rounds
    );
    eprintln!(
        "[lever] truncate: fast/ORIG median {:.4} p5 {:.4} p95 {:.4} -> {}",
        lever.median,
        lever.p5,
        lever.p95,
        if lever.decidable_against(&null) {
            if lever.median < 1.0 {
                "DECIDABLE WIN (ASCII fast-path faster)"
            } else {
                "DECIDABLE REGRESSION"
            }
        } else {
            "INSIDE NULL FLOOR (not decidable)"
        }
    );

    let mut g = c.benchmark_group("truncate_chars");
    g.sample_size(30);
    g.bench_function("char_indices", |b| b.iter(run_slow));
    g.bench_function("ascii_fast", |b| b.iter(run_fast));
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
