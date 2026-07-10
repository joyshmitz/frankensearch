//! Index-time A/B: bulk-copy `cass_build_preview` vs the char-by-char original.
//!
//! `cass_build_preview` runs per document at index time, truncating content to N chars (+ `…`). The
//! original pushed up to `max_chars` chars one at a time into an unallocated `String` — re-encoding
//! each char and growing the buffer through repeated reallocations. The shipped version does one
//! `char_indices` scan for the cut offset, then a single `push_str` bulk-copy into a pre-sized
//! buffer. Byte-for-byte identical (`cass_build_preview_matches_slow`), so no index/search behaviour
//! change.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- cargo bench -p frankensearch-lexical --features bench-internals --bench preview_build_ab
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_core::bench_support::paired_median_ratio;
use frankensearch_lexical::cass_compat::{cass_build_preview, cass_build_preview_slow};

const PREVIEW_MAX_CHARS: usize = 400; // the production cap (`cass_build_content_prefix_and_preview`)

/// A realistic message body ~2 KiB (longer than the preview cap, so both arms hit the truncation
/// path — the common case for real documents).
fn message() -> String {
    let words = [
        "the",
        "quick",
        "brown",
        "fox",
        "search",
        "index",
        "vector",
        "embedding",
        "rerank",
        "tokenizer",
        "document",
        "content",
        "preview",
        "bd-q3fy",
        "message",
        "café",
    ];
    let mut s = String::with_capacity(2048);
    let mut r = 0x9e37_79b9_7f4a_7c15_u64;
    while s.len() < 2000 {
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
    let content = message();
    assert_eq!(
        cass_build_preview(&content, PREVIEW_MAX_CHARS),
        cass_build_preview_slow(&content, PREVIEW_MAX_CHARS),
    );

    // A batch of previews per timed region (the real per-doc index workload).
    let run_fast = || {
        for _ in 0..256 {
            black_box(cass_build_preview(black_box(&content), PREVIEW_MAX_CHARS));
        }
    };
    let run_slow = || {
        for _ in 0..256 {
            black_box(cass_build_preview_slow(
                black_box(&content),
                PREVIEW_MAX_CHARS,
            ));
        }
    };

    // NULL (slow vs slow) then lever (slow=ORIG vs fast). Ratio = fast/ORIG, <1.0 = bulk-copy wins.
    let null = paired_median_ratio(41, 4, run_slow, run_slow);
    let lever = paired_median_ratio(41, 4, run_slow, run_fast);
    eprintln!(
        "[null]  preview: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
        null.median, null.p5, null.p95, null.rounds
    );
    eprintln!(
        "[lever] preview: fast/ORIG median {:.4} p5 {:.4} p95 {:.4} -> {}",
        lever.median,
        lever.p5,
        lever.p95,
        if lever.decidable_against(&null) {
            if lever.median < 1.0 {
                "DECIDABLE WIN (bulk-copy faster)"
            } else {
                "DECIDABLE REGRESSION"
            }
        } else {
            "INSIDE NULL FLOOR (not decidable)"
        }
    );

    let mut g = c.benchmark_group("preview_build");
    g.sample_size(30);
    g.bench_function("char_push", |b| b.iter(run_slow));
    g.bench_function("bulk_copy", |b| b.iter(run_fast));
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
