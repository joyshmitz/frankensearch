//! Index-time A/B: ASCII-fast `cass_generate_edge_ngrams` vs the char_indices-re-decode original.
//!
//! `cass_generate_edge_ngrams` runs per document at index time on the ≤ 4 KiB content prefix. The
//! original re-decoded every word with `char_indices` to find prefix boundaries — a second UTF-8
//! decode on top of the `split`. The shipped version slices ASCII words' prefixes directly from byte
//! positions (char boundary == byte position), skipping the re-decode; non-ASCII words fall back.
//! Byte-for-byte identical output (`cass_generate_edge_ngrams_matches_slow`), so no index/search
//! change.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- cargo bench -p frankensearch-lexical --features bench-internals --bench edge_ngrams_ab
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_core::bench_support::paired_median_ratio;
use frankensearch_lexical::cass_compat::{
    cass_generate_edge_ngrams, cass_generate_edge_ngrams_slow,
};

/// A realistic ~4 KiB content prefix (English + code identifiers + a little Unicode) — what the
/// production `cass_generate_edge_ngrams(prefix_source)` call actually sees.
fn prefix_text() -> String {
    let words = [
        "the",
        "quick",
        "brown",
        "fox",
        "search",
        "index",
        "tokenizer",
        "vector",
        "embedding",
        "rerank",
        "document",
        "content",
        "bd-q3fy",
        "ID_42",
        "camelCase",
        "snake_case_name",
        "café", // one accented word so it isn't 100% ASCII
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
    let text = prefix_text();
    assert_eq!(
        cass_generate_edge_ngrams(&text),
        cass_generate_edge_ngrams_slow(&text),
    );

    let run_fast = || {
        black_box(cass_generate_edge_ngrams(black_box(&text)));
    };
    let run_slow = || {
        black_box(cass_generate_edge_ngrams_slow(black_box(&text)));
    };

    // NULL (slow vs slow) then lever (slow=ORIG vs fast). Ratio = fast/ORIG, <1.0 = ASCII-fast wins.
    let null = paired_median_ratio(41, 4, run_slow, run_slow);
    let lever = paired_median_ratio(41, 4, run_slow, run_fast);
    eprintln!(
        "[null]  edge_ngrams: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
        null.median, null.p5, null.p95, null.rounds
    );
    eprintln!(
        "[lever] edge_ngrams: fast/ORIG median {:.4} p5 {:.4} p95 {:.4} -> {}",
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

    let mut g = c.benchmark_group("edge_ngrams");
    g.sample_size(30);
    g.bench_function("char_indices", |b| b.iter(run_slow));
    g.bench_function("ascii_fast", |b| b.iter(run_fast));
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
