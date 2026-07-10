//! Tokenizer hot-path A/B: ASCII byte fast-path in `next_char_from` vs the UTF-8-decode original.
//!
//! `next_char_from` is called for every character the cass tokenizer scans (at index time per doc
//! and query time per query), doing `text[off..].chars().next()` + `len_utf8()` — a full UTF-8
//! decode. The shipped version now returns single-byte ASCII directly (`b < 128 → (b as char,
//! off+1)`), skipping the decode for the common case (English, code, IDs). It is bit-identical
//! (`cass_compat::tests::next_char_from_ascii_matches_decode`), so token boundaries — hence
//! recall/ordering — are unchanged. This measures the speedup on realistic ASCII-heavy text.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- cargo bench -p frankensearch-lexical --features bench-internals --bench tokenizer_char_walk_ab
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_core::bench_support::paired_median_ratio;
use frankensearch_lexical::cass_compat::{cass_char_walk_fast, cass_char_walk_slow};

/// A realistic, mostly-ASCII document corpus (English prose + code identifiers + a little Unicode),
/// which is what the cass tokenizer actually sees.
fn corpus() -> String {
    let words = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "search",
        "index",
        "tokenizer",
        "bd-q3fy",
        "ID_42",
        "camelCase",
        "snake_case_name",
        "http",
        "vector",
        "embedding",
        "rerank",
        "café", // one accented word so the corpus isn't 100% ASCII
    ];
    let mut s = String::with_capacity(64 * 1024);
    let mut r = 0x2545_f491_4f6c_dd1d_u64;
    while s.len() < 48 * 1024 {
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
    let text = corpus();
    // Parity before timing (the shipped path must equal the decode path).
    assert_eq!(cass_char_walk_fast(&text), cass_char_walk_slow(&text));

    let run_fast = || {
        black_box(cass_char_walk_fast(black_box(&text)));
    };
    let run_slow = || {
        black_box(cass_char_walk_slow(black_box(&text)));
    };

    // NULL (slow vs slow) then lever (slow=ORIG vs fast). Ratio = fast/ORIG, <1.0 = fast-path wins.
    let null = paired_median_ratio(41, 4, run_slow, run_slow);
    let lever = paired_median_ratio(41, 4, run_slow, run_fast);
    eprintln!(
        "[null]  char_walk/{}KiB: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
        text.len() / 1024,
        null.median,
        null.p5,
        null.p95,
        null.rounds
    );
    eprintln!(
        "[lever] char_walk/{}KiB: fast/ORIG median {:.4} p5 {:.4} p95 {:.4} -> {}",
        text.len() / 1024,
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

    let mut g = c.benchmark_group("tokenizer_char_walk");
    g.sample_size(30);
    g.bench_function("decode", |b| b.iter(run_slow));
    g.bench_function("ascii_fast", |b| b.iter(run_fast));
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
