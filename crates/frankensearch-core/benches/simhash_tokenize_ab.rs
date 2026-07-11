//! Ingest A/B: byte-fast whitespace tokenizer vs `str::split_whitespace` in the simhash.
//!
//! `semantic_simhash_text` (per document at ingest, inside `DocumentFingerprint::compute`) tokenizes
//! by whitespace. The original `str::split_whitespace` decoded every char to test
//! `char::is_whitespace`; the shipped tokenizer classifies ASCII bytes with a cheap byte test
//! (`is_ascii_whitespace() || b == 0x0B`, matching `char::is_whitespace` on ASCII) and decodes only
//! non-ASCII lead bytes. Token-identical (`split_whitespace_fast_matches_std`), so the simhash — and
//! all dedup decisions — are unchanged. This isolates the tokenization scan (fold token lengths, no
//! alloc); it is ~15-25% of `semantic_simhash_text` (the no-alloc window FNV hashing is the rest).
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- cargo bench -p frankensearch-core --features bench-internals --bench simhash_tokenize_ab
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_core::bench_support::paired_median_ratio;
use frankensearch_core::fingerprint::{split_whitespace_fast_lensum, split_whitespace_std_lensum};

/// A ~4 KiB all-ASCII document (the case the fast path targets; mixed docs stay correct via the
/// per-byte Unicode fallback — covered by `split_whitespace_fast_matches_std`).
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
    let mut r = 0x2545_f491_4f6c_dd1d_u64;
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
    assert_eq!(
        split_whitespace_fast_lensum(&text),
        split_whitespace_std_lensum(&text),
    );

    let run_fast = || {
        black_box(split_whitespace_fast_lensum(black_box(&text)));
    };
    let run_slow = || {
        black_box(split_whitespace_std_lensum(black_box(&text)));
    };

    // NULL (std vs std) then lever (std=ORIG vs fast). Ratio = fast/ORIG, <1.0 = byte-fast wins.
    let null = paired_median_ratio(41, 4, run_slow, run_slow);
    let lever = paired_median_ratio(41, 4, run_slow, run_fast);
    eprintln!(
        "[null]  simhash_tok: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
        null.median, null.p5, null.p95, null.rounds
    );
    eprintln!(
        "[lever] simhash_tok: fast/ORIG median {:.4} p5 {:.4} p95 {:.4} -> {}",
        lever.median,
        lever.p5,
        lever.p95,
        if lever.decidable_against(&null) {
            if lever.median < 1.0 {
                "DECIDABLE WIN (byte-fast tokenizer faster)"
            } else {
                "DECIDABLE REGRESSION"
            }
        } else {
            "INSIDE NULL FLOOR (not decidable)"
        }
    );

    let mut g = c.benchmark_group("simhash_tokenize");
    g.sample_size(30);
    g.bench_function("split_whitespace", |b| b.iter(run_slow));
    g.bench_function("byte_fast", |b| b.iter(run_fast));
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
