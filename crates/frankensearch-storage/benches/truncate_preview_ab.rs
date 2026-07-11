//! Ingest content-preview truncation A/B: `IngestPipeline` builds a `MAX_CONTENT_PREVIEW_CHARS`
//! (400) preview from each document's full canonical text via `truncate_chars`. The former
//! implementation did `value.chars().count()` — decoding EVERY char of the (typically long)
//! document — just to learn it exceeds 400, then `take(400)` decoded again. Since real documents
//! are far longer than a 400-char preview, this paid a full-document UTF-8 decode per doc. The
//! lever walks at most `max_chars + 1` chars via `char_indices().nth(max_chars)` and slices.
//!
//! Byte-identical (`truncate_chars == truncate_chars_slow`, proven by `truncate_chars_matches_slow`
//! and re-asserted here). Single-threaded, so the paired ratio is contention-robust. Gate on the
//! MEDIAN of CAND/ORIG against the A/A null floor.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- cargo bench -p frankensearch-storage --features bench-internals --bench truncate_preview_ab
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_core::bench_support::paired_median_ratio;
use frankensearch_storage::pipeline::{truncate_chars, truncate_chars_slow};

const PREVIEW: usize = 400; // MAX_CONTENT_PREVIEW_CHARS

fn ascii_doc(chars: usize) -> String {
    "the quick brown fox jumps over the lazy dog. "
        .chars()
        .cycle()
        .take(chars)
        .collect()
}

fn multibyte_doc(chars: usize) -> String {
    "café déjà 日本語 τест 🚀 ".chars().cycle().take(chars).collect()
}

fn bench(c: &mut Criterion) {
    let inner: u32 = std::env::var("TRUNC_AB_INNER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    // (label, doc): real documents are ≫ the 400-char preview, so all exercise the over-limit path.
    let docs = [
        ("ascii_2k", ascii_doc(2_000)),
        ("ascii_16k", ascii_doc(16_000)),
        ("multibyte_16k", multibyte_doc(16_000)),
    ];

    for (label, doc) in &docs {
        assert_eq!(
            truncate_chars(doc, PREVIEW),
            truncate_chars_slow(doc, PREVIEW),
            "truncate_chars != slow for {label}"
        );

        let run_orig = || {
            black_box(truncate_chars_slow(black_box(doc), PREVIEW));
        };
        let run_cand = || {
            black_box(truncate_chars(black_box(doc), PREVIEW));
        };

        let null = paired_median_ratio(41, inner, run_orig, run_orig);
        let lever = paired_median_ratio(41, inner, run_orig, run_cand);
        eprintln!(
            "[null]  trunc/{label}: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
            null.median, null.p5, null.p95, null.rounds
        );
        eprintln!(
            "[lever] trunc/{label}: cand/ORIG median {:.4} p5 {:.4} p95 {:.4} -> {}",
            lever.median,
            lever.p5,
            lever.p95,
            if lever.decidable_against(&null) {
                if lever.median < 1.0 {
                    "DECIDABLE WIN"
                } else {
                    "DECIDABLE REGRESSION"
                }
            } else {
                "INSIDE NULL FLOOR (not decidable)"
            }
        );
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
