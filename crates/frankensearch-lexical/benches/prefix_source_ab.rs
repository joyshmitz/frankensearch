//! Index-time A/B: O(1) `cass_prefix_source` (backward `is_char_boundary` walk) vs the O(n) forward
//! `char_indices` scan.
//!
//! `cass_prefix_source` runs per document at index time, taking the ≤ `max_bytes` (4 KiB) char-
//! boundary prefix of content that feeds edge-ngram generation. For content over the cap, the
//! original decoded ~`max_bytes` chars forward just to find the boundary near byte 4096; the shipped
//! version walks back from `max_bytes` at most one UTF-8 char width (≤ 4 `is_char_boundary` checks).
//! Byte-for-byte identical prefix (`cass_prefix_source_matches_slow`), so no index/search change.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- cargo bench -p frankensearch-lexical --features bench-internals --bench prefix_source_ab
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_core::bench_support::paired_median_ratio;
use frankensearch_lexical::cass_compat::{cass_prefix_source_fast_bench, cass_prefix_source_slow};

const MAX_BYTES: usize = 4 * 1024; // production `CONTENT_PREFIX_MAX_BYTES`

/// A ~64 KiB document (well over the cap, so both arms take the truncation path — the case the
/// lever targets; small docs early-return identically in both).
fn big_content() -> String {
    let words = [
        "the",
        "quick",
        "brown",
        "fox",
        "search",
        "index",
        "vector",
        "content",
        "café",
        "日本語",
    ];
    let mut s = String::with_capacity(64 * 1024);
    let mut r = 0x2545_f491_4f6c_dd1d_u64;
    while s.len() < 64 * 1024 {
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
    let content = big_content();
    assert_eq!(
        cass_prefix_source_fast_bench(&content, MAX_BYTES),
        cass_prefix_source_slow(&content, MAX_BYTES).len(),
    );

    // A batch per timed region (per-doc index workload).
    let run_fast = || {
        for _ in 0..256 {
            black_box(cass_prefix_source_fast_bench(
                black_box(&content),
                MAX_BYTES,
            ));
        }
    };
    let run_slow = || {
        for _ in 0..256 {
            black_box(cass_prefix_source_slow(black_box(&content), MAX_BYTES).len());
        }
    };

    // NULL (slow vs slow) then lever (slow=ORIG vs fast). Ratio = fast/ORIG, <1.0 = O(1) walk wins.
    let null = paired_median_ratio(41, 4, run_slow, run_slow);
    let lever = paired_median_ratio(41, 4, run_slow, run_fast);
    eprintln!(
        "[null]  prefix_source: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
        null.median, null.p5, null.p95, null.rounds
    );
    eprintln!(
        "[lever] prefix_source: fast/ORIG median {:.4} p5 {:.4} p95 {:.4} -> {}",
        lever.median,
        lever.p5,
        lever.p95,
        if lever.decidable_against(&null) {
            if lever.median < 1.0 {
                "DECIDABLE WIN (O(1) boundary walk faster)"
            } else {
                "DECIDABLE REGRESSION"
            }
        } else {
            "INSIDE NULL FLOOR (not decidable)"
        }
    );

    let mut g = c.benchmark_group("prefix_source");
    g.sample_size(30);
    g.bench_function("forward_scan", |b| b.iter(run_slow));
    g.bench_function("boundary_walk", |b| b.iter(run_fast));
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
