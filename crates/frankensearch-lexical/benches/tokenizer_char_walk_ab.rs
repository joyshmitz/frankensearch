//! Tokenizer hot-path A/B: ASCII byte fast-path in `next_char_from` vs the UTF-8-decode original.
//!
//! `next_char_from` is called for every character the cass tokenizer scans (at index time per doc
//! and query time per query), doing `text[off..].chars().next()` + `len_utf8()` — a full UTF-8
//! decode. The shipped version now returns single-byte ASCII directly (`b < 128 → (b as char,
//! off+1)`), skipping the decode for the common case (English, code, IDs). It is bit-identical
//! (`cass_compat::tests::next_char_from_ascii_matches_decode`), so token boundaries — hence
//! recall/ordering — are unchanged. This measures the speedup on realistic ASCII-heavy text.
//!
//! The `boolean_operator_reparse` group separately compares the full cass query builder's
//! production two-parse path with a retained single-parse candidate. Both arms build and drop
//! the same Tantivy query trees; a paired A/A null control determines whether the difference is
//! resolvable on the worker.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- cargo bench -p frankensearch-lexical --features bench-internals \
//!     --profile release --bench tokenizer_char_walk_ab
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_core::bench_support::paired_median_ratio;
use frankensearch_lexical::cass_compat::{
    CassQueryFilters, cass_build_schema, cass_build_tantivy_query,
    cass_build_tantivy_query_single_parse, cass_char_walk_fast, cass_char_walk_slow,
    cass_fields_from_schema,
};

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

fn bench_boolean_operator_reparse(c: &mut Criterion) {
    let schema = cass_build_schema();
    let fields = cass_fields_from_schema(&schema).expect("cass fields");
    let filters = CassQueryFilters::default();
    let queries = [
        "auth token cache",
        "auth AND token OR cache",
        "\"exact phrase\" OR cache",
        "-legacy identifier",
        "search 搜索 NOT stale",
        "workspace agent source path conversation title content AND embedding OR tokenizer NOT deprecated",
    ];

    for query in queries {
        assert_eq!(
            format!("{:?}", cass_build_tantivy_query(query, &filters, &fields)),
            format!(
                "{:?}",
                cass_build_tantivy_query_single_parse(query, &filters, &fields)
            ),
            "query tree changed for {query:?}"
        );
    }

    let run_legacy = || {
        for query in queries {
            black_box(cass_build_tantivy_query(
                black_box(query),
                black_box(&filters),
                black_box(&fields),
            ));
        }
    };
    let run_single_parse = || {
        for query in queries {
            black_box(cass_build_tantivy_query_single_parse(
                black_box(query),
                black_box(&filters),
                black_box(&fields),
            ));
        }
    };

    // NULL (legacy vs legacy), then a full-builder A/B. Both arms construct and
    // drop the same Tantivy query tree; only the candidate omits the second
    // boolean-token parse/allocation.
    // Keep each arm in the millisecond range so worker scheduling jitter does
    // not dominate this small full-builder delta (the earlier inner=16 run
    // produced an unusably wide 0.52..2.18 A/A band).
    let null = paired_median_ratio(41, 256, run_legacy, run_legacy);
    let lever = paired_median_ratio(41, 256, run_legacy, run_single_parse);
    eprintln!(
        "[null]  boolean_operator_reparse/{}queries: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
        queries.len(),
        null.median,
        null.p5,
        null.p95,
        null.rounds
    );
    eprintln!(
        "[lever] boolean_operator_reparse/{}queries: single_parse_builder/ORIG median {:.4} p5 {:.4} p95 {:.4} -> {}",
        queries.len(),
        lever.median,
        lever.p5,
        lever.p95,
        if lever.decidable_against(&null) {
            if lever.median < 1.0 {
                "DECIDABLE WIN (full builder, duplicate parse removed)"
            } else {
                "DECIDABLE REGRESSION"
            }
        } else {
            "INSIDE NULL FLOOR (not decidable)"
        }
    );

    let mut group = c.benchmark_group("boolean_operator_reparse");
    group.sample_size(30);
    group.bench_function("legacy_builder_two_parses", |b| b.iter(run_legacy));
    group.bench_function("single_parse_builder", |b| b.iter(run_single_parse));
    group.finish();
}

criterion_group!(benches, bench, bench_boolean_operator_reparse);
criterion_main!(benches);
