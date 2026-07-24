//! Route-next probe from the 2026-06-27 "count-free top-k is a mixed result" ledger:
//! `search_doc_ids` gates the count-free `execute_top_k` (`TopDocs` alone → block-max
//! WAND pruning) to ≤2-term syntax-free queries and otherwise runs the counted
//! `execute_query_with_offset` (`TopDocs` + `Count` → full scan, `total_count` then
//! discarded). The open question: does the WAND regression (measured ~2× on a broad
//! natural-language query with a corpus-saturating term) appear at exactly 3 terms —
//! i.e. is bumping the gate `term_count > 2` → `> 3` safe, or is per-term IDF/selectivity
//! awareness required to extend it?
//!
//! This A/Bs count-free vs counted for the SAME parsed query (ranked hits are identical —
//! asserted), over 2/3/4-term selective (rare-term) queries and queries that include the
//! corpus-saturating `search` term. Ratio is free/counted (`<1` = count-free faster).
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- \
//!   cargo bench -p frankensearch-lexical --profile release \
//!   --features bench-internals --bench count_free_gate_terms_ab
//! ```

use std::fmt::Write as _;
use std::hint::black_box;

use frankensearch_core::bench_support::{PairedRatio, paired_median_ratio};
use frankensearch_lexical::{execute_query_with_offset, execute_top_k};
use tantivy::query::QueryParser;
use tantivy::schema::{STORED, STRING, Schema, TEXT};
use tantivy::{Index, doc};

const N: usize = 40_000;
const VOCAB: u64 = 5_000;
const LIMIT: usize = 100;

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

struct Fixture {
    index: Index,
}

fn build_fixture() -> Fixture {
    let mut sb = Schema::builder();
    let _id = sb.add_text_field("id", STRING | STORED);
    let content = sb.add_text_field("content", TEXT | STORED);
    let schema = sb.build();
    let id = schema.get_field("id").unwrap();

    let index = Index::create_in_ram(schema);
    let mut writer: tantivy::IndexWriter = index.writer(120_000_000).expect("writer");
    for i in 0..N {
        let mut state = (i as u64).wrapping_add(1);
        // Every doc carries the corpus-saturating term `search` (100% doc frequency,
        // the low-IDF term that defeats WAND pruning), plus 13 rare terms drawn from a
        // large vocab (each ~N*13/VOCAB ≈ 100 docs ≈ 0.26% doc frequency → selective).
        let mut body = String::with_capacity(160);
        body.push_str("search ");
        for _ in 0..13 {
            let t =
                usize::try_from(xorshift(&mut state) % VOCAB).expect("VOCAB remainder fits usize");
            write!(&mut body, "term{t} ").expect("writing to String cannot fail");
        }
        writer
            .add_document(doc!(id => format!("doc-{i:06}"), content => body))
            .expect("add");
    }
    writer.commit().expect("commit");
    Fixture { index }
}

fn verdict(lever: &PairedRatio, null: &PairedRatio) -> &'static str {
    if lever.decidable_against(null) {
        if lever.median < 1.0 {
            "DECIDABLE (count-free faster)"
        } else {
            "DECIDABLE REGRESSION (count-free slower)"
        }
    } else {
        "INSIDE NULL FLOOR (not decidable)"
    }
}

fn main() {
    let fx = build_fixture();
    let reader = fx.index.reader().expect("reader");
    let searcher = reader.searcher();
    let content = fx.index.schema().get_field("content").unwrap();
    let parser = QueryParser::for_index(&fx.index, vec![content]);

    // (label, query string). Rare `termN` are selective; `search` is corpus-saturating.
    let queries = [
        ("sel2", "term7 term113"),
        ("sel3", "term7 term113 term509"),
        ("sel4", "term7 term113 term509 term888"),
        ("broad3_sat", "search term7 term113"),
        ("broad5_sat", "search term7 term113 term509 term888"),
    ];

    let inner = std::env::var("CF_GATE_AB_INNER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);

    for (label, qstr) in queries {
        let query = parser.parse_query(qstr).expect("parse");

        // Parity: count-free and counted return identical ranked hits (same top-k).
        let counted = execute_query_with_offset(&searcher, &*query, LIMIT, 0).expect("counted");
        let free = execute_top_k(&searcher, &*query, LIMIT, 0).expect("free");
        let counted_ids: Vec<_> = counted.hits.iter().map(|h| h.doc_address).collect();
        let free_ids: Vec<_> = free.iter().map(|h| h.doc_address).collect();
        assert_eq!(
            counted_ids, free_ids,
            "count-free and counted ranked hits differ for {label} ({qstr})"
        );

        let run_counted = || {
            black_box(execute_query_with_offset(&searcher, &*query, LIMIT, 0).expect("counted"));
        };
        let run_free = || {
            black_box(execute_top_k(&searcher, &*query, LIMIT, 0).expect("free"));
        };
        let null = paired_median_ratio(41, inner, run_counted, run_counted);
        let lever = paired_median_ratio(41, inner, run_counted, run_free);
        eprintln!(
            "[null]  cf_gate/{label}: median {:.4} p5 {:.4} p95 {:.4} (matches={}, {} rounds)",
            null.median,
            null.p5,
            null.p95,
            counted_ids.len(),
            null.rounds
        );
        eprintln!(
            "[lever] cf_gate/{label}: FREE/counted median {:.4} p5 {:.4} p95 {:.4} -> {}",
            lever.median,
            lever.p5,
            lever.p95,
            verdict(&lever, &null)
        );
    }
}
