//! Validate the IDF/selectivity-aware count-free gate (route-next from the 2026-07-14
//! "count-free WAND gate is on the wrong axis" ledger). The count-free `execute_top_k`
//! (TopDocs → block-max WAND) beats the counted `execute_query_with_offset` (TopDocs +
//! Count full scan) precisely when a query has a SELECTIVE anchor term (skewed scores →
//! WAND prunes) and loses on FLAT mid-IDF disjunctions (no anchor → WAND can't prune).
//!
//! This fixture carries THREE term-frequency classes so both regimes appear in one
//! binary: `search` (100% doc frequency, saturating), `commonN` (~50%, mid-IDF, the flat
//! pathology), `rareN` (~0.2%, selective anchors). For each query it prints the parsed
//! query's `min(doc_freq)` (the candidate gate signal) and the measured count-free-vs-
//! counted ratio, so we can check whether `min(doc_freq) < threshold` picks the faster
//! collector for every shape — including the crucial MIXED case (mid-IDF + one rare anchor).
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- \
//!   cargo bench -p frankensearch-lexical --profile release \
//!   --features bench-internals --bench count_free_idf_gate_ab
//! ```

use std::hint::black_box;

use frankensearch_core::bench_support::{PairedRatio, paired_median_ratio};
use frankensearch_lexical::{execute_query_with_offset, execute_top_k};
use tantivy::query::QueryParser;
use tantivy::schema::{STORED, STRING, Schema, TEXT};
use tantivy::{Index, Term, doc};

const N: usize = 50_000;
const RARE_VOCAB: usize = 5_000;
const COMMON_VOCAB: usize = 8;
const LIMIT: usize = 100;

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn build_index() -> Index {
    let mut sb = Schema::builder();
    let _id = sb.add_text_field("id", STRING | STORED);
    let content = sb.add_text_field("content", TEXT | STORED);
    let schema = sb.build();
    let id = schema.get_field("id").unwrap();

    let index = Index::create_in_ram(schema);
    let mut writer: tantivy::IndexWriter = index.writer(150_000_000).expect("writer");
    for i in 0..N {
        let mut state = (i as u64).wrapping_add(1);
        let mut body = String::with_capacity(220);
        // Saturating term (100% doc frequency).
        body.push_str("search ");
        // Mid-IDF terms: each `commonN` present in ~50% of docs (independent coin flips)
        // → the flat-score pathology when a query is all-common.
        for j in 0..COMMON_VOCAB {
            if xorshift(&mut state) & 1 == 1 {
                body.push_str(&format!("common{j} "));
            }
        }
        // Rare/selective terms: 10 draws from a 5k vocab → each ~0.2% doc frequency.
        for _ in 0..10 {
            let t = (xorshift(&mut state) as usize) % RARE_VOCAB;
            body.push_str(&format!("rare{t} "));
        }
        writer
            .add_document(doc!(id => format!("doc-{i:06}"), content => body))
            .expect("add");
    }
    writer.commit().expect("commit");
    index
}

fn verdict(lever: &PairedRatio, null: &PairedRatio) -> &'static str {
    if lever.decidable_against(null) {
        if lever.median < 1.0 {
            "COUNT-FREE FASTER"
        } else {
            "COUNT-FREE SLOWER (regression)"
        }
    } else {
        "neutral (inside null floor)"
    }
}

fn main() {
    let index = build_index();
    let reader = index.reader().expect("reader");
    let searcher = reader.searcher();
    let content = index.schema().get_field("content").unwrap();
    let parser = QueryParser::for_index(&index, vec![content]);

    // Candidate gate threshold: "has a selective anchor" = min per-term doc_freq below 1%
    // of the corpus. Rare terms (~0.2%) pass; mid-IDF (~50%) and saturating (100%) fail.
    let selective_threshold = (N as u64) / 100;

    let queries = [
        ("flat3_common", "common0 common1 common2"),
        ("flat5_sat_common", "search common0 common1 common2 common3"),
        ("skewed3_sat_rare", "search rare7 rare113"),
        ("mixed3_common_rare", "common0 common1 rare7"),
        ("sel3_rare", "rare7 rare113 rare509"),
    ];

    let inner = std::env::var("CF_IDF_AB_INNER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);

    eprintln!("threshold(min doc_freq < {selective_threshold}) => predict COUNT-FREE");
    for (label, qstr) in queries {
        let query = parser.parse_query(qstr).expect("parse");

        // Candidate gate signal: min doc_freq over the parsed query's indexed terms.
        let mut min_df = u64::MAX;
        query.query_terms(&mut |term: &Term, _pos| {
            let df = searcher.doc_freq(term).unwrap_or(0);
            min_df = min_df.min(df);
        });
        let predict_free = min_df < selective_threshold;

        // Ranking parity: WAND (count-free) can break score TIES differently than the
        // full scan (counted), so the top-k doc set is not guaranteed identical. Report it
        // per query rather than assert — a differing set is itself a key finding.
        let counted = execute_query_with_offset(&searcher, &*query, LIMIT, 0).expect("counted");
        let free = execute_top_k(&searcher, &*query, LIMIT, 0).expect("free");
        let counted_ids: Vec<_> = counted.hits.iter().map(|h| h.doc_address).collect();
        let free_ids: Vec<_> = free.iter().map(|h| h.doc_address).collect();
        let hits_match = counted_ids == free_ids;
        let counted_set: std::collections::HashSet<_> = counted_ids.iter().copied().collect();
        let free_set: std::collections::HashSet<_> = free_ids.iter().copied().collect();
        let set_match = counted_set == free_set;

        let run_counted = || {
            black_box(execute_query_with_offset(&searcher, &*query, LIMIT, 0).expect("counted"));
        };
        let run_free = || {
            black_box(execute_top_k(&searcher, &*query, LIMIT, 0).expect("free"));
        };
        let null = paired_median_ratio(41, inner, run_counted, run_counted);
        let lever = paired_median_ratio(41, inner, run_counted, run_free);
        eprintln!(
            "[{label}] min_df={min_df} predict={} order_id={} set_id={} | free/counted median {:.4} p5 {:.4} p95 {:.4} (matches={}, null p5 {:.4} p95 {:.4}) -> {}",
            if predict_free { "FREE" } else { "counted" },
            hits_match,
            set_match,
            lever.median,
            lever.p5,
            lever.p95,
            counted_ids.len(),
            null.p5,
            null.p95,
            verdict(&lever, &null),
        );
    }
}
