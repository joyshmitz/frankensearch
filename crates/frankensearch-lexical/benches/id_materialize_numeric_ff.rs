//! Hypothesis test for the ONE un-rejected lexical materialization lever
//! (NEGATIVE_EVIDENCE.md route-next after the str-FAST-field reject):
//!
//!   `search_doc_ids` materializes each hit's `doc_id` via `searcher.doc(addr)`,
//!   which **decompresses the whole stored document** (id + content + title +
//!   metadata) just to read `id`. The str-FAST-field variant was REJECTED
//!   (2.65–18.3× slower) because `StrColumn::ord_to_str` does a dictionary
//!   SSTable seek per hit. The route-next: a **numeric u64 fast field carrying a
//!   dense doc ordinal** (a flat packed column — NO dictionary) plus an external
//!   `ordinal -> doc_id` table (`Vec<String>`, O(1) index). Materialization then
//!   reads `ord = ff.first(local_doc)` (bit-unpack) and clones `table[ord]` once,
//!   skipping BOTH the docstore decompress AND the dictionary seek.
//!
//! This bench isolates that core read-cost question on a fresh in-RAM index
//! (no deletes/merges — the production version would need segment/delete-aware
//! ordinal mapping; this only measures whether the read path is worth it):
//!
//! - `docstore`  : `searcher.doc(addr)` -> read `id` field (current production).
//! - `numeric_ff`: read `ord` u64 fast field -> `table[ord]` -> clone.
//!
//! Both arms return identical `doc_id`s (asserted once before timing). Content
//! is `STORED` so the docstore arm pays a realistic decompress, matching the
//! production schema (`content` is `TEXT | STORED`).
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-lexical --profile release \
//!   --bench id_materialize_numeric_ff
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use tantivy::collector::TopDocs;
use tantivy::columnar::Column;
use tantivy::query::QueryParser;
use tantivy::schema::{FAST, STORED, STRING, Schema, TEXT, Value};
use tantivy::{DocAddress, Index, TantivyDocument, doc};

const N: usize = 20_000;
const KS: &[usize] = &[30, 100, 300, 1000];

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

const VOCAB: &[&str] = &[
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "search", "engine",
    "vector", "lexical", "ranking", "relevance", "document", "query", "hybrid", "fusion",
];

struct Fixture {
    index: Index,
    id_field: tantivy::schema::Field,
    table: Vec<String>,
}

fn build_fixture() -> Fixture {
    let mut sb = Schema::builder();
    let id_field = sb.add_text_field("id", STRING | STORED);
    let content = sb.add_text_field("content", TEXT | STORED);
    // Dense insertion ordinal as a numeric fast field (flat packed column, no
    // dictionary) AND stored (irrelevant to the fast read; kept for parity).
    let ord_field = sb.add_u64_field("ord", FAST | STORED);
    let schema = sb.build();

    let index = Index::create_in_ram(schema);
    let mut writer: tantivy::IndexWriter = index.writer(60_000_000).expect("writer");
    let mut table = Vec::with_capacity(N);
    for i in 0..N {
        let mut state = (i as u64).wrapping_add(1);
        let doc_id = format!("doc-{i:06}");
        // ~14-token body so the stored block has realistic decompress work, plus
        // a guaranteed high-fanout term ("search") in every doc.
        let mut body = String::with_capacity(160);
        body.push_str("search ");
        for _ in 0..13 {
            let w = VOCAB[(xorshift(&mut state) as usize) % VOCAB.len()];
            body.push_str(w);
            body.push(' ');
        }
        body.push_str(&doc_id);
        writer
            .add_document(doc!(
                id_field => doc_id.clone(),
                content => body,
                ord_field => i as u64,
            ))
            .expect("add");
        table.push(doc_id);
    }
    writer.commit().expect("commit");

    Fixture {
        index,
        id_field,
        table,
    }
}

fn materialize_docstore(
    searcher: &tantivy::Searcher,
    id_field: tantivy::schema::Field,
    hits: &[(f32, DocAddress)],
) -> Vec<String> {
    let mut out = Vec::with_capacity(hits.len());
    for &(_, addr) in hits {
        let doc: TantivyDocument = searcher.doc(addr).expect("doc");
        let id = doc
            .get_first(id_field)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_owned();
        out.push(id);
    }
    out
}

fn materialize_numeric_ff(
    searcher: &tantivy::Searcher,
    table: &[String],
    hits: &[(f32, DocAddress)],
) -> Vec<String> {
    // One column handle per segment (cheap; columnar slice, no dictionary).
    let seg_count = searcher.segment_readers().len();
    let mut cols: Vec<Option<Column<u64>>> = Vec::with_capacity(seg_count);
    for sr in searcher.segment_readers() {
        cols.push(sr.fast_fields().u64("ord").ok());
    }
    let mut out = Vec::with_capacity(hits.len());
    for &(_, addr) in hits {
        let seg = addr.segment_ord as usize;
        let ord = cols[seg]
            .as_ref()
            .and_then(|c| c.first(addr.doc_id))
            .expect("ord fast field") as usize;
        out.push(table[ord].clone());
    }
    out
}

fn bench(c: &mut Criterion) {
    let fx = build_fixture();
    let reader = fx.index.reader().expect("reader");
    let searcher = reader.searcher();
    let parser = QueryParser::for_index(
        &fx.index,
        vec![fx.index.schema().get_field("content").unwrap()],
    );
    let query = parser.parse_query("search").expect("query");

    let max_k = *KS.iter().max().unwrap();
    let hits_all: Vec<(f32, DocAddress)> = searcher
        .search(&query, &TopDocs::with_limit(max_k).order_by_score())
        .expect("search");

    // Prove the two paths are bit-identical before timing.
    let a = materialize_docstore(&searcher, fx.id_field, &hits_all);
    let b = materialize_numeric_ff(&searcher, &fx.table, &hits_all);
    assert_eq!(a, b, "docstore and numeric-ff doc_ids must match");

    let mut g = c.benchmark_group("id_materialize");
    for &k in KS {
        let hits = &hits_all[..k.min(hits_all.len())];
        g.bench_function(format!("docstore/k{k}"), |bn| {
            bn.iter(|| black_box(materialize_docstore(&searcher, fx.id_field, black_box(hits))));
        });
        g.bench_function(format!("numeric_ff/k{k}"), |bn| {
            bn.iter(|| black_box(materialize_numeric_ff(&searcher, &fx.table, black_box(hits))));
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
