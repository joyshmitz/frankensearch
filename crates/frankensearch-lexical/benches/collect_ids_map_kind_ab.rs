//! Sibling-consistency A/B for the `collect_id_hits` per-segment column cache.
//!
//! Production `TantivyIndex::collect_id_hits` caches the opened `ord` fast-field
//! column per segment in a **lazy `HashMap<u32, Option<Column<u64>>>`**, hashing
//! `segment_ord` once per hit (k SipHash-of-u32 per query). The numeric-ff bench
//! prototype (`id_materialize_numeric_ff`) instead indexed a `Vec` by
//! `segment_ord` (O(1), no hash) — but opened every segment eagerly. This bench
//! measures the best-of-both candidate: a **lazily-populated `Vec<Option<Option<
//! Column>>>`** (O(1) index, no hash, still only opens touched segments) vs the
//! shipped lazy HashMap. Both are byte-identical (same columns, same ords, same
//! id clones); the single-segment fixture isolates the per-hit map access so the
//! ratio is the pure SipHash-vs-index cost against a String-clone-dominated loop.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- \
//!   cargo bench -p frankensearch-lexical --profile release \
//!   --features bench-internals --bench collect_ids_map_kind_ab
//! ```

use std::collections::HashMap;
use std::hint::black_box;

use frankensearch_core::bench_support::{PairedRatio, paired_median_ratio};
use tantivy::collector::TopDocs;
use tantivy::columnar::Column;
use tantivy::query::QueryParser;
use tantivy::schema::{FAST, STORED, STRING, Schema, TEXT};
use tantivy::{DocAddress, Index, doc};

const N: usize = 20_000;
const KS: &[usize] = &[30, 100, 300, 1000];

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

const VOCAB: &[&str] = &[
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "search",
    "engine",
    "vector",
    "lexical",
    "ranking",
    "relevance",
    "document",
    "query",
    "hybrid",
    "fusion",
];

struct Fixture {
    index: Index,
    table: Vec<String>,
}

fn build_fixture() -> Fixture {
    let mut sb = Schema::builder();
    let _id_field = sb.add_text_field("id", STRING | STORED);
    let content = sb.add_text_field("content", TEXT | STORED);
    let ord_field = sb.add_u64_field("ord", FAST | STORED);
    let schema = sb.build();
    let id_field = schema.get_field("id").unwrap();

    let index = Index::create_in_ram(schema);
    let mut writer: tantivy::IndexWriter = index.writer(60_000_000).expect("writer");
    let mut table = Vec::with_capacity(N);
    for i in 0..N {
        let mut state = (i as u64).wrapping_add(1);
        let doc_id = format!("doc-{i:06}");
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

    Fixture { index, table }
}

/// Shipped shape: lazy `HashMap<u32, Option<Column>>` — one `segment_ord` hash per hit.
fn materialize_hashmap_lazy(
    searcher: &tantivy::Searcher,
    table: &[String],
    hits: &[(f32, DocAddress)],
) -> Vec<String> {
    let mut cols: HashMap<u32, Option<Column<u64>>> = HashMap::new();
    let mut out = Vec::with_capacity(hits.len());
    for &(_, addr) in hits {
        let ord = cols
            .entry(addr.segment_ord)
            .or_insert_with(|| {
                searcher
                    .segment_reader(addr.segment_ord)
                    .fast_fields()
                    .u64("ord")
                    .ok()
            })
            .as_ref()
            .and_then(|c| c.first(addr.doc_id))
            .expect("ord fast field") as usize;
        out.push(table[ord].clone());
    }
    out
}

/// Candidate: lazy `Vec<Option<Option<Column>>>` indexed by `segment_ord` — no hash,
/// still opens only touched segments (outer `None` = not-yet-opened).
fn materialize_vec_lazy(
    searcher: &tantivy::Searcher,
    table: &[String],
    hits: &[(f32, DocAddress)],
) -> Vec<String> {
    let seg_count = searcher.segment_readers().len();
    let mut cols: Vec<Option<Option<Column<u64>>>> = Vec::new();
    cols.resize_with(seg_count, || None);
    let mut out = Vec::with_capacity(hits.len());
    for &(_, addr) in hits {
        let seg = addr.segment_ord as usize;
        let slot = &mut cols[seg];
        if slot.is_none() {
            *slot = Some(
                searcher
                    .segment_reader(addr.segment_ord)
                    .fast_fields()
                    .u64("ord")
                    .ok(),
            );
        }
        let ord = slot
            .as_ref()
            .expect("slot populated")
            .as_ref()
            .and_then(|c| c.first(addr.doc_id))
            .expect("ord fast field") as usize;
        out.push(table[ord].clone());
    }
    out
}

fn verdict(lever: &PairedRatio, null: &PairedRatio) -> &'static str {
    if lever.decidable_against(null) {
        if lever.median < 1.0 {
            "DECIDABLE WIN"
        } else {
            "DECIDABLE REGRESSION"
        }
    } else {
        "INSIDE NULL FLOOR (not decidable)"
    }
}

fn main() {
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
    eprintln!(
        "segments={} hits={}",
        searcher.segment_readers().len(),
        hits_all.len()
    );

    let inner = std::env::var("COLLECT_IDS_AB_INNER")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(512);

    for &k in KS {
        let hits = &hits_all[..k.min(hits_all.len())];
        let base = materialize_hashmap_lazy(&searcher, &fx.table, hits);
        let cand = materialize_vec_lazy(&searcher, &fx.table, hits);
        assert_eq!(base, cand, "hashmap and vec doc_ids differ for k={k}");

        let run_orig = || {
            black_box(materialize_hashmap_lazy(
                &searcher,
                &fx.table,
                black_box(hits),
            ));
        };
        let run_cand = || {
            black_box(materialize_vec_lazy(&searcher, &fx.table, black_box(hits)));
        };
        let null = paired_median_ratio(41, inner, run_orig, run_orig);
        let lever = paired_median_ratio(41, inner, run_orig, run_cand);
        eprintln!(
            "[null]  collect_ids/k{k}: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
            null.median, null.p5, null.p95, null.rounds
        );
        eprintln!(
            "[lever] collect_ids/k{k}: vec/HASHMAP median {:.4} p5 {:.4} p95 {:.4} -> {}",
            lever.median,
            lever.p5,
            lever.p95,
            verdict(&lever, &null)
        );
    }
}
