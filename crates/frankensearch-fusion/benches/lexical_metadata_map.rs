//! A/B for the lexical-metadata re-attachment in the async hybrid's
//! `fused_hits_to_scored_results` (searcher.rs). The map is keyed by doc_id and
//! only the `fused` (top-k) hits read it, but the old code cloned EVERY lexical
//! candidate's `serde_json::Value` metadata into the map (incl. candidates
//! dropped from the top-k). The new code borrows `&Value` into the map and
//! clones only the per-winner lookup.
//!
//! - `clone_all` : `AHashMap<&str, Value>` (clone every candidate's metadata).
//! - `borrow`    : `AHashMap<&str, &Value>` (clone only the k winners).
//!
//! Both produce identical `Option<Value>` per winner. The win = `(N - k)`
//! avoided `Value` clones, scaling with metadata size. Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
//!   rch exec -- cargo bench -p frankensearch-fusion --profile release \
//!   --bench lexical_metadata_map
//! ```

use std::hint::black_box;

use ahash::AHashMap;
use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_core::types::{ScoreSource, ScoredResult};
use serde_json::{Value, json};

const N: usize = 60; // lexical candidates
const K: usize = 10; // fused top-k winners

fn small_metadata(i: usize) -> Value {
    json!({ "cluster": (i % 6).to_string() })
}

fn large_metadata(i: usize) -> Value {
    json!({
        "cluster": (i % 6).to_string(),
        "title": format!("Document number {i:06} about hybrid search and ranking"),
        "tags": ["rust", "search", "vector", "lexical", "fusion"],
        "source": { "kind": "corpus", "shard": i % 8, "path": format!("/data/docs/{i:06}.txt") },
        "score_hint": i as f64 * 1.5,
        "lang": "en",
        "extra": "deterministic benchmark metadata payload kept identical across arms",
    })
}

fn make_candidates(meta: impl Fn(usize) -> Value) -> Vec<ScoredResult> {
    (0..N)
        .map(|i| ScoredResult {
            doc_id: format!("doc-{i:06}").into(),
            score: 1.0,
            source: ScoreSource::Lexical,
            index: None,
            fast_score: None,
            quality_score: None,
            lexical_score: Some(1.0),
            rerank_score: None,
            explanation: None,
            metadata: Some(meta(i)),
        })
        .collect()
}

fn winners() -> Vec<String> {
    (0..K).map(|i| format!("doc-{i:06}")).collect()
}

/// Old path: clone every candidate's metadata into the map, lookup+clone winners.
fn clone_all(lexical: &[ScoredResult], fused: &[String]) -> Vec<Option<Value>> {
    let map: AHashMap<&str, Value> = lexical
        .iter()
        .filter_map(|r| r.metadata.as_ref().map(|m| (r.doc_id.as_str(), m.clone())))
        .collect();
    fused
        .iter()
        .map(|id| map.get(id.as_str()).cloned())
        .collect()
}

/// New path: borrow metadata into the map, clone only the winners.
fn borrow(lexical: &[ScoredResult], fused: &[String]) -> Vec<Option<Value>> {
    let map: AHashMap<&str, &Value> = lexical
        .iter()
        .filter_map(|r| r.metadata.as_ref().map(|m| (r.doc_id.as_str(), m)))
        .collect();
    fused
        .iter()
        .map(|id| map.get(id.as_str()).copied().cloned())
        .collect()
}

fn bench(c: &mut Criterion) {
    let fused = winners();
    let small = make_candidates(small_metadata);
    let large = make_candidates(large_metadata);

    // Identical output guard.
    assert_eq!(clone_all(&small, &fused), borrow(&small, &fused));
    assert_eq!(clone_all(&large, &fused), borrow(&large, &fused));

    let mut g = c.benchmark_group("lexical_metadata_map");
    for (label, lexical) in [("small", &small), ("large", &large)] {
        g.bench_function(format!("clone_all/{label}"), |b| {
            b.iter(|| black_box(clone_all(black_box(lexical), black_box(&fused))));
        });
        g.bench_function(format!("borrow/{label}"), |b| {
            b.iter(|| black_box(borrow(black_box(lexical), black_box(&fused))));
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
