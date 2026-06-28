//! A/B for the MMR rerank reorder in the async hybrid (`searcher.rs`). When MMR
//! diversity reranking is enabled, the top `pool` fused results are reordered by
//! the `mmr_rerank` permutation. The old code cloned the pool into `head`, then
//! cloned each element again while emitting it in MMR order (2×pool full
//! `ScoredResult` clones, each carrying a metadata `Value`). The new code splits
//! off the tail and MOVES each head element into its MMR slot (zero clones).
//!
//! - `clone_reorder` : clone head + clone-per-order-index (old).
//! - `move_reorder`  : split_off tail + `Option::take` move per index (new).
//!
//! Both produce identical output (asserted). `order` is a distinct permutation,
//! as `mmr_rerank` guarantees. Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
//!   rch exec -- cargo bench -p frankensearch-fusion --profile release \
//!   --bench mmr_reorder
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_core::types::{ScoreSource, ScoredResult};
use serde_json::{Value, json};

const POOL: usize = 30;
const TAIL: usize = 10;

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

fn make_results(meta: impl Fn(usize) -> Value) -> Vec<ScoredResult> {
    (0..POOL + TAIL)
        .map(|i| ScoredResult {
            doc_id: format!("doc-{i:06}"),
            score: (POOL + TAIL - i) as f32,
            source: ScoreSource::Hybrid,
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

// 7 is coprime with 30, so this is a valid permutation of 0..POOL.
fn order() -> Vec<usize> {
    (0..POOL).map(|i| (i * 7) % POOL).collect()
}

fn clone_reorder(results: Vec<ScoredResult>, order: &[usize]) -> Vec<ScoredResult> {
    let head = results.iter().take(POOL).cloned().collect::<Vec<_>>();
    let mut reranked = Vec::with_capacity(results.len());
    for &idx in order {
        if let Some(item) = head.get(idx) {
            reranked.push(item.clone());
        }
    }
    reranked.extend(results.into_iter().skip(POOL));
    reranked
}

fn move_reorder(mut results: Vec<ScoredResult>, order: &[usize]) -> Vec<ScoredResult> {
    let split = POOL.min(results.len());
    let tail = results.split_off(split);
    let mut slots: Vec<Option<ScoredResult>> = results.into_iter().map(Some).collect();
    let mut reranked = Vec::with_capacity(slots.len() + tail.len());
    for &idx in order {
        if let Some(item) = slots.get_mut(idx).and_then(Option::take) {
            reranked.push(item);
        }
    }
    reranked.extend(tail);
    reranked
}

fn bench(c: &mut Criterion) {
    let ord = order();

    let small = make_results(small_metadata);
    let large = make_results(large_metadata);
    // ScoredResult isn't PartialEq; the reorder only permutes (payloads
    // unchanged), so identical doc_id ordering ⇒ identical output.
    let ids = |v: &[ScoredResult]| v.iter().map(|r| r.doc_id.clone()).collect::<Vec<_>>();
    assert_eq!(
        ids(&clone_reorder(small.clone(), &ord)),
        ids(&move_reorder(small.clone(), &ord))
    );
    assert_eq!(
        ids(&clone_reorder(large.clone(), &ord)),
        ids(&move_reorder(large.clone(), &ord))
    );

    let mut g = c.benchmark_group("mmr_reorder");
    for (label, template) in [("small", &small), ("large", &large)] {
        g.bench_function(format!("clone_reorder/{label}"), |b| {
            b.iter_batched(
                || template.clone(),
                |r| black_box(clone_reorder(r, black_box(&ord))),
                criterion::BatchSize::SmallInput,
            );
        });
        g.bench_function(format!("move_reorder/{label}"), |b| {
            b.iter_batched(
                || template.clone(),
                |r| black_box(move_reorder(r, black_box(&ord))),
                criterion::BatchSize::SmallInput,
            );
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
