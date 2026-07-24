//! Full `rrf_fuse` head-to-head: the shipped map version (`rrf_fuse_with_graph`,
//! N-entry value hashmap → random-order collect → from-scratch sort) vs the
//! merge-structured version (`rrf_fuse_with_graph_merge`, small contribution maps
//! plus semantic-ordered emission → near-sorted adaptive sort). Both are verified
//! byte-identical by the `merge_matches_map_fusion` unit test.
//!
//! Models the `limit_all` shape (the gap row): all-N semantic candidates in
//! vector-score order, a lexical subset (~20% overlap), no graph, `limit = N`.
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-fusion --bench rrf_merge_fuse
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_core::types::{ScoreSource, ScoredResult, VectorHit};
use frankensearch_fusion::rrf::{
    RrfConfig, rrf_fuse_with_graph, rrf_fuse_with_graph_merge, rrf_fuse_with_graph_merge_unique,
};

fn lexical(doc_id: String, score: f32) -> ScoredResult {
    ScoredResult {
        doc_id: doc_id.into(),
        score,
        source: ScoreSource::Lexical,
        index: None,
        fast_score: None,
        quality_score: None,
        lexical_score: Some(score),
        rerank_score: None,
        explanation: None,
        metadata: None,
    }
}

/// Build limit_all-shaped inputs: N semantic hits already in descending score
/// order (vector tier output), and a lexical subset (~20%) overlapping them.
fn build(n: usize) -> (Vec<ScoredResult>, Vec<VectorHit>) {
    let semantic: Vec<VectorHit> = (0..n)
        .map(|i| VectorHit {
            index: u32::try_from(i).expect("benchmark index fits u32"),
            #[allow(clippy::cast_precision_loss)]
            score: 1.0 - (i as f32) / (n as f32), // strictly descending
            doc_id: format!("doc-{i:06}").into(),
        })
        .collect();
    // Lexical: every 5th doc (20%), with its own ranking order.
    let lex: Vec<ScoredResult> = (0..n)
        .step_by(5)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            lexical(format!("doc-{i:06}"), (i % 97) as f32 * 0.01)
        })
        .collect();
    (lex, semantic)
}

fn bench(c: &mut Criterion) {
    let cfg = RrfConfig::default();
    let mut group = c.benchmark_group("rrf_merge_fuse");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(30);

    for &n in &[10_000_usize, 50_000, 100_000] {
        let (lex, sem) = build(n);

        // Sanity: identical output (length + first/last doc_id).
        let a = rrf_fuse_with_graph(&lex, &sem, &[], 0.0, n, 0, &cfg);
        let b = rrf_fuse_with_graph_merge(&lex, &sem, &[], 0.0, n, 0, &cfg);
        assert_eq!(a.len(), b.len());
        assert_eq!(a[0].doc_id, b[0].doc_id);
        assert_eq!(a[a.len() - 1].doc_id, b[b.len() - 1].doc_id);

        group.bench_with_input(BenchmarkId::new("map", n), &n, |bch, _| {
            bch.iter(|| {
                black_box(rrf_fuse_with_graph(
                    black_box(&lex),
                    black_box(&sem),
                    &[],
                    0.0,
                    n,
                    0,
                    &cfg,
                ));
            });
        });
        group.bench_with_input(BenchmarkId::new("merge", n), &n, |bch, _| {
            bch.iter(|| {
                black_box(rrf_fuse_with_graph_merge(
                    black_box(&lex),
                    black_box(&sem),
                    &[],
                    0.0,
                    n,
                    0,
                    &cfg,
                ));
            });
        });
        // Unique-semantic fast path (skips the seen_semantic dedup set).
        let u = rrf_fuse_with_graph_merge_unique(&lex, &sem, &[], 0.0, n, 0, &cfg);
        assert_eq!(u.len(), b.len());
        assert_eq!(u[0].doc_id, b[0].doc_id);
        group.bench_with_input(BenchmarkId::new("merge_unique", n), &n, |bch, _| {
            bch.iter(|| {
                black_box(rrf_fuse_with_graph_merge_unique(
                    black_box(&lex),
                    black_box(&sem),
                    &[],
                    0.0,
                    n,
                    0,
                    &cfg,
                ));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
