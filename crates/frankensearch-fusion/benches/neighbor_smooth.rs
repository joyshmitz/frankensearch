//! Latency A/B: graph score-diffusion (kNN neighbor smoothing) vs the no-smooth ORIGINAL.
//!
//! [`neighbor_smooth`] is a label-propagation pass that rescues below-threshold relevants
//! sitting in confident clusters — measured **+0.0052…+0.0114 hybrid nDCG@10 on recall-bound
//! BEIR corpora** (docs/NEGATIVE_EVIDENCE.md; pool-restricted deployable form). This bench
//! confirms it is nearly free atop an existing k-NN neighbor graph: the LEGACY ORIGINAL is
//! the pipeline with no smoothing (identity, α=0 — a plain pool clone); the candidate is the
//! full diffusion pass (α=0.3, M=10) and its reciprocal-kNN refinement. Cost is O(pool · M).
//!
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-fusion --profile release --bench neighbor_smooth
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_core::{DocumentGraph, EdgeType, VectorHit};
use frankensearch_fusion::{SmoothConfig, neighbor_smooth};

/// A pool of `pool` candidates with heavy-tailed cosine scores, and an M-nearest `Similar`
/// graph where each doc links to the `m` following docs (a synthetic cluster chain, all
/// in-pool so the kernel does real averaging work).
fn build(pool: usize, m: usize) -> (Vec<VectorHit>, DocumentGraph) {
    let hits: Vec<VectorHit> = (0..pool)
        .map(|i| VectorHit {
            index: i as u32,
            score: 1.0 / ((i as f32) + 1.0), // heavy-tailed like real cosine pools
            doc_id: format!("doc{i:06}").into(),
        })
        .collect();
    let mut graph = DocumentGraph::new();
    for i in 0..pool {
        for j in 1..=m {
            let nbr = (i + j) % pool;
            graph.add_edge(
                format!("doc{i:06}"),
                format!("doc{nbr:06}"),
                EdgeType::Similar,
                1.0 - (j as f32) * 0.01,
            );
            // add the reverse edge so mutual-kNN has real reciprocal work to do
            graph.add_edge(
                format!("doc{nbr:06}"),
                format!("doc{i:06}"),
                EdgeType::Similar,
                1.0 - (j as f32) * 0.01,
            );
        }
    }
    (hits, graph)
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("neighbor_smooth");
    g.sample_size(50);
    g.warm_up_time(Duration::from_millis(400));
    g.measurement_time(Duration::from_millis(2000));

    let identity = SmoothConfig { alpha: 0.0, m: 10, mutual: false };
    let smooth = SmoothConfig { alpha: 0.3, m: 10, mutual: false };
    let mutual = SmoothConfig { alpha: 0.3, m: 10, mutual: true };

    for &pool in &[50usize, 100, 1000] {
        let (hits, graph) = build(pool, 10);
        // Sanity: smoothing changes the scores (real work happens).
        let s = neighbor_smooth(&hits, &graph, &smooth);
        assert_eq!(s.len(), pool);
        assert!((s[10].score - hits[10].score).abs() > 1e-9);

        g.bench_with_input(BenchmarkId::new("identity", pool), &(), |b, ()| {
            b.iter(|| black_box(neighbor_smooth(black_box(&hits), black_box(&graph), &identity)));
        });
        g.bench_with_input(BenchmarkId::new("smooth", pool), &(), |b, ()| {
            b.iter(|| black_box(neighbor_smooth(black_box(&hits), black_box(&graph), &smooth)));
        });
        g.bench_with_input(BenchmarkId::new("mutual", pool), &(), |b, ()| {
            b.iter(|| black_box(neighbor_smooth(black_box(&hits), black_box(&graph), &mutual)));
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
