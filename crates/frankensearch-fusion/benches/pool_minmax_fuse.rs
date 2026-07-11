//! Latency A/B: pool-local min-max SCORE fusion vs RRF (LEGACY ORIGINAL).
//!
//! `pool_minmax_fuse` is a drop-in for `rrf_fuse` that recovers the score MAGNITUDE
//! the rank transform discards — measured **+0.0038 mean nDCG@10 over RRF across 4
//! BEIR corpora, never-negative** (docs/NEGATIVE_EVIDENCE.md, `45530fb`). This bench
//! confirms it is **latency-neutral** vs RRF: both are O(pool) hashmap accumulation +
//! a windowed sort; pool-min-max adds two O(pool) min/max passes and a per-doc
//! normalize (a couple of extra flops per candidate), so it should land within noise
//! of RRF. Calls the REAL public fuse functions (not reimplementations).
//!
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-fusion --profile release --bench pool_minmax_fuse
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_core::{ScoreSource, ScoredResult, VectorHit};
use frankensearch_fusion::{RrfConfig, pool_minmax_fuse, rrf_fuse};

fn lexical(doc: &str, score: f32) -> ScoredResult {
    ScoredResult {
        doc_id: doc.into(),
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

fn semantic(doc: &str, score: f32, index: u32) -> VectorHit {
    VectorHit {
        index,
        score,
        doc_id: doc.into(),
    }
}

/// Two pools of `pool` candidates with ~50% doc overlap (shared docs in the middle),
/// heavy-tailed scores like real BM25 / cosine pools.
fn build(pool: usize) -> (Vec<ScoredResult>, Vec<VectorHit>) {
    let lex: Vec<ScoredResult> = (0..pool)
        .map(|i| lexical(&format!("doc{i:06}"), (pool - i) as f32 * 0.5 + 1.0))
        .collect();
    let half = pool / 2;
    let sem: Vec<VectorHit> = (half..pool + half)
        .map(|i| {
            semantic(
                &format!("doc{i:06}"),
                1.0 / ((i - half) as f32 + 1.0),
                i as u32,
            )
        })
        .collect();
    (lex, sem)
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("pool_minmax_fuse");
    g.sample_size(50);
    g.warm_up_time(Duration::from_millis(400));
    g.measurement_time(Duration::from_millis(2000));
    let config = RrfConfig::default();

    for &pool in &[50usize, 100, 1000] {
        let (lex, sem) = build(pool);
        // Sanity: both return a non-empty top-10.
        assert_eq!(rrf_fuse(&lex, &sem, 10, 0, &config).len(), 10);
        assert_eq!(pool_minmax_fuse(&lex, &sem, 10, 0, &config).len(), 10);

        g.bench_with_input(BenchmarkId::new("rrf", pool), &(), |b, ()| {
            b.iter(|| black_box(rrf_fuse(black_box(&lex), black_box(&sem), 10, 0, &config)));
        });
        g.bench_with_input(BenchmarkId::new("pool_minmax", pool), &(), |b, ()| {
            b.iter(|| {
                black_box(pool_minmax_fuse(
                    black_box(&lex),
                    black_box(&sem),
                    10,
                    0,
                    &config,
                ))
            });
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
