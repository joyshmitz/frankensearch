//! Latency A/B: merge-structured pool-min-max fusion vs the value-map ORIGINAL.
//!
//! `pool_minmax_fuse` (LEGACY ORIGINAL) accumulates every doc into one `N`-entry
//! `HashMap<&str, FusedHitScratch>`, then `into_values()` in **random** hash order and sorts
//! from scratch — an `O(N log N)` sort on random input for the `limit_all` shape (window ≥ N,
//! e.g. returning a full ranked feed to a reranker). `pool_minmax_fuse_merge` keeps only a small
//! `&str → (rank, score)` lexical contribution map and walks the already-score-sorted `semantic`
//! slice once in order, so `results` is **near-sorted** and the final sort runs near-`O(N)`
//! (pdqsort is adaptive). This is the same primitive that made `rrf_fuse_with_graph_merge` beat
//! the RRF value-map (`4aeb66b`, 1.31–1.46× on limit_all); it had never been ported to the
//! pool-min-max quality operator (`a9e53b4`, +0.0038 nDCG@10 over RRF).
//!
//! Two shapes: `limit_all` (window ≥ N — where the merge pays) and `top10` (select_nth dominates
//! — a regression guard: the merge must not lose the small-window path). Bit-identity is gated
//! before timing.
//!
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-copper \
//!   rch exec -- cargo bench -p frankensearch-fusion --profile release --bench pool_minmax_merge_ab
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_core::{FusedHit, ScoreSource, ScoredResult, VectorHit};
use frankensearch_fusion::{RrfConfig, pool_minmax_fuse, pool_minmax_fuse_merge};

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
    VectorHit { index, score, doc_id: doc.into() }
}

/// Two pools of `pool` candidates with ~50% doc overlap (shared docs in the middle), heavy-tailed
/// scores like real BM25 / cosine pools. Semantic is score-sorted descending (as a vector index
/// returns), so the merge path's near-sorted assumption is exercised.
fn build(pool: usize) -> (Vec<ScoredResult>, Vec<VectorHit>) {
    let lex: Vec<ScoredResult> = (0..pool)
        .map(|i| lexical(&format!("doc{i:06}"), (pool - i) as f32 * 0.5 + 1.0))
        .collect();
    let half = pool / 2;
    let sem: Vec<VectorHit> = (half..pool + half)
        .map(|i| semantic(&format!("doc{i:06}"), 1.0 / ((i - half) as f32 + 1.0), i as u32))
        .collect();
    (lex, sem)
}

fn assert_bit_identical(
    lex: &[ScoredResult],
    sem: &[VectorHit],
    limit: usize,
    cfg: &RrfConfig,
    tag: &str,
) {
    let a: Vec<FusedHit> = pool_minmax_fuse(lex, sem, limit, 0, cfg);
    let b: Vec<FusedHit> = pool_minmax_fuse_merge(lex, sem, limit, 0, cfg);
    assert_eq!(a.len(), b.len(), "{tag}: length differs");
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert_eq!(x.doc_id, y.doc_id, "{tag} row {i}: doc_id");
        assert_eq!(
            x.rrf_score.to_bits(),
            y.rrf_score.to_bits(),
            "{tag} row {i}: fused score ({} vs {})",
            x.rrf_score,
            y.rrf_score
        );
        assert_eq!(x.semantic_index, y.semantic_index, "{tag} row {i}: semantic_index");
        assert_eq!(x.in_both_sources, y.in_both_sources, "{tag} row {i}: in_both");
    }
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("pool_minmax_merge");
    g.sample_size(50);
    g.warm_up_time(Duration::from_millis(400));
    g.measurement_time(Duration::from_millis(2000));
    let cfg = RrfConfig::default();

    for &pool in &[50usize, 100, 1000] {
        let (lex, sem) = build(pool);
        let all = 3 * pool; // window ≥ unique N (~1.5·pool) → full sort (limit_all shape)

        assert_bit_identical(&lex, &sem, all, &cfg, &format!("limit_all/{pool}"));
        assert_bit_identical(&lex, &sem, 10, &cfg, &format!("top10/{pool}"));

        // limit_all — the shape the merge structure targets.
        g.bench_with_input(BenchmarkId::new("limit_all_ORIG", pool), &(), |b, ()| {
            b.iter(|| black_box(pool_minmax_fuse(black_box(&lex), black_box(&sem), all, 0, &cfg)));
        });
        g.bench_with_input(BenchmarkId::new("limit_all_merge", pool), &(), |b, ()| {
            b.iter(|| {
                black_box(pool_minmax_fuse_merge(black_box(&lex), black_box(&sem), all, 0, &cfg))
            });
        });
        // top10 — regression guard (select_nth dominates; merge must not lose here).
        g.bench_with_input(BenchmarkId::new("top10_ORIG", pool), &(), |b, ()| {
            b.iter(|| black_box(pool_minmax_fuse(black_box(&lex), black_box(&sem), 10, 0, &cfg)));
        });
        g.bench_with_input(BenchmarkId::new("top10_merge", pool), &(), |b, ()| {
            b.iter(|| {
                black_box(pool_minmax_fuse_merge(black_box(&lex), black_box(&sem), 10, 0, &cfg))
            });
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
