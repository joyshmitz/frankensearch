//! `limit_all` end-to-end A/B for the discarded-phase clone elision.
//!
//! `SyncTwoTierSearcher::search_collect` returns only `(final_results, metrics)`
//! and throws away `outcome.phases` — yet `search_internal` used to build those
//! phases by cloning the full `Vec<ScoredResult>` (N owned doc_ids each) once per
//! phase (Initial + Refined). At `limit_all` (k = N) that is up to 2·N wasted
//! short-`String` allocations per query. `search_collect` now passes
//! `want_phases = false` and skips them; only `search_iter` (the streaming API)
//! still builds phases.
//!
//! This measures the exact end-to-end delta using only public APIs on the same
//! searcher over the same queries:
//!   - `collect` (new): `search_collect` — no phase clones.
//!   - `iter`    (old): `search_iter` — builds both phases (the pre-change
//!                       `search_collect` behavior); drained so nothing is elided.
//! Both run identical search work (fast scan → quality rescore → blend), so the
//! ratio is purely the phase-clone cost. `final_results` are unchanged.
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-fusion --bench collect_limit_all
//! ```

use std::hint::black_box;
use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_core::{SearchPhase, TwoTierConfig};
use frankensearch_fusion::SyncTwoTierSearcher;
use frankensearch_index::{InMemoryTwoTierIndex, InMemoryVectorIndex};

const N: usize = 10_000;
const DIM: usize = 384;
const QUERIES: usize = 16;
const CLUSTERS: usize = 64;
const NOISE: f32 = 0.30;

fn raw_vector(seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut v = Vec::with_capacity(DIM);
    for _ in 0..DIM {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        v.push((state >> 40) as f32 / (1u64 << 23) as f32 - 1.0);
    }
    v
}

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

fn make_vector(centroids: &[Vec<f32>], c: usize, noise_seed: u64) -> Vec<f32> {
    let centroid = &centroids[c % centroids.len()];
    let noise = raw_vector(noise_seed);
    normalize(
        centroid
            .iter()
            .zip(&noise)
            .map(|(a, n)| a + NOISE * n)
            .collect(),
    )
}

fn bench_collect_limit_all(c: &mut Criterion) {
    let centroids: Vec<Vec<f32>> = (0..CLUSTERS)
        .map(|i| normalize(raw_vector(0xc000_0000 + i as u64)))
        .collect();
    let ids: Vec<String> = (0..N).map(|i| format!("doc-{i:06}")).collect();
    let fast_vecs: Vec<Vec<f32>> = (0..N)
        .map(|i| make_vector(&centroids, i % CLUSTERS, i as u64 + 1))
        .collect();
    let quality_vecs: Vec<Vec<f32>> = (0..N)
        .map(|i| make_vector(&centroids, i % CLUSTERS, 0xbeef_0000 + i as u64))
        .collect();
    let fast = InMemoryVectorIndex::from_vectors(ids.clone(), fast_vecs, DIM).expect("fast");
    let quality = InMemoryVectorIndex::from_vectors(ids, quality_vecs, DIM).expect("quality");
    let index = Arc::new(InMemoryTwoTierIndex::new(fast, Some(quality)));
    let searcher = SyncTwoTierSearcher::new(index, TwoTierConfig::default());

    let queries: Vec<Vec<f32>> = (0..QUERIES)
        .map(|q| make_vector(&centroids, q % CLUSTERS, 0xdead_0000 + q as u64))
        .collect();

    // Sanity: both APIs agree on the final ranking (identical doc_id order).
    let collect_ids: Vec<String> = searcher
        .search_collect(&queries[0], N)
        .expect("collect")
        .0
        .iter()
        .map(|r| r.doc_id.to_string())
        .collect();
    let iter_final = searcher
        .search_iter(&queries[0], N)
        .last()
        .expect("iter has a final phase");
    let iter_results = match &iter_final {
        SearchPhase::Initial { results, .. }
        | SearchPhase::Refined { results, .. }
        | SearchPhase::Reranked { results, .. } => results,
        SearchPhase::RefinementFailed {
            initial_results, ..
        } => initial_results,
    };
    let iter_ids: Vec<String> = iter_results.iter().map(|r| r.doc_id.to_string()).collect();
    assert_eq!(collect_ids, iter_ids, "collect vs iter final ranking differs");

    let mut qi = 0usize;
    let mut g = c.benchmark_group("collect_limit_all");
    // New: search_collect skips phase construction (no discarded clones).
    g.bench_function("collect", |b| {
        b.iter(|| {
            let q = &queries[qi % QUERIES];
            qi += 1;
            black_box(searcher.search_collect(black_box(q), N).expect("collect"))
        });
    });
    // Old-equivalent: search_iter builds both phases (the pre-change collect cost).
    g.bench_function("iter", |b| {
        b.iter(|| {
            let q = &queries[qi % QUERIES];
            qi += 1;
            black_box(searcher.search_iter(black_box(q), N).count())
        });
    });
    g.finish();
}

criterion_group!(benches, bench_collect_limit_all);
criterion_main!(benches);
