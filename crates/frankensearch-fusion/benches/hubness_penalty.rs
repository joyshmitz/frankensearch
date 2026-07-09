//! Latency A/B: query-hubness dense-score correction vs the no-correction ORIGINAL.
//!
//! [`apply_hubness_penalty`] demotes high-hub docs before fusion — measured **all-positive
//! +0.0033 mean hybrid nDCG@10 (β=0.2, leakage-free) across 4 BEIR corpora**, with genuine
//! dense-tier gains on stance/citation corpora (docs/NEGATIVE_EVIDENCE.md). The query-time
//! kernel is a trivial O(pool) subtract; the LEGACY ORIGINAL is the pipeline with no correction
//! (identity, β=0 — a plain pool clone). We also bench the offline `compute_query_hubness`
//! builder (amortized: recomputed periodically from the query log, not per query).
//!
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-fusion --profile release --bench hubness_penalty
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_core::VectorHit;
use frankensearch_fusion::{HubnessConfig, apply_hubness_penalty, compute_query_hubness};

fn build_pool(pool: usize) -> (Vec<VectorHit>, Vec<f32>) {
    let hits: Vec<VectorHit> = (0..pool)
        .map(|i| VectorHit {
            index: i as u32,
            score: 1.0 / ((i as f32) + 1.0),
            doc_id: format!("doc{i:06}").into(),
        })
        .collect();
    // per-doc hubness in [0,1], heavy near the front (hubs)
    let hubness: Vec<f32> = (0..pool).map(|i| 0.9 - (i as f32 / pool as f32) * 0.6).collect();
    (hits, hubness)
}

/// Deterministic normalized vectors (no RNG in benches).
fn vecs(n: usize, dim: usize, seed: u32) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| {
            let mut v: Vec<f32> = (0..dim)
                .map(|j| (((i as u32 * 2_654_435_761 + j as u32 * 40_503 + seed) % 997) as f32) / 997.0 - 0.5)
                .collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            for x in &mut v {
                *x /= norm;
            }
            v
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("hubness_penalty");
    g.sample_size(50);
    g.warm_up_time(Duration::from_millis(400));
    g.measurement_time(Duration::from_millis(2000));

    let identity = HubnessConfig { beta: 0.0, kq: 10 };
    let correct = HubnessConfig { beta: 0.2, kq: 10 };

    for &pool in &[50usize, 100, 1000] {
        let (hits, hub) = build_pool(pool);
        let s = apply_hubness_penalty(&hits, &hub, &correct);
        assert_eq!(s.len(), pool);
        assert!((s[10].score - hits[10].score).abs() > 1e-9);

        g.bench_with_input(BenchmarkId::new("identity", pool), &(), |b, ()| {
            b.iter(|| black_box(apply_hubness_penalty(black_box(&hits), black_box(&hub), &identity)));
        });
        g.bench_with_input(BenchmarkId::new("correct", pool), &(), |b, ()| {
            b.iter(|| black_box(apply_hubness_penalty(black_box(&hits), black_box(&hub), &correct)));
        });
    }
    g.finish();

    // Offline builder cost (amortized): per-doc mean-cos to kq nearest of a query sample.
    let mut gb = c.benchmark_group("hubness_build");
    gb.sample_size(20);
    gb.warm_up_time(Duration::from_millis(300));
    gb.measurement_time(Duration::from_millis(2000));
    let dim = 384;
    for &(docs, queries) in &[(2000usize, 200usize), (5000, 500usize)] {
        let dv = vecs(docs, dim, 1);
        let qv = vecs(queries, dim, 7);
        let dref: Vec<&[f32]> = dv.iter().map(Vec::as_slice).collect();
        let qref: Vec<&[f32]> = qv.iter().map(Vec::as_slice).collect();
        gb.bench_with_input(BenchmarkId::new("build", format!("{docs}x{queries}")), &(), |b, ()| {
            b.iter(|| black_box(compute_query_hubness(black_box(&dref), black_box(&qref), 10)));
        });
    }
    gb.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
