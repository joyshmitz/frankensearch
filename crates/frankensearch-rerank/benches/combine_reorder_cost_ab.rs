//! Latency of the shipped reranker combine step: `RerankCombine::PureReorder` (one sort)
//! vs `RerankCombine::RrfCombine` (rank-fuse the pre-rerank order with the rerank order).
//! The RRF-combine arms compare the original five-vector bookkeeping against the compact
//! order-vector implementation used by `pipeline.rs`.
//!
//! The reorder logic mirrors `pipeline.rs::apply_rrf_combine` / `compare_by_rerank_score`
//! (both private), replicated here over real `ScoredResult`s so the clone/permute cost is
//! faithful.
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
//!   rch exec -- cargo bench -p frankensearch-rerank --bench combine_reorder_cost_ab
//! ```

use std::cmp::Ordering;
use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_core::types::{ScoreSource, ScoredResult};

const K: f64 = 60.0;

fn candidate(i: usize, rerank_score: f32) -> ScoredResult {
    ScoredResult {
        doc_id: format!("doc-{i:06}").into(),
        score: 1.0 - (i as f32) * 0.001,
        source: ScoreSource::Reranked,
        index: None,
        fast_score: None,
        quality_score: None,
        lexical_score: None,
        rerank_score: Some(rerank_score),
        explanation: None,
        metadata: None,
    }
}

/// A reranked window in pre-rerank (fused) order — index i == pre-rerank rank — with an
/// interleaved rerank score so the two orders genuinely differ.
fn window(n: usize) -> Vec<ScoredResult> {
    (0..n)
        .map(|i| {
            // rerank score roughly reverses the pre-order, with jitter, to force a real reorder.
            let rs = ((n - i) as f32) + ((i * 7 % 13) as f32) * 0.1;
            candidate(i, rs)
        })
        .collect()
}

fn finite(c: &ScoredResult) -> f32 {
    c.rerank_score
        .filter(|s| s.is_finite())
        .unwrap_or(f32::NEG_INFINITY)
}

fn cmp_rerank(a: &ScoredResult, b: &ScoredResult) -> Ordering {
    finite(b)
        .total_cmp(&finite(a))
        .then_with(|| a.doc_id.cmp(&b.doc_id))
}

/// `PureReorder`: one sort by rerank score descending.
fn pure_reorder(win: &[ScoredResult]) -> Vec<ScoredResult> {
    let mut v = win.to_vec();
    v.sort_by(cmp_rerank);
    v
}

#[derive(Clone, Copy)]
struct RrfOrder {
    position: usize,
    fused_key: f64,
}

/// Original `RrfCombine`: argsort, inverted rank map, key vector, permutation, copy.
#[allow(clippy::cast_precision_loss)]
fn rrf_combine_current(win: &[ScoredResult]) -> Vec<ScoredResult> {
    let n = win.len();
    let mut by_rerank: Vec<usize> = (0..n).collect();
    by_rerank.sort_by(|&a, &b| cmp_rerank(&win[a], &win[b]));
    let mut rerank_rank = vec![0usize; n];
    for (rank, &pos) in by_rerank.iter().enumerate() {
        rerank_rank[pos] = rank;
    }
    let key: Vec<f64> = (0..n)
        .map(|i| 1.0 / (K + i as f64) + 1.0 / (K + rerank_rank[i] as f64))
        .collect();
    let mut perm: Vec<usize> = (0..n).collect();
    perm.sort_by(|&a, &b| {
        key[b]
            .total_cmp(&key[a])
            .then_with(|| win[a].doc_id.cmp(&win[b].doc_id))
    });
    perm.into_iter().map(|i| win[i].clone()).collect()
}

/// New `RrfCombine`: one order vector carries the rerank-rank and fused-key stages.
#[allow(clippy::cast_precision_loss)]
fn rrf_combine_order_vec(win: &[ScoredResult]) -> Vec<ScoredResult> {
    let n = win.len();
    let mut order: Vec<RrfOrder> = (0..n)
        .map(|position| RrfOrder {
            position,
            fused_key: 0.0,
        })
        .collect();
    order.sort_by(|a, b| cmp_rerank(&win[a.position], &win[b.position]));
    for (rerank_rank, entry) in order.iter_mut().enumerate() {
        entry.fused_key = 1.0 / (K + entry.position as f64) + 1.0 / (K + rerank_rank as f64);
    }
    order.sort_by(|a, b| {
        b.fused_key
            .total_cmp(&a.fused_key)
            .then_with(|| win[a.position].doc_id.cmp(&win[b.position].doc_id))
    });
    order
        .into_iter()
        .map(|entry| win[entry.position].clone())
        .collect()
}

fn comparable_order(win: &[ScoredResult]) -> Vec<(&str, Option<f32>)> {
    win.iter()
        .map(|item| (item.doc_id.as_ref(), item.rerank_score))
        .collect()
}

#[derive(Debug)]
struct PreparedMapping {
    included_count: usize,
    indices: Option<Vec<usize>>,
}

fn prepare_mapping_allocating(win: &[ScoredResult]) -> PreparedMapping {
    let mut indices = Vec::with_capacity(win.len());
    for (index, candidate) in win.iter().enumerate() {
        if candidate.rerank_score.is_some() {
            indices.push(index);
        }
    }
    PreparedMapping {
        included_count: indices.len(),
        indices: Some(indices),
    }
}

fn prepare_mapping_lazy(win: &[ScoredResult]) -> PreparedMapping {
    let mut included_count = 0;
    let mut indices: Option<Vec<usize>> = None;
    for (index, candidate) in win.iter().enumerate() {
        if candidate.rerank_score.is_some() {
            included_count += 1;
            if let Some(indices) = indices.as_mut() {
                indices.push(index);
            }
        } else if indices.is_none() {
            let mut gapped = Vec::with_capacity(win.len());
            gapped.extend(0..index);
            indices = Some(gapped);
        }
    }
    PreparedMapping {
        included_count,
        indices,
    }
}

fn mapping_checksum(mapping: &PreparedMapping) -> usize {
    (0..mapping.included_count).fold(0_usize, |checksum, rank| {
        let index = mapping
            .indices
            .as_ref()
            .map_or(rank, |indices| indices[rank]);
        checksum.wrapping_mul(31).wrapping_add(index)
    })
}

fn resolved_indices(mapping: &PreparedMapping) -> Vec<usize> {
    match &mapping.indices {
        Some(indices) => indices.clone(),
        None => (0..mapping.included_count).collect(),
    }
}

fn run_mapping(
    win: &[ScoredResult],
    prepare: fn(&[ScoredResult]) -> PreparedMapping,
) -> (PreparedMapping, usize) {
    let mapping = prepare(win);
    let checksum = mapping_checksum(&mapping);
    (mapping, checksum)
}

fn bench_mapping(c: &mut Criterion) {
    let all_text = window(32);
    let mut gapped = all_text.clone();
    for index in [3_usize, 9, 22] {
        gapped[index].rerank_score = None;
    }

    for fixture in [&all_text, &gapped] {
        let allocating = prepare_mapping_allocating(fixture);
        let lazy = prepare_mapping_lazy(fixture);
        assert_eq!(allocating.included_count, lazy.included_count);
        assert_eq!(resolved_indices(&allocating), resolved_indices(&lazy));
        assert_eq!(mapping_checksum(&allocating), mapping_checksum(&lazy));
    }

    let mut group = c.benchmark_group("rerank_prepare_mapping");
    group.warm_up_time(Duration::from_millis(50));
    group.measurement_time(Duration::from_millis(150));
    group.sample_size(10);
    group.bench_with_input(BenchmarkId::new("allocating_a", 32), &all_text, |b, win| {
        b.iter(|| {
                    black_box(run_mapping(black_box(win), prepare_mapping_allocating));
        });
    });
    group.bench_with_input(BenchmarkId::new("lazy", 32), &all_text, |b, win| {
        b.iter(|| {
            black_box(run_mapping(black_box(win), prepare_mapping_lazy));
        });
    });
    group.bench_with_input(BenchmarkId::new("allocating_b", 32), &all_text, |b, win| {
        b.iter(|| {
                    black_box(run_mapping(black_box(win), prepare_mapping_allocating));
        });
    });
    group.finish();
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("rerank_combine_reorder");
    for &n in &[20_usize, 50, 100, 200] {
        let win = window(n);
        let current = rrf_combine_current(&win);
        let order_vec = rrf_combine_order_vec(&win);
        assert_eq!(comparable_order(&current), comparable_order(&order_vec));

        g.bench_with_input(BenchmarkId::new("pure_reorder", n), &win, |b, w| {
            b.iter(|| black_box(pure_reorder(black_box(w))));
        });
        g.bench_with_input(BenchmarkId::new("rrf_combine_current", n), &win, |b, w| {
            b.iter(|| black_box(rrf_combine_current(black_box(w))));
        });
        g.bench_with_input(
            BenchmarkId::new("rrf_combine_order_vec", n),
            &win,
            |b, w| {
                b.iter(|| black_box(rrf_combine_order_vec(black_box(w))));
            },
        );
    }
    g.finish();
}

criterion_group!(benches, bench, bench_mapping);
criterion_main!(benches);
