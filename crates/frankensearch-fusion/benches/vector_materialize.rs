//! Final no-lexical vector materialization: old borrowed converter
//! (`&[VectorHit]` + `HashSet` dedup + `doc_id.clone()`) vs new owned converter
//! (`Vec<VectorHit>` + `doc_id` move). The new path is only used for
//! `blend_two_tier` output, which is already unique because it is collected from
//! a doc-id keyed map.

use std::collections::{HashMap, HashSet};
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};

#[derive(Clone)]
struct VHit {
    index: usize,
    score: f32,
    doc_id: String,
}

#[allow(dead_code)]
struct Scored {
    doc_id: String,
    score: f32,
    index: Option<usize>,
    fast_score: Option<f32>,
    quality_score: Option<f32>,
}

fn old_mat(
    hits: &[VHit],
    k: usize,
    fast_scores: &HashMap<&str, f32>,
    quality_scores: &HashMap<&str, f32>,
) -> Vec<Scored> {
    let mut seen = HashSet::with_capacity(hits.len());
    hits.iter()
        .filter(|hit| seen.insert(hit.doc_id.as_str()))
        .take(k)
        .map(|hit| Scored {
            doc_id: hit.doc_id.clone(),
            score: hit.score,
            index: Some(hit.index),
            fast_score: fast_scores
                .get(hit.doc_id.as_str())
                .copied()
                .or(Some(hit.score)),
            quality_score: quality_scores.get(hit.doc_id.as_str()).copied(),
        })
        .collect()
}

fn new_mat(
    hits: Vec<VHit>,
    k: usize,
    fast_scores: &HashMap<&str, f32>,
    quality_scores: &HashMap<&str, f32>,
) -> Vec<Scored> {
    hits.into_iter()
        .take(k)
        .map(|hit| {
            let fast_score = fast_scores
                .get(hit.doc_id.as_str())
                .copied()
                .or(Some(hit.score));
            let quality_score = quality_scores.get(hit.doc_id.as_str()).copied();
            Scored {
                doc_id: hit.doc_id,
                score: hit.score,
                index: Some(hit.index),
                fast_score,
                quality_score,
            }
        })
        .collect()
}

fn make_hits(n: usize, offset: usize) -> Vec<VHit> {
    (0..n)
        .map(|i| VHit {
            index: offset + i,
            score: 1.0 / (i as f32 + 1.0),
            doc_id: format!("doc-{:06}", offset + i),
        })
        .collect()
}

fn bench_vector_materialize(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_materialize");
    for n in [20usize, 60, 200] {
        let blended = make_hits(n, 0);
        let score_source = make_hits(n, 0);
        let fast_scores = score_source
            .iter()
            .map(|hit| (hit.doc_id.as_str(), hit.score))
            .collect::<HashMap<&str, f32>>();
        let quality_scores = score_source
            .iter()
            .map(|hit| (hit.doc_id.as_str(), hit.score * 0.9))
            .collect::<HashMap<&str, f32>>();
        debug_assert_eq!(
            old_mat(&blended, n, &fast_scores, &quality_scores).len(),
            new_mat(blended.clone(), n, &fast_scores, &quality_scores).len()
        );

        let id = format!("n{n}");
        group.bench_with_input(BenchmarkId::new("clone_dedup", &id), &(), |b, ()| {
            b.iter_batched(
                || blended.clone(),
                |owned| black_box(old_mat(black_box(&owned), n, &fast_scores, &quality_scores)),
                BatchSize::SmallInput,
            );
        });
        group.bench_with_input(BenchmarkId::new("move_unique", &id), &(), |b, ()| {
            b.iter_batched(
                || blended.clone(),
                |owned| black_box(new_mat(black_box(owned), n, &fast_scores, &quality_scores)),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_vector_materialize);
criterion_main!(benches);
