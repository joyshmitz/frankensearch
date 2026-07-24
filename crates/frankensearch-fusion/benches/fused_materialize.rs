//! `fused_hits_to_scored_results` materialization: old (`&[FusedHit]` + clone
//! `doc_id`) vs new (`Vec<FusedHit>` + move `doc_id`). The `rrf_fuse` result is a
//! fresh temporary, so moving `doc_ids` is bit-identical and drops a per-result
//! `String` clone. `iter_batched` clones the input per iteration so both arms pay
//! the same setup; the measured delta is the `doc_id` clone vs move.
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

#[derive(Clone)]
struct Fused {
    doc_id: String,
    rrf_score: f64,
    semantic_index: Option<u32>,
    semantic_score: Option<f32>,
    lexical_score: Option<f32>,
}
struct Scored {
    doc_id: String,
    score: f32,
    index: Option<u32>,
    fast_score: Option<f32>,
    lexical_score: Option<f32>,
}

#[allow(clippy::cast_possible_truncation)]
fn old_mat(hits: &[Fused], k: usize) -> Vec<Scored> {
    hits.iter()
        .take(k)
        .map(|h| Scored {
            doc_id: h.doc_id.clone(),
            score: h.rrf_score as f32,
            index: h.semantic_index,
            fast_score: h.semantic_score,
            lexical_score: h.lexical_score,
        })
        .collect()
}
#[allow(clippy::cast_possible_truncation)]
fn new_mat(hits: Vec<Fused>, k: usize) -> Vec<Scored> {
    hits.into_iter()
        .take(k)
        .map(|h| Scored {
            doc_id: h.doc_id,
            score: h.rrf_score as f32,
            index: h.semantic_index,
            fast_score: h.semantic_score,
            lexical_score: h.lexical_score,
        })
        .collect()
}
fn consume_scored(rows: &[Scored]) -> usize {
    rows.iter()
        .map(|row| {
            row.doc_id.len()
                + usize::from(row.score > 0.0)
                + row.index.unwrap_or_default() as usize
                + usize::from(row.fast_score.is_some())
                + usize::from(row.lexical_score.is_some())
        })
        .sum()
}
fn make(n: usize) -> Vec<Fused> {
    (0..n)
        .map(|i| Fused {
            doc_id: format!("doc-{i:06}"),
            rrf_score: 1.0 / (i as f64 + 1.0),
            semantic_index: Some(u32::try_from(i).expect("benchmark index fits u32")),
            semantic_score: Some(0.5),
            lexical_score: Some(0.3),
        })
        .collect()
}
fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("fused_materialize");
    for n in [20usize, 60, 200] {
        let input = make(n);
        let k = n; // materialize all
        let id = format!("n{n}");
        g.bench_with_input(BenchmarkId::new("clone", &id), &(), |b, ()| {
            b.iter_batched(
                || input.clone(),
                |owned| {
                    let rows = old_mat(black_box(&owned), k);
                    black_box(consume_scored(&rows));
                },
                BatchSize::SmallInput,
            );
        });
        g.bench_with_input(BenchmarkId::new("move", &id), &(), |b, ()| {
            b.iter_batched(
                || input.clone(),
                |owned| {
                    let rows = new_mat(owned, k);
                    black_box(consume_scored(&rows));
                },
                BatchSize::SmallInput,
            );
        });
    }
    g.finish();
}
criterion_group!(benches, bench);
criterion_main!(benches);
