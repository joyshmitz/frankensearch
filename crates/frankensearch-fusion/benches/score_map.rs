//! sync fast/quality score-map build: old `HashMap<String, f32>` (clone `doc_id`)
//! vs borrowed `HashMap<&str, f32>` vs the aligned numeric lookup used by the
//! vector-index refined path.

use std::collections::HashMap;
use std::hint::black_box;

use ahash::AHashMap;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

#[derive(Clone)]
struct VHit {
    index: u32,
    score: f32,
    doc_id: String,
}

fn build_old(fast: &[VHit], quality: &[VHit]) -> usize {
    let f: HashMap<String, f32> = fast.iter().map(|h| (h.doc_id.clone(), h.score)).collect();
    let q: HashMap<String, f32> = quality
        .iter()
        .map(|h| (h.doc_id.clone(), h.score))
        .collect();
    f.len() + q.len()
}

fn build_new(fast: &[VHit], quality: &[VHit]) -> usize {
    let f: HashMap<&str, f32> = fast.iter().map(|h| (h.doc_id.as_str(), h.score)).collect();
    let q: HashMap<&str, f32> = quality
        .iter()
        .map(|h| (h.doc_id.as_str(), h.score))
        .collect();
    f.len() + q.len()
}

fn build_lookup_current(fast: &[VHit], quality_scores: &[Option<f32>], blended: &[VHit]) -> f32 {
    let fast_scores = fast
        .iter()
        .map(|hit| (hit.doc_id.as_str(), hit.score))
        .collect::<AHashMap<&str, f32>>();
    let quality_scores = fast
        .iter()
        .zip(quality_scores.iter())
        .filter_map(|(hit, score)| score.map(|s| (hit.doc_id.as_str(), s)))
        .collect::<AHashMap<&str, f32>>();

    blended.iter().fold(0.0, |acc, hit| {
        let fast = fast_scores
            .get(hit.doc_id.as_str())
            .copied()
            .unwrap_or(hit.score);
        let quality = quality_scores
            .get(hit.doc_id.as_str())
            .copied()
            .unwrap_or(0.0);
        acc + fast + quality
    })
}

fn build_lookup_aligned(fast: &[VHit], quality_scores: &[Option<f32>], blended: &[VHit]) -> f32 {
    let Some(first) = fast.first() else {
        return 0.0;
    };
    let (mut min_index, mut max_index) = (first.index, first.index);
    for hit in &fast[1..] {
        min_index = min_index.min(hit.index);
        max_index = max_index.max(hit.index);
    }

    let span = max_index.saturating_sub(min_index).saturating_add(1);
    let dense_limit =
        u32::try_from(fast.len().saturating_mul(4).saturating_add(1024)).unwrap_or(u32::MAX);

    if span <= dense_limit {
        let span = usize::try_from(span).expect("bench span fits usize");
        let mut scores = vec![None; span];
        for (position, hit) in fast.iter().enumerate() {
            let slot =
                usize::try_from(hit.index - min_index).expect("bench dense offset fits usize");
            scores[slot] = Some((hit.score, quality_scores[position]));
        }
        blended.iter().fold(0.0, |acc, hit| {
            let slot =
                usize::try_from(hit.index - min_index).expect("bench dense offset fits usize");
            let (fast, quality) = scores[slot].unwrap_or((hit.score, None));
            acc + fast + quality.unwrap_or(0.0)
        })
    } else {
        let score_map = fast
            .iter()
            .zip(quality_scores.iter())
            .map(|(hit, quality)| (hit.index, (hit.score, *quality)))
            .collect::<AHashMap<u32, (f32, Option<f32>)>>();
        blended.iter().fold(0.0, |acc, hit| {
            let (fast, quality) = score_map
                .get(&hit.index)
                .copied()
                .unwrap_or((hit.score, None));
            acc + fast + quality.unwrap_or(0.0)
        })
    }
}

fn make(n: usize, off: usize) -> Vec<VHit> {
    (0..n)
        .map(|i| VHit {
            index: u32::try_from(off + i).expect("bench index fits u32"),
            score: 1.0 / (i as f32 + 1.0),
            doc_id: format!("doc-{:06}", off + i),
        })
        .collect()
}

fn quality_scores(n: usize) -> Vec<Option<f32>> {
    (0..n)
        .map(|i| {
            if i % 10 == 7 {
                None
            } else {
                Some(0.5 + 0.5 * (((i * 2_654_435_761usize) % n) as f32) / n as f32)
            }
        })
        .collect()
}

fn shuffled(mut hits: Vec<VHit>) -> Vec<VHit> {
    hits.sort_by(|left, right| {
        let lhs = (left.index as usize).wrapping_mul(2_654_435_761usize);
        let rhs = (right.index as usize).wrapping_mul(2_654_435_761usize);
        lhs.cmp(&rhs)
    });
    hits
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("score_map");
    for n in [30usize, 90, 300] {
        let fast = make(n, 0);
        let quality = make(n, n / 3);
        debug_assert_eq!(build_old(&fast, &quality), build_new(&fast, &quality));
        let id = format!("n{n}");
        g.bench_with_input(BenchmarkId::new("clone", &id), &(), |b, ()| {
            b.iter(|| black_box(build_old(black_box(&fast), black_box(&quality))));
        });
        g.bench_with_input(BenchmarkId::new("borrow", &id), &(), |b, ()| {
            b.iter(|| black_box(build_new(black_box(&fast), black_box(&quality))));
        });
    }

    for n in [10_000usize, 100_000] {
        let fast = make(n, 0);
        let quality = quality_scores(n);
        let blended = shuffled(fast.clone());
        let current = build_lookup_current(&fast, &quality, &blended);
        let aligned = build_lookup_aligned(&fast, &quality, &blended);
        assert_eq!(current.to_bits(), aligned.to_bits());

        g.bench_with_input(BenchmarkId::new("lookup_current", n), &(), |b, ()| {
            b.iter(|| {
                black_box(build_lookup_current(
                    black_box(&fast),
                    black_box(&quality),
                    black_box(&blended),
                ));
            });
        });
        g.bench_with_input(BenchmarkId::new("lookup_aligned", n), &(), |b, ()| {
            b.iter(|| {
                black_box(build_lookup_aligned(
                    black_box(&fast),
                    black_box(&quality),
                    black_box(&blended),
                ));
            });
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
