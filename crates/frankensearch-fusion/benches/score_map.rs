//! sync fast/quality score-map build: old HashMap<String,f32> (clone doc_id) vs
//! new HashMap<&str,f32> (borrow). The maps are only .get()-looked-up by &str, so
//! borrowing is bit-identical and drops a per-candidate String clone.
use std::collections::HashMap;
use std::hint::black_box;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

#[derive(Clone)]
struct VHit { index: u32, score: f32, doc_id: String }

fn build_old(fast: &[VHit], quality: &[VHit]) -> usize {
    let f: HashMap<String, f32> = fast.iter().map(|h| (h.doc_id.clone(), h.score)).collect();
    let q: HashMap<String, f32> = quality.iter().map(|h| (h.doc_id.clone(), h.score)).collect();
    f.len() + q.len()
}
fn build_new(fast: &[VHit], quality: &[VHit]) -> usize {
    let f: HashMap<&str, f32> = fast.iter().map(|h| (h.doc_id.as_str(), h.score)).collect();
    let q: HashMap<&str, f32> = quality.iter().map(|h| (h.doc_id.as_str(), h.score)).collect();
    f.len() + q.len()
}
fn make(n: usize, off: usize) -> Vec<VHit> {
    (0..n).map(|i| VHit { index: (off + i) as u32, score: 1.0 / (i as f32 + 1.0), doc_id: format!("doc-{:06}", off + i) }).collect()
}
fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("score_map");
    for n in [30usize, 90, 300] {
        let fast = make(n, 0);
        let quality = make(n, n / 3);
        debug_assert_eq!(build_old(&fast, &quality), build_new(&fast, &quality));
        let id = format!("n{n}");
        g.bench_with_input(BenchmarkId::new("clone", &id), &(), |b, ()| b.iter(|| black_box(build_old(black_box(&fast), black_box(&quality)))));
        g.bench_with_input(BenchmarkId::new("borrow", &id), &(), |b, ()| b.iter(|| black_box(build_new(black_box(&fast), black_box(&quality)))));
    }
    g.finish();
}
criterion_group!(benches, bench);
criterion_main!(benches);
