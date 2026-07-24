//! `compute_rank_changes` rank-map build: old (clone into `Vec<VectorHit>`) vs new
//! (borrow `doc_id` straight from `ScoredResult`). Bit-identical maps; the new path
//! drops 2 `Vec` allocs + 2*N `String` clones per sync query.
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::collections::HashMap;
use std::hint::black_box;

#[derive(Clone)]
struct Scored {
    doc_id: String,
    score: f32,
    index: Option<u32>,
}
#[derive(Clone)]
struct VHit {
    _index: u32,
    _score: f32,
    doc_id: String,
}

fn rank_map_vhits(hits: &[VHit]) -> HashMap<&str, usize> {
    let mut m = HashMap::with_capacity(hits.len());
    for (r, h) in hits.iter().enumerate() {
        m.entry(h.doc_id.as_str()).or_insert(r);
    }
    m
}
// OLD: clone into Vec<VHit>, then build maps.
fn old_build(initial: &[Scored], refined: &[Scored]) -> usize {
    let ih: Vec<VHit> = initial
        .iter()
        .enumerate()
        .map(|(i, h)| VHit {
            _index: h
                .index
                .unwrap_or_else(|| u32::try_from(i).expect("benchmark indices fit in u32")),
            _score: h.score,
            doc_id: h.doc_id.clone(),
        })
        .collect();
    let rh: Vec<VHit> = refined
        .iter()
        .enumerate()
        .map(|(i, h)| VHit {
            _index: h
                .index
                .unwrap_or_else(|| u32::try_from(i).expect("benchmark indices fit in u32")),
            _score: h.score,
            doc_id: h.doc_id.clone(),
        })
        .collect();
    let im = rank_map_vhits(&ih);
    let rm = rank_map_vhits(&rh);
    im.len() + rm.len()
}
// NEW: borrow doc_id straight from Scored.
fn new_build(initial: &[Scored], refined: &[Scored]) -> usize {
    fn rm(hits: &[Scored]) -> HashMap<&str, usize> {
        let mut m = HashMap::with_capacity(hits.len());
        for (r, h) in hits.iter().enumerate() {
            m.entry(h.doc_id.as_str()).or_insert(r);
        }
        m
    }
    let im = rm(initial);
    let rm2 = rm(refined);
    im.len() + rm2.len()
}
fn make(n: usize, off: usize) -> Vec<Scored> {
    (0..n)
        .map(|i| Scored {
            doc_id: format!("doc-{:06}", off + i),
            score: 1.0 / (i as f32 + 1.0),
            index: Some(u32::try_from(off + i).expect("benchmark indices fit in u32")),
        })
        .collect()
}
fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("rank_map");
    for n in [20usize, 60, 200] {
        let initial = make(n, 0);
        let refined = make(n, n / 4); // partial overlap
        debug_assert_eq!(old_build(&initial, &refined), new_build(&initial, &refined));
        let id = format!("n{n}");
        g.bench_with_input(BenchmarkId::new("clone", &id), &(), |b, ()| {
            b.iter(|| black_box(old_build(black_box(&initial), black_box(&refined))));
        });
        g.bench_with_input(BenchmarkId::new("borrow", &id), &(), |b, ()| {
            b.iter(|| black_box(new_build(black_box(&initial), black_box(&refined))));
        });
    }
    g.finish();
}
criterion_group!(benches, bench);
criterion_main!(benches);
