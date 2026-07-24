//! Does building the RRF sort input in semantic order (near-sorted) beat the
//! shipped random `hits.into_values()` order, AFTER charging the reorder cost?
//!
//! The shipped path: `hits.into_values().collect()` (random hashmap order) then
//! `sort_unstable_by`. pdqsort is adaptive — for `limit_all` the semantic list is
//! already in fused order for the vector-only majority, so reordering the input
//! into semantic order before the sort is bit-identical (same total order) yet
//! the sort runs in ~O(N). The honest question is whether the O(N) reorder pass
//! (N hashmap `remove`s, which `into_values` does NOT pay) is repaid by the sort.
//!
//! Both arms clone the same prebuilt `AHashMap` (common overhead) inside the
//! timed region, then differ only in collect-method + the resulting sort:
//! - `current` : `into_values().collect()` (random) → sort.
//! - `reorder` : drain in semantic order via `remove` → sort (near-sorted input).
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-fusion --bench rrf_sort_order
//! ```

use std::cmp::Ordering;
use std::hint::black_box;
use std::time::Duration;

use ahash::AHashMap;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

#[derive(Clone)]
#[allow(dead_code)]
struct Scratch {
    doc_id: String,
    rrf_score: f64,
    lexical_rank: Option<usize>,
    semantic_rank: Option<usize>,
    semantic_index: Option<u32>,
    graph_rank: Option<usize>,
    lexical_score: Option<f32>,
    semantic_score: Option<f32>,
    graph_score: Option<f32>,
    in_both_sources: bool,
}

fn cmp_current(a: &Scratch, b: &Scratch) -> Ordering {
    b.rrf_score
        .total_cmp(&a.rrf_score)
        .then(b.in_both_sources.cmp(&a.in_both_sources))
        .then_with(|| {
            let la = a.lexical_score.unwrap_or(f32::NEG_INFINITY);
            let lb = b.lexical_score.unwrap_or(f32::NEG_INFINITY);
            lb.total_cmp(&la)
        })
        .then_with(|| a.doc_id.cmp(&b.doc_id))
}

/// Returns (map keyed by `doc_id`, semantic-ordered `doc_id` list). Semantic order =
/// ascending `semantic_rank` = fused order for the vector-only majority.
fn build(n: usize) -> (AHashMap<String, Scratch>, Vec<String>) {
    let mut map = AHashMap::with_capacity(n);
    let mut sem: Vec<(usize, String)> = Vec::with_capacity(n);
    for i in 0..n {
        let rank = i % (n * 4 / 5).max(1);
        let rrf_score = 1.0 / (60.0 + rank as f64);
        let in_both = i % 5 == 0;
        let lexical_score = if in_both {
            #[allow(clippy::cast_precision_loss)]
            Some((i % 97) as f32 * 0.01)
        } else {
            None
        };
        let doc_id = format!("doc-{i:06}");
        sem.push((rank, doc_id.clone()));
        map.insert(
            doc_id.clone(),
            Scratch {
                doc_id,
                rrf_score,
                lexical_rank: in_both.then_some(i),
                semantic_rank: Some(rank),
                semantic_index: Some(u32::try_from(i).expect("benchmark indices fit in u32")),
                graph_rank: None,
                lexical_score,
                semantic_score: Some(0.5),
                graph_score: None,
                in_both_sources: in_both,
            },
        );
    }
    // Semantic-score order (what the vector tier already produces).
    sem.sort_unstable_by_key(|(r, _)| *r);
    (map, sem.into_iter().map(|(_, d)| d).collect())
}

#[allow(
    clippy::many_single_char_names,
    reason = "the benchmark uses compact conventional names for sort arms and rows"
)]
fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("rrf_sort_order");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(30);

    for &n in &[10_000_usize, 50_000] {
        let (map, sem_order) = build(n);

        // Equivalence: both produce identical sorted output.
        let mut a: Vec<Scratch> = map.clone().into_values().collect();
        a.sort_unstable_by(cmp_current);
        let mut m2 = map.clone();
        let mut b: Vec<Scratch> = Vec::with_capacity(m2.len());
        for k in &sem_order {
            if let Some(v) = m2.remove(k) {
                b.push(v);
            }
        }
        b.extend(m2.into_values());
        b.sort_unstable_by(cmp_current);
        assert!(
            a.iter().zip(&b).all(|(x, y)| x.doc_id == y.doc_id),
            "reorder changes sorted output at n={n}"
        );

        group.bench_with_input(BenchmarkId::new("current", n), &n, |bch, _| {
            bch.iter(|| {
                let mut v: Vec<Scratch> = map.clone().into_values().collect();
                v.sort_unstable_by(cmp_current);
                black_box(&v[0].doc_id);
            });
        });
        group.bench_with_input(BenchmarkId::new("reorder", n), &n, |bch, _| {
            bch.iter(|| {
                let mut m = map.clone();
                let mut v: Vec<Scratch> = Vec::with_capacity(m.len());
                for k in &sem_order {
                    if let Some(s) = m.remove(k) {
                        v.push(s);
                    }
                }
                v.extend(m.into_values());
                v.sort_unstable_by(cmp_current);
                black_box(&v[0].doc_id);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
