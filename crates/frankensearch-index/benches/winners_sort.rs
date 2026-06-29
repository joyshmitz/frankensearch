//! Exact-search final winners-ordering benchmark: stable `sort_by` vs `sort_unstable_by`.
//!
//! `search.rs` / `in_memory.rs` finish a search by ordering the collected
//! `Vec<HeapEntry>` best-first with a **stable** `sort_by(compare_best_first)` —
//! three sites, including the `limit_all` scan-all path (`search.rs:183`,
//! `scan_wal_collect_all`), where `winners` holds *every* match. The comparator
//! is a strict total order (`score_key.total_cmp`, then a unique `index` tiebreak),
//! so a stable sort buys nothing but its mergesort scratch allocation + constant
//! factors. This bench isolates that final sort over realistic winners sets
//! (mostly-distinct cosine-like f32 scores, unique indices). `HeapEntry` and
//! `compare_best_first` are private, so they are replicated here with the same
//! shape (cf. the `blend_reorder` / `rrf_fuse` benches).
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-index --bench winners_sort
//! ```

use std::cmp::Ordering;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

#[derive(Clone, Copy)]
struct HeapEntry {
    index: usize,
    score: f32,
}

/// Mirrors `search::score_key`: NaN sorts as the worst score.
#[inline]
fn score_key(score: f32) -> f32 {
    if score.is_nan() { f32::NEG_INFINITY } else { score }
}

/// Mirrors `search::compare_best_first` — strict total order (unique `index` tiebreak).
#[inline]
fn compare_best_first(left: &HeapEntry, right: &HeapEntry) -> Ordering {
    match score_key(right.score).total_cmp(&score_key(left.score)) {
        Ordering::Equal => left.index.cmp(&right.index),
        other => other,
    }
}

/// Realistic winners set: unique indices, mostly-distinct cosine-like scores in
/// [-1, 1]. The Knuth multiplicative scramble distributes scores so the input is
/// not pre-sorted; collisions are rare so the `index` tiebreak almost never fires.
fn make_winners(n: usize) -> Vec<HeapEntry> {
    (0..n)
        .map(|i| {
            let scrambled = i.wrapping_mul(2_654_435_761) % n;
            HeapEntry {
                index: i,
                // Map to ~[-1, 1] with fine granularity (distinct per scrambled).
                score: (scrambled as f32 / n as f32).mul_add(2.0, -1.0),
            }
        })
        .collect()
}

fn sort_stable(mut v: Vec<HeapEntry>) -> usize {
    v.sort_by(compare_best_first);
    v[0].index
}

fn sort_unstable(mut v: Vec<HeapEntry>) -> usize {
    v.sort_unstable_by(compare_best_first);
    v[0].index
}

fn bench_winners_sort(c: &mut Criterion) {
    let mut g = c.benchmark_group("winners_sort");
    // n100 = bounded top-k; n10000/n50000 = limit_all scan-all winners.
    for &n in &[100usize, 10_000, 50_000] {
        let base = make_winners(n);
        let id = format!("n{n}");
        debug_assert_eq!(sort_stable(base.clone()), sort_unstable(base.clone()));
        g.bench_with_input(BenchmarkId::new("stable", &id), &base, |b, v| {
            b.iter_batched(
                || v.clone(),
                |v| black_box(sort_stable(v)),
                criterion::BatchSize::LargeInput,
            );
        });
        g.bench_with_input(BenchmarkId::new("unstable", &id), &base, |b, v| {
            b.iter_batched(
                || v.clone(),
                |v| black_box(sort_unstable(v)),
                criterion::BatchSize::LargeInput,
            );
        });
    }
    g.finish();
}

criterion_group!(benches, bench_winners_sort);
criterion_main!(benches);
