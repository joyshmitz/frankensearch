//! Two-tier blend final-sort benchmark: stable `sort_by` vs `sort_unstable_by`.
//!
//! `blend_two_tier` (the main two-tier quality-phase blend, `blend.rs`) finishes
//! by sorting the deduped `Vec<VectorHit>` with a **stable** `sort_by`, while the
//! sibling RRF fuse path (`rrf.rs`) uses `sort_unstable_by`. The comparator is a
//! strict total order — `score.total_cmp` then a `doc_id` tiebreak — and the
//! `blended` vec is built from an `AHashMap<&str, _>` so every `doc_id` is unique;
//! no two elements compare Equal, so a stable sort buys nothing but its mergesort
//! scratch allocation + constant factors. This bench isolates that final sort over
//! a realistic blended set (unique ids, varied f32 scores).
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
//!   rch exec -- cargo bench -p frankensearch-fusion --bench blend_reorder
//! ```

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_core::VectorHit;

/// Deterministic blended set modeling a real two-tier blend: unique doc ids and
/// **mostly-distinct** f32 scores (the real score is `alpha*q + (1-alpha)*f` over
/// normalized inputs, so collisions are rare and the `doc_id` tiebreak almost
/// never fires). The Knuth multiplicative scramble gives distinct, non-pre-sorted
/// scores; a tiny `% (n/16)` band of repeats keeps a realistic sprinkle of ties.
fn make_blended(n: usize) -> Vec<VectorHit> {
    (0..n)
        .map(|i| {
            let scrambled = i.wrapping_mul(2_654_435_761) % n;
            // Distinct base score + a small fractional jitter; ~6% of rows share a
            // coarse bucket so the tiebreak is exercised but does not dominate.
            let tie = (i % (n / 16).max(1)) as f32 * 1e-6;
            VectorHit {
                index: i as u32,
                score: scrambled as f32 * 0.5 + tie,
                doc_id: format!("doc_{i:08}"),
            }
        })
        .collect()
}

#[inline]
fn sanitize_score(s: f32) -> f32 {
    if s.is_finite() { s } else { 0.0 }
}

fn sort_stable(mut v: Vec<VectorHit>) -> VectorHit {
    v.sort_by(|left, right| {
        sanitize_score(right.score)
            .total_cmp(&sanitize_score(left.score))
            .then_with(|| left.doc_id.cmp(&right.doc_id))
    });
    v.swap_remove(0)
}

fn sort_unstable(mut v: Vec<VectorHit>) -> VectorHit {
    v.sort_unstable_by(|left, right| {
        sanitize_score(right.score)
            .total_cmp(&sanitize_score(left.score))
            .then_with(|| left.doc_id.cmp(&right.doc_id))
    });
    v.swap_remove(0)
}

fn bench_blend_sort(c: &mut Criterion) {
    let mut g = c.benchmark_group("blend_reorder");
    for &n in &[200usize, 2000] {
        let base = make_blended(n);
        let id = format!("n{n}");
        // Confirm identical top element (and, by total order, identical full order).
        debug_assert_eq!(sort_stable(base.clone()).doc_id, sort_unstable(base.clone()).doc_id);
        g.bench_with_input(BenchmarkId::new("stable", &id), &base, |b, v| {
            b.iter_batched(|| v.clone(), |v| black_box(sort_stable(v)), criterion::BatchSize::LargeInput);
        });
        g.bench_with_input(BenchmarkId::new("unstable", &id), &base, |b, v| {
            b.iter_batched(|| v.clone(), |v| black_box(sort_unstable(v)), criterion::BatchSize::LargeInput);
        });
    }
    g.finish();
}

criterion_group!(benches, bench_blend_sort);
criterion_main!(benches);
