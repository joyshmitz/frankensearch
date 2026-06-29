//! limit_all materialization clone cost (bd-tjkm-adjacent / VectorHit<'a> lever).
//!
//! The RRF fuse returns `Vec<FusedHit>` whose `doc_id: String` are `into_owned`
//! clones of the borrowed lexical/semantic inputs (the inputs are dropped). For
//! `limit_all` (k = N) that is N short-String allocations. The PERF_LEDGER estimated
//! eliding them (the invasive `VectorHit<'a>` borrow refactor) at "~3-5% of
//! limit_all" — but that was never measured. This isolates the actual cost: building
//! `Vec<String>` (clone, current) vs `Vec<&str>` (borrow, the refactor target) over
//! N doc_ids, so the lever's real magnitude can be compared to limit_all's ~1869 µs
//! (10k) p50 before deciding whether the cross-crate refactor is worth it.
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
//!   rch exec -- cargo bench -p frankensearch-fusion --bench materialize_clone
//! ```

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

fn doc_ids(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("doc-{i:06}")).collect()
}

/// Current: own each winner's doc_id (the RRF `into_owned`).
fn collect_owned(ids: &[String]) -> Vec<String> {
    ids.iter().map(|s| s.clone()).collect()
}

/// Refactor target: borrow (no per-hit allocation).
fn collect_borrowed(ids: &[String]) -> Vec<&str> {
    ids.iter().map(String::as_str).collect()
}

fn bench_materialize_clone(c: &mut Criterion) {
    let mut g = c.benchmark_group("materialize_clone");
    for n in [10_000usize, 100_000] {
        let ids = doc_ids(n);
        g.bench_with_input(BenchmarkId::new("owned_clone", n), &ids, |b, ids| {
            b.iter(|| black_box(collect_owned(black_box(ids))));
        });
        g.bench_with_input(BenchmarkId::new("borrowed", n), &ids, |b, ids| {
            b.iter(|| black_box(collect_borrowed(black_box(ids))));
        });
    }
    g.finish();
}

criterion_group!(benches, bench_materialize_clone);
criterion_main!(benches);
