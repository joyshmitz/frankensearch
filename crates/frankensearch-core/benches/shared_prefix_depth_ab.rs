//! Paired A/B for `fsfs::ranking_priors::shared_prefix_depth`.
//!
//! The `path_proximity` ranking prior counts how many leading path components a
//! query origin path and a candidate's file path share, per candidate, on every
//! ranked query where the prior is enabled. The shipping helper collected both
//! paths' components into two `Vec<&str>` and then `zip`ped them — but the vectors
//! were used for nothing except that `zip`, so walking the split iterators in
//! lockstep is byte-identical and drops two per-call allocations.
//!
//! This bench (self-contained — the function is a pure string algorithm, hosted in
//! `core` to avoid the ~10-minute `fsfs` compile) mirrors both variants and asserts
//! identical counts before timing:
//!
//! - `alloc` : collect two `Vec<&str>`, then `zip` (the shipping path).
//! - `zip`   : `zip` the split iterators directly (the new path).
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/<lane> \
//!   rch exec -- cargo bench -p frankensearch-core --bench shared_prefix_depth_ab
//! ```

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

/// Shipping path: collect both component lists, then zip.
fn shared_prefix_depth_alloc(path_a: &str, path_b: &str) -> usize {
    let components_a: Vec<&str> = path_a.split('/').filter(|c| !c.is_empty()).collect();
    let components_b: Vec<&str> = path_b.split('/').filter(|c| !c.is_empty()).collect();
    components_a
        .iter()
        .zip(components_b.iter())
        .take_while(|(a, b)| a == b)
        .count()
}

/// New path: zip the split iterators directly (zero allocation).
fn shared_prefix_depth_zip(path_a: &str, path_b: &str) -> usize {
    let components_a = path_a.split('/').filter(|c| !c.is_empty());
    let components_b = path_b.split('/').filter(|c| !c.is_empty());
    components_a
        .zip(components_b)
        .take_while(|(a, b)| a == b)
        .count()
}

/// (`query_path`, `candidate_path`) pairs with varied shared-prefix depths, shaped
/// like a real source tree — the query origin plus a fanned-out candidate set.
fn pairs() -> Vec<(&'static str, &'static str)> {
    let query = "crates/frankensearch-fsfs/src/ranking_priors.rs";
    let candidates = [
        "crates/frankensearch-fsfs/src/query_execution.rs", // shared 3
        "crates/frankensearch-fsfs/src/ranking_priors.rs",  // shared 4 (identical)
        "crates/frankensearch-fsfs/benches/fuse_merge_ab.rs", // shared 2
        "crates/frankensearch-core/src/traits.rs",          // shared 1
        "crates/frankensearch-index/src/simd.rs",           // shared 1
        "docs/PERF_LEDGER.md",                              // shared 0
        "crates/frankensearch-fsfs/src/query_planning.rs",  // shared 3
        "crates/frankensearch-fsfs/src/orchestration.rs",   // shared 3
    ];
    candidates.into_iter().map(|c| (query, c)).collect()
}

fn run_alloc(ps: &[(&str, &str)]) -> usize {
    ps.iter()
        .map(|(a, b)| shared_prefix_depth_alloc(a, b))
        .sum()
}

fn run_zip(ps: &[(&str, &str)]) -> usize {
    ps.iter().map(|(a, b)| shared_prefix_depth_zip(a, b)).sum()
}

fn bench(c: &mut Criterion) {
    let ps = pairs();

    // Parity gate: identical counts per pair.
    for (a, b) in &ps {
        assert_eq!(
            shared_prefix_depth_alloc(a, b),
            shared_prefix_depth_zip(a, b),
            "parity for ({a:?}, {b:?})"
        );
    }

    let mut group = c.benchmark_group("shared_prefix_depth");
    for &n in &[8usize, 100, 500] {
        // Replicate the candidate set to a realistic per-query candidate count.
        let batch: Vec<(&str, &str)> = ps.iter().copied().cycle().take(n).collect();
        assert_eq!(run_alloc(&batch), run_zip(&batch), "batch parity (n={n})");

        group.bench_with_input(BenchmarkId::new("alloc", n), &batch, |b, batch| {
            b.iter(|| black_box(run_alloc(black_box(batch))));
        });
        group.bench_with_input(BenchmarkId::new("zip", n), &batch, |b, batch| {
            b.iter(|| black_box(run_zip(black_box(batch))));
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
