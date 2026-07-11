//! `SearchFilterExpr::matches_doc_id` A/B: the fsfs search filter (runtime.rs) is
//! evaluated PER CANDIDATE in `apply_search_filter` and both merge loops, and the
//! current impl unconditionally allocates `doc_id.to_ascii_lowercase()` on every
//! call — even for `Extension`-only filters that never read `lowered`. The fix is
//! an alloc-free ASCII-case-insensitive substring test (needles are already
//! lowercased at parse time, and `to_ascii_lowercase` only folds ASCII bytes, so a
//! byte-wise fold is bit-identical). This bench measures the per-candidate filter
//! over a realistic candidate set for a PathContains filter (the allocating case)
//! and an Extension filter (old allocates uselessly), old vs new. Identical
//! match verdicts asserted.
use std::hint::black_box;
use std::path::Path;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

enum Clause {
    PathContains(String), // already ASCII-lowercase (as produced by parse)
    Extension(String),    // already ASCII-lowercase
}

/// Current production: unconditional `to_ascii_lowercase` allocation per call.
fn matches_old(clauses: &[Clause], doc_id: &str) -> bool {
    let lowered = doc_id.to_ascii_lowercase();
    clauses.iter().all(|clause| match clause {
        Clause::PathContains(needle) => lowered.contains(needle),
        Clause::Extension(expected) => Path::new(doc_id)
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case(expected)),
    })
}

/// Proposed: allocate `lowered` ONLY when a PathContains clause needs it (the flag
/// is precomputed once per query at parse time, modeled here by `has_path`). Path
/// filters keep the fast SIMD `str::contains` (identical to old); extension-only
/// filters allocate nothing.
fn matches_new(clauses: &[Clause], has_path: bool, doc_id: &str) -> bool {
    let lowered = has_path.then(|| doc_id.to_ascii_lowercase());
    clauses.iter().all(|clause| match clause {
        Clause::PathContains(needle) => lowered.as_deref().is_some_and(|l| l.contains(needle)),
        Clause::Extension(expected) => Path::new(doc_id)
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case(expected)),
    })
}

fn has_path_contains(clauses: &[Clause]) -> bool {
    clauses.iter().any(|c| matches!(c, Clause::PathContains(_)))
}

fn docs(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| {
            let ext = ["rs", "md", "py", "txt", "json"][i % 5];
            format!("docs/Section-{:02}/File-{:05}.{ext}", i % 24, i)
        })
        .collect()
}

fn run_old(clauses: &[Clause], docs: &[String]) -> usize {
    docs.iter().filter(|d| matches_old(clauses, d)).count()
}

fn run_new(clauses: &[Clause], docs: &[String]) -> usize {
    let has_path = has_path_contains(clauses); // precomputed once per query
    docs.iter()
        .filter(|d| matches_new(clauses, has_path, d))
        .count()
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("filter_match");
    let path_filter = vec![Clause::PathContains("section-1".to_owned())];
    let ext_filter = vec![Clause::Extension("md".to_owned())];
    for &n in &[200usize, 1000, 4000] {
        let d = docs(n);
        // Bit-identical verdicts, both filter kinds.
        assert_eq!(run_old(&path_filter, &d), run_new(&path_filter, &d));
        assert_eq!(run_old(&ext_filter, &d), run_new(&ext_filter, &d));

        let id = format!("path_n{n}");
        g.bench_with_input(BenchmarkId::new("old", &id), &(), |b, ()| {
            b.iter(|| black_box(run_old(black_box(&path_filter), black_box(&d))));
        });
        g.bench_with_input(BenchmarkId::new("new", &id), &(), |b, ()| {
            b.iter(|| black_box(run_new(black_box(&path_filter), black_box(&d))));
        });

        let id = format!("ext_n{n}");
        g.bench_with_input(BenchmarkId::new("old", &id), &(), |b, ()| {
            b.iter(|| black_box(run_old(black_box(&ext_filter), black_box(&d))));
        });
        g.bench_with_input(BenchmarkId::new("new", &id), &(), |b, ()| {
            b.iter(|| black_box(run_new(black_box(&ext_filter), black_box(&d))));
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
