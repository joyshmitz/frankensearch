//! Same-worker A/B for the `ScoredResult.metadata` clone at `limit_all`.
//!
//! The async searcher materializes each winner's metadata by **deep-cloning** its
//! `serde_json::Value` (`searcher.rs:2514`, `lexical_metadata_by_doc.get(..).cloned()`).
//! At `limit_all` (k = N) that is N deep clones of a nested JSON object per query —
//! each re-allocates the map + every string + nested arrays. The BOLD proxy uses
//! *tiny* metadata so this cost is invisible there; realistic document metadata
//! (title/path/tags/…) is not tiny.
//!
//! This measures the lever: deep `Value` clone (current `Option<Value>`) vs an
//! `Arc<Value>` refcount bump (the candidate `Option<Arc<Value>>` refactor), both
//! arms in ONE binary on ONE worker over N realistic metadata objects. If Arc is
//! materially cheaper, `ScoredResult.metadata: Option<Arc<Value>>` is worth the
//! (public-API) change for rich-metadata + `limit_all` workloads.
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-fusion --bench metadata_clone_ab
//! ```

use std::hint::black_box;
use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use serde_json::json;

/// A representative document-metadata object (what a real corpus attaches, not the
/// BOLD proxy's tiny stub): a handful of string fields + a nested tag array.
fn realistic_metadata(i: usize) -> serde_json::Value {
    json!({
        "title": format!("Document number {i} — a reasonably descriptive title"),
        "path": format!("/home/user/projects/corpus/section-{}/doc-{i:06}.md", i % 32),
        "language": "markdown",
        "size_bytes": 4096 + i,
        "modified": "2026-07-02T00:00:00Z",
        "author": "corpus-generator",
        "tags": ["reference", "indexed", "hybrid-search", "section"],
        "line_count": 120 + (i % 400),
    })
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("metadata_clone_ab");
    for n in [10_000usize, 100_000] {
        let values: Vec<serde_json::Value> = (0..n).map(realistic_metadata).collect();
        let arcs: Vec<Arc<serde_json::Value>> =
            values.iter().cloned().map(Arc::new).collect();

        // Current: N deep clones of Option<Value> (the limit_all materialization).
        g.bench_with_input(BenchmarkId::new("value_deep_clone", n), &values, |b, vs| {
            b.iter(|| {
                let out: Vec<Option<serde_json::Value>> =
                    black_box(vs).iter().map(|v| Some(v.clone())).collect();
                black_box(out)
            });
        });
        // Candidate: N Arc refcount bumps of Option<Arc<Value>>.
        g.bench_with_input(BenchmarkId::new("arc_clone", n), &arcs, |b, as_| {
            b.iter(|| {
                let out: Vec<Option<Arc<serde_json::Value>>> =
                    black_box(as_).iter().map(|a| Some(Arc::clone(a))).collect();
                black_box(out)
            });
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
