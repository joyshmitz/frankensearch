//! Filtered-scan prescreen: `matches()` (re-hash per `doc_id`) vs
//! `matches_doc_id_hash()` (precomputed hash) — the per-vector cost the in-memory
//! filtered scan pays for a `BitsetFilter`.
//!
//! The committed in-memory scan called `SearchFilter::matches(doc_id)`, which for
//! `BitsetFilter` re-hashes the `doc_id` **string** every vector. The FSVI scan
//! already pre-screens with a precomputed hash via `matches_doc_id_hash`; this
//! bench quantifies the per-vector saving (then in-memory was wired to match).
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
//!   rch exec -- cargo bench -p frankensearch-core --bench filter_prescreen
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_core::filter::{BitsetFilter, SearchFilter, fnv1a_hash};

fn bench_filter_prescreen(c: &mut Criterion) {
    const N: usize = 10_000;
    let doc_ids: Vec<String> = (0..N).map(|i| format!("doc-{i:06}")).collect();
    // Filter allows ~half the docs (selective filter — the scan checks every vector).
    let filter = BitsetFilter::from_doc_ids(doc_ids.iter().step_by(2).cloned());
    let precomputed: Vec<u64> = doc_ids.iter().map(|id| fnv1a_hash(id.as_bytes())).collect();

    let mut g = c.benchmark_group("filter_prescreen");
    g.bench_function("matches_rehash", |b| {
        b.iter(|| {
            let mut kept = 0usize;
            for id in &doc_ids {
                if filter.matches(black_box(id), None) {
                    kept += 1;
                }
            }
            black_box(kept)
        });
    });
    g.bench_function("matches_doc_id_hash", |b| {
        b.iter(|| {
            let mut kept = 0usize;
            for &h in &precomputed {
                if filter.matches_doc_id_hash(black_box(h), None) == Some(true) {
                    kept += 1;
                }
            }
            black_box(kept)
        });
    });
    g.finish();
}

criterion_group!(benches, bench_filter_prescreen);
criterion_main!(benches);
