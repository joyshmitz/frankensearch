//! `doc_id` materialization clone cost: `String` (current) vs SSO `CompactString`
//! vs `Arc<str>` — for the `limit_all` `doc_id`-clone lever (~23% of `limit_all`, the
//! largest remaining frankensearch-owned slice, per `NEGATIVE_EVIDENCE`).
//!
//! The RRF `into_owned` + `resolve_heap` + blend materialize N owned `doc_id`s per
//! `limit_all` query. Each `String` clone is a heap alloc + memcpy (~43 ns for a
//! `doc-NNNNNN` id, measured 432 µs/10k in `materialize_clone`). Two drop-in-ish
//! alternatives make the clone cheap:
//!
//! - `CompactString`: ids ≤ 24 bytes are stored **inline**, so clone is a stack
//!   memcpy — no alloc. `.as_str()`/Deref/serde all work (near-drop-in for
//!   `String`), and it degrades gracefully to a heap alloc for long ids.
//! - `Arc<str>`: clone is a refcount bump, but lacks `.as_str()` (bigger refactor
//!   surface) and needs serde `rc`.
//!
//! This measures the raw N-clone cost for a realistic short id, deciding which is
//! the right target for the (cross-crate) materialization refactor.
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-fusion --bench doc_id_clone_sso
//! ```

use std::hint::black_box;
use std::sync::Arc;

use compact_str::CompactString;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

fn short_ids(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("doc-{i:06}")).collect() // 10 bytes ⇒ inline for CompactString
}

/// Longer ids (36-byte UUID-like) — beyond `CompactString`'s 24-byte inline cap, so
/// its clone falls back to a heap alloc (parity with String). Confirms the SSO win
/// is short-id-specific and there's no regression for long ids.
fn long_ids(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| format!("urn:uuid:00000000-0000-0000-0000-{i:012}"))
        .collect()
}

fn bench_doc_id_clone(c: &mut Criterion) {
    let mut g = c.benchmark_group("doc_id_clone_sso");
    for (label, ids) in [("short", short_ids(10_000)), ("long", long_ids(10_000))] {
        // Current: N String clones (each a heap alloc + memcpy).
        let strings = ids.clone();
        g.bench_with_input(BenchmarkId::new("string", label), &strings, |b, s| {
            b.iter(|| black_box(s.clone()));
        });
        // CompactString: inline (no alloc) for short ids, heap for long.
        let compacts: Vec<CompactString> = ids.iter().map(CompactString::from).collect();
        g.bench_with_input(BenchmarkId::new("compact", label), &compacts, |b, c| {
            b.iter(|| black_box(c.clone()));
        });
        // Arc<str>: refcount bump.
        let arcs: Vec<Arc<str>> = ids.iter().map(|s| Arc::from(s.as_str())).collect();
        g.bench_with_input(BenchmarkId::new("arc", label), &arcs, |b, a| {
            b.iter(|| black_box(a.clone()));
        });
    }
    g.finish();
}

criterion_group!(benches, bench_doc_id_clone);
criterion_main!(benches);
