//! RRF final-sort **indirect index sort** probe (the untested route-next whose
//! prior rejection rested on a false premise).
//!
//! `FusedHitScratch` (`rrf.rs`) is a ~112-byte, 10-field struct, but the sort
//! comparator (`cmp_for_ranking`) reads only 4 fields, and its `doc_id` is
//! `&'a str` (BORROWED). NEGATIVE_EVIDENCE 2026-06-29 dismissed "sort a separate
//! (key, idx) array then gather" because "gather-by-index needs an unsafe move-out
//! of the FusedHitScratch Vec" — but that is wrong: gathering by index reads Copy
//! fields and `into_owned`s the borrowed `&str` (a clone, no move, no unsafe).
//!
//! So this A/B (both bit-identical output, asserted): sort the fat structs in place
//! (`struct_sort`, production today) vs sort a `Vec<u32>` of 4-byte indices with an
//! indirect comparator, then gather by index (`index_sort`). 4-byte swaps vs
//! ~112-byte swaps, at the cost of scattered comparator + gather reads — the
//! empirical question this measures. The gather (doc_id `into_owned` + score) is
//! charged to both. `iter_batched` clones the base in untimed setup.
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-fusion --bench rrf_index_sort
//! ```

use std::cmp::Ordering;
use std::hint::black_box;
use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};

/// Mirrors the real `FusedHitScratch` (borrowed `doc_id: &str`, ~112 bytes).
#[derive(Clone)]
#[allow(dead_code)]
struct Scratch<'a> {
    doc_id: &'a str,
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

/// The shipped comparator (RRF desc, in_both desc, lexical desc, doc_id asc).
#[inline]
fn cmp_current(a: &Scratch, b: &Scratch) -> Ordering {
    b.rrf_score
        .total_cmp(&a.rrf_score)
        .then(b.in_both_sources.cmp(&a.in_both_sources))
        .then_with(|| {
            let la = a.lexical_score.unwrap_or(f32::NEG_INFINITY);
            let lb = b.lexical_score.unwrap_or(f32::NEG_INFINITY);
            lb.total_cmp(&la)
        })
        .then_with(|| a.doc_id.cmp(b.doc_id))
}

/// Output row: owned doc_id (the `into_owned` the real fuse pays) + score. Charged
/// to both arms so the delta is purely the sort + read-order.
type Row = (String, f32);

#[allow(clippy::cast_possible_truncation)]
fn gather_struct(v: &[Scratch]) -> Vec<Row> {
    v.iter()
        .map(|s| (s.doc_id.to_owned(), s.rrf_score as f32))
        .collect()
}

#[allow(clippy::cast_possible_truncation)]
fn gather_index(v: &[Scratch], idx: &[u32]) -> Vec<Row> {
    idx.iter()
        .map(|&i| {
            let s = &v[i as usize];
            (s.doc_id.to_owned(), s.rrf_score as f32)
        })
        .collect()
}

fn doc_ids(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("doc-{i:06}")).collect()
}

/// Realistic `limit_all` shape: RRF scores `1/(60+rank)` with a wrap that forces
/// pervasive ties, ~20% in-both.
fn build<'a>(ids: &'a [String]) -> Vec<Scratch<'a>> {
    let n = ids.len();
    ids.iter()
        .enumerate()
        .map(|(i, id)| {
            let rank = i % (n * 4 / 5).max(1);
            let in_both = i % 5 == 0;
            Scratch {
                doc_id: id.as_str(),
                rrf_score: 1.0 / (60.0 + rank as f64),
                lexical_rank: in_both.then_some(i),
                semantic_rank: Some(rank),
                semantic_index: Some(i as u32),
                graph_rank: None,
                #[allow(clippy::cast_precision_loss)]
                lexical_score: in_both.then_some((i % 97) as f32 * 0.01),
                semantic_score: Some(0.5),
                graph_score: None,
                in_both_sources: in_both,
            }
        })
        .collect()
}

fn struct_sort(mut v: Vec<Scratch>) -> Vec<Row> {
    v.sort_unstable_by(cmp_current);
    gather_struct(&v)
}

#[allow(clippy::cast_possible_truncation)]
fn index_sort(v: Vec<Scratch>) -> Vec<Row> {
    let mut idx: Vec<u32> = (0..v.len() as u32).collect();
    idx.sort_unstable_by(|&a, &b| cmp_current(&v[a as usize], &v[b as usize]));
    gather_index(&v, &idx)
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("rrf_index_sort");
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(3));
    g.sample_size(30);

    for &n in &[10_000usize, 50_000] {
        let ids = doc_ids(n);
        let base = build(&ids);

        // Bit-identity: same final ordering (doc_id sequence).
        let a = struct_sort(base.clone());
        let b = index_sort(base.clone());
        assert_eq!(a, b, "index_sort reorders vs struct_sort at n={n}");

        g.bench_with_input(BenchmarkId::new("struct_sort", n), &n, |bch, _| {
            bch.iter_batched(
                || base.clone(),
                |v| black_box(struct_sort(v)),
                BatchSize::LargeInput,
            );
        });
        g.bench_with_input(BenchmarkId::new("index_sort", n), &n, |bch, _| {
            bch.iter_batched(
                || base.clone(),
                |v| black_box(index_sort(v)),
                BatchSize::LargeInput,
            );
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
