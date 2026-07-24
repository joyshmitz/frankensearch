//! `RRF` final-sort comparator probe: the shipped `cmp_for_ranking`
//! (`f64::total_cmp` + `f32::total_cmp` + `str::cmp` per comparison, evaluated
//! O(N log N)×) vs a comparator over **precomputed** monotonic integer keys
//! computed O(N)× — `total_cmp`-equivalent `i64`/`i32` keys for the score levels and
//! a big-endian `u64` `doc_id` prefix that resolves the pervasive `String` tiebreak in
//! one integer compare (full `str::cmp` fallback only on prefix-tie).
//!
//! This is the largest single slice of the `limit_all` gap (the `RRF` full sort,
//! ~22% of `limit_all`, `rrf.rs:341`). The radix refutation (`NEGATIVE_EVIDENCE`
//! 2026-06-29) rejected *replacing the sort algorithm*; this keeps
//! `sort_unstable_by` and only makes the comparator cheaper — bit-identical order
//! (asserted before timing).
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-fusion --bench rrf_sort_key
//! ```

use std::cmp::Ordering;
use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

/// Mirrors the private `FusedHitScratch` shape (current production: no keys).
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

/// Proposed shape: `FusedHitScratch` + 3 precomputed sort keys (bigger struct).
#[derive(Clone)]
#[allow(dead_code)]
struct ScratchKeyed {
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
    rrf_key: i64,
    lex_key: i32,
    prefix: u64,
}

/// `total_cmp`-equivalent ascending `i64` key for an `f64` (the exact transform
/// `f64::total_cmp` applies internally).
#[inline]
fn key_f64(x: f64) -> i64 {
    let b = x.to_bits().cast_signed();
    b ^ ((b >> 63).cast_unsigned() >> 1).cast_signed()
}

#[inline]
fn key_f32(x: f32) -> i32 {
    let b = x.to_bits().cast_signed();
    b ^ ((b >> 31).cast_unsigned() >> 1).cast_signed()
}

/// First 8 bytes of `doc_id`, big-endian, zero-padded — orders identically to
/// `str::cmp` on the first 8 bytes (ASCII, no interior NUL).
#[inline]
fn doc_prefix(s: &str) -> u64 {
    let bytes = s.as_bytes();
    let n = bytes.len().min(8);
    let mut buf = [0u8; 8];
    buf[..n].copy_from_slice(&bytes[..n]);
    u64::from_be_bytes(buf)
}

/// The shipped comparator (replicated): `RRF` desc, `in_both` desc, lexical desc,
/// `doc_id` asc.
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

/// Precomputed-key comparator: integer compares for the score levels + `doc_id`
/// prefix, full `str::cmp` only on prefix-tie. Bit-identical order.
fn cmp_fast(a: &ScratchKeyed, b: &ScratchKeyed) -> Ordering {
    b.rrf_key
        .cmp(&a.rrf_key)
        .then(b.in_both_sources.cmp(&a.in_both_sources))
        .then(b.lex_key.cmp(&a.lex_key))
        .then(a.prefix.cmp(&b.prefix))
        .then_with(|| a.doc_id.cmp(&b.doc_id))
}

/// Build a realistic `limit_all`-shaped candidate set: N docs, RRF scores from
/// `1/(60+rank)` with a wrap that forces pervasive ties, ~20% in-both.
fn build(n: usize) -> Vec<Scratch> {
    (0..n)
        .map(|i| {
            let rank = i % (n * 4 / 5).max(1); // wrap → pervasive score ties
            let rrf_score = 1.0 / (60.0 + rank as f64);
            let in_both = i % 5 == 0;
            let lexical_score = if in_both {
                #[allow(clippy::cast_precision_loss)]
                Some((i % 97) as f32 * 0.01)
            } else {
                None
            };
            Scratch {
                doc_id: format!("doc-{i:06}"),
                rrf_score,
                lexical_rank: in_both.then_some(i),
                semantic_rank: Some(rank),
                semantic_index: Some(u32::try_from(i).expect("benchmark index fits u32")),
                graph_rank: None,
                lexical_score,
                semantic_score: Some(0.5),
                graph_score: None,
                in_both_sources: in_both,
            }
        })
        .collect()
}

/// Promote a current-shape scratch to the keyed shape WITHOUT computing keys
/// (keys are filled in the timed region to charge the precompute fairly).
fn to_keyed(s: &Scratch) -> ScratchKeyed {
    ScratchKeyed {
        doc_id: s.doc_id.clone(),
        rrf_score: s.rrf_score,
        lexical_rank: s.lexical_rank,
        semantic_rank: s.semantic_rank,
        semantic_index: s.semantic_index,
        graph_rank: s.graph_rank,
        lexical_score: s.lexical_score,
        semantic_score: s.semantic_score,
        graph_score: s.graph_score,
        in_both_sources: s.in_both_sources,
        rrf_key: 0,
        lex_key: 0,
        prefix: 0,
    }
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("rrf_sort_key");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(30);

    for &n in &[10_000_usize, 50_000] {
        let base = build(n);
        let keyed_base: Vec<ScratchKeyed> = base.iter().map(to_keyed).collect();

        // Equivalence: identical order from both comparators.
        let mut va = base.clone();
        let mut vb = keyed_base.clone();
        for h in &mut vb {
            h.rrf_key = key_f64(h.rrf_score);
            h.lex_key = key_f32(h.lexical_score.unwrap_or(f32::NEG_INFINITY));
            h.prefix = doc_prefix(&h.doc_id);
        }
        va.sort_unstable_by(cmp_current);
        vb.sort_unstable_by(cmp_fast);
        assert!(
            va.iter().zip(&vb).all(|(x, y)| x.doc_id == y.doc_id),
            "fast comparator reorders vs current at n={n}"
        );

        // "current" arm: small struct + total_cmp/str comparator (production today).
        group.bench_with_input(BenchmarkId::new("current", n), &n, |bch, _| {
            bch.iter(|| {
                let mut v = base.clone();
                v.sort_unstable_by(cmp_current);
                black_box(&v[0].doc_id);
            });
        });
        // "fast" arm: bigger keyed struct + the O(N) precompute pass + int-key sort.
        group.bench_with_input(BenchmarkId::new("fast", n), &n, |bch, _| {
            bch.iter(|| {
                let mut v = keyed_base.clone();
                for h in &mut v {
                    h.rrf_key = key_f64(h.rrf_score);
                    h.lex_key = key_f32(h.lexical_score.unwrap_or(f32::NEG_INFINITY));
                    h.prefix = doc_prefix(&h.doc_id);
                }
                v.sort_unstable_by(cmp_fast);
                black_box(&v[0].doc_id);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
