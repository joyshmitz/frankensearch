//! Same-worker A/B for the `DocId = CompactString` landing (`8529084`).
//!
//! `doc_id_clone_sso` measured the *bare* doc_id clone (`String` 438 µs vs
//! `CompactString` 14.73 µs / 10k = 29.8×). This bench folds that clone into the
//! **full `FusedHit` materialization** — the exact `FusedHitScratch::into_owned`
//! the landing changed (rrf.rs:113): building the 10-field struct (9 `Copy`
//! fields + `doc_id: self.doc_id.into()`) over N borrowed `&str` winners, the
//! `limit_all` output shape. Both arms are identical except the `doc_id` target
//! type, run in ONE binary on ONE worker, so the ratio is the landing's real
//! per-query materialization delta with the `Copy`-field overhead charged to
//! BOTH arms (no cross-worker noise, unlike comparing to a saved baseline).
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-fusion --bench docid_materialize_ab
//! ```

use std::hint::black_box;

use compact_str::CompactString;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

/// Borrowed winner: mirrors `FusedHitScratch<'a>` (Copy fields + `doc_id: &str`).
#[derive(Clone, Copy)]
struct Winner<'a> {
    doc_id: &'a str,
    rrf_score: f64,
    lexical_rank: Option<usize>,
    semantic_rank: Option<usize>,
    semantic_index: Option<u32>,
    lexical_score: Option<f32>,
    semantic_score: Option<f32>,
    in_both_sources: bool,
}

/// `FusedHit` with a `String` doc_id — the pre-landing production type.
struct FusedHitString {
    doc_id: String,
    rrf_score: f64,
    lexical_rank: Option<usize>,
    semantic_rank: Option<usize>,
    semantic_index: Option<u32>,
    lexical_score: Option<f32>,
    semantic_score: Option<f32>,
    in_both_sources: bool,
}

/// `FusedHit` with a `CompactString` doc_id — the landed production type (~96 B).
struct FusedHitCompact {
    doc_id: CompactString,
    rrf_score: f64,
    lexical_rank: Option<usize>,
    semantic_rank: Option<usize>,
    semantic_index: Option<u32>,
    lexical_score: Option<f32>,
    semantic_score: Option<f32>,
    in_both_sources: bool,
}

/// Hypothetical packed `FusedHit` (~56 B): the `Option<usize>`/`Option<u32>` rank
/// fields become `u32` with a `u32::MAX` sentinel, `Option<f32>` scores become
/// `f32` with a `NaN` sentinel. Same doc_id (CompactString). Tests whether the
/// ~40 % struct-size cut speeds the `limit_all` materialize (smaller memcpy +
/// smaller Vec alloc) enough to justify the public-API break it would require.
struct FusedHitPacked {
    doc_id: CompactString,
    rrf_score: f64,
    lexical_rank: u32,
    semantic_rank: u32,
    semantic_index: u32,
    lexical_score: f32,
    semantic_score: f32,
    in_both_sources: bool,
}

fn winners<'a>(ids: &'a [String]) -> Vec<Winner<'a>> {
    ids.iter()
        .enumerate()
        .map(|(i, s)| Winner {
            doc_id: s.as_str(),
            rrf_score: 1.0 / (i as f64 + 60.0),
            lexical_rank: Some(i),
            semantic_rank: if i % 2 == 0 { Some(i) } else { None },
            semantic_index: Some(i as u32),
            lexical_score: Some(1.0 - i as f32 * 1e-4),
            semantic_score: if i % 2 == 0 { Some(0.9) } else { None },
            in_both_sources: i % 2 == 0,
        })
        .collect()
}

#[inline]
fn to_string(w: &Winner<'_>) -> FusedHitString {
    FusedHitString {
        doc_id: w.doc_id.into(),
        rrf_score: w.rrf_score,
        lexical_rank: w.lexical_rank,
        semantic_rank: w.semantic_rank,
        semantic_index: w.semantic_index,
        lexical_score: w.lexical_score,
        semantic_score: w.semantic_score,
        in_both_sources: w.in_both_sources,
    }
}

#[inline]
fn to_compact(w: &Winner<'_>) -> FusedHitCompact {
    FusedHitCompact {
        doc_id: w.doc_id.into(),
        rrf_score: w.rrf_score,
        lexical_rank: w.lexical_rank,
        semantic_rank: w.semantic_rank,
        semantic_index: w.semantic_index,
        lexical_score: w.lexical_score,
        semantic_score: w.semantic_score,
        in_both_sources: w.in_both_sources,
    }
}

#[inline]
fn to_packed(w: &Winner<'_>) -> FusedHitPacked {
    FusedHitPacked {
        doc_id: w.doc_id.into(),
        rrf_score: w.rrf_score,
        lexical_rank: w.lexical_rank.map_or(u32::MAX, |r| r as u32),
        semantic_rank: w.semantic_rank.map_or(u32::MAX, |r| r as u32),
        semantic_index: w.semantic_index.unwrap_or(u32::MAX),
        lexical_score: w.lexical_score.unwrap_or(f32::NAN),
        semantic_score: w.semantic_score.unwrap_or(f32::NAN),
        in_both_sources: w.in_both_sources,
    }
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("docid_materialize_ab");
    for n in [10_000usize, 100_000] {
        // Short, SSO-inline ids (≤24 B) — the common frankensearch doc_id shape.
        let ids: Vec<String> = (0..n).map(|i| format!("doc-{i:06}")).collect();
        let ws = winners(&ids);

        g.bench_with_input(BenchmarkId::new("string", n), &ws, |b, ws| {
            b.iter(|| {
                let out: Vec<FusedHitString> =
                    black_box(ws).iter().map(to_string).collect();
                black_box(out)
            });
        });
        g.bench_with_input(BenchmarkId::new("compact", n), &ws, |b, ws| {
            b.iter(|| {
                let out: Vec<FusedHitCompact> =
                    black_box(ws).iter().map(to_compact).collect();
                black_box(out)
            });
        });
        g.bench_with_input(BenchmarkId::new("packed", n), &ws, |b, ws| {
            b.iter(|| {
                let out: Vec<FusedHitPacked> =
                    black_box(ws).iter().map(to_packed).collect();
                black_box(out)
            });
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
