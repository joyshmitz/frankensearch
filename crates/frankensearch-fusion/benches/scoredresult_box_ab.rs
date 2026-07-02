//! Same-worker A/B for boxing `ScoredResult.explanation`.
//!
//! `ScoredResult.explanation: Option<HitExplanation>` stores `HitExplanation`
//! INLINE. `HitExplanation` is ~96 B (`Vec<ScoreComponent>` + `f64` +
//! `Option<RankMovement>{String}`), so every `ScoredResult` reserves that ~96 B
//! **even when explanation is None** (the common, `explain=false` case) — the enum
//! is sized for its largest variant. That bloats `ScoredResult` to ~176 B; at
//! `limit_all` (k=N) the searcher materializes N of them (`fused_hits_to_scored_results`)
//! and clones them per progressive phase, so the struct size is memcpy'd N (or 2N)
//! times per query.
//!
//! Boxing (`Option<Box<HitExplanation>>` → 8 B) nearly halves `ScoredResult` for the
//! common None case, and the `Box` is only allocated when `explain=true` (rare).
//! Bit-identical: `Box<HitExplanation>` derefs to `HitExplanation`. This measures
//! the materialization delta (N-vec build) inline vs boxed, both with explanation
//! = None (the limit_all common case).
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-fusion --bench scoredresult_box_ab
//! ```

use std::hint::black_box;
use std::mem::size_of;
use std::sync::Arc;

use compact_str::CompactString;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_core::explanation::HitExplanation;
use frankensearch_core::types::ScoreSource;

/// Mirrors the production `ScoredResult` (explanation INLINE).
struct SrInline {
    doc_id: CompactString,
    score: f32,
    source: ScoreSource,
    index: Option<u32>,
    fast_score: Option<f32>,
    quality_score: Option<f32>,
    lexical_score: Option<f32>,
    rerank_score: Option<f32>,
    explanation: Option<HitExplanation>,
    metadata: Option<Arc<serde_json::Value>>,
}

/// Candidate: explanation BOXED.
struct SrBoxed {
    doc_id: CompactString,
    score: f32,
    source: ScoreSource,
    index: Option<u32>,
    fast_score: Option<f32>,
    quality_score: Option<f32>,
    lexical_score: Option<f32>,
    rerank_score: Option<f32>,
    explanation: Option<Box<HitExplanation>>,
    metadata: Option<Arc<serde_json::Value>>,
}

/// Borrowed winner shape (Copy fields + &str doc_id), like FusedHitScratch.
#[derive(Clone, Copy)]
struct Winner<'a> {
    doc_id: &'a str,
    score: f32,
    index: u32,
    lexical_score: f32,
}

#[inline]
fn to_inline(w: &Winner<'_>) -> SrInline {
    SrInline {
        doc_id: w.doc_id.into(),
        score: w.score,
        source: ScoreSource::Hybrid,
        index: Some(w.index),
        fast_score: Some(w.score),
        quality_score: None,
        lexical_score: Some(w.lexical_score),
        rerank_score: None,
        explanation: None,
        metadata: None,
    }
}

#[inline]
fn to_boxed(w: &Winner<'_>) -> SrBoxed {
    SrBoxed {
        doc_id: w.doc_id.into(),
        score: w.score,
        source: ScoreSource::Hybrid,
        index: Some(w.index),
        fast_score: Some(w.score),
        quality_score: None,
        lexical_score: Some(w.lexical_score),
        rerank_score: None,
        explanation: None,
        metadata: None,
    }
}

fn bench(c: &mut Criterion) {
    eprintln!(
        "SIZE_OF SrInline={} SrBoxed={} (HitExplanation={})",
        size_of::<SrInline>(),
        size_of::<SrBoxed>(),
        size_of::<HitExplanation>()
    );
    let mut g = c.benchmark_group("scoredresult_box_ab");
    for n in [10_000usize, 100_000] {
        let ids: Vec<String> = (0..n).map(|i| format!("doc-{i:06}")).collect();
        let ws: Vec<Winner> = ids
            .iter()
            .enumerate()
            .map(|(i, s)| Winner {
                doc_id: s.as_str(),
                score: 1.0 - i as f32 * 1e-6,
                index: i as u32,
                lexical_score: 1.0,
            })
            .collect();

        g.bench_with_input(BenchmarkId::new("inline", n), &ws, |b, ws| {
            b.iter(|| {
                let out: Vec<SrInline> = black_box(ws).iter().map(to_inline).collect();
                black_box(out)
            });
        });
        g.bench_with_input(BenchmarkId::new("boxed", n), &ws, |b, ws| {
            b.iter(|| {
                let out: Vec<SrBoxed> = black_box(ws).iter().map(to_boxed).collect();
                black_box(out)
            });
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
