//! Latency cost of the shipped `RrfConfig` knobs — per-tier weights (`7ccda28`) and
//! the neutral hash tiebreak (`05472cd`). These were landed default-preserving; this
//! bench MEASURES (rather than assumes) that *using* the tuned config carries no latency
//! penalty over the default, and quantifies the hash tiebreak's cost under maximum ties.
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
//!   rch exec -- cargo bench -p frankensearch-fusion --bench rrf_config_cost_ab
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_core::types::{ScoreSource, ScoredResult, VectorHit};
use frankensearch_fusion::bench_support::paired_median_ratio;
use frankensearch_fusion::rrf::{RrfConfig, RrfTiebreak, rrf_fuse};

fn lexical(doc_id: String, score: f32) -> ScoredResult {
    ScoredResult {
        doc_id: doc_id.into(),
        score,
        source: ScoreSource::Lexical,
        index: None,
        fast_score: None,
        quality_score: None,
        lexical_score: Some(score),
        rerank_score: None,
        explanation: None,
        metadata: None,
    }
}

/// Realistic inputs: strictly descending scores, ~20% lexical/semantic overlap (few
/// exact `rrf_score` ties — the common case).
#[allow(clippy::cast_precision_loss)]
fn build_realistic(n: usize) -> (Vec<ScoredResult>, Vec<VectorHit>) {
    let semantic: Vec<VectorHit> = (0..n)
        .map(|i| VectorHit {
            index: u32::try_from(i).expect("benchmark indices fit in u32"),
            score: 1.0 - (i as f32) / (n as f32),
            doc_id: format!("doc-{i:06}").into(),
        })
        .collect();
    let lex: Vec<ScoredResult> = (0..n)
        .step_by(5)
        .map(|i| lexical(format!("doc-{i:06}"), (i % 97) as f32 * 0.01))
        .collect();
    (lex, semantic)
}

/// Adversarial inputs for the tiebreak: disjoint lexical/semantic doc sets with parallel
/// ranks, so the lexical-only doc at rank `i` ties with the semantic-only doc at rank `i`
/// (both contribute `1/(k+i+1)`, neither is in both sources) — `n` exact `rrf_score` ties,
/// the worst case for tiebreak work.
#[allow(clippy::cast_precision_loss)]
fn build_tied(n: usize) -> (Vec<ScoredResult>, Vec<VectorHit>) {
    let semantic: Vec<VectorHit> = (0..n)
        .map(|i| VectorHit {
            index: u32::try_from(i).expect("benchmark indices fit in u32"),
            score: 1.0 - (i as f32) / (n as f32),
            doc_id: format!("sem-{i:06}").into(),
        })
        .collect();
    let lex: Vec<ScoredResult> = (0..n)
        .map(|i| lexical(format!("lex-{i:06}"), 1.0 - (i as f32) / (n as f32)))
        .collect();
    (lex, semantic)
}

#[allow(
    clippy::significant_drop_tightening,
    reason = "Criterion benchmark groups intentionally span their complete arm sets"
)]
fn bench(c: &mut Criterion) {
    let n = 2000;
    let (lex, sem) = build_realistic(n);
    let limit = lex.len() + sem.len();

    let default = RrfConfig::default();
    let weighted = RrfConfig {
        semantic_weight: 1.3,
        ..RrfConfig::default()
    };
    let hash_tb = RrfConfig {
        tiebreak: RrfTiebreak::Hash,
        ..RrfConfig::default()
    };

    let mut g = c.benchmark_group("rrf_config_cost/realistic");
    g.bench_function("default", |b| {
        b.iter(|| {
            black_box(rrf_fuse(
                black_box(&lex),
                black_box(&sem),
                limit,
                0,
                &default,
            ))
        });
    });
    g.bench_function("tier_weighted", |b| {
        b.iter(|| {
            black_box(rrf_fuse(
                black_box(&lex),
                black_box(&sem),
                limit,
                0,
                &weighted,
            ))
        });
    });
    g.bench_function("hash_tiebreak", |b| {
        b.iter(|| {
            black_box(rrf_fuse(
                black_box(&lex),
                black_box(&sem),
                limit,
                0,
                &hash_tb,
            ))
        });
    });
    g.finish();

    // ── DECIDABILITY: alternating-round paired sampler + A/A null control ──
    //
    // The criterion arms above CANNOT decide these levers: criterion runs them as
    // separate benchmarks minutes apart, so worker drift between them is not
    // cancelled. The paired sampler runs both arms in ONE routine in alternating
    // rounds and takes the median per-round ratio; gate on the median against the
    // A/A null's observed spread, not on cv.
    let fuse_default = || {
        black_box(rrf_fuse(
            black_box(&lex),
            black_box(&sem),
            limit,
            0,
            &default,
        ));
    };
    let fuse_weighted = || {
        black_box(rrf_fuse(
            black_box(&lex),
            black_box(&sem),
            limit,
            0,
            &weighted,
        ));
    };
    let fuse_hash = || {
        black_box(rrf_fuse(
            black_box(&lex),
            black_box(&sem),
            limit,
            0,
            &hash_tb,
        ));
    };
    let null = paired_median_ratio(41, 8, fuse_default, fuse_default);
    let lever_weighted = paired_median_ratio(41, 8, fuse_default, fuse_weighted);
    let lever_hash = paired_median_ratio(41, 8, fuse_default, fuse_hash);
    eprintln!(
        "[null]  rrf_config_cost/realistic: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
        null.median, null.p5, null.p95, null.rounds
    );
    eprintln!(
        "[lever] rrf_config_cost/realistic: tier_weighted median {:.4} p5 {:.4} p95 {:.4} -> {}",
        lever_weighted.median,
        lever_weighted.p5,
        lever_weighted.p95,
        if lever_weighted.decidable_against(&null) {
            "DECIDABLE"
        } else {
            "INSIDE NULL FLOOR (not decidable)"
        }
    );
    eprintln!(
        "[lever] rrf_config_cost/realistic: hash_tiebreak median {:.4} p5 {:.4} p95 {:.4} -> {}",
        lever_hash.median,
        lever_hash.p5,
        lever_hash.p95,
        if lever_hash.decidable_against(&null) {
            "DECIDABLE"
        } else {
            "INSIDE NULL FLOOR (not decidable)"
        }
    );

    // Tiebreak-stressed: n exact ties. Measures the hash tiebreak's worst case vs legacy.
    let (tlex, tsem) = build_tied(n);
    let tlimit = tlex.len() + tsem.len();
    let mut gt = c.benchmark_group("rrf_config_cost/tie_heavy");
    gt.bench_function("lexical_tiebreak", |b| {
        b.iter(|| {
            black_box(rrf_fuse(
                black_box(&tlex),
                black_box(&tsem),
                tlimit,
                0,
                &default,
            ))
        });
    });
    gt.bench_function("hash_tiebreak", |b| {
        b.iter(|| {
            black_box(rrf_fuse(
                black_box(&tlex),
                black_box(&tsem),
                tlimit,
                0,
                &hash_tb,
            ))
        });
    });
    gt.finish();

    // ── DECIDABILITY: paired sampler + A/A null control (tie-heavy inputs) ──
    let fuse_tied_default = || {
        black_box(rrf_fuse(
            black_box(&tlex),
            black_box(&tsem),
            tlimit,
            0,
            &default,
        ));
    };
    let fuse_tied_hash = || {
        black_box(rrf_fuse(
            black_box(&tlex),
            black_box(&tsem),
            tlimit,
            0,
            &hash_tb,
        ));
    };
    let null = paired_median_ratio(41, 8, fuse_tied_default, fuse_tied_default);
    let lever = paired_median_ratio(41, 8, fuse_tied_default, fuse_tied_hash);
    eprintln!(
        "[null]  rrf_config_cost/tie_heavy: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
        null.median, null.p5, null.p95, null.rounds
    );
    eprintln!(
        "[lever] rrf_config_cost/tie_heavy: hash_tiebreak median {:.4} p5 {:.4} p95 {:.4} -> {}",
        lever.median,
        lever.p5,
        lever.p95,
        if lever.decidable_against(&null) {
            "DECIDABLE"
        } else {
            "INSIDE NULL FLOOR (not decidable)"
        }
    );
}

criterion_group!(benches, bench);
criterion_main!(benches);
