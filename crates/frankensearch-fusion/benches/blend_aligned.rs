//! `limit_all` quality-tier blend: materialized `quality_hits` (current) vs the
//! aligned, clone-free blend (`blend_two_tier_aligned`).
//!
//! In `sync_searcher::search_internal` the quality tier is a re-scored subset of
//! the fast tier (it shares every `doc_id`/`index`). The committed path built an
//! intermediate `Vec<VectorHit>` — cloning one `String` doc_id per quality hit —
//! purely to pass a `&[VectorHit]` to `blend_two_tier`, even though that slice's
//! doc_ids are only ever read as `&str` (by the blend and by the downstream
//! `quality_scores` borrow-map). `blend_two_tier_aligned` blends straight from
//! the aligned `Vec<Option<f32>>` quality scores, borrowing each doc_id from
//! `fast_hits` — eliding N short-String allocations. Output is bit-identical
//! (asserted once before timing).
//!
//! Each blend arm times the full differing region (quality-side prep + blend +
//! the downstream `quality_scores` borrow-map build), so the `quality_hits`
//! clone is charged honestly to the `current` arm. The `quality_score_prep`
//! group separately measures the async searcher's residual aligned-score
//! rebuild against reusing the owned vector in place.
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-fusion --bench blend_aligned
//! ```

use std::collections::HashMap;
use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_core::VectorHit;
use frankensearch_fusion::{blend_two_tier, blend_two_tier_aligned, blend_two_tier_aligned_unique};

const BLEND_FACTOR: f32 = 0.7;

/// Realistic fast tier: N candidates, distinct doc_ids, score descending.
fn fast_hits(n: usize) -> Vec<VectorHit> {
    (0..n)
        .map(|i| VectorHit {
            index: i as u32,
            score: 1.0 - (i as f32) / (n as f32),
            doc_id: format!("doc-{i:06}").into(),
        })
        .collect()
}

/// Aligned quality scores: ~90% present (the rest fell out of the quality index),
/// re-scored so the quality ranking differs from the fast ranking.
fn quality_scores(n: usize) -> Vec<Option<f32>> {
    (0..n)
        .map(|i| {
            if i % 10 == 7 {
                None
            } else {
                // A different monotone mapping so blend actually reorders.
                Some(0.5 + 0.5 * (((i * 2654435761) % n) as f32) / (n as f32))
            }
        })
        .collect()
}

#[derive(Clone, Copy)]
struct BenchCalibrator {
    scale: f64,
    bias: f64,
}

fn calibrate(config: Option<BenchCalibrator>, score: f32) -> f32 {
    let Some(config) = config else {
        return score;
    };
    let calibrated = (f64::from(score) * config.scale + config.bias) as f32;
    if calibrated.is_finite() {
        calibrated
    } else {
        0.0
    }
}

/// Pre-change async Phase-2 prep: borrow the owned input and rebuild every slot.
fn quality_prep_collect_copy(
    scores: Vec<Option<f32>>,
    config: Option<BenchCalibrator>,
) -> Vec<Option<f32>> {
    scores
        .iter()
        .map(|score| score.map(|value| calibrate(config, value)))
        .collect()
}

/// Candidate prep: retain the owned allocation and only walk it when configured.
fn quality_prep_in_place(
    mut scores: Vec<Option<f32>>,
    config: Option<BenchCalibrator>,
) -> Vec<Option<f32>> {
    if config.is_some() {
        for score in scores.iter_mut().flatten() {
            *score = calibrate(config, *score);
        }
    }
    scores
}

fn option_bits(scores: &[Option<f32>]) -> Vec<Option<u32>> {
    scores.iter().map(|score| score.map(f32::to_bits)).collect()
}

/// Current path: materialize the quality subset (cloning doc_ids), blend, then
/// build the downstream borrow-map from the materialized hits.
fn current(fast: &[VectorHit], qscores: &[Option<f32>]) -> (Vec<VectorHit>, usize) {
    let quality_hits = fast
        .iter()
        .zip(qscores.iter())
        .filter_map(|(f, s)| {
            s.map(|v| VectorHit {
                index: f.index,
                doc_id: f.doc_id.clone(),
                score: v,
            })
        })
        .collect::<Vec<_>>();
    let blended = blend_two_tier(fast, &quality_hits, BLEND_FACTOR);
    let qmap = quality_hits
        .iter()
        .map(|h| (h.doc_id.as_str(), h.score))
        .collect::<HashMap<&str, f32>>();
    (blended, qmap.len())
}

/// Aligned path: blend straight from the aligned scores (no clone), build the
/// borrow-map from `fast` + scores.
fn aligned(fast: &[VectorHit], qscores: &[Option<f32>]) -> (Vec<VectorHit>, usize) {
    let blended = blend_two_tier_aligned(fast, qscores, BLEND_FACTOR);
    let qmap = fast
        .iter()
        .zip(qscores.iter())
        .filter_map(|(h, s)| s.map(|v| (h.doc_id.as_str(), v)))
        .collect::<HashMap<&str, f32>>();
    (blended, qmap.len())
}

/// Unique aligned path: same production precondition as vector-index search
/// results, which already have at most one hit per doc_id.
fn aligned_unique(fast: &[VectorHit], qscores: &[Option<f32>]) -> (Vec<VectorHit>, usize) {
    let blended = blend_two_tier_aligned_unique(fast, qscores, BLEND_FACTOR);
    let qmap = fast
        .iter()
        .zip(qscores.iter())
        .filter_map(|(h, s)| s.map(|v| (h.doc_id.as_str(), v)))
        .collect::<HashMap<&str, f32>>();
    (blended, qmap.len())
}

fn bench_blend_aligned(c: &mut Criterion) {
    let mut g = c.benchmark_group("blend_aligned");
    for n in [10_000usize, 50_000, 100_000] {
        let fast = fast_hits(n);
        let qscores = quality_scores(n);

        // Bit-identity guard: same blended ranking (doc_id + score) and same
        // quality-map size, otherwise the A/B is meaningless.
        let (b_cur, m_cur) = current(&fast, &qscores);
        let (b_ali, m_ali) = aligned(&fast, &qscores);
        let (b_uni, m_uni) = aligned_unique(&fast, &qscores);
        assert_eq!(b_cur.len(), b_ali.len(), "blended length mismatch (n={n})");
        assert_eq!(m_cur, m_ali, "quality-map size mismatch (n={n})");
        for (x, y) in b_cur.iter().zip(b_ali.iter()) {
            assert_eq!(x.doc_id, y.doc_id, "blended doc_id order mismatch (n={n})");
            assert_eq!(
                x.score.to_bits(),
                y.score.to_bits(),
                "blended score mismatch (n={n})"
            );
            assert_eq!(x.index, y.index, "blended index mismatch (n={n})");
        }
        assert_eq!(b_ali.len(), b_uni.len(), "unique length mismatch (n={n})");
        assert_eq!(m_ali, m_uni, "unique quality-map size mismatch (n={n})");
        for (x, y) in b_ali.iter().zip(b_uni.iter()) {
            assert_eq!(x.doc_id, y.doc_id, "unique doc_id order mismatch (n={n})");
            assert_eq!(
                x.score.to_bits(),
                y.score.to_bits(),
                "unique score mismatch (n={n})"
            );
            assert_eq!(x.index, y.index, "unique index mismatch (n={n})");
        }

        g.bench_with_input(
            BenchmarkId::new("current", n),
            &(&fast, &qscores),
            |b, _| {
                b.iter(|| black_box(current(black_box(&fast), black_box(&qscores))));
            },
        );
        g.bench_with_input(
            BenchmarkId::new("aligned", n),
            &(&fast, &qscores),
            |b, _| {
                b.iter(|| black_box(aligned(black_box(&fast), black_box(&qscores))));
            },
        );
        g.bench_with_input(
            BenchmarkId::new("aligned_unique", n),
            &(&fast, &qscores),
            |b, _| {
                b.iter(|| black_box(aligned_unique(black_box(&fast), black_box(&qscores))));
            },
        );
    }
    g.finish();
}

fn bench_quality_score_prep(c: &mut Criterion) {
    let parity_input = vec![
        None,
        Some(-0.0),
        Some(0.25),
        Some(f32::NAN),
        Some(f32::INFINITY),
        Some(f32::NEG_INFINITY),
    ];
    for config in [
        None,
        Some(BenchCalibrator {
            scale: 0.75,
            bias: 0.125,
        }),
    ] {
        let current = quality_prep_collect_copy(parity_input.clone(), config);
        let candidate = quality_prep_in_place(parity_input.clone(), config);
        assert_eq!(
            option_bits(&current),
            option_bits(&candidate),
            "aligned calibration prep must preserve every score bit"
        );
    }

    let mut g = c.benchmark_group("quality_score_prep");
    for n in [32usize, 10_000, 100_000] {
        let scores = quality_scores(n);
        let config = black_box(None::<BenchCalibrator>);
        g.bench_with_input(BenchmarkId::new("collect_copy", n), &scores, |b, input| {
            b.iter_batched(
                || input.clone(),
                |owned| {
                    black_box(quality_prep_collect_copy(
                        black_box(owned),
                        black_box(config),
                    ))
                },
                BatchSize::LargeInput,
            );
        });
        g.bench_with_input(BenchmarkId::new("in_place", n), &scores, |b, input| {
            b.iter_batched(
                || input.clone(),
                |owned| black_box(quality_prep_in_place(black_box(owned), black_box(config))),
                BatchSize::LargeInput,
            );
        });
    }
    g.finish();
}

criterion_group!(benches, bench_blend_aligned, bench_quality_score_prep);
criterion_main!(benches);
