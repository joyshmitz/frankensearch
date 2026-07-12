//! Score normalization utilities for fusion and blending.
//!
//! Different retrieval sources emit scores on different scales (for example
//! unbounded BM25 versus bounded cosine similarity), so normalization is often
//! required before weighted blending. RRF itself is rank-based and does not
//! require normalization.

use serde::{Deserialize, Serialize};

const NON_FINITE_FALLBACK: f32 = 0.0;
const DEGENERATE_VALUE: f32 = 0.5;
const Z_SCORE_CLIP_SIGMAS: f32 = 3.0;
const NUMERIC_EPSILON: f32 = 1e-10;

/// Supported normalization strategies for score vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum NormalizationMethod {
    /// Min-max normalization into `[0, 1]`.
    #[default]
    MinMax,
    /// Z-score normalization mapped into `[0, 1]` after clipping to ±3σ.
    ZScore,
    /// Leave scores unchanged.
    None,
}

/// In-place min-max normalization.
///
/// Finite values are scaled into `[0, 1]`. Non-finite values (`NaN`/`±∞`) are
/// mapped to `0.0`. If all finite values are effectively identical, finite
/// values are mapped to `0.5`.
pub fn min_max_normalize(scores: &mut [f32]) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut saw_finite = false;

    for &value in scores.iter() {
        if value.is_finite() {
            min = min.min(value);
            max = max.max(value);
            saw_finite = true;
        }
    }

    if !saw_finite {
        scores.fill(NON_FINITE_FALLBACK);
        return;
    }

    let range = max - min;
    if range.abs() <= NUMERIC_EPSILON {
        for score in scores.iter_mut() {
            *score = if score.is_finite() {
                DEGENERATE_VALUE
            } else {
                NON_FINITE_FALLBACK
            };
        }
        return;
    }

    for score in scores.iter_mut() {
        if score.is_finite() {
            *score = ((*score - min) / range).clamp(0.0, 1.0);
        } else {
            *score = NON_FINITE_FALLBACK;
        }
    }
}

/// Query-commitment signal (NQC): the population coefficient of variation (σ/μ) of a
/// score slice — higher means a more peaked/"committed" retrieval.
///
/// Non-finite values are ignored. Returns `0.0` for empty input, no finite values, or a
/// non-positive mean (the intended input is the top-k BM25 scores of a query, which are
/// positive in practice). Accumulation is in `f64` for numerical stability.
///
/// This is the label-free, dense-free signal behind the opt-in *NQC dense down-weight*
/// (`docs/NEGATIVE_EVIDENCE.md`, 2026-07-12: dense is net-neutral/harmful on ~3/4 of
/// queries; down-weighting it on high-NQC queries is a small aggregate-significant nDCG
/// gain, pooled 95% CI `[+0.0008, +0.0035]`). This is the foundational statistic only —
/// the per-deployment cv→percentile CDF mapping and the fusion weight application are
/// separate pieces of that (not-yet-wired, default-off) feature.
#[must_use]
#[allow(clippy::cast_possible_truncation)] // f64 stats -> f32 score domain; precision loss is intentional
pub fn nqc_cv(scores: &[f32]) -> f32 {
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut count = 0_u32;
    for &value in scores {
        if value.is_finite() {
            let v = f64::from(value);
            sum += v;
            sum_sq += v * v;
            count += 1;
        }
    }
    if count == 0 {
        return 0.0;
    }
    let n = f64::from(count);
    let mean = sum / n;
    if mean <= f64::from(NUMERIC_EPSILON) {
        return 0.0;
    }
    let variance = (sum_sq / n - mean * mean).max(0.0);
    (variance.sqrt() / mean) as f32
}

/// Maps a query's NQC (see [`nqc_cv`]) to a per-query dense-tier weight multiplier for the
/// opt-in *NQC dense down-weight* (`docs/SEARCH_QUALITY_FINDINGS.md`, 2026-07-12).
///
/// Built from a rolling **sample** of observed NQC values (the query stream), so a raw `cv`
/// is mapped to its distribution **percentile** — a fixed `β·cv` does NOT transfer, because
/// the NQC scale is corpus-dependent (`docs/NEGATIVE_EVIDENCE.md`, 2026-07-12). Rebuild
/// periodically from a fresh sample. An empty sample yields a neutral weight of `1.0` (no
/// down-weight until the sketch has warmed up), so wiring it in is safe at startup.
///
/// A caller realizes the down-weight with **no fusion-kernel change**: multiply
/// `RrfConfig::semantic_weight` per query by [`NqcDenseWeight::dense_weight`].
#[derive(Debug, Clone, Default)]
pub struct NqcDenseWeight {
    /// Ascending sample of observed NQC (`nqc_cv`) values.
    sorted_cv: Vec<f32>,
}

impl NqcDenseWeight {
    /// An empty sketch (yields a neutral `1.0` weight until populated). `const` so it can
    /// initialize a searcher field in a `const fn` constructor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            sorted_cv: Vec::new(),
        }
    }

    /// Build from a sample of observed NQC values (non-finite samples are dropped).
    #[must_use]
    pub fn from_sample(sample: &[f32]) -> Self {
        let mut sorted_cv: Vec<f32> = sample.iter().copied().filter(|v| v.is_finite()).collect();
        sorted_cv.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Self { sorted_cv }
    }

    /// Number of retained samples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sorted_cv.len()
    }

    /// Whether the sketch has no samples (a neutral, no-down-weight state).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sorted_cv.is_empty()
    }

    /// Empirical CDF: the fraction of sampled NQC values `<= cv`, in `[0, 1]`. Returns
    /// `0.0` for an empty sample (→ neutral weight in [`dense_weight`](Self::dense_weight)).
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // sample-fraction; counts are small vs f32 mantissa
    pub fn percentile(&self, cv: f32) -> f32 {
        if self.sorted_cv.is_empty() {
            return 0.0;
        }
        let below_or_equal = self.sorted_cv.partition_point(|&v| v <= cv);
        below_or_equal as f32 / self.sorted_cv.len() as f32
    }

    /// Per-query dense-tier multiplier `clip(1 − β·CDF(cv), w_min, 1)`. Higher NQC (a more
    /// committed lexical retrieval, where the dense tier tends to add little or hurt) →
    /// a lower dense weight. `beta` ∈ ~`[0, 1]` (≈0.5 measured best); `w_min` floors it.
    /// `beta <= 0` (or an empty sketch) returns the neutral `1.0`.
    #[must_use]
    pub fn dense_weight(&self, cv: f32, beta: f32, w_min: f32) -> f32 {
        if beta <= 0.0 {
            return 1.0;
        }
        (1.0 - beta * self.percentile(cv)).clamp(w_min.clamp(0.0, 1.0), 1.0)
    }
}

/// In-place z-score normalization.
///
/// Finite values are standardized and then linearly mapped to `[0, 1]` by
/// clipping z-scores to ±3σ and applying `(z + 3) / 6`. Non-finite values are
/// mapped to `0.0`. If standard deviation is effectively zero, finite values
/// are mapped to `0.5`.
pub fn z_score_normalize(scores: &mut [f32]) {
    let mut count = 0.0_f32;
    let mut mean = 0.0_f32;
    let mut m2 = 0.0_f32;

    // Welford running variance for numerical stability.
    for &value in scores.iter() {
        if value.is_finite() {
            count += 1.0;
            let delta = value - mean;
            mean += delta / count;
            let delta2 = value - mean;
            m2 += delta * delta2;
        }
    }

    if count <= NUMERIC_EPSILON {
        scores.fill(NON_FINITE_FALLBACK);
        return;
    }

    let std_dev = (m2 / count).sqrt();
    if std_dev <= NUMERIC_EPSILON {
        for score in scores.iter_mut() {
            *score = if score.is_finite() {
                DEGENERATE_VALUE
            } else {
                NON_FINITE_FALLBACK
            };
        }
        return;
    }

    let denominator = 2.0 * Z_SCORE_CLIP_SIGMAS;
    for score in scores.iter_mut() {
        if score.is_finite() {
            let z = (*score - mean) / std_dev;
            let clipped = z.clamp(-Z_SCORE_CLIP_SIGMAS, Z_SCORE_CLIP_SIGMAS);
            *score = (clipped + Z_SCORE_CLIP_SIGMAS) / denominator;
        } else {
            *score = NON_FINITE_FALLBACK;
        }
    }
}

/// Applies a selected normalization method in-place.
pub fn normalize_in_place(scores: &mut [f32], method: NormalizationMethod) {
    match method {
        NormalizationMethod::MinMax => min_max_normalize(scores),
        NormalizationMethod::ZScore => z_score_normalize(scores),
        NormalizationMethod::None => {}
    }
}

/// Returns min-max normalized scores.
#[must_use]
pub fn normalize_scores(scores: &[f32]) -> Vec<f32> {
    normalize_scores_with_method(scores, NormalizationMethod::MinMax)
}

/// Returns normalized scores using the provided method.
#[must_use]
pub fn normalize_scores_with_method(scores: &[f32], method: NormalizationMethod) -> Vec<f32> {
    let mut normalized = scores.to_vec();
    normalize_in_place(&mut normalized, method);
    normalized
}

#[cfg(test)]
mod tests {
    use super::{
        NormalizationMethod, NqcDenseWeight, min_max_normalize, nqc_cv, normalize_in_place,
        normalize_scores, normalize_scores_with_method, z_score_normalize,
    };

    const EPSILON: f32 = 1e-6;

    fn assert_approx_slice(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (lhs, rhs)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (*lhs - *rhs).abs() <= EPSILON,
                "index {idx}: {lhs} != {rhs} within {EPSILON}"
            );
        }
    }

    #[test]
    fn min_max_normalize_spans_unit_interval() {
        let mut scores = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        min_max_normalize(&mut scores);
        assert_approx_slice(&scores, &[0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn min_max_normalize_identical_values_to_midpoint() {
        let mut scores = vec![3.0, 3.0, 3.0];
        min_max_normalize(&mut scores);
        assert_approx_slice(&scores, &[0.5, 0.5, 0.5]);
    }

    #[test]
    fn min_max_normalize_handles_non_finite_values() {
        let mut scores = vec![5.0, f32::NAN, f32::INFINITY, 10.0];
        min_max_normalize(&mut scores);
        assert_approx_slice(&scores, &[0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn nqc_cv_matches_population_coefficient_of_variation() {
        // mean=3, population var=2, std=sqrt(2) -> cv = sqrt(2)/3.
        let cv = nqc_cv(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((cv - (2.0_f32).sqrt() / 3.0).abs() <= 1e-5, "cv={cv}");
    }

    #[test]
    fn nqc_cv_zero_variance_is_zero() {
        assert_eq!(nqc_cv(&[5.0, 5.0, 5.0]), 0.0);
    }

    #[test]
    fn nqc_cv_empty_and_no_finite_is_zero() {
        assert_eq!(nqc_cv(&[]), 0.0);
        assert_eq!(nqc_cv(&[f32::NAN, f32::INFINITY]), 0.0);
    }

    #[test]
    fn nqc_cv_ignores_non_finite_values() {
        let with = nqc_cv(&[1.0, 2.0, 3.0, f32::NAN, f32::INFINITY, 5.0]);
        let without = nqc_cv(&[1.0, 2.0, 3.0, 5.0]);
        assert!((with - without).abs() <= 1e-6, "with={with} without={without}");
    }

    #[test]
    fn nqc_cv_more_peaked_scores_have_higher_cv() {
        // A committed retrieval (one dominant score) is more peaked than a flat one.
        let peaked = nqc_cv(&[10.0, 1.0, 1.0, 1.0]);
        let flat = nqc_cv(&[4.0, 3.0, 3.0, 4.0]);
        assert!(peaked > flat, "peaked={peaked} flat={flat}");
    }

    #[test]
    fn nqc_weight_from_sample_filters_non_finite_and_sorts() {
        let w = NqcDenseWeight::from_sample(&[0.3, f32::NAN, 0.1, f32::INFINITY, 0.2]);
        assert_eq!(w.len(), 3);
        // percentile is monotone in cv over the retained {0.1, 0.2, 0.3}
        assert!(w.percentile(0.05) <= w.percentile(0.15));
        assert!(w.percentile(0.15) <= w.percentile(0.25));
    }

    #[test]
    fn nqc_weight_percentile_spans_unit_interval() {
        let w = NqcDenseWeight::from_sample(&[0.1, 0.2, 0.3, 0.4]);
        assert!((w.percentile(0.0) - 0.0).abs() <= 1e-6, "below all -> 0");
        assert!((w.percentile(1.0) - 1.0).abs() <= 1e-6, "above all -> 1");
        assert!((w.percentile(0.2) - 0.5).abs() <= 1e-6, "<=0.2 is 2/4");
    }

    #[test]
    fn nqc_weight_empty_and_beta_zero_are_neutral() {
        let empty = NqcDenseWeight::default();
        assert_eq!(empty.dense_weight(1.23, 0.5, 0.0), 1.0, "empty -> neutral");
        let w = NqcDenseWeight::from_sample(&[0.1, 0.2, 0.3]);
        assert_eq!(w.dense_weight(0.3, 0.0, 0.0), 1.0, "beta=0 -> neutral");
    }

    #[test]
    fn nqc_weight_down_weights_high_commitment_and_clamps() {
        let w = NqcDenseWeight::from_sample(&[0.1, 0.2, 0.3, 0.4]);
        // High cv (percentile 1.0) with beta=0.5 -> 1 - 0.5*1.0 = 0.5.
        assert!((w.dense_weight(1.0, 0.5, 0.0) - 0.5).abs() <= 1e-6);
        // Low cv (percentile 0.0) -> neutral 1.0.
        assert!((w.dense_weight(0.0, 0.5, 0.0) - 1.0).abs() <= 1e-6);
        // Monotone: higher cv never increases the weight.
        assert!(w.dense_weight(0.4, 0.5, 0.0) <= w.dense_weight(0.15, 0.5, 0.0));
        // w_min floors it: beta=1.0 at percentile 1.0 would be 0.0, clamped up to 0.3.
        assert!((w.dense_weight(1.0, 1.0, 0.3) - 0.3).abs() <= 1e-6);
    }

    #[test]
    fn z_score_normalize_zero_variance_to_midpoint() {
        let mut scores = vec![42.0, 42.0, 42.0];
        z_score_normalize(&mut scores);
        assert_approx_slice(&scores, &[0.5, 0.5, 0.5]);
    }

    #[test]
    fn z_score_normalize_symmetric_distribution_centers_at_half() {
        let mut scores = vec![-1.0, 0.0, 1.0];
        z_score_normalize(&mut scores);

        let mean = scores.iter().copied().sum::<f32>() / 3.0;
        assert!((mean - 0.5).abs() <= EPSILON);
        assert!(scores[0] < scores[1] && scores[1] < scores[2]);
    }

    #[test]
    fn normalize_in_place_none_keeps_scores() {
        let original = vec![0.1, 0.4, 0.9];
        let mut scores = original.clone();

        normalize_in_place(&mut scores, NormalizationMethod::None);
        assert_approx_slice(&scores, &original);
    }

    #[test]
    fn normalize_scores_uses_min_max() {
        let scores = vec![10.0, 20.0, 30.0];
        let normalized = normalize_scores(&scores);
        assert_approx_slice(&normalized, &[0.0, 0.5, 1.0]);
    }

    #[test]
    fn normalize_scores_with_method_respects_selection() {
        let scores = vec![1.0, 2.0, 3.0];
        let min_max = normalize_scores_with_method(&scores, NormalizationMethod::MinMax);
        let z_score = normalize_scores_with_method(&scores, NormalizationMethod::ZScore);

        assert!(min_max[0] < min_max[1] && min_max[1] < min_max[2]);
        assert!(z_score[0] < z_score[1] && z_score[1] < z_score[2]);
    }

    #[test]
    fn min_max_normalize_all_negative_values() {
        let mut scores = vec![-5.0, -3.0, -1.0];
        min_max_normalize(&mut scores);
        assert_approx_slice(&scores, &[0.0, 0.5, 1.0]);
    }

    #[test]
    fn z_score_normalize_all_negative_values_preserves_order() {
        let mut scores = vec![-10.0, -5.0, -1.0];
        z_score_normalize(&mut scores);
        assert!(scores.iter().all(|s| (0.0..=1.0).contains(s)));
        assert!(scores[0] < scores[1] && scores[1] < scores[2]);
    }

    #[test]
    fn min_max_normalize_single_element() {
        let mut scores = vec![42.0];
        min_max_normalize(&mut scores);
        assert_approx_slice(&scores, &[0.5]);
    }

    #[test]
    fn z_score_normalize_single_element() {
        let mut scores = vec![42.0];
        z_score_normalize(&mut scores);
        assert_approx_slice(&scores, &[0.5]);
    }

    #[test]
    fn min_max_normalize_empty_is_noop() {
        let mut scores: Vec<f32> = vec![];
        min_max_normalize(&mut scores);
        assert!(scores.is_empty());
    }

    #[test]
    fn z_score_normalize_empty_is_noop() {
        let mut scores: Vec<f32> = vec![];
        z_score_normalize(&mut scores);
        assert!(scores.is_empty());
    }

    #[test]
    fn min_max_normalize_all_nan_maps_to_zero() {
        let mut scores = vec![f32::NAN, f32::NAN, f32::NAN];
        min_max_normalize(&mut scores);
        assert!(scores.iter().all(|s| *s == 0.0));
    }

    #[test]
    fn z_score_normalize_all_nan_maps_to_zero() {
        let mut scores = vec![f32::NAN, f32::NAN, f32::NAN];
        z_score_normalize(&mut scores);
        assert!(scores.iter().all(|s| *s == 0.0));
    }
}
