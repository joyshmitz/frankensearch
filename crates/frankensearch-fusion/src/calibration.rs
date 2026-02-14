//! Score calibration service for heterogeneous raw model scores.
//!
//! Converts raw scores (BM25, cosine similarity, reranker logits) into
//! calibrated probabilities `[0, 1]` before fusion, making blend and RRF
//! score combination mathematically meaningful.
//!
//! # Implementations
//!
//! | Calibrator          | Complexity  | Best For                        |
//! |---------------------|-------------|----------------------------------|
//! | [`Identity`]        | O(1)        | Passthrough (backward compat)    |
//! | [`TemperatureScaling`] | O(1)     | Reranker logits, cosine scores  |
//! | [`PlattScaling`]    | O(1)        | BM25, RRF scores                |
//! | [`IsotonicRegression`] | O(log n) | Any source with calibration data |
//!
//! # Monitoring
//!
//! Expected Calibration Error (ECE) is computed via [`compute_ece`] to detect
//! when a calibrator has gone stale. When ECE exceeds the configured threshold,
//! consumers should fall back to [`Identity`] and trigger recalibration.
//!
//! # References
//!
//! - Platt (1999) "Probabilistic outputs for SVMs"
//! - Zadrozny & Elkan (2002) "Transforming classifier scores"
//! - Guo et al. (2017) "On Calibration of Modern Neural Networks"

use serde::{Deserialize, Serialize};

// ─── ScoreCalibrator trait ───────────────────────────────────────────────────

/// Trait for calibrating raw scores into probabilities.
///
/// All implementations must:
/// - Map raw scores to `[0.0, 1.0]`.
/// - Be monotonic: higher raw score → higher calibrated score.
/// - Be deterministic: same input always produces same output.
/// - Handle non-finite inputs gracefully (NaN/Inf → 0.0).
pub trait ScoreCalibrator: Send + Sync {
    /// Calibrate a single raw score to a probability in `[0.0, 1.0]`.
    fn calibrate(&self, raw_score: f64) -> f64;

    /// Calibrate a batch of scores in-place.
    ///
    /// Default implementation calls [`calibrate`](Self::calibrate) in a loop.
    fn calibrate_batch(&self, scores: &mut [f64]) {
        for score in scores.iter_mut() {
            *score = self.calibrate(*score);
        }
    }

    /// Human-readable name of this calibrator.
    fn name(&self) -> &'static str;
}

// ─── Identity ────────────────────────────────────────────────────────────────

/// Passthrough calibrator that returns scores unchanged.
///
/// Used as the default when no calibration data is available.
/// Clamps output to `[0.0, 1.0]` and maps non-finite values to `0.0`.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Identity;

impl ScoreCalibrator for Identity {
    fn calibrate(&self, raw_score: f64) -> f64 {
        if !raw_score.is_finite() {
            return 0.0;
        }
        raw_score.clamp(0.0, 1.0)
    }

    fn name(&self) -> &'static str {
        "identity"
    }
}

// ─── Temperature Scaling ─────────────────────────────────────────────────────

/// Single-parameter temperature scaling: `calibrated = sigmoid(score / T)`.
///
/// Best for reranker logits and cosine similarity scores.
/// Temperature `T` is learned offline via NLL minimization on a validation set.
/// `T = 1.0` is equivalent to a standard sigmoid.
///
/// # Parameters
///
/// - `temperature`: Positive scaling factor. Higher values produce softer
///   (closer to 0.5) probabilities; lower values produce sharper outputs.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TemperatureScaling {
    /// Temperature parameter (must be > 0).
    pub temperature: f64,
}

impl TemperatureScaling {
    /// Create a new temperature scaling calibrator.
    ///
    /// # Panics
    ///
    /// Panics if `temperature <= 0.0` or is non-finite.
    #[must_use]
    pub fn new(temperature: f64) -> Self {
        assert!(
            temperature > 0.0 && temperature.is_finite(),
            "temperature must be positive and finite, got {temperature}"
        );
        Self { temperature }
    }
}

impl ScoreCalibrator for TemperatureScaling {
    fn calibrate(&self, raw_score: f64) -> f64 {
        if !raw_score.is_finite() {
            return 0.0;
        }
        sigmoid(raw_score / self.temperature)
    }

    fn name(&self) -> &'static str {
        "temperature_scaling"
    }
}

// ─── Platt Scaling ───────────────────────────────────────────────────────────

/// Logistic regression calibration: `calibrated = sigmoid(a * score + b)`.
///
/// Parameters `(a, b)` are fit offline via maximum likelihood on held-out data.
/// Best for BM25 and RRF scores where the score-to-probability relationship
/// is approximately logistic but with an offset.
///
/// # Parameters
///
/// - `a`: Slope (typically negative for scores where higher = better).
/// - `b`: Intercept.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PlattScaling {
    /// Slope parameter.
    pub a: f64,
    /// Intercept parameter.
    pub b: f64,
}

impl PlattScaling {
    /// Create a new Platt scaling calibrator.
    ///
    /// # Panics
    ///
    /// Panics if either parameter is non-finite.
    #[must_use]
    pub fn new(a: f64, b: f64) -> Self {
        assert!(
            a.is_finite() && b.is_finite(),
            "Platt parameters must be finite, got a={a}, b={b}"
        );
        Self { a, b }
    }
}

impl ScoreCalibrator for PlattScaling {
    fn calibrate(&self, raw_score: f64) -> f64 {
        if !raw_score.is_finite() {
            return 0.0;
        }
        sigmoid(self.a.mul_add(raw_score, self.b))
    }

    fn name(&self) -> &'static str {
        "platt_scaling"
    }
}

// ─── Isotonic Regression ─────────────────────────────────────────────────────

/// Non-parametric monotonic calibration via piecewise-constant mapping.
///
/// Learned offline by fitting a monotone non-decreasing step function to
/// `(raw_score, relevance_label)` pairs. Guarantees monotonicity by
/// construction: higher raw scores always map to equal or higher calibrated
/// probabilities.
///
/// At inference time, uses binary search over breakpoints for O(log n)
/// lookup per score.
///
/// # Parameters
///
/// - `breakpoints`: Sorted (ascending) raw score thresholds.
/// - `values`: Calibrated probability for each interval. `values[i]` is the
///   output for scores in `[breakpoints[i], breakpoints[i+1])`.
///   `values.len() == breakpoints.len()`.
///
/// Scores below `breakpoints[0]` return `values[0]`.
/// Scores at or above `breakpoints[last]` return `values[last]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsotonicRegression {
    /// Sorted raw score breakpoints (ascending).
    pub breakpoints: Vec<f64>,
    /// Calibrated probability values (non-decreasing, `[0, 1]`).
    pub values: Vec<f64>,
}

impl IsotonicRegression {
    /// Create a new isotonic regression calibrator.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `breakpoints` and `values` have different lengths.
    /// - `breakpoints` is empty.
    /// - `breakpoints` is not sorted ascending.
    /// - `values` is not monotonically non-decreasing.
    /// - Any value is outside `[0.0, 1.0]`.
    #[must_use]
    pub fn new(breakpoints: Vec<f64>, values: Vec<f64>) -> Self {
        assert!(
            !breakpoints.is_empty(),
            "isotonic regression requires at least one breakpoint"
        );
        assert_eq!(
            breakpoints.len(),
            values.len(),
            "breakpoints and values must have same length"
        );
        // Verify sorted breakpoints.
        for w in breakpoints.windows(2) {
            assert!(
                w[0] <= w[1],
                "breakpoints must be sorted ascending: {} > {}",
                w[0],
                w[1]
            );
        }
        // Verify non-decreasing values in [0, 1].
        for (i, &v) in values.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&v),
                "values[{i}] = {v} is outside [0, 1]"
            );
            if i > 0 {
                assert!(
                    v >= values[i - 1],
                    "values must be non-decreasing: values[{}] = {} > values[{i}] = {}",
                    i - 1,
                    values[i - 1],
                    v
                );
            }
        }
        Self {
            breakpoints,
            values,
        }
    }

    /// Fit an isotonic regression from raw scores and binary relevance labels.
    ///
    /// Uses the pool-adjacent-violators algorithm (PAVA):
    /// 1. Sort pairs by raw score ascending.
    /// 2. Merge adjacent pairs that violate monotonicity.
    /// 3. Output the resulting step function.
    ///
    /// # Panics
    ///
    /// Panics if `scores` and `labels` have different lengths or are empty.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn fit(scores: &[f64], labels: &[f64]) -> Self {
        assert_eq!(scores.len(), labels.len(), "scores and labels must match");
        assert!(!scores.is_empty(), "need at least one data point");

        // Sort by score ascending.
        let mut pairs: Vec<(f64, f64)> =
            scores.iter().copied().zip(labels.iter().copied()).collect();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

        // Pool Adjacent Violators Algorithm (PAVA).
        // Each block is (sum_of_labels, count, representative_score).
        let mut blocks: Vec<(f64, usize, f64)> = pairs
            .iter()
            .map(|&(score, label)| (label, 1, score))
            .collect();

        let mut i = 0;
        while i < blocks.len().saturating_sub(1) {
            let mean_i = blocks[i].0 / {
                #[allow(clippy::cast_precision_loss)]
                {
                    blocks[i].1 as f64
                }
            };
            let mean_next = blocks[i + 1].0 / blocks[i + 1].1 as f64;
            if mean_i > mean_next {
                // Merge blocks[i] and blocks[i+1].
                blocks[i].0 += blocks[i + 1].0;
                blocks[i].1 += blocks[i + 1].1;
                // Keep the midpoint score as representative.
                blocks[i].2 = f64::midpoint(blocks[i].2, blocks[i + 1].2);
                blocks.remove(i + 1);
                // Step back to check if new merge created another violation.
                i = i.saturating_sub(1);
            } else {
                i += 1;
            }
        }

        let breakpoints: Vec<f64> = blocks.iter().map(|b| b.2).collect();
        let values: Vec<f64> = blocks
            .iter()
            .map(|b| (b.0 / b.1 as f64).clamp(0.0, 1.0))
            .collect();

        Self {
            breakpoints,
            values,
        }
    }
}

impl ScoreCalibrator for IsotonicRegression {
    fn calibrate(&self, raw_score: f64) -> f64 {
        if !raw_score.is_finite() {
            return 0.0;
        }
        // Binary search for the rightmost breakpoint <= raw_score.
        match self
            .breakpoints
            .binary_search_by(|bp| bp.total_cmp(&raw_score))
        {
            Ok(idx) => self.values[idx],
            Err(0) => self.values[0],
            Err(idx) if idx >= self.breakpoints.len() => *self.values.last().unwrap_or(&0.0),
            Err(idx) => self.values[idx - 1],
        }
    }

    fn name(&self) -> &'static str {
        "isotonic_regression"
    }
}

// ─── ECE Computation ─────────────────────────────────────────────────────────

/// Expected Calibration Error (ECE) computation.
///
/// Partitions calibrated probabilities into `num_bins` equal-width bins
/// and computes the weighted average of `|avg_confidence - accuracy|`
/// per bin.
///
/// Lower is better. Typical threshold for "well-calibrated": ECE < 0.05.
///
/// # Parameters
///
/// - `predictions`: Calibrated probabilities in `[0, 1]`.
/// - `labels`: Binary relevance labels (`0.0` or `1.0`).
/// - `num_bins`: Number of equal-width bins (typically 10 or 15).
///
/// # Returns
///
/// ECE value in `[0, 1]`. Returns `0.0` if inputs are empty.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn compute_ece(predictions: &[f64], labels: &[f64], num_bins: usize) -> f64 {
    if predictions.is_empty() || labels.is_empty() || num_bins == 0 {
        return 0.0;
    }
    let n = predictions.len().min(labels.len());

    let mut bin_sums = vec![0.0_f64; num_bins];
    let mut bin_correct = vec![0.0_f64; num_bins];
    let mut bin_counts = vec![0_usize; num_bins];

    for i in 0..n {
        let p = predictions[i].clamp(0.0, 1.0);
        // Determine bin index: [0, 1/B), [1/B, 2/B), ..., [(B-1)/B, 1].
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let bin = ((p * num_bins as f64) as usize).min(num_bins - 1);
        bin_sums[bin] += p;
        bin_correct[bin] += labels[i];
        bin_counts[bin] += 1;
    }

    let mut ece = 0.0_f64;
    #[allow(clippy::cast_precision_loss)]
    for bin in 0..num_bins {
        if bin_counts[bin] > 0 {
            let avg_confidence = bin_sums[bin] / bin_counts[bin] as f64;
            let accuracy = bin_correct[bin] / bin_counts[bin] as f64;
            ece += (bin_counts[bin] as f64 / n as f64) * (avg_confidence - accuracy).abs();
        }
    }
    ece
}

/// Brier score: mean squared error of calibrated probabilities vs labels.
///
/// Lower is better. Range: `[0, 1]`. A perfectly calibrated model on balanced
/// data has Brier score ~0.25 (from the irreducible uncertainty).
///
/// Returns `0.0` if inputs are empty.
#[must_use]
pub fn compute_brier_score(predictions: &[f64], labels: &[f64]) -> f64 {
    if predictions.is_empty() || labels.is_empty() {
        return 0.0;
    }
    let n = predictions.len().min(labels.len());
    let sum_sq: f64 = predictions
        .iter()
        .zip(labels.iter())
        .take(n)
        .map(|(&p, &l)| (p - l).powi(2))
        .sum();
    #[allow(clippy::cast_precision_loss)]
    let brier = sum_sq / n as f64;
    brier
}

// ─── Serializable Calibrator ─────────────────────────────────────────────────

/// A serializable calibrator container for persistence.
///
/// Wraps the four calibrator types in an enum so they can be serialized
/// to JSON for storage in `data_dir/calibration/<scorer_id>.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CalibratorConfig {
    /// Passthrough calibration.
    Identity(Identity),
    /// Temperature scaling calibration.
    TemperatureScaling(TemperatureScaling),
    /// Platt (logistic) scaling calibration.
    PlattScaling(PlattScaling),
    /// Isotonic regression calibration.
    IsotonicRegression(IsotonicRegression),
}

impl CalibratorConfig {
    /// Get a reference to the inner calibrator as a trait object.
    #[must_use]
    pub fn as_calibrator(&self) -> &dyn ScoreCalibrator {
        match self {
            Self::Identity(c) => c,
            Self::TemperatureScaling(c) => c,
            Self::PlattScaling(c) => c,
            Self::IsotonicRegression(c) => c,
        }
    }

    /// Calibrate a single score using the contained calibrator.
    #[must_use]
    pub fn calibrate(&self, raw_score: f64) -> f64 {
        self.as_calibrator().calibrate(raw_score)
    }

    /// Name of the contained calibrator.
    #[must_use]
    pub fn name(&self) -> &str {
        self.as_calibrator().name()
    }
}

/// Summary diagnostics for one batch-calibration pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalibrationSummary {
    /// Number of `(score, label)` pairs processed.
    pub count: usize,
    /// ECE before calibration.
    pub ece_before: f64,
    /// ECE after calibration.
    pub ece_after: f64,
    /// Brier score before calibration.
    pub brier_before: f64,
    /// Brier score after calibration.
    pub brier_after: f64,
}

/// Calibrate a score batch with labels and emit one structured tracing event.
///
/// This helper is intended for integration points where callers want a single
/// call that:
/// 1. calibrates a batch,
/// 2. computes before/after diagnostics (ECE + Brier), and
/// 3. emits a structured event for observability.
///
/// The input is truncated to `min(raw_scores.len(), labels.len())`.
/// Non-finite values are sanitized to `0.0`, and all probabilities are clamped
/// to `[0.0, 1.0]` for metric computation.
#[must_use]
pub fn calibrate_scores_with_labels(
    calibrator: &dyn ScoreCalibrator,
    raw_scores: &[f64],
    labels: &[f64],
    num_bins: usize,
) -> (Vec<f64>, CalibrationSummary) {
    let count = raw_scores.len().min(labels.len());
    if count == 0 {
        return (
            Vec::new(),
            CalibrationSummary {
                count: 0,
                ece_before: 0.0,
                ece_after: 0.0,
                brier_before: 0.0,
                brier_after: 0.0,
            },
        );
    }

    let bounded_raw: Vec<f64> = raw_scores
        .iter()
        .take(count)
        .copied()
        .map(bounded_probability)
        .collect();
    let bounded_labels: Vec<f64> = labels
        .iter()
        .take(count)
        .copied()
        .map(bounded_probability)
        .collect();

    let ece_before = compute_ece(&bounded_raw, &bounded_labels, num_bins);
    let brier_before = compute_brier_score(&bounded_raw, &bounded_labels);

    let mut calibrated: Vec<f64> = raw_scores.iter().take(count).copied().collect();
    calibrator.calibrate_batch(&mut calibrated);
    for score in &mut calibrated {
        *score = bounded_probability(*score);
    }

    let ece_after = compute_ece(&calibrated, &bounded_labels, num_bins);
    let brier_after = compute_brier_score(&calibrated, &bounded_labels);

    tracing::info!(
        event = "scores_calibrated",
        calibrator = calibrator.name(),
        count,
        ece_before,
        ece_after,
        brier_before,
        brier_after,
        "scores calibrated"
    );

    (
        calibrated,
        CalibrationSummary {
            count,
            ece_before,
            ece_after,
            brier_before,
            brier_after,
        },
    )
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Standard sigmoid function: `1 / (1 + exp(-x))`.
///
/// Numerically stable for all finite inputs. Returns `0.0` for `-inf`
/// and `1.0` for `+inf`.
#[must_use]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Sanitize a raw scalar into a valid probability.
#[must_use]
const fn bounded_probability(value: f64) -> f64 {
    if !value.is_finite() {
        return 0.0;
    }
    value.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_test::traced_test;

    // ─── Sigmoid ─────────────────────────────────────────────────────

    #[test]
    fn sigmoid_zero_is_half() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn sigmoid_large_positive_near_one() {
        assert!((sigmoid(100.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn sigmoid_large_negative_near_zero() {
        assert!(sigmoid(-100.0) < 1e-10);
    }

    #[test]
    fn sigmoid_symmetry() {
        let x = 2.5;
        assert!((sigmoid(x) + sigmoid(-x) - 1.0).abs() < 1e-10);
    }

    // ─── Identity ────────────────────────────────────────────────────

    #[test]
    fn identity_passthrough() {
        let cal = Identity;
        assert!((cal.calibrate(0.5) - 0.5).abs() < 1e-10);
        assert!((cal.calibrate(0.0) - 0.0).abs() < 1e-10);
        assert!((cal.calibrate(1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn identity_clamps_out_of_range() {
        let cal = Identity;
        assert!((cal.calibrate(1.5) - 1.0).abs() < 1e-10);
        assert!((cal.calibrate(-0.5) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn identity_handles_nan() {
        let cal = Identity;
        assert!((cal.calibrate(f64::NAN) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn identity_handles_infinity() {
        let cal = Identity;
        assert!((cal.calibrate(f64::INFINITY) - 0.0).abs() < 1e-10);
        assert!((cal.calibrate(f64::NEG_INFINITY) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn identity_name() {
        assert_eq!(Identity.name(), "identity");
    }

    // ─── Temperature Scaling ─────────────────────────────────────────

    #[test]
    fn temperature_t1_is_sigmoid() {
        let cal = TemperatureScaling::new(1.0);
        let x = 2.0;
        assert!((cal.calibrate(x) - sigmoid(x)).abs() < 1e-10);
    }

    #[test]
    fn temperature_higher_t_softer_output() {
        let soft = TemperatureScaling::new(5.0);
        let sharp = TemperatureScaling::new(0.5);
        // At x=2.0, higher T should give output closer to 0.5.
        let soft_out = soft.calibrate(2.0);
        let sharp_out = sharp.calibrate(2.0);
        assert!((soft_out - 0.5).abs() < (sharp_out - 0.5).abs());
    }

    #[test]
    fn temperature_zero_input_is_half() {
        let cal = TemperatureScaling::new(3.0);
        assert!((cal.calibrate(0.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn temperature_handles_nan() {
        let cal = TemperatureScaling::new(1.0);
        assert!((cal.calibrate(f64::NAN) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn temperature_monotonicity() {
        let cal = TemperatureScaling::new(2.0);
        let scores: Vec<f64> = (-50..=50).map(|i| f64::from(i) * 0.1).collect();
        for w in scores.windows(2) {
            assert!(
                cal.calibrate(w[1]) >= cal.calibrate(w[0]),
                "monotonicity violated: cal({}) = {} > cal({}) = {}",
                w[0],
                cal.calibrate(w[0]),
                w[1],
                cal.calibrate(w[1])
            );
        }
    }

    #[test]
    fn temperature_name() {
        assert_eq!(TemperatureScaling::new(1.0).name(), "temperature_scaling");
    }

    #[test]
    #[should_panic(expected = "temperature must be positive")]
    fn temperature_rejects_zero() {
        let _ = TemperatureScaling::new(0.0);
    }

    #[test]
    #[should_panic(expected = "temperature must be positive")]
    fn temperature_rejects_negative() {
        let _ = TemperatureScaling::new(-1.0);
    }

    // ─── Platt Scaling ───────────────────────────────────────────────

    #[test]
    fn platt_zero_params_is_sigmoid() {
        // sigmoid(0*x + 0) = sigmoid(0) = 0.5 for all x.
        let cal = PlattScaling::new(0.0, 0.0);
        assert!((cal.calibrate(999.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn platt_identity_slope() {
        // a=1, b=0: equivalent to plain sigmoid.
        let cal = PlattScaling::new(1.0, 0.0);
        let x = 3.0;
        assert!((cal.calibrate(x) - sigmoid(x)).abs() < 1e-10);
    }

    #[test]
    fn platt_negative_slope() {
        // For BM25 where higher raw score = more relevant,
        // negative slope + positive intercept can shift the mapping.
        let cal = PlattScaling::new(-1.0, 5.0);
        // sigmoid(-1*10 + 5) = sigmoid(-5) ≈ 0.0067
        assert!(cal.calibrate(10.0) < 0.01);
        // sigmoid(-1*0 + 5) = sigmoid(5) ≈ 0.993
        assert!(cal.calibrate(0.0) > 0.99);
    }

    #[test]
    fn platt_handles_nan() {
        let cal = PlattScaling::new(1.0, 0.0);
        assert!((cal.calibrate(f64::NAN) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn platt_monotonicity_positive_slope() {
        let cal = PlattScaling::new(2.0, -1.0);
        let scores: Vec<f64> = (-50..=50).map(|i| f64::from(i) * 0.1).collect();
        for w in scores.windows(2) {
            assert!(cal.calibrate(w[1]) >= cal.calibrate(w[0]));
        }
    }

    #[test]
    fn platt_name() {
        assert_eq!(PlattScaling::new(1.0, 0.0).name(), "platt_scaling");
    }

    #[test]
    #[should_panic(expected = "Platt parameters must be finite")]
    fn platt_rejects_nan_params() {
        let _ = PlattScaling::new(f64::NAN, 0.0);
    }

    // ─── Isotonic Regression ─────────────────────────────────────────

    #[test]
    fn isotonic_simple_step_function() {
        let cal = IsotonicRegression::new(vec![0.0, 0.5, 1.0], vec![0.1, 0.5, 0.9]);
        assert!((cal.calibrate(0.0) - 0.1).abs() < 1e-10);
        assert!((cal.calibrate(0.5) - 0.5).abs() < 1e-10);
        assert!((cal.calibrate(1.0) - 0.9).abs() < 1e-10);
    }

    #[test]
    fn isotonic_interpolation_between_breakpoints() {
        let cal = IsotonicRegression::new(vec![0.0, 1.0], vec![0.2, 0.8]);
        // Score 0.5 is between breakpoints 0.0 and 1.0.
        // It falls in the interval [0.0, 1.0), so value = 0.2.
        assert!((cal.calibrate(0.5) - 0.2).abs() < 1e-10);
    }

    #[test]
    fn isotonic_below_first_breakpoint() {
        let cal = IsotonicRegression::new(vec![1.0, 2.0], vec![0.3, 0.7]);
        // Score below all breakpoints returns first value.
        assert!((cal.calibrate(0.0) - 0.3).abs() < 1e-10);
    }

    #[test]
    fn isotonic_above_last_breakpoint() {
        let cal = IsotonicRegression::new(vec![0.0, 1.0], vec![0.2, 0.8]);
        // Score above all breakpoints returns last value.
        assert!((cal.calibrate(5.0) - 0.8).abs() < 1e-10);
    }

    #[test]
    fn isotonic_handles_nan() {
        let cal = IsotonicRegression::new(vec![0.0, 1.0], vec![0.2, 0.8]);
        assert!((cal.calibrate(f64::NAN) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn isotonic_single_breakpoint() {
        let cal = IsotonicRegression::new(vec![0.5], vec![0.7]);
        assert!((cal.calibrate(0.0) - 0.7).abs() < 1e-10);
        assert!((cal.calibrate(0.5) - 0.7).abs() < 1e-10);
        assert!((cal.calibrate(1.0) - 0.7).abs() < 1e-10);
    }

    #[test]
    fn isotonic_monotonicity_guarantee() {
        let cal = IsotonicRegression::new(
            vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            vec![0.05, 0.15, 0.35, 0.55, 0.75, 0.95],
        );
        let scores: Vec<f64> = (0..=100).map(|i| f64::from(i) * 0.01).collect();
        for w in scores.windows(2) {
            assert!(
                cal.calibrate(w[1]) >= cal.calibrate(w[0]),
                "isotonic monotonicity violated at {} -> {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn isotonic_name() {
        let cal = IsotonicRegression::new(vec![0.0], vec![0.5]);
        assert_eq!(cal.name(), "isotonic_regression");
    }

    #[test]
    #[should_panic(expected = "at least one breakpoint")]
    fn isotonic_rejects_empty() {
        let _ = IsotonicRegression::new(vec![], vec![]);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn isotonic_rejects_length_mismatch() {
        let _ = IsotonicRegression::new(vec![0.0, 1.0], vec![0.5]);
    }

    #[test]
    #[should_panic(expected = "sorted ascending")]
    fn isotonic_rejects_unsorted_breakpoints() {
        let _ = IsotonicRegression::new(vec![1.0, 0.0], vec![0.5, 0.9]);
    }

    #[test]
    #[should_panic(expected = "non-decreasing")]
    fn isotonic_rejects_non_monotonic_values() {
        let _ = IsotonicRegression::new(vec![0.0, 1.0], vec![0.9, 0.1]);
    }

    #[test]
    #[should_panic(expected = "outside [0, 1]")]
    fn isotonic_rejects_out_of_range_values() {
        let _ = IsotonicRegression::new(vec![0.0], vec![1.5]);
    }

    // ─── Isotonic Regression Fit ─────────────────────────────────────

    #[test]
    fn isotonic_fit_perfectly_calibrated() {
        // Data already monotonic: no merging needed.
        let scores = vec![0.0, 0.5, 1.0];
        let labels = vec![0.0, 0.5, 1.0];
        let cal = IsotonicRegression::fit(&scores, &labels);
        // Should have 3 breakpoints with matching values.
        assert_eq!(cal.breakpoints.len(), 3);
        assert!((cal.calibrate(0.0) - 0.0).abs() < 1e-10);
        assert!((cal.calibrate(1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn isotonic_fit_with_violations() {
        // Scores: 1, 2, 3 with labels: 0.8, 0.2, 0.6
        // The pair (2, 0.2) violates monotonicity with (1, 0.8).
        let scores = vec![1.0, 2.0, 3.0];
        let labels = vec![0.8, 0.2, 0.6];
        let cal = IsotonicRegression::fit(&scores, &labels);
        // PAVA should merge violating pairs.
        // All outputs should be in [0, 1].
        for &v in &cal.values {
            assert!((0.0..=1.0).contains(&v));
        }
        // Monotonicity must hold.
        for w in cal.values.windows(2) {
            assert!(w[1] >= w[0], "fit result not monotonic");
        }
    }

    #[test]
    fn isotonic_fit_all_same_label() {
        let scores = vec![0.0, 0.5, 1.0];
        let labels = vec![0.7, 0.7, 0.7];
        let cal = IsotonicRegression::fit(&scores, &labels);
        // Should merge into one or more blocks all with value 0.7.
        assert!((cal.calibrate(0.5) - 0.7).abs() < 1e-10);
    }

    #[test]
    fn isotonic_fit_single_point() {
        let cal = IsotonicRegression::fit(&[3.0], &[0.5]);
        assert_eq!(cal.breakpoints.len(), 1);
        assert!((cal.calibrate(3.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn isotonic_fit_unsorted_input() {
        // Input doesn't need to be sorted — fit() sorts internally.
        let scores = vec![3.0, 1.0, 2.0];
        let labels = vec![0.9, 0.1, 0.5];
        let cal = IsotonicRegression::fit(&scores, &labels);
        // Should produce monotonic output.
        for w in cal.values.windows(2) {
            assert!(w[1] >= w[0]);
        }
    }

    // ─── Batch Calibration ───────────────────────────────────────────

    #[test]
    fn batch_calibration_matches_sequential() {
        let cal = PlattScaling::new(1.5, -0.3);
        let raw_scores = vec![0.0, 0.5, 1.0, 2.0, -1.0];
        let expected: Vec<f64> = raw_scores.iter().map(|&s| cal.calibrate(s)).collect();
        let mut batch = raw_scores;
        cal.calibrate_batch(&mut batch);
        for (e, b) in expected.iter().zip(batch.iter()) {
            assert!((e - b).abs() < 1e-10);
        }
    }

    // ─── ECE Computation ─────────────────────────────────────────────

    #[test]
    fn ece_well_calibrated() {
        // Predictions that match the true probabilities well.
        // Each prediction roughly equals the fraction of positive labels
        // in its neighborhood.
        let predictions = vec![0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.7, 0.7, 0.9, 0.9];
        let labels = vec![0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let ece = compute_ece(&predictions, &labels, 5);
        // With 5 bins, this should have reasonably low ECE.
        assert!(ece < 0.3, "ECE = {ece}, expected < 0.3 for well-calibrated");
    }

    #[test]
    fn ece_completely_miscalibrated() {
        // All predictions are 0.9 but all labels are 0.0.
        let predictions = vec![0.9; 10];
        let labels = vec![0.0; 10];
        let ece = compute_ece(&predictions, &labels, 10);
        // ECE should be high (~0.9).
        assert!(ece > 0.8, "ECE = {ece}, expected > 0.8 for miscalibrated");
    }

    #[test]
    fn ece_empty_inputs() {
        assert!((compute_ece(&[], &[], 10) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn ece_zero_bins() {
        assert!((compute_ece(&[0.5], &[1.0], 0) - 0.0).abs() < 1e-10);
    }

    // ─── Brier Score ─────────────────────────────────────────────────

    #[test]
    fn brier_score_perfect() {
        let predictions = vec![1.0, 0.0, 1.0, 0.0];
        let labels = vec![1.0, 0.0, 1.0, 0.0];
        assert!((compute_brier_score(&predictions, &labels) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn brier_score_worst() {
        let predictions = vec![0.0, 1.0];
        let labels = vec![1.0, 0.0];
        assert!((compute_brier_score(&predictions, &labels) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn brier_score_empty() {
        assert!((compute_brier_score(&[], &[]) - 0.0).abs() < 1e-10);
    }

    // ─── CalibratorConfig Serialization ──────────────────────────────

    #[test]
    fn calibrator_config_identity_roundtrip() {
        let config = CalibratorConfig::Identity(Identity);
        let json = serde_json::to_string(&config).unwrap();
        let decoded: CalibratorConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.name(), "identity");
    }

    #[test]
    fn calibrator_config_temperature_roundtrip() {
        let config = CalibratorConfig::TemperatureScaling(TemperatureScaling::new(2.5));
        let json = serde_json::to_string(&config).unwrap();
        let decoded: CalibratorConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.name(), "temperature_scaling");
        assert!((decoded.calibrate(0.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn calibrator_config_platt_roundtrip() {
        let config = CalibratorConfig::PlattScaling(PlattScaling::new(1.5, -0.3));
        let json = serde_json::to_string(&config).unwrap();
        let decoded: CalibratorConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.name(), "platt_scaling");
        // Verify the calibrated value is identical after roundtrip.
        let expected = PlattScaling::new(1.5, -0.3).calibrate(2.0);
        assert!((decoded.calibrate(2.0) - expected).abs() < 1e-10);
    }

    #[test]
    fn calibrator_config_isotonic_roundtrip() {
        let config = CalibratorConfig::IsotonicRegression(IsotonicRegression::new(
            vec![0.0, 0.5, 1.0],
            vec![0.1, 0.5, 0.9],
        ));
        let json = serde_json::to_string(&config).unwrap();
        let decoded: CalibratorConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.name(), "isotonic_regression");
        assert!((decoded.calibrate(0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn calibrator_config_tagged_json() {
        // Verify the JSON includes the "type" tag for deserialization.
        let config = CalibratorConfig::PlattScaling(PlattScaling::new(1.0, 0.0));
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"type\":\"PlattScaling\""));
    }

    // ─── Trait Object Safety ─────────────────────────────────────────

    #[test]
    fn calibrator_trait_is_object_safe() {
        fn _takes_dyn(_: &dyn ScoreCalibrator) {}
    }

    #[test]
    fn calibrator_via_dyn_dispatch() {
        let calibrators: Vec<Box<dyn ScoreCalibrator>> = vec![
            Box::new(Identity),
            Box::new(TemperatureScaling::new(1.0)),
            Box::new(PlattScaling::new(1.0, 0.0)),
            Box::new(IsotonicRegression::new(vec![0.0, 1.0], vec![0.2, 0.8])),
        ];
        // All should handle the same input without panicking.
        for cal in &calibrators {
            let _ = cal.calibrate(0.5);
            assert!(!cal.name().is_empty());
        }
    }

    // ─── Batch Calibration Integration Helper ───────────────────────

    #[test]
    fn calibrate_scores_with_labels_identity_passthrough() {
        let raw = vec![0.2, 0.8, 1.5, f64::NAN];
        let labels = vec![0.0, 1.0, 1.0, 0.0];

        let (calibrated, summary) = calibrate_scores_with_labels(&Identity, &raw, &labels, 10);

        assert_eq!(summary.count, 4);
        assert_eq!(calibrated.len(), 4);
        assert!((calibrated[0] - 0.2).abs() < 1e-10);
        assert!((calibrated[1] - 0.8).abs() < 1e-10);
        assert!((calibrated[2] - 1.0).abs() < 1e-10); // clamped
        assert!((calibrated[3] - 0.0).abs() < 1e-10); // NaN sanitized
    }

    #[test]
    fn calibrate_scores_with_labels_reduces_ece_with_isotonic_fit() {
        // Deliberately miscalibrated raw scores: highest scores are not always relevant.
        let raw = vec![0.95, 0.85, 0.75, 0.25, 0.15, 0.05];
        let labels = vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        let isotonic = IsotonicRegression::fit(&raw, &labels);

        let (_calibrated, summary) = calibrate_scores_with_labels(&isotonic, &raw, &labels, 6);

        assert!(
            summary.ece_after <= summary.ece_before + 1e-12,
            "expected isotonic fit to not worsen ECE: before={} after={}",
            summary.ece_before,
            summary.ece_after
        );
    }

    #[test]
    #[traced_test]
    fn calibrate_scores_with_labels_emits_scores_calibrated_log() {
        let raw = vec![0.1, 0.9];
        let labels = vec![0.0, 1.0];
        let _ = calibrate_scores_with_labels(&Identity, &raw, &labels, 5);

        assert!(logs_contain("scores calibrated"));
        assert!(logs_contain("scores_calibrated"));
    }
}
