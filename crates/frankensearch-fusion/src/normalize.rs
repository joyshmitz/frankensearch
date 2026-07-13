//! Score normalization utilities for fusion and blending.
//!
//! Different retrieval sources emit scores on different scales (for example
//! unbounded BM25 versus bounded cosine similarity), so normalization is often
//! required before weighted blending. RRF itself is rank-based and does not
//! require normalization.

use std::collections::VecDeque;

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
    nqc_cv_iter(scores.iter().copied())
}

/// Iterator form of [`nqc_cv`] for callers whose scores already live inside another record type.
/// Keeps the exact same accumulation order without materializing a temporary score vector.
pub(crate) fn nqc_cv_iter(scores: impl IntoIterator<Item = f32>) -> f32 {
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut count = 0_u32;
    for value in scores {
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

    #[cfg(any(test, feature = "bench-internals"))]
    fn from_values(sample: impl IntoIterator<Item = f32>) -> Self {
        let mut sorted_cv: Vec<f32> = sample.into_iter().filter(|v| v.is_finite()).collect();
        sorted_cv.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Self { sorted_cv }
    }

    /// Build directly from a batch of per-query lexical score slices — the deployment path
    /// (replay a query log / rolling window through the lexical tier, feed each query's top-k
    /// BM25 scores). Computes each query's NQC via [`nqc_cv`] and retains the sample. Empty
    /// batches (no scores) contribute a `0.0` NQC, which `from_sample` keeps.
    #[must_use]
    pub fn from_query_scores<'a>(queries: impl IntoIterator<Item = &'a [f32]>) -> Self {
        let sample: Vec<f32> = queries.into_iter().map(nqc_cv).collect();
        Self::from_sample(&sample)
    }

    /// Exact pre-optimization sample-builder path retained for same-binary benchmarks.
    #[cfg(feature = "bench-internals")]
    #[doc(hidden)]
    #[must_use]
    #[allow(clippy::needless_collect)]
    pub fn bench_from_query_scores_collect<'a>(
        queries: impl IntoIterator<Item = &'a [f32]>,
    ) -> Self {
        let sample: Vec<f32> = queries.into_iter().map(nqc_cv).collect();
        Self::from_sample(&sample)
    }

    /// Single-allocation sample-builder candidate for same-binary benchmarks.
    #[cfg(feature = "bench-internals")]
    #[doc(hidden)]
    #[must_use]
    pub fn bench_from_query_scores_iter<'a>(queries: impl IntoIterator<Item = &'a [f32]>) -> Self {
        Self::from_values(queries.into_iter().map(nqc_cv))
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

    /// Rebuild the sketch IN PLACE from already-finite values, reusing the existing `Vec`
    /// capacity (no per-rebuild allocation after warm-up) and an unstable total-order sort. This
    /// is the alloc-free deployment-rebuild path ([`NqcCvSampler::iter`] guarantees finiteness, so
    /// the [`from_sample`](Self::from_sample) filter is skipped). Byte-identical to a fresh
    /// `from_sample` for `percentile`/`dense_weight`: same finite multiset, sorted (`total_cmp` ==
    /// `partial_cmp` order on finite f32; equal-element order is irrelevant to the CDF).
    pub(crate) fn rebuild_from_finite(&mut self, values: impl IntoIterator<Item = f32>) {
        self.sorted_cv.clear();
        self.sorted_cv.extend(values);
        self.sorted_cv.sort_unstable_by(f32::total_cmp);
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

/// A bounded rolling sample of observed query NQC values that rebuilds a [`NqcDenseWeight`]
/// on demand — the deployment source [`NqcDenseWeight`]'s docs call for ("a rolling sample of
/// observed NQC values (the query stream); rebuild periodically"). Retains the most-recent
/// `capacity` observations so the installed sketch tracks the live query distribution rather
/// than a stale startup batch (the NQC scale is corpus-dependent, so the percentile map must
/// be estimated from the deployment's own queries).
///
/// Typical wiring: `observe` each query's lexical top-k BM25 scores, then periodically install
/// `sketch(min_samples)` as the searcher's [`NqcDenseWeight`]. Below `min_samples` the sketch
/// is empty, so the down-weight stays neutral (byte-identical fusion) during warm-up.
#[derive(Debug, Clone)]
pub struct NqcCvSampler {
    /// Most-recent observed NQC (`nqc_cv`) values, front = oldest.
    recent: VecDeque<f32>,
    capacity: usize,
}

impl NqcCvSampler {
    /// Create a sampler retaining the most-recent `capacity` NQC observations. A `capacity`
    /// of `0` is clamped to `1` so at least one observation is always retained.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            recent: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Record one query's NQC. Non-finite observations are ignored; when the sample is full the
    /// oldest observation is evicted (bounded rolling window).
    pub fn observe(&mut self, cv: f32) {
        if !cv.is_finite() {
            return;
        }
        if self.recent.len() == self.capacity {
            self.recent.pop_front();
        }
        self.recent.push_back(cv);
    }

    /// Record one query from its lexical (BM25) top-k scores — computes [`nqc_cv`] and observes
    /// it. A degenerate score set (empty / zero-mean) yields a `0.0` NQC, a valid low-commitment
    /// observation that is retained (matching [`NqcDenseWeight::from_query_scores`]).
    pub fn observe_lexical_scores(&mut self, scores: &[f32]) {
        self.observe(nqc_cv(scores));
    }

    /// Number of retained observations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.recent.len()
    }

    /// Whether no observations have been retained yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.recent.is_empty()
    }

    /// The retention bound (the most-recent-N rolling-window size).
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Iterate the retained (all-finite) observations oldest→newest, without allocating —
    /// feeds [`NqcDenseWeight::rebuild_from_finite`] for an alloc-free periodic rebuild.
    pub fn iter(&self) -> impl Iterator<Item = f32> + '_ {
        self.recent.iter().copied()
    }

    /// Build a [`NqcDenseWeight`] from the current sample once at least `min_samples` (and at
    /// least one) observations have accumulated; below that returns an empty (neutral) sketch so
    /// the down-weight stays off during warm-up. `min_samples` guards against a percentile map
    /// estimated from too few queries.
    #[must_use]
    pub fn sketch(&self, min_samples: usize) -> NqcDenseWeight {
        if self.recent.len() < min_samples.max(1) {
            return NqcDenseWeight::new();
        }
        let sample: Vec<f32> = self.recent.iter().copied().collect();
        NqcDenseWeight::from_sample(&sample)
    }
}

/// A self-driving rolling NQC dense down-weight: owns an [`NqcCvSampler`] plus a cached
/// [`NqcDenseWeight`], observes each query's NQC, and periodically rebuilds the percentile
/// sketch so the down-weight tracks the live query distribution with no external sample
/// management. This is the deployment brain that turns the (dormant) static
/// [`NqcDenseWeight`] into a live default: a searcher holds one behind interior mutability
/// (matching the crate's existing per-search `Mutex` state) and calls
/// [`weight_for_cv`](Self::weight_for_cv) each query.
///
/// A query is scored against the sketch built from **prior** observations, then contributes
/// to the sample — so a query never influences its own weight, and the whole thing starts
/// **neutral** (`1.0`, byte-identical fusion) until `min_samples` observations warm the
/// sketch. `beta <= 0` disables it entirely (always `1.0`, no observation).
#[derive(Debug, Clone)]
pub struct AdaptiveNqcDenseWeight {
    sampler: NqcCvSampler,
    sketch: NqcDenseWeight,
    beta: f32,
    w_min: f32,
    min_samples: usize,
    rebuild_every: usize,
    since_rebuild: usize,
}

impl AdaptiveNqcDenseWeight {
    /// Create a rolling down-weight. `beta` (≈0.5 measured best) and `w_min` are the
    /// [`NqcDenseWeight::dense_weight`] parameters; `capacity` bounds the rolling sample
    /// window; `min_samples` is the minimum observations before the sketch activates (below
    /// it the weight stays neutral); `rebuild_every` is how many observations between sketch
    /// rebuilds (`0` is clamped to `1`). Starts neutral (empty sketch).
    #[must_use]
    pub fn new(
        beta: f32,
        w_min: f32,
        capacity: usize,
        min_samples: usize,
        rebuild_every: usize,
    ) -> Self {
        Self {
            sampler: NqcCvSampler::new(capacity),
            sketch: NqcDenseWeight::new(),
            beta,
            w_min,
            min_samples,
            rebuild_every: rebuild_every.max(1),
            since_rebuild: 0,
        }
    }

    /// Whether the down-weight is enabled (`beta > 0`). When disabled, [`weight_for_cv`] is a
    /// no-op returning the neutral `1.0`.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.beta > 0.0
    }

    /// The dense-tier multiplier for a query with NQC `cv`, using the sketch built from prior
    /// observations; then records `cv` into the rolling sample and rebuilds the sketch when due.
    /// Returns the neutral `1.0` while disabled (`beta <= 0`) or during warm-up (empty sketch).
    pub fn weight_for_cv(&mut self, cv: f32) -> f32 {
        if self.beta <= 0.0 {
            return 1.0;
        }
        let factor = self.sketch.dense_weight(cv, self.beta, self.w_min);
        self.sampler.observe(cv);
        self.since_rebuild += 1;
        if self.since_rebuild >= self.rebuild_every {
            // In-place rebuild: reuse the sketch's Vec capacity + a fast unstable total-order
            // sort (the sampler is all-finite). An empty iter → empty (neutral) sketch during the
            // pre-`min_samples` warm-up. Avoids the two per-rebuild allocations + stable
            // partial_cmp sort of the old `sampler.sketch()` path (measured ~3x cheaper).
            if self.sampler.len() >= self.min_samples {
                self.sketch.rebuild_from_finite(self.sampler.iter());
            } else {
                self.sketch.rebuild_from_finite(std::iter::empty());
            }
            self.since_rebuild = 0;
        }
        factor
    }

    /// [`weight_for_cv`](Self::weight_for_cv) from a query's lexical (BM25) top-k scores,
    /// computing its NQC via [`nqc_cv`].
    pub fn weight_for_lexical_scores(&mut self, scores: &[f32]) -> f32 {
        self.weight_for_cv(nqc_cv(scores))
    }

    /// The blessed production defaults for the rolling dense down-weight: `beta = 0.5`
    /// (measured best — floors the dense multiplier at `0.5`, so the tier is never fully
    /// dropped and keeps its minority upside), `w_min = 0.5` (the natural floor at `beta=0.5`,
    /// `> 0` so a scaled weight is never sanitized back to neutral), a `2048`-query rolling
    /// window, `min_samples = 128` before the sketch activates, and a rebuild every `64`
    /// observations. Realizes the measured aggregate +0.0022 nDCG@10 (pooled 95% CI
    /// `[+0.0008, +0.0035]`) once warmed; neutral (byte-identical) during the 128-query warm-up.
    #[must_use]
    pub fn production_default() -> Self {
        Self::new(0.5, 0.5, 2048, 128, 64)
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
        AdaptiveNqcDenseWeight, NormalizationMethod, NqcCvSampler, NqcDenseWeight,
        min_max_normalize, normalize_in_place, normalize_scores, normalize_scores_with_method,
        nqc_cv, nqc_cv_iter, z_score_normalize,
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
    fn nqc_sampler_capacity_zero_retains_at_least_one() {
        let mut sampler = NqcCvSampler::new(0);
        assert_eq!(sampler.capacity(), 1);
        sampler.observe(0.3);
        sampler.observe(0.5);
        assert_eq!(sampler.len(), 1, "capacity clamps to 1 -> only the most-recent is kept");
    }

    #[test]
    fn nqc_sampler_bounded_rolling_window_evicts_oldest() {
        let mut sampler = NqcCvSampler::new(3);
        for cv in [0.1, 0.2, 0.3, 0.4, 0.5] {
            sampler.observe(cv);
        }
        assert_eq!(sampler.len(), 3, "window bounded to the 3 most-recent");
        // The sketch reflects only the retained window {0.3, 0.4, 0.5}.
        let weight = sampler.sketch(1);
        assert_eq!(weight.len(), 3);
        assert!(
            (weight.percentile(0.25) - 0.0).abs() <= EPSILON,
            "0.25 is below the retained window"
        );
        assert!(
            (weight.percentile(0.5) - 1.0).abs() <= EPSILON,
            "0.5 is the max of the retained window"
        );
    }

    #[test]
    fn nqc_sampler_ignores_non_finite_observations() {
        let mut sampler = NqcCvSampler::new(8);
        sampler.observe(f32::NAN);
        sampler.observe(f32::INFINITY);
        sampler.observe(f32::NEG_INFINITY);
        sampler.observe(0.2);
        assert_eq!(sampler.len(), 1, "only the finite observation is retained");
    }

    #[test]
    fn nqc_sampler_sketch_below_min_samples_is_neutral() {
        let mut sampler = NqcCvSampler::new(8);
        sampler.observe(0.1);
        sampler.observe(0.2);
        let weight = sampler.sketch(5);
        assert!(weight.is_empty(), "below min_samples -> empty (neutral) sketch");
        assert!(
            (weight.dense_weight(0.15, 0.5, 0.1) - 1.0).abs() <= EPSILON,
            "empty sketch -> neutral dense weight (byte-identical fusion during warm-up)"
        );
    }

    #[test]
    fn nqc_sampler_observe_lexical_scores_matches_from_query_scores() {
        // Degenerate score sets contribute a 0.0 NQC, retained just like from_query_scores.
        let mut sampler = NqcCvSampler::new(8);
        sampler.observe_lexical_scores(&[]); // empty -> 0.0
        sampler.observe_lexical_scores(&[5.0, 5.0, 5.0]); // zero-variance -> 0.0
        sampler.observe_lexical_scores(&[10.0, 1.0, 0.5]); // peaked -> positive
        assert_eq!(sampler.len(), 3);
        let weight = sampler.sketch(1);
        assert_eq!(weight.len(), 3);
        // Two 0.0 observations and one positive -> CDF(0.0) = 2/3.
        assert!(
            (weight.percentile(0.0) - 2.0 / 3.0).abs() <= EPSILON,
            "percentile(0.0)={}",
            weight.percentile(0.0)
        );
    }

    #[test]
    fn adaptive_nqc_warmup_is_neutral_then_activates() {
        // capacity 100, min_samples 10, rebuild every 5.
        let mut adaptive = AdaptiveNqcDenseWeight::new(0.5, 0.1, 100, 10, 5);
        assert!(adaptive.is_active());
        // First rebuild fires at 5 observations but 5 < min_samples(10) -> still empty/neutral.
        for i in 0..9 {
            let w = adaptive.weight_for_cv(0.1 + 0.02 * i as f32);
            assert!((w - 1.0).abs() <= EPSILON, "warm-up observation {i} must be neutral, got {w}");
        }
        // The 10th observation triggers the 2nd rebuild (10 obs >= min_samples) -> sketch active.
        let _ = adaptive.weight_for_cv(0.3);
        // A very high cv now maps to a high percentile -> a down-weight strictly below 1.0.
        let high = adaptive.weight_for_cv(10.0);
        assert!(high < 1.0, "high-NQC query must be down-weighted after warm-up, got {high}");
        assert!(high >= 0.1 - EPSILON, "floored at w_min");
    }

    #[test]
    fn adaptive_nqc_beta_zero_is_always_neutral() {
        let mut adaptive = AdaptiveNqcDenseWeight::new(0.0, 0.1, 100, 1, 1);
        assert!(!adaptive.is_active());
        for cv in [0.0, 0.5, 5.0, 50.0] {
            assert!((adaptive.weight_for_cv(cv) - 1.0).abs() <= EPSILON, "beta=0 stays neutral");
        }
    }

    #[test]
    fn adaptive_nqc_query_does_not_influence_its_own_weight() {
        // Warm a stable low-cv distribution, then a single high-cv query is scored against the
        // PRIOR (low-cv) sketch -> high percentile -> down-weighted, proving score-before-observe.
        let mut adaptive = AdaptiveNqcDenseWeight::new(1.0, 0.0, 100, 4, 4);
        for _ in 0..8 {
            let _ = adaptive.weight_for_cv(0.05);
        }
        let outlier = adaptive.weight_for_cv(9.0);
        assert!(outlier < 0.5, "an outlier scored against the prior low-cv sketch is down-weighted");
    }

    #[test]
    fn adaptive_nqc_production_default_is_active_and_warms_up_neutral() {
        let mut adaptive = AdaptiveNqcDenseWeight::production_default();
        assert!(adaptive.is_active(), "production default enables the down-weight (beta=0.5)");
        // The 128-sample warm-up keeps the first observations neutral (byte-identical fusion).
        for i in 0..20 {
            let w = adaptive.weight_for_cv(0.1 + 0.01 * i as f32);
            assert!((w - 1.0).abs() <= EPSILON, "obs {i} neutral during the 128-query warm-up");
        }
    }

    #[test]
    fn adaptive_nqc_lexical_scores_path_matches_cv_path() {
        let scores = [10.0_f32, 1.0, 0.5];
        let cv = nqc_cv(&scores);
        let mut by_scores = AdaptiveNqcDenseWeight::new(0.5, 0.1, 16, 1, 1);
        let mut by_cv = AdaptiveNqcDenseWeight::new(0.5, 0.1, 16, 1, 1);
        assert!(
            (by_scores.weight_for_lexical_scores(&scores) - by_cv.weight_for_cv(cv)).abs()
                <= EPSILON,
            "lexical-scores path == precomputed-cv path"
        );
    }

    #[test]
    fn rebuild_from_finite_matches_from_sample() {
        // The alloc-free in-place rebuild yields a sketch byte-identical (for percentile /
        // dense_weight) to a fresh from_sample of the same finite values — even when reusing a
        // pre-populated Vec of a different length.
        let sample = [0.3_f32, 0.1, 0.5, 0.2, 0.4, 0.15, 0.25];
        let fresh = NqcDenseWeight::from_sample(&sample);
        let mut reused = NqcDenseWeight::from_sample(&[0.9, 0.8]); // capacity reused, cleared first
        reused.rebuild_from_finite(sample.iter().copied());
        assert_eq!(fresh.len(), reused.len());
        for &cv in &[0.0_f32, 0.12, 0.2, 0.31, 0.5, 0.99] {
            assert!(
                (fresh.percentile(cv) - reused.percentile(cv)).abs() <= EPSILON,
                "percentile mismatch at cv={cv}: {} vs {}",
                fresh.percentile(cv),
                reused.percentile(cv)
            );
            assert!(
                (fresh.dense_weight(cv, 0.5, 0.1) - reused.dense_weight(cv, 0.5, 0.1)).abs()
                    <= EPSILON
            );
        }
        // Empty rebuild → neutral (warm-up path).
        reused.rebuild_from_finite(std::iter::empty());
        assert!(reused.is_empty());
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
    fn nqc_cv_iterator_is_bit_identical_to_slice() {
        let cases: &[&[f32]] = &[
            &[],
            &[1.0, 2.0, 3.0, 5.0],
            &[f32::NAN, 1.0, f32::INFINITY, 2.0, f32::NEG_INFINITY],
            &[-4.0, -2.0, 0.0],
            &[f32::MIN_POSITIVE, 1.0e-20, 1.0e-10, 1.0],
        ];
        for &scores in cases {
            assert_eq!(
                nqc_cv(scores).to_bits(),
                nqc_cv_iter(scores.iter().copied()).to_bits(),
                "iterator reduction changed bits for {scores:?}"
            );
        }
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
    fn nqc_weight_from_query_scores_computes_and_ranks_nqc() {
        let peaked: &[f32] = &[10.0, 1.0]; // high NQC (cv ≈ 0.818)
        let uniform: &[f32] = &[5.0, 5.0, 5.0]; // NQC = 0
        let mid: &[f32] = &[3.0, 2.0];
        let w = NqcDenseWeight::from_query_scores([peaked, uniform, mid]);
        assert_eq!(w.len(), 3);
        // The peaked query sits above the uniform (NQC 0) one in the sampled distribution.
        assert!(w.percentile(nqc_cv(peaked)) >= w.percentile(nqc_cv(uniform)));
        // So enabling the down-weight demotes the dense tier more on the high-NQC query.
        assert!(w.dense_weight(nqc_cv(peaked), 0.5, 0.0) <= w.dense_weight(nqc_cv(uniform), 0.5, 0.0));
    }

    #[test]
    fn nqc_weight_iterator_builder_is_bit_identical_to_collect_builder() {
        let queries: &[&[f32]] = &[
            &[],
            &[10.0, 1.0],
            &[5.0, 5.0, 5.0],
            &[f32::NAN, 4.0, 2.0, f32::INFINITY],
            &[-4.0, -2.0, 0.0],
        ];
        let original = NqcDenseWeight::from_query_scores(queries.iter().copied());
        let candidate = NqcDenseWeight::from_values(queries.iter().copied().map(nqc_cv));
        assert_eq!(
            original
                .sorted_cv
                .iter()
                .copied()
                .map(f32::to_bits)
                .collect::<Vec<_>>(),
            candidate
                .sorted_cv
                .iter()
                .copied()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        );
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
