//! Distribution-free recall certification for approximate nearest-neighbour search.
//!
//! The ANN search reports `SearchStats.estimated_recall` via [`crate::hnsw`]'s
//! `estimate_recall`, a magic-constant heuristic — `0.9 + 0.1·log2(ef/k)`, clamped
//! — with **no empirical grounding and no guarantee**. That guess is exactly the
//! kind of "magic threshold" the alien-graveyard §12.1 (Conformal Prediction,
//! Tier S) is meant to remove: a distribution-free, finite-sample-valid bound that
//! holds under only an exchangeability assumption — no assumptions about the
//! embedding distribution, no asymptotics.
//!
//! This module turns a *calibration sample of measured per-query recalls*
//! (produced by the crate's own `recall_at_k` bruteforce comparison) into a
//! **certified recall lower bound**. It is the automated replacement for the human
//! "recall-budget sign-off" that blocks turning on aggressive ANN (`ef` tuned down
//! for a 2.6–5× vector-tier speed-up): the certificate *is* the sign-off.
//!
//! Two complementary, both distribution-free and finite-sample-valid, certificates:
//!
//! * [`conformal_recall_lower_bound`] — a split-conformal **per-query lower
//!   tolerance bound** `L`: a fresh query's recall is `≥ L` with probability
//!   `≥ 1 − α` (rank-based; the graveyard §12.1 primitive).
//! * [`mean_recall_lower_bound`] — a Hoeffding **lower confidence bound on the
//!   mean** recall over queries: `E[recall] ≥ L` with confidence `≥ 1 − δ`.
//!
//! Both return the trivial bound `0.0` when the sample is too small to certify
//! anything at the requested error level (rather than inventing a number, which is
//! precisely the failure mode of the heuristic they replace).

/// Split-conformal **lower tolerance bound** on per-query recall.
///
/// Given `recalls`, a calibration sample of per-query recall@k values in `[0, 1]`
/// (each the fraction of the true top-k a single query's ANN result recovered),
/// and a miss rate `alpha ∈ (0, 1)`, returns `L` such that, for a fresh
/// exchangeable query,
///
/// ```text
///     P(recall_new ≥ L) ≥ 1 − alpha.
/// ```
///
/// `L` is the `⌊alpha·(n+1)⌋`-th smallest calibration recall (1-indexed). The
/// coverage guarantee is finite-sample exact under exchangeability (it follows
/// from the rank of `recall_new` among the `n+1` values being uniform), with **no**
/// distributional assumptions. When `⌊alpha·(n+1)⌋ = 0` the sample is too small to
/// certify any positive bound at this `alpha`, and `0.0` (the trivial, always-valid
/// bound, since recall ≥ 0) is returned. Non-finite calibration entries are ignored.
///
/// This is monotone: a smaller `alpha` (stronger guarantee) never returns a larger
/// bound, and adding calibration data can only tighten it toward the true quantile.
#[must_use]
pub fn conformal_recall_lower_bound(recalls: &[f64], alpha: f64) -> f64 {
    if !(0.0..1.0).contains(&alpha) {
        return 0.0;
    }
    let mut sorted: Vec<f64> = recalls.iter().copied().filter(|r| r.is_finite()).collect();
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    // Rank k = floor(alpha * (n + 1)) in [0, n]. k == 0 => cannot certify > 0.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let rank = (alpha * (n as f64 + 1.0)).floor() as usize;
    if rank == 0 {
        return 0.0;
    }
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // 1-indexed rank -> 0-indexed; clamp defensively (rank <= n always holds here).
    let idx = (rank - 1).min(n - 1);
    sorted[idx].clamp(0.0, 1.0)
}

/// Hoeffding **lower confidence bound on the mean** recall.
///
/// Given per-query `recalls` in `[0, 1]` and a confidence parameter `delta ∈ (0, 1)`,
/// returns `L` such that
///
/// ```text
///     P(E[recall] ≥ L) ≥ 1 − delta,
/// ```
///
/// namely `mean − sqrt(ln(1/delta) / (2n))`, clamped to `[0, 1]`. Hoeffding's
/// inequality gives this for any distribution of bounded `[0, 1]` variables with no
/// further assumptions. Returns `0.0` for an empty sample.
#[must_use]
pub fn mean_recall_lower_bound(recalls: &[f64], delta: f64) -> f64 {
    if !(0.0..1.0).contains(&delta) {
        return 0.0;
    }
    let finite: Vec<f64> = recalls.iter().copied().filter(|r| r.is_finite()).collect();
    let n = finite.len();
    if n == 0 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let n_f = n as f64;
    let mean = finite.iter().sum::<f64>() / n_f;
    let radius = ((1.0 / delta).ln() / (2.0 * n_f)).sqrt();
    (mean - radius).clamp(0.0, 1.0)
}

/// A certified `ef_search` choice: the smallest `ef` whose conformal recall lower
/// bound meets the target, together with the certified bound and whether the target
/// was actually met.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CertifiedEf {
    /// The chosen `ef_search`.
    pub ef_search: usize,
    /// The certified per-query recall lower bound at this `ef` (holds w.p. ≥ 1−α).
    pub certified_recall: f64,
    /// `true` iff `certified_recall ≥ target` — i.e. the recall budget is met with
    /// a finite-sample guarantee (the automated sign-off).
    pub meets_target: bool,
}

/// Select the **minimal `ef_search` whose certified recall lower bound meets
/// `target`** — the automated replacement for a human recall-budget sign-off.
///
/// `calibration` pairs each candidate `ef_search` with its measured per-query
/// recall sample. Returns the smallest `ef` for which
/// [`conformal_recall_lower_bound`]`(recalls, alpha) ≥ target` (cheapest ANN that
/// is *certified* to hit the budget). If none qualifies, returns the candidate with
/// the highest certified bound and `meets_target = false`, so the caller always
/// learns the best certifiable option instead of silently trusting a guess. Returns
/// `None` only for an empty `calibration`.
#[must_use]
pub fn certified_min_ef(
    calibration: &[(usize, Vec<f64>)],
    target: f64,
    alpha: f64,
) -> Option<CertifiedEf> {
    let mut best: Option<CertifiedEf> = None;
    // Ascending ef so the FIRST that meets the target is the minimal (cheapest) one.
    let mut sorted: Vec<&(usize, Vec<f64>)> = calibration.iter().collect();
    sorted.sort_by_key(|(ef, _)| *ef);
    for (ef, recalls) in sorted {
        let bound = conformal_recall_lower_bound(recalls, alpha);
        let candidate = CertifiedEf {
            ef_search: *ef,
            certified_recall: bound,
            meets_target: bound >= target,
        };
        if candidate.meets_target {
            return Some(candidate);
        }
        // Track the best-certifiable fallback (highest bound).
        let better = match best {
            Some(b) => bound > b.certified_recall,
            None => true,
        };
        if better {
            best = Some(candidate);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LCG so tests need no `rand` dep and no clock.
    struct Lcg(u64);
    impl Lcg {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            self.0
        }
        /// Uniform f64 in [0, 1).
        #[allow(clippy::cast_precision_loss)]
        fn unit(&mut self) -> f64 {
            ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    #[test]
    fn conformal_bound_is_trivial_when_sample_too_small() {
        // alpha=0.05, n=10 -> floor(0.05*11)=0 -> cannot certify > 0.
        let recalls = vec![0.9; 10];
        assert_eq!(conformal_recall_lower_bound(&recalls, 0.05), 0.0);
        // n=19 -> floor(0.05*20)=1 -> certifiable.
        let recalls = vec![0.9; 19];
        assert!(conformal_recall_lower_bound(&recalls, 0.05) > 0.0);
        // Degenerate inputs.
        assert_eq!(conformal_recall_lower_bound(&[], 0.1), 0.0);
        assert_eq!(conformal_recall_lower_bound(&[0.9, 0.8], 0.0), 0.0);
        assert_eq!(conformal_recall_lower_bound(&[0.9, 0.8], 1.0), 0.0);
    }

    #[test]
    fn conformal_bound_recovers_the_order_statistic() {
        // 99 sorted values 0.01..=0.99; alpha=0.1 -> floor(0.1*100)=10 -> 10th
        // smallest = 0.10.
        let recalls: Vec<f64> = (1..=99).map(|i| f64::from(i) / 100.0).collect();
        let bound = conformal_recall_lower_bound(&recalls, 0.10);
        assert!((bound - 0.10).abs() < 1e-9, "got {bound}");
    }

    #[test]
    fn conformal_bound_is_monotone_in_alpha() {
        let mut lcg = Lcg(42);
        let recalls: Vec<f64> = (0..500).map(|_| lcg.unit()).collect();
        // Smaller alpha (stronger guarantee) => smaller-or-equal bound.
        let strong = conformal_recall_lower_bound(&recalls, 0.01);
        let weak = conformal_recall_lower_bound(&recalls, 0.20);
        assert!(strong <= weak, "strong={strong} weak={weak}");
    }

    /// THE validity test: over many independent draws, a fresh query's recall
    /// falls below the certified conformal lower bound at most `alpha` of the time
    /// (within sampling noise) — for an *arbitrary* recall distribution.
    #[test]
    fn conformal_bound_has_valid_finite_sample_coverage() {
        let alpha = 0.10;
        let n_cal = 200;
        let trials = 4000;
        let mut lcg = Lcg(0x5eed);
        let mut misses = 0usize;
        for _ in 0..trials {
            // A deliberately skewed, arbitrary recall law: Beta-ish via min of two
            // uniforms shifted — NOT anything the bound is allowed to assume.
            let draw = |lcg: &mut Lcg| -> f64 {
                let a = lcg.unit();
                let b = lcg.unit();
                // Concentrates near 1.0 with a left tail — realistic for recall.
                1.0 - (a * b) * 0.4
            };
            let cal: Vec<f64> = (0..n_cal).map(|_| draw(&mut lcg)).collect();
            let bound = conformal_recall_lower_bound(&cal, alpha);
            let fresh = draw(&mut lcg);
            if fresh < bound {
                misses += 1;
            }
        }
        #[allow(clippy::cast_precision_loss)]
        let miss_rate = misses as f64 / trials as f64;
        // Finite-sample guarantee: miss rate <= alpha in expectation. Allow a small
        // Monte-Carlo slack (3 s.e. ~ 0.014 at these sizes).
        assert!(
            miss_rate <= alpha + 0.02,
            "conformal coverage violated: miss_rate={miss_rate:.4} > alpha={alpha}"
        );
    }

    #[test]
    fn mean_bound_lower_bounds_and_tightens_with_n() {
        // All recalls = 0.95: mean LCB = 0.95 - sqrt(ln(1/delta)/(2n)), rising to
        // the mean as n grows.
        let small = mean_recall_lower_bound(&vec![0.95; 30], 0.05);
        let large = mean_recall_lower_bound(&vec![0.95; 3000], 0.05);
        assert!(small < large, "small={small} large={large}");
        assert!(large <= 0.95 && large > 0.90, "large={large}");
        assert!(small >= 0.0);
    }

    #[test]
    fn mean_bound_coverage_holds() {
        let delta = 0.05;
        let n = 300;
        let trials = 3000;
        let mut lcg = Lcg(0xc0ffee);
        let true_mean = 0.9;
        let mut misses = 0usize;
        for _ in 0..trials {
            // Bernoulli(0.9) recalls (0/1) — a worst-ish case for a bounded mean.
            let cal: Vec<f64> = (0..n)
                .map(|_| if lcg.unit() < true_mean { 1.0 } else { 0.0 })
                .collect();
            if true_mean < mean_recall_lower_bound(&cal, delta) {
                misses += 1;
            }
        }
        #[allow(clippy::cast_precision_loss)]
        let miss_rate = misses as f64 / trials as f64;
        assert!(miss_rate <= delta, "mean LCB coverage violated: {miss_rate:.4} > {delta}");
    }

    #[test]
    fn certified_min_ef_picks_the_cheapest_certified_option() {
        // ef=40 misses, ef=100 meets the target -> pick 100.
        let calibration = vec![
            (40usize, vec![0.80; 300]),
            (100usize, vec![0.99; 300]),
            (200usize, vec![0.999; 300]),
        ];
        let choice = certified_min_ef(&calibration, 0.95, 0.05).unwrap();
        assert_eq!(choice.ef_search, 100);
        assert!(choice.meets_target);
        assert!(choice.certified_recall >= 0.95);
    }

    #[test]
    fn certified_min_ef_falls_back_to_best_when_none_meets_target() {
        let calibration = vec![(40usize, vec![0.70; 300]), (100usize, vec![0.85; 300])];
        let choice = certified_min_ef(&calibration, 0.99, 0.05).unwrap();
        assert!(!choice.meets_target);
        assert_eq!(choice.ef_search, 100); // higher certified bound
        assert!(certified_min_ef(&[], 0.9, 0.05).is_none());
    }

    /// The heuristic `estimate_recall` can be *over-optimistic* on a distribution
    /// where the conformal certificate correctly refuses the budget — the whole
    /// point of replacing a guess with a guarantee.
    #[test]
    fn certificate_catches_heuristic_overconfidence() {
        // ef/k = 100/10 -> heuristic estimate = 0.9 + 0.1*log2(10) ≈ 1.232 -> 1.0.
        let heuristic = 0.1_f64.mul_add((100.0_f64 / 10.0).log2(), 0.9).clamp(0.0, 1.0);
        assert!(heuristic >= 0.999, "heuristic claims ~perfect recall: {heuristic}");
        // But measured recall at that ef is actually ~0.85 on this corpus:
        let measured = vec![0.85; 500];
        let certified = conformal_recall_lower_bound(&measured, 0.05);
        assert!(
            certified < 0.9,
            "certificate should refuse the heuristic's optimism, got {certified}"
        );
    }
}
