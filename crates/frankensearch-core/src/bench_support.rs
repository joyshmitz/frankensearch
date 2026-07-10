//! Bench-only measurement harness. Not a shipping path (`feature = "bench-internals"`).
//!
//! Lives in `frankensearch-core` so **every** crate's benches can share one decidability harness —
//! including `frankensearch-index`, which cannot depend on `frankensearch-fusion` (that would be a
//! dependency cycle). This is what makes an int8-ADC-scan A/B (`bd-b5wl`) decidable on the same
//! footing as a fusion A/B. Std-only (`black_box` + `Instant`); zero new dependencies.
//!
//! # Why criterion alone cannot decide a small lever
//!
//! Registering ORIG and CAND as two criterion benchmarks — even with each one internally
//! interleaving a timed and an untimed half — does **not** cancel worker drift, because criterion
//! runs the two benchmarks *sequentially*, often minutes apart. The internal interleaving only
//! equalizes cache/branch state *within* an arm. Measured consequence (`neighbor_smooth`, worker
//! `hz1`, 120 samples): an A/A null control — the identical function registered as both arms —
//! reported a median ratio of **1.1265×** at pool 50 and **0.9268×** at pool 100, a range that does
//! not even contain 1.000. Any lever whose effect is smaller than that is undecidable on that
//! harness, and a WIN or REJECT resting on one is meaningless.
//!
//! # What this does instead
//!
//! [`paired_median_ratio`] runs both arms inside **one** measured routine, in **alternating rounds**:
//! round `r` times `(a, b)` when `r` is even and `(b, a)` when odd, so first-mover and cache-warm
//! bias cancel across rounds. It forms the ratio **per round**, so drift is shared by the two arms
//! within a few microseconds of each other rather than across minutes, then reports the **median**
//! ratio with a p5/p95 spread.
//!
//! Gate on the **median against the null's observed spread**, not on `cv_pct` — `cv < 5%` is
//! unattainable on this fleet. The floor is **per-function**: calibrate it for the function you are
//! actually measuring by running `paired_median_ratio(rounds, inner, base, base)` (an A/A null)
//! before trusting `paired_median_ratio(rounds, inner, base, cand)`.

use std::hint::black_box;
use std::time::{Duration, Instant};

/// Median ratio `b/a` with a p5/p95 spread, plus the per-round sample count.
#[derive(Debug, Clone, Copy)]
pub struct PairedRatio {
    /// Median of the per-round `b/a` ratios. For an A/A null this should sit at ~1.000.
    pub median: f64,
    /// 5th percentile of the per-round ratios.
    pub p5: f64,
    /// 95th percentile of the per-round ratios.
    pub p95: f64,
    /// Rounds actually measured.
    pub rounds: usize,
}

impl PairedRatio {
    /// Whether `self` (a candidate) lies clearly outside `null`'s observed spread — the decidability
    /// test. A candidate median inside the null's p5..p95 band is indistinguishable from noise.
    #[must_use]
    pub fn decidable_against(&self, null: &Self) -> bool {
        self.median > null.p95 || self.median < null.p5
    }
}

/// Time `inner` back-to-back calls of `f`, returning the elapsed duration for the whole batch.
///
/// Batching amortizes the `Instant::now()` pair; the caller divides by `inner` for a per-call cost.
fn time_batch<F: FnMut()>(inner: u32, f: &mut F) -> Duration {
    let t = Instant::now();
    for _ in 0..inner {
        f();
    }
    t.elapsed()
}

/// Run `a` and `b` in alternating rounds within one routine and return the median `b/a` ratio.
///
/// Each round times `inner` calls of each arm. Even rounds run `a` then `b`; odd rounds run `b` then
/// `a`. Callers must `black_box` their inputs and results inside the closures — this function
/// `black_box`es the closures themselves but cannot see through them.
///
/// Panics if `rounds == 0` or `inner == 0`.
#[must_use]
pub fn paired_median_ratio<A: FnMut(), B: FnMut()>(
    rounds: usize,
    inner: u32,
    mut a: A,
    mut b: B,
) -> PairedRatio {
    assert!(rounds > 0 && inner > 0, "rounds and inner must be non-zero");

    // Warm both arms so the first measured round is not a cold-code outlier.
    for _ in 0..2 {
        black_box(time_batch(inner, &mut a));
        black_box(time_batch(inner, &mut b));
    }

    let mut ratios: Vec<f64> = Vec::with_capacity(rounds);
    for r in 0..rounds {
        let (ta, tb) = if r % 2 == 0 {
            let ta = time_batch(inner, &mut a);
            let tb = time_batch(inner, &mut b);
            (ta, tb)
        } else {
            let tb = time_batch(inner, &mut b);
            let ta = time_batch(inner, &mut a);
            (ta, tb)
        };
        let ta = ta.as_secs_f64();
        if ta > 0.0 {
            ratios.push(tb.as_secs_f64() / ta);
        }
    }

    assert!(
        !ratios.is_empty(),
        "no round produced a positive base timing"
    );
    ratios.sort_unstable_by(f64::total_cmp);
    let n = ratios.len();
    // `q ∈ [0,1]` and `n ≥ 1`, so the product is a finite non-negative index ≤ n-1.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let pct = |q: f64| ratios[((n - 1) as f64 * q).round() as usize];
    PairedRatio {
        median: pct(0.5),
        p5: pct(0.05),
        p95: pct(0.95),
        rounds: n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An A/A null over identical closures must land at ~1.0 and bracket it.
    #[test]
    fn null_control_of_identical_work_is_near_one() {
        let work = || {
            let mut acc = 0u64;
            for i in 0..2_000u64 {
                acc = acc.wrapping_add(black_box(i).wrapping_mul(2_654_435_761));
            }
            black_box(acc);
        };
        let null = paired_median_ratio(41, 8, work, work);
        assert!(
            null.median > 0.75 && null.median < 1.33,
            "A/A null median {} strayed far from 1.0",
            null.median
        );
        assert!(null.p5 <= null.median && null.median <= null.p95);
        assert_eq!(null.rounds, 41);
    }

    /// A candidate doing ~4x the base work must be decidable against that null.
    #[test]
    fn a_large_effect_is_decidable_against_the_null() {
        let base = || {
            let mut acc = 0u64;
            for i in 0..2_000u64 {
                acc = acc.wrapping_add(black_box(i).wrapping_mul(2_654_435_761));
            }
            black_box(acc);
        };
        let cand = || {
            let mut acc = 0u64;
            for i in 0..8_000u64 {
                acc = acc.wrapping_add(black_box(i).wrapping_mul(2_654_435_761));
            }
            black_box(acc);
        };
        let null = paired_median_ratio(41, 8, base, base);
        let lever = paired_median_ratio(41, 8, base, cand);
        assert!(lever.median > 2.0, "expected ~4x, got {}", lever.median);
        assert!(
            lever.decidable_against(&null),
            "4x effect (median {}) should clear the null band [{}, {}]",
            lever.median,
            null.p5,
            null.p95
        );
    }
}
