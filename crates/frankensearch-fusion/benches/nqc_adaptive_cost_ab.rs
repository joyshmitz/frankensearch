//! Same-binary gate for the DEFAULT-ON `AdaptiveNqcDenseWeight` rolling-order lever.
//!
//! Both arms execute the complete adaptive production boundary with precomputed NQC values:
//! - ORIG: retain insertion order and sort all 2,048 values every 64 queries.
//! - CAND: maintain the exact rolling multiset in sorted order on each observation, then copy
//!   that order into the 64-query snapshot without sorting.
//! Before timing, the harness proves every returned weight bit-identical across warm-up,
//! eviction, duplicates, signed zero, and ignored non-finite observations. Timings use paired
//! AB/BA rounds against an ORIG/ORIG A/A null floor.
//!
//! Run with:
//! ```bash
//! AGENT_NAME=cc_fse CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cod \
//!   rch exec -- cargo bench -p frankensearch-fusion --features bench-internals \
//!     --profile release --bench nqc_adaptive_cost_ab
//! ```
#![allow(clippy::doc_markdown)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use frankensearch_fusion::AdaptiveNqcDenseWeight;

const PROFILE_ROUNDS: usize = 41;
const PAIRED_ROUND_PAIRS: usize = 41;
const INNER: usize = 4096;

/// Production default config ([`AdaptiveNqcDenseWeight::production_default`]).
const BETA: f32 = 0.5;
const W_MIN: f32 = 0.5;
const CAPACITY: usize = 2048;
const MIN_SAMPLES: usize = 128;
const REBUILD_EVERY: usize = 64;

/// Deterministic xorshift → a stream of realistic NQC (cv) values in ~`[0, 1.5)`.
struct CvStream {
    state: u64,
}
impl CvStream {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_cv(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        // 0..1.5
        (self.state >> 40) as f32 / (1_u32 << 24) as f32 * 1.5
    }
}

#[derive(Clone, Copy)]
struct RatioDistribution {
    median: f64,
    p5: f64,
    p95: f64,
    round_pairs: usize,
}
impl RatioDistribution {
    fn null_contains_one(self) -> bool {
        self.p5 <= 1.0 && 1.0 <= self.p95
    }
    fn verdict_against(self, null: Self) -> &'static str {
        if !null.null_contains_one() {
            "BIASED_NULL_UNDECIDABLE"
        } else if self.median < null.p5 {
            "CANDIDATE_FASTER"
        } else if self.median > null.p95 {
            "CANDIDATE_SLOWER"
        } else {
            "INSIDE_NULL_FLOOR"
        }
    }
}

fn percentile(sorted: &[Duration], pct: usize) -> Duration {
    sorted[((sorted.len() - 1) * pct + 50) / 100]
}
fn ratio_distribution(mut samples: Vec<f64>) -> RatioDistribution {
    samples.sort_unstable_by(f64::total_cmp);
    let index = |pct: usize| ((samples.len() - 1) * pct + 50) / 100;
    RatioDistribution {
        median: samples[index(50)],
        p5: samples[index(5)],
        p95: samples[index(95)],
        round_pairs: samples.len(),
    }
}

/// One timed batch of INNER adaptive weightings (lookup + observe + periodic snapshot).
fn time_adaptive(adaptive: &mut AdaptiveNqcDenseWeight, stream: &mut CvStream) -> Duration {
    let started = Instant::now();
    let mut acc = 0.0_f32;
    for _ in 0..INNER {
        acc += adaptive.weight_for_cv(stream.next_cv());
    }
    let elapsed = started.elapsed();
    black_box(acc);
    elapsed
}

fn warm_adaptive(legacy: bool) -> AdaptiveNqcDenseWeight {
    let mut adaptive = if legacy {
        AdaptiveNqcDenseWeight::bench_legacy(BETA, W_MIN, CAPACITY, MIN_SAMPLES, REBUILD_EVERY)
    } else {
        AdaptiveNqcDenseWeight::new(BETA, W_MIN, CAPACITY, MIN_SAMPLES, REBUILD_EVERY)
    };
    // Drive it to steady state (full window + several rebuilds).
    let mut warm = CvStream::new(0x1234_5678_9ABC_DEF0);
    for _ in 0..(CAPACITY * 2) {
        black_box(adaptive.weight_for_cv(warm.next_cv()));
    }
    adaptive
}

fn median_ns_adaptive(adaptive: &mut AdaptiveNqcDenseWeight) -> (f64, f64) {
    let mut stream = CvStream::new(0xDEAD_BEEF_CAFE_1234);
    for _ in 0..3 {
        black_box(time_adaptive(adaptive, &mut stream));
    }
    let mut samples: Vec<Duration> = (0..PROFILE_ROUNDS)
        .map(|_| time_adaptive(adaptive, &mut stream))
        .collect();
    samples.sort_unstable();
    (
        percentile(&samples, 50).as_secs_f64() * 1e9 / INNER as f64,
        percentile(&samples, 95).as_secs_f64() * 1e9 / INNER as f64,
    )
}

fn paired_ratio(lever: bool) -> RatioDistribution {
    let mut original = warm_adaptive(true);
    let mut compared = warm_adaptive(!lever);
    let mut a_stream = CvStream::new(0x0F0F_0F0F_0F0F_0F0F);
    let mut b_stream = CvStream::new(0x0F0F_0F0F_0F0F_0F0F);
    let mut sample = |record: bool| {
        // AB then BA; ORIG=periodic full sort, CAND=incremental order (or ORIG for null).
        let ab_a = time_adaptive(&mut original, &mut a_stream);
        let ab_b = time_adaptive(&mut compared, &mut b_stream);
        let ba_b = time_adaptive(&mut compared, &mut b_stream);
        let ba_a = time_adaptive(&mut original, &mut a_stream);
        if record {
            let ab = ab_b.as_secs_f64() / ab_a.as_secs_f64();
            let ba = ba_b.as_secs_f64() / ba_a.as_secs_f64();
            Some((ab * ba).sqrt())
        } else {
            black_box((ab_a, ab_b, ba_b, ba_a));
            None
        }
    };
    for _ in 0..3 {
        let _ = sample(false);
    }
    ratio_distribution(
        (0..PAIRED_ROUND_PAIRS)
            .map(|_| sample(true).expect("round"))
            .collect(),
    )
}

fn assert_exact_parity() {
    let mut original = warm_adaptive(true);
    let mut candidate = warm_adaptive(false);
    let mut stream = CvStream::new(0xA11C_E5E5_1234_5678);
    for index in 0..(CAPACITY * 4) {
        let cv = if index % 521 == 0 {
            f32::NAN
        } else if index % 257 == 0 {
            f32::INFINITY
        } else if index % 113 == 0 {
            -0.0
        } else if index % 97 == 0 {
            0.5
        } else {
            stream.next_cv()
        };
        let expected = original.weight_for_cv(cv);
        let actual = candidate.weight_for_cv(cv);
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "weight bits diverged at observation {index}: original={expected:?} candidate={actual:?}"
        );
    }
    eprintln!(
        "[parity] exact weight bits across {} post-warm observations (eviction, duplicates, signed zero, non-finite)",
        CAPACITY * 4
    );
}

fn main() {
    eprintln!(
        "[config] beta={BETA} w_min={W_MIN} capacity={CAPACITY} min_samples={MIN_SAMPLES} rebuild_every={REBUILD_EVERY} inner={INNER}"
    );
    assert_exact_parity();
    let mut original = warm_adaptive(true);
    let mut candidate = warm_adaptive(false);

    let (original_med, original_p95) = median_ns_adaptive(&mut original);
    let (candidate_med, candidate_p95) = median_ns_adaptive(&mut candidate);
    eprintln!(
        "[profile] periodic_full_sort median_ns/q={original_med:.2} p95_ns/q={original_p95:.2}"
    );
    eprintln!(
        "[profile] incremental_order  median_ns/q={candidate_med:.2} p95_ns/q={candidate_p95:.2}"
    );
    eprintln!(
        "[saving] incremental-minus-original median_ns/q={:.2} p95_ns/q={:.2}",
        candidate_med - original_med,
        candidate_p95 - original_p95
    );

    let null = paired_ratio(false);
    let lever = paired_ratio(true);
    eprintln!(
        "[paired] null_original_original median={:.4} p5={:.4} p95={:.4} ({} pairs)",
        null.median, null.p5, null.p95, null.round_pairs
    );
    eprintln!(
        "[paired] incremental_vs_original median={:.4} p5={:.4} p95={:.4} ({} pairs)",
        lever.median, lever.p5, lever.p95, lever.round_pairs
    );
    eprintln!(
        "[verdict] {} candidate_original_ratio={:.4}x (<1 = incremental order is faster)",
        lever.verdict_against(null),
        lever.median
    );
    eprintln!(
        "[context] candidate saves ~{:.1} ns/query = {:.4}% of a 500us search",
        original_med - candidate_med,
        (original_med - candidate_med) / 500_000.0 * 100.0
    );
}
