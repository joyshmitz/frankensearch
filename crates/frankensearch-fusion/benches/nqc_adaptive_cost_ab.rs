//! Quantifies the per-query overhead the DEFAULT-ON `AdaptiveNqcDenseWeight` (ac081b7d) adds
//! beyond a bare static sketch lookup — i.e. the cost of the *online* machinery (rolling
//! `observe` + the every-`rebuild_every`-queries sketch rebuild), which `nqc_cv_cost_ab`
//! (static-path only) does not cover. Validates the shipped "latency-neutral" claim with a
//! number, and flags whether the rebuild (a `Vec<f32>` alloc + sort of the whole window every
//! 64 queries) is worth a buffer-reuse lever.
//!
//! Both arms are fed a PRECOMPUTED cv (so `nqc_cv` cancels out and only the weight machinery is
//! measured), over a warmed steady-state sketch:
//! - ORIG: `NqcDenseWeight::dense_weight(cv, β, w_min)` — a read-only `partition_point` lookup.
//! - CAND: `AdaptiveNqcDenseWeight::weight_for_cv(cv)` — the same lookup PLUS `observe` + the
//!   amortized periodic rebuild.
//! The CAND/ORIG median ratio is the overhead of the rolling adaptivity; p95 catches the rebuild
//! spikes. Within-process paired AB/BA against an A/A null floor.
//!
//! Run with:
//! ```bash
//! AGENT_NAME=cc_fse CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cod \
//!   rch exec -- cargo bench -p frankensearch-fusion --profile release --bench nqc_adaptive_cost_ab
//! ```
#![allow(clippy::doc_markdown)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use frankensearch_fusion::{AdaptiveNqcDenseWeight, NqcDenseWeight};

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

/// A warmed steady-state sketch: `CAPACITY` observed cvs, sorted.
fn warmed_sample() -> Vec<f32> {
    let mut stream = CvStream::new(0x9E37_79B9_7F4A_7C15);
    (0..CAPACITY).map(|_| stream.next_cv()).collect()
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
        } else if self.median > null.p95 {
            "CAND_COSTS_MORE"
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

/// One timed batch of INNER static lookups.
fn time_static(sketch: &NqcDenseWeight, stream: &mut CvStream) -> Duration {
    let started = Instant::now();
    let mut acc = 0.0_f32;
    for _ in 0..INNER {
        acc += sketch.dense_weight(stream.next_cv(), BETA, W_MIN);
    }
    let elapsed = started.elapsed();
    black_box(acc);
    elapsed
}

/// One timed batch of INNER adaptive weightings (lookup + observe + amortized rebuild).
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

fn warm_adaptive() -> AdaptiveNqcDenseWeight {
    let mut adaptive =
        AdaptiveNqcDenseWeight::new(BETA, W_MIN, CAPACITY, MIN_SAMPLES, REBUILD_EVERY);
    // Drive it to steady state (full window + several rebuilds).
    let mut warm = CvStream::new(0x1234_5678_9ABC_DEF0);
    for _ in 0..(CAPACITY * 2) {
        black_box(adaptive.weight_for_cv(warm.next_cv()));
    }
    adaptive
}

fn median_us_static(sketch: &NqcDenseWeight) -> (f64, f64) {
    let mut stream = CvStream::new(0xDEAD_BEEF_CAFE_1234);
    for _ in 0..3 {
        black_box(time_static(sketch, &mut stream));
    }
    let mut samples: Vec<Duration> = (0..PROFILE_ROUNDS)
        .map(|_| time_static(sketch, &mut stream))
        .collect();
    samples.sort_unstable();
    (
        percentile(&samples, 50).as_secs_f64() * 1e9 / INNER as f64,
        percentile(&samples, 95).as_secs_f64() * 1e9 / INNER as f64,
    )
}

fn median_us_adaptive(adaptive: &mut AdaptiveNqcDenseWeight) -> (f64, f64) {
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

fn paired_ratio(sketch: &NqcDenseWeight, adaptive: &mut AdaptiveNqcDenseWeight, lever: bool) -> RatioDistribution {
    let mut a_stream = CvStream::new(0x0F0F_0F0F_0F0F_0F0F);
    let mut b_stream = CvStream::new(0x00FF_00FF_00FF_00FF);
    let mut sample = |record: bool| {
        // AB then BA; ORIG=static, CAND=adaptive (or static for the A/A null).
        let ab_a = time_static(sketch, &mut a_stream);
        let ab_b = if lever {
            time_adaptive(adaptive, &mut b_stream)
        } else {
            time_static(sketch, &mut b_stream)
        };
        let ba_b = if lever {
            time_adaptive(adaptive, &mut b_stream)
        } else {
            time_static(sketch, &mut b_stream)
        };
        let ba_a = time_static(sketch, &mut a_stream);
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
    ratio_distribution((0..PAIRED_ROUND_PAIRS).map(|_| sample(true).expect("round")).collect())
}

fn main() {
    eprintln!(
        "[config] beta={BETA} w_min={W_MIN} capacity={CAPACITY} min_samples={MIN_SAMPLES} rebuild_every={REBUILD_EVERY} inner={INNER}"
    );
    let sketch = NqcDenseWeight::from_sample(&warmed_sample());
    let mut adaptive = warm_adaptive();

    let (static_med, static_p95) = median_us_static(&sketch);
    let (adaptive_med, adaptive_p95) = median_us_adaptive(&mut adaptive);
    eprintln!("[profile] static_lookup     median_ns/q={static_med:.2} p95_ns/q={static_p95:.2}");
    eprintln!("[profile] adaptive_weightfor median_ns/q={adaptive_med:.2} p95_ns/q={adaptive_p95:.2}");
    eprintln!(
        "[overhead] adaptivity_adds median_ns/q={:.2} p95_ns/q={:.2} (observe + amortized rebuild)",
        adaptive_med - static_med,
        adaptive_p95 - static_p95
    );

    let null = paired_ratio(&sketch, &mut adaptive, false);
    let lever = paired_ratio(&sketch, &mut adaptive, true);
    eprintln!(
        "[paired] null_static_static median={:.4} p5={:.4} p95={:.4} ({} pairs)",
        null.median, null.p5, null.p95, null.round_pairs
    );
    eprintln!(
        "[paired] adaptive_vs_static median={:.4} p5={:.4} p95={:.4} ({} pairs)",
        lever.median, lever.p5, lever.p95, lever.round_pairs
    );
    eprintln!(
        "[verdict] {} adaptivity_overhead_ratio={:.4}x (>1 = adaptive costs more than a bare lookup)",
        lever.verdict_against(null),
        lever.median
    );
    // Context: a hybrid search is ~hundreds of microseconds; the adaptivity overhead is per query.
    eprintln!(
        "[context] adaptivity adds ~{:.1} ns/query = {:.4}% of a 500us search",
        adaptive_med - static_med,
        (adaptive_med - static_med) / 500_000.0 * 100.0
    );
}
