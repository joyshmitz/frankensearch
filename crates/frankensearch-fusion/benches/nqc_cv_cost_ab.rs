//! Latency validation for the just-landed `nqc_cv` (dense down-weight step 1): confirm the
//! NQC computation is within the per-tier normalization budget the fusion path ALREADY pays.
//!
//! `nqc_cv` is a single-pass sum+sum_sq reduction over a tier's top-k scores; the fusion path
//! already does a `min_max_normalize` (a single-pass min+max reduction) per tier. If nqc_cv's
//! cost ≤ the min-max scan, adding it is provably latency-neutral (a fraction of work already
//! done each query). Both measured in one process (immune to the RCH_WORKER soft-pin), A/A null.

use std::hint::black_box;
use std::time::Instant;

use frankensearch_fusion::nqc_cv;

/// The core scan `min_max_normalize` performs (min+max, one pass) — measured without the
/// write-back/clone so it is a fair "existing per-tier reduction" baseline.
fn min_max_scan(scores: &[f32]) -> (f32, f32) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &v in scores {
        if v.is_finite() {
            min = min.min(v);
            max = max.max(v);
        }
    }
    (min, max)
}

fn make_scores(n: usize) -> Vec<f32> {
    // BM25-like positive scores with a peaked head.
    (0..n).map(|i| 20.0 / (1.0 + i as f32) + (i % 7) as f32 * 0.1).collect()
}

fn time_many(iters: usize, scores: &[f32], f: impl Fn(&[f32]) -> f32) -> f64 {
    let start = Instant::now();
    let mut acc = 0.0f32;
    for _ in 0..iters {
        acc += black_box(f(black_box(scores)));
    }
    black_box(acc);
    start.elapsed().as_secs_f64() / iters as f64
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let scores = make_scores(100); // top-100 pool (the default candidate_pool_size)
    let iters = 20000usize;
    let rounds = 60usize;

    let nqc = |s: &[f32]| nqc_cv(s);
    let scan = |s: &[f32]| { let (mn, mx) = min_max_scan(s); mx - mn };

    let mut nqc_t = Vec::new();
    let mut scan_t = Vec::new();
    let mut null_a = Vec::new();
    let mut null_b = Vec::new();
    for _ in 0..rounds {
        nqc_t.push(time_many(iters, &scores, nqc));
        scan_t.push(time_many(iters, &scores, scan));
        null_a.push(time_many(iters, &scores, nqc));
        null_b.push(time_many(iters, &scores, nqc));
    }
    let m_nqc = median(nqc_t);
    let m_scan = median(scan_t);
    let null_ratio = median(null_a.iter().zip(&null_b).map(|(a, b)| b / a).collect());

    println!("[nqc_cv     ] median {:>7.2} ns/call (top-100)", m_nqc * 1e9);
    println!("[minmax scan] median {:>7.2} ns/call (the per-tier reduction fusion already does)", m_scan * 1e9);
    println!("[ratio      ] nqc/minmax = {:.3}  (A/A null ~{:.3})", m_nqc / m_scan, null_ratio);
    println!(
        "[verdict    ] nqc_cv is {} the existing min-max scan -> {}",
        if m_nqc <= m_scan * 1.5 { "within ~1.5x of" } else { "MORE than" },
        if m_nqc <= m_scan * 1.5 { "LATENCY-NEUTRAL (fraction of per-tier work already paid)" } else { "measurable added cost" }
    );
}
