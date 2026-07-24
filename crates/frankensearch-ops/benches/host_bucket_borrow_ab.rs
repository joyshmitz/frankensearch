//! Within-process paired A/B for the timeline `host_bucket` allocation elision.
//!
//! `host_bucket(instance_id) -> String` used to `.to_owned()` a subslice of the input;
//! in the hot `filtered_events` filter loop the result is only compared
//! (`eq_ignore_ascii_case`) and dropped — a throwaway heap allocation per event. The
//! landed change returns a borrowed `&str` (all three branches are subslices), eliding
//! the allocation. This bench reproduces that filter-site workload for both arms in one
//! process (immune to the fleet's `RCH_WORKER` soft-pin: both arms share one core), with
//! an A/A null floor. Verdict `CANDIDATE_FASTER` iff the borrowed median < the null p5.

use std::hint::black_box;
use std::time::Instant;

/// ORIG: allocate an owned prefix (what production did before the change).
#[inline]
fn host_bucket_owned(instance_id: &str) -> String {
    if let Some((host, _)) = instance_id.split_once(':') {
        return host.to_owned();
    }
    if let Some((host, _)) = instance_id.split_once('-') {
        return host.to_owned();
    }
    instance_id.to_owned()
}

/// NEW: return a borrowed subslice (what production does now).
#[inline]
fn host_bucket_borrowed(instance_id: &str) -> &str {
    if let Some((host, _)) = instance_id.split_once(':') {
        return host;
    }
    if let Some((host, _)) = instance_id.split_once('-') {
        return host;
    }
    instance_id
}

/// Representative fleet instance ids: `host-<n>:instance-<m>` (colon form dominates),
/// with a fraction of dash-only and separatorless ids to exercise all three branches.
fn make_ids(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| match i % 8 {
            7 => format!("standalone{i}"),
            6 => format!("server-{i}"),
            _ => format!("host-{}:instance-{i}", i % 32),
        })
        .collect()
}

/// The filter-site workload: bucket every id and compare it to the active host filter,
/// counting matches (so the comparison — and thus the alloc in the owned arm — is live).
fn run_owned(ids: &[String], host_filter: &str) -> usize {
    ids.iter()
        .filter(|id| host_bucket_owned(id).eq_ignore_ascii_case(host_filter))
        .count()
}

fn run_borrowed(ids: &[String], host_filter: &str) -> usize {
    ids.iter()
        .filter(|id| host_bucket_borrowed(id).eq_ignore_ascii_case(host_filter))
        .count()
}

fn time_many(iters: usize, mut f: impl FnMut() -> usize) -> f64 {
    let start = Instant::now();
    let mut acc = 0usize;
    for _ in 0..iters {
        acc = acc.wrapping_add(black_box(f()));
    }
    black_box(acc);
    start.elapsed().as_secs_f64() / iters as f64
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn p5(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 20]
}

fn main() {
    let n = 2000usize;
    let ids = make_ids(n);
    let host_filter = "host-3"; // a bucket that matches ~1/32 of the colon ids
    let iters = 400usize;
    let rounds = 60usize;

    // Warm up + sanity: both arms must count identically (byte-identical behavior).
    let c_owned = run_owned(&ids, host_filter);
    let c_borrowed = run_borrowed(&ids, host_filter);
    assert_eq!(c_owned, c_borrowed, "arms disagree — not byte-identical");
    println!("[sanity] n={n} matches={c_owned} (arms agree)");

    let mut owned = Vec::new();
    let mut borrowed = Vec::new();
    let mut null_a = Vec::new();
    let mut null_b = Vec::new();
    // Interleave arms round-robin so slow drift hits both equally.
    for _ in 0..rounds {
        owned.push(time_many(iters, || run_owned(&ids, host_filter)));
        borrowed.push(time_many(iters, || run_borrowed(&ids, host_filter)));
        // A/A null: same arm twice → the noise floor of "median difference".
        null_a.push(time_many(iters, || run_borrowed(&ids, host_filter)));
        null_b.push(time_many(iters, || run_borrowed(&ids, host_filter)));
    }

    let m_owned = median(owned.clone());
    let m_borrowed = median(borrowed.clone());
    // Null floor: the spread between two identical arms (b vs a), as a ratio.
    let null_ratio_med = median(
        null_a
            .iter()
            .zip(&null_b)
            .map(|(a, b)| b / a)
            .collect::<Vec<_>>(),
    );
    let null_p5 = p5(null_a.iter().zip(&null_b).map(|(a, b)| b / a).collect());

    let speedup = m_owned / m_borrowed; // >1 means borrowed is faster
    let lever_ratio = m_borrowed / m_owned; // <1 means borrowed faster

    println!(
        "[owned   ] median {:>9.2} ns/call",
        m_owned * 1e9 / n as f64
    );
    println!(
        "[borrowed] median {:>9.2} ns/call",
        m_borrowed * 1e9 / n as f64
    );
    println!("[speedup ] owned/borrowed = {speedup:.4}x  (borrowed faster if >1)");
    println!(
        "[null A/A] median ratio {null_ratio_med:.4}  p5 {null_p5:.4}  (noise floor of a null diff)"
    );
    // Verdict: borrowed is a real win iff its advantage exceeds the A/A noise floor.
    let verdict = if lever_ratio < null_p5 {
        "CANDIDATE_FASTER (lever ratio beats A/A p5 floor)"
    } else {
        "INCONCLUSIVE (within A/A noise)"
    };
    println!("[lever   ] borrowed/owned = {lever_ratio:.4}  vs null p5 {null_p5:.4} -> {verdict}");
}
