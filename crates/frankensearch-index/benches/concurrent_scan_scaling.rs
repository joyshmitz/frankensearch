//! NEW WORKLOAD PROXY: concurrent int8 flat-scan throughput scaling.
//!
//! The single-query µbenches (`fsvi_int8_two_pass`, `int8_two_pass`) measure ONE
//! thread scanning. Real serving runs many concurrent queries against the SAME
//! shared index. This bench spawns M threads that each flat-scan the same shared
//! int8 slab and measures how aggregate throughput scales with M.
//!
//! The question it answers (which the single-thread benches structurally cannot):
//! is the shared-index int8 scan **compute-bound** (scales ~linearly with cores —
//! the memory `frontier-exhausted` conclusion "int8 dominates 4-bit, AVX2 dot eats
//! 4-bit's bandwidth edge" holds under load) or **memory-bandwidth-bound** under
//! concurrency (throughput saturates — where 4-bit's half-footprint would win,
//! reversing the single-thread conclusion → a real lever)?
//!
//! Run: `rch exec -- cargo bench -p frankensearch-index --profile release
//!   --bench concurrent_scan_scaling`

use std::hint::black_box;
use std::time::Instant;

use frankensearch_index::dot_i8_i8;

const N: usize = 131_072; // 128k vectors
const DIM: usize = 384; // int8 slab = 48 MiB (>> L3, so bandwidth is in play)
const ROUNDS: usize = 40; // full scans per thread per measurement

fn make_slab() -> Vec<i8> {
    (0..N * DIM)
        .map(|i| {
            let raw = (i.wrapping_mul(2_654_435_761) % 251) as i64 - 125;
            raw as i8
        })
        .collect()
}

fn make_query() -> Vec<i8> {
    (0..DIM)
        .map(|i| ((i.wrapping_mul(97) % 251) as i64 - 125) as i8)
        .collect()
}

/// One full flat scan: dot the query against every stored vector; return a checksum
/// so the compiler cannot elide the work.
#[inline]
fn scan(slab: &[i8], query: &[i8]) -> i64 {
    let mut acc: i64 = 0;
    for v in 0..N {
        let row = &slab[v * DIM..(v + 1) * DIM];
        acc = acc.wrapping_add(i64::from(dot_i8_i8(row, query)));
    }
    acc
}

/// Median over `samples` of the per-round wall time when `threads` threads each run
/// `ROUNDS` scans concurrently. Returns aggregate throughput (scans/sec).
fn measure_throughput(slab: &[i8], query: &[i8], threads: usize, samples: usize) -> f64 {
    let mut per_sample: Vec<f64> = Vec::with_capacity(samples);
    for _ in 0..samples {
        let started = Instant::now();
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| {
                    let mut acc = 0i64;
                    for _ in 0..ROUNDS {
                        acc = acc.wrapping_add(scan(slab, query));
                    }
                    black_box(acc);
                });
            }
        });
        let elapsed = started.elapsed();
        let total_scans = (threads * ROUNDS) as f64;
        per_sample.push(total_scans / elapsed.as_secs_f64());
    }
    per_sample.sort_unstable_by(f64::total_cmp);
    per_sample[per_sample.len() / 2]
}

fn main() {
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(4);
    let slab = make_slab();
    let query = make_query();
    let slab_mib = (N * DIM) as f64 / (1024.0 * 1024.0);
    eprintln!(
        "[profile-config] n={N} dim={DIM} rounds={ROUNDS} int8_slab_mib={slab_mib:.1} available_cores={cores}"
    );
    eprintln!(
        "[profile-config] binary_path={}",
        std::env::current_exe()
            .expect("resolve measured binary")
            .display()
    );

    // Warm up + establish the single-thread baseline.
    black_box(scan(&slab, &query));
    let base = measure_throughput(&slab, &query, 1, 7);
    eprintln!("[baseline] threads=1 scans_per_sec={base:.2}");

    let thread_counts: Vec<usize> = [2usize, 4, 8, cores]
        .into_iter()
        .filter(|&m| m > 1 && m <= cores.max(2))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    let mut min_efficiency = 1.0f64;
    for &m in &thread_counts {
        let agg = measure_throughput(&slab, &query, m, 7);
        let ideal = base * m as f64;
        let efficiency = agg / ideal;
        min_efficiency = min_efficiency.min(efficiency);
        eprintln!(
            "[scaling] threads={m} agg_scans_per_sec={agg:.2} ideal={ideal:.2} efficiency={efficiency:.4}"
        );
    }

    // Verdict: <~0.7 at high thread counts ⇒ a shared resource (memory bandwidth)
    // is saturating, so 4-bit's half-footprint could win under concurrency (lever);
    // ~1.0 ⇒ compute-bound, scales with cores, single-thread int8 conclusion holds.
    let verdict = if min_efficiency < 0.70 {
        "BANDWIDTH_BOUND_UNDER_CONCURRENCY (4-bit lever plausible)"
    } else if min_efficiency < 0.90 {
        "MILD_CONTENTION"
    } else {
        "COMPUTE_BOUND_SCALES (int8 conclusion holds under load)"
    };
    eprintln!("[verdict] min_scaling_efficiency={min_efficiency:.4} => {verdict}");
}
