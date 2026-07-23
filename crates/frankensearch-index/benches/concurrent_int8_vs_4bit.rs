//! CONFIRM the concurrent 4-bit lever (route-next from `f027a193`).
//!
//! `concurrent_scan_scaling` showed the int8 flat scan is memory-bandwidth-bound under
//! concurrency (efficiency 0.53 @ 8 cores). This bench scans the SAME shared corpus in
//! BOTH int8 (48 MiB, `dot_i8_i8`) and 4-bit (24 MiB packed, `dot_4bit_prepared`)
//! layouts at 1/2/4/8 threads and reports each layout's scaling efficiency AND absolute
//! aggregate throughput. Prediction: at 8 cores int8 saturates (~0.53) while 4-bit's
//! half-footprint stays higher, so 4-bit becomes the aggregate-throughput winner for
//! concurrent serving — reversing the single-thread ship-int8-not-4bit call for that
//! regime.
//!
//! Run: `rch exec -- cargo bench -p frankensearch-index --profile release
//!   --bench concurrent_int8_vs_4bit`
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::hint::black_box;
use std::time::Instant;

use frankensearch_index::{PreparedQuery4bit, dot_4bit_prepared, dot_i8_i8, prepare_4bit_query};

const N: usize = 131_072;
const DIM: usize = 384;
const ROUNDS: usize = 40;
const SAMPLES: usize = 7;

fn gen_f32(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    // map to roughly [-1, 1]
    ((*state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

fn quantize_i8(x: f32) -> i8 {
    let s = (x * 127.0).round();
    s.clamp(-127.0, 127.0) as i8
}

fn quantize_4bit(x: f32) -> i8 {
    let s = (x * 7.0).round();
    s.clamp(-7.0, 7.0) as i8
}

fn pack_4bit_vector(v: &[f32]) -> Vec<u8> {
    v.chunks(2)
        .map(|pair| {
            let lo = quantize_4bit(pair[0]) as u8 & 0x0f;
            let hi = pair
                .get(1)
                .map_or(0, |x| (quantize_4bit(*x) as u8 & 0x0f) << 4);
            lo | hi
        })
        .collect()
}

/// Contiguous int8 slab (N*DIM) and packed 4-bit slab (N*DIM/2), plus the queries.
struct Corpus {
    slab_i8: Vec<i8>,
    query_i8: Vec<i8>,
    slab_4bit: Vec<u8>,
    q4: PreparedQuery4bit,
    packed_per_vec: usize,
}

fn build() -> Corpus {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let query_f32: Vec<f32> = (0..DIM).map(|_| gen_f32(&mut state)).collect();
    let query_i8: Vec<i8> = query_f32.iter().copied().map(quantize_i8).collect();
    let q4 = prepare_4bit_query(&pack_4bit_vector(&query_f32));

    let packed_per_vec = DIM.div_ceil(2);
    let mut slab_i8 = Vec::with_capacity(N * DIM);
    let mut slab_4bit = Vec::with_capacity(N * packed_per_vec);
    for _ in 0..N {
        let v: Vec<f32> = (0..DIM).map(|_| gen_f32(&mut state)).collect();
        slab_i8.extend(v.iter().copied().map(quantize_i8));
        slab_4bit.extend(pack_4bit_vector(&v));
    }
    Corpus {
        slab_i8,
        query_i8,
        slab_4bit,
        q4,
        packed_per_vec,
    }
}

#[inline]
fn scan_i8(slab: &[i8], query: &[i8]) -> i64 {
    let mut acc = 0i64;
    for v in 0..N {
        acc = acc.wrapping_add(i64::from(dot_i8_i8(&slab[v * DIM..(v + 1) * DIM], query)));
    }
    acc
}

#[inline]
fn scan_4bit(slab: &[u8], q4: &PreparedQuery4bit, per: usize) -> i64 {
    let mut acc = 0i64;
    for v in 0..N {
        acc = acc.wrapping_add(i64::from(dot_4bit_prepared(
            &slab[v * per..(v + 1) * per],
            q4,
        )));
    }
    acc
}

/// Median aggregate throughput (scans/sec) with `threads` concurrent workers.
fn throughput(threads: usize, round_fn: impl Fn() -> i64 + Copy + Send + Sync) -> f64 {
    let mut samples: Vec<f64> = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let started = Instant::now();
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(move || {
                    let mut acc = 0i64;
                    for _ in 0..ROUNDS {
                        acc = acc.wrapping_add(round_fn());
                    }
                    black_box(acc);
                });
            }
        });
        let secs = started.elapsed().as_secs_f64();
        samples.push((threads * ROUNDS) as f64 / secs);
    }
    samples.sort_unstable_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn main() {
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(4);
    let c = build();
    eprintln!(
        "[profile-config] n={N} dim={DIM} rounds={ROUNDS} int8_mib={:.1} fourbit_mib={:.1} cores={cores}",
        (N * DIM) as f64 / (1024.0 * 1024.0),
        (N * c.packed_per_vec) as f64 / (1024.0 * 1024.0)
    );

    let i8_round = || scan_i8(&c.slab_i8, &c.query_i8);
    let b4_round = || scan_4bit(&c.slab_4bit, &c.q4, c.packed_per_vec);
    black_box(i8_round());
    black_box(b4_round());

    let i8_base = throughput(1, i8_round);
    let b4_base = throughput(1, b4_round);
    eprintln!(
        "[baseline] int8_scans_per_sec={i8_base:.1} fourbit_scans_per_sec={b4_base:.1} fourbit_over_int8_1thread={:.4}",
        b4_base / i8_base
    );

    let counts: Vec<usize> = [2usize, 4, 8, cores]
        .into_iter()
        .filter(|&m| m > 1 && m <= cores.max(2))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    for &m in &counts {
        let i8_agg = throughput(m, i8_round);
        let b4_agg = throughput(m, b4_round);
        let i8_eff = i8_agg / (i8_base * m as f64);
        let b4_eff = b4_agg / (b4_base * m as f64);
        eprintln!(
            "[scaling] threads={m} int8_agg={i8_agg:.1} int8_eff={i8_eff:.4} | fourbit_agg={b4_agg:.1} fourbit_eff={b4_eff:.4} | fourbit_over_int8_agg={:.4}",
            b4_agg / i8_agg
        );
    }

    // Verdict at the highest thread count.
    let m = *counts.last().unwrap_or(&1);
    let i8_agg = throughput(m, i8_round);
    let b4_agg = throughput(m, b4_round);
    let verdict = if b4_agg > i8_agg * 1.05 {
        "FOURBIT_WINS_CONCURRENT (bandwidth lever confirmed — 4-bit for concurrent serving)"
    } else if b4_agg > i8_agg * 0.95 {
        "PARITY_CONCURRENT (4-bit closes the single-thread gap under load)"
    } else {
        "INT8_STILL_WINS (bandwidth not the binding constraint at these cores)"
    };
    eprintln!("[verdict] threads={m} int8_agg={i8_agg:.1} fourbit_agg={b4_agg:.1} => {verdict}");
}
