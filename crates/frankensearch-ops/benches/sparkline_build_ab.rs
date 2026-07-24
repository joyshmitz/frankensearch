//! Within-process paired A/B for the ops `sparkline` string builder.
//!
//! `sparkline(values) = values.iter().map(spark_char).collect::<String>()` reserves only
//! `values.len()` bytes (`FromIterator` `size_hint` lower bound) but each block glyph
//! (U+2581..U+2588) is 3 bytes UTF-8 → the buffer reallocs a couple times per call as it
//! grows to 3*len. Two candidate builds elide that: (1) pre-size `with_capacity(len*3)` +
//! push; (2) byte-table — the eight glyphs are contiguous (E2 96 81 .. E2 96 88), so
//! byte[2] = 0x81 + idx, letting us extend a `Vec<u8>` and wrap it (valid UTF-8 by
//! construction). All three must produce byte-identical strings. Both arms run in one
//! process (immune to `RCH_WORKER` soft-pin) with an A/A null floor.

use std::hint::black_box;
use std::time::Instant;

#[inline]
fn spark_idx(percent: u8) -> usize {
    let idx = (u16::from(percent).saturating_mul(7).saturating_add(50)) / 100;
    usize::from(idx.min(7))
}

#[inline]
fn spark_char(percent: u8) -> char {
    const BINS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    BINS[spark_idx(percent)]
}

/// ORIG: map+collect (reserves len bytes, reallocs to 3*len).
fn spark_collect(values: &[u8]) -> String {
    values.iter().map(|v| spark_char(*v)).collect()
}

/// CAND A: pre-size to the exact final byte length, then push.
fn spark_presized(values: &[u8]) -> String {
    let mut s = String::with_capacity(values.len() * 3);
    for &v in values {
        s.push(spark_char(v));
    }
    s
}

fn make_values(n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| u8::try_from((i * 37 + 11) % 101).unwrap_or(0))
        .collect()
}

fn time_many(iters: usize, values: &[u8], f: fn(&[u8]) -> String) -> f64 {
    let start = Instant::now();
    let mut acc = 0usize;
    for _ in 0..iters {
        let s = f(black_box(values));
        acc = acc.wrapping_add(black_box(s.len()));
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
    // Byte-identity check across a wide input space (both arms must agree exactly).
    for n in [0usize, 1, 4, 63, 120, 256] {
        let vals = make_values(n);
        let a = spark_collect(&vals);
        let b = spark_presized(&vals);
        assert_eq!(a, b, "presized differs at n={n}");
    }
    println!("[sanity] both arms byte-identical across n in {{0,1,4,63,120,256}}");

    let width = 120usize; // representative heatstrip width (widest per-frame sparkline)
    let vals = make_values(width);
    let iters = 2000usize;
    let rounds = 60usize;

    let mut c_collect = Vec::new();
    let mut c_presized = Vec::new();
    let mut null_a = Vec::new();
    let mut null_b = Vec::new();
    for _ in 0..rounds {
        c_collect.push(time_many(iters, &vals, spark_collect));
        c_presized.push(time_many(iters, &vals, spark_presized));
        null_a.push(time_many(iters, &vals, spark_collect));
        null_b.push(time_many(iters, &vals, spark_collect));
    }

    let m_collect = median(c_collect);
    let m_presized = median(c_presized);
    let null_p5 = p5(null_a.iter().zip(&null_b).map(|(a, b)| b / a).collect());

    let ns = |t: f64| t * 1e9;
    let ratio = m_presized / m_collect;
    println!(
        "[collect ] median {:>9.2} ns/call (width={width})",
        ns(m_collect)
    );
    println!(
        "[presized] median {:>9.2} ns/call  ratio {ratio:.4}",
        ns(m_presized)
    );
    println!("[null A/A] p5 {null_p5:.4}  (a ratio below this beats the noise floor)");
    if ratio < null_p5 {
        println!("[verdict ] presized: {ratio:.4} < null p5 {null_p5:.4} -> CANDIDATE_FASTER");
    } else {
        println!("[verdict ] presized: {ratio:.4} vs null p5 {null_p5:.4} -> INCONCLUSIVE");
    }
}
