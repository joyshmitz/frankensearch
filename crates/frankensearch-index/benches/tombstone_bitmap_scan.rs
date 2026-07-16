//! int8 pass-1 scan: LEGACY strided per-record tombstone flag vs a SUCCINCT
//! contiguous tombstone BITMAP (1 bit/vector) with all-dead-word skipping.
//!
//! `search.rs::int8_scan_range` reads each candidate's live/tombstone flag as a
//! 2-byte field at offset 14 of its 16-byte record in the `records` region — a
//! SECOND memory stream, strided 16 bytes apart, separate from the contiguous int8
//! slab. Structurally-different primitive (succinct-structure / data-layout): replace
//! that per-record flag read with a contiguous 1-bit-per-vector bitmap (~N/8 bytes,
//! L1/L2-resident), which (a) removes the second strided stream (TLB / hw-prefetcher
//! pressure) and (b) lets an all-dead u64 word skip 64 candidates' dots at once.
//!
//! Both arms compute the IDENTICAL cutoff-gated top-K over the SAME live set (parity
//! asserted). Two tombstone regimes: `scattered` (~1%, freshly-compacted index — the
//! common case, no word-skip possible) and `clustered` (25% in contiguous runs — a
//! partially-compacted / batch-deleted index, where all-dead words let the bitmap skip).
//!
//! Uses the real `frankensearch_index::simd::dot_i8_i8` (AVX2) so the dot cost is
//! identical to production; only the flag-access primitive differs between arms.
//!
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-index --profile release --bench tombstone_bitmap_scan
//! ```

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_core::bench_support::paired_median_ratio;
use frankensearch_index::simd::dot_i8_i8;

const DIM: usize = 384;
const K: usize = 100;
const RECORD_STRIDE: usize = 16; // bytes/record in the `records` region
const FLAG_OFFSET: usize = 14; // low byte of the u16 flags field

/// Bounded cutoff-gated top-K, mirroring `int8_scan_range`: the dot is computed for
/// every LIVE candidate, then a `>= cutoff` gate avoids the heap push for losers.
/// Returns the top-K as (score, index) sorted desc with index tie-break.
#[inline]
fn push_candidate(
    heap: &mut BinaryHeap<Reverse<(i32, usize)>>,
    cutoff: &mut i32,
    s: i32,
    j: usize,
) {
    if heap.len() < K || s > *cutoff {
        heap.push(Reverse((s, j)));
        if heap.len() > K {
            heap.pop();
        }
        if heap.len() >= K {
            *cutoff = heap.peek().unwrap().0.0;
        }
    }
}

fn finish(heap: BinaryHeap<Reverse<(i32, usize)>>) -> Vec<(i32, usize)> {
    let mut out: Vec<(i32, usize)> = heap.into_iter().map(|r| r.0).collect();
    out.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    out
}

// ── LEGACY: strided per-record 2-byte flag read (mirrors int8_scan_range) ─────
fn scan_records_flag(slab: &[i8], records: &[u8], query: &[i8], n: usize) -> Vec<(i32, usize)> {
    let mut heap: BinaryHeap<Reverse<(i32, usize)>> = BinaryHeap::with_capacity(K + 1);
    let mut cutoff = i32::MIN;
    for j in 0..n {
        let flag = records[j * RECORD_STRIDE + FLAG_OFFSET];
        if (flag & 0x01) == 0 {
            let s = dot_i8_i8(&slab[j * DIM..j * DIM + DIM], query);
            push_candidate(&mut heap, &mut cutoff, s, j);
        }
    }
    finish(heap)
}

// ── SUCCINCT: contiguous 1-bit/vector bitmap (1 = live), all-dead-word skip ────
fn scan_bitmap(slab: &[i8], bitmap: &[u64], query: &[i8], n: usize) -> Vec<(i32, usize)> {
    let mut heap: BinaryHeap<Reverse<(i32, usize)>> = BinaryHeap::with_capacity(K + 1);
    let mut cutoff = i32::MIN;
    let words = n.div_ceil(64);
    for w in 0..words {
        let mut bits = bitmap[w];
        if bits == 0 {
            continue; // all 64 candidates tombstoned — skip their dots entirely
        }
        let base = w * 64;
        // Iterate only the set (live) bits via trailing-zeros scan.
        while bits != 0 {
            let b = bits.trailing_zeros() as usize;
            let j = base + b;
            if j < n {
                let s = dot_i8_i8(&slab[j * DIM..j * DIM + DIM], query);
                push_candidate(&mut heap, &mut cutoff, s, j);
            }
            bits &= bits - 1; // clear lowest set bit
        }
    }
    finish(heap)
}

/// Deterministic xorshift for reproducible fixtures (Date/rand unavailable in benches).
struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn build_fixtures(n: usize, clustered: bool) -> (Vec<i8>, Vec<u8>, Vec<u64>, Vec<i8>) {
    let mut rng = XorShift(0x2545_f491_4f6c_dd1d);
    // Clustered-ish int8 embeddings: small centered range, a few shared centroids.
    let mut slab = vec![0_i8; n * DIM];
    for v in slab.iter_mut() {
        *v = ((rng.next() >> 40) as i32 % 40 - 20) as i8;
    }
    let mut query = vec![0_i8; DIM];
    for v in query.iter_mut() {
        *v = ((rng.next() >> 40) as i32 % 40 - 20) as i8;
    }

    let mut records = vec![0_u8; n * RECORD_STRIDE];
    let mut bitmap = vec![0_u64; n.div_ceil(64)]; // 1 = live
    let mut tombstoned = vec![false; n];
    if clustered {
        // 25% tombstoned in contiguous runs of ~192 (spans multiple full u64 words).
        let mut j = 0;
        while j < n {
            let run = 128 + (rng.next() as usize % 256);
            let dead = (rng.next() % 100) < 25; // ~25% of runs are dead runs
            for _ in 0..run {
                if j >= n {
                    break;
                }
                tombstoned[j] = dead;
                j += 1;
            }
        }
    } else {
        // ~1% scattered tombstones (freshly-compacted common case).
        for t in tombstoned.iter_mut() {
            *t = (rng.next() % 100) == 0;
        }
    }
    for (j, &dead) in tombstoned.iter().enumerate() {
        if dead {
            records[j * RECORD_STRIDE + FLAG_OFFSET] = 0x01;
        } else {
            bitmap[j / 64] |= 1_u64 << (j % 64);
        }
    }
    (slab, records, bitmap, query)
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("tombstone_bitmap_scan");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(400));
    group.measurement_time(Duration::from_millis(1500));

    let n = 100_000;
    for &(name, clustered) in &[("scattered", false), ("clustered", true)] {
        let (slab, records, bitmap, query) = build_fixtures(n, clustered);
        // Parity: identical top-K from both flag-access strategies.
        let a = scan_records_flag(&slab, &records, &query, n);
        let b = scan_bitmap(&slab, &bitmap, &query, n);
        assert!(
            a == b,
            "bitmap scan diverged from records scan ({name}) — must match"
        );

        // DRIFT-CANCELLED interleaved paired A/B (bd-6m8p). The criterion arms below run
        // SEQUENTIALLY (records_flag fully, then bitmap fully), so worker frequency/thermal drift
        // between the two full measurements biases their ratio. `paired_median_ratio` interleaves
        // both arms per round and times only its own, cancelling drift, and an A/A null control
        // (flag/flag) gives the noise floor the bitmap/flag ratio must clear to be a real effect.
        let run_flag = || {
            black_box(scan_records_flag(
                black_box(&slab),
                black_box(&records),
                black_box(&query),
                n,
            ));
        };
        let run_bitmap = || {
            black_box(scan_bitmap(
                black_box(&slab),
                black_box(&bitmap),
                black_box(&query),
                n,
            ));
        };
        let null = paired_median_ratio(41, 4, run_flag, run_flag);
        let lever = paired_median_ratio(41, 4, run_flag, run_bitmap);
        eprintln!(
            "[paired-null  {name}] flag/flag   median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
            null.median, null.p5, null.p95, null.rounds
        );
        eprintln!(
            "[paired-lever {name}] bitmap/flag median {:.4} p5 {:.4} p95 {:.4} -> {} (<1 ⇒ bitmap faster)",
            lever.median,
            lever.p5,
            lever.p95,
            if lever.decidable_against(&null) {
                "DECIDABLE"
            } else {
                "INSIDE NULL FLOOR"
            }
        );

        group.bench_with_input(BenchmarkId::new("records_flag", name), &(), |bn, ()| {
            bn.iter(|| {
                black_box(scan_records_flag(
                    black_box(&slab),
                    black_box(&records),
                    black_box(&query),
                    n,
                ))
            });
        });
        group.bench_with_input(BenchmarkId::new("bitmap", name), &(), |bn, ()| {
            bn.iter(|| {
                black_box(scan_bitmap(
                    black_box(&slab),
                    black_box(&bitmap),
                    black_box(&query),
                    n,
                ))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
