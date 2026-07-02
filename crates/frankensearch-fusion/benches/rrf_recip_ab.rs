//! Ceiling measurement for the RRF reciprocal-LUT lever.
//!
//! `rank_contribution(k, rank) = 1.0 / (k + rank + 1.0)` is a float DIVIDE called
//! per candidate per source in the fuse loop (rrf.rs:464/488). At `limit_all` that
//! is ~2–3·N divides per query. A precomputed reciprocal table (indexed by rank,
//! computed once for the fixed `rrf_k`, reused across queries) replaces each
//! ~10–20-cycle divide with a ~1-cycle lookup — bit-identical if the table stores
//! the same computed reciprocals.
//!
//! This isolates the CEILING: the raw cost of N divides vs N LUT lookups (the fuse
//! loop does more work, so the real-fuse win is ≤ this). If this delta is small,
//! the lever is dead (the loop's divides are already hidden by out-of-order exec
//! behind the merge/store work); if large, it justifies wiring the LUT into the fuse.
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-fusion --bench rrf_recip_ab
//! ```

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

const K: f64 = 60.0;

#[inline]
fn rank_contribution(k: f64, rank: usize) -> f64 {
    let rank_u32 = u32::try_from(rank).unwrap_or(u32::MAX);
    1.0 / (k + f64::from(rank_u32) + 1.0)
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("rrf_recip_ab");
    for n in [10_000usize, 100_000] {
        // ~2 sources per candidate at limit_all (lexical + semantic rank).
        let ranks: Vec<usize> = (0..n).flat_map(|i| [i, i / 2]).collect();

        // Current: a divide per contribution.
        g.bench_with_input(BenchmarkId::new("divide", n), &ranks, |b, ranks| {
            b.iter(|| {
                let mut acc = 0.0_f64;
                for &r in black_box(ranks) {
                    acc += rank_contribution(K, r);
                }
                black_box(acc)
            });
        });

        // Candidate: precomputed reciprocal LUT (built ONCE outside the timed loop,
        // amortized across queries in production), indexed by rank. Bit-identical:
        // `lut[r]` == `rank_contribution(K, r)`.
        let max_rank = ranks.iter().copied().max().unwrap_or(0);
        let lut: Vec<f64> = (0..=max_rank).map(|r| rank_contribution(K, r)).collect();
        g.bench_with_input(BenchmarkId::new("lut", n), &ranks, |b, ranks| {
            b.iter(|| {
                let mut acc = 0.0_f64;
                for &r in black_box(ranks) {
                    acc += black_box(&lut)[r];
                }
                black_box(acc)
            });
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
