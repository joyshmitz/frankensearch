//! Historical analytics percentile benchmark.
//!
//! Legacy ORIG sorted the entire telemetry vector to read one percentile.
//! Candidate keeps the exact same rank math and uses `select_nth_unstable`
//! to partition directly to the requested order statistic.
//!
//! Run with:
//! ```bash
//! AGENT_NAME=SearchCod CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cod \
//!   rch exec -- cargo bench -p frankensearch-ops --profile release --bench percentile_select
//! ```

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

fn percentile_index(len: usize, pct: u8) -> usize {
    let len_minus_one = len.saturating_sub(1);
    let pct_usize = usize::from(pct.min(100));
    let numerator = len_minus_one.saturating_mul(pct_usize).saturating_add(50);
    numerator.saturating_div(100).min(len_minus_one)
}

fn percentile_legacy_sort(values: &[u64], pct: u8) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted
        .get(percentile_index(sorted.len(), pct))
        .copied()
        .unwrap_or(0)
}

fn percentile_select(values: &[u64], pct: u8) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut selected = values.to_vec();
    let index = percentile_index(selected.len(), pct);
    let (_, value, _) = selected.select_nth_unstable(index);
    *value
}

fn build_latency_values(n: usize) -> Vec<u64> {
    let mut x = 0x9e37_79b9_7f4a_7c15_u64 ^ u64::try_from(n).unwrap_or(u64::MAX);
    (0..n)
        .map(|i| {
            x ^= x << 7;
            x ^= x >> 9;
            x ^= x << 8;
            let trend = (u64::try_from(i).unwrap_or(u64::MAX) % 257).saturating_mul(31);
            let burst = if i % 19 == 0 { 14_000 } else { 0 };
            (x % 4_096).saturating_add(trend).saturating_add(burst)
        })
        .collect()
}

fn bench_percentile_select(c: &mut Criterion) {
    let mut group = c.benchmark_group("ops_percentile_p95");
    for &n in &[64_usize, 512, 4_096, 16_384] {
        let values = build_latency_values(n);
        assert_eq!(
            percentile_legacy_sort(&values, 95),
            percentile_select(&values, 95)
        );

        group.bench_with_input(
            BenchmarkId::new("legacy_sort_ORIG", n),
            &values,
            |b, values| {
                b.iter(|| percentile_legacy_sort(black_box(values), black_box(95)));
            },
        );
        group.bench_with_input(BenchmarkId::new("select_nth", n), &values, |b, values| {
            b.iter(|| percentile_select(black_box(values), black_box(95)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_percentile_select);
criterion_main!(benches);
