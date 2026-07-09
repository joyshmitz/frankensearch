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
use frankensearch_ops::FrameQualityTracker;

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

#[derive(Clone)]
struct LegacyFrameQualityTracker {
    window: usize,
    samples: Vec<u16>,
    cursor: usize,
    total_frames: u64,
}

impl LegacyFrameQualityTracker {
    fn new(window: usize) -> Self {
        let window = window.max(1);
        Self {
            window,
            samples: vec![0; window],
            cursor: 0,
            total_frames: 0,
        }
    }

    fn record(&mut self, duration_ms: u16) {
        self.samples[self.cursor] = duration_ms;
        self.cursor = (self.cursor + 1) % self.window;
        self.total_frames += 1;
    }

    fn p95_frame_time_ms(&self) -> u16 {
        let window_u64 = u64::try_from(self.window).unwrap_or(u64::MAX);
        let active = usize::try_from(self.total_frames.min(window_u64)).unwrap_or(self.window);
        if active == 0 {
            return 0;
        }
        let mut sorted: Vec<u16> = self.samples[..active].to_vec();
        sorted.sort_unstable();
        let idx = active.saturating_mul(95).div_ceil(100);
        sorted[idx.min(active - 1)]
    }
}

fn build_frame_durations(window: usize) -> Vec<u16> {
    let mut x = 0xd1b5_4a32_d192_ed03_u64 ^ u64::try_from(window).unwrap_or(u64::MAX);
    (0..(window * 3))
        .map(|i| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let jitter = u16::try_from(x % 11).unwrap_or(0);
            if i % 251 == 0 {
                300 + jitter
            } else if i % 19 == 0 {
                34 + jitter
            } else if i % 7 == 0 {
                18 + jitter
            } else {
                10 + jitter
            }
        })
        .collect()
}

fn bench_frame_quality_p95(c: &mut Criterion) {
    let mut group = c.benchmark_group("ops_frame_quality_p95");
    for &window in &[64_usize, 128, 512, 2_048] {
        let durations = build_frame_durations(window);
        let mut legacy = LegacyFrameQualityTracker::new(window);
        let mut current = FrameQualityTracker::new(window);
        for duration_ms in durations {
            legacy.record(duration_ms);
            current.record(duration_ms);
        }
        assert_eq!(legacy.p95_frame_time_ms(), current.p95_frame_time_ms());

        group.bench_with_input(
            BenchmarkId::new("legacy_sort_ORIG", window),
            &legacy,
            |b, tracker| {
                b.iter(|| black_box(tracker.p95_frame_time_ms()));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("histogram", window),
            &current,
            |b, tracker| {
                b.iter(|| black_box(tracker.p95_frame_time_ms()));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_percentile_select, bench_frame_quality_p95);
criterion_main!(benches);
