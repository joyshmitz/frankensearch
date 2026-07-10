//! Isolated latency A/B (scan lever): FMA f16 dot vs the shipped mul+add f16 dot.
//!
//! The f16 flat scan (`scan_range_chunk`) scores each vector with `dot_product_f16_bytes_f32`, whose
//! SIMD body does a separate `_mm256_mul_ps` + `_mm256_add_ps` per chunk. `_mm256_fmadd_ps` fuses
//! those into one op. Whether that helps depends on whether the kernel is FP-port-bound or purely
//! `cvtph2ps`-decode-bound — this bench decides it, since profiling is blocked (perf only on one
//! worker, rch can't pin; bd-e41k). FMA is sub-ULP-different but order-preserving (gated in
//! `simd.rs::fma_f16_dot_is_ulp_close_and_order_preserving`).
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- cargo bench -p frankensearch-index --features bench-internals --bench f16_dot_fma_ab
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_core::bench_support::paired_median_ratio;
use frankensearch_index::{dot_product_f16_bytes_f32, dot_product_f16_bytes_f32_fma};

const DIM: usize = 384;
const BATCH: usize = 4096;

fn f16_bytes(seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    let mut f: Vec<f32> = (0..DIM)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / 1e6) - 0.5
        })
        .collect();
    let n = f.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    for x in &mut f {
        *x /= n;
    }
    f.iter()
        .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
        .collect()
}

fn norm_query(seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    let mut v: Vec<f32> = (0..DIM)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / 1e6) - 0.5
        })
        .collect();
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    for x in &mut v {
        *x /= n;
    }
    v
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("f16_dot_fma");
    g.sample_size(30);

    let stored: Vec<Vec<u8>> = (0..BATCH).map(|v| f16_bytes(0x1000 + v as u64)).collect();
    let query = norm_query(0xBEEF);

    // Sanity: the two agree to sub-ULP (they are different roundings of the same dot).
    for s in &stored {
        let a = dot_product_f16_bytes_f32(s, &query).expect("base");
        let b = dot_product_f16_bytes_f32_fma(s, &query);
        assert!(
            (a - b).abs() <= a.abs().max(1e-6) * 1e-4,
            "FMA vs mul+add drifted beyond sub-ULP: {a} vs {b}"
        );
    }

    let run_base = || {
        let mut acc = 0.0f32;
        for s in &stored {
            acc += dot_product_f16_bytes_f32(black_box(s), black_box(&query)).unwrap();
        }
        black_box(acc);
    };
    let run_fma = || {
        let mut acc = 0.0f32;
        for s in &stored {
            acc += dot_product_f16_bytes_f32_fma(black_box(s), black_box(&query));
        }
        black_box(acc);
    };

    let null = paired_median_ratio(41, 4, run_base, run_base);
    let lever = paired_median_ratio(41, 4, run_base, run_fma);
    eprintln!(
        "[null]  f16_dot/{DIM}x{BATCH}: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
        null.median, null.p5, null.p95, null.rounds
    );
    eprintln!(
        "[lever] f16_dot/{DIM}x{BATCH}: fma/ORIG median {:.4} p5 {:.4} p95 {:.4} -> {}",
        lever.median,
        lever.p5,
        lever.p95,
        if lever.decidable_against(&null) {
            if lever.median < 1.0 {
                "DECIDABLE WIN"
            } else {
                "DECIDABLE REGRESSION"
            }
        } else {
            "INSIDE NULL FLOOR (not decidable)"
        }
    );

    g.bench_function("mul_add", |b| b.iter(run_base));
    g.bench_function("fma", |b| b.iter(run_fma));
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
