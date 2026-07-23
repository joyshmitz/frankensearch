//! Paired A/B for the FSVI two-tier quality-rescore per-hit scorer.
//!
//! `TwoTierIndex::score_quality_for_fast_index` scored each fast-tier hit against
//! the f16 quality slab by calling `VectorIndex::vector_at_f32` — which allocates a
//! `Vec<f32>` and runs a **scalar** `f16::to_f32` decode loop — and then a separate
//! `dot_product_f32_f32`. The brute-force scan (`search.rs`) already scores the same
//! f16 slab with the fused `dot_product_f16_bytes_f32` (hardware `vcvtph2ps` decode,
//! no allocation), so the rescore path was the slow, allocating outlier. This bench
//! measures that exact per-hit swap:
//!
//! - `decode_then_f32dot` : mirror `vector_at_f32` (F16) — decode bytes → `Vec<f32>`
//!   — then `dot_product_f32_f32` (the shipping path).
//! - `fused_bytes_dot`    : `dot_product_f16_bytes_f32` on the borrowed slab bytes
//!   (the new `dot_query_at` path for `dim % 32 == 0`).
//!
//! For `dim % 32 == 0` the two are **bit-identical** (asserted before timing): the
//! 4-accumulator SIMD grouping/reduction coincides and f16→f32 decode is exact.
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/<lane> \
//!   rch exec -- cargo bench -p frankensearch-index --bench quality_rescore_fused_dot_ab
//! ```

#![allow(clippy::chunks_exact_to_as_chunks)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_index::{dot_product_f16_bytes_f32, dot_product_f32_f32};
use half::f16;

const DIM: usize = 384; // dim % 32 == 0 → fused path is bit-identical
const N_DOCS: usize = 50_000;

fn next(state: &mut u64) -> u64 {
    // xorshift64* — deterministic, no external RNG.
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

/// Map a raw draw to `[-1.0, 1.0)`. The top-24 bits fit f32's mantissa exactly.
fn unit_f32(bits: u64) -> f32 {
    ((bits >> 40) as f32) / 16_777_216.0 * 2.0 - 1.0
}

/// A flat f16 byte slab of `N_DOCS * DIM` little-endian halves, exactly as an FSVI
/// F16 vector index stores them.
fn build_slab() -> Vec<u8> {
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut slab = Vec::with_capacity(N_DOCS * DIM * 2);
    for _ in 0..N_DOCS * DIM {
        let h = f16::from_f32(unit_f32(next(&mut state)));
        slab.extend_from_slice(&h.to_le_bytes());
    }
    slab
}

fn make_query(state: &mut u64) -> Vec<f32> {
    (0..DIM).map(|_| unit_f32(next(state))).collect()
}

/// The candidate hit positions (arbitrary spread over the corpus).
fn make_positions(m: usize) -> Vec<usize> {
    let stride = (N_DOCS / m).max(1);
    (0..m).map(|j| (j * stride) % N_DOCS).collect()
}

#[inline]
fn slab_bytes(slab: &[u8], pos: usize) -> &[u8] {
    &slab[pos * DIM * 2..(pos + 1) * DIM * 2]
}

/// Shipping path: `vector_at_f32` (F16) allocation + scalar decode, then f32 dot.
#[inline]
fn decode_then_f32dot(slab: &[u8], positions: &[usize], query: &[f32]) -> Vec<f32> {
    positions
        .iter()
        .map(|&pos| {
            let bytes = slab_bytes(slab, pos);
            let mut decoded = Vec::with_capacity(DIM);
            for chunk in bytes.chunks_exact(2) {
                decoded.push(f16::from_le_bytes([chunk[0], chunk[1]]).to_f32());
            }
            dot_product_f32_f32(&decoded, query).unwrap()
        })
        .collect()
}

/// New path: fused byte-based f16·f32 dot, no allocation, hardware decode.
#[inline]
fn fused_bytes_dot(slab: &[u8], positions: &[usize], query: &[f32]) -> Vec<f32> {
    positions
        .iter()
        .map(|&pos| dot_product_f16_bytes_f32(slab_bytes(slab, pos), query).unwrap())
        .collect()
}

fn bench(c: &mut Criterion) {
    let slab = build_slab();
    let mut qstate = 0x0f0f_0f0f_dead_beef_u64;
    let query = make_query(&mut qstate);

    let mut group = c.benchmark_group("quality_rescore_fused_dot");
    for &m in &[32usize, 128, 300] {
        let positions = make_positions(m);

        // Bit-identical parity gate (dim % 32 == 0).
        let base = decode_then_f32dot(&slab, &positions, &query);
        let cand = fused_bytes_dot(&slab, &positions, &query);
        assert_eq!(base.len(), cand.len(), "length parity (m={m})");
        for (a, b) in base.iter().zip(cand.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "score parity (m={m})");
        }

        group.bench_with_input(
            BenchmarkId::new("decode_then_f32dot", m),
            &positions,
            |b, positions| {
                b.iter(|| black_box(decode_then_f32dot(&slab, black_box(positions), &query)));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("fused_bytes_dot", m),
            &positions,
            |b, positions| {
                b.iter(|| black_box(fused_bytes_dot(&slab, black_box(positions), &query)));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
