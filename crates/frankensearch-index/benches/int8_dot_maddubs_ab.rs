//! Isolated latency A/B (bd-b5wl): `vpmaddubs` int8 dot vs the shipped `vpmovsxbw`+`vpmaddwd` kernel.
//!
//! # Why isolated, not end-to-end
//!
//! The full int8 two-pass scan's end-to-end A/B is **undecidable under the current fleet** — cod's
//! null control measured a per-function floor of CV 32–35% (`docs/NEGATIVE_EVIDENCE.md`, worker
//! contention), far wider than any single-kernel effect after Amdahl (the int8 dot leaf is
//! ~23–44% of the scan). This bench instead measures the **dot kernel in isolation**, where the
//! kernel is ~100% of the timed work and the null floor is tight, so the kernel-level ratio is
//! decidable even while the scan-level ratio is not.
//!
//! # The lever
//!
//! The shipped `dot_i8_i8_avx2` sign-extends BOTH operands (4× `vpmovsxbw` per 32 int8) before
//! `vpmaddwd`. `dot_i8_i8_maddubs` shifts `stored` to the u8 domain with one `vpxor` and uses
//! `vpmaddubs` (u8·i8 → i16, one op, no stored widening), folding the `128·Σq` bias into a
//! per-query scalar. It is APPROXIMATE (`vpmaddubs` saturates i16 pair-sums). This bench uses
//! `|x| ≤ 40` so the two arms are **bit-identical** (asserted before timing) — a clean speed A/B
//! with no quality variable. That is NOT the full deployment distribution: the shipped `127/max_abs`
//! quantizer leaves a ±127 tail that does saturate, so on real data maddubs is
//! approximate-but-recall-preserving — proven deterministically in
//! `simd.rs::maddubs_pass1_preserves_f32_recall_under_real_saturation`. Instruction count is
//! value-independent, so the speed ratio here holds for the real distribution too.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- cargo bench -p frankensearch-index --features bench-internals --bench int8_dot_maddubs_ab
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_core::bench_support::paired_median_ratio;
use frankensearch_index::{dot_i8_i8, dot_i8_i8_maddubs, maddubs_query_bias};

/// Deterministic int8 vector, `|x| ≤ bound` (models int8-quantized L2-normalized cosine vectors,
/// where a 384-dim component is typically ±6 and rarely past ±40 — well below the `vpmaddubs`
/// saturation ceiling, so the approximate kernel is bit-exact here).
fn i8_vec(n: usize, seed: u64, bound: i8) -> Vec<i8> {
    let mut s = seed | 1;
    let span = u64::from(2 * bound.unsigned_abs() + 1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            i8::try_from(i64::try_from(s % span).unwrap_or(0) - i64::from(bound)).unwrap_or(0)
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("int8_dot_maddubs");
    g.sample_size(30);

    // One dim (384, the fast-tier width) run as a batch of many dots per timed region, which is the
    // real scan's inner shape (N dots per query).
    let dim = 384usize;
    let batch = 4096usize;
    let stored: Vec<Vec<i8>> = (0..batch)
        .map(|v| i8_vec(dim, 0x1000 + v as u64, 40))
        .collect();
    let query = i8_vec(dim, 0xBEEF, 40);
    let bias = maddubs_query_bias(&query, dim);

    // PARITY GATE: on realistic magnitudes the approximate kernel is bit-exact, so this is a pure
    // speed A/B, not a speed/quality trade. Assert before timing.
    for s in &stored {
        assert_eq!(
            dot_i8_i8(s, &query),
            dot_i8_i8_maddubs(s, &query, bias),
            "maddubs must be bit-exact on |x|≤40 inputs"
        );
    }

    let run_orig = || {
        let mut acc = 0i64;
        for s in &stored {
            acc += i64::from(dot_i8_i8(black_box(s), black_box(&query)));
        }
        black_box(acc);
    };
    let run_maddubs = || {
        let mut acc = 0i64;
        for s in &stored {
            acc += i64::from(dot_i8_i8_maddubs(
                black_box(s),
                black_box(&query),
                black_box(bias),
            ));
        }
        black_box(acc);
    };

    // NULL CONTROL first (A/A on the shipped kernel), then the lever. Ratio = maddubs/ORIG, so
    // <1.0 means the new kernel is faster. Gate on median vs the null p5..p95 spread.
    let null = paired_median_ratio(41, 4, run_orig, run_orig);
    let lever = paired_median_ratio(41, 4, run_orig, run_maddubs);
    eprintln!(
        "[null]  int8_dot/{dim}x{batch}: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
        null.median, null.p5, null.p95, null.rounds
    );
    eprintln!(
        "[lever] int8_dot/{dim}x{batch}: maddubs/ORIG median {:.4} p5 {:.4} p95 {:.4} -> {}",
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

    // Retain criterion arms as profile diagnostics (not the A/B ratio).
    g.bench_function("orig_vpmaddwd", |b| b.iter(run_orig));
    g.bench_function("maddubs", |b| b.iter(run_maddubs));
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
