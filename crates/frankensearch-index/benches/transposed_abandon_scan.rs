//! Candidate-TRANSPOSED SIMD dense scan with CLUSTER-GROUPED early-abandon — a
//! radically different execution model vs both the row-major flat scan and the
//! row-major early-abandon (`early_abandon_scan`).
//!
//! MOTIVATION: the row-major early-abandon's measured bottleneck is the per-block
//! horizontal `reduce_add` (one per candidate per block; it dominates once the
//! cutoff loosens at larger k — see `docs/PERF_LEDGER.md` early-abandon rows).
//! The batched multi-query GEMM scan was already rejected (bandwidth is NOT the
//! bottleneck; the scan is FMA-compute-bound — `NEGATIVE_EVIDENCE.md` 73a77fc).
//! So this does NOT try to save bandwidth; it attacks the reduce + the FMAs a
//! DIFFERENT way:
//!
//! PRIMITIVE — process candidates in GROUPS of 8, stored candidate-major within
//! a group (for group g, dim d, lane l: `tvecs[g·DIM·8 + d·8 + l]`), and energy-
//! reorder the dims (as in `early_abandon_scan`). The 8 candidates' running
//! partial dots live in ONE `f32x8` accumulator — accumulation is VERTICAL
//! (`acc += splat(q[d]) · v_lanes[d]`), so there is NO horizontal reduce during
//! the scan. After each 32-dim block, the Cauchy–Schwarz bound
//! `acc + ‖q_suffix‖·‖v_suffix‖` is checked per lane; if the group's MAX bound ≤
//! the k-th best (one max-fold over 8 lanes — 8× fewer reduces than row-major),
//! ALL 8 candidates provably cannot enter top-k → **abandon the whole group**.
//!
//! CLUSTER-SORTING (one-time index-build transform, free at query time) stores a
//! cluster's vectors contiguously, so a group of 8 is intra-cluster: a far
//! cluster abandons all 8 lanes together at block ~1 (no SIMD divergence), while
//! the query's own cluster computes fully. Without it, round-robin cluster
//! assignment would put 8 different clusters in each group → the group can't
//! abandon until its nearest lane does (maximal divergence).
//!
//! EXACT: a completed group's `acc` lanes are byte-identical full dots; an
//! abandoned group's every lane provably < cutoff. Parity vs the flat scan is
//! asserted before timing. Swept over k∈{10,100}.
//!
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-index --profile release \
//!     --bench transposed_abandon_scan
//! ```

#![allow(
    clippy::cast_lossless,
    clippy::needless_range_loop,
    clippy::range_plus_one
)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use wide::f32x8;

const N: usize = 50_000;
const DIM: usize = 384;
const QUERIES: usize = 32;
const CLUSTERS: usize = 64;
const NOISE: f32 = 0.07;
const LANES: usize = 8;
const BLOCK: usize = 32;
const KS: [usize; 2] = [10, 100];

fn raw_vector(seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut v = Vec::with_capacity(DIM);
    for _ in 0..DIM {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        v.push((state >> 40) as f32 / (1u64 << 23) as f32 - 1.0);
    }
    v
}
fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-12 {
        for x in &mut v {
            *x /= n;
        }
    }
    v
}
/// Skewed spectral envelope ≈ real embedding singular-value decay (see
/// `early_abandon_scan`): concentrates energy so the suffix norms collapse fast.
fn shape(mut v: Vec<f32>) -> Vec<f32> {
    let tau = DIM as f32 * 0.15;
    for (d, x) in v.iter_mut().enumerate() {
        *x *= (-(d as f32) / tau).exp() + 0.03;
    }
    v
}
fn make_vector(centroids: &[Vec<f32>], c: usize, noise_seed: u64) -> Vec<f32> {
    let centroid = &centroids[c % centroids.len()];
    let noise = raw_vector(noise_seed);
    normalize(
        centroid
            .iter()
            .zip(&noise)
            .map(|(a, n)| a + NOISE * n)
            .collect(),
    )
}

/// SIMD block dot over `len` (multiple of 8) consecutive f32s, 4-accumulator ILP
/// — the row-major baseline kernel (identical to `early_abandon_scan`).
#[inline]
fn dot_block(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut acc0 = f32x8::splat(0.0);
    let mut acc1 = f32x8::splat(0.0);
    let mut acc2 = f32x8::splat(0.0);
    let mut acc3 = f32x8::splat(0.0);
    let mut j = 0;
    let mut chunk = [0.0f32; 8];
    let mut chb = [0.0f32; 8];
    while j + 32 <= len {
        chunk.copy_from_slice(&a[j..j + 8]);
        chb.copy_from_slice(&b[j..j + 8]);
        acc0 += f32x8::from(chunk) * f32x8::from(chb);
        chunk.copy_from_slice(&a[j + 8..j + 16]);
        chb.copy_from_slice(&b[j + 8..j + 16]);
        acc1 += f32x8::from(chunk) * f32x8::from(chb);
        chunk.copy_from_slice(&a[j + 16..j + 24]);
        chb.copy_from_slice(&b[j + 16..j + 24]);
        acc2 += f32x8::from(chunk) * f32x8::from(chb);
        chunk.copy_from_slice(&a[j + 24..j + 32]);
        chb.copy_from_slice(&b[j + 24..j + 32]);
        acc3 += f32x8::from(chunk) * f32x8::from(chb);
        j += 32;
    }
    while j + 8 <= len {
        chunk.copy_from_slice(&a[j..j + 8]);
        chb.copy_from_slice(&b[j..j + 8]);
        acc0 += f32x8::from(chunk) * f32x8::from(chb);
        j += 8;
    }
    ((acc0 + acc1) + (acc2 + acc3)).reduce_add()
}

#[inline]
fn offer(top: &mut Vec<(f32, usize)>, k: usize, score: f32, idx: usize) {
    if top.len() < k {
        let pos = top
            .iter()
            .position(|&(s, _)| s < score)
            .unwrap_or(top.len());
        top.insert(pos, (score, idx));
    } else if score > top[k - 1].0 {
        let pos = top.iter().position(|&(s, _)| s < score).unwrap_or(k - 1);
        top.insert(pos, (score, idx));
        top.truncate(k);
    }
}

/// Row-major flat scan baseline (ORIG).
fn topk_full(rq: &[f32], rvecs: &[f32], k: usize) -> Vec<(f32, usize)> {
    let mut top = Vec::with_capacity(k + 1);
    for i in 0..N {
        let v = &rvecs[i * DIM..(i + 1) * DIM];
        offer(&mut top, k, dot_block(rq, v, DIM), i);
    }
    top
}

fn energy_order(vectors: &[Vec<f32>]) -> Vec<usize> {
    let mut energy = vec![0.0f64; DIM];
    for v in vectors {
        for (d, &x) in v.iter().enumerate() {
            energy[d] += (x as f64) * (x as f64);
        }
    }
    let mut order: Vec<usize> = (0..DIM).collect();
    order.sort_by(|&a, &b| energy[b].partial_cmp(&energy[a]).unwrap());
    order
}
fn reorder(v: &[f32], order: &[usize]) -> Vec<f32> {
    order.iter().map(|&d| v[d]).collect()
}
fn suffix_norms(rv: &[f32], block: usize) -> Vec<f32> {
    let nb = DIM / block;
    let mut out = vec![0.0f32; nb + 1];
    let mut acc = 0.0f32;
    for b in (0..nb).rev() {
        for d in b * block..(b + 1) * block {
            acc += rv[d] * rv[d];
        }
        out[b] = acc.sqrt();
    }
    out
}

/// Candidate-transposed, cluster-grouped early-abandon top-k.
/// `tvecs`: group-major, dim-major, lane-minor. `tvsuf`: group-major,
/// block-major, lane-minor (nb+1 boundaries). `cand_ids[g*LANES + l]` = original id.
fn topk_transposed(
    rq: &[f32],
    qsuf: &[f32],
    tvecs: &[f32],
    tvsuf: &[f32],
    cand_ids: &[usize],
    ngroups: usize,
    k: usize,
) -> (Vec<(f32, usize)>, u64) {
    let nb = DIM / BLOCK;
    let mut top: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
    let mut blocks_done: u64 = 0;
    let mut chunk = [0.0f32; LANES];
    let mut sn = [0.0f32; LANES];
    for g in 0..ngroups {
        let vbase = g * DIM * LANES;
        let sbase = g * (nb + 1) * LANES;
        let full = top.len() == k;
        let cutoff = if full {
            top[k - 1].0
        } else {
            f32::NEG_INFINITY
        };
        let mut acc = f32x8::splat(0.0);
        let mut abandoned = false;
        for b in 0..nb {
            for d in b * BLOCK..(b + 1) * BLOCK {
                chunk.copy_from_slice(&tvecs[vbase + d * LANES..vbase + d * LANES + LANES]);
                acc += f32x8::splat(rq[d]) * f32x8::from(chunk);
            }
            blocks_done += LANES as u64;
            if full {
                sn.copy_from_slice(
                    &tvsuf[sbase + (b + 1) * LANES..sbase + (b + 1) * LANES + LANES],
                );
                let bound = acc + f32x8::splat(qsuf[b + 1]) * f32x8::from(sn);
                let maxb = bound
                    .to_array()
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                if maxb <= cutoff {
                    abandoned = true;
                    break;
                }
            }
        }
        if !abandoned {
            let scores = acc.to_array();
            for lane in 0..LANES {
                offer(&mut top, k, scores[lane], cand_ids[g * LANES + lane]);
            }
        }
    }
    (top, blocks_done)
}

fn bench_transposed(c: &mut Criterion) {
    assert_eq!(N % LANES, 0, "N must be a multiple of LANES for this bench");
    let centroids: Vec<Vec<f32>> = (0..CLUSTERS)
        .map(|i| normalize(shape(raw_vector(0xc000_0000 + i as u64))))
        .collect();
    let vectors: Vec<Vec<f32>> = (0..N)
        .map(|i| make_vector(&centroids, i % CLUSTERS, i as u64 + 1))
        .collect();
    let queries: Vec<Vec<f32>> = (0..QUERIES)
        .map(|q| make_vector(&centroids, q % CLUSTERS, 0xdead_0000 + q as u64))
        .collect();

    // ── index build: energy-reorder dims + row-major store (for the baseline) ──
    let order = energy_order(&vectors);
    let mut rvecs = vec![0.0f32; N * DIM];
    for (i, v) in vectors.iter().enumerate() {
        rvecs[i * DIM..(i + 1) * DIM].copy_from_slice(&reorder(v, &order));
    }
    let rqueries: Vec<Vec<f32>> = queries.iter().map(|q| reorder(q, &order)).collect();

    // ── cluster-sort candidates for contiguous groups ──
    let mut perm: Vec<usize> = (0..N).collect();
    perm.sort_by_key(|&i| i % CLUSTERS);
    let ngroups = N / LANES;

    // ── transposed candidate-major store + per-lane suffix norms ──
    let nb = DIM / BLOCK;
    let mut tvecs = vec![0.0f32; N * DIM];
    let mut tvsuf = vec![0.0f32; ngroups * (nb + 1) * LANES];
    let mut cand_ids = vec![0usize; N];
    for g in 0..ngroups {
        for lane in 0..LANES {
            let cand = perm[g * LANES + lane];
            cand_ids[g * LANES + lane] = cand;
            let rv = &rvecs[cand * DIM..(cand + 1) * DIM];
            let vbase = g * DIM * LANES;
            for d in 0..DIM {
                tvecs[vbase + d * LANES + lane] = rv[d];
            }
            let s = suffix_norms(rv, BLOCK);
            let sbase = g * (nb + 1) * LANES;
            for b in 0..nb + 1 {
                tvsuf[sbase + b * LANES + lane] = s[b];
            }
        }
    }
    let qsuf: Vec<Vec<f32>> = rqueries.iter().map(|q| suffix_norms(q, BLOCK)).collect();

    let mut group = c.benchmark_group("transposed_abandon_scan");
    for &k in &KS {
        // ── parity + abandonment report ──
        // The transposed layout accumulates each candidate's dot in a different
        // summation order (per-dim sequential into one f32x8 lane) than the
        // row-major 4-accumulator kernel, so dots differ at the ULP level → the
        // true top-k is recovered up to f32 reorder, NOT bit-identical. Verify the
        // top-k SCORES match position-wise within epsilon (equivalent ranking) and
        // count boundary id-swaps (ULP-ties at the k-th place) for honesty.
        let mut total_blocks = 0u64;
        let mut id_swaps = 0usize;
        for (qi, rq) in rqueries.iter().enumerate() {
            let full = topk_full(rq, &rvecs, k);
            let (t, bd) = topk_transposed(rq, &qsuf[qi], &tvecs, &tvsuf, &cand_ids, ngroups, k);
            total_blocks += bd;
            let fs: Vec<f32> = full.iter().map(|&(s, _)| s).collect();
            let ts: Vec<f32> = t.iter().map(|&(s, _)| s).collect();
            for j in 0..k {
                let tol = 1e-4 * fs[j].abs().max(1.0);
                assert!(
                    (fs[j] - ts[j]).abs() <= tol,
                    "transposed top-k SCORE mismatch at rank {j} (k={k}, q={qi}): {} vs {}",
                    fs[j],
                    ts[j]
                );
            }
            let fids: std::collections::HashSet<usize> = full.iter().map(|&(_, i)| i).collect();
            id_swaps += t.iter().filter(|&&(_, i)| !fids.contains(&i)).count();
        }
        let frac = total_blocks as f64 / (QUERIES as f64 * N as f64 * nb as f64);
        eprintln!(
            "[transposed] k={k} N={N} dim={DIM} lanes={LANES}: blocks computed = {:.1}% of full ({:.2} of {nb} blocks/cand avg); boundary id-swaps = {id_swaps}/{} (ULP-ties, scores match <1e-4)",
            frac * 100.0,
            frac * nb as f64,
            QUERIES * k
        );

        let mut qi = 0usize;
        group.bench_function(format!("full_k{k}"), |b| {
            b.iter(|| {
                let rq = &rqueries[qi % QUERIES];
                qi += 1;
                black_box(topk_full(black_box(rq), &rvecs, k))
            });
        });
        let mut qi = 0usize;
        group.bench_function(format!("transposed_k{k}"), |b| {
            b.iter(|| {
                let rq = &rqueries[qi % QUERIES];
                let qs = &qsuf[qi % QUERIES];
                qi += 1;
                black_box(topk_transposed(
                    black_box(rq),
                    qs,
                    &tvecs,
                    &tvsuf,
                    &cand_ids,
                    ngroups,
                    k,
                ))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_transposed);
criterion_main!(benches);
