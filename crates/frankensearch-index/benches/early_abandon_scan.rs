//! Early-abandonment top-k dense scan — a radically different primitive vs the
//! incumbent "compute every candidate's full dot, then top-k" flat scan.
//!
//! PRIMITIVE (ADSampling / early-abandoning-dot class, exact variant): store the
//! dense vectors with dimensions **reordered by descending per-dim energy** (a
//! one-time index-build transform) and precompute per-vector **block suffix L2
//! norms**. At query time, accumulate each candidate's dot in `BLOCK`-dim chunks;
//! after each chunk, the Cauchy–Schwarz bound `partial + ‖q_suffix‖·‖v_suffix‖`
//! is the *maximum achievable* full dot. Once the top-k heap is full, if that
//! bound ≤ the k-th best score, the candidate can never enter top-k → **abandon**
//! the rest of its dims. Energy-ordering makes the suffix norm collapse fast, so
//! far-from-query candidates abandon after 1–2 blocks.
//!
//! This is EXACT (bit-identical top-k): both arms use the same block summation
//! order, so a candidate that is *not* abandoned has a byte-identical partial dot
//! to the full scan; an abandoned candidate provably cannot enter top-k. The bench
//! asserts identical top-k (ids + order) before timing.
//!
//! A/B (swept over k∈{10,100}): `full_k{k}` (no prune, reordered layout) vs
//! `abandon_block{32,64,128}_k{k}` (check-every-block prune) vs
//! `abandon_stride3_k{k}` (prune with a strided bound-check: reduce+check after
//! block 0, then every 3 blocks — fewer latency-bound horizontal reduces). Both
//! prune arms call the identical SIMD block-dot, so the ratio isolates pruning.
//! Finding: check-every-block wins at k=10 but regresses at k=100 (the reduce
//! overhead scales with blocks-survived as the cutoff loosens); the strided
//! variant wins at BOTH k by amortizing the reduce over several blocks.
//!
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/fs-op \
//!   rch exec -- cargo bench -p frankensearch-index --profile release \
//!     --bench early_abandon_scan
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use wide::f32x8;

const N: usize = 50_000;
const DIM: usize = 384;
/// Swept top-k depths: k=10 (final retrieval) and k=100 (fusion-pool /
/// rerank-feed candidate-gen depth). The check-every-block abandon wins at k=10
/// but regresses at k=100 (looser cutoff → more blocks survived → the
/// per-block horizontal reduce dominates); the strided abandon wins at both.
const KS: [usize; 2] = [10, 100];
const QUERIES: usize = 32;
const CLUSTERS: usize = 64;
// Real-embedding within-cluster tightness (cos(vec,centroid) ~0.75) — NOT the
// prior NOISE=0.30 diffuse regime, which produced cos~0.28 loose clusters and
// (like the false ANN rejection in `ann-in-bold-viable`) starved every
// distribution-sensitive vector lever of the structure it exploits.
const NOISE: f32 = 0.07;

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

/// Skewed spectral envelope mimicking real embedding singular-value decay: a few
/// high-variance dims carry most of the signal. This is the property the
/// early-abandon primitive exploits (energy-ordering → suffix norms collapse
/// fast). Synthetic white-noise vectors LACK it — a flat spectrum keeps every
/// suffix norm large, so the Cauchy–Schwarz bound stays loose and no candidate
/// abandons early. Shaping the shared cluster subspace makes the synthetic
/// distribution spectrally faithful to a real dense index.
fn shape(mut v: Vec<f32>) -> Vec<f32> {
    let tau = DIM as f32 * 0.15;
    for (d, x) in v.iter_mut().enumerate() {
        *x *= (-(d as f32) / tau).exp() + 0.03;
    }
    v
}

/// SIMD block dot over `len` (multiple of 8) consecutive f32s, 4-accumulator ILP.
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

/// Insert (score, idx) into a descending top-k list (k tiny → linear insert).
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

/// Full-dot baseline (reordered layout, no pruning).
fn topk_full(rq: &[f32], rvecs: &[f32], k: usize) -> Vec<(f32, usize)> {
    let mut top = Vec::with_capacity(k + 1);
    for i in 0..N {
        let v = &rvecs[i * DIM..(i + 1) * DIM];
        let s = dot_block(rq, v, DIM);
        offer(&mut top, k, s, i);
    }
    top
}

/// Early-abandon top-k. `qsuf[b]` = ‖rq[b*block..]‖, `vsuf[i*(nb+1)+b]` = ‖rvec_i[b*block..]‖.
/// Returns (top-k, total_blocks_computed) for abandonment-rate reporting.
fn topk_abandon(
    rq: &[f32],
    qsuf: &[f32],
    rvecs: &[f32],
    vsuf: &[f32],
    block: usize,
    k: usize,
) -> (Vec<(f32, usize)>, u64) {
    let nb = DIM / block;
    let mut top: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
    let mut blocks_done: u64 = 0;
    for i in 0..N {
        let v = &rvecs[i * DIM..(i + 1) * DIM];
        let vs = &vsuf[i * (nb + 1)..(i + 1) * (nb + 1)];
        let full = top.len() == k;
        let cutoff = if full { top[k - 1].0 } else { f32::NEG_INFINITY };
        let mut partial = 0.0f32;
        let mut abandoned = false;
        for b in 0..nb {
            partial += dot_block(&rq[b * block..], &v[b * block..], block);
            blocks_done += 1;
            if full {
                let bound = partial + qsuf[b + 1] * vs[b + 1];
                if bound <= cutoff {
                    abandoned = true;
                    break;
                }
            }
        }
        if !abandoned {
            offer(&mut top, k, partial, i);
        }
    }
    (top, blocks_done)
}

/// Accumulate one 32-dim block into 4 live f32x8 accumulators WITHOUT reducing.
#[inline]
fn accum_block32(acc: &mut [f32x8; 4], a: &[f32], b: &[f32]) {
    let mut chunk = [0.0f32; 8];
    let mut chb = [0.0f32; 8];
    chunk.copy_from_slice(&a[0..8]);
    chb.copy_from_slice(&b[0..8]);
    acc[0] += f32x8::from(chunk) * f32x8::from(chb);
    chunk.copy_from_slice(&a[8..16]);
    chb.copy_from_slice(&b[8..16]);
    acc[1] += f32x8::from(chunk) * f32x8::from(chb);
    chunk.copy_from_slice(&a[16..24]);
    chb.copy_from_slice(&b[16..24]);
    acc[2] += f32x8::from(chunk) * f32x8::from(chb);
    chunk.copy_from_slice(&a[24..32]);
    chb.copy_from_slice(&b[24..32]);
    acc[3] += f32x8::from(chunk) * f32x8::from(chb);
}
#[inline]
fn reduce4(acc: &[f32x8; 4]) -> f32 {
    ((acc[0] + acc[1]) + (acc[2] + acc[3])).reduce_add()
}

/// Adaptive-stride early-abandon: block=32 accumulation into LIVE accumulators,
/// but only reduce+check the Cauchy–Schwarz bound after block 0 (to catch the
/// far-candidate majority in one reduce) and then every `STRIDE` blocks. Fewer
/// horizontal reduces than the check-every-block variant. EXACT: checking less
/// often can only DELAY an abandon, never cause a wrong one (the bound remains a
/// valid upper bound at each check), so top-k is unchanged.
fn topk_abandon_stride(
    rq: &[f32],
    qsuf: &[f32],
    rvecs: &[f32],
    vsuf: &[f32],
    k: usize,
    stride: usize,
) -> (Vec<(f32, usize)>, u64) {
    const BLOCK: usize = 32;
    let nb = DIM / BLOCK;
    let mut top: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
    let mut blocks_done: u64 = 0;
    for i in 0..N {
        let v = &rvecs[i * DIM..(i + 1) * DIM];
        let vs = &vsuf[i * (nb + 1)..(i + 1) * (nb + 1)];
        let full = top.len() == k;
        let cutoff = if full { top[k - 1].0 } else { f32::NEG_INFINITY };
        let mut acc = [f32x8::splat(0.0); 4];
        let mut abandoned = false;
        for b in 0..nb {
            accum_block32(&mut acc, &rq[b * BLOCK..], &v[b * BLOCK..]);
            blocks_done += 1;
            if full && (b == 0 || (b + 1) % stride == 0) {
                let bound = reduce4(&acc) + qsuf[b + 1] * vs[b + 1];
                if bound <= cutoff {
                    abandoned = true;
                    break;
                }
            }
        }
        if !abandoned {
            offer(&mut top, k, reduce4(&acc), i);
        }
    }
    (top, blocks_done)
}

/// Descending energy order of dimensions across the corpus.
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

/// Suffix L2 norms at every block boundary (inclusive of the empty tail = 0).
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

fn bench_early_abandon(c: &mut Criterion) {
    let centroids: Vec<Vec<f32>> = (0..CLUSTERS)
        .map(|i| normalize(shape(raw_vector(0xc000_0000 + i as u64))))
        .collect();
    let vectors: Vec<Vec<f32>> = (0..N)
        .map(|i| make_vector(&centroids, i % CLUSTERS, i as u64 + 1))
        .collect();
    let queries: Vec<Vec<f32>> = (0..QUERIES)
        .map(|q| make_vector(&centroids, q % CLUSTERS, 0xdead_0000 + q as u64))
        .collect();

    // ── one-time index build: energy-reordered layout + suffix norms ──
    let order = energy_order(&vectors);
    let mut rvecs = vec![0.0f32; N * DIM];
    for (i, v) in vectors.iter().enumerate() {
        let rv = reorder(v, &order);
        rvecs[i * DIM..(i + 1) * DIM].copy_from_slice(&rv);
    }
    let rqueries: Vec<Vec<f32>> = queries.iter().map(|q| reorder(q, &order)).collect();

    const STRIDE: usize = 3;
    let mut group = c.benchmark_group("early_abandon_scan");
    for block in [32usize, 64, 128] {
        let nb = DIM / block;
        let mut vsuf = vec![0.0f32; N * (nb + 1)];
        for i in 0..N {
            let s = suffix_norms(&rvecs[i * DIM..(i + 1) * DIM], block);
            vsuf[i * (nb + 1)..(i + 1) * (nb + 1)].copy_from_slice(&s);
        }
        let qsuf: Vec<Vec<f32>> = rqueries.iter().map(|q| suffix_norms(q, block)).collect();

        for &k in &KS {
            // ── parity + abandonment-rate report (check-every-block abandon) ──
            let mut total_blocks = 0u64;
            for (qi, rq) in rqueries.iter().enumerate() {
                let full = topk_full(rq, &rvecs, k);
                let (pruned, bd) = topk_abandon(rq, &qsuf[qi], &rvecs, &vsuf, block, k);
                total_blocks += bd;
                assert_eq!(
                    full.iter().map(|&(_, i)| i).collect::<Vec<_>>(),
                    pruned.iter().map(|&(_, i)| i).collect::<Vec<_>>(),
                    "top-k mismatch (block={block}, k={k}, q={qi})"
                );
            }
            let avg_frac = total_blocks as f64 / (QUERIES as f64 * N as f64 * nb as f64);
            eprintln!(
                "[early_abandon] block={block} nb={nb} N={N} dim={DIM} k={k}: \
                 blocks computed = {:.1}% of full ({:.2} of {nb} blocks/cand avg)",
                avg_frac * 100.0,
                avg_frac * nb as f64
            );

            if block == 32 {
                // ── strided abandon: parity + abandonment report ──
                let mut stride_blocks = 0u64;
                for (qi, rq) in rqueries.iter().enumerate() {
                    let full = topk_full(rq, &rvecs, k);
                    let (st, bd) = topk_abandon_stride(rq, &qsuf[qi], &rvecs, &vsuf, k, STRIDE);
                    stride_blocks += bd;
                    assert_eq!(
                        full.iter().map(|&(_, i)| i).collect::<Vec<_>>(),
                        st.iter().map(|&(_, i)| i).collect::<Vec<_>>(),
                        "stride top-k mismatch (k={k}, q={qi})"
                    );
                }
                let sfrac = stride_blocks as f64 / (QUERIES as f64 * N as f64 * nb as f64);
                eprintln!(
                    "[early_abandon] stride={STRIDE} k={k}: blocks computed = {:.1}% of full ({:.2} of {nb} blocks/cand avg)",
                    sfrac * 100.0,
                    sfrac * nb as f64
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
                group.bench_function(format!("abandon_stride3_k{k}"), |b| {
                    b.iter(|| {
                        let rq = &rqueries[qi % QUERIES];
                        let qs = &qsuf[qi % QUERIES];
                        qi += 1;
                        black_box(topk_abandon_stride(black_box(rq), qs, &rvecs, &vsuf, k, STRIDE))
                    });
                });
            }
            let mut qi = 0usize;
            group.bench_function(format!("abandon_block{block}_k{k}"), |b| {
                b.iter(|| {
                    let rq = &rqueries[qi % QUERIES];
                    let qs = &qsuf[qi % QUERIES];
                    qi += 1;
                    black_box(topk_abandon(black_box(rq), qs, &rvecs, &vsuf, block, k))
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_early_abandon);
criterion_main!(benches);
