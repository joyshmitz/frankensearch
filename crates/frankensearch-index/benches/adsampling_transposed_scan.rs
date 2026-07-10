//! ADSampling-style CONCENTRATION-BOUND early-abandon on the candidate-transposed
//! cluster-grouped scan — an APPROXIMATE regime the exact bound cannot reach.
//!
//! The exact primitive (`transposed_abandon_scan`, landed) abandons a candidate
//! group when the worst-case Cauchy–Schwarz suffix bound `partial + ‖q_suf‖·‖v_suf‖`
//! ≤ the k-th best. That bound is loose: for an energy-ordered suffix the residual
//! `q·v_suffix = ‖q_suf‖·‖v_suf‖·cos θ`, and for effectively-random residual
//! directions in `m` remaining dims, `cos θ` concentrates around 0 with std ~1/√m.
//! So the residual is typically ~√m SMALLER than the worst case. ADSampling's
//! insight: abandon on the CONCENTRATION bound
//!   `partial + z·‖q_suf‖·‖v_suf‖/√m`
//! which is ~√m ≈ 18× tighter (m≈352 after block 0) → candidates abandon FAR
//! earlier, at a per-decision error rate set by the confidence multiplier `z`.
//! `z→∞` (or the CS bound) is exact; smaller `z` trades a little recall for a lot
//! of speed. This is the approximate lever the exact early-abandon arc could not
//! touch; recall vs the exact top-k is MEASURED per `z`.
//!
//! Reuses the transposed layout (candidate-major groups of 8, cluster-sorted,
//! energy-reordered dims) so the reduce-elimination win composes with the tighter
//! pruning. Arms: `full` (flat ORIG) vs `transposed_exact` (CS bound, z=∞) vs
//! `transposed_ads_z{3,2}` (concentration). Swept over k∈{10,100}.
//!
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-index --profile release \
//!     --bench adsampling_transposed_scan
//! ```

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

/// Transposed cluster-grouped abandon. `z = None` → exact worst-case
/// Cauchy–Schwarz bound; `z = Some(zc)` → ADSampling concentration bound
/// `partial + zc·‖q_suf‖·‖v_suf‖/√(remaining_dims)` (approximate).
fn topk_transposed(
    rq: &[f32],
    qsuf: &[f32],
    tvecs: &[f32],
    tvsuf: &[f32],
    cand_ids: &[usize],
    ngroups: usize,
    k: usize,
    factor: f32,
    conc: bool,
) -> (Vec<(f32, usize)>, u64) {
    let nb = DIM / BLOCK;
    // Per-boundary bound scale. exact: factor=1, conc=false → qsuf[b+1] (worst-case
    // Cauchy–Schwarz). ε-relaxed: factor<1, conc=false → factor·qsuf[b+1] (prune a
    // little more). concentration: conc=true → factor·qsuf[b+1]/√(remaining_dims).
    let mut scale = vec![0.0f32; nb + 1];
    for b in 0..nb {
        let rem = (DIM - (b + 1) * BLOCK) as f32;
        scale[b + 1] = if conc && rem > 0.0 {
            factor * qsuf[b + 1] / rem.sqrt()
        } else {
            factor * qsuf[b + 1]
        };
    }
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
                let bound = acc + f32x8::splat(scale[b + 1]) * f32x8::from(sn);
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

fn bench_ads(c: &mut Criterion) {
    assert_eq!(N % LANES, 0);
    let centroids: Vec<Vec<f32>> = (0..CLUSTERS)
        .map(|i| normalize(shape(raw_vector(0xc000_0000 + i as u64))))
        .collect();
    let vectors: Vec<Vec<f32>> = (0..N)
        .map(|i| make_vector(&centroids, i % CLUSTERS, i as u64 + 1))
        .collect();
    let queries: Vec<Vec<f32>> = (0..QUERIES)
        .map(|q| make_vector(&centroids, q % CLUSTERS, 0xdead_0000 + q as u64))
        .collect();

    let order = energy_order(&vectors);
    let mut rvecs = vec![0.0f32; N * DIM];
    for (i, v) in vectors.iter().enumerate() {
        rvecs[i * DIM..(i + 1) * DIM].copy_from_slice(&reorder(v, &order));
    }
    let rqueries: Vec<Vec<f32>> = queries.iter().map(|q| reorder(q, &order)).collect();

    let mut perm: Vec<usize> = (0..N).collect();
    perm.sort_by_key(|&i| i % CLUSTERS);
    let ngroups = N / LANES;

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

    // (name, factor, concentration?). exact = worst-case CS; ads_z3 = concentration
    // (recall-destroying, kept as documented rejection); eps* = ε-relaxed CS.
    let variants: [(&str, f32, bool); 4] = [
        ("exact", 1.0, false),
        ("ads_z3", 3.0, true),
        ("eps097", 0.97, false),
        ("eps090", 0.90, false),
    ];

    let mut group = c.benchmark_group("adsampling_transposed_scan");
    for &k in &KS {
        // ── recall + abandonment report per variant (exact = recall 1.0 by def) ──
        for (name, factor, conc) in variants {
            let mut total_blocks = 0u64;
            let mut recalls = 0.0f64;
            for (qi, rq) in rqueries.iter().enumerate() {
                let full = topk_full(rq, &rvecs, k);
                let (t, bd) = topk_transposed(
                    rq, &qsuf[qi], &tvecs, &tvsuf, &cand_ids, ngroups, k, factor, conc,
                );
                total_blocks += bd;
                let fids: std::collections::HashSet<usize> = full.iter().map(|&(_, i)| i).collect();
                let hit = t.iter().filter(|&&(_, i)| fids.contains(&i)).count();
                recalls += hit as f64 / k as f64;
                if factor == 1.0 && !conc {
                    // exact: also assert score-equivalence (fp-reorder tolerant)
                    let fs: Vec<f32> = full.iter().map(|&(s, _)| s).collect();
                    let ts: Vec<f32> = t.iter().map(|&(s, _)| s).collect();
                    for j in 0..k {
                        let tol = 1e-4 * fs[j].abs().max(1.0);
                        assert!(
                            (fs[j] - ts[j]).abs() <= tol,
                            "exact score mismatch rank {j} (k={k},q={qi})"
                        );
                    }
                }
            }
            let frac = total_blocks as f64 / (QUERIES as f64 * N as f64 * nb as f64);
            eprintln!(
                "[ads] k={k} {name}: blocks = {:.1}% of full ({:.2}/{nb} avg); recall@{k} = {:.4}",
                frac * 100.0,
                frac * nb as f64,
                recalls / QUERIES as f64
            );
        }

        let mut qi = 0usize;
        group.bench_function(format!("full_k{k}"), |b| {
            b.iter(|| {
                let rq = &rqueries[qi % QUERIES];
                qi += 1;
                black_box(topk_full(black_box(rq), &rvecs, k))
            });
        });
        for (name, factor, conc) in variants {
            let mut qi = 0usize;
            group.bench_function(format!("transposed_{name}_k{k}"), |b| {
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
                        factor,
                        conc,
                    ))
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_ads);
criterion_main!(benches);
