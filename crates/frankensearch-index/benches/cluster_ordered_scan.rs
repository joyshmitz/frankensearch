//! QUERY-DIRECTED cluster traversal for the candidate-transposed early-abandon
//! scan — visit clusters in descending `q·centroid` order so the near cluster is
//! scanned FIRST, setting a tight top-k cutoff immediately; every far cluster then
//! abandons at block 0.
//!
//! The landed transposed scan (`transposed_abandon_scan`, `7afaadf`) processes
//! cluster-groups in arbitrary storage order. Measured waste: far clusters visited
//! BEFORE the query's own cluster don't abandon (the cutoff is still loose from
//! low-scoring far candidates), so ~half the far clusters get scanned deep before
//! the cutoff tightens (28–30% of blocks computed).
//!
//! PRIMITIVE: precompute per-cluster centroids `μ_c` (one-time, free at query time).
//! At query, compute the 64 cheap dots `q·μ_c` (~0.13% of a full scan), sort
//! clusters by descending similarity, and process cluster-groups in that order. The
//! near cluster is scanned first → cutoff jumps to the true top-k level → every
//! subsequent far cluster's group abandons after block 0. This is EXACT: no cluster
//! is skipped (unlike a loose cluster-bound prune — the max-radius residual bound is
//! useless in 384-dim); the traversal ORDER only changes WHEN the cutoff tightens,
//! and the per-candidate suffix-norm bound still guarantees no true top-k is dropped.
//!
//! Arms: `full` (flat ORIG) vs `transposed_unord` (identity cluster order = the
//! landed scan) vs `transposed_ord` (centroid-directed). Swept over k∈{10,100}.
//!
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-index --profile release \
//!     --bench cluster_ordered_scan
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
        offer(&mut top, k, dot_block(rq, &rvecs[i * DIM..(i + 1) * DIM], DIM), i);
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

/// Cluster ids sorted by descending `q·μ_c` (near cluster first). Cost = 64 dots.
fn cluster_order(rq: &[f32], centroids: &[f32]) -> Vec<usize> {
    let mut cs: Vec<(f32, usize)> = (0..CLUSTERS)
        .map(|c| (dot_block(rq, &centroids[c * DIM..(c + 1) * DIM], DIM), c))
        .collect();
    cs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    cs.into_iter().map(|(_, c)| c).collect()
}

/// Transposed cluster-grouped early-abandon, visiting clusters in `order`.
fn topk_ordered(
    rq: &[f32],
    qsuf: &[f32],
    tvecs: &[f32],
    tvsuf: &[f32],
    cand_ids: &[usize],
    order: &[usize],
    cluster_groups: &[Vec<usize>],
    k: usize,
) -> (Vec<(f32, usize)>, u64) {
    let nb = DIM / BLOCK;
    let mut top: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
    let mut blocks_done: u64 = 0;
    let mut chunk = [0.0f32; LANES];
    let mut sn = [0.0f32; LANES];
    for &c in order {
        for &g in &cluster_groups[c] {
            let vbase = g * DIM * LANES;
            let sbase = g * (nb + 1) * LANES;
            let full = top.len() == k;
            let cutoff = if full { top[k - 1].0 } else { f32::NEG_INFINITY };
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
    }
    (top, blocks_done)
}

fn bench_cluster_ordered(c: &mut Criterion) {
    assert_eq!(N % LANES, 0);
    let centroids0: Vec<Vec<f32>> = (0..CLUSTERS)
        .map(|i| normalize(shape(raw_vector(0xc000_0000 + i as u64))))
        .collect();
    let vectors: Vec<Vec<f32>> = (0..N)
        .map(|i| make_vector(&centroids0, i % CLUSTERS, i as u64 + 1))
        .collect();
    let queries: Vec<Vec<f32>> = (0..QUERIES)
        .map(|q| make_vector(&centroids0, q % CLUSTERS, 0xdead_0000 + q as u64))
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

    // Per-cluster centroids in the reordered space (mean of members).
    let mut cent = vec![0.0f32; CLUSTERS * DIM];
    let mut counts = vec![0usize; CLUSTERS];
    for i in 0..N {
        let cl = i % CLUSTERS;
        let rv = &rvecs[i * DIM..(i + 1) * DIM];
        for d in 0..DIM {
            cent[cl * DIM + d] += rv[d];
        }
        counts[cl] += 1;
    }
    for cl in 0..CLUSTERS {
        if counts[cl] > 0 {
            for d in 0..DIM {
                cent[cl * DIM + d] /= counts[cl] as f32;
            }
        }
    }
    // Group index lists per cluster (boundary groups → first member's cluster).
    let mut cluster_groups: Vec<Vec<usize>> = vec![Vec::new(); CLUSTERS];
    for g in 0..ngroups {
        cluster_groups[cand_ids[g * LANES] % CLUSTERS].push(g);
    }
    let ident: Vec<usize> = (0..CLUSTERS).collect();

    let mut group = c.benchmark_group("cluster_ordered_scan");
    for &k in &KS {
        // ── parity (score-equivalence + set) + blocks report ──
        let mut blk_unord = 0u64;
        let mut blk_ord = 0u64;
        let mut swaps = 0usize;
        for (qi, rq) in rqueries.iter().enumerate() {
            let full = topk_full(rq, &rvecs, k);
            let (u, bu) =
                topk_ordered(rq, &qsuf[qi], &tvecs, &tvsuf, &cand_ids, &ident, &cluster_groups, k);
            let ord = cluster_order(rq, &cent);
            let (o, bo) =
                topk_ordered(rq, &qsuf[qi], &tvecs, &tvsuf, &cand_ids, &ord, &cluster_groups, k);
            blk_unord += bu;
            blk_ord += bo;
            let fs: Vec<f32> = full.iter().map(|&(s, _)| s).collect();
            for (label, res) in [("unord", &u), ("ord", &o)] {
                let ts: Vec<f32> = res.iter().map(|&(s, _)| s).collect();
                for j in 0..k {
                    let tol = 1e-4 * fs[j].abs().max(1.0);
                    assert!(
                        (fs[j] - ts[j]).abs() <= tol,
                        "{label} score mismatch rank {j} (k={k},q={qi})"
                    );
                }
            }
            let fids: std::collections::HashSet<usize> = full.iter().map(|&(_, i)| i).collect();
            swaps += o.iter().filter(|&&(_, i)| !fids.contains(&i)).count();
        }
        let denom = (QUERIES as f64) * (N as f64) * (nb as f64);
        eprintln!(
            "[cluster_ordered] k={k}: blocks unord={:.1}% ord={:.1}% ; boundary id-swaps={swaps}/{}",
            blk_unord as f64 / denom * 100.0,
            blk_ord as f64 / denom * 100.0,
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
        group.bench_function(format!("transposed_unord_k{k}"), |b| {
            b.iter(|| {
                let rq = &rqueries[qi % QUERIES];
                let qs = &qsuf[qi % QUERIES];
                qi += 1;
                black_box(topk_ordered(
                    black_box(rq),
                    qs,
                    &tvecs,
                    &tvsuf,
                    &cand_ids,
                    &ident,
                    &cluster_groups,
                    k,
                ))
            });
        });
        let mut qi = 0usize;
        group.bench_function(format!("transposed_ord_k{k}"), |b| {
            b.iter(|| {
                let rq = &rqueries[qi % QUERIES];
                let qs = &qsuf[qi % QUERIES];
                qi += 1;
                let ord = cluster_order(black_box(rq), &cent);
                black_box(topk_ordered(
                    black_box(rq),
                    qs,
                    &tvecs,
                    &tvsuf,
                    &cand_ids,
                    &ord,
                    &cluster_groups,
                    k,
                ))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_cluster_ordered);
criterion_main!(benches);
