//! RESIDUAL-BUCKETED transposed groups — recover per-candidate pruning power
//! INSIDE the fast (reduce-free) transposed layout via an exact pre-block-0
//! whole-group skip.
//!
//! Two prior rounds bracketed the frontier into one precise tension:
//!  * the TIGHT bound `q·v ≤ q·μ_c + ‖q‖·‖v−μ_c‖` (per-candidate residual)
//!    cut blocks 12.6→7.6% — but only in the SLOW row-major layout
//!    (`rowmajor_prefilter_scan`, REJECTED: row-major's per-block horizontal
//!    reduce outweighs the skips → loses to the transposed traversal).
//!  * the FAST transposed layout (`cluster_ordered_scan`, `2945ed8`) can only
//!    use a LOOSE group bound — a group skips only if all 8 lanes clear
//!    (~0.4⁸), so an explicit group-skip never fires when groups are random.
//!
//! SYNTHESIS: sort each cluster's members by residual `dist[i]=‖v_i−μ_c‖` at
//! INDEX time (query-independent), then pack transposed groups of 8. Now each
//! group is residual-HOMOGENEOUS → its exact upper bound `q·μ_c + max_lane(dist)`
//! is TIGHT (all 8 lanes ≈ equal, so the group max ≈ the group min, not the
//! whole cluster's max-radius) → whole far-residual groups clear the cutoff and
//! skip BEFORE block 0. This ports the per-candidate pruning into the layout
//! that already won on reduce-elimination.
//!
//! EXACT (recall 1.0): the skip is a valid Cauchy–Schwarz upper bound for every
//! lane in the group (‖q‖=1 here). Boundary groups whose 8 lanes straddle two
//! clusters (dist measured against different centroids) are flagged unskippable.
//! Fallback-safe: the skip is one compare per group; if within-cluster residual
//! variance is low it simply never fires and the scan degrades to `2945ed8`.
//!
//! Arms: `full` (flat ORIG) vs `transposed_ord` (`2945ed8`: cluster-sorted
//! layout, cluster-ordered traversal, NO skip) vs `bucketed_skip` (residual-
//! bucketed layout + pre-block-0 group-skip). Swept over k∈{10,100}.
//!
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-index --profile release \
//!     --bench bucketed_transposed_scan
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

/// A candidate-transposed layout for one member permutation.
struct Layout {
    tvecs: Vec<f32>,
    tvsuf: Vec<f32>,
    cand_ids: Vec<usize>,
    /// Per-group max residual `max_lane ‖v−μ_c‖` (valid only for intra-cluster groups).
    gmax_dist: Vec<f32>,
    /// Per-group: all 8 lanes share one cluster → the group-skip bound is valid.
    gskippable: Vec<bool>,
    /// Group index lists per cluster (boundary groups → first member's cluster).
    cluster_groups: Vec<Vec<usize>>,
}

/// Pack `perm` (member order) into the transposed layout + per-group skip metadata.
fn build_layout(perm: &[usize], rvecs: &[f32], dist: &[f32]) -> Layout {
    let nb = DIM / BLOCK;
    let ngroups = N / LANES;
    let mut tvecs = vec![0.0f32; N * DIM];
    let mut tvsuf = vec![0.0f32; ngroups * (nb + 1) * LANES];
    let mut cand_ids = vec![0usize; N];
    let mut gmax_dist = vec![0.0f32; ngroups];
    let mut gskippable = vec![true; ngroups];
    for g in 0..ngroups {
        let c0 = perm[g * LANES] % CLUSTERS;
        let mut gmax = 0.0f32;
        for lane in 0..LANES {
            let cand = perm[g * LANES + lane];
            cand_ids[g * LANES + lane] = cand;
            if cand % CLUSTERS != c0 {
                gskippable[g] = false; // straddles clusters → dist vs different μ
            }
            gmax = gmax.max(dist[cand]);
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
        gmax_dist[g] = gmax;
    }
    let mut cluster_groups: Vec<Vec<usize>> = vec![Vec::new(); CLUSTERS];
    for g in 0..ngroups {
        cluster_groups[cand_ids[g * LANES] % CLUSTERS].push(g);
    }
    Layout { tvecs, tvsuf, cand_ids, gmax_dist, gskippable, cluster_groups }
}

/// Transposed cluster-grouped early-abandon, visiting clusters in `order`.
/// `skip=true` adds the exact pre-block-0 residual group-skip
/// `qmu[c] + gmax_dist[g] ≤ cutoff` (valid: ‖q‖=1, group intra-cluster).
#[allow(clippy::too_many_arguments)]
fn topk_ordered(
    rq: &[f32],
    qsuf: &[f32],
    lay: &Layout,
    qmu: &[f32],
    order: &[usize],
    k: usize,
    skip: bool,
) -> (Vec<(f32, usize)>, u64) {
    let nb = DIM / BLOCK;
    let mut top: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
    let mut blocks_done: u64 = 0;
    let mut chunk = [0.0f32; LANES];
    let mut sn = [0.0f32; LANES];
    for &c in order {
        let qc = qmu[c];
        for &g in &lay.cluster_groups[c] {
            let vbase = g * DIM * LANES;
            let sbase = g * (nb + 1) * LANES;
            let full = top.len() == k;
            let cutoff = if full { top[k - 1].0 } else { f32::NEG_INFINITY };
            // Exact pre-block-0 whole-group skip: every lane obeys
            // q·v ≤ q·μ_c + ‖q‖·‖v−μ_c‖ ≤ qc + gmax_dist[g] (‖q‖=1).
            if skip && full && lay.gskippable[g] && qc + lay.gmax_dist[g] <= cutoff {
                continue;
            }
            let mut acc = f32x8::splat(0.0);
            let mut abandoned = false;
            for b in 0..nb {
                for d in b * BLOCK..(b + 1) * BLOCK {
                    chunk.copy_from_slice(&lay.tvecs[vbase + d * LANES..vbase + d * LANES + LANES]);
                    acc += f32x8::splat(rq[d]) * f32x8::from(chunk);
                }
                blocks_done += LANES as u64;
                if full {
                    sn.copy_from_slice(
                        &lay.tvsuf[sbase + (b + 1) * LANES..sbase + (b + 1) * LANES + LANES],
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
                    offer(&mut top, k, scores[lane], lay.cand_ids[g * LANES + lane]);
                }
            }
        }
    }
    (top, blocks_done)
}

/// `qmu[c] = q·μ_c` for all clusters + cluster ids sorted by descending qmu.
fn query_centroids(rq: &[f32], cent: &[f32]) -> (Vec<f32>, Vec<usize>) {
    let mut qmu = vec![0.0f32; CLUSTERS];
    let mut cs: Vec<(f32, usize)> = Vec::with_capacity(CLUSTERS);
    for c in 0..CLUSTERS {
        let d = dot_block(rq, &cent[c * DIM..(c + 1) * DIM], DIM);
        qmu[c] = d;
        cs.push((d, c));
    }
    cs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    (qmu, cs.into_iter().map(|(_, c)| c).collect())
}

fn bench_bucketed(c: &mut Criterion) {
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
    // Per-vector residual dist to its OWN cluster centroid (query-independent).
    let mut dist = vec![0.0f32; N];
    for i in 0..N {
        let cl = i % CLUSTERS;
        let rv = &rvecs[i * DIM..(i + 1) * DIM];
        let mut s = 0.0f32;
        for d in 0..DIM {
            let e = rv[d] - cent[cl * DIM + d];
            s += e * e;
        }
        dist[i] = s.sqrt();
    }

    // Layout A: cluster-sorted (== 2945ed8 baseline).
    let mut perm_cluster: Vec<usize> = (0..N).collect();
    perm_cluster.sort_by_key(|&i| i % CLUSTERS);
    let lay_cluster = build_layout(&perm_cluster, &rvecs, &dist);

    // Layout B: residual-bucketed — within each cluster, core (low dist) first
    // so the near cluster tightens the cutoff before its edge groups are skipped.
    let mut perm_bucket: Vec<usize> = (0..N).collect();
    perm_bucket.sort_by(|&a, &b| {
        let (ca, cb) = (a % CLUSTERS, b % CLUSTERS);
        ca.cmp(&cb)
            .then(dist[a].partial_cmp(&dist[b]).unwrap())
            .then(a.cmp(&b))
    });
    let lay_bucket = build_layout(&perm_bucket, &rvecs, &dist);

    let nb = DIM / BLOCK;

    let mut group = c.benchmark_group("bucketed_transposed_scan");
    for &k in &KS {
        // ── parity (score-equivalence + set) + blocks report ──
        let mut blk_base = 0u64;
        let mut blk_bkt_noskip = 0u64;
        let mut blk_bkt_skip = 0u64;
        let mut swaps = 0usize;
        let mut skipped_groups = 0u64;
        for (qi, rq) in rqueries.iter().enumerate() {
            let full = topk_full(rq, &rvecs, k);
            let (qmu, ord) = query_centroids(rq, &cent);
            let (base, bb) = topk_ordered(rq, &qsuf[qi], &lay_cluster, &qmu, &ord, k, false);
            let (bkt0, b0) = topk_ordered(rq, &qsuf[qi], &lay_bucket, &qmu, &ord, k, false);
            let (bkts, bs) = topk_ordered(rq, &qsuf[qi], &lay_bucket, &qmu, &ord, k, true);
            blk_base += bb;
            blk_bkt_noskip += b0;
            blk_bkt_skip += bs;
            let fs: Vec<f32> = full.iter().map(|&(s, _)| s).collect();
            for (label, res) in [("base", &base), ("bkt_noskip", &bkt0), ("bkt_skip", &bkts)] {
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
            swaps += bkts.iter().filter(|&&(_, i)| !fids.contains(&i)).count();
            // approximate skip-firing count for reporting (bound holds on final cutoff)
            if !full.is_empty() {
                let cut = full[k - 1].0;
                for g in 0..(N / LANES) {
                    let cg = lay_bucket.cand_ids[g * LANES] % CLUSTERS;
                    if lay_bucket.gskippable[g] && qmu[cg] + lay_bucket.gmax_dist[g] <= cut {
                        skipped_groups += 1;
                    }
                }
            }
        }
        let denom = (QUERIES as f64) * (N as f64) * (nb as f64);
        eprintln!(
            "[bucketed] k={k}: blocks base={:.1}% bkt_noskip={:.1}% bkt_skip={:.1}% ; \
             groups skippable-at-final-cutoff={:.1}% ; boundary id-swaps={swaps}/{}",
            blk_base as f64 / denom * 100.0,
            blk_bkt_noskip as f64 / denom * 100.0,
            blk_bkt_skip as f64 / denom * 100.0,
            skipped_groups as f64 / (QUERIES as f64 * (N / LANES) as f64) * 100.0,
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
        group.bench_function(format!("transposed_ord_k{k}"), |b| {
            b.iter(|| {
                let rq = &rqueries[qi % QUERIES];
                let qs = &qsuf[qi % QUERIES];
                qi += 1;
                let (qmu, ord) = query_centroids(black_box(rq), &cent);
                black_box(topk_ordered(black_box(rq), qs, &lay_cluster, &qmu, &ord, k, false))
            });
        });
        let mut qi = 0usize;
        group.bench_function(format!("bucketed_skip_k{k}"), |b| {
            b.iter(|| {
                let rq = &rqueries[qi % QUERIES];
                let qs = &qsuf[qi % QUERIES];
                qi += 1;
                let (qmu, ord) = query_centroids(black_box(rq), &cent);
                black_box(topk_ordered(black_box(rq), qs, &lay_bucket, &qmu, &ord, k, true))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_bucketed);
criterion_main!(benches);
