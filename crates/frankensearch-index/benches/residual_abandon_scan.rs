//! RESIDUAL-SPACE (centroid-centered) early-abandon — tighten the Cauchy–Schwarz
//! slack by scanning `q·(v−μ_c)` instead of `q·v`.
//!
//! The landed bucketed scan (`05b57f3`) bounds `q·v ≤ q·v[0:b] + ‖q_suf‖·‖v_suf‖`
//! with `‖v_suf‖ ≈ ‖v‖ = 1`. But `q·v = q·μ_c + q·r`, `r = v − μ_c`, and in tight
//! clusters `‖r‖ ≈ 0.3 ≪ 1`. Store the CENTERED residual `r` transposed (with
//! residual suffix norms) and the CS slack `‖q_suf‖·‖r_suf‖` shrinks ~3× → the
//! bound tightens ~3× faster → more groups abandon at block 0 AND the near
//! cluster's non-top-k candidates (currently scanned deep because `‖v_suf‖` keeps
//! the raw bound loose) start abandoning too.
//!
//! EXACT: `q·v = q·μ_c + q·r` reconstructed per lane via a per-lane `qmu` vector
//! `gqmu[l] = q·μ_{c(lane)}` — so boundary groups (lanes straddling clusters) need
//! NO special-case; each lane uses its own centroid dot. The pre-block-0 skip is
//! just the `b=0` case (`gqmu + ‖q‖·‖r‖ ≤ cutoff`), and check-then-compute abandons
//! one block earlier than the landed compute-then-check. Same (cluster,dist)-bucketed
//! layout + query-directed cluster traversal as `05b57f3`.
//!
//! Arms: `full` (flat ORIG) vs `bucketed_skip` (= `05b57f3`, raw v) vs
//! `residual_scan` (centered r). Swept over k∈{10,100}.
//!
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-index --profile release \
//!     --bench residual_abandon_scan
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

/// Candidate-transposed layout for one member permutation (raw or residual body).
struct Layout {
    /// Transposed vectors: `body[g*DIM*L + d*L + lane]`.
    body: Vec<f32>,
    /// Transposed suffix norms: `suf[g*(nb+1)*L + b*L + lane]`.
    suf: Vec<f32>,
    cand_ids: Vec<usize>,
    /// Cluster of each candidate (== cand % CLUSTERS), lane-aligned to `cand_ids`.
    cand_cluster: Vec<usize>,
    /// Per-group max residual (for the raw `bucketed_skip` baseline skip).
    gmax_dist: Vec<f32>,
    gskippable: Vec<bool>,
    cluster_groups: Vec<Vec<usize>>,
}

/// Pack `perm` into the transposed layout. `body_of(cand)` supplies the per-vector
/// body (raw `v` or centered residual `r`); `dist` is the residual norm sidecar.
fn build_layout(
    perm: &[usize],
    body_src: &[f32],
    dist: &[f32],
) -> Layout {
    let nb = DIM / BLOCK;
    let ngroups = N / LANES;
    let mut body = vec![0.0f32; N * DIM];
    let mut suf = vec![0.0f32; ngroups * (nb + 1) * LANES];
    let mut cand_ids = vec![0usize; N];
    let mut cand_cluster = vec![0usize; N];
    let mut gmax_dist = vec![0.0f32; ngroups];
    let mut gskippable = vec![true; ngroups];
    for g in 0..ngroups {
        let c0 = perm[g * LANES] % CLUSTERS;
        let mut gmax = 0.0f32;
        for lane in 0..LANES {
            let cand = perm[g * LANES + lane];
            cand_ids[g * LANES + lane] = cand;
            cand_cluster[g * LANES + lane] = cand % CLUSTERS;
            if cand % CLUSTERS != c0 {
                gskippable[g] = false;
            }
            gmax = gmax.max(dist[cand]);
            let bv = &body_src[cand * DIM..(cand + 1) * DIM];
            let vbase = g * DIM * LANES;
            for d in 0..DIM {
                body[vbase + d * LANES + lane] = bv[d];
            }
            let s = suffix_norms(bv, BLOCK);
            let sbase = g * (nb + 1) * LANES;
            for b in 0..nb + 1 {
                suf[sbase + b * LANES + lane] = s[b];
            }
        }
        gmax_dist[g] = gmax;
    }
    let mut cluster_groups: Vec<Vec<usize>> = vec![Vec::new(); CLUSTERS];
    for g in 0..ngroups {
        cluster_groups[cand_ids[g * LANES] % CLUSTERS].push(g);
    }
    Layout { body, suf, cand_ids, cand_cluster, gmax_dist, gskippable, cluster_groups }
}

/// `qmu[c] = q·μ_c` + cluster ids sorted by descending qmu (near cluster first).
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

/// RAW `bucketed_skip` baseline (== `05b57f3`): body stores v, suffix norms of v,
/// pre-block-0 skip `qmu[c] + gmax_dist[g] ≤ cutoff`, compute-then-check bound.
#[allow(clippy::too_many_arguments)]
fn topk_bucketed(
    rq: &[f32],
    qsuf: &[f32],
    lay: &Layout,
    qmu: &[f32],
    order: &[usize],
    k: usize,
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
            if full && lay.gskippable[g] && qc + lay.gmax_dist[g] <= cutoff {
                continue;
            }
            let mut acc = f32x8::splat(0.0);
            let mut abandoned = false;
            for b in 0..nb {
                for d in b * BLOCK..(b + 1) * BLOCK {
                    chunk.copy_from_slice(&lay.body[vbase + d * LANES..vbase + d * LANES + LANES]);
                    acc += f32x8::splat(rq[d]) * f32x8::from(chunk);
                }
                blocks_done += LANES as u64;
                if full {
                    sn.copy_from_slice(&lay.suf[sbase + (b + 1) * LANES..sbase + (b + 1) * LANES + LANES]);
                    let bound = acc + f32x8::splat(qsuf[b + 1]) * f32x8::from(sn);
                    let maxb = bound.to_array().iter().copied().fold(f32::NEG_INFINITY, f32::max);
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

/// RESIDUAL scan: body stores r = v − μ_c, suffix norms of r. Score reconstructed
/// per lane as `gqmu[lane] + q·r`. Check-then-compute: bound `gqmu + acc + qsuf[b]·rsuf[b]`
/// (acc over blocks 0..b) is a valid per-lane CS upper bound; the b=0 check is the skip.
fn topk_residual(
    rq: &[f32],
    qsuf: &[f32],
    lay: &Layout,
    qmu: &[f32],
    order: &[usize],
    k: usize,
) -> (Vec<(f32, usize)>, u64) {
    let nb = DIM / BLOCK;
    let mut top: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
    let mut blocks_done: u64 = 0;
    let mut chunk = [0.0f32; LANES];
    let mut sn = [0.0f32; LANES];
    let mut gq = [0.0f32; LANES];
    for &c in order {
        for &g in &lay.cluster_groups[c] {
            let vbase = g * DIM * LANES;
            let sbase = g * (nb + 1) * LANES;
            let full = top.len() == k;
            let cutoff = if full { top[k - 1].0 } else { f32::NEG_INFINITY };
            // Per-lane centroid dot (exact for boundary groups too).
            for lane in 0..LANES {
                gq[lane] = qmu[lay.cand_cluster[g * LANES + lane]];
            }
            let gqmu = f32x8::from(gq);
            let mut acc = f32x8::splat(0.0);
            let mut abandoned = false;
            for b in 0..nb {
                if full {
                    // check-then-compute: bound uses residual suffix from block b.
                    sn.copy_from_slice(&lay.suf[sbase + b * LANES..sbase + b * LANES + LANES]);
                    let bound = gqmu + acc + f32x8::splat(qsuf[b]) * f32x8::from(sn);
                    let maxb = bound.to_array().iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    if maxb <= cutoff {
                        abandoned = true;
                        break;
                    }
                }
                for d in b * BLOCK..(b + 1) * BLOCK {
                    chunk.copy_from_slice(&lay.body[vbase + d * LANES..vbase + d * LANES + LANES]);
                    acc += f32x8::splat(rq[d]) * f32x8::from(chunk);
                }
                blocks_done += LANES as u64;
            }
            if !abandoned {
                let scores = (gqmu + acc).to_array();
                for lane in 0..LANES {
                    offer(&mut top, k, scores[lane], lay.cand_ids[g * LANES + lane]);
                }
            }
        }
    }
    (top, blocks_done)
}

fn bench_residual(c: &mut Criterion) {
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
    // Per-vector residual r = v − μ_c and its norm.
    let mut resid = vec![0.0f32; N * DIM];
    let mut dist = vec![0.0f32; N];
    for i in 0..N {
        let cl = i % CLUSTERS;
        let mut s = 0.0f32;
        for d in 0..DIM {
            let e = rvecs[i * DIM + d] - cent[cl * DIM + d];
            resid[i * DIM + d] = e;
            s += e * e;
        }
        dist[i] = s.sqrt();
    }

    // Residual-bucketed permutation (== 05b57f3): (cluster asc, dist asc, id).
    let mut perm: Vec<usize> = (0..N).collect();
    perm.sort_by(|&a, &b| {
        let (ca, cb) = (a % CLUSTERS, b % CLUSTERS);
        ca.cmp(&cb)
            .then(dist[a].partial_cmp(&dist[b]).unwrap())
            .then(a.cmp(&b))
    });
    let lay_v = build_layout(&perm, &rvecs, &dist); // raw v body (bucketed_skip)
    let lay_r = build_layout(&perm, &resid, &dist); // centered r body (residual)

    let nb = DIM / BLOCK;
    let mut group = c.benchmark_group("residual_abandon_scan");
    for &k in &KS {
        // ── parity (score-equivalence + set) + blocks report ──
        let mut blk_bkt = 0u64;
        let mut blk_res = 0u64;
        let mut swaps = 0usize;
        for (qi, rq) in rqueries.iter().enumerate() {
            let full = topk_full(rq, &rvecs, k);
            let (qmu, ord) = query_centroids(rq, &cent);
            let (bkt, bb) = topk_bucketed(rq, &qsuf[qi], &lay_v, &qmu, &ord, k);
            let (res, br) = topk_residual(rq, &qsuf[qi], &lay_r, &qmu, &ord, k);
            blk_bkt += bb;
            blk_res += br;
            let fs: Vec<f32> = full.iter().map(|&(s, _)| s).collect();
            for (label, r) in [("bkt", &bkt), ("res", &res)] {
                let ts: Vec<f32> = r.iter().map(|&(s, _)| s).collect();
                for j in 0..k {
                    let tol = 1e-4 * fs[j].abs().max(1.0);
                    assert!(
                        (fs[j] - ts[j]).abs() <= tol,
                        "{label} score mismatch rank {j} (k={k},q={qi}): {} vs {}",
                        fs[j],
                        ts[j]
                    );
                }
            }
            let fids: std::collections::HashSet<usize> = full.iter().map(|&(_, i)| i).collect();
            swaps += res.iter().filter(|&&(_, i)| !fids.contains(&i)).count();
        }
        let denom = (QUERIES as f64) * (N as f64) * (nb as f64);
        eprintln!(
            "[residual] k={k}: blocks bucketed={:.1}% residual={:.1}% ; boundary id-swaps={swaps}/{}",
            blk_bkt as f64 / denom * 100.0,
            blk_res as f64 / denom * 100.0,
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
        group.bench_function(format!("bucketed_skip_k{k}"), |b| {
            b.iter(|| {
                let rq = &rqueries[qi % QUERIES];
                let qs = &qsuf[qi % QUERIES];
                qi += 1;
                let (qmu, ord) = query_centroids(black_box(rq), &cent);
                black_box(topk_bucketed(black_box(rq), qs, &lay_v, &qmu, &ord, k))
            });
        });
        let mut qi = 0usize;
        group.bench_function(format!("residual_scan_k{k}"), |b| {
            b.iter(|| {
                let rq = &rqueries[qi % QUERIES];
                let qs = &qsuf[qi % QUERIES];
                qi += 1;
                let (qmu, ord) = query_centroids(black_box(rq), &cent);
                black_box(topk_residual(black_box(rq), qs, &lay_r, &qmu, &ord, k))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_residual);
criterion_main!(benches);
