//! THROUGHPUT-optimized transposed residual group kernel — attack the per-group
//! OVERHEAD and the latency-bound block dot that the landed residual scan (`8d13097`)
//! left on the table (its "block cut 25/28% > time cut 10/19%" gap).
//!
//! Two exact reassociations, isolated as separate arms:
//!  (1) `gqmu` SPLAT-not-gather: the score reconstruction needs `q·μ_{c(lane)}` per
//!      lane. In the (cluster,dist)-bucketed layout 99% of groups are intra-cluster
//!      (all 8 lanes share cluster c), so `gqmu = splat(qmu[c])` — collapsing 8
//!      gathers to one splat. Only boundary groups (flagged) gather per-lane.
//!  (2) 4-ACCUMULATOR ILP: the transposed block dot `acc += splat(rq[d])·body[d]` is
//!      a SINGLE f32x8 accumulator = a dependent FMA chain (latency-bound). Unlike
//!      the register-spilling f16 *widen* kernel, this is a plain-f32 load (4 accs +
//!      query splat fit YMM), and unlike f32×f32 dots it's a scalar-broadcast (rq in
//!      L1, ~1 load/FMA) → more ILP headroom. Split the 32-dim block over 4 accs.
//!
//! Both are exact up to f32 summation order (verified via score-equivalence + zero
//! boundary swaps), same layout + traversal + bound as `8d13097`.
//!
//! Arms: `full` (flat ORIG) vs `residual_base` (= `8d13097`) vs `residual_splat`
//! (gqmu splat only) vs `residual_fast` (splat + 4-acc). Swept over k∈{10,100}.
//!
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-index --profile release \
//!     --bench residual_ilp_scan
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
        offer(
            &mut top,
            k,
            dot_block(rq, &rvecs[i * DIM..(i + 1) * DIM], DIM),
            i,
        );
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

struct Layout {
    body: Vec<f32>,
    suf: Vec<f32>,
    cand_ids: Vec<usize>,
    cand_cluster: Vec<usize>,
    /// true = all 8 lanes share one cluster → gqmu is a splat of qmu[c].
    intra: Vec<bool>,
    cluster_groups: Vec<Vec<usize>>,
}

fn build_layout(perm: &[usize], body_src: &[f32]) -> Layout {
    let nb = DIM / BLOCK;
    let ngroups = N / LANES;
    let mut body = vec![0.0f32; N * DIM];
    let mut suf = vec![0.0f32; ngroups * (nb + 1) * LANES];
    let mut cand_ids = vec![0usize; N];
    let mut cand_cluster = vec![0usize; N];
    let mut intra = vec![true; ngroups];
    for g in 0..ngroups {
        let c0 = perm[g * LANES] % CLUSTERS;
        for lane in 0..LANES {
            let cand = perm[g * LANES + lane];
            cand_ids[g * LANES + lane] = cand;
            cand_cluster[g * LANES + lane] = cand % CLUSTERS;
            if cand % CLUSTERS != c0 {
                intra[g] = false;
            }
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
    }
    let mut cluster_groups: Vec<Vec<usize>> = vec![Vec::new(); CLUSTERS];
    for g in 0..ngroups {
        cluster_groups[cand_ids[g * LANES] % CLUSTERS].push(g);
    }
    Layout {
        body,
        suf,
        cand_ids,
        cand_cluster,
        intra,
        cluster_groups,
    }
}

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

#[inline]
fn maxlane(v: f32x8) -> f32 {
    v.to_array()
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max)
}

/// BASE (== 8d13097): single accumulator, per-lane gqmu gather always.
fn topk_base(
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
            let cutoff = if full {
                top[k - 1].0
            } else {
                f32::NEG_INFINITY
            };
            for lane in 0..LANES {
                gq[lane] = qmu[lay.cand_cluster[g * LANES + lane]];
            }
            let gqmu = f32x8::from(gq);
            let mut acc = f32x8::splat(0.0);
            let mut abandoned = false;
            for b in 0..nb {
                if full {
                    sn.copy_from_slice(&lay.suf[sbase + b * LANES..sbase + b * LANES + LANES]);
                    let bound = gqmu + acc + f32x8::splat(qsuf[b]) * f32x8::from(sn);
                    if maxlane(bound) <= cutoff {
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

/// SPLAT: single accumulator, gqmu splat for intra-cluster groups (gather only at boundaries).
fn topk_splat(
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
        let qc = qmu[c];
        for &g in &lay.cluster_groups[c] {
            let vbase = g * DIM * LANES;
            let sbase = g * (nb + 1) * LANES;
            let full = top.len() == k;
            let cutoff = if full {
                top[k - 1].0
            } else {
                f32::NEG_INFINITY
            };
            let gqmu = if lay.intra[g] {
                f32x8::splat(qc)
            } else {
                for lane in 0..LANES {
                    gq[lane] = qmu[lay.cand_cluster[g * LANES + lane]];
                }
                f32x8::from(gq)
            };
            let mut acc = f32x8::splat(0.0);
            let mut abandoned = false;
            for b in 0..nb {
                if full {
                    sn.copy_from_slice(&lay.suf[sbase + b * LANES..sbase + b * LANES + LANES]);
                    let bound = gqmu + acc + f32x8::splat(qsuf[b]) * f32x8::from(sn);
                    if maxlane(bound) <= cutoff {
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

/// FAST: gqmu splat + 4-accumulator block dot (break the FMA latency chain).
fn topk_fast(
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
    let mut sn = [0.0f32; LANES];
    let mut gq = [0.0f32; LANES];
    let mut c0 = [0.0f32; LANES];
    let mut c1 = [0.0f32; LANES];
    let mut c2 = [0.0f32; LANES];
    let mut c3 = [0.0f32; LANES];
    for &c in order {
        let qc = qmu[c];
        for &g in &lay.cluster_groups[c] {
            let vbase = g * DIM * LANES;
            let sbase = g * (nb + 1) * LANES;
            let full = top.len() == k;
            let cutoff = if full {
                top[k - 1].0
            } else {
                f32::NEG_INFINITY
            };
            let gqmu = if lay.intra[g] {
                f32x8::splat(qc)
            } else {
                for lane in 0..LANES {
                    gq[lane] = qmu[lay.cand_cluster[g * LANES + lane]];
                }
                f32x8::from(gq)
            };
            let mut acc0 = f32x8::splat(0.0);
            let mut acc1 = f32x8::splat(0.0);
            let mut acc2 = f32x8::splat(0.0);
            let mut acc3 = f32x8::splat(0.0);
            let mut abandoned = false;
            for b in 0..nb {
                if full {
                    let combined = (acc0 + acc1) + (acc2 + acc3);
                    sn.copy_from_slice(&lay.suf[sbase + b * LANES..sbase + b * LANES + LANES]);
                    let bound = gqmu + combined + f32x8::splat(qsuf[b]) * f32x8::from(sn);
                    if maxlane(bound) <= cutoff {
                        abandoned = true;
                        break;
                    }
                }
                let base = b * BLOCK;
                let mut j = 0;
                while j < BLOCK {
                    let p = vbase + (base + j) * LANES;
                    c0.copy_from_slice(&lay.body[p..p + LANES]);
                    c1.copy_from_slice(&lay.body[p + LANES..p + 2 * LANES]);
                    c2.copy_from_slice(&lay.body[p + 2 * LANES..p + 3 * LANES]);
                    c3.copy_from_slice(&lay.body[p + 3 * LANES..p + 4 * LANES]);
                    acc0 += f32x8::splat(rq[base + j]) * f32x8::from(c0);
                    acc1 += f32x8::splat(rq[base + j + 1]) * f32x8::from(c1);
                    acc2 += f32x8::splat(rq[base + j + 2]) * f32x8::from(c2);
                    acc3 += f32x8::splat(rq[base + j + 3]) * f32x8::from(c3);
                    j += 4;
                }
                blocks_done += LANES as u64;
            }
            if !abandoned {
                let combined = (acc0 + acc1) + (acc2 + acc3);
                let scores = (gqmu + combined).to_array();
                for lane in 0..LANES {
                    offer(&mut top, k, scores[lane], lay.cand_ids[g * LANES + lane]);
                }
            }
        }
    }
    (top, blocks_done)
}

fn bench_ilp(c: &mut Criterion) {
    assert_eq!(N % LANES, 0);
    assert_eq!(BLOCK % 4, 0);
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
    let mut perm: Vec<usize> = (0..N).collect();
    perm.sort_by(|&a, &b| {
        let (ca, cb) = (a % CLUSTERS, b % CLUSTERS);
        ca.cmp(&cb)
            .then(dist[a].partial_cmp(&dist[b]).unwrap())
            .then(a.cmp(&b))
    });
    let lay = build_layout(&perm, &resid);

    let nb = DIM / BLOCK;
    let mut group = c.benchmark_group("residual_ilp_scan");
    for &k in &KS {
        let mut swaps = 0usize;
        for (qi, rq) in rqueries.iter().enumerate() {
            let full = topk_full(rq, &rvecs, k);
            let (qmu, ord) = query_centroids(rq, &cent);
            let (base, _) = topk_base(rq, &qsuf[qi], &lay, &qmu, &ord, k);
            let (splat, _) = topk_splat(rq, &qsuf[qi], &lay, &qmu, &ord, k);
            let (fast, _) = topk_fast(rq, &qsuf[qi], &lay, &qmu, &ord, k);
            let fs: Vec<f32> = full.iter().map(|&(s, _)| s).collect();
            for (label, r) in [("base", &base), ("splat", &splat), ("fast", &fast)] {
                let ts: Vec<f32> = r.iter().map(|&(s, _)| s).collect();
                for j in 0..k {
                    let tol = 1e-4 * fs[j].abs().max(1.0);
                    assert!(
                        (fs[j] - ts[j]).abs() <= tol,
                        "{label} score mismatch rank {j} (k={k},q={qi})"
                    );
                }
            }
            let fids: std::collections::HashSet<usize> = full.iter().map(|&(_, i)| i).collect();
            swaps += fast.iter().filter(|&&(_, i)| !fids.contains(&i)).count();
        }
        eprintln!(
            "[residual_ilp] k={k}: boundary id-swaps (fast)={swaps}/{}",
            QUERIES * k
        );
        let _ = nb;

        let mut qi = 0usize;
        group.bench_function(format!("full_k{k}"), |b| {
            b.iter(|| {
                let rq = &rqueries[qi % QUERIES];
                qi += 1;
                black_box(topk_full(black_box(rq), &rvecs, k))
            });
        });
        for (name, f) in [
            ("residual_base", 0u8),
            ("residual_splat", 1u8),
            ("residual_fast", 2u8),
        ] {
            let mut qi = 0usize;
            group.bench_function(format!("{name}_k{k}"), |b| {
                b.iter(|| {
                    let rq = &rqueries[qi % QUERIES];
                    let qs = &qsuf[qi % QUERIES];
                    qi += 1;
                    let (qmu, ord) = query_centroids(black_box(rq), &cent);
                    let out = match f {
                        0 => topk_base(black_box(rq), qs, &lay, &qmu, &ord, k),
                        1 => topk_splat(black_box(rq), qs, &lay, &qmu, &ord, k),
                        _ => topk_fast(black_box(rq), qs, &lay, &qmu, &ord, k),
                    };
                    black_box(out)
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_ilp);
criterion_main!(benches);
