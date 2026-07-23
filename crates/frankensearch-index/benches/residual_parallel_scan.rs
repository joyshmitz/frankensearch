//! Does the exact early-abandon arc SURVIVE parallelism? — the production-relevant test.
//!
//! The whole arc (`2945ed8`→`0c107f3`, 8.3×/7.5× vs flat) was benched SERIAL-vs-serial.
//! But the real product `InMemoryVectorIndex::search_top_k` is ALREADY rayon-parallel
//! (NEGATIVE_EVIDENCE 2026-06-26/24). On a 64-core box, parallel flat (~40-64× over
//! serial flat) can beat a serial 8.3× algorithmic win outright — so the question that
//! matters is: does a PARALLEL early-abandon beat the already-parallel flat scan, or is
//! the algorithmic win subsumed by parallelism?
//!
//! PARALLEL EARLY-ABANDON (hybrid): serial-scan the near-cluster CORE (first `SEED`
//! groups of the query-directed order, which are the true-top-k-dense low-residual
//! groups) to seed a tight cutoff; then rayon-parallel-scan all remaining groups with
//! that cutoff as each chunk's initial heap. Exact: a chunk abandons only when its bound
//! ≤ its local cutoff (≥ the seed cutoff, ≤ the true k-th) → never drops a true top-k
//! (recall 1.0). The seed cutoff is looser than a fully-tightened serial cutoff, so
//! chunks over-scan slightly — parallelism repays it many times over.
//!
//! Arms (k∈{10,100}): `full_serial` (serial flat, the arc's OLD orig) · `full_parallel`
//! (parallel flat, the PRODUCTION orig) · `residual_serial` (= `0c107f3`) ·
//! `residual_parallel` (new). Key ratios: residual_parallel / full_parallel (does the
//! arc beat the real product?) and residual_parallel / residual_serial (parallel speedup).
//!
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-index --profile release \
//!     --bench residual_parallel_scan
//! ```

#![allow(
    clippy::cast_lossless,
    clippy::doc_markdown,
    clippy::needless_range_loop,
    clippy::range_plus_one,
    clippy::similar_names
)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use rayon::prelude::*;
use wide::f32x8;

const N: usize = 50_000;
const DIM: usize = 384;
const QUERIES: usize = 32;
const CLUSTERS: usize = 64;
const NOISE: f32 = 0.07;
const LANES: usize = 8;
const BLOCK: usize = 32;
const SEED: usize = 32; // near-core groups scanned serially to seed the cutoff
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

/// Parallel flat scan — the PRODUCTION baseline (chunk the candidates, local top-k, merge).
fn topk_full_par(rq: &[f32], rvecs: &[f32], k: usize) -> Vec<(f32, usize)> {
    let nt = rayon::current_num_threads().max(1);
    let csz = N.div_ceil(nt);
    let mut locals: Vec<Vec<(f32, usize)>> = (0..nt)
        .into_par_iter()
        .map(|t| {
            let lo = t * csz;
            let hi = ((t + 1) * csz).min(N);
            let mut top = Vec::with_capacity(k + 1);
            for i in lo..hi {
                offer(
                    &mut top,
                    k,
                    dot_block(rq, &rvecs[i * DIM..(i + 1) * DIM], DIM),
                    i,
                );
            }
            top
        })
        .collect();
    let mut top = locals.pop().unwrap_or_default();
    for l in locals {
        for (s, i) in l {
            offer(&mut top, k, s, i);
        }
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

/// Scan a list of groups (residual body, gqmu-splat kernel = `0c107f3`) starting from
/// `top`; returns the refined top-k + blocks computed. Pure over `top` → parallel-safe.
fn scan_group_list(
    rq: &[f32],
    qsuf: &[f32],
    lay: &Layout,
    qmu: &[f32],
    groups: &[usize],
    k: usize,
    mut top: Vec<(f32, usize)>,
) -> (Vec<(f32, usize)>, u64) {
    let nb = DIM / BLOCK;
    let mut blocks: u64 = 0;
    let mut chunk = [0.0f32; LANES];
    let mut sn = [0.0f32; LANES];
    let mut gq = [0.0f32; LANES];
    for &g in groups {
        let vbase = g * DIM * LANES;
        let sbase = g * (nb + 1) * LANES;
        let full = top.len() == k;
        let cutoff = if full {
            top[k - 1].0
        } else {
            f32::NEG_INFINITY
        };
        let gqmu = if lay.intra[g] {
            f32x8::splat(qmu[lay.cand_cluster[g * LANES]])
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
            for d in b * BLOCK..(b + 1) * BLOCK {
                chunk.copy_from_slice(&lay.body[vbase + d * LANES..vbase + d * LANES + LANES]);
                acc += f32x8::splat(rq[d]) * f32x8::from(chunk);
            }
            blocks += LANES as u64;
        }
        if !abandoned {
            let scores = (gqmu + acc).to_array();
            for lane in 0..LANES {
                offer(&mut top, k, scores[lane], lay.cand_ids[g * LANES + lane]);
            }
        }
    }
    (top, blocks)
}

fn ordered_groups(lay: &Layout, order: &[usize]) -> Vec<usize> {
    order
        .iter()
        .flat_map(|&c| lay.cluster_groups[c].iter().copied())
        .collect()
}

/// SERIAL residual early-abandon (= `0c107f3`): one pass over the ordered groups.
fn topk_residual_serial(
    rq: &[f32],
    qsuf: &[f32],
    lay: &Layout,
    qmu: &[f32],
    order: &[usize],
    k: usize,
) -> (Vec<(f32, usize)>, u64) {
    let og = ordered_groups(lay, order);
    scan_group_list(rq, qsuf, lay, qmu, &og, k, Vec::with_capacity(k + 1))
}

/// PARALLEL residual early-abandon: serial seed of the near core, then parallel chunks.
fn topk_residual_par(
    rq: &[f32],
    qsuf: &[f32],
    lay: &Layout,
    qmu: &[f32],
    order: &[usize],
    k: usize,
) -> (Vec<(f32, usize)>, u64) {
    let og = ordered_groups(lay, order);
    let sn = SEED.min(og.len());
    let (seed_top, seed_blk) =
        scan_group_list(rq, qsuf, lay, qmu, &og[0..sn], k, Vec::with_capacity(k + 1));
    let rest = &og[sn..];
    if rest.is_empty() {
        return (seed_top, seed_blk);
    }
    let nt = rayon::current_num_threads().max(1);
    let csz = rest.len().div_ceil(nt).max(1);
    let init = seed_top.clone();
    let results: Vec<(Vec<(f32, usize)>, u64)> = rest
        .par_chunks(csz)
        .map(|ch| scan_group_list(rq, qsuf, lay, qmu, ch, k, init.clone()))
        .collect();
    // Merge with DEDUP by id: each chunk was seeded with `seed_top` (for the cutoff), so
    // its returned heap carries the seed entries — offering them all back would duplicate.
    let mut blk = seed_blk;
    let mut byid: std::collections::HashMap<usize, f32> =
        std::collections::HashMap::with_capacity(k * (results.len() + 1));
    for &(s, i) in &seed_top {
        byid.insert(i, s);
    }
    for (t, b) in &results {
        blk += *b;
        for &(s, i) in t {
            byid.insert(i, s);
        }
    }
    let mut top: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
    for (i, s) in byid {
        offer(&mut top, k, s, i);
    }
    (top, blk)
}

fn bench_parallel(c: &mut Criterion) {
    assert_eq!(N % LANES, 0);
    eprintln!(
        "[residual_parallel] rayon threads = {}",
        rayon::current_num_threads()
    );
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

    let mut group = c.benchmark_group("residual_parallel_scan");
    for &k in &KS {
        // parity: serial + parallel residual + parallel flat vs serial flat
        let mut swaps_s = 0usize;
        let mut swaps_p = 0usize;
        let mut swaps_fp = 0usize;
        for (qi, rq) in rqueries.iter().enumerate() {
            let full = topk_full(rq, &rvecs, k);
            let (qmu, ord) = query_centroids(rq, &cent);
            let (rs, _) = topk_residual_serial(rq, &qsuf[qi], &lay, &qmu, &ord, k);
            let (rp, _) = topk_residual_par(rq, &qsuf[qi], &lay, &qmu, &ord, k);
            let fp = topk_full_par(rq, &rvecs, k);
            let fids: std::collections::HashSet<usize> = full.iter().map(|&(_, i)| i).collect();
            let fs: Vec<f32> = full.iter().map(|&(s, _)| s).collect();
            for (lbl, r) in [("serial", &rs), ("par", &rp)] {
                let ts: Vec<f32> = r.iter().map(|&(s, _)| s).collect();
                for j in 0..k {
                    let tol = 1e-4 * fs[j].abs().max(1.0);
                    assert!(
                        (fs[j] - ts[j]).abs() <= tol,
                        "{lbl} score mismatch rank {j} (k={k},q={qi})"
                    );
                }
            }
            swaps_s += rs.iter().filter(|&&(_, i)| !fids.contains(&i)).count();
            swaps_p += rp.iter().filter(|&&(_, i)| !fids.contains(&i)).count();
            swaps_fp += fp.iter().filter(|&&(_, i)| !fids.contains(&i)).count();
        }
        eprintln!(
            "[residual_parallel] k={k}: swaps serial={swaps_s} par={swaps_p} full_par={swaps_fp} (/{})",
            QUERIES * k
        );

        let mut qi = 0usize;
        group.bench_function(format!("full_serial_k{k}"), |b| {
            b.iter(|| {
                let rq = &rqueries[qi % QUERIES];
                qi += 1;
                black_box(topk_full(black_box(rq), &rvecs, k))
            });
        });
        let mut qi = 0usize;
        group.bench_function(format!("full_parallel_k{k}"), |b| {
            b.iter(|| {
                let rq = &rqueries[qi % QUERIES];
                qi += 1;
                black_box(topk_full_par(black_box(rq), &rvecs, k))
            });
        });
        let mut qi = 0usize;
        group.bench_function(format!("residual_serial_k{k}"), |b| {
            b.iter(|| {
                let rq = &rqueries[qi % QUERIES];
                let qs = &qsuf[qi % QUERIES];
                qi += 1;
                let (qmu, ord) = query_centroids(black_box(rq), &cent);
                black_box(topk_residual_serial(black_box(rq), qs, &lay, &qmu, &ord, k))
            });
        });
        let mut qi = 0usize;
        group.bench_function(format!("residual_parallel_k{k}"), |b| {
            b.iter(|| {
                let rq = &rqueries[qi % QUERIES];
                let qs = &qsuf[qi % QUERIES];
                qi += 1;
                let (qmu, ord) = query_centroids(black_box(rq), &cent);
                black_box(topk_residual_par(black_box(rq), qs, &lay, &qmu, &ord, k))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_parallel);
criterion_main!(benches);
