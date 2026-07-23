//! Per-candidate CENTROID-DISTANCE pre-filter on a row-major query-directed scan —
//! can we abandon far candidates at ZERO blocks (no dot at all)?
//!
//! The query-directed transposed traversal (`cluster_ordered_scan`, `2945ed8`) is
//! block-floor-bound: every far candidate still computes block 0 (its group can't
//! skip individual lanes). Idea to break that floor: precompute per candidate
//! `d_v = ‖v − μ_{c(v)}‖`; since `q·v = q·μ_c + q·(v−μ_c) ≤ q·μ_c + ‖q‖·d_v`, a
//! candidate can be skipped with NO dot when `q·μ_{c(v)} + d_v ≤ cutoff`. This is a
//! valid EXACT upper bound, and per-candidate (so tighter than the failed cluster-
//! level max-radius bound — that used one loose radius for the whole cluster).
//! Requires a ROW-MAJOR scan (per-candidate skip), which trades away the transposed
//! reduce-elimination — whether the trade wins is the open question this measures.
//!
//! Arms: `full` (flat ORIG) vs `rowmajor_ord` (row-major + query-directed traversal,
//! no prefilter) vs `rowmajor_ord_prefilter` (+ the per-candidate zero-block skip).
//! Swept over k∈{10,100}. Compare the best row-major arm to the known transposed
//! `2945ed8` (0.242×/0.272×) to decide if per-candidate skips beat the group scan.
//!
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-index --profile release \
//!     --bench rowmajor_prefilter_scan
//! ```

#![allow(clippy::cast_lossless, clippy::needless_range_loop)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use wide::f32x8;

const N: usize = 50_000;
const DIM: usize = 384;
const QUERIES: usize = 32;
const CLUSTERS: usize = 64;
const NOISE: f32 = 0.07;
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

/// `q·μ_c` for all clusters, plus cluster ids sorted by it (near first). 64 dots.
fn centroid_dots(rq: &[f32], cent: &[f32]) -> (Vec<f32>, Vec<usize>) {
    let qmu: Vec<f32> = (0..CLUSTERS)
        .map(|c| dot_block(rq, &cent[c * DIM..(c + 1) * DIM], DIM))
        .collect();
    let mut ord: Vec<usize> = (0..CLUSTERS).collect();
    ord.sort_by(|&a, &b| qmu[b].partial_cmp(&qmu[a]).unwrap());
    (qmu, ord)
}

/// Row-major query-directed early-abandon. `prefilter`: if true, skip a candidate
/// at zero blocks when `q·μ_{c} + d_v ≤ cutoff`.
fn topk_rowmajor(
    rq: &[f32],
    qsuf: &[f32],
    rvecs: &[f32],
    rvsuf: &[f32],
    dist: &[f32],
    qmu: &[f32],
    order: &[usize],
    members: &[Vec<usize>],
    k: usize,
    prefilter: bool,
) -> (Vec<(f32, usize)>, u64) {
    let nb = DIM / BLOCK;
    let mut top: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
    let mut blocks_done: u64 = 0;
    for &c in order {
        let qc = qmu[c];
        for &i in &members[c] {
            let full = top.len() == k;
            let cutoff = if full {
                top[k - 1].0
            } else {
                f32::NEG_INFINITY
            };
            if prefilter && full && qc + dist[i] <= cutoff {
                continue; // exact zero-block skip
            }
            let v = &rvecs[i * DIM..(i + 1) * DIM];
            let vs = &rvsuf[i * (nb + 1)..(i + 1) * (nb + 1)];
            let mut partial = 0.0f32;
            let mut abandoned = false;
            for b in 0..nb {
                partial += dot_block(&rq[b * BLOCK..], &v[b * BLOCK..], BLOCK);
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
    }
    (top, blocks_done)
}

fn bench_rowmajor_prefilter(c: &mut Criterion) {
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

    let nb = DIM / BLOCK;
    let mut rvsuf = vec![0.0f32; N * (nb + 1)];
    for i in 0..N {
        let s = suffix_norms(&rvecs[i * DIM..(i + 1) * DIM], BLOCK);
        rvsuf[i * (nb + 1)..(i + 1) * (nb + 1)].copy_from_slice(&s);
    }
    let qsuf: Vec<Vec<f32>> = rqueries.iter().map(|q| suffix_norms(q, BLOCK)).collect();

    // Per-cluster centroids (reordered space) + per-candidate distance to own centroid.
    let mut cent = vec![0.0f32; CLUSTERS * DIM];
    let mut counts = vec![0usize; CLUSTERS];
    for i in 0..N {
        let cl = i % CLUSTERS;
        for d in 0..DIM {
            cent[cl * DIM + d] += rvecs[i * DIM + d];
        }
        counts[cl] += 1;
    }
    for cl in 0..CLUSTERS {
        for d in 0..DIM {
            cent[cl * DIM + d] /= counts[cl] as f32;
        }
    }
    let mut dist = vec![0.0f32; N];
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); CLUSTERS];
    for i in 0..N {
        let cl = i % CLUSTERS;
        let mut s = 0.0f32;
        for d in 0..DIM {
            let e = rvecs[i * DIM + d] - cent[cl * DIM + d];
            s += e * e;
        }
        dist[i] = s.sqrt();
        members[cl].push(i);
    }

    let mut group = c.benchmark_group("rowmajor_prefilter_scan");
    for &k in &KS {
        // ── parity (score-equivalence) + blocks/skip report ──
        let mut blk_ord = 0u64;
        let mut blk_pf = 0u64;
        for (qi, rq) in rqueries.iter().enumerate() {
            let full = topk_full(rq, &rvecs, k);
            let (qmu, ord) = centroid_dots(rq, &cent);
            let (o, bo) = topk_rowmajor(
                rq, &qsuf[qi], &rvecs, &rvsuf, &dist, &qmu, &ord, &members, k, false,
            );
            let (p, bp) = topk_rowmajor(
                rq, &qsuf[qi], &rvecs, &rvsuf, &dist, &qmu, &ord, &members, k, true,
            );
            blk_ord += bo;
            blk_pf += bp;
            let fs: Vec<f32> = full.iter().map(|&(s, _)| s).collect();
            for (label, res) in [("ord", &o), ("pf", &p)] {
                let ts: Vec<f32> = res.iter().map(|&(s, _)| s).collect();
                for j in 0..k {
                    let tol = 1e-4 * fs[j].abs().max(1.0);
                    assert!(
                        (fs[j] - ts[j]).abs() <= tol,
                        "{label} score mismatch rank {j} (k={k},q={qi})"
                    );
                }
            }
        }
        let denom = (QUERIES as f64) * (N as f64) * (nb as f64);
        eprintln!(
            "[rowmajor_pf] k={k}: blocks ord={:.1}% prefilter={:.1}%",
            blk_ord as f64 / denom * 100.0,
            blk_pf as f64 / denom * 100.0
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
        group.bench_function(format!("rowmajor_ord_k{k}"), |b| {
            b.iter(|| {
                let rq = &rqueries[qi % QUERIES];
                let qs = &qsuf[qi % QUERIES];
                qi += 1;
                let (qmu, ord) = centroid_dots(black_box(rq), &cent);
                black_box(topk_rowmajor(
                    black_box(rq),
                    qs,
                    &rvecs,
                    &rvsuf,
                    &dist,
                    &qmu,
                    &ord,
                    &members,
                    k,
                    false,
                ))
            });
        });
        let mut qi = 0usize;
        group.bench_function(format!("rowmajor_ord_prefilter_k{k}"), |b| {
            b.iter(|| {
                let rq = &rqueries[qi % QUERIES];
                let qs = &qsuf[qi % QUERIES];
                qi += 1;
                let (qmu, ord) = centroid_dots(black_box(rq), &cent);
                black_box(topk_rowmajor(
                    black_box(rq),
                    qs,
                    &rvecs,
                    &rvsuf,
                    &dist,
                    &qmu,
                    &ord,
                    &members,
                    k,
                    true,
                ))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_rowmajor_prefilter);
criterion_main!(benches);
