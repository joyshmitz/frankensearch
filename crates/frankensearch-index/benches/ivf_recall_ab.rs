//! IMPLEMENT + VALIDATE the IVF lever: a real IVF (k-means clustering → inverted lists → probe-P
//! search) with the SPEED/RECALL Pareto. `ivf_crossover_ab` measured only speed (IVF scans
//! ~8/√N of candidates → 40-82× vs flat); the landability question is RECALL — does probing P of
//! C clusters actually return the true top-k? This runs full k-means, builds the index, and for a
//! probe sweep reports recall@k (vs the flat ground truth) AND median ns/query + speedup, so the
//! speed/recall trade is concrete: which probe-P buys acceptable recall, at what speedup.
//!
//! Synthetic CLUSTERED data (Gaussian mixture, so IVF has structure to exploit). L2 nearest-
//! neighbour (min distance): flat = exact top-k over all N; IVF = nearest-P centroids' lists.
//! Recall is measured on synthetic — a real-corpus number will differ, but this is the first
//! end-to-end validation of the mechanism (recall as a function of probe).
//!
//! Run with:
//! ```bash
//! AGENT_NAME=cc_fse env -u CARGO_TARGET_DIR \
//!   rch exec -- cargo bench -p frankensearch-index --profile release --bench ivf_recall_ab
//! ```
#![allow(
    clippy::cast_possible_truncation,
    clippy::doc_markdown,
    clippy::many_single_char_names
)]

use std::hint::black_box;
use std::time::Instant;

use wide::f32x8;

// Sized so k-means (N·NLIST·ITERS distance calls) finishes fast; the RECALL-vs-probe curve is
// the finding here (the speed/N crossover is in ivf_crossover_ab). NLIST≈√N keeps ~N/NLIST per list.
const DIM: usize = 128;
const N: usize = 32_768;
const NLIST: usize = 181; // ~sqrt(N)
const KMEANS_ITERS: usize = 6;
const K: usize = 10;
const NQUERY: usize = 64;
const PROBES: &[usize] = &[1, 2, 4, 8, 16, 32, 64];

#[inline]
fn load8(s: &[f32]) -> f32x8 {
    f32x8::from([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]])
}

/// Squared L2 distance (f32x8, 4-acc).
#[inline]
fn l2sq(a: &[f32], b: &[f32], dim: usize) -> f32 {
    let per = dim / 8;
    let mut acc = [f32x8::splat(0.0); 4];
    let quad = per / 4;
    for q in 0..quad {
        let c = q * 4;
        for (l, off) in (0..4).map(|l| (l, (c + l) * 8)) {
            let d = load8(&a[off..]) - load8(&b[off..]);
            acc[l] += d * d;
        }
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])).reduce_add()
}

struct Xorshift(u64);
impl Xorshift {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1_u32 << 24) as f32 * 2.0 - 1.0
    }
    fn next_usize(&mut self, bound: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 33) as usize % bound
    }
}

/// Ground-truth top-K indices (smallest L2), ascending distance.
fn flat_topk(query: &[f32], slab: &[f32], n: usize) -> Vec<usize> {
    let mut best: Vec<(f32, usize)> = Vec::with_capacity(K + 1);
    let mut cutoff = f32::INFINITY;
    for i in 0..n {
        let d = l2sq(query, &slab[i * DIM..], DIM);
        if best.len() < K {
            best.push((d, i));
            if best.len() == K {
                cutoff = best
                    .iter()
                    .map(|&(x, _)| x)
                    .fold(f32::NEG_INFINITY, f32::max);
            }
        } else if d < cutoff {
            let mut mi = 0;
            for j in 1..best.len() {
                if best[j].0 > best[mi].0 {
                    mi = j;
                }
            }
            best[mi] = (d, i);
            cutoff = best
                .iter()
                .map(|&(x, _)| x)
                .fold(f32::NEG_INFINITY, f32::max);
        }
    }
    best.sort_by(|a, b| a.0.total_cmp(&b.0));
    best.into_iter().map(|(_, i)| i).collect()
}

/// IVF top-K: nearest `probe` centroids (by L2), scan their lists.
fn ivf_topk(
    query: &[f32],
    slab: &[f32],
    centroids: &[f32],
    lists: &[Vec<u32>],
    probe: usize,
) -> Vec<usize> {
    let mut cd: Vec<(f32, usize)> = (0..NLIST)
        .map(|c| (l2sq(query, &centroids[c * DIM..], DIM), c))
        .collect();
    let take = probe.min(NLIST);
    cd.select_nth_unstable_by(take - 1, |a, b| a.0.total_cmp(&b.0));
    let mut best: Vec<(f32, usize)> = Vec::with_capacity(K + 1);
    let mut cutoff = f32::INFINITY;
    for &(_, c) in cd.iter().take(take) {
        for &i in &lists[c] {
            let i = i as usize;
            let d = l2sq(query, &slab[i * DIM..], DIM);
            if best.len() < K {
                best.push((d, i));
                if best.len() == K {
                    cutoff = best
                        .iter()
                        .map(|&(x, _)| x)
                        .fold(f32::NEG_INFINITY, f32::max);
                }
            } else if d < cutoff {
                let mut mi = 0;
                for j in 1..best.len() {
                    if best[j].0 > best[mi].0 {
                        mi = j;
                    }
                }
                best[mi] = (d, i);
                cutoff = best
                    .iter()
                    .map(|&(x, _)| x)
                    .fold(f32::NEG_INFINITY, f32::max);
            }
        }
    }
    best.into_iter().map(|(_, i)| i).collect()
}

fn kmeans(slab: &[f32], n: usize, r: &mut Xorshift) -> (Vec<f32>, Vec<Vec<u32>>) {
    // Init: NLIST distinct random points as centroids.
    let mut centroids = vec![0.0_f32; NLIST * DIM];
    for c in 0..NLIST {
        let p = r.next_usize(n);
        centroids[c * DIM..(c + 1) * DIM].copy_from_slice(&slab[p * DIM..(p + 1) * DIM]);
    }
    let mut assign = vec![0_u32; n];
    for _ in 0..KMEANS_ITERS {
        for i in 0..n {
            let v = &slab[i * DIM..];
            let mut bd = f32::INFINITY;
            let mut bc = 0u32;
            for c in 0..NLIST {
                let d = l2sq(v, &centroids[c * DIM..], DIM);
                if d < bd {
                    bd = d;
                    bc = c as u32;
                }
            }
            assign[i] = bc;
        }
        let mut sums = vec![0.0_f32; NLIST * DIM];
        let mut counts = vec![0u32; NLIST];
        for i in 0..n {
            let c = assign[i] as usize;
            counts[c] += 1;
            let (s, v) = (&mut sums[c * DIM..(c + 1) * DIM], &slab[i * DIM..]);
            for d in 0..DIM {
                s[d] += v[d];
            }
        }
        for c in 0..NLIST {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f32;
                for d in 0..DIM {
                    centroids[c * DIM + d] = sums[c * DIM + d] * inv;
                }
            }
        }
    }
    let mut lists: Vec<Vec<u32>> = vec![Vec::new(); NLIST];
    for i in 0..n {
        lists[assign[i] as usize].push(i as u32);
    }
    (centroids, lists)
}

fn main() {
    eprintln!(
        "[config] dim={DIM} n={N} nlist={NLIST} kmeans_iters={KMEANS_ITERS} k={K} nquery={NQUERY}"
    );
    let mut r = Xorshift(0x9E37_79B9_7F4A_7C15);
    // Clustered corpus.
    let ntrue = 256;
    let centers: Vec<f32> = (0..ntrue * DIM).map(|_| r.next_f32() * 4.0).collect();
    let mut slab = vec![0.0_f32; N * DIM];
    for i in 0..N {
        let cl = r.next_usize(ntrue);
        for d in 0..DIM {
            slab[i * DIM + d] = centers[cl * DIM + d] + r.next_f32() * 0.4;
        }
    }
    // Queries near random true centers.
    let queries: Vec<Vec<f32>> = (0..NQUERY)
        .map(|_| {
            let cl = r.next_usize(ntrue);
            (0..DIM)
                .map(|d| centers[cl * DIM + d] + r.next_f32() * 0.4)
                .collect()
        })
        .collect();

    let build0 = Instant::now();
    let (centroids, lists) = kmeans(&slab, N, &mut r);
    eprintln!("[build] kmeans {:.2}s", build0.elapsed().as_secs_f64());

    // Ground truth.
    let truth: Vec<Vec<usize>> = queries.iter().map(|q| flat_topk(q, &slab, N)).collect();

    // Flat timing.
    let f0 = Instant::now();
    for q in &queries {
        black_box(flat_topk(q, &slab, N));
    }
    let flat_ns = f0.elapsed().as_secs_f64() * 1e9 / NQUERY as f64;
    eprintln!("[flat] ns/query={flat_ns:.0}");

    for &probe in PROBES {
        let mut hit = 0usize;
        for (qi, q) in queries.iter().enumerate() {
            let got = ivf_topk(q, &slab, &centroids, &lists, probe);
            let t: std::collections::HashSet<usize> = truth[qi].iter().copied().collect();
            hit += got.iter().filter(|i| t.contains(i)).count();
        }
        let recall = hit as f64 / (NQUERY * K) as f64;
        let p0 = Instant::now();
        for q in &queries {
            black_box(ivf_topk(q, &slab, &centroids, &lists, probe));
        }
        let ivf_ns = p0.elapsed().as_secs_f64() * 1e9 / NQUERY as f64;
        eprintln!(
            "[pareto] probe={probe} recall@{K}={recall:.4} ivf_ns/query={ivf_ns:.0} speedup_vs_flat={:.1}x",
            flat_ns / ivf_ns
        );
    }
}
