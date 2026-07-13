//! ALGORITHMIC candidate-reduction: IVF (probe P of C clusters, scan only their members) vs the
//! shipped FLAT scan (all N candidates). The vector-scan dot is FMA-throughput-saturated on Zen 3
//! (see NEGATIVE_EVIDENCE 2026-07-13), so the only remaining vector-scan slack is scanning FEWER
//! candidates. IVF is sub-linear (~N·P/C scanned) but pays a fixed overhead (C centroid dots +
//! list indirection). This finds the concrete SPEED CROSSOVER N above which IVF beats flat — the
//! "N ≫ 130k crossover" route-next from `ann-in-bold-viable` (RECALL is a separate, real-corpus
//! question; this measures SPEED only, which is dot-type / recall-independent).
//!
//! Synthetic CLUSTERED data (so IVF has structure to exploit): N vectors in C_true Gaussian
//! clusters; the query sits near one cluster. FLAT and IVF use the SAME f32 dot + bounded top-k;
//! only the candidate SET differs. Within-process paired profile; reports ns/query at each N.
//!
//! Run with:
//! ```bash
//! AGENT_NAME=cc_fse CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cod \
//!   rch exec -- cargo bench -p frankensearch-index --profile release --bench ivf_crossover_ab
//! ```
#![allow(clippy::doc_markdown)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use wide::f32x8;

const PROFILE_ROUNDS: usize = 31;
const DIM: usize = 128;
const K: usize = 10;
const PROBE: usize = 8; // clusters probed per query

/// N candidate counts spanning the suspected crossover.
const SHAPES: &[usize] = &[8_192, 32_768, 131_072, 524_288];

#[inline]
fn load8(s: &[f32]) -> f32x8 {
    f32x8::from([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]])
}

#[inline]
fn dot(query_v: &[f32x8], vec: &[f32], dim: usize) -> f32 {
    let per = dim / 8;
    let mut acc = [f32x8::splat(0.0); 4];
    let quad = per / 4;
    for q in 0..quad {
        let c = q * 4;
        acc[0] += query_v[c] * load8(&vec[c * 8..]);
        acc[1] += query_v[c + 1] * load8(&vec[(c + 1) * 8..]);
        acc[2] += query_v[c + 2] * load8(&vec[(c + 2) * 8..]);
        acc[3] += query_v[c + 3] * load8(&vec[(c + 3) * 8..]);
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
}

struct Corpus {
    query_v: Vec<f32x8>,
    slab: Vec<f32>,          // n*DIM, row-major
    centroids: Vec<f32>,     // c*DIM
    lists: Vec<Vec<usize>>,  // cluster -> member indices
    n: usize,
    c: usize,
}

/// Build N vectors in `c` Gaussian clusters; query near cluster 0.
fn make_corpus(n: usize) -> Corpus {
    let c = (n as f64).sqrt().ceil() as usize; // classic IVF nlist ~= sqrt(N)
    let mut r = Xorshift(0x9E37_79B9_7F4A_7C15);
    // Cluster centers.
    let centroids: Vec<f32> = (0..c * DIM).map(|_| r.next_f32() * 4.0).collect();
    let mut slab = vec![0.0_f32; n * DIM];
    let mut lists: Vec<Vec<usize>> = vec![Vec::new(); c];
    for i in 0..n {
        let cl = (r.0 as usize) % c;
        // advance r deterministically
        r.next_f32();
        for d in 0..DIM {
            slab[i * DIM + d] = centroids[cl * DIM + d] + r.next_f32() * 0.3;
        }
        lists[cl].push(i);
    }
    // Query near cluster 0's centroid.
    let mut query = vec![0.0_f32; DIM];
    for d in 0..DIM {
        query[d] = centroids[d] + r.next_f32() * 0.3;
    }
    let query_v: Vec<f32x8> = (0..DIM / 8).map(|k| load8(&query[k * 8..])).collect();
    Corpus { query_v, slab, centroids, lists, n, c }
}

fn retain(heap: &mut Vec<(f32, usize)>, score: f32, idx: usize, cutoff: &mut f32) {
    if heap.len() < K {
        heap.push((score, idx));
        if heap.len() == K {
            *cutoff = heap.iter().map(|&(s, _)| s).fold(f32::INFINITY, f32::min);
        }
    } else if score > *cutoff {
        // replace the current min entry (avoid nested-tuple `a.1.0` parse gotcha)
        let mut mi = 0;
        for j in 1..heap.len() {
            if heap[j].0 < heap[mi].0 {
                mi = j;
            }
        }
        heap[mi] = (score, idx);
        *cutoff = heap.iter().map(|&(s, _)| s).fold(f32::INFINITY, f32::min);
    }
}

fn scan_flat(c: &Corpus) -> usize {
    let mut heap: Vec<(f32, usize)> = Vec::with_capacity(K + 1);
    let mut cutoff = f32::NEG_INFINITY;
    for i in 0..c.n {
        let s = dot(&c.query_v, &c.slab[i * DIM..], DIM);
        retain(&mut heap, s, i, &mut cutoff);
    }
    heap.len()
}

fn scan_ivf(c: &Corpus) -> usize {
    // Rank clusters by query·centroid, take the PROBE nearest.
    let mut cdist: Vec<(f32, usize)> = (0..c.c)
        .map(|cl| (dot(&c.query_v, &c.centroids[cl * DIM..], DIM), cl))
        .collect();
    let take = PROBE.min(c.c);
    cdist.select_nth_unstable_by(take - 1, |a, b| b.0.total_cmp(&a.0));
    let mut heap: Vec<(f32, usize)> = Vec::with_capacity(K + 1);
    let mut cutoff = f32::NEG_INFINITY;
    for &(_, cl) in cdist.iter().take(take) {
        for &i in &c.lists[cl] {
            let s = dot(&c.query_v, &c.slab[i * DIM..], DIM);
            retain(&mut heap, s, i, &mut cutoff);
        }
    }
    heap.len()
}

fn median_ns(c: &Corpus, ivf: bool) -> f64 {
    for _ in 0..3 {
        black_box(if ivf { scan_ivf(c) } else { scan_flat(c) });
    }
    let mut samples: Vec<Duration> = (0..PROFILE_ROUNDS)
        .map(|_| {
            let s = Instant::now();
            black_box(if ivf { scan_ivf(c) } else { scan_flat(c) });
            s.elapsed()
        })
        .collect();
    samples.sort_unstable();
    samples[samples.len() / 2].as_secs_f64() * 1e9
}

fn main() {
    eprintln!("[config] dim={DIM} k={K} probe={PROBE} nlist=sqrt(N) profile_rounds={PROFILE_ROUNDS}");
    let mut crossover = None;
    for &n in SHAPES {
        let c = make_corpus(n);
        // Measure IVF's scanned fraction (candidate-reduction).
        let mut cdist: Vec<(f32, usize)> = (0..c.c)
            .map(|cl| (dot(&c.query_v, &c.centroids[cl * DIM..], DIM), cl))
            .collect();
        let take = PROBE.min(c.c);
        cdist.select_nth_unstable_by(take - 1, |a, b| b.0.total_cmp(&a.0));
        let scanned: usize = cdist.iter().take(take).map(|&(_, cl)| c.lists[cl].len()).sum();

        let flat_ns = median_ns(&c, false);
        let ivf_ns = median_ns(&c, true);
        let speedup = flat_ns / ivf_ns;
        if speedup > 1.0 && crossover.is_none() {
            crossover = Some(n);
        }
        eprintln!(
            "[profile] n={n} nlist={} scanned={scanned} ({:.1}% of N) flat_us={:.1} ivf_us={:.1} speedup={speedup:.2}x",
            c.c,
            scanned as f64 / n as f64 * 100.0,
            flat_ns / 1000.0,
            ivf_ns / 1000.0
        );
    }
    eprintln!(
        "[crossover] IVF first beats flat at N={} (probe={PROBE}, nlist=sqrt(N))",
        crossover.map_or_else(|| "none-in-range".to_string(), |n| n.to_string())
    );
}
