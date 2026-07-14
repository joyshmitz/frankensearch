//! Captures the IVF recall@k-vs-probe Pareto through the TEST channel (this session's rch drops
//! `cargo bench` stderr on retrieval but returns `cargo test` output). Recall is deterministic
//! (fixed synthetic data + k-means seed), so a test that asserts monotonicity + a probe-32 floor
//! is a stable validation; the exact values print via `-- --nocapture`, and a failing assert
//! surfaces the actual number. Small N so it runs in well under a second.
//!
//! Run: `AGENT_NAME=cc_fse env -u CARGO_TARGET_DIR rch exec -- \
//!   cargo test -p frankensearch-index --release --test ivf_recall_test -- --nocapture`

const DIM: usize = 64;
const N: usize = 16_384;
const NLIST: usize = 128;
const KMEANS_ITERS: usize = 6;
const K: usize = 10;
const NQUERY: usize = 48;
const NTRUE: usize = 128;

struct Xs(u64);
impl Xs {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1_u32 << 24) as f32 * 2.0 - 1.0
    }
    fn u(&mut self, b: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 33) as usize % b
    }
}

#[inline]
fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for d in 0..DIM {
        let x = a[d] - b[d];
        s += x * x;
    }
    s
}

fn topk_flat(q: &[f32], slab: &[f32]) -> Vec<usize> {
    let mut v: Vec<(f32, usize)> = (0..N).map(|i| (l2sq(q, &slab[i * DIM..]), i)).collect();
    v.select_nth_unstable_by(K - 1, |a, b| a.0.total_cmp(&b.0));
    v.truncate(K);
    v.into_iter().map(|(_, i)| i).collect()
}

fn topk_ivf(q: &[f32], slab: &[f32], cents: &[f32], lists: &[Vec<u32>], probe: usize) -> Vec<usize> {
    let mut cd: Vec<(f32, usize)> = (0..NLIST).map(|c| (l2sq(q, &cents[c * DIM..]), c)).collect();
    cd.select_nth_unstable_by(probe - 1, |a, b| a.0.total_cmp(&b.0));
    let mut best: Vec<(f32, usize)> = Vec::new();
    for &(_, c) in cd.iter().take(probe) {
        for &i in &lists[c] {
            best.push((l2sq(q, &slab[i as usize * DIM..]), i as usize));
        }
    }
    let kk = K.min(best.len());
    if kk > 0 {
        best.select_nth_unstable_by(kk - 1, |a, b| a.0.total_cmp(&b.0));
        best.truncate(kk);
    }
    best.into_iter().map(|(_, i)| i).collect()
}

fn kmeans(slab: &[f32], r: &mut Xs) -> (Vec<f32>, Vec<Vec<u32>>) {
    let mut cents = vec![0.0f32; NLIST * DIM];
    for c in 0..NLIST {
        let p = r.u(N);
        cents[c * DIM..(c + 1) * DIM].copy_from_slice(&slab[p * DIM..(p + 1) * DIM]);
    }
    let mut asg = vec![0u32; N];
    for _ in 0..KMEANS_ITERS {
        for i in 0..N {
            let v = &slab[i * DIM..];
            let (mut bd, mut bc) = (f32::INFINITY, 0u32);
            for c in 0..NLIST {
                let d = l2sq(v, &cents[c * DIM..]);
                if d < bd {
                    bd = d;
                    bc = c as u32;
                }
            }
            asg[i] = bc;
        }
        let mut sums = vec![0.0f32; NLIST * DIM];
        let mut cnt = vec![0u32; NLIST];
        for i in 0..N {
            let c = asg[i] as usize;
            cnt[c] += 1;
            for d in 0..DIM {
                sums[c * DIM + d] += slab[i * DIM + d];
            }
        }
        for c in 0..NLIST {
            if cnt[c] > 0 {
                let inv = 1.0 / cnt[c] as f32;
                for d in 0..DIM {
                    cents[c * DIM + d] = sums[c * DIM + d] * inv;
                }
            }
        }
    }
    let mut lists = vec![Vec::new(); NLIST];
    for i in 0..N {
        lists[asg[i] as usize].push(i as u32);
    }
    (cents, lists)
}

#[test]
fn ivf_recall_pareto() {
    let mut r = Xs(0x9E37_79B9_7F4A_7C15);
    let centers: Vec<f32> = (0..NTRUE * DIM).map(|_| r.f() * 4.0).collect();
    let mut slab = vec![0.0f32; N * DIM];
    for i in 0..N {
        let cl = r.u(NTRUE);
        for d in 0..DIM {
            slab[i * DIM + d] = centers[cl * DIM + d] + r.f() * 0.4;
        }
    }
    let queries: Vec<Vec<f32>> = (0..NQUERY)
        .map(|_| {
            let cl = r.u(NTRUE);
            (0..DIM).map(|d| centers[cl * DIM + d] + r.f() * 0.4).collect()
        })
        .collect();
    let (cents, lists) = kmeans(&slab, &mut r);
    let truth: Vec<Vec<usize>> = queries.iter().map(|q| topk_flat(q, &slab)).collect();

    let probes = [1usize, 2, 4, 8, 16, 32];
    let mut recalls = Vec::new();
    for &p in &probes {
        let mut hit = 0usize;
        for (qi, q) in queries.iter().enumerate() {
            let got = topk_ivf(q, &slab, &cents, &lists, p);
            let t: std::collections::HashSet<usize> = truth[qi].iter().copied().collect();
            hit += got.iter().filter(|i| t.contains(i)).count();
        }
        let recall = hit as f64 / (NQUERY * K) as f64;
        // Fraction of N scanned at this probe (avg list ~ N/NLIST).
        let scanned: usize = {
            let mut cd: Vec<(f32, usize)> =
                (0..NLIST).map(|c| (l2sq(&queries[0], &cents[c * DIM..]), c)).collect();
            cd.select_nth_unstable_by(p - 1, |a, b| a.0.total_cmp(&b.0));
            cd.iter().take(p).map(|&(_, c)| lists[c].len()).sum()
        };
        println!(
            "[ivf-pareto] probe={p} recall@{K}={recall:.4} scanned={scanned} ({:.1}% of N, ~{:.0}x fewer dots)",
            scanned as f64 / N as f64 * 100.0,
            N as f64 / scanned.max(1) as f64
        );
        recalls.push(recall);
    }
    // Monotone non-decreasing in probe (more clusters can only add candidates).
    for w in recalls.windows(2) {
        assert!(w[1] >= w[0] - 1e-9, "recall not monotone in probe: {recalls:?}");
    }
    // Probe-32 (of 128 lists) should recover essentially all true neighbours on well-clustered data.
    let r32 = *recalls.last().unwrap();
    assert!(r32 >= 0.90, "probe=32 recall too low: {r32:.4} (full pareto: {recalls:?})");
}
