//! Scan-level A/B (bd-b5wl): does the `vpmaddubs` pass-1 kernel make the WHOLE int8 two-pass scan
//! faster, end-to-end, on a real file-backed `VectorIndex`?
//!
//! The kernel is 1.23× in isolation (`int8_dot_maddubs_ab`) and preserves f32 recall
//! (`simd::tests::maddubs_pass1_preserves_f32_recall_under_real_saturation`). This measures the
//! part that was previously called "undecidable": the scan-level ratio, using the tight
//! alternating-round null control (`frankensearch_core::bench_support`) rather than the
//! contention-wide balanced-pair substrate. Both arms run in ONE routine per round; the ratio is
//! taken per round so worker drift cancels. Gate on the median vs the null p5..p95 spread.
//!
//! RECALL is asserted first (deterministic): the maddubs scan must match the exact-flat top-k
//! recall of the production scan — swapping the pass-1 kernel must not change results.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- cargo bench -p frankensearch-index --features bench-internals --bench int8_scan_maddubs_ab
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_core::bench_support::paired_median_ratio;
use frankensearch_index::VectorIndex;

const DIM: usize = 384;
const N: usize = 40_000;
const CLUSTERS: usize = 32;
const QUERIES: usize = 16;
const K: usize = 10;
const MULT: usize = 5;

fn raw(seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..DIM)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
        })
        .collect()
}

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    for x in &mut v {
        *x /= n;
    }
    v
}

/// A clustered vector: centroid + small deterministic jitter (realistic ANN corpus).
fn make_vector(centroids: &[Vec<f32>], cluster: usize, seed: u64) -> Vec<f32> {
    let c = &centroids[cluster];
    let j = raw(seed);
    normalize(
        c.iter()
            .zip(&j)
            .map(|(a, b)| a + 0.15 * b)
            .collect::<Vec<f32>>(),
    )
}

fn recall(exact: &[String], approx: &[String]) -> f64 {
    let hit = approx.iter().filter(|id| exact.contains(id)).count();
    hit as f64 / exact.len().max(1) as f64
}

fn bench(c: &mut Criterion) {
    let dir = std::env::temp_dir().join(format!("int8_scan_maddubs_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("bench dir");
    let path = dir.join("index.idx");

    let centroids: Vec<Vec<f32>> = (0..CLUSTERS)
        .map(|i| normalize(raw(0xc000_0000 + i as u64)))
        .collect();
    let mut writer = VectorIndex::create(&path, "bench-384", DIM).expect("create fsvi");
    for i in 0..N {
        let v = make_vector(&centroids, i % CLUSTERS, i as u64 + 1);
        writer
            .write_record(&format!("doc-{i:06}"), &v)
            .expect("write record");
    }
    writer.finish().expect("finish fsvi");
    let index = VectorIndex::open(&path).expect("open fsvi");

    let queries: Vec<Vec<f32>> = (0..QUERIES)
        .map(|q| make_vector(&centroids, q % CLUSTERS, 0xdead_0000 + q as u64))
        .collect();

    // ── RECALL GATE (deterministic): maddubs scan must not lose recall vs the production scan. ──
    let mut orig_recall = 0.0;
    let mut maddubs_recall = 0.0;
    for q in &queries {
        let exact: Vec<String> = index
            .search_top_k(q, K, None)
            .expect("flat")
            .into_iter()
            .map(|h| h.doc_id.to_string())
            .collect();
        let orig: Vec<String> = index
            .bench_search_top_k_int8_two_pass_orig(q, K, MULT)
            .expect("orig scan")
            .into_iter()
            .map(|h| h.doc_id.to_string())
            .collect();
        let maddubs: Vec<String> = index
            .bench_search_top_k_int8_two_pass_maddubs(q, K, MULT)
            .expect("maddubs scan")
            .into_iter()
            .map(|h| h.doc_id.to_string())
            .collect();
        orig_recall += recall(&exact, &orig);
        maddubs_recall += recall(&exact, &maddubs);
    }
    orig_recall /= QUERIES as f64;
    maddubs_recall /= QUERIES as f64;
    eprintln!("[recall] orig={orig_recall:.4} maddubs={maddubs_recall:.4} (mult {MULT}, N {N})");
    assert!(
        maddubs_recall >= orig_recall - 1e-9,
        "maddubs scan must not lose recall vs production: maddubs={maddubs_recall} orig={orig_recall}"
    );

    // ── SCAN-LEVEL A/B: null control, then lever, alternating rounds in one routine. ──
    // ORIG arm = the exact-int8 pass-1 kernel explicitly (the shipped default is now maddubs, so
    // `search_top_k_int8_two_pass` would be an A/A here — keep the bench a true ORIG-vs-maddubs A/B).
    let run_orig = || {
        for q in &queries {
            black_box(
                index
                    .bench_search_top_k_int8_two_pass_orig(black_box(q), K, MULT)
                    .expect("orig"),
            );
        }
    };
    let run_maddubs = || {
        for q in &queries {
            black_box(
                index
                    .bench_search_top_k_int8_two_pass_maddubs(black_box(q), K, MULT)
                    .expect("maddubs"),
            );
        }
    };

    // `inner` averages this many scan batches into ONE timed sample. Under a busy shared fleet the
    // int8 pass-1 scan is rayon-parallel (all cores), so ms-scale contention bursts hit one paired
    // arm but not the other and blow up the per-round ratio (null p5 ~0.5 at inner=2). Averaging
    // 32 batches/sample dilutes sub-batch bursts and tightens the floor WITHOUT biasing the median
    // ratio estimator — a more-powerful test of the same true effect (bd-b5wl worker-isolation retry).
    let inner: u32 = std::env::var("SCAN_AB_INNER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let null = paired_median_ratio(41, inner, run_orig, run_orig);
    let lever = paired_median_ratio(41, inner, run_orig, run_maddubs);
    eprintln!(
        "[null]  scan/{N}: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
        null.median, null.p5, null.p95, null.rounds
    );
    eprintln!(
        "[lever] scan/{N}: maddubs/ORIG median {:.4} p5 {:.4} p95 {:.4} -> {}",
        lever.median,
        lever.p5,
        lever.p95,
        if lever.decidable_against(&null) {
            if lever.median < 1.0 {
                "DECIDABLE WIN"
            } else {
                "DECIDABLE REGRESSION"
            }
        } else {
            "INSIDE NULL FLOOR (not decidable)"
        }
    );

    let mut g = c.benchmark_group("int8_scan_maddubs");
    g.sample_size(20);
    g.bench_function("orig", |b| b.iter(run_orig));
    g.bench_function("maddubs", |b| b.iter(run_maddubs));
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
