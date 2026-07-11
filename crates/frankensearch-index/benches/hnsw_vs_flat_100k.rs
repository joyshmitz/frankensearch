//! HNSW (ANN) vs flat brute-force vector search **at 100k** — the scale where
//! ANN's O(log N) navigation should beat the O(N) rayon flat scan (the 10k
//! `hnsw_vs_flat` showed flat wins except at low `ef`, because at 10k the parallel
//! flat scan is already ~175 µs). This measures HNSW `knn_search` latency vs flat
//! `search_top_k`, plus **recall@10** of HNSW vs the exact flat top-10 (printed to
//! stderr), swept over `ef_search` — the data that decides whether ANN-in-BOLD is
//! worth wiring (a big latency win at high recall) at BOLD's larger corpus size.
//! Validation only; nothing here is wired into product code.
//!
//! Run with (the `ann` feature is required):
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-index --features ann --bench hnsw_vs_flat_100k
//! ```
use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(feature = "ann")]
fn bench_hnsw_vs_flat_100k(c: &mut Criterion) {
    use std::hint::black_box;

    use frankensearch_index::{
        HNSW_DEFAULT_EF_SEARCH, HnswConfig, HnswIndex, Quantization, VectorIndex, certified_min_ef,
        certified_min_ef_mean,
    };

    const N: usize = 100_000;
    const DIM: usize = 128;
    const K: usize = 10;
    const QUERIES: usize = 32;

    const CLUSTERS: usize = 256;
    // Tighter clusters (0.15 vs the prior 0.30): real semantic embeddings put
    // similar docs genuinely close, unlike the diffuse 0.30 spread that is
    // near-worst-case for HNSW navigation. Tests whether the M-swept rejection is
    // a synthetic-diffuseness artifact or robust to realistic cluster tightness.
    const NOISE: f32 = 0.15;

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
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-12 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }

    fn make_vector(centroids: &[Vec<f32>], c: usize, noise_seed: u64) -> Vec<f32> {
        let centroid = &centroids[c % centroids.len()];
        let noise = raw_vector(noise_seed);
        let v: Vec<f32> = centroid
            .iter()
            .zip(&noise)
            .map(|(a, n)| a + NOISE * n)
            .collect();
        normalize(v)
    }

    fn recall_at_k(flat: &[String], ann: &[String]) -> f64 {
        let hits = ann.iter().filter(|id| flat.contains(id)).count();
        hits as f64 / flat.len().max(1) as f64
    }

    let centroids: Vec<Vec<f32>> = (0..CLUSTERS)
        .map(|c| normalize(raw_vector(0xc000_0000 + c as u64)))
        .collect();

    let path = std::env::temp_dir().join(format!("fs_hnsw_bench_100k_{}.fsvi", std::process::id()));
    {
        let mut writer =
            VectorIndex::create_with_revision(&path, "hash", "bench", DIM, Quantization::F32)
                .expect("create writer");
        for i in 0..N {
            let v = make_vector(&centroids, i % CLUSTERS, i as u64 + 1);
            writer
                .write_record(&format!("doc-{i:06}"), &v)
                .expect("write record");
        }
        writer.finish().expect("finish");
    }
    let index = VectorIndex::open(&path).expect("open index");

    // F16 copy of the SAME corpus for the int8 two-pass arm. The production fast-tier
    // default is int8 two-pass (lossless candidate gen, ~7x vs flat), and it requires an
    // F16-quantized slab (`search_top_k_int8_two_pass` falls back to exact on F32). Same
    // vectors, so its latency and recall are directly comparable to `flat` and HNSW here —
    // this is the baseline ANN would actually have to beat, not the naive flat scan.
    let path_f16 =
        std::env::temp_dir().join(format!("fs_hnsw_bench_100k_f16_{}.fsvi", std::process::id()));
    {
        let mut writer =
            VectorIndex::create_with_revision(&path_f16, "hash", "bench", DIM, Quantization::F16)
                .expect("create f16 writer");
        for i in 0..N {
            let v = make_vector(&centroids, i % CLUSTERS, i as u64 + 1);
            writer
                .write_record(&format!("doc-{i:06}"), &v)
                .expect("write f16 record");
        }
        writer.finish().expect("finish f16");
    }
    let index_f16 = VectorIndex::open(&path_f16).expect("open f16 index");
    // Production fast-tier candidate multiplier (see sync_searcher::search_fast_hits).
    const INT8_MULT: usize = 3;

    // M is HNSW's primary recall knob (default 16). Bump to 32 (standard high-recall
    // setting) to test whether the default-M recall rejection is config-dependent or
    // fundamental. ef_construction kept at the 200 default.
    let hnsw_config = HnswConfig {
        m: 32,
        ..HnswConfig::default()
    };
    let hnsw = HnswIndex::build_from_vector_index(&index, hnsw_config).expect("build hnsw");

    let queries: Vec<Vec<f32>> = (0..QUERIES)
        .map(|q| make_vector(&centroids, q % CLUSTERS, 0xdead_0000 + q as u64))
        .collect();

    let flat_topk: Vec<Vec<String>> = queries
        .iter()
        .map(|q| {
            index
                .search_top_k(q, K, None)
                .expect("flat")
                .into_iter()
                .map(|h| h.doc_id.to_string())
                .collect()
        })
        .collect();

    // ── Certificate-driven ef selection, TWO guarantee modes on shared calibration:
    //    (1) PER-QUERY TAIL (split-conformal): a fresh query's recall >= target w.p.
    //        >= 1-alpha — the strong guarantee. (2) AVERAGE recall (empirical-Bernstein
    //        mean LCB): E[recall] >= target w.p. >= 1-delta — a weaker budget that
    //        certifies a CHEAPER ef. Recall is measured ONCE per candidate ef on the
    //        calibration set (exact top-k is ef-independent), then each mode selects.
    const CAL_QUERIES: usize = 1000;
    const HOLDOUT_QUERIES: usize = 300;
    const TARGET_RECALL: f64 = 0.95;
    const ALPHA: f64 = 0.1; // per-query tail confidence
    const DELTA: f64 = 0.05; // mean-mode confidence
    const CANDIDATE_EFS: [usize; 5] = [10, 20, 40, 100, 200];

    let calibration: Vec<Vec<f32>> = (0..CAL_QUERIES)
        .map(|q| make_vector(&centroids, q % CLUSTERS, 0xca11_0000 + q as u64))
        .collect();
    // Exact top-k ids per calibration query, computed ONCE (ef-independent).
    let cal_exact: Vec<Vec<String>> = calibration
        .iter()
        .map(|q| {
            index
                .search_top_k(q, K, None)
                .expect("flat")
                .into_iter()
                .map(|h| h.doc_id.to_string())
                .collect()
        })
        .collect();
    // Per-ef recall sample over the calibration set.
    let cal_samples: Vec<(usize, Vec<f64>)> = CANDIDATE_EFS
        .iter()
        .map(|&ef| {
            let recalls: Vec<f64> = calibration
                .iter()
                .zip(&cal_exact)
                .map(|(q, exact)| {
                    let ann: Vec<String> = hnsw
                        .knn_search(q, K, ef)
                        .expect("ann")
                        .into_iter()
                        .map(|h| h.doc_id.to_string())
                        .collect();
                    recall_at_k(exact, &ann)
                })
                .collect();
            (ef, recalls)
        })
        .collect();
    let tail = certified_min_ef(&cal_samples, TARGET_RECALL, ALPHA).expect("tail");
    let mean = certified_min_ef_mean(&cal_samples, TARGET_RECALL, DELTA).expect("mean");
    let (tail_ef, mean_ef) = (tail.ef_search, mean.ef_search);

    // Out-of-sample recall for each chosen ef on fresh held-out queries.
    let holdout: Vec<Vec<f32>> = (0..HOLDOUT_QUERIES)
        .map(|q| make_vector(&centroids, q % CLUSTERS, 0x401d_0000 + q as u64))
        .collect();
    let holdout_recall_at = |ef: usize| -> f64 {
        let sum: f64 = holdout
            .iter()
            .map(|q| {
                let exact: Vec<String> = index
                    .search_top_k(q, K, None)
                    .expect("flat")
                    .into_iter()
                    .map(|h| h.doc_id.to_string())
                    .collect();
                let ann: Vec<String> = hnsw
                    .knn_search(q, K, ef)
                    .expect("ann")
                    .into_iter()
                    .map(|h| h.doc_id.to_string())
                    .collect();
                recall_at_k(&exact, &ann)
            })
            .sum();
        sum / HOLDOUT_QUERIES as f64
    };
    let tail_holdout = holdout_recall_at(tail_ef);
    let mean_holdout = holdout_recall_at(mean_ef);

    let mut qi = 0usize;
    let mut g = c.benchmark_group("hnsw_vs_flat_100k");
    g.bench_function("flat", |b| {
        b.iter(|| {
            let q = &queries[qi % QUERIES];
            qi += 1;
            black_box(index.search_top_k(black_box(q), K, None).expect("flat"))
        });
    });
    // The REAL baseline: the production fast-tier default (int8 two-pass, recall-preserving).
    g.bench_function("int8_two_pass", |b| {
        b.iter(|| {
            let q = &queries[qi % QUERIES];
            qi += 1;
            black_box(
                index_f16
                    .search_top_k_int8_two_pass(black_box(q), K, INT8_MULT)
                    .expect("int8"),
            )
        });
    });
    for ef in [10usize, 20, 40, HNSW_DEFAULT_EF_SEARCH] {
        g.bench_function(format!("hnsw_ef{ef}"), |b| {
            b.iter(|| {
                let q = &queries[qi % QUERIES];
                qi += 1;
                black_box(hnsw.knn_search(black_box(q), K, ef).expect("ann"))
            });
        });
    }
    // The two certified efs — each one's latency vs `flat` above is the certified
    // ANN-in-BOLD speedup under that guarantee mode.
    g.bench_function(format!("hnsw_tail_certified_ef{tail_ef}"), |b| {
        b.iter(|| {
            let q = &queries[qi % QUERIES];
            qi += 1;
            black_box(hnsw.knn_search(black_box(q), K, tail_ef).expect("ann"))
        });
    });
    g.bench_function(format!("hnsw_mean_certified_ef{mean_ef}"), |b| {
        b.iter(|| {
            let q = &queries[qi % QUERIES];
            qi += 1;
            black_box(hnsw.knn_search(black_box(q), K, mean_ef).expect("ann"))
        });
    });
    g.finish();

    // Recall@K sweep printed AFTER criterion (to stdout) so it lands in the
    // captured output tail (rch keeps only the tail; a setup-time eprintln was
    // truncated). HNSW is only a real win where recall is high at a fast `ef`.
    const EF_SWEEP: [usize; 5] = [10, 20, 40, 100, 200];
    for ef in EF_SWEEP {
        let mut total = 0.0;
        for (qi, query) in queries.iter().enumerate() {
            let ann: Vec<String> = hnsw
                .knn_search(query, K, ef)
                .expect("ann")
                .into_iter()
                .map(|h| h.doc_id.to_string())
                .collect();
            total += recall_at_k(&flat_topk[qi], &ann);
        }
        println!(
            "RECALL_RESULT N={N} dim={DIM} k={K} ef_search={ef} recall@{K}={:.4}",
            total / QUERIES as f64
        );
    }

    // int8 two-pass recall@K vs the F32 exact ground truth (the production default it
    // would have to beat). Its latency arm above is the number ANN must undercut.
    {
        let mut total = 0.0;
        for (qi, query) in queries.iter().enumerate() {
            let hits: Vec<String> = index_f16
                .search_top_k_int8_two_pass(query, K, INT8_MULT)
                .expect("int8")
                .into_iter()
                .map(|h| h.doc_id.to_string())
                .collect();
            total += recall_at_k(&flat_topk[qi], &hits);
        }
        println!(
            "INT8_RESULT N={N} dim={DIM} k={K} mult={INT8_MULT} recall@{K}={:.4}",
            total / QUERIES as f64
        );
    }

    println!(
        "CERTIFIED_RESULT target={TARGET_RECALL} n_cal={CAL_QUERIES} | \
         TAIL(per-query, alpha={ALPHA}): ef={tail_ef} lower_bound={:.4} meets={} holdout@{K}={:.4} | \
         MEAN(average, Bernstein, delta={DELTA}): ef={mean_ef} lower_bound={:.4} meets={} holdout@{K}={:.4} \
         (compare hnsw_tail_certified_ef{tail_ef} and hnsw_mean_certified_ef{mean_ef} vs flat for the two certified speedups)",
        tail.certified_recall,
        tail.meets_target,
        tail_holdout,
        mean.certified_recall,
        mean.meets_target,
        mean_holdout
    );

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&path_f16);
}

#[cfg(not(feature = "ann"))]
fn bench_hnsw_vs_flat_100k(_c: &mut Criterion) {
    // HNSW lives behind the `ann` feature; build with `--features ann` to run it.
}

criterion_group!(benches, bench_hnsw_vs_flat_100k);
criterion_main!(benches);
