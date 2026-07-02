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
        HNSW_DEFAULT_EF_SEARCH, HnswConfig, HnswIndex, Quantization, VectorIndex,
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

    let mut qi = 0usize;
    let mut g = c.benchmark_group("hnsw_vs_flat_100k");
    g.bench_function("flat", |b| {
        b.iter(|| {
            let q = &queries[qi % QUERIES];
            qi += 1;
            black_box(index.search_top_k(black_box(q), K, None).expect("flat"))
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

    let _ = std::fs::remove_file(&path);
}

#[cfg(not(feature = "ann"))]
fn bench_hnsw_vs_flat_100k(_c: &mut Criterion) {
    // HNSW lives behind the `ann` feature; build with `--features ann` to run it.
}

criterion_group!(benches, bench_hnsw_vs_flat_100k);
criterion_main!(benches);
