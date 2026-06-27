//! FSVI (file-backed `VectorIndex`) int8 two-pass vs exact f16 scan — lever
//! validation for standalone large-N vector search on clustered data.
//!
//! `VectorIndex::search_top_k_int8_two_pass` does a fast parallel int8 pass-1 over
//! all main records + an exact f16 rescore of the top `k·mult` candidates. This is
//! the file-backed twin of the validated in-memory `int8_two_pass` lever. It is NOT
//! wired into the BOLD hybrid (that gap is not vector-bound — see
//! docs/NEGATIVE_EVIDENCE.md); this measures pure vector-search latency + recall.
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
//!   rch exec -- cargo bench -p frankensearch-index --bench fsvi_int8_two_pass
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_index::VectorIndex;

const N: usize = 100_000;
const DIM: usize = 384;
const K: usize = 10;
const QUERIES: usize = 32;
const CLUSTERS: usize = 64;
const NOISE: f32 = 0.30;

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
    normalize(
        centroid
            .iter()
            .zip(&noise)
            .map(|(a, n)| a + NOISE * n)
            .collect(),
    )
}

fn recall_at_k(exact: &[String], approx: &[String]) -> f64 {
    let hits = approx.iter().filter(|id| exact.contains(id)).count();
    hits as f64 / exact.len().max(1) as f64
}

fn bench_fsvi_int8_two_pass(c: &mut Criterion) {
    let dir = std::env::temp_dir().join(format!("fsvi_int8_two_pass_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create bench dir");
    let path = dir.join("index.idx");

    let centroids: Vec<Vec<f32>> = (0..CLUSTERS)
        .map(|i| normalize(raw_vector(0xc000_0000 + i as u64)))
        .collect();
    let mut writer = VectorIndex::create(&path, "bench-384", DIM).expect("create fsvi index");
    for i in 0..N {
        let vector = make_vector(&centroids, i % CLUSTERS, i as u64 + 1);
        writer
            .write_record(&format!("doc-{i:06}"), &vector)
            .expect("write record");
    }
    writer.finish().expect("finish fsvi index");
    let index = VectorIndex::open(&path).expect("open fsvi index");

    let queries: Vec<Vec<f32>> = (0..QUERIES)
        .map(|q| make_vector(&centroids, q % CLUSTERS, 0xdead_0000 + q as u64))
        .collect();

    // Exact flat top-K (recall reference).
    let exact: Vec<Vec<String>> = queries
        .iter()
        .map(|q| {
            index
                .search_top_k(q, K, None)
                .expect("flat")
                .into_iter()
                .map(|h| h.doc_id)
                .collect()
        })
        .collect();

    // ── Recall@K sweep over candidate_multiplier. ──
    for mult in [2usize, 3, 5, 10] {
        let mut total = 0.0;
        for (qi, query) in queries.iter().enumerate() {
            let approx: Vec<String> = index
                .search_top_k_int8_two_pass(query, K, mult)
                .expect("int8")
                .into_iter()
                .map(|h| h.doc_id)
                .collect();
            total += recall_at_k(&exact[qi], &approx);
        }
        eprintln!(
            "[fsvi_int8_two_pass] N={N} dim={DIM} k={K} mult={mult} recall@{K}={:.4}",
            total / QUERIES as f64
        );
    }

    // ── Latency: flat exact vs int8 two-pass at mult=5 and mult=10. ──
    let mut qi = 0usize;
    let mut g = c.benchmark_group("fsvi_int8_two_pass");
    g.bench_function("flat", |b| {
        b.iter(|| {
            let q = &queries[qi % QUERIES];
            qi += 1;
            black_box(index.search_top_k(black_box(q), K, None).expect("flat"))
        });
    });
    for mult in [5usize, 10] {
        g.bench_function(format!("int8_mult{mult}"), |b| {
            b.iter(|| {
                let q = &queries[qi % QUERIES];
                qi += 1;
                black_box(
                    index
                        .search_top_k_int8_two_pass(black_box(q), K, mult)
                        .expect("int8"),
                )
            });
        });
    }
    g.finish();

    std::fs::remove_dir_all(&dir).ok();
}

criterion_group!(benches, bench_fsvi_int8_two_pass);
criterion_main!(benches);
