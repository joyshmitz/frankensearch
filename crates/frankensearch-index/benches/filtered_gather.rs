//! Selective-filter **gather** fast-path vs the per-document filtered scan.
//!
//! A filtered vector search with a [`BitsetFilter`] allow-list currently scans the
//! **whole corpus** and probes the filter once per document. When the allow-set is
//! a small fraction of the corpus (the common real-world case — "search within
//! these K docs"), that is mostly wasted work: every rejected document still pays a
//! membership probe, and most documents are rejected.
//!
//! The gather fast-path inverts the loop: it iterates the (small) allow-set, maps
//! each hash to its position via a lazily-built `hash → pos` table, and exact
//! f16-scans **only** those positions — `O(|allow-set|)` instead of `O(corpus)`.
//! It is bit-identical to the per-document scan (the passing set is identical and
//! both rank by the `(score, index)` total order).
//!
//! This bench sweeps selectivity to (a) prove bit-identity and (b) find the
//! crossover where gather stops beating the scan, which sets
//! `GATHER_SELECTIVITY_DIVISOR`.
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
//!   rch exec -- cargo bench -p frankensearch-index --bench filtered_gather
//! ```

use std::collections::HashSet;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_core::filter::{BitsetFilter, fnv1a_hash};
use frankensearch_index::{InMemoryVectorIndex, VectorIndex};

const N: usize = 50_000;
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

/// Build a `BitsetFilter` allowing `count` documents spread evenly across the
/// corpus (a realistic "this subset is in scope" filter, not a contiguous block).
fn allow_filter(count: usize) -> BitsetFilter {
    let stride = (N / count.max(1)).max(1);
    let allowed: HashSet<u64> = (0..N)
        .step_by(stride)
        .take(count)
        .map(|i| fnv1a_hash(format!("doc-{i:06}").as_bytes()))
        .collect();
    BitsetFilter::from_hashes(allowed)
}

fn ids(hits: &[frankensearch_core::VectorHit]) -> Vec<String> {
    hits.iter().map(|h| h.doc_id.to_string()).collect()
}

fn bench_index_path() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "frankensearch-filtered-gather-{}-{nanos}.fsvi",
        std::process::id()
    ))
}

fn build_fsvi_index(doc_ids: &[String], vectors: &[Vec<f32>]) -> VectorIndex {
    let path = bench_index_path();
    let mut writer =
        VectorIndex::create(&path, "filtered-gather-bench", DIM).expect("create file-backed index");
    for (doc_id, vector) in doc_ids.iter().zip(vectors) {
        writer
            .write_record(doc_id, vector)
            .expect("write file-backed vector");
    }
    writer.finish().expect("finish file-backed index");
    VectorIndex::open(&path).expect("open file-backed index")
}

fn bench_filtered_gather(c: &mut Criterion) {
    let centroids: Vec<Vec<f32>> = (0..CLUSTERS)
        .map(|i| normalize(raw_vector(0xc000_0000 + i as u64)))
        .collect();
    let doc_ids: Vec<String> = (0..N).map(|i| format!("doc-{i:06}")).collect();
    let vectors: Vec<Vec<f32>> = (0..N)
        .map(|i| make_vector(&centroids, i % CLUSTERS, i as u64 + 1))
        .collect();
    let index =
        InMemoryVectorIndex::from_vectors(doc_ids, vectors, DIM).expect("build in-memory index");

    let queries: Vec<Vec<f32>> = (0..QUERIES)
        .map(|q| make_vector(&centroids, q % CLUSTERS, 0xdead_0000 + q as u64))
        .collect();

    // Selectivity points as a fraction of the corpus.
    let selectivities: [f64; 8] = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50];

    // Correctness gate: gather == per-document scan (bit-identical) at every point.
    for &s in &selectivities {
        let allow = (N as f64 * s) as usize;
        let filter = allow_filter(allow);
        for q in &queries {
            let scan = ids(&index
                .bench_scan_filtered(q, K, Some(&filter))
                .expect("scan"));
            let gather = ids(&index.bench_gather_filtered(q, K, &filter).expect("gather"));
            assert_eq!(
                scan, gather,
                "gather must be bit-identical to scan (sel={s})"
            );
        }
        eprintln!("[filtered_gather] sel={s} allow≈{allow} parity OK");
    }

    // Latency A/B per selectivity: scan (baseline) vs gather (candidate).
    let mut qi = 0usize;
    for &s in &selectivities {
        let allow = (N as f64 * s) as usize;
        let filter = allow_filter(allow);
        let pct = (s * 1000.0) as usize; // permille, for a stable arm name
        let mut g = c.benchmark_group(format!("filtered_gather/sel_permille_{pct}"));
        g.bench_function("scan", |b| {
            b.iter(|| {
                let q = &queries[qi % QUERIES];
                qi += 1;
                black_box(
                    index
                        .bench_scan_filtered(black_box(q), K, Some(&filter))
                        .expect("scan"),
                )
            });
        });
        g.bench_function("gather", |b| {
            b.iter(|| {
                let q = &queries[qi % QUERIES];
                qi += 1;
                black_box(
                    index
                        .bench_gather_filtered(black_box(q), K, &filter)
                        .expect("gather"),
                )
            });
        });
        g.finish();
    }
}

fn bench_fsvi_filtered_gather(c: &mut Criterion) {
    let centroids: Vec<Vec<f32>> = (0..CLUSTERS)
        .map(|i| normalize(raw_vector(0xc000_0000 + i as u64)))
        .collect();
    let doc_ids: Vec<String> = (0..N).map(|i| format!("doc-{i:06}")).collect();
    let vectors: Vec<Vec<f32>> = (0..N)
        .map(|i| make_vector(&centroids, i % CLUSTERS, i as u64 + 1))
        .collect();
    let index = build_fsvi_index(&doc_ids, &vectors);

    let queries: Vec<Vec<f32>> = (0..QUERIES)
        .map(|q| make_vector(&centroids, q % CLUSTERS, 0xbeef_0000 + q as u64))
        .collect();

    let selectivities: [f64; 8] = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50];

    for &s in &selectivities {
        let allow = (N as f64 * s) as usize;
        let filter = allow_filter(allow);
        for q in &queries {
            let scan = ids(&index
                .bench_scan_filtered(q, K, Some(&filter))
                .expect("scan"));
            let gather = ids(&index.bench_gather_filtered(q, K, &filter).expect("gather"));
            assert_eq!(
                scan, gather,
                "file-backed gather must be bit-identical to scan (sel={s})"
            );
            let public = ids(&index.search_top_k(q, K, Some(&filter)).expect("public"));
            if allow.saturating_mul(50) < N {
                assert_eq!(
                    public, scan,
                    "public file-backed search must take the gather-equivalent result (sel={s})"
                );
            }
        }
        eprintln!("[fsvi_filtered_gather] sel={s} allow≈{allow} parity OK");
    }

    let mut qi = 0usize;
    for &s in &selectivities {
        let allow = (N as f64 * s) as usize;
        let filter = allow_filter(allow);
        let pct = (s * 1000.0) as usize;
        let mut g = c.benchmark_group(format!("fsvi_filtered_gather/sel_permille_{pct}"));
        g.bench_function("scan", |b| {
            b.iter(|| {
                let q = &queries[qi % QUERIES];
                qi += 1;
                black_box(
                    index
                        .bench_scan_filtered(black_box(q), K, Some(&filter))
                        .expect("scan"),
                )
            });
        });
        g.bench_function("gather", |b| {
            b.iter(|| {
                let q = &queries[qi % QUERIES];
                qi += 1;
                black_box(
                    index
                        .bench_gather_filtered(black_box(q), K, &filter)
                        .expect("gather"),
                )
            });
        });
        g.finish();
    }
}

criterion_group!(benches, bench_filtered_gather, bench_fsvi_filtered_gather);
criterion_main!(benches);
