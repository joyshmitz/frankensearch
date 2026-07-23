//! Vector-tier known-item recall vs QUERY LENGTH on real embeddings — the light
//! (index-only) half of the hybrid query-length question: does semantic (vector)
//! retrieval hold up for SHORT / vague queries, where BM25 keyword-matching
//! struggles? Query = first W words of a held-out doc (embedded separately); the
//! relevant answer is that source doc.
//!
//! ```bash
//! FS_CORPUS_SLAB=real_big.bin FS_DIM=256 \
//! FS_QSLABS=queries_w3.bin,queries_w5.bin,queries_w10.bin \
//! FS_QIDS=query_srcids_w3.txt,query_srcids_w5.txt,query_srcids_big.txt \
//! FS_QLABELS=w3,w5,w10 \
//!   cargo bench -p frankensearch-index --bench real_qlen_vector
//! ```

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_index::InMemoryVectorIndex;

fn read_slab(path: &str, dim: usize) -> Vec<Vec<f32>> {
    let bytes = std::fs::read(path).expect("read slab");
    let n = bytes.len() / (dim * 4);
    (0..n)
        .map(|i| {
            (0..dim)
                .map(|d| {
                    let o = (i * dim + d) * 4;
                    f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]])
                })
                .collect()
        })
        .collect()
}

fn read_ids(path: &str) -> Vec<usize> {
    std::fs::read_to_string(path)
        .expect("read ids")
        .lines()
        .filter_map(|s| s.trim().parse().ok())
        .collect()
}

fn bench_real_qlen_vector(_c: &mut Criterion) {
    let Ok(corpus_slab) = std::env::var("FS_CORPUS_SLAB") else {
        eprintln!("[real_qlen_vector] FS_CORPUS_SLAB unset — skipping.");
        return;
    };
    let dim: usize = std::env::var("FS_DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let qslabs: Vec<String> = std::env::var("FS_QSLABS")
        .unwrap_or_default()
        .split(',')
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect();
    let qids: Vec<String> = std::env::var("FS_QIDS")
        .unwrap_or_default()
        .split(',')
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect();
    let qlabels: Vec<String> = std::env::var("FS_QLABELS")
        .unwrap_or_default()
        .split(',')
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect();

    const K: usize = 10;
    let corpus = read_slab(&corpus_slab, dim);
    let n = corpus.len();
    let doc_ids: Vec<String> = (0..n).map(|i| format!("doc-{i:06}")).collect();
    let index = InMemoryVectorIndex::from_vectors(doc_ids, corpus, dim).expect("index");
    eprintln!("[real_qlen_vector] corpus={n} dim={dim} k={K}");

    for ((slab, ids_path), label) in qslabs.iter().zip(&qids).zip(&qlabels) {
        let qvecs = read_slab(slab, dim);
        let srcids = read_ids(ids_path);
        let q = qvecs.len().min(srcids.len());
        let mut recall = 0.0f64;
        let mut mrr = 0.0f64;
        for i in 0..q {
            let target = format!("doc-{:06}", srcids[i]);
            let hits = index.search_top_k(&qvecs[i], K, None).expect("search");
            if let Some(rank) = hits.iter().position(|h| h.doc_id == target) {
                recall += 1.0;
                mrr += 1.0 / (rank as f64 + 1.0);
            }
        }
        let qf = q as f64;
        eprintln!(
            "[qlen] {label:>4} ({q} queries): vector recall@{K}={:.4} MRR@{K}={:.4}",
            recall / qf,
            mrr / qf
        );
    }
}

criterion_group!(benches, bench_real_qlen_vector);
criterion_main!(benches);
