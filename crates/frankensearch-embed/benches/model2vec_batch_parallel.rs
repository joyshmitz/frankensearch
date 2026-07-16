//! Model2Vec `embed_batch` parallel-dispatch A/B.
//!
//! `Model2VecEmbedder::embed_batch` now dispatches per-document `embed_sync` across Rayon
//! threads once a batch reaches `PARALLEL_BATCH_MIN` — the sibling of the FNV hash embedder's
//! batch parallelization (`0b560edc`). Each document is independent ~0.5 ms CPU work: a static
//! embedding-row gather over a ~30 MB table (cache-miss-bound) + mean-pool + L2-normalize.
//!
//! This measures the parallel-dispatch scaling on that per-document kernel with synthetic data
//! (no model file needed). It is exactly the work `embed_batch` fans out across docs — minus
//! tokenization, which parallelizes identically. `serial` = the shipped-before loop; `parallel`
//! = the new Rayon dispatch. Parity-asserted bit-identical (per-doc work is deterministic and
//! `par_iter().collect()` preserves order).
//!
//! Run with:
//! ```bash
//!   rch exec -- cargo bench -p frankensearch-embed --features model2vec \
//!     --bench model2vec_batch_parallel
//! ```

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_embed::simd::accumulate_model2vec_rows;
use rayon::prelude::*;

const VOCAB: usize = 30_000; // potion-base-8M-class vocab (30k × 256 × 4 ≈ 30 MB table)
const DIM: usize = 256;
const TOKENS_PER_DOC: usize = 220; // ~a paragraph of tokenized text

/// Deterministic xorshift for reproducible fixtures.
struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn build_embeddings() -> Vec<f32> {
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    let mut emb = vec![0.0_f32; VOCAB * DIM];
    for v in &mut emb {
        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        let val = ((rng.next() >> 40) as i64 % 2000 - 1000) as f32 / 1000.0;
        *v = val;
    }
    emb
}

fn build_corpus(n_docs: usize) -> Vec<Vec<u32>> {
    let mut rng = XorShift(0x2545_F491_4F6C_DD1D);
    (0..n_docs)
        .map(|_| {
            (0..TOKENS_PER_DOC)
                .map(|_| {
                    #[allow(clippy::cast_possible_truncation)]
                    let id = (rng.next() % VOCAB as u64) as u32;
                    id
                })
                .collect()
        })
        .collect()
}

/// One document's embedding: static-row gather + mean-pool + L2-normalize.
/// Mirrors `Model2VecEmbedder::embed_sync` arithmetic, minus tokenization.
fn embed_doc(emb: &[f32], ids: &[u32]) -> Vec<f32> {
    let mut sum = vec![0.0_f32; DIM];
    let count = accumulate_model2vec_rows(&mut sum, emb, ids, VOCAB);
    if count == 0 {
        return sum;
    }
    #[allow(clippy::cast_precision_loss)]
    let inv = 1.0 / count as f32;
    for s in &mut sum {
        *s *= inv;
    }
    let norm_sq: f32 = sum.iter().map(|x| x * x).sum();
    if norm_sq.is_finite() && norm_sq > f32::EPSILON {
        let inv_norm = 1.0 / norm_sq.sqrt();
        for s in &mut sum {
            *s *= inv_norm;
        }
    } else {
        sum.fill(0.0);
    }
    sum
}

fn batch_serial(emb: &[f32], corpus: &[Vec<u32>]) -> Vec<Vec<f32>> {
    corpus.iter().map(|ids| embed_doc(emb, ids)).collect()
}

fn batch_parallel(emb: &[f32], corpus: &[Vec<u32>]) -> Vec<Vec<f32>> {
    corpus.par_iter().map(|ids| embed_doc(emb, ids)).collect()
}

fn bench(c: &mut Criterion) {
    let emb = build_embeddings();
    let mut group = c.benchmark_group("model2vec_batch_parallel");
    for &n in &[64_usize, 256] {
        let corpus = build_corpus(n);
        // Parallel dispatch must be bit-identical to the serial loop.
        assert_eq!(
            batch_serial(&emb, &corpus),
            batch_parallel(&emb, &corpus),
            "parallel batch diverged from serial at n={n}"
        );
        #[allow(clippy::cast_possible_truncation)]
        group.throughput(criterion::Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("serial", n), &(), |b, ()| {
            b.iter(|| black_box(batch_serial(black_box(&emb), black_box(&corpus))));
        });
        group.bench_with_input(BenchmarkId::new("parallel", n), &(), |b, ()| {
            b.iter(|| black_box(batch_parallel(black_box(&emb), black_box(&corpus))));
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
