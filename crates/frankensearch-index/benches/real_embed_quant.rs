//! REAL-embedding validation of the shipped int8/4-bit two-pass quantization,
//! plus the anisotropic-quant lever #3 (random-rotation preprocessing).
//!
//! Every prior ANN/quant recall number in this repo rests on a SYNTHETIC
//! clustered-Gaussian corpus (isotropic noise around random centroids). This
//! bench replaces that foundation with genuine static (Model2Vec/potion)
//! embeddings of real English text — which carry the true per-dimension
//! anisotropy / outlier structure that quantization actually has to survive.
//!
//! Generate the slab first (LOCAL, needs the model on disk):
//! ```bash
//! cargo run --release -p frankensearch-embed --features model2vec \
//!   --example potion_embed_corpus -- <model_dir> corpus.txt real.bin
//! FS_REAL_SLAB=real.bin FS_REAL_DIM=256 \
//!   cargo bench -p frankensearch-index --bench real_embed_quant
//! ```
//!
//! Measures, on the real corpus:
//!   1. Per-dimension anisotropy (variance profile, top-dim share, excess kurtosis)
//!      — how far real embeddings are from the synthetic isotropic assumption.
//!   2. Plain int8 / 4-bit two-pass recall@k vs exact — validates the shipped
//!      quant claims on REAL data (they were only ever measured on synthetic).
//!   3. Lever #3: the SAME quant recall after an orthonormal random rotation
//!      (inner products preserved exactly; only the quantization grid changes).
//!      If rotation lifts 4-bit recall to plain-int8 levels, that is a 2×
//!      recall-per-byte win on the bandwidth-bound flat scan.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_index::InMemoryVectorIndex;

const K: usize = 10;

// ── deterministic RNG (xorshift64* + Box-Muller), no external dep ──
struct Rng(u64);
impl Rng {
    fn u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn unit(&mut self) -> f32 {
        // (0,1)
        ((self.u64() >> 11) as f64 / (1u64 << 53) as f64) as f32
    }
    fn gauss(&mut self) -> f32 {
        let u1 = self.unit().max(1e-9);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

/// Random orthonormal `dim × dim` matrix (row-major), via Gaussian + modified
/// Gram–Schmidt. Orthonormal ⇒ `<Rx, Ry> = <x, y>` exactly, so it preserves all
/// inner products and only reshapes the per-dimension distribution the quantizer
/// grid sees.
fn random_orthonormal(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed | 1);
    let mut m = vec![0f32; dim * dim];
    for v in &mut m {
        *v = rng.gauss();
    }
    // Modified Gram–Schmidt over rows.
    for i in 0..dim {
        // Orthogonalize row i against rows 0..i.
        for j in 0..i {
            let mut dot = 0f32;
            for d in 0..dim {
                dot += m[i * dim + d] * m[j * dim + d];
            }
            for d in 0..dim {
                m[i * dim + d] -= dot * m[j * dim + d];
            }
        }
        // Normalize row i.
        let mut norm = 0f32;
        for d in 0..dim {
            norm += m[i * dim + d] * m[i * dim + d];
        }
        let inv = 1.0 / norm.sqrt().max(1e-20);
        for d in 0..dim {
            m[i * dim + d] *= inv;
        }
    }
    m
}

fn rotate(r: &[f32], x: &[f32], dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; dim];
    for i in 0..dim {
        let row = &r[i * dim..i * dim + dim];
        let mut acc = 0f32;
        for d in 0..dim {
            acc += row[d] * x[d];
        }
        out[i] = acc;
    }
    out
}

fn load_slab(path: &str, dim: usize) -> Vec<Vec<f32>> {
    let bytes = std::fs::read(path).expect("read slab");
    assert!(bytes.len() % (dim * 4) == 0, "slab size not a multiple of dim*4");
    let n = bytes.len() / (dim * 4);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut v = Vec::with_capacity(dim);
        for d in 0..dim {
            let off = (i * dim + d) * 4;
            v.push(f32::from_le_bytes([
                bytes[off],
                bytes[off + 1],
                bytes[off + 2],
                bytes[off + 3],
            ]));
        }
        out.push(v);
    }
    out
}

fn recall_at_k(exact: &[String], approx: &[String]) -> f64 {
    let hits = approx.iter().filter(|id| exact.contains(id)).count();
    hits as f64 / exact.len().max(1) as f64
}

fn report_anisotropy(vectors: &[Vec<f32>], dim: usize, label: &str) {
    let n = vectors.len() as f64;
    let mut mean = vec![0f64; dim];
    for v in vectors {
        for d in 0..dim {
            mean[d] += v[d] as f64;
        }
    }
    for m in &mut mean {
        *m /= n;
    }
    let mut var = vec![0f64; dim];
    let mut m4 = vec![0f64; dim];
    for v in vectors {
        for d in 0..dim {
            let z = v[d] as f64 - mean[d];
            var[d] += z * z;
            m4[d] += z * z * z * z;
        }
    }
    for d in 0..dim {
        var[d] /= n;
        m4[d] /= n;
    }
    let total: f64 = var.iter().sum();
    let mut sorted = var.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let top1 = sorted[0] / total * 100.0;
    let top5: f64 = sorted[..5.min(dim)].iter().sum::<f64>() / total * 100.0;
    let top10: f64 = sorted[..10.min(dim)].iter().sum::<f64>() / total * 100.0;
    // Average per-dim excess kurtosis (0 = Gaussian; >0 = heavy tails/outliers).
    let mut exkurt = 0f64;
    let mut cnt = 0f64;
    for d in 0..dim {
        if var[d] > 1e-12 {
            exkurt += m4[d] / (var[d] * var[d]) - 3.0;
            cnt += 1.0;
        }
    }
    exkurt /= cnt.max(1.0);
    eprintln!(
        "[anisotropy {label}] n={} dim={dim} var(total)={total:.4} top1-dim%={top1:.1} top5-dim%={top5:.1} top10-dim%={top10:.1} avg-excess-kurtosis={exkurt:.2}",
        vectors.len()
    );
}

fn bench_real_embed_quant(c: &mut Criterion) {
    let slab = match std::env::var("FS_REAL_SLAB") {
        Ok(p) => p,
        Err(_) => {
            eprintln!(
                "[real_embed_quant] FS_REAL_SLAB unset — skipping (generate with the potion_embed_corpus example)."
            );
            return;
        }
    };
    let dim: usize = std::env::var("FS_REAL_DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let n_queries: usize = std::env::var("FS_REAL_QUERIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);

    let mut all = load_slab(&slab, dim);
    eprintln!("[real_embed_quant] loaded {} vectors × {dim}", all.len());
    assert!(all.len() > n_queries + 100, "corpus too small");

    // Hold out the last n_queries vectors as queries; the rest are the corpus.
    let queries: Vec<Vec<f32>> = all.split_off(all.len() - n_queries);
    let docs = all;
    let n = docs.len();
    let doc_ids: Vec<String> = (0..n).map(|i| format!("doc-{i:06}")).collect();

    report_anisotropy(&docs, dim, "real");

    // ── Random orthonormal rotation (inner-product preserving). ──
    let rot = random_orthonormal(dim, 0x9e37_79b9_7f4a_7c15);
    let docs_rot: Vec<Vec<f32>> = docs.iter().map(|v| rotate(&rot, v, dim)).collect();
    report_anisotropy(&docs_rot, dim, "rotated");
    let queries_rot: Vec<Vec<f32>> = queries.iter().map(|v| rotate(&rot, v, dim)).collect();

    let index =
        InMemoryVectorIndex::from_vectors(doc_ids.clone(), docs, dim).expect("build plain index");
    let index_rot =
        InMemoryVectorIndex::from_vectors(doc_ids, docs_rot, dim).expect("build rotated index");

    // Exact f16 top-K reference (rotation preserves inner products, so the plain
    // exact top-K is the reference for BOTH the plain and rotated approx paths).
    let exact: Vec<Vec<String>> = queries
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

    // ── Recall@K sweeps: plain vs rotated, int8 and 4-bit, over mult. ──
    eprintln!("[real_embed_quant] N={n} dim={dim} k={K} queries={n_queries}");
    for mult in [2usize, 3, 5, 10, 20] {
        let mut r_i8 = 0.0;
        let mut r_i8_rot = 0.0;
        let mut r_4b = 0.0;
        let mut r_4b_rot = 0.0;
        for (qi, q) in queries.iter().enumerate() {
            let qr = &queries_rot[qi];
            let ids = |hits: Vec<frankensearch_core::types::VectorHit>| -> Vec<String> {
                hits.into_iter().map(|h| h.doc_id.to_string()).collect()
            };
            r_i8 += recall_at_k(
                &exact[qi],
                &ids(index.search_top_k_int8_two_pass(q, K, mult).expect("i8")),
            );
            r_i8_rot += recall_at_k(
                &exact[qi],
                &ids(index_rot
                    .search_top_k_int8_two_pass(qr, K, mult)
                    .expect("i8r")),
            );
            r_4b += recall_at_k(
                &exact[qi],
                &ids(index.search_top_k_4bit_two_pass(q, K, mult).expect("4b")),
            );
            r_4b_rot += recall_at_k(
                &exact[qi],
                &ids(index_rot
                    .search_top_k_4bit_two_pass(qr, K, mult)
                    .expect("4br")),
            );
        }
        let q = n_queries as f64;
        eprintln!(
            "[recall] mult={mult:2}  int8={:.4} int8+rot={:.4}   4bit={:.4} 4bit+rot={:.4}",
            r_i8 / q,
            r_i8_rot / q,
            r_4b / q,
            r_4b_rot / q
        );
    }

    // ── Latency: flat exact vs int8/4bit two-pass (plain) at a representative mult. ──
    let mut qi = 0usize;
    let mut g = c.benchmark_group("real_embed_quant");
    g.bench_function("flat", |b| {
        b.iter(|| {
            let q = &queries[qi % n_queries];
            qi += 1;
            black_box(index.search_top_k(black_box(q), K, None).expect("flat"))
        });
    });
    for mult in [5usize, 10] {
        g.bench_function(format!("int8_mult{mult}"), |b| {
            b.iter(|| {
                let q = &queries[qi % n_queries];
                qi += 1;
                black_box(
                    index
                        .search_top_k_int8_two_pass(black_box(q), K, mult)
                        .expect("i8"),
                )
            });
        });
        g.bench_function(format!("fourbit_mult{mult}"), |b| {
            b.iter(|| {
                let q = &queries[qi % n_queries];
                qi += 1;
                black_box(
                    index
                        .search_top_k_4bit_two_pass(black_box(q), K, mult)
                        .expect("4b"),
                )
            });
        });
    }
    g.finish();
}

criterion_group!(benches, bench_real_embed_quant);
criterion_main!(benches);
