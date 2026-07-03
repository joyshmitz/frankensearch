//! REAL-embedding validation of the HNSW ANN recall-certificate arc.
//!
//! The entire cert arc (`certify_ef_search`, conformal tail + Bernstein-mean ef
//! selection; ledger `acfb33b`/`7f8de36`/`c30004c`) and the "ANN-in-BOLD viable"
//! numbers (tail 1.76×/mean 3.70× vs flat) were measured only on SYNTHETIC
//! clustered-Gaussian corpora. This bench re-runs the same measurement on genuine
//! Model2Vec/potion embeddings of real English text, so the recall / certified-ef
//! / speedup claims are validated (or corrected) on real data.
//!
//! Generate the slab first (LOCAL), then:
//! ```bash
//! FS_REAL_SLAB=real.bin FS_REAL_DIM=256 \
//!   cargo bench -p frankensearch-index --features ann --bench real_embed_ann
//! ```

use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(feature = "ann")]
fn bench_real_embed_ann(c: &mut Criterion) {
    use std::hint::black_box;

    use frankensearch_index::{HnswConfig, HnswIndex, Quantization, VectorIndex};

    const K: usize = 10;

    fn load_slab(path: &str, dim: usize) -> Vec<Vec<f32>> {
        let bytes = std::fs::read(path).expect("read slab");
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

    fn recall_at_k(exact: &[String], ann: &[String]) -> f64 {
        let hits = ann.iter().filter(|id| exact.contains(id)).count();
        hits as f64 / exact.len().max(1) as f64
    }

    let slab = match std::env::var("FS_REAL_SLAB") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("[real_embed_ann] FS_REAL_SLAB unset — skipping.");
            return;
        }
    };
    let dim: usize = std::env::var("FS_REAL_DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let cap: usize = std::env::var("FS_REAL_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);

    let mut all = load_slab(&slab, dim);
    if all.len() > cap + 1400 {
        all.truncate(cap + 1400);
    }
    eprintln!("[real_embed_ann] using {} vectors × {dim}", all.len());

    // Split: holdout queries (64), calibration (1000), rest = corpus.
    const BENCH_Q: usize = 64;
    const CAL_Q: usize = 1000;
    let holdout: Vec<Vec<f32>> = all.split_off(all.len() - BENCH_Q);
    let calibration: Vec<Vec<f32>> = all.split_off(all.len() - CAL_Q);
    let docs = all;
    let n = docs.len();

    // Build a real VectorIndex (FSVI) from the real embeddings.
    let path = std::env::temp_dir().join(format!("fs_real_ann_{}.fsvi", std::process::id()));
    {
        let mut writer =
            VectorIndex::create_with_revision(&path, "potion", "real", dim, Quantization::F32)
                .expect("create writer");
        for (i, v) in docs.iter().enumerate() {
            writer
                .write_record(&format!("doc-{i:06}"), v)
                .expect("write record");
        }
        writer.finish().expect("finish");
    }
    let index = VectorIndex::open(&path).expect("open index");

    // M=32 (high-recall setting, matching hnsw_vs_flat_100k).
    let hnsw_config = HnswConfig {
        m: 32,
        ..HnswConfig::default()
    };
    let hnsw = HnswIndex::build_from_vector_index(&index, hnsw_config).expect("build hnsw");
    eprintln!("[real_embed_ann] N={n} dim={dim} k={K} — HNSW M=32 built");

    // ── Recall@ef sweep vs exact flat (real data). ──
    let holdout_exact: Vec<Vec<String>> = holdout
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

    for ef in [40usize, 100, 200, 400, 800] {
        let mut total = 0.0;
        for (qi, q) in holdout.iter().enumerate() {
            let ann: Vec<String> = hnsw
                .knn_search(q, K, ef)
                .expect("ann")
                .into_iter()
                .map(|h| h.doc_id.to_string())
                .collect();
            total += recall_at_k(&holdout_exact[qi], &ann);
        }
        eprintln!(
            "[real_embed_ann] RECALL ef={ef:3} recall@{K}={:.4}",
            total / holdout.len() as f64
        );
    }

    // ── Conformal certificate on REAL calibration data (tail mode). ──
    const CANDIDATE_EFS: [usize; 5] = [100, 200, 400, 800, 1600];
    let cert = hnsw
        .certify_ef_search(&index, &calibration, &CANDIDATE_EFS, K, 0.95, 0.1)
        .expect("certify");
    let certified_ef = match &cert {
        Some(cal) => {
            eprintln!(
                "[real_embed_ann] CERTIFIED tail target=0.95 alpha=0.1 -> ef={} lower_bound={:.4} meets={}",
                cal.chosen.ef_search, cal.chosen.certified_recall, cal.chosen.meets_target
            );
            cal.chosen.ef_search
        }
        None => {
            eprintln!("[real_embed_ann] CERTIFIED: none certifiable at target=0.95");
            200
        }
    };

    // ── Latency: flat exact vs HNSW at the certified ef (and ef=40/100). ──
    let mut qi = 0usize;
    let mut g = c.benchmark_group("real_embed_ann");
    g.bench_function("flat", |b| {
        b.iter(|| {
            let q = &holdout[qi % BENCH_Q];
            qi += 1;
            black_box(index.search_top_k(black_box(q), K, None).expect("flat"))
        });
    });
    let mut efs = vec![40usize, 100, 200, certified_ef];
    efs.sort_unstable();
    efs.dedup();
    for ef in efs {
        g.bench_function(format!("hnsw_ef{ef}"), |b| {
            b.iter(|| {
                let q = &holdout[qi % BENCH_Q];
                qi += 1;
                black_box(hnsw.knn_search(black_box(q), K, ef).expect("ann"))
            });
        });
    }
    g.finish();

    let _ = std::fs::remove_file(&path);
}

#[cfg(not(feature = "ann"))]
fn bench_real_embed_ann(_c: &mut Criterion) {
    // HNSW lives behind the `ann` feature; build with `--features ann` to run it.
}

criterion_group!(benches, bench_real_embed_ann);
criterion_main!(benches);
