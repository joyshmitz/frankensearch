//! Latency benchmark for the pure-Rust f32 `NativeReranker` (bd-1nl13.10).
//!
//! ## Method
//!
//! `rerank_sync` scores each `(query, doc)` pair with an independent BERT
//! cross-encoder forward in a sequential loop, so total rerank latency is
//! `per_doc_latency * doc_count` plus negligible fixed overhead. We therefore
//! benchmark the *per-doc* forward at sequence lengths {128, 256, 512} (fast,
//! criterion-friendly) and extrapolate the doc-count table {10, 50, 100},
//! validated by a 10-doc linearity check. Benchmarking 100 docs @ 512 directly
//! under criterion's sampler would take tens of minutes for no extra signal,
//! because each seq-512 forward is ~hundreds of ms single-threaded in f32.
//!
//! ## Latency budget (bd-1nl13.10)
//!
//! Canonical budget point: p95 latency for 50 docs @ seq 256. The measured
//! baseline and the full extrapolated table are recorded in the bd-1nl13.10
//! bead comment and in this crate's commit message; re-run this benchmark to
//! refresh them after any change to the f32 forward or to frankentorch kernels.
//!
//! Run:
//! ```text
//! cargo bench -p frankensearch-rerank --features native --bench native_rerank
//! ```
//! If the staged model dir is absent the benchmark logs a SKIP and exits 0.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use frankensearch_core::traits::{RerankDocument, SyncRerank};
use frankensearch_rerank::NativeReranker;

/// Staged reference model (model_f32.safetensors + tokenizer.json).
const MODEL_DIR: &str = "/private/tmp/ee-reranker-port/model";
const QUERY: &str = "what is the capital city of france and its population";

/// Word counts chosen to land near 128 / 256 / 512 total tokens for the
/// `[CLS] query [SEP] doc [SEP]` input; the actual per-doc latency for each is
/// logged at startup so the numbers stay interpretable even if WordPiece
/// tokenization drifts from the nominal target.
const SEQ_CONFIGS: &[(&str, usize)] = &[("seq~128", 90), ("seq~256", 215), ("seq~512", 460)];

/// Build a dense filler passage of `n_words` multi-token words.
fn filler_doc(n_words: usize) -> String {
    const WORDS: &[&str] = &[
        "transformer",
        "embedding",
        "relevance",
        "document",
        "passage",
        "retrieval",
        "semantic",
        "ranking",
        "context",
        "language",
    ];
    let mut s = String::with_capacity(n_words * 10);
    for i in 0..n_words {
        if i > 0 {
            s.push(' ');
        }
        s.push_str(WORDS[i % WORDS.len()]);
    }
    s
}

/// Load the staged reranker, or `None` (with a logged SKIP) if it is absent.
fn load() -> Option<NativeReranker> {
    let dir = PathBuf::from(MODEL_DIR);
    if !dir.join("tokenizer.json").is_file() {
        eprintln!(
            "[native_rerank bench] SKIP: model dir {MODEL_DIR} has no tokenizer.json. \
             Stage the reranker model there to run this benchmark."
        );
        return None;
    }
    match NativeReranker::load(&dir) {
        Ok(r) => Some(r),
        Err(e) => {
            eprintln!("[native_rerank bench] SKIP: NativeReranker::load failed: {e}");
            None
        }
    }
}

fn bench_native_rerank(c: &mut Criterion) {
    let Some(reranker) = load() else {
        return;
    };

    // Detailed informational logging: a warm single-call per-doc timing for
    // each sequence length, plus the extrapolated doc-count table. This lands
    // in the bench log immediately, before criterion's longer sampling.
    eprintln!("[native_rerank bench] warm per-doc timings (f32, single-threaded):");
    let mut per_doc_ms: Vec<(&str, f64)> = Vec::new();
    for (label, nw) in SEQ_CONFIGS {
        let docs = vec![RerankDocument {
            doc_id: "d0".into(),
            text: filler_doc(*nw),
        }];
        let _ = reranker.rerank_sync(QUERY, &docs); // warm caches
        let t = Instant::now();
        let _ = reranker.rerank_sync(QUERY, &docs).expect("rerank");
        let ms = t.elapsed().as_secs_f64() * 1e3;
        eprintln!("  {label} (words={nw}): per_doc = {ms:.2} ms");
        per_doc_ms.push((label, ms));
    }
    eprintln!("[native_rerank bench] extrapolated rerank latency = per_doc * doc_count:");
    eprintln!("  docs |   seq~128 |   seq~256 |   seq~512");
    for n in [10usize, 50, 100] {
        eprintln!(
            "  {n:>4} | {:>7.1}ms | {:>7.1}ms | {:>7.1}ms",
            per_doc_ms[0].1 * n as f64,
            per_doc_ms[1].1 * n as f64,
            per_doc_ms[2].1 * n as f64,
        );
    }

    let mut g = c.benchmark_group("native_rerank_per_doc");
    g.sample_size(10);
    g.warm_up_time(Duration::from_secs(2));
    g.measurement_time(Duration::from_secs(8));
    for (label, nw) in SEQ_CONFIGS {
        let docs = vec![RerankDocument {
            doc_id: "d0".into(),
            text: filler_doc(*nw),
        }];
        g.bench_with_input(BenchmarkId::from_parameter(label), &docs, |b, docs| {
            b.iter(|| reranker.rerank_sync(QUERY, docs).expect("rerank"));
        });
    }
    g.finish();

    // Linearity check: 10 docs @ seq~256 should ≈ 10 × per-doc(seq~256),
    // validating the extrapolation used for the 50/100-doc table.
    let mut g2 = c.benchmark_group("native_rerank_linearity");
    g2.sample_size(10);
    g2.warm_up_time(Duration::from_secs(2));
    g2.measurement_time(Duration::from_secs(12));
    let docs10: Vec<RerankDocument> = (0..10)
        .map(|i| RerankDocument {
            doc_id: format!("d{i}").into(),
            text: filler_doc(215),
        })
        .collect();
    g2.bench_function("10docs_seq~256", |b| {
        b.iter(|| reranker.rerank_sync(QUERY, &docs10).expect("rerank"));
    });
    g2.finish();
}

criterion_group!(benches, bench_native_rerank);
criterion_main!(benches);
