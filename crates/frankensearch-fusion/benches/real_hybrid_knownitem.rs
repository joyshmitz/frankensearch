//! REAL-data "ratio vs Tantivy": does frankensearch's HYBRID (lexical + vector via
//! RRF) retrieve better than Tantivy-lexical-alone — the core value-add of the
//! hybrid tier — measured by label-free **known-item retrieval** on real text.
//!
//! Setup (all real): index a real English corpus in Tantivy (lexical) AND in the
//! vector index (real potion embeddings). Each query is the **first ~10 words** of a
//! held-out corpus doc (embedded separately, so it is a genuine partial-query, not a
//! trivial exact-vector match); the one relevant answer is that source doc. Measures
//! recall@10 + MRR for lexical-alone (Tantivy BM25) vs vector-alone vs hybrid (RRF).
//!
//! Run (LOCAL — needs the corpus/embedding artifacts on disk):
//! ```bash
//! FS_CORPUS_TXT=corpus.txt FS_CORPUS_SLAB=real.bin FS_QUERY_TXT=queries.txt \
//! FS_QUERY_SLAB=queries.bin FS_QUERY_IDS=query_srcids.txt FS_DIM=256 \
//!   cargo bench -p frankensearch-fusion --features lexical --bench real_hybrid_knownitem
//! ```

use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(feature = "lexical")]
fn bench_real_hybrid_knownitem(c: &mut Criterion) {
    use std::hint::black_box;

    use frankensearch_core::traits::LexicalSearch;
    use frankensearch_core::types::{IndexableDocument, ScoreSource, ScoredResult};
    use frankensearch_fusion::rrf::{RrfConfig, rrf_fuse};
    use frankensearch_index::InMemoryVectorIndex;
    use frankensearch_lexical::TantivyIndex;

    const K: usize = 10;

    fn env(name: &str) -> Option<String> {
        std::env::var(name).ok()
    }
    let (Some(corpus_txt), Some(corpus_slab), Some(query_txt), Some(query_slab), Some(query_ids)) = (
        env("FS_CORPUS_TXT"),
        env("FS_CORPUS_SLAB"),
        env("FS_QUERY_TXT"),
        env("FS_QUERY_SLAB"),
        env("FS_QUERY_IDS"),
    ) else {
        eprintln!("[real_hybrid_knownitem] FS_CORPUS_TXT/SLAB + FS_QUERY_TXT/SLAB/IDS unset — skipping.");
        return;
    };
    let dim: usize = env("FS_DIM").and_then(|s| s.parse().ok()).unwrap_or(256);

    fn read_lines(p: &str) -> Vec<String> {
        std::fs::read_to_string(p)
            .expect("read text")
            .lines()
            .map(str::to_owned)
            .collect()
    }
    fn read_slab(p: &str, dim: usize) -> Vec<Vec<f32>> {
        let bytes = std::fs::read(p).expect("read slab");
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

    let corpus_texts = read_lines(&corpus_txt);
    let corpus_vecs = read_slab(&corpus_slab, dim);
    let n = corpus_texts.len().min(corpus_vecs.len());
    let query_texts = read_lines(&query_txt);
    let query_vecs = read_slab(&query_slab, dim);
    let query_src: Vec<usize> = read_lines(&query_ids)
        .iter()
        .filter_map(|s| s.parse().ok())
        .collect();
    let q = query_texts.len().min(query_vecs.len()).min(query_src.len());
    eprintln!("[real_hybrid_knownitem] corpus={n} docs dim={dim}  queries={q}  k={K}");

    let doc_ids: Vec<String> = (0..n).map(|i| format!("doc-{i:06}")).collect();

    // ── Vector index (real embeddings). ──
    let vindex = InMemoryVectorIndex::from_vectors(
        doc_ids.clone(),
        corpus_vecs[..n].to_vec(),
        dim,
    )
    .expect("vector index");

    // ── Tantivy lexical index (real text) — async build. ──
    let runtime = asupersync::runtime::RuntimeBuilder::current_thread()
        .build()
        .expect("runtime");
    let tantivy = TantivyIndex::in_memory().expect("tantivy in_memory");
    runtime.block_on(async {
        let cx = asupersync::Cx::for_testing();
        let docs: Vec<IndexableDocument> = (0..n)
            .map(|i| IndexableDocument {
                id: doc_ids[i].clone(),
                content: corpus_texts[i].clone(),
                title: None,
                metadata: Default::default(),
            })
            .collect();
        tantivy.index_documents(&cx, &docs).await.expect("index");
        tantivy.commit(&cx).await.expect("commit");
    });

    let cx = asupersync::Cx::for_testing();
    let rrf = RrfConfig::default();

    // ── Known-item eval: recall@K + MRR for lexical / vector / hybrid. ──
    let (mut rl, mut rv, mut rh) = (0.0f64, 0.0f64, 0.0f64);
    let (mut ml, mut mv, mut mh) = (0.0f64, 0.0f64, 0.0f64);
    let rank_of = |ids: &[String], target: &str| -> Option<usize> {
        ids.iter().position(|d| d == target)
    };
    for qi in 0..q {
        let target = format!("doc-{:06}", query_src[qi]);
        // Lexical (Tantivy BM25).
        let lex_hits = tantivy
            .search_doc_ids(&cx, &query_texts[qi], K)
            .expect("lex");
        let lex_ids: Vec<String> = lex_hits.iter().map(|h| h.doc_id.to_string()).collect();
        let lex_scored: Vec<ScoredResult> = lex_hits
            .iter()
            .map(|h| ScoredResult {
                doc_id: h.doc_id.clone(),
                score: h.bm25_score,
                source: ScoreSource::Lexical,
                index: None,
                fast_score: None,
                quality_score: None,
                lexical_score: Some(h.bm25_score),
                rerank_score: None,
                explanation: None,
                metadata: None,
            })
            .collect();
        // Vector.
        let vec_hits = vindex.search_top_k(&query_vecs[qi], K, None).expect("vec");
        let vec_ids: Vec<String> = vec_hits.iter().map(|h| h.doc_id.to_string()).collect();
        // Hybrid (RRF).
        let fused = rrf_fuse(&lex_scored, &vec_hits, K, 0, &rrf);
        let hyb_ids: Vec<String> = fused.iter().map(|h| h.doc_id.to_string()).collect();

        for (ids, rc, mc) in [
            (&lex_ids, &mut rl, &mut ml),
            (&vec_ids, &mut rv, &mut mv),
            (&hyb_ids, &mut rh, &mut mh),
        ] {
            if let Some(rank) = rank_of(ids, &target) {
                *rc += 1.0;
                *mc += 1.0 / (rank as f64 + 1.0);
            }
        }
    }
    let qf = q as f64;
    eprintln!("[knownitem] recall@{K}:  lexical={:.4}  vector={:.4}  hybrid={:.4}", rl / qf, rv / qf, rh / qf);
    eprintln!("[knownitem] MRR@{K}:     lexical={:.4}  vector={:.4}  hybrid={:.4}", ml / qf, mv / qf, mh / qf);

    // ── Latency: lexical vs vector vs hybrid per query. ──
    let mut qi = 0usize;
    let mut g = c.benchmark_group("real_hybrid_knownitem");
    g.bench_function("lexical", |b| {
        b.iter(|| {
            let i = qi % q;
            qi += 1;
            black_box(tantivy.search_doc_ids(&cx, black_box(&query_texts[i]), K).expect("lex"))
        });
    });
    g.bench_function("vector", |b| {
        b.iter(|| {
            let i = qi % q;
            qi += 1;
            black_box(vindex.search_top_k(black_box(&query_vecs[i]), K, None).expect("vec"))
        });
    });
    g.finish();
}

#[cfg(not(feature = "lexical"))]
fn bench_real_hybrid_knownitem(_c: &mut Criterion) {
    // Lexical (Tantivy) lives behind the `lexical` feature; build with `--features lexical`.
}

criterion_group!(benches, bench_real_hybrid_knownitem);
criterion_main!(benches);
