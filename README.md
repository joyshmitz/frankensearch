# frankensearch

<div align="center">

[![CI](https://github.com/Dicklesworthstone/frankensearch/actions/workflows/ci.yml/badge.svg)](https://github.com/Dicklesworthstone/frankensearch/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/frankensearch.svg)](https://crates.io/crates/frankensearch)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-edition_2024-orange.svg)](https://doc.rust-lang.org/edition-guide/)

**Two-tier hybrid search for Rust: sub-millisecond initial results, quality-refined rankings in 150ms.**

Stitched together from the best parts of three battle-tested codebases — hence the name.

</div>

---

## TL;DR

**The Problem:** You want both speed *and* quality from local search, but fast embedding models sacrifice accuracy and quality models are too slow for interactive use. You also want lexical keyword matching combined with semantic understanding, not one or the other.

**The Solution:** frankensearch runs a two-tier progressive search pipeline. A fast static embedder (potion-128M, 0.57ms) delivers initial results instantly, then a quality transformer (MiniLM-L6-v2, 128ms) refines the rankings in the background. Lexical BM25 and semantic vector search are fused via Reciprocal Rank Fusion. Your UI gets results to display in under 15ms, with refined rankings arriving ~150ms later.

### Why frankensearch?

| Feature | What It Does |
|---------|--------------|
| **Progressive search** | Initial results in <15ms, refined in ~150ms via `SearchPhase` iterator |
| **Hybrid fusion** | Lexical (Tantivy BM25) + semantic (vector cosine) combined with RRF (K=60) |
| **Two-tier embedding** | Fast tier (potion-128M, 0.57ms) + quality tier (MiniLM-L6-v2, 128ms) |
| **Graceful degradation** | Quality model unavailable? Falls back to fast-only. No models at all? Hash embedder works everywhere |
| **Feature-gated** | Pay only for what you compile: `hash` (0 deps) through `full` (everything) |
| **SIMD vector search** | `wide::f32x8` portable SIMD across x86 SSE2/AVX2 and ARM NEON |
| **f16 quantization** | 50% memory savings on vector indices with <1% quality loss |
| **Zero unsafe code** | `#![forbid(unsafe_code)]` throughout |

---

## Quick Example

```rust
use frankensearch::prelude::*;

// Build a two-tier searcher with defaults
let config = TwoTierConfig::default();
let searcher = TwoTierSearcher::new(&index, &embedder_stack, config)
    .with_lexical(&tantivy_index);

// Progressive search: fast results first, then refined
for phase in searcher.search("distributed consensus algorithm", 10) {
    match phase {
        SearchPhase::Initial(results) => {
            // Display these immediately (~15ms)
            for hit in &results {
                println!("  {} (rrf: {:.4})", hit.doc_id, hit.rrf_score);
            }
        }
        SearchPhase::Refined(results) => {
            // Update the display with refined rankings (~150ms)
            for hit in &results {
                println!("  {} (rrf: {:.4}, blended)", hit.doc_id, hit.rrf_score);
            }
        }
        SearchPhase::RefinementFailed(results) => {
            // Quality model failed; initial results are still valid
        }
    }
}
```

### Indexing Documents

```rust
use frankensearch::prelude::*;

// Auto-detect the best available embedder
let embedder_stack = EmbedderStack::auto_detect()?;

// Build a vector index
let mut builder = VectorIndexBuilder::new(embedder_stack.fast_embedder());
for doc in &documents {
    let text = canonicalize(&doc.content); // NFC + markdown strip + code collapse
    let embedding = embedder_stack.fast_embedder().embed(&text)?;
    builder.add(&doc.id, &embedding)?;
}
builder.save("index.fsvi")?; // FSVI binary format, f16 quantized, fsync'd

// Build a Tantivy lexical index
let lexical = LexicalIndexBuilder::new()
    .add_documents(&documents)?
    .build("tantivy_index/")?;
```

### Minimal Setup (Hash Embedder Only)

```rust
// Zero dependencies beyond frankensearch-core
// Works everywhere, no model downloads needed
use frankensearch::hash::FnvHashEmbedder;

let embedder = FnvHashEmbedder::new(384); // 384-dim, deterministic
let embedding = embedder.embed("hello world")?;
```

---

## Design Philosophy

### 1. Progressive Over Blocking

Traditional search makes you wait for the best answer. frankensearch yields fast approximate results immediately (via `SearchPhase::Initial`), then upgrades them when the quality model finishes. The consumer decides how to present this — swap in place, animate a transition, or ignore the refinement entirely.

### 2. Hybrid Over Single-Signal

Pure semantic search misses exact keyword matches. Pure lexical search misses meaning. frankensearch fuses both via Reciprocal Rank Fusion (RRF), which is rank-based and doesn't depend on score normalization. Documents appearing in both lexical and semantic results get a natural boost.

### 3. Pay For What You Use

The default feature set (`hash`) compiles with zero ML dependencies. Add `model2vec` for the fast tier (~128MB model), `fastembed` for the quality tier (~90MB model + ONNX runtime), `lexical` for Tantivy, or `full` for everything. Feature flags control compilation, not runtime behavior.

### 4. No Domain Leakage

The vector index stores only `(doc_id, embedding)`. frankensearch doesn't know or care whether your documents are tweets, chat messages, code files, or research papers. Domain-specific metadata belongs in your storage layer.

### 5. Deterministic Results

NaN-safe `total_cmp()` ordering in the top-k heap. Four-level tie-breaking in RRF: score descending, in-both-sources preference, lexical score descending, doc_id ascending. Same input always produces the same output.

---

## How frankensearch Compares

| Feature | frankensearch | tantivy (alone) | qdrant | meilisearch |
|---------|---------------|-----------------|--------|-------------|
| Semantic search | Two-tier (fast + quality) | Via plugin | Single model | Experimental |
| Lexical search | Tantivy BM25 | Native | Basic | Native |
| Hybrid fusion | RRF built-in | Manual | RRF | Manual |
| Progressive results | Native iterator | N/A | N/A | N/A |
| Deployment | Embedded library | Embedded library | Server | Server |
| Model management | Auto-detect + download | N/A | External | Built-in |
| f16 quantization | Default | N/A | Optional | N/A |
| Portable SIMD | `wide` (x86 + ARM) | N/A | Platform-specific | N/A |
| Unsafe code | Forbidden | Minimal | Present | Present |

**Use frankensearch when:** you need an embedded search library with both speed and quality, hybrid lexical+semantic fusion, and progressive result delivery — all without running a server.

**Use something else when:** you need a distributed search cluster, GPU-accelerated inference, or a standalone search API with a REST interface.

---

## Installation

### As a Dependency

```toml
# Cargo.toml — pick your feature set

# Minimal: hash embedder only (zero ML deps, always works)
[dependencies]
frankensearch = "0.1"

# Fast semantic search (potion-128M, ~0.57ms embeddings)
[dependencies]
frankensearch = { version = "0.1", features = ["model2vec"] }

# Full hybrid search (semantic + lexical + RRF)
[dependencies]
frankensearch = { version = "0.1", features = ["hybrid"] }

# Everything: all models, reranking, ANN, model downloads
[dependencies]
frankensearch = { version = "0.1", features = ["full"] }
```

### Feature Flags

| Feature | Dependencies Added | What You Get |
|---------|--------------------|--------------|
| `hash` (default) | None | FNV-1a hash embedder, vector index, SIMD search |
| `model2vec` | safetensors, tokenizers | potion-128M static embedder (fast tier) |
| `fastembed` | fastembed (ONNX runtime) | MiniLM-L6-v2 transformer embedder (quality tier) |
| `lexical` | tantivy | BM25 full-text search + RRF fusion |
| `rerank` | ort, tokenizers | FlashRank cross-encoder reranking |
| `ann` | hnsw_rs | HNSW approximate nearest neighbor index |
| `download` | reqwest | Model auto-download from HuggingFace |
| `semantic` | model2vec + fastembed | All embedding models |
| `hybrid` | semantic + lexical | Full hybrid search pipeline |
| `full` | Everything | All features enabled |

### From Source

```bash
git clone https://github.com/Dicklesworthstone/frankensearch.git
cd frankensearch
cargo build --release --features full
cargo test --features full
```

Requires Rust nightly (edition 2024). A `rust-toolchain.toml` is included.

---

## Architecture

```
                         ┌──────────────────┐
                         │   User Query     │
                         └────────┬─────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │   Text Canonicalization      │
                    │   NFC → Markdown Strip →     │
                    │   Code Collapse → Truncate   │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │   Query Classification      │
                    │   Empty│Identifier│Short│NL  │
                    │   → Adaptive candidate       │
                    │     budgets per class         │
                    └─────────────┬───────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                                       ▼
   ┌─────────────────────┐                 ┌─────────────────────┐
   │  Fast Tier Embed    │                 │  Tantivy BM25       │
   │  potion-128M        │                 │  Lexical Search     │
   │  ~0.57ms, 256d      │                 │                     │
   └──────────┬──────────┘                 └──────────┬──────────┘
              │                                       │
              ▼                                       │
   ┌─────────────────────┐                            │
   │  Vector Index       │                            │
   │  FSVI (f16, mmap)   │                            │
   │  SIMD dot product   │                            │
   └──────────┬──────────┘                            │
              │                                       │
              └──────────────┬────────────────────────┘
                             ▼
               ┌─────────────────────────────┐
               │    RRF Fusion (K=60)        │
               │    score = Σ 1/(K+rank+1)   │
               └─────────────┬───────────────┘
                             │
                             ▼
               ┌─────────────────────────────┐
               │  yield SearchPhase::Initial │  ← ~15ms
               └─────────────┬───────────────┘
                             │
                             ▼
               ┌─────────────────────────────┐
               │  Quality Tier Embed         │
               │  MiniLM-L6-v2, ~128ms       │
               │  Re-embed top candidates    │
               └─────────────┬───────────────┘
                             │
                             ▼
               ┌─────────────────────────────┐
               │  Two-Tier Blending          │
               │  0.7 quality + 0.3 fast     │
               └─────────────┬───────────────┘
                             │
                             ▼
               ┌─────────────────────────────┐
               │  Optional: FlashRank        │
               │  Cross-encoder reranking    │
               └─────────────┬───────────────┘
                             │
                             ▼
               ┌─────────────────────────────┐
               │  yield SearchPhase::Refined │  ← ~150ms
               └─────────────────────────────┘
```

### Crate Structure

```
frankensearch/                         # Facade crate (re-exports everything)
├── Cargo.toml                         # Workspace root
└── crates/
    ├── frankensearch-core/            # Zero-dep traits, types, errors
    │   └── src/lib.rs                 #   Embedder, Reranker, SearchError,
    │                                  #   ScoredResult, VectorHit, FusedHit,
    │                                  #   Canonicalizer, QueryClass
    │
    ├── frankensearch-embed/           # Embedder implementations
    │   └── src/
    │       ├── hash_embedder.rs       #   FNV-1a (0 deps, always available)
    │       ├── model2vec_embedder.rs  #   potion-128M (fast tier)
    │       ├── fastembed_embedder.rs  #   MiniLM-L6-v2 (quality tier)
    │       └── auto_detect.rs         #   EmbedderStack auto-detection
    │
    ├── frankensearch-index/           # Vector storage & search
    │   └── src/
    │       ├── format.rs              #   FSVI binary format I/O
    │       ├── simd.rs                #   wide::f32x8 dot product
    │       ├── search.rs              #   Brute-force top-k
    │       └── hnsw.rs                #   Optional HNSW ANN
    │
    ├── frankensearch-lexical/         # Full-text search
    │   └── src/lib.rs                 #   Tantivy schema, indexing, queries
    │
    ├── frankensearch-fusion/          # Result combination
    │   └── src/
    │       ├── rrf.rs                 #   Reciprocal Rank Fusion
    │       ├── normalize.rs           #   Score normalization (min-max)
    │       ├── blend.rs               #   Two-tier score blending
    │       ├── two_tier_searcher.rs   #   Progressive iterator orchestrator
    │       └── query_class.rs         #   Query classification & budgets
    │
    └── frankensearch-rerank/          # Reranking
        └── src/lib.rs                 #   FlashRank cross-encoder
```

### Dependency Graph Between Crates

```
frankensearch-core       ← everything depends on this (zero external deps)
    │
    ├── frankensearch-embed   (+ safetensors, tokenizers, fastembed)
    ├── frankensearch-index   (+ wide, half, memmap2)
    ├── frankensearch-lexical (+ tantivy)
    ├── frankensearch-fusion  (depends on embed + index + lexical)
    └── frankensearch-rerank  (+ ort, tokenizers)

frankensearch (facade)   ← re-exports from all crates
```

---

## Core Types

### SearchPhase — Progressive Results

```rust
pub enum SearchPhase {
    /// Fast results from potion-128M + BM25 + RRF (~15ms)
    Initial(Vec<FusedHit>),
    /// Quality-refined results from MiniLM-L6-v2 blending (~150ms)
    Refined(Vec<FusedHit>),
    /// Quality model failed; initial results are your final answer
    RefinementFailed(Vec<FusedHit>),
}
```

### FusedHit — Hybrid Search Result

```rust
pub struct FusedHit {
    pub doc_id: String,
    pub rrf_score: f64,
    pub lexical_rank: Option<usize>,
    pub semantic_rank: Option<usize>,
    pub lexical_score: Option<f32>,
    pub semantic_score: Option<f32>,
    pub in_both_sources: bool,
}
```

### Embedder Trait

```rust
pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> SearchResult<Vec<f32>>;
    fn embed_batch(&self, texts: &[&str]) -> SearchResult<Vec<Vec<f32>>>;
    fn dimension(&self) -> usize;
    fn id(&self) -> &str;
    fn is_semantic(&self) -> bool;
    fn category(&self) -> ModelCategory;
    fn supports_mrl(&self) -> bool; // Matryoshka dim truncation
}
```

### TwoTierConfig

```rust
let config = TwoTierConfig {
    quality_weight: 0.7,        // 70% quality, 30% fast in blend
    rrf_k: 60.0,                // RRF constant (literature standard)
    candidate_multiplier: 3,    // Fetch 3x limit from each source
    quality_timeout_ms: 500,    // Max wait for quality model
    fast_only: false,           // Set true to skip quality refinement
    ..Default::default()
};
```

### SearchError

```rust
pub enum SearchError {
    EmbedderUnavailable { model: String, reason: String },
    EmbeddingFailed { model: String, source: Box<dyn Error> },
    IndexCorrupted { path: PathBuf, detail: String },
    DimensionMismatch { expected: usize, found: usize },
    QueryParseError { query: String, detail: String },
    SearchTimeout { elapsed_ms: u64, budget_ms: u64 },
    // ... and more
}
```

---

## FSVI Vector Index Format

frankensearch stores embeddings in a custom binary format optimized for memory-mapped SIMD search:

```
┌─────────────────────────────────────┐
│ Header                              │
│   magic: "FSVI" (4 bytes)           │
│   version: u16                      │
│   embedder_id: variable UTF-8       │
│   dimension: u32                    │
│   quantization: u8 (0=f32, 1=f16)  │
│   record_count: u64                 │
│   vectors_offset: u64               │
│   header_crc32: u32                 │
├─────────────────────────────────────┤
│ Record Table (16 bytes/record)      │
│   doc_id_hash: u64 (FNV-1a)        │
│   doc_id_offset: u32               │
│   doc_id_len: u16                  │
│   flags: u16                        │
├─────────────────────────────────────┤
│ String Table                        │
│   Concatenated UTF-8 doc_id strings │
├─────────────────────────────────────┤
│ Vector Slab (64-byte aligned)       │
│   f16 or f32 vectors, contiguous    │
│   Aligned for AVX2/cache lines      │
└─────────────────────────────────────┘
```

- **f16 by default**: 50% memory reduction, <1% quality loss for cosine similarity
- **Memory-mapped**: Zero-copy access via `memmap2`, OS handles page caching
- **64-byte aligned vectors**: Cache-line and AVX2 friendly
- **fsync on save**: Durability guarantee for the index file

---

## Performance

| Operation | Latency | Notes |
|-----------|---------|-------|
| FNV-1a hash embed | ~0.07ms | Pure algorithm, zero deps |
| potion-128M embed | ~0.57ms | Static token lookup + mean pool |
| MiniLM-L6-v2 embed | ~128ms | ONNX transformer inference |
| 384-dim f16 dot product | <2us | `wide::f32x8` SIMD |
| Phase 1 (Initial results) | <15ms | Fast embed + vector search + BM25 + RRF |
| Phase 2 (Refined results) | ~150ms | Quality embed + blend (quality embed is bottleneck) |
| Top-k search (10K docs) | <15ms | Brute-force with heap guard pattern |
| Top-k search (100K docs) | <150ms | Rayon parallel chunks of 1024 |

### Embedder Bakeoff Results

| Model | p50 Latency | Embeddings/sec | Dimensions | Semantic Quality |
|-------|-------------|----------------|------------|------------------|
| FNV-1a hash | 0.07ms | 14,000+ | 384 (configurable) | None (deterministic) |
| potion-multilingual-128M | 0.57ms | 52,144 | 256 | Good (223x faster than MiniLM) |
| all-MiniLM-L6-v2 | 128ms | 234 | 384 | Excellent (baseline) |

The 223x speed gap between potion and MiniLM is exactly why the two-tier design exists.

---

## Configuration

### Environment Variable Overrides

| Variable | Description | Default |
|----------|-------------|---------|
| `FRANKENSEARCH_MODEL_DIR` | Model file directory | `~/.cache/frankensearch/models/` |
| `FRANKENSEARCH_FAST_MODEL` | Fast tier model name | `potion-multilingual-128M` |
| `FRANKENSEARCH_QUALITY_MODEL` | Quality tier model name | `all-MiniLM-L6-v2` |
| `FRANKENSEARCH_FAST_ONLY` | Skip quality refinement | `false` |
| `FRANKENSEARCH_QUALITY_WEIGHT` | Blend factor (0.0-1.0) | `0.7` |
| `FRANKENSEARCH_RRF_K` | RRF constant | `60.0` |
| `FRANKENSEARCH_PARALLEL_SEARCH` | Enable rayon parallel search | `auto` |
| `FRANKENSEARCH_LOG` | Tracing filter directive | `info` |

### Model Search Paths

Models are located by checking these paths in order:

1. `$FRANKENSEARCH_MODEL_DIR` (explicit override)
2. `~/.cache/frankensearch/models/`
3. `~/.local/share/frankensearch/models/`
4. HuggingFace cache (`~/.cache/huggingface/hub/`)

### Embedder Auto-Detection

When you call `EmbedderStack::auto_detect()`, frankensearch probes for available models:

1. **FastEmbed** (MiniLM-L6-v2) — check for `model.onnx` in model paths
2. **Model2Vec** (potion-128M) — check for `model.safetensors` + `tokenizer.json`
3. **FNV-1a Hash** — always available (fallback)

The best available model becomes the quality tier; the fastest becomes the fast tier. If only one model is found, both tiers use it (no refinement phase). If only the hash embedder is available, search works but without semantic understanding.

---

## Text Canonicalization

All text is preprocessed before embedding to maximize search quality:

```rust
use frankensearch::core::DefaultCanonicalizer;

let canon = DefaultCanonicalizer::default();
let clean = canon.canonicalize(raw_text);
```

The default pipeline applies these steps in order:

1. **NFC Unicode normalization** — ensures hash stability across different Unicode representations
2. **Markdown stripping** — removes `#`, `**`, `*`, `_`, `[text](url)` → `text`
3. **Code block collapsing** — keeps first 20 + last 10 lines of fenced code blocks
4. **Low-signal filtering** — removes pure-URL lines, import-only blocks, empty sections
5. **Length truncation** — caps at 2000 characters (configurable)

Query canonicalization is simpler (no markdown stripping or code collapsing) since queries are typically short natural language.

---

## Query Classification

frankensearch adapts its retrieval strategy based on query type:

| Query Class | Example | Strategy |
|-------------|---------|----------|
| `Empty` | `""` | Return empty results |
| `Identifier` | `"br-123"`, `"src/main.rs"` | Lean heavily lexical (exact match matters) |
| `ShortKeyword` | `"error handling"` | Balanced lexical + semantic |
| `NaturalLanguage` | `"how does the search pipeline work?"` | Lean heavily semantic |

Each class gets adaptive candidate budgets — identifiers fetch more lexical candidates, natural language queries fetch more semantic candidates. This avoids wasting compute on the wrong retrieval path.

---

## Troubleshooting

### "EmbedderUnavailable: potion-multilingual-128M"

The Model2Vec model files aren't found. Either download them or use the hash embedder:

```bash
# Option 1: Download models (requires 'download' feature)
frankensearch download potion-multilingual-128M

# Option 2: Set model directory
export FRANKENSEARCH_MODEL_DIR=/path/to/models

# Option 3: Use hash-only (always works, no semantic understanding)
# In code: let embedder = FnvHashEmbedder::new(384);
```

### "DimensionMismatch: expected 256, found 384"

Your vector index was built with a different embedder than you're querying with. Rebuild the index with the correct embedder, or use the embedder that matches the index dimension.

### "IndexCorrupted: bad magic bytes"

The FSVI file is damaged or not a valid frankensearch index. Re-index your documents.

### Quality refinement never completes

Check that the quality model (MiniLM-L6-v2) is downloaded and accessible. If ONNX Runtime can't load it, frankensearch yields `SearchPhase::RefinementFailed` with the fast results as your final answer. Set `FRANKENSEARCH_FAST_ONLY=true` to skip refinement entirely.

### High memory usage with large indices

FSVI indices are memory-mapped. The OS manages page caching, so `top`/`htop` may show high virtual memory but actual resident memory depends on access patterns. For very large indices, ensure your system has enough RAM to keep the hot portion cached, or consider using HNSW (ANN) to avoid scanning every vector.

---

## Limitations

- **CPU-only inference**: ONNX Runtime runs on CPU; no GPU acceleration support yet
- **Single-node**: Designed as an embedded library, not a distributed search cluster
- **English-optimized**: MiniLM-L6-v2 works best on English text; potion-multilingual-128M handles multiple languages for the fast tier
- **Brute-force default**: Vector search scans all vectors by default; HNSW ANN is optional and trades accuracy for speed at large scale
- **No incremental index updates**: Adding documents requires rebuilding the FSVI index (append support planned)
- **Model download size**: Full setup requires ~220MB of model files (potion: ~128MB, MiniLM: ~90MB)

---

## FAQ

### Why "frankensearch"?

It's stitched together from the best parts of three separate codebases (cass, xf, mcp_agent_mail_rust) — like Frankenstein's monster, but for search. Each project independently developed similar hybrid search systems; frankensearch extracts and unifies the common core.

### Why two tiers instead of one model?

The 223x speed gap between potion-128M (0.57ms) and MiniLM-L6-v2 (128ms) means you can show results to the user *before* the quality model even starts. In interactive applications, perceived latency matters more than final quality — and with two tiers, you get both.

### Why RRF instead of learned fusion?

Reciprocal Rank Fusion is rank-based, so it doesn't need score calibration between lexical and semantic sources. It's simple, well-studied, and produces consistently good results across diverse query types. The K=60 constant comes from the original Cormack et al. literature and has been validated across all three source codebases.

### Can I use frankensearch without Tantivy?

Yes. Without the `lexical` feature, you get pure semantic search (vector similarity only). The RRF fusion layer is skipped, and you get ranked results from vector search alone.

### Can I bring my own embedding model?

Yes. Implement the `Embedder` trait for your model and pass it to the searcher. The trait is object-safe (`dyn Embedder`) for runtime polymorphism.

### Does frankensearch phone home or send telemetry?

No. Everything runs locally. No network calls are made unless you explicitly enable the `download` feature and call the model download function.

---

## Acknowledgments

Built with:
- [Tantivy](https://github.com/quickwit-oss/tantivy) — Full-text search engine for Rust
- [fastembed-rs](https://github.com/Anush008/fastembed-rs) — ONNX-based text embeddings
- [wide](https://crates.io/crates/wide) — Portable SIMD for Rust
- [half](https://crates.io/crates/half) — IEEE 754 half-precision floats
- [memmap2](https://crates.io/crates/memmap2) — Memory-mapped file I/O
- [Model2Vec](https://github.com/MinishLab/model2vec) — Static token embedding models

Informed by the hybrid search implementations in:
- [cass](https://github.com/Dicklesworthstone/cass) — Claude Agent Session Search
- [xf](https://github.com/Dicklesworthstone/xf) — X/Twitter archive search
- [mcp_agent_mail_rust](https://github.com/Dicklesworthstone/mcp_agent_mail_rust) — Agent coordination mail system

---

## About Contributions

Please don't take this the wrong way, but I do not accept outside contributions for any of my projects. I simply don't have the mental bandwidth to review anything, and it's my name on the thing, so I'm responsible for any problems it causes; thus, the risk-reward is highly asymmetric from my perspective. I'd also have to worry about other "stakeholders," which seems unwise for tools I mostly make for myself for free. Feel free to submit issues, and even PRs if you want to illustrate a proposed fix, but know I won't merge them directly. Instead, I'll have Claude or Codex review submissions via `gh` and independently decide whether and how to address them. Bug reports in particular are welcome. Sorry if this offends, but I want to avoid wasted time and hurt feelings. I understand this isn't in sync with the prevailing open-source ethos that seeks community contributions, but it's the only way I can move at this velocity and keep my sanity.

---

## License

MIT
