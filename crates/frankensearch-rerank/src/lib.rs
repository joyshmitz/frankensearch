//! Cross-encoder reranking for frankensearch.
//!
//! The pure-Rust [`NativeReranker`] (feature `native`) runs a BERT
//! cross-encoder forward pass on frankentorch tensors — no ONNX / `ort` /
//! native C++ dependency — producing `sigmoid(logit)` relevance scores in
//! `[0, 1]`. An optional FastEmbed backend is available behind
//! `fastembed-reranker`.
//!
//! ```text
//! (query, document) → tokenize → frankentorch forward → logit → sigmoid → score ∈ [0, 1]
//! ```

pub mod pipeline;

#[cfg(feature = "fastembed-reranker")]
pub mod fastembed_reranker;

pub use pipeline::{DEFAULT_MIN_CANDIDATES, DEFAULT_TOP_K_RERANK, rerank_step};

#[cfg(feature = "fastembed-reranker")]
pub use fastembed_reranker::FastEmbedReranker;

#[cfg(feature = "native")]
pub mod native;

#[cfg(feature = "native")]
pub mod native_embedder;

#[cfg(feature = "native")]
pub use native::NativeReranker;

#[cfg(feature = "native")]
pub use native_embedder::NativeEmbedder;

/// Default model directory name for the cross-encoder reranker.
pub const DEFAULT_MODEL_NAME: &str = "rerank-default-v1";

/// Default maximum input token length for cross-encoder query/document pairs.
pub const DEFAULT_MAX_LENGTH: usize = 512;
