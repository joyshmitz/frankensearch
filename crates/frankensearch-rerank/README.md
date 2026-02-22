# frankensearch-rerank

Cross-encoder reranking for frankensearch using FlashRank and FastEmbed.

## Overview

This crate provides cross-encoder reranking, which processes the query and each candidate document *together* through a transformer model for dramatically more accurate relevance scoring than bi-encoder (embedding) approaches. The primary implementation uses ONNX Runtime for inference with sigmoid activation on raw logits.

Cross-encoders cannot pre-compute embeddings, so they are used as a reranking step on a shortlist of candidates produced by faster retrieval methods.

```text
(query, document) -> tokenize -> ONNX -> logit -> sigmoid -> score in [0, 1]
```

## Key Types

- `FlashRankReranker` - ONNX Runtime cross-encoder reranker implementing the `Reranker` trait
- `FastEmbedReranker` - alternative reranker using the fastembed library (feature: `fastembed-reranker`)
- `rerank_step` - utility function that runs reranking on a candidate list with configurable top-k and minimum candidate thresholds
- `DEFAULT_TOP_K_RERANK` - default number of top results to keep after reranking
- `DEFAULT_MIN_CANDIDATES` - minimum candidates required before reranking is triggered

## Model Layout

Required files in the model directory:

- `onnx/model.onnx` (preferred) or `model.onnx` (legacy)
- `tokenizer.json`

## Features

| Feature | Description |
|---------|-------------|
| `fastembed-reranker` | Enables `FastEmbedReranker` as an alternative backend |

## Usage

```rust
use frankensearch_rerank::FlashRankReranker;

// Load a cross-encoder model
let reranker = FlashRankReranker::load("/path/to/flashrank/model")
    .expect("load reranker model");

// Rerank candidates (via the Reranker trait)
// let reranked = reranker.rerank(&cx, "search query", &documents, 10).await?;
```

## Dependency Graph Position

```
frankensearch-core
  ^
  |
frankensearch-rerank
  ^
  |-- frankensearch-fusion (optional, feature: rerank)
  |-- frankensearch (root, optional, feature: rerank)
```

## License

MIT
