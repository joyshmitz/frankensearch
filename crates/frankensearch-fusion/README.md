# frankensearch-fusion

RRF fusion, score blending, and two-tier progressive search orchestration for frankensearch.

## Overview

This crate is the search orchestration layer of frankensearch. It combines results from vector search and lexical search using Reciprocal Rank Fusion (RRF), blends fast-tier and quality-tier scores, and manages the progressive two-phase search pipeline. It also provides adaptive fusion, score calibration, circuit breaking, feedback collection, and federated multi-index search.

## Key Types

### Fusion and Blending

- `rrf_fuse` - Reciprocal Rank Fusion (K=60) with 4-level tie-breaking
- `blend_two_tier` - score blending (default 0.7 quality / 0.3 fast)
- `RrfConfig` - RRF parameter configuration
- `MmrConfig` / `mmr_rerank` - Maximal Marginal Relevance for diversity-aware reranking

### Search Orchestration

- `TwoTierSearcher` - progressive iterator orchestrator yielding `SearchPhase` results
- `IncrementalSearcher` - incremental search with configurable strategy and budget
- `FederatedSearcher` / `FederatedFusion` - search across multiple indices with result merging

### Score Calibration

- `ScoreCalibrator` - trait for score normalization
- `PlattScaling` - Platt sigmoid calibration
- `IsotonicRegression` - monotone score calibration
- `TemperatureScaling` - temperature-based softmax calibration

### Adaptive and Quality Control

- `AdaptiveFusion` - Bayesian-adaptive fusion weight tuning per query class
- `CircuitBreaker` - quality-tier health monitoring with automatic skip on failure
- `PhaseGate` - anytime-valid sequential testing for phase transition decisions
- `FeedbackCollector` - implicit relevance feedback with exponentially-decaying boost map

### Caching and Refresh

- `IndexCache` / `SentinelFileDetector` - index staleness detection and cache invalidation
- `RefreshWorker` - background index refresh with configurable polling
- `EmbeddingQueue` - async embedding job queue with priority and backpressure

### Normalization

- `normalize_scores` - min-max, z-score, or identity score normalization
- `NormalizationMethod` - configurable normalization strategy

### Pseudo-Relevance Feedback

- `prf_expand` - pseudo-relevance feedback query expansion

### Graph Ranking (feature: `graph`)

- `GraphRanker` - graph-boosted ranking using document relationships

## Usage

```rust
use std::sync::Arc;
use frankensearch_fusion::{TwoTierSearcher, rrf_fuse, RrfConfig};
use frankensearch_core::{TwoTierConfig, VectorHit, ScoredResult};

// Fuse vector and lexical results
let vector_hits = vec![/* VectorHit results */];
let lexical_hits = vec![/* ScoredResult results */];
let config = RrfConfig::default();
let fused = rrf_fuse(&vector_hits, &lexical_hits, &config);

// Two-tier progressive search
// let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());
// let (results, metrics) = searcher.search_collect(&cx, "query", 10).await?;
```

## Dependency Graph Position

```
frankensearch-core
  ^    ^    ^
  |    |    |
  embed index lexical (optional)  rerank (optional)
    \   |   /                      /
     \  |  /                      /
  frankensearch-fusion -----------
        ^
        |-- frankensearch (root)
```

## License

MIT
