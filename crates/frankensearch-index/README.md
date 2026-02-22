# frankensearch-index

FSVI vector index, SIMD dot product, and top-k search for frankensearch.

## Overview

This crate implements the FSVI (FrankenSearch Vector Index) binary format for storing and searching dense vector embeddings. It provides memory-mapped I/O for zero-copy access, brute-force exact top-k search with SIMD acceleration, and an optional HNSW approximate nearest-neighbor index for large-scale deployments.

The FSVI format is designed for cache-line and SIMD friendliness, with 64-byte aligned vector slabs supporting both f32 and f16 quantization.

## FSVI File Layout

```text
+-------------------------------------------+
| Header (variable length)                  |
|   magic: b"FSVI"              (4 bytes)   |
|   version: u16                (2 bytes)   |
|   embedder_id + revision      (variable)  |
|   dimension: u32              (4 bytes)   |
|   quantization: u8            (1 byte)    |
|   record_count: u64           (8 bytes)   |
|   vectors_offset: u64         (8 bytes)   |
|   header_crc32: u32           (4 bytes)   |
+-------------------------------------------+
| Record Table (16 bytes per record)        |
+-------------------------------------------+
| String Table (UTF-8 doc IDs)              |
+-------------------------------------------+
| Padding (to 64-byte alignment)            |
+-------------------------------------------+
| Vector Slab (record_count x dim x elem)   |
+-------------------------------------------+
```

## Key Types

- `VectorIndex` - memory-mapped FSVI index reader with brute-force search
- `VectorIndexWriter` - builder for writing new FSVI files
- `TwoTierIndex` / `TwoTierIndexBuilder` - manages fast + quality index pair for two-tier search
- `VectorMetadata` - parsed header metadata (embedder ID, dimension, quantization, record count)
- `Quantization` - element type enum (F32 or F16)
- `VacuumStats` - statistics from tombstone compaction
- `SearchParams` - search configuration (top-k, parallel thresholds)
- `ScalarQuantizer` - f32-to-f16 quantization utilities
- `HnswIndex` / `HnswConfig` - HNSW approximate nearest-neighbor index (feature: `ann`)
- `MrlConfig` - Matryoshka Representation Learning configuration for progressive dimensionality
- `WalConfig` / `CompactionStats` - write-ahead log for atomic index updates
- `WarmUpConfig` / `WarmUpStrategy` - adaptive index warmup with heat map tracking

## SIMD Functions

- `dot_product_f16_f32` - SIMD dot product between f16 and f32 vectors
- `dot_product_f32_f32` - SIMD dot product between two f32 vectors
- `cosine_similarity_f16` - cosine similarity for f16 vectors

## Usage

```rust
use frankensearch_index::{VectorIndex, TwoTierIndex, TwoTierIndexBuilder};
use frankensearch_core::TwoTierConfig;

// Open a two-tier index (fast + quality)
let index = TwoTierIndex::open("./my_index", TwoTierConfig::default())
    .expect("open index");

// Low-level: open a single FSVI file
let vi = VectorIndex::open("./my_index/vectors_fast.fsvi")
    .expect("open vector index");
println!("Records: {}, Dim: {}", vi.metadata().record_count, vi.metadata().dimension);
```

## Dependency Graph Position

```
frankensearch-core
  ^
  |
frankensearch-index
  ^
  |-- frankensearch-fusion
  |-- frankensearch-fsfs
  |-- frankensearch (root)
```

## License

MIT
