# frankensearch-durability

RaptorQ durability primitives for frankensearch indices.

## Overview

This crate provides a durability layer for frankensearch index artifacts using RaptorQ fountain codes (via FrankenSQLite's `SymbolCodec`). It enables self-healing of corrupted vector index files and Tantivy segments by encoding repair symbols into binary trailers appended to protected files. If corruption is detected (via CRC32 or xxHash verification), the repair data can reconstruct the original content without external backups.

## Key Types

### Codec

- `RepairCodec` / `RepairCodecConfig` - configurable RaptorQ encode/decode facade
- `DefaultSymbolCodec` / `CodecFacade` - default symbol codec implementation
- `EncodedPayload` / `DecodedPayload` - encoded/decoded data wrappers
- `VerifyResult` - verification outcome (intact, repairable, unrecoverable)
- `classify_decode_failure` - categorizes decode failures for diagnostics

### File Protection

- `FileProtector` - protects, verifies, and repairs arbitrary files using repair trailers
- `DurabilityProvider` - trait for pluggable durability backends
- `NoopDurability` - no-op implementation for when durability is disabled
- `FileHealth` / `FileVerifyResult` - file integrity status
- `FileRepairOutcome` / `FileProtectionResult` - repair and protection results
- `DirectoryHealthReport` / `DirectoryProtectionReport` - directory-wide health summaries
- `RepairPipelineConfig` - configuration for multi-file repair orchestration

### FSVI Protection

- `FsviProtector` - specialized protector for FSVI vector index files
- `FsviVerifyResult` / `FsviRepairResult` / `FsviProtectionResult` - FSVI-specific outcomes

### Tantivy Segment Protection

- `DurableTantivyIndex` - durability wrapper for Tantivy lexical indices
- `TantivySegmentProtector` - per-segment protection for Tantivy index files
- `SegmentHealthReport` / `SegmentProtectionReport` - segment-level health diagnostics

### Repair Trailer Format

- `RepairTrailerHeader` - binary trailer header with magic bytes and version
- `RepairSymbol` - individual repair symbol stored in the trailer
- `serialize_repair_trailer` / `deserialize_repair_trailer` - trailer I/O

### Metrics

- `DurabilityMetrics` / `DurabilityMetricsSnapshot` - protection/repair telemetry counters
- `DurabilityConfig` - global durability configuration

## Usage

```rust
use frankensearch_durability::{FileProtector, DurabilityConfig};

let config = DurabilityConfig::default();
let protector = FileProtector::new(config);

// Protect a file by appending repair symbols
// let result = protector.protect("/path/to/vectors.fsvi")?;

// Verify integrity
// let health = protector.verify("/path/to/vectors.fsvi")?;

// Repair if corrupted
// let outcome = protector.repair("/path/to/vectors.fsvi")?;
```

## Dependency Graph Position

```
frankensearch-core
  ^
  |
frankensearch-durability
  ^
  |-- frankensearch (root, optional, feature: durability)
```

## License

MIT
