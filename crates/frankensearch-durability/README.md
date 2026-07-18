# frankensearch-durability

RaptorQ durability primitives for frankensearch indices.

## Overview

This crate provides a durability layer for frankensearch index artifacts using RaptorQ fountain codes (via FrankenSQLite's `SymbolCodec`). It enables self-healing of corrupted index files by encoding repair symbols into binary trailers appended to protected files. If corruption is detected (via CRC32 or xxHash verification), the repair data can reconstruct the original content without external backups.

The crate is engine-neutral by construction (bd-tkjm): Tantivy appears nowhere in its dependency graph. Engine-specific wrappers compose with the generic `FileProtector` boundary from their own crates; the retired `DurableTantivyIndex`/`TantivySegmentProtector` wrapper (`src/tantivy_wrapper.rs`) is no longer part of the build, and its removal proposal rides with the post-flip retirement sweep (quill-e9.3).

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

### Tantivy Segment Protection (retired)

`DurableTantivyIndex` and `TantivySegmentProtector` were removed from the build in the Quill migration (bd-tkjm) so this crate stays engine-neutral; `src/tantivy_wrapper.rs` remains on disk only until the quill-e9.3 retirement sweep lands its approved removal.

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
