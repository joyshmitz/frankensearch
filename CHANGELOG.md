# Changelog

All notable changes to [frankensearch](https://github.com/Dicklesworthstone/frankensearch) are documented here.

Entries correspond to [GitHub Releases](https://github.com/Dicklesworthstone/frankensearch/releases) unless noted otherwise. Tags that share a commit with another release are called out explicitly. Each entry links to representative commits using full commit URLs.

---

## [Unreleased](https://github.com/Dicklesworthstone/frankensearch/compare/v1.1.4...HEAD)

---

## [v1.1.4](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.1.4) -- 2026-03-22

> **Release with binary assets for all platforms** -- [Full diff from v1.1.3](https://github.com/Dicklesworthstone/frankensearch/compare/v1.1.3...v1.1.4)
>
> The CI was fixed in March to include aarch64-linux-musl and x86_64-apple-darwin targets,
> but no release was tagged after v1.1.3. This release makes binaries available.
> Closes #7, closes #23.
>
> Commits since v1.1.3 (2026-02-23) through v1.1.4 (2026-03-22).

### Cloud API Embedding Providers

- Add pluggable cloud API embedding abstraction supporting OpenAI and Gemini backends, with HTTP transport, automatic retry, token-bucket rate limiting, and L2 normalization ([e5b7bab](https://github.com/Dicklesworthstone/frankensearch/commit/e5b7bab7d6a303c3503b6b7e99509d94b484c812), [d37d506](https://github.com/Dicklesworthstone/frankensearch/commit/d37d506da533fa7ded49ceb0b36af3e5214e0a40))
- Support query-param authentication for Gemini via `request_url()` trait method ([9846bb0](https://github.com/Dicklesworthstone/frankensearch/commit/9846bb03d4e1097138960fd500b44b83e4da426e))
- Fix rate limiter token drain and separate OpenAI/Gemini auto-detect paths ([e57f2de](https://github.com/Dicklesworthstone/frankensearch/commit/e57f2de8c4fe0f5e557a58be8e5e399acb03834e))
- Thread `Cx` through download and API embedding pipelines for asupersync 0.2.9 cancellation support ([8e369b4](https://github.com/Dicklesworthstone/frankensearch/commit/8e369b4c640510ea109d751659bd73ff2dde8f2a))

### Tokenizer & Search

- Preserve hyphenated bead IDs (e.g. `bd-q3fy`) in cass tokenizer by replacing `SimpleTokenizer` with a regex tokenizer and `HyphenDecompose` filter; schema bumped v6 to v7 ([11db96a](https://github.com/Dicklesworthstone/frankensearch/commit/11db96ae541659f7422c9b66f98c52f9d294b872))
- Use `[a-zA-Z0-9]` instead of `\w` in tokenizer regex to exclude underscores, fixing index/query mismatch for underscore-separated terms ([472322f](https://github.com/Dicklesworthstone/frankensearch/commit/472322fdcedea90d72209e79f845875e1d8703e3))

### In-Memory Vector Index

- Add fully-resident in-memory vector index with f16 quantization, enabling use cases that skip disk entirely ([eee7b73](https://github.com/Dicklesworthstone/frankensearch/commit/eee7b73dee846d4601cee9b8b67ad66147e4f880))
- Add synchronous two-tier search API alongside in-memory index improvements ([e081bc5](https://github.com/Dicklesworthstone/frankensearch/commit/e081bc578ada4ed9e9dab06476b3737e3dad6541))

### WAL-Based Incremental Mutations (fsfs CLI)

- Add `append-batch`, `delete`, `compact`, and `daemon` commands for WAL-based incremental index mutation without full rebuilds ([0fadc4d](https://github.com/Dicklesworthstone/frankensearch/commit/0fadc4d3fdfc7c5f0ccafcbf5eb3f93747e375af))
- Document WAL commands in help text and shell completions ([1b88d5e](https://github.com/Dicklesworthstone/frankensearch/commit/1b88d5e46e494116defb13dde03499f22397c2bc))

### Async Runtime Compatibility

- Rename `LockError::PolledAfterCompletion` to `Cancelled` across all crates for clarity ([5151d47](https://github.com/Dicklesworthstone/frankensearch/commit/5151d473b1ff09836c89e381ed49f317f4ec7ffb))
- Handle `PolledAfterCompletion` LockError variant in embed, lexical, and rerank crates ([1359551](https://github.com/Dicklesworthstone/frankensearch/commit/1359551eaaa8c098474083738e68f8593456921c))
- Update `asupersync` to 0.2.8 and then 0.2.9 ([ba3ab85](https://github.com/Dicklesworthstone/frankensearch/commit/ba3ab85f0c8a0d713706d15556101cf32fce9683), [ede9fc8](https://github.com/Dicklesworthstone/frankensearch/commit/ede9fc8ec4f64f8e452e02ce9a0fbbd97535c0a8))
- Remove 6 unreachable duplicate `Cancelled` match arms across lexical, rerank, and embed crates ([3e6b29e](https://github.com/Dicklesworthstone/frankensearch/commit/3e6b29e4a6898ebc9ae9d8dea50eb95e2958c3e3))

### Build & CI

- Decouple release CI builds from quality gate so tag pushes produce artifacts ([9b9accb](https://github.com/Dicklesworthstone/frankensearch/commit/9b9accb51ea9c2c264aec428df886c91f71fc471))
- Exclude `tools/optimize_params` from default build; add `--recurse-submodules` to install.sh ([35becff](https://github.com/Dicklesworthstone/frankensearch/commit/35becff5ceda7865e49f3c4923410a6bcb1c61e3))
- Remove invalid `const fn` qualifiers from phase gate builder methods ([900898b](https://github.com/Dicklesworthstone/frankensearch/commit/900898b8c770df2b758793331870e82c8fd607c9))

### Tests

- Add comprehensive test coverage for fsfs update verification path ([da7acf5](https://github.com/Dicklesworthstone/frankensearch/commit/da7acf55a633b5a28e6f82118bd14b4b43ce1383))

---

## [v1.1.3](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.1.3) -- 2026-02-23

> **Bug fixes, PDF extraction, lite builds** -- [Full diff from v1.1.2](https://github.com/Dicklesworthstone/frankensearch/compare/v1.1.2...v1.1.3)
>
> Annotated tag. Published as a GitHub Release.

### PDF & Document Processing

- Native PDF text extraction -- `fsfs index` and `fsfs search` can now process PDF files directly without external tooling ([efc26cc](https://github.com/Dicklesworthstone/frankensearch/commit/efc26cc2fd5b6444efb1efcfa45cabb8a030982e))

### Lite / Offline Builds

- `embedded-models` feature flag for lite/offline builds that bundle models at compile time ([9fbdbd6](https://github.com/Dicklesworthstone/frankensearch/commit/9fbdbd6a82c4a0443aef5756c5b433d24cdd42d7))

### Search Quality & Observability

- Rank movement explanations -- `TwoTierSearcher` now surfaces why a result moved between Phase 1 and Phase 2 ([117f955](https://github.com/Dicklesworthstone/frankensearch/commit/117f95539b4387c1a1e96516618cfc386d663b59))

### Installer & Update Pipeline

- Beautiful download progress with file-size display ([2886323](https://github.com/Dicklesworthstone/frankensearch/commit/2886323fa2de42738e3e51681ed2f1f8fef7f53a))
- Fix six security and correctness issues in installer and update logic ([97ac428](https://github.com/Dicklesworthstone/frankensearch/commit/97ac428f729eb830e06f19f7f6b04f2049f9506f))
- Harden release asset URL and checksum handling ([20aa045](https://github.com/Dicklesworthstone/frankensearch/commit/20aa045be191d4ce76b7dbf490362c7ae906e9b3))

### Bug Fixes

- Fix seven code-review bugs: ANSI box border rendering, `.max` vs `.min` confusion, `pdf_extract` panic guard, model ID lookup, else-if cleanup, `--expand`+`--daemon` warning ([f21ba72](https://github.com/Dicklesworthstone/frankensearch/commit/f21ba7225e0c181c0eafd1e161236a8eb4802507))
- Improve searcher module reliability ([97e340d](https://github.com/Dicklesworthstone/frankensearch/commit/97e340da0ddff51ba2bdb2a76b8bab018f6b151a))

---

## [v1.1.2](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.1.2) -- 2026-02-22

> **Fix version reporting, update mechanism, and TUI visibility** -- [Full diff from v1.1.0](https://github.com/Dicklesworthstone/frankensearch/compare/v1.1.0...v1.1.2)
>
> Lightweight tag. Published as a GitHub Release.

### Release & Update Infrastructure

- Overhaul release asset system with proper Windows support and SHA256SUMS, refactor pipeline dedup logic ([7f4f8c7](https://github.com/Dicklesworthstone/frankensearch/commit/7f4f8c732f0918adcf3a3ce974fa84bf78dbae6b))
- **Version reporting** -- binary now correctly reports its actual version instead of `v0.1.0`
- **Update mechanism** -- `fsfs update` constructs correct download URLs matching release asset naming (`fsfs-{version}-{triple}.{ext}`)
- **SHA256SUMS** -- update verification downloads the release-level checksum file instead of per-artifact sidecars
- **Windows target triple** -- correct detection for `x86_64-pc-windows-msvc` and `aarch64-pc-windows-msvc`

### CLI & UX

- **TUI visibility** -- running `fsfs` with no args now prints diagnostic messages when the TUI exits
- Tighten root probe limits, expand excluded directories, and improve first-run UX ([3cdae25](https://github.com/Dicklesworthstone/frankensearch/commit/3cdae25fa2e47360edab7bed9e9e6cd3fe07db1d))

### Performance & Correctness

- Use `HashSet` for O(1) duplicate `doc_id` detection in `TwoTierIndexBuilder` ([8cd272a](https://github.com/Dicklesworthstone/frankensearch/commit/8cd272a9b714dfdf7ce5ad294d1095c3e24d9ce4))
- Use `BTreeSet` for deterministic path ordering in `PendingEvents` ([378af78](https://github.com/Dicklesworthstone/frankensearch/commit/378af788b60319a1715cc56a9c3e069723a258e7))
- Populate actual index path in WAL `IndexCorrupted` error for better diagnostics ([b72bfd4](https://github.com/Dicklesworthstone/frankensearch/commit/b72bfd41e4b9088dedbc382f1ebb7252e797c54b))

---

## [v1.1.1](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.1.1) -- 2026-02-22

> [Full diff from v1.1.0](https://github.com/Dicklesworthstone/frankensearch/compare/v1.1.0...v1.1.1)
>
> Lightweight tag pointing to the same commit as v1.1.0 (`82c18ff`). Published as a separate GitHub Release with patched release notes describing fixes that were folded into the v1.1.0 binary.

**Note:** v1.1.0 and v1.1.1 share the same commit. v1.1.1 was cut as a quick-follow release to document first-run fixes that shipped in the v1.1.0 binary but warranted explicit callout.

### Bug Fixes (documented retroactively)

- **Fix first-run hang on macOS** -- reduced filesystem probe budget (depth 2, 10K entries max) and excluded macOS system directories (`Library`, `Pictures`, `Movies`, etc.)
- **Add progress indicator** -- prints "Scanning for indexable directories..." before filesystem probe begins
- **SHA256SUMS filename** -- checksum file now has the correct name

### Platforms

| Platform | Asset |
|----------|-------|
| Linux x86_64 | `fsfs-1.1.1-x86_64-unknown-linux-musl.tar.xz` |
| macOS arm64 (Apple Silicon) | `fsfs-1.1.1-aarch64-apple-darwin.tar.xz` |
| Windows x86_64 | `fsfs-1.1.1-x86_64-pc-windows-msvc.zip` |

---

## [v1.1.0](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.1.0) -- 2026-02-22

> **Crates.io publishing, resilient indexing, macOS Apple Silicon support** -- [Full diff from v1.0.0](https://github.com/Dicklesworthstone/frankensearch/compare/v1.0.0...v1.1.0)
>
> Lightweight tag. Published as a GitHub Release.

### Crates.io Publishing & Dependency Migration

- Switch all workspace dependencies from local path references to crates.io registry versions ([1fceef4](https://github.com/Dicklesworthstone/frankensearch/commit/1fceef4a5e62a33ce2b50ab5edede0516763a145))
- Bump core to v0.1.2 and index to v0.1.2 for crates.io republish ([c1676fb](https://github.com/Dicklesworthstone/frankensearch/commit/c1676fbe5b0d78f0ec0ac2735b02d2572fa8d25f))
- Add README.md files for all 12 crates for crates.io presentation ([790cbb0](https://github.com/Dicklesworthstone/frankensearch/commit/790cbb0d45a6b7cf241d44415e1e6b971f6d8700))
- Adapt to ftui-text 0.2.1 API changes ([be772f7](https://github.com/Dicklesworthstone/frankensearch/commit/be772f7e80286b6b66fc0cf6103f8d316451ae9a))

### Resilient Indexing Pipeline

- Checkpoint resume, embedding retries, degraded-mode completion, watcher auto-restart, and heap/normalization fixes ([4547323](https://github.com/Dicklesworthstone/frankensearch/commit/45473238fa4cdcf5b3408a5303b8efd026c48563))
- Prevent infinite loop in snapshot walker on symlink cycles ([af02a5d](https://github.com/Dicklesworthstone/frankensearch/commit/af02a5d73693b99ad59cb761c6519e3b096c319e))
- Recognize Johnson-Lindenstrauss embedders as hash embedders in storage ([1738a6b](https://github.com/Dicklesworthstone/frankensearch/commit/1738a6bd2f87da22f55eaf87acbbbd0a787911b0))

### Search Pipeline Hardening

- Fix nested markdown links, optimize diff, stabilize MMR, add bounds checks ([ee88129](https://github.com/Dicklesworthstone/frankensearch/commit/ee881297165bb75912a9a0df158d1b5d22474539))
- Improve identifier detection, WAL-first lookups, score normalization, and job queue dedup ([a74d3f2](https://github.com/Dicklesworthstone/frankensearch/commit/a74d3f2c0fe6e597cd5e7a73d191776c3bad042b))
- Model manifest expansion, index reconciliation, and storage pipeline hardening ([adc3f45](https://github.com/Dicklesworthstone/frankensearch/commit/adc3f454a189179eb95ab39bc568c761c5b9f3fd))
- Deduplicate WAL entries and fix ULID generation with telemetry prefix ([595aa48](https://github.com/Dicklesworthstone/frankensearch/commit/595aa48937d56fb48677daaa54f0f1f39dfc9d59))

### Platform Support

- **macOS arm64 (Apple Silicon)** added as a first-class release target with pre-built binary
- Tighten root probe limits and expand excluded directories for better first-run UX on macOS ([3cdae25](https://github.com/Dicklesworthstone/frankensearch/commit/3cdae25fa2e47360edab7bed9e9e6cd3fe07db1d))
- Fix `version` subcommand (use subcommand instead of `--version` flag) in install script self-test ([35dbcd5](https://github.com/Dicklesworthstone/frankensearch/commit/35dbcd5f3cf8a21ef86745bbe1444fda4727039c))

### Platforms

| Platform | Asset |
|----------|-------|
| Linux x86_64 | `fsfs-1.1.0-x86_64-unknown-linux-musl.tar.xz` |
| macOS arm64 (Apple Silicon) | `fsfs-1.1.0-aarch64-apple-darwin.tar.xz` |
| Windows x86_64 | `fsfs-1.1.0-x86_64-pc-windows-msvc.zip` |

---

## [v1.0.0](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.0.0) -- 2026-02-21

> **First stable release** -- [Full diff from v0.1.0](https://github.com/Dicklesworthstone/frankensearch/compare/v0.1.0...v1.0.0)
>
> Lightweight tag. Published as a GitHub Release.

First stable release of the `fsfs` CLI, marking the two-tier hybrid local search engine as production-ready.

### Canonicalization & Reranking Overhaul

- Rewrite the query canonicalization pipeline and add fastembed-based cross-encoder reranker ([1445c78](https://github.com/Dicklesworthstone/frankensearch/commit/1445c7805cad0b6768b724a5e93b74f19cf9de65))
- Add multi-model ONNX embedder support via `OnnxEmbedderConfig` ([41c69be](https://github.com/Dicklesworthstone/frankensearch/commit/41c69be0791bf5689fee97134983a7e69e9ebdfc))

### Stability & Correctness

- Work around VDBE `sqlite_master` parameterized query limitation ([567d816](https://github.com/Dicklesworthstone/frankensearch/commit/567d8160efe1d6ef74c7070667888b174fa6e65a))
- Fix daemon modules, exclude pattern suffix matching, and clippy warnings ([2ca9040](https://github.com/Dicklesworthstone/frankensearch/commit/2ca90406e8fcf829b9a8f3ac29063916cff9fa7d))
- Resolve workspace-level dead code errors and fix 2 failing fsfs tests ([d7f144f](https://github.com/Dicklesworthstone/frankensearch/commit/d7f144f71573e04f6ae6b4173f2b787422131223))
- Resolve compilation errors, lifetime annotations, and clippy warnings across workspace ([3a280c6](https://github.com/Dicklesworthstone/frankensearch/commit/3a280c663f59e3dcaf0cbee62ea0157195467ef4))
- Fix Windows build portability and installer checksum fallback ([7f8649e](https://github.com/Dicklesworthstone/frankensearch/commit/7f8649e04a0a1953a21d9a43970d12611f3a681c))
- Remove spurious lifetime annotations, update sibling dep refs, fix embed auto_detect ([0bdbac9](https://github.com/Dicklesworthstone/frankensearch/commit/0bdbac941b9e3deefeeb280a2debbf9a9f4105ae))

### Runtime & Configuration

- Refine FSFS runtime configuration and update ops dashboard screens ([139b70d](https://github.com/Dicklesworthstone/frankensearch/commit/139b70dfdd6d07f82df85b92cb6c83d7e43038e8))

### Platforms

| Platform | Asset |
|----------|-------|
| Linux x86_64 | `fsfs-1.0.0-x86_64-unknown-linux-musl.tar.xz` |
| macOS arm64 (Apple Silicon) | `fsfs-1.0.0-aarch64-apple-darwin.tar.xz` |
| Windows x86_64 | `fsfs-1.0.0-x86_64-pc-windows-msvc.zip` |

---

## [v0.1.0](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v0.1.0) -- 2026-02-19

> **Initial public release** -- [Full diff from initial commit](https://github.com/Dicklesworthstone/frankensearch/compare/42156796c494...v0.1.0)
>
> Annotated tag. Published as a GitHub Release.

The first public release of frankensearch -- a two-tier hybrid local search engine for Rust with the `fsfs` standalone CLI. This release includes the full workspace of 11 crates and approximately 240 commits over 6 days of intensive multi-agent development (2026-02-13 through 2026-02-19).

### Two-Tier Progressive Search Engine

- Fast initial results (<15 ms target) followed by quality-refined results (~150 ms target)
- Reciprocal Rank Fusion (RRF) combining lexical BM25 and semantic vector similarity
- Configurable score blending between fast and quality tiers (`quality_weight`)
- Graceful degradation -- `SearchPhase::RefinementFailed` preserves Phase 1 results when quality tier errors or times out
- Deterministic ordering for reproducible ranking via stable tie-break logic
- Query classification (`identifier`, `short keyword`, `natural language`) with adaptive budgets
- Exclusion queries (`-term`) for filtering unwanted results
- Negation-aware canonicalization pipeline

### Embedding System

- Hash embedder (zero model downloads) for dev/CI
- `model2vec` embedder (potion-multilingual-128M for fast tier)
- `fastembed` embedder (all-MiniLM-L6-v2 for quality tier)
- Automatic embedder stack detection and fallback ([54b11f6](https://github.com/Dicklesworthstone/frankensearch/commit/54b11f63f66b201d6df62731fee1c871c2c4ce04))
- Streaming model downloads with manifest validation and batch size clamping ([5a09ad9](https://github.com/Dicklesworthstone/frankensearch/commit/5a09ad9dfdb16ed159db0c376c61ef48f1fc8e9f))
- Filesystem-backed verification cache for model manifests ([0d1b78a](https://github.com/Dicklesworthstone/frankensearch/commit/0d1b78a1feb2023f566d7b18549afdb17ffdef9f))
- Dimension reduction support

### Vector Index (FSVI Format)

- Memory-mapped on-disk format with f16 quantization by default
- SIMD dot products with NaN-safe total ordering
- Zero-alloc byte-level SIMD dot products eliminating thread-local scratch buffers ([3ae2caa](https://github.com/Dicklesworthstone/frankensearch/commit/3ae2caa70d9078189170ad33b529c45120ea61b4))
- Heap-based top-k selection
- WAL journaling for crash recovery with corrupted trailer detection ([cb43051](https://github.com/Dicklesworthstone/frankensearch/commit/cb43051f3da60527360d03576b45831b635c6c98))
- HNSW approximate nearest-neighbor path with persistence and graph-ranking integration ([3409623](https://github.com/Dicklesworthstone/frankensearch/commit/3409623b66fecb3a1d7f7880e1e9ddb39ffa6a55), [3ad4326](https://github.com/Dicklesworthstone/frankensearch/commit/3ad4326fd63a3f53d0d90d48f0e5ea1c74c2a4da))
- Soft-delete with rollback support ([8dd09e6](https://github.com/Dicklesworthstone/frankensearch/commit/8dd09e6b8e2e7ce030271acb17053dd2509f1d7a))
- Mmap-backed VectorIndex ([a0a96fd](https://github.com/Dicklesworthstone/frankensearch/commit/a0a96fd4dee03ca5eb1d559bcae8ba6148109858))

### Fusion Pipeline

- Adaptive fusion with circuit breaker and phase gating ([e9551bb](https://github.com/Dicklesworthstone/frankensearch/commit/e9551bb60254826c451bc9435017b2be597e31ad))
- Pseudo-relevance feedback (PRF) and Maximal Marginal Relevance (MMR)
- Conformal calibration for score confidence
- Query-biased PageRank graph ranking and 3-input RRF ([da611e6](https://github.com/Dicklesworthstone/frankensearch/commit/da611e6596ef05e099bcb2697869300bc1f7bbfc))
- RRF explain support ([b1541ca](https://github.com/Dicklesworthstone/frankensearch/commit/b1541ca254c387f9dfd4ac348aad3906b0d27faf))
- NaN/Infinity fallback in RRF scoring ([5dba362](https://github.com/Dicklesworthstone/frankensearch/commit/5dba3628ae604a1f45634ac6d2b5901d3de03ff2))
- Federated search with interaction testing infrastructure ([de2bddc](https://github.com/Dicklesworthstone/frankensearch/commit/de2bddc80961a5d2c4f9e3d96577d0ab051a79c7))

### FSFS CLI Product

- `fsfs search` with progressive delivery (`--stream`, `--format jsonl/toon/csv/table/json`)
- `fsfs index` with filesystem watching and incremental updates
- `fsfs explain` for result explanation surfaces
- `fsfs doctor` for health checks
- `fsfs status` for runtime diagnostics
- Daemon transport with query caching, unbounded-recall tuning, and fallback when daemon unavailable ([e4870f8](https://github.com/Dicklesworthstone/frankensearch/commit/e4870f82809ea42969ddf05a513ee9a60ab35321), [068121d](https://github.com/Dicklesworthstone/frankensearch/commit/068121dca3382a618a8dd8ad770c9f2b7dfe8bb7))
- Auto-detect output format for non-TTY environments ([b7813295](https://github.com/Dicklesworthstone/frankensearch/commit/b7813295ca0f280197f958f0e5db11d28c6f22f8))
- Adaptive debounce engine for search execution ([85a4380](https://github.com/Dicklesworthstone/frankensearch/commit/85a4380838ef7fe81d1e430e1401480f2159f77a))
- Semantic VOI gate, cass-compatible Tantivy index, and TUI render/timing overhaul ([3e5bcfb](https://github.com/Dicklesworthstone/frankensearch/commit/3e5bcfba3b9ad093f5cbdac52d8fa9f9b9e5053e))
- Pressure-aware backoff, redaction hardening, and repro diagnostic expansion ([40839e1](https://github.com/Dicklesworthstone/frankensearch/commit/40839e1abfec2ab1197fde30737e9677f60cf4a2))
- Batch indexing pipeline, vector skip diagnostics, and WAL stale-detection fix ([5d40423](https://github.com/Dicklesworthstone/frankensearch/commit/5d40423d3f23f7935b359fafdbebb8b56b2bdf16))

### Storage & Durability

- FrankenSQLite-backed metadata persistence with content-hash dedup
- Immediate transactions, ingest pipeline, staleness detection ([fe4fca4](https://github.com/Dicklesworthstone/frankensearch/commit/fe4fca4d09f5e76328778fc6c27d539c7d8a8622))
- Concurrent schema bootstrap with race-safe upsert ([780b676](https://github.com/Dicklesworthstone/frankensearch/commit/780b676e78314750414c722963f5c7e8fa4a60cf))
- File protector with e2e corruption recovery and atomic operations ([edf2c4a](https://github.com/Dicklesworthstone/frankensearch/commit/edf2c4a82f65d78628d1b27eeb41eecc4e22ee00))
- fsync hardening for all production metadata writes, including sidecar and repair files ([477b713](https://github.com/Dicklesworthstone/frankensearch/commit/477b7137c4f7552f767701a04b012fb69941150a), [391f1fa](https://github.com/Dicklesworthstone/frankensearch/commit/391f1fa8e01766bec72b61295faf537374f1c832), [3a2e61b](https://github.com/Dicklesworthstone/frankensearch/commit/3a2e61b5484a8d761f9ba1da5f69ef433c99cea2))
- StorageDataSource with storage-backed ingest pipeline and IndexBuilder durability/lexical wiring ([bf2ed68](https://github.com/Dicklesworthstone/frankensearch/commit/bf2ed68ac78cf00887281f38aaa89f263b0603f6))

### Ops Dashboard & Telemetry

- Multi-screen TUI dashboard: alerts/SLO, fleet, resources, analytics, timeline ([fc155e3](https://github.com/Dicklesworthstone/frankensearch/commit/fc155e344bcfeda0d805219670b0bd6eb1eedde3), [590f3e0](https://github.com/Dicklesworthstone/frankensearch/commit/590f3e04c1a343a35e5a2116ff4e855338f84954))
- Telemetry ingest pipeline with backpressure, attribution, and lifecycle tracking ([dac02d3](https://github.com/Dicklesworthstone/frankensearch/commit/dac02d3556c9dba1931b48c5e767304ae7e0fbd6))
- Control-plane health alerting and self-monitoring ([814331a](https://github.com/Dicklesworthstone/frankensearch/commit/814331aff12b0752877e44ce08dff6c48825f663))
- Live resource telemetry collection and ops ingestion pipeline ([2f61155](https://github.com/Dicklesworthstone/frankensearch/commit/2f61155e644e95cafced01baf39b9a96a1fae3d5))
- FrankenSQLite-backed telemetry storage ([4e18736](https://github.com/Dicklesworthstone/frankensearch/commit/4e1873664b4640c50232c231a1ab8ae524c7779c))
- Migration to FrankenTUI backend ([85e44e0](https://github.com/Dicklesworthstone/frankensearch/commit/85e44e09589512e18d57a4a4dc942634312f7661))

### Resilience & Safety

- Graceful lock recovery (no panic-on-poison) across all concurrent components ([968a6b8](https://github.com/Dicklesworthstone/frankensearch/commit/968a6b8f5a1036861fc57edfcb8027566065f3a3))
- Fix 14 NaN-blindness and safety bugs across 8 files ([c0eea15](https://github.com/Dicklesworthstone/frankensearch/commit/c0eea158a4e9407fda63949a8bf6636b7da9b7b1))
- Eliminate TOCTOU race in sentinel and PID file acquisition ([28f4ab3](https://github.com/Dicklesworthstone/frankensearch/commit/28f4ab34ef3017f42e5514ff3aa556974aaee08f))
- Block path traversal in model manifests and harden searcher indexing ([b994a1c](https://github.com/Dicklesworthstone/frankensearch/commit/b994a1c6fac194708d04bfb4f2644d503514773b))
- Harden self-update pipeline, add ULID telemetry IDs, improve WAL atomicity ([08569f5](https://github.com/Dicklesworthstone/frankensearch/commit/08569f5d638b3f6b3894ae3793b0ff6dedabf81d))

### Cross-Platform & Packaging

- Linux x86_64 (`musl` static binary)
- Windows x86_64 (zip + standalone exe) with CI build target ([af41cdc](https://github.com/Dicklesworthstone/frankensearch/commit/af41cdc3089350b3006869679936dd0cbe823715))
- curl|bash installer with HTTP proxy forwarding and background daemon service installation (systemd / launchd / schtasks) ([83f3298](https://github.com/Dicklesworthstone/frankensearch/commit/83f32989373b16f3c950dd6220568cad7be4a185))
- MIT License with OpenAI/Anthropic Rider ([9f60b71](https://github.com/Dicklesworthstone/frankensearch/commit/9f60b71daefe58496503d78a0c5303de5f716c70))
- Published to crates.io (core crates v0.1.1) ([a8d64a0](https://github.com/Dicklesworthstone/frankensearch/commit/a8d64a0605b373ea5875b4fa95266c23a3529fec))

### Build System & Quality

- Feature-flag tier system: `default` (hash), `semantic`, `hybrid`, `persistent`, `durable`, `full`, `full-fts5`
- CMA-ES hyperparameter optimizer for fusion pipeline tuning ([979b09c](https://github.com/Dicklesworthstone/frankensearch/commit/979b09c0309c02180d9cad5cb770a1ec9119fc12))
- IR evaluation metrics: nDCG@K, MRR, Recall@K, MAP with bootstrap confidence intervals
- Benchmark baseline matrix and pressure simulation harness ([dff3387](https://github.com/Dicklesworthstone/frankensearch/commit/dff33875558285ee8cf6f06bf223b73fd6618d49))
- Dependency upgrades: tantivy 0.25, fastembed 5.8, ort rc10 ([df53b6e](https://github.com/Dicklesworthstone/frankensearch/commit/df53b6ee43b8d5d63f3a7c769c4ef0cd90f20a61))
- Pin asupersync to v0.2.0 ([3818edb](https://github.com/Dicklesworthstone/frankensearch/commit/3818edbc973d1bde7ffeee13043d2217c1df2b2f))

### Concurrency Model

- Built on `asupersync` and capability context (`Cx`), not Tokio
- Cancellation-aware search phases and timeouts
- Deterministic TUI replay and evidence ledger hooks ([251e618](https://github.com/Dicklesworthstone/frankensearch/commit/251e618c9a0443df866b11fca74ab37203a0a107))

### Architecture (11-Crate Workspace)

| Crate | Responsibility |
|-------|---------------|
| `frankensearch` | Facade crate with top-level public API and re-exports |
| `frankensearch-core` | Shared types, traits, errors, config, query canonicalization/classification, metrics/eval helpers |
| `frankensearch-embed` | Embedding backends and fallback stack (`hash`, `model2vec`, `fastembed`), streaming model downloads |
| `frankensearch-index` | FSVI vector storage, SIMD dot products, top-k search, WAL, optional ANN (HNSW) |
| `frankensearch-lexical` | Tantivy schema/index/search for BM25 lexical retrieval |
| `frankensearch-fusion` | RRF fusion, two-tier orchestration, blending, adaptive fusion, PRF, MMR, circuit-breaker, federated search |
| `frankensearch-rerank` | Cross-encoder reranking integration |
| `frankensearch-storage` | FrankenSQLite metadata persistence, dedup/content-hash tracking, embedding queue |
| `frankensearch-durability` | Repair/protection primitives for index artifacts and segment health |
| `frankensearch-tui` | Shared TUI shell, input, theme, replay framework |
| `frankensearch-ops` | Fleet observability/control-plane TUI and telemetry materialization |

---

## Pre-Release History

> 2026-02-13 -- Project scaffolding and planning phase before the first tagged release.

- Initial commit: project scaffolding, README, AGENTS.md, and beads task graph ([4215679](https://github.com/Dicklesworthstone/frankensearch/commit/42156796c494a74d4dfad83301d25bec04058c61))
- Document `asupersync` as mandatory async runtime, purge Tokio references ([3f29794](https://github.com/Dicklesworthstone/frankensearch/commit/3f2979450fb9c924cbcea07ecf76a8279b183f5a))
- Initialize Rust workspace with six-crate hybrid search architecture ([f39c793](https://github.com/Dicklesworthstone/frankensearch/commit/f39c793a94e1e687cae0db104c6f1c5f6df02f89))
- Implement full search pipeline across embed, index, lexical, fusion, and rerank crates ([3965991](https://github.com/Dicklesworthstone/frankensearch/commit/39659917d5a3a55be73a68cbc39aa30eba3b39c1))
- Add storage, durability, fsfs, tui, and ops crates to workspace ([484f9cc](https://github.com/Dicklesworthstone/frankensearch/commit/484f9cc9e266f02348b44ff3c09557838eb06be0))
- Expand workspace config, facade crate, and README with feature-flag tier system ([efe8ca6](https://github.com/Dicklesworthstone/frankensearch/commit/efe8ca6d4f92d6da2346e9a58a479ea1af5ca22f))
