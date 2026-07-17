# WIZARD_IDEAS_GMI.md

## Top 5 Additive/Corrective Ideas for Quill Lexical Engine

### 1. Snapshot-Epoch 2D BM25 Memoization Table
**What it is:** Argus replaces inner-loop BM25 floating-point arithmetic with a 64KB L1-resident 2D array lookup `table[freq][fieldnorm]`.
**Why it wins:** BM25's `tf_part` depends on `avgdl`, which is invariant per snapshot. When Keeper publishes a snapshot, it computes a `256 x 256` array of `f32` (handling term frequencies 1-255, with a scalar fallback for `> 255`). Quiver's exhaustive and BMW top-k kernels become pure integer array lookups + f32 sums, completely eliminating division and multiplication from the hot loop. This directly attacks QG-6 (Query latency) by stripping CPU overhead from the tightest loop in the engine.
**Implementation sketch:** Define `struct Bm25Table([[f32; 256]; 256])`. Build it during `Arc<Snapshot>` creation. Pass a reference into the Argus collectors.
**Risk/Cost:** ~64KB memory per snapshot per field (negligible). Trivial compute cost at snapshot publish time.
**Self-assessed confidence:** 99%. An algorithmic slam dunk that exploits the existing quantization table.

### 2. Page-Fault Pipelining via `memmap2::Advice::WillNeed`
**What it is:** Overlap I/O and CPU in Argus by issuing asynchronous OS prefetch hints for Quiver posting blocks immediately after dictionary lookups.
**Why it wins:** FSLX relies on mmap. Cold-cache queries stall the CPU on page faults during posting list iteration. By scanning Grimoire (term dictionary) first, Argus gets the exact byte offsets in the `POSTINGS` and `BLOCKMAX` sections. Hinting the OS immediately allows I/O to fetch pages while Argus sets up MaxScore/BMW heaps and parses the query tree, crushing p99 tail latency.
**Implementation sketch:** After the Grimoire probe, call `.advise(Advice::WillNeed, offset, len)` on the `Mmap` slice for the matched term extents before yielding to the collector setup.
**Risk/Cost:** Minimal code footprint. The OS can choose to ignore the hint under extreme memory pressure, resulting in baseline performance.
**Self-assessed confidence:** 95%. Directly mitigates the primary vulnerability of mmap-based read paths.

### 3. Quarantine-and-Degrade Open Protocol
**What it is:** If Keeper's `open()` encounters a corrupted segment (e.g., failed `xxh3` checksum) that the `.fec` FileProtector cannot repair, quarantine the segment and gracefully open the index with the remaining segments instead of returning `SearchError::IndexCorrupted`.
**Why it wins:** Strict crash-only open (§11.4) is architecturally sound, but holding the entire corpus hostage for one bad mini-segment violates frankensearch's "graceful degradation" principle. A quarantine mode drops the bad segment from the snapshot and emits a `Quarantined` telemetry warning. Missing documents will naturally backfill via watch-mode incremental updates, keeping the searcher 99% operational.
**Implementation sketch:** `Keeper::recover()` catches section-level checksum errors, moves the `seg-*.fslx` to a `.quarantine` extension, skips it in snapshot construction, and logs a high-priority `tracing::warn!`.
**Risk/Cost:** Users get partial results without a hard failure. Must ensure ops telemetry prominently flags the quarantine state so administrators know repair/re-indexing is needed.
**Self-assessed confidence:** 90%.

### 4. Shadow-Traffic "Dark Launch" Gate (G2.5)
**What it is:** A transitional rollout phase where `fsfs` indexes into both engines, serves queries from Tantivy, but asynchronously evaluates the same query against Quill and logs divergences.
**Why it wins:** The plan relies on static fixtures and generated queries (G2). Real monorepos and chaotic user typing produce states synthetic tests inevitably miss. Dark launching validates Argus and Scribe on the user's actual hardware and corpus *before* the G3 flip commit, de-risking the "leapfrog" claim against zero-day correctness regressions in production.
**Implementation sketch:** `TwoTierSearcher` spawns a background `asupersync` task to query Quill. A canonicalizing differ runs and writes to `frankensearch-ops` SQLite if results violate the `ScoreEpsilon` class.
**Risk/Cost:** Temporarily doubles lexical indexing cost and adds async search overhead. Mitigated by restricting to an opt-in config flag or a time-boxed internal rollout window.
**Self-assessed confidence:** 100% on value for de-risking, 85% on system overhead tolerance.

### 5. Zero-Copy SWAR Snippet Windowing
**What it is:** Implement snippet generation (§9.5) using SWAR (SIMD Within A Register) `memmem` to locate term byte-slices directly inside the UTF-8 `STOREDMETA` mmap, bypassing `String` allocation and char-boundary decoding until the final window is selected.
**Why it wins:** Snippet generation is a hidden latency killer in Phase 1 (`Initial`). If Argus spends time allocating and iterating chars for every top-k document, it risks missing the <15ms target. Operating purely on byte-slices with `wide` or scalar SWAR early-exit keeps the phase 1 envelope strictly I/O and scoring bound.
**Implementation sketch:** Use `[u8]::windows` or a `wide` vectorized byte search to find term offsets. Calculate greedy window coverage based on byte distances, then do a single UTF-8 boundary snap and `CompactString` allocation for the winning window.
**Risk/Cost:** Must ensure byte-slice matching doesn't incorrectly match inside multi-byte characters (e.g., ASCII term bytes inside a CJK character). Bounded by validating UTF-8 boundaries only on the handful of candidate hits.
**Self-assessed confidence:** 90%.

### Appendix: 25 Additional Ideas Considered
1. **Elias-Fano for IDMAP section:** Use Elias-Fano encoding for docid mapping to get dense O(1) rank/select, accelerating ID materialization.
2. **Bloom filters for hapax legomena:** Add small Bloom filters for terms appearing exactly once to skip Grimoire binary search entirely on misses.
3. **Asymmetric P-core/E-core rayon weights:** Assign dense Quiver blocks to P-cores and sparse blocks to E-cores during Argus query fan-out.
4. **Delta segment write-stall circuit breaker:** Return `SearchError::Backpressure` if the delta segment doubles its budget while waiting for the Sealer, preventing OOM.
5. **Dictionary Prefix-Block Interpolation Search:** Use interpolation search instead of binary search in the two-level index since term bytes are uniformly distributed.
6. **JIT compiled boolean iterators:** Compile the cursor evaluation loop at runtime for complex boolean + phrase queries to avoid nested trait/virtual call overhead.
7. **Mmap lifetime tied zero-copy `DocId` strings:** Hydrate `DocId` as a `&'a str` tied to the snapshot lifetime to eliminate String allocations in ID returns.
8. **TUI live segment/tombstone visualization:** Surface live segments and delta segment size graphically in the `frankensearch-ops` TUI.
9. **Background `io_uring` for sealing:** Use `io_uring` for segment sealing to avoid blocking background worker threads with `fsync`.
10. **Query result caching via FNV-1a:** Cache top-k results for frequent queries in watch-mode, invalidating only when matching docs are appended.
11. **In-register decoding for fast-path queries:** Special-case 1-term queries to decode postings and score entirely in registers without writing to the MaxScore heap until full.
12. **"Explain" tree output for BM25 tuning:** Augment `QueryExplanation` to return a nested tree of tf/idf weights for specific documents to aid debugging.
13. **Strict memory budget for fusion candidates:** Hard-limit the size of `ScoredResult` allocations during `search_fusion_candidates` to prevent OOM on pathologically broad queries.
14. **Pure boolean "Exact Match" syntax flag:** Allow queries to explicitly bypass BM25 (score=1.0) for ultra-fast filtering passes prior to semantic search.
15. **Fuzzing FSLX reader with cargo-afl:** Use LibFuzzer on the FSLX open/read path to ensure corrupted bytes never induce panics or UB.
16. **Cross-language oracle validation:** Dump indexes into a minimal Java Lucene harness to verify BM25 scores against the industry standard, not just Tantivy.
17. **Deterministic network/disk partition simulation:** Use LabRuntime to simulate slow/stalled disk I/O on the sealing thread to verify backpressure behavior.
18. **Adaptive `quality_weight` blending:** Dynamically lower the semantic blend weight if the lexical BM25 exact match confidence is exceptionally high.
19. **Dynamic batch size based on Scribe flush latency:** Adjust the document lease size per shard dynamically based on how fast the Sealer is completing I/O.
20. **Read-only lockless fallback for crashed writers:** Ensure a new reader can open the index using the manifest if the write lock is orphaned by a crashed process.
21. **Dictionary common-prefix byte compression:** Compress the block-first-terms array itself using front-coding to fit entirely in L1 cache.
22. **SIMD-accelerated CJK bigram generation:** Vectorize the generation of overlapping bigrams in `CjkBigramDecompose` using byte-shifts.
23. **Parallelized `.fec` sidecar generation:** Generate the RaptorQ sidecars on a background rayon thread since they are compute-intensive.
24. **Tiered storage directory structure:** Allow configuring separate directories for Delta/Recent segments (NVMe) vs Cold sealed segments (HDD).
25. **Automatic deep-compaction scheduling:** Schedule the expensive tombstone renumbering compaction pass automatically during idle CPU hours detected by ops telemetry.
