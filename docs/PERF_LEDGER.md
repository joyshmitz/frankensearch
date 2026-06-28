# PERF_LEDGER.md — frankensearch measured wins

> Head-to-head measured performance wins **kept** in the tree. Each row cites the
> exact bench workload, the before/after timings, and the ratio (new/old; lower is
> faster). Dead-ends and regressions live in `docs/NEGATIVE_EVIDENCE.md`.

Build/bench protocol (per-crate ONLY, never workspace-wide):
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p <crate> --release
```

Current Cargo rejects `--release` for `cargo bench` in this checkout; successful
Criterion runs use the equivalent `--profile release` form and record that protocol
mismatch in `docs/NEGATIVE_EVIDENCE.md`.

The dominance baseline is a Tantivy/Lucene/Meilisearch-class original comparator
on identical corpora and query streams. The `bd-ui41` comparator harness now
lives in `frankensearch/benches/search_bench.rs`; its current result is negative
and recorded in `docs/NEGATIVE_EVIDENCE.md`. Rows below remain frankensearch
pre-change baselines or before/after local hot-path ratios unless explicitly
marked as original-comparator wins.

## Original-comparator wins

### 2026-06-25 — BOLD-VERIFY non-semantic zero-hit lexical gate (BlackThrush)

**Lever:** the BOLD hash-hybrid path and `TwoTierSearcher` now stop before hash-vector search
when a non-semantic fast embedder has no quality tier and lexical search produces zero candidates.
The same gate also permits lexical-only return for saturated natural-language rows in that
non-semantic/no-quality mode; that reduced the old catastrophic vector-scan residual but is still
not a universal Tantivy-class win, so the slower natural-language rows are recorded in
`docs/NEGATIVE_EVIDENCE.md`.

**Measured command (RCH worker `hz2`, per-crate, warm target dir):**
```bash
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
  rch exec -- env \
  FRANKENSEARCH_BOLD_VERIFY_EMIT=1 \
  RUST_LOG=error \
  cargo bench -p frankensearch --features lexical --profile release \
  --bench search_bench bold_verify_tantivy_class \
  -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Artifact: `/data/projects/.rch-targets/frankensearch-cod-b/criterion/bold_verify/summary.md`
and `summary.jsonl`.

| Workload | Corpus hash | Tantivy-class p50 | frankensearch p50 | Ratio vs Tantivy-class | Status |
|----------|-------------|-------------------|-------------------|------------------------|--------|
| `bold_verify/top10/100000` `zero_hit` | `13f1b0153f5adec9` | 67 us | 62 us | **0.925** | KEEP |

**Scope:** this is a zero-hit incumbent win for the non-semantic hash/no-quality lane, not a
claim that hybrid search dominates Tantivy/Lucene/Meilisearch-class BM25 overall. The 10k zero-hit
row is a p50 tie (43 us vs 44 us; ratio 0.977) with better tails, and natural-language rows remain
slower at p50; both are recorded as residual evidence.

### 2026-06-25 — BOLD-VERIFY lexical-saturated short-circuit (BlackThrush)

**Lever:** after successful fast embedding and lexical search, `TwoTierSearcher` now returns
lexical-only initial results for `Identifier` / `ShortKeyword` queries when lexical already has
at least `k` hits and graph ranking is disabled. That skips the phase-1 vector scan and RRF work
for lexical-saturated queries while preserving the hybrid path for natural-language and zero-hit
queries.

**Measured command (local fallback after RCH worker `vmi1153651` stalled twice; per-crate,
warm target dir):**
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
FRANKENSEARCH_BOLD_VERIFY_EMIT=1 RUST_LOG=error \
  cargo bench -p frankensearch --features lexical --profile release \
  --bench search_bench bold_verify_tantivy_class \
  -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Artifact: `/data/projects/.rch-targets/frankensearch-cod-a/criterion/bold_verify/summary.md`
and `summary.jsonl`.

| Workload | Corpus hash | Tantivy-class p50 | frankensearch p50 | Ratio vs Tantivy-class | Status |
|----------|-------------|-------------------|-------------------|------------------------|--------|
| `bold_verify/top10/10000` `high_fanout` | `2e78365a46a7c3b9` | 114 us | 81 us | **0.711** | KEEP |
| `bold_verify/top10/100000` `exact_identifier` | `13f1b0153f5adec9` | 1.241 ms | 1.090 ms | **0.878** | KEEP |

**Scope:** this is a targeted lexical-saturation win vs a Tantivy/Lucene/Meilisearch-class
incumbent, not a universal dominance claim. The same BOLD run still shows slower/noisy rows for
10k exact identifiers, short-keyword rows, `limit_all`, natural-language, and zero-hit queries;
those ratios are recorded in `docs/NEGATIVE_EVIDENCE.md`.

### 2026-06-25 — BOLD-VERIFY lexical prefetch budget gate (BlackThrush)

**Lever:** the BOLD hash-hybrid harness now mirrors the product's lexical short-circuit budget:
for query classes that can legally return lexical-only (`Identifier`, `ShortKeyword`,
`NaturalLanguage`), it asks Tantivy for only `k` lexical candidates instead of prefetching `3k`
and then throwing the surplus away. This is an evidence-harness correction for the shipped
short-circuit path, not a new product ranking algorithm.

**Measured command (RCH local fallback; no admissible workers:
`insufficient_slots=5,hard_preflight=1`; per-crate, warm target dir):**
```bash
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
  rch exec -- env \
  FRANKENSEARCH_BOLD_VERIFY_EMIT=1 \
  FRANKENSEARCH_BOLD_VERIFY_COMMAND='CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b rch exec -- env FRANKENSEARCH_BOLD_VERIFY_EMIT=1 RUST_LOG=error cargo bench -p frankensearch --features lexical --profile release --bench search_bench bold_verify_tantivy_class -- --sample-size 10 --warm-up-time 1 --measurement-time 1' \
  RUST_LOG=error \
  cargo bench -p frankensearch --features lexical --profile release \
  --bench search_bench bold_verify_tantivy_class \
  -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Artifact: `/data/projects/.rch-targets/frankensearch-cod-b/criterion/bold_verify/summary.md`
and `summary.jsonl`.

| Workload | Corpus hash | Tantivy-class p50 | frankensearch p50 | Ratio vs Tantivy-class | Status |
|----------|-------------|-------------------|-------------------|------------------------|--------|
| `bold_verify/top10/10000` `exact_identifier` | `2e78365a46a7c3b9` | 205 us | 178 us | **0.868** | KEEP |
| `bold_verify/top10/10000` `natural_language` | `2e78365a46a7c3b9` | 335 us | 322 us | **0.961** | KEEP |
| `bold_verify/top10/100000` `exact_identifier` | `13f1b0153f5adec9` | 1.596 ms | 1.540 ms | **0.965** | KEEP, p95 noisy |
| `bold_verify/top10/100000` `short_keyword` | `13f1b0153f5adec9` | 359 us | 297 us | **0.827** | KEEP |
| `bold_verify/top10/100000` `natural_language` | `13f1b0153f5adec9` | 972 us | 935 us | **0.962** | KEEP |
| `bold_verify/top10/100000` `high_fanout` | `13f1b0153f5adec9` | 895 us | 783 us | **0.875** | KEEP |
| `bold_verify/top10/100000` `zero_hit` | `13f1b0153f5adec9` | 36 us | 24 us | **0.667** | KEEP, tail regressed |

**Scope:** this fixes the BOLD evidence path for lexical-saturated and natural-language
short-circuit rows. It does not establish universal dominance over Tantivy/Lucene/Meilisearch-class
BM25; the 10k short-keyword/high-fanout/zero-hit rows, 100k quoted phrase, and `limit_all` remain
negative or noisy and are recorded in `docs/NEGATIVE_EVIDENCE.md`.

| Date | Crate | Lever | Workload (bench id) | Before | After | Ratio | Status |
|------|-------|-------|---------------------|--------|-------|-------|--------|
| 2026-06-25 | frankensearch-core | **ASCII fast-path for NFC canonicalization** (analyzer hot path) | `nfc/ascii_short` | 1.207 µs | 26.7 ns | **0.022 (~45×)** | KEEP (`9d7e8d0`) |
| 2026-06-25 | frankensearch-core | **ASCII fast-path for NFC canonicalization** (analyzer hot path) | `nfc/ascii_doc` (2.2 KB) | 56.35 µs | 153 ns | **0.0027 (~368×)** | KEEP (`9d7e8d0`) |
| 2026-06-25 | frankensearch-core | ASCII fast-path for NFC canonicalization | `nfc/non_ascii` | 40.86 µs | 41.32 µs | 1.011 (neutral) | KEEP |
| 2026-06-25 | frankensearch-core | **`filter_low_signal`: ASCII compare instead of full-doc `to_lowercase`** | `filter_low_signal/ascii_doc` (2.25 KB) | 137.8 ns | 9.2 ns | **0.067 (~15×)** | KEEP |
| 2026-06-25 | frankensearch-core | `filter_low_signal`: ASCII compare instead of full-doc `to_lowercase` | `filter_low_signal/ascii_short` | 16.7 ns | 8.2 ns | **0.49 (~2×)** | KEEP |
| 2026-06-25 | frankensearch-core | **`strip_markdown_line`: skip the 4-pass inline replace chain for lines with no inline-markdown chars** | `strip_markdown_inline` (80 plain lines) | 10.17 µs | 4.41 µs | **0.434 (~2.3×)** | KEEP |
| 2026-06-25 | frankensearch-core | `strip_markdown_line`: chain `#`/`>` prefix trims as one `&str` (1 alloc, not 2) | `prefix_trim` (80 lines) | 2.45 µs | 1.34 µs | **0.549 (~1.8×)** | KEEP |
| 2026-06-25 | frankensearch-core | **canonicalize tail: drop the 2nd whole-document copy** (`filter_low_signal`→`is_low_signal` bool predicate; pass the owned `ws_normalized` buffer straight to `truncate_to_chars`) | `pipeline_tail` (2.25 KB ascii_doc) | 91.4 ns | 59.4 ns | **0.649 (~1.54×)** | KEEP (BlackThrush) |
| 2026-06-25 | frankensearch-core | **`strip_markdown_line` fast path: trim the borrowed line directly** (drop the `line.to_string()` copy; only `strip_list_marker` allocates) | `strip_markdown_fastpath` (80 plain lines) | 2.328 µs | 1.306 µs | **0.561 (~1.78×)** | KEEP (BlackThrush) |
| 2026-06-25 | frankensearch-core | **`strip_markdown_line` → `Cow<str>`: zero-alloc plain lines** (`strip_list_marker` returns borrowed slices; plain line borrows straight to the caller's single `push_str`) | `strip_markdown_cow` (80 plain lines, full push_str loop) | 2.166 µs | 1.390 µs | **0.642 (~1.56×)** | KEEP (BlackThrush) |
| 2026-06-25 | frankensearch-core | **`QueryClass::classify`: compute `has_whitespace` once + `rsplit_once`** (was up to 4× whitespace rescans + a `rsplitn().collect()` Vec, per query) | `query_class` (11-query mix) | 732.4 ns | 512.4 ns | **0.700 (~1.43×)** | KEEP (BlackThrush) |
| 2026-06-25 | frankensearch-core | **`strip_italic_underscores`: single pass, 3 allocs → 1** (drop the `Vec<char>` + `Vec<bool>` + final `collect`; build output directly with `prev`/`peek`) | `strip_italic_underscores` (40 snake_case code lines) | 9.770 µs | 2.982 µs | **0.305 (~3.28×)** | KEEP (BlackThrush) |
| 2026-06-25 | frankensearch-core | **`strip_markdown_line` slow path: guard each transform by its trigger char** (one scan sets `*`/`_`/`` ` ``/`[` flags; a snake_case line skips the `**`/`*`/`` ` ``/link no-op allocating passes) | `strip_markdown_slowpath` (snake_case line) | 307.9 ns | 48.3 ns | **0.157 (~6.37×)** | KEEP (BlackThrush) |
| 2026-06-25 | frankensearch-core | **`nfc_normalize` → `Cow<str>`: ASCII borrows instead of `to_owned`** (eliminates the whole-document copy at the `canonicalize` entry; ASCII is already NFC and the next stage only needs `&str`) | `nfc_ascii_copy` (2.25 KB ascii_doc) | 39.45 ns | 1.17 ns | **0.030 (~33×)** | KEEP (BlackThrush) |
| 2026-06-25 | frankensearch-core | **`apply_hash_votes` branchless SimHash vote** (`2*b-1` instead of a per-bit `if`/`else` on random hash bits → no ~50% branch mispredict) | `simhash_votes` (~300-token doc) | 18.480 µs | 16.073 µs | **0.870 (~1.15×)** | superseded by table |
| 2026-06-26 | frankensearch-core | **`apply_hash_votes` table-driven SimHash vote** (8 byte-indexed lookups into a compile-time `[[i32;8];256]` + vectorizable 8-wide slice adds, vs 64 per-bit `shift/mask/mul`) | `simhash_votes` (~300-token doc) | 15.810 µs (branchless) | 10.930 µs | **0.691 (~1.45×) vs branchless, ~0.60× vs original branch** | KEEP (BlackThrush) |
| 2026-06-26 | frankensearch-core | **`ParsedQuery::parse` no-negation fast path** (queries without `-`/`"`/`\` skip the `Vec<char>` + char-by-char parse; whitespace-normalize via split + `push_str`) | `parsed_query` (plain multi-word query) | 503.4 ns | 109.6 ns | **0.218 (~4.59×)** | KEEP (BlackThrush) |
| 2026-06-27 | frankensearch-core | **`ParsedQuery::parse` NOT-keyword check: direct ASCII case match instead of allocating a 3-char `String`** on every tested token boundary in the negation-capable parser path. This is the graveyard/data-structure lever in miniature: compile a fixed keyword recognizer to branch checks instead of constructing transient heap data. Behavior is unchanged because the prior `eq_ignore_ascii_case("NOT")` was ASCII-only and only the three-byte keyword is accepted; surrounding whitespace/boundary checks are unchanged. | `parsed_query/not_phrase` (two `NOT "phrase"` exclusions + one `-term`) | 746.73 ns | 668.39 ns | **0.895 (~1.12×)** | KEEP (BlackThrush) |
| 2026-06-25 | frankensearch-embed | **hash embedder: drop 2 per-embed allocs** (lazy `tokenize` iterator + `l2_normalize_in_place` on the owned accumulator) | `hash_embed_fnv` (~100-word doc, dim384) | 2.318 µs | 1.961 µs | **0.846 (~1.18×)** | KEEP (BlackThrush) |
| 2026-06-25 | frankensearch-embed | hash embedder alloc elision — JL path (compute-bound, alloc negligible) | `hash_embed_jl` (~100-word doc, dim384) | 100.27 µs | 102.07 µs | 1.018 (neutral) | KEEP (no regression) |
| 2026-06-27 | frankensearch-embed | **JL projection: interleave 4 independent xorshift token chains** — the JL path was still latency-bound after allocation elision because each token's xorshift recurrence serializes three shift/xor steps per dimension. Buffering four token-chain seeds and advancing them together exposes instruction-level parallelism while preserving the exact scalar output: each dimension accumulator is an exact small integer-valued `f32` sum of +/-1 signs before normalization. Verified by new bit-identical scalar-reference tests across dimensions, seeds, and token-count tails. The requested literal `cargo bench --release` form was tried first and Cargo rejected it; the successful final per-crate run used `--profile release` through `rch exec` with `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a` (local fallback because no RCH worker was admissible). | `hash_embed_jl` (~100-word doc, dim384) | 105.00 µs | 88.945 µs | **0.847 (~1.18×)** | KEEP (BlackThrush) |
| 2026-06-25 | frankensearch-fusion | **RRF fuse: one `entry` lookup instead of `get`+`entry`** (halve per-candidate hashing of the `AHashMap<&str,_>` accumulator) | `rrf_fuse` (1000 lexical + 1000 semantic, ~50% overlap) | 29.11 µs | 23.07 µs | **0.793 (~1.26×)** | KEEP (BlackThrush) |
| 2026-06-24 | frankensearch-index | `f32_bytes` fixed-array decode + 4 accumulators | `dot/dim256/f32_bytes` | 10.839 ms | 3.647 ms | **0.336** | KEEP |
| 2026-06-24 | frankensearch-index | `f32_bytes` fixed-array decode + 4 accumulators | `dot/dim384/f32_bytes` | 14.084 ms | 5.333 ms | **0.379** | KEEP |
| 2026-06-24 | frankensearch-index | `f32_bytes` fixed-array decode + 4 accumulators (`BlueGull` pinned-worker confirmation) | `dot/dim256/f32_bytes/10000` | 3.4835 ms | 0.66126 ms | **0.190** | KEEP (`vmi1149989`) |
| 2026-06-24 | frankensearch-index | `f32_bytes` fixed-array decode + 4 accumulators (`BlueGull` pinned-worker confirmation) | `dot/dim384/f32_bytes/10000` | 5.1487 ms | 1.8811 ms | **0.365** | KEEP (`vmi1149989`) |
| 2026-06-26 | frankensearch-index | **`dot_product_f32_f32`: 4 independent `f32x8` accumulators** over 32-element blocks (plain f32 slice path; cheap decode, real callers in WAL-entry scoring and MRL truncated search). Existing tests allow tiny f32-order differences; SIMD gate passed. BlackThrush landed after remote same-process A/B superseded the older local-fallback negative row. | `dot/dim256/f32_slice/10000` (remote `vmi1227854`, in-process vs single-acc bench baseline) | 2.3594 ms | 750.64 µs | **0.318 (~3.14×)** | KEEP |
| 2026-06-26 | frankensearch-index | `dot_product_f32_f32` 4 accumulators (MiniLM/quality dim) | `dot/dim384/f32_slice/10000` (remote `vmi1227854`) | 3.7372 ms | 2.2784 ms | **0.610 (~1.64×)** | KEEP |
| 2026-06-24 | frankensearch-index | **branchless SIMD f16→f32 widen** (default path) | `dot/dim256/f16_bytes` | 4.733 ms | 1.632 ms | **0.345** | KEEP |
| 2026-06-24 | frankensearch-index | **branchless SIMD f16→f32 widen** (default path) | `dot/dim384/f16_bytes` | 7.363 ms | 2.332 ms | **0.317** | KEEP |
| 2026-06-24 | frankensearch-index | branchless SIMD f16→f32 widen | `dot/dim256/f16_slice` | 3.699 ms | 1.348 ms | **0.364** | KEEP |
| 2026-06-24 | frankensearch-index | branchless SIMD f16→f32 widen | `dot/dim384/f16_slice` | 5.536 ms | 2.181 ms | **0.394** | KEEP |
| 2026-06-26 | frankensearch-index | **in-memory filtered scan: precomputed-hash prescreen** (`matches_doc_id_hash` with a lazy `doc_id_hashes` slab) instead of re-hashing each `doc_id` string per vector via `matches()` — matches the FSVI scan, which already did this | `filter_prescreen` (10k `BitsetFilter` checks) | 183.5 µs | 88.4 µs | **0.482 (~2.08×)** | KEEP (BlackThrush) |
| 2026-06-26 | frankensearch-index | **`dot_i8_i8`: 4 independent `i32x8` accumulators** (int8 two-pass pass-1) — the i8→i16 decode is a cheap sign-extend, so the kernel is sum-chain-bound (unlike the decode-bound f16 dot, where 4 accs regress — see NEGATIVE_EVIDENCE). Integer sum is associative → bit-identical. Landed as `957d608`+`28aa022`; ledger row added by BlackThrush | `dot_dim384/i8_dot` (10k, in-process vs single-acc baseline) | 481.3 µs | 412.0 µs | **0.856 (~1.17×)** | KEEP |
| 2026-06-26 | frankensearch-index | `dot_i8_i8` 4 accumulators (smaller dim) | `dot_dim256/i8_dot` (10k) | 318.2 µs | 298.8 µs | **0.939 (~1.06×)** | KEEP |
| 2026-06-26 | frankensearch-index | **`search_top_k_int8_two_pass_filtered`** — extended the int8 ADC two-pass to accept a `SearchFilter` (pass-1 pre-screens by the precomputed `doc_id` hash). The fast int8 path was previously filter-incapable, so filtered large-N vector search fell back to the exact scan; now it gets the int8 speedup. Bit-identical to the exact filtered top-k (`int8_two_pass_filtered_matches_exact_filtered`) | `int8_two_pass` filtered (10k clustered, `BitsetFilter` ~½ corpus, mult=5): `flat_filtered` vs `int8_filtered_mult5` | 314.4 µs | 248.7 µs | **0.791 (~1.26×)** | KEEP (BlackThrush) |
| 2026-06-26 | frankensearch-index | **int8 two-pass pass-1 cutoff fast-path** — the int8 pass-1 called `insert_candidate` for *every* vector; the exact `scan_range` skips it for sub-cutoff scores (`heap.len() < cap \|\| score_key(score) >= cutoff`). Added the same guard to pass-1 (track the bounded-heap min as cutoff). Result is bit-identical (a sub-cutoff score never enters the full heap; `int8_two_pass_matches_exact_topk` + filtered test pass). The selection overhead was large — without the cutoff, mult=10 (2278 µs) was **slower than the exact flat scan** (~2131 µs); the cutoff restores the int8 win. In-process A/B (100k clustered, cutoff vs no-cutoff baseline) | `int8_two_pass` mult=5 / mult=10 | 1170.4 / 2278.6 µs | 900.9 / 1140.2 µs | **0.770 (~1.30×) / 0.500 (~2.00×)** | KEEP (BlackThrush) |
| 2026-06-26 | frankensearch-index | **MRL truncated-scan cutoff fast-path** — same fix applied to `mrl_truncated_scan`: it called `insert_mrl_candidate` for every vector; added the bounded-heap cutoff guard (`heap.len() < limit \|\| nan_safe(score) >= cutoff`) the exact scan + int8 pass-1 use. Bit-identical (the 46 `mrl::` tests pass unchanged). Smaller win than the int8 pass-1 (the MRL scan is single-threaded with a cheap 64-dim truncated dot, so `insert` is a smaller fraction). In-process A/B (100k FSVI F16, clustered, search_dims=64, limit=30): cutoff vs no-cutoff baseline | `mrl_cutoff` scan | 23056.9 µs | 21045.9 µs | **0.913 (~1.10×)** | KEEP (BlackThrush) |
| 2026-06-26 | frankensearch-index | **MRL truncated-scan parallelization** — `mrl_truncated_scan` was **single-threaded** (a plain `for index in 0..record_count()` byte-walk), while the exact `scan_range` + int8 pass-1 are rayon-parallel. Refactored the byte-walk into a per-chunk `mrl_scan_chunk` (fn-pointer dot collapses the F16/F32 arms) and scan disjoint record ranges in parallel above `PARALLEL_THRESHOLD`, merging per-chunk bounded heaps (`merge_mrl_partials`). Order-independent (score+index total order), so **bit-identical** to the sequential top-k — verified by a new `mrl_search_parallel_path_covers_all_records` test (12k records, rescore-all == exact `search_top_k`) + all 47 `mrl::` tests. In-process A/B (100k FSVI F16, clustered, search_dims=64, limit=30): parallel vs forced-sequential | `mrl_parallel` scan | 5828.5 µs | 674.9 µs | **0.116 (~8.64×)** | KEEP (BlackThrush) |
| 2026-06-26 | frankensearch-fusion | **`SyncTwoTierSearcher` fast-tier fetch → int8 two-pass (default path)** — the hybrid's vector candidate fetch (`search_fast_hits`) called the exact f16 `search_top_k`. The fast tier is a *reranked candidate generator* (quality-tier + RRF re-score its hits), so its contract is approximate candidates; routed the **default path** (no explicit `SearchParams`) to `search_top_k_int8_two_pass_filtered` (parallel + cutoff + filter, mult=10, recall=1.0). **Wires the validated int8 two-pass into a real production hybrid path** (it was previously unused by callers). Lossless → identical fused top-k (6 sync_searcher + 9 integration tests pass); explicit `SearchParams` still uses the exact scan. End-to-end hybrid A/B (100k in-memory two-tier, no lexical): exact-fetch vs int8-fetch | `sync_int8_fetch` | 6449.6 µs | 5498.6 µs | **0.853 (~1.17×)** | KEEP (BlackThrush) |
| 2026-06-26 | frankensearch-index | **`InMemoryTwoTierIndex` quality rerank: O(hits·N) → O(hits)** — `quality_scores_for_hits` (the two-tier quality rerank, on the wired sync-hybrid path) did `quality.doc_ids.iter().position(\|id\| id == hit.doc_id)` — a **linear O(N) doc-id scan per hit** ⇒ O(hits·N) (≈3M string-compares at N=100k, hits=30). Replaced with a lazy `doc_id → index` `HashMap` (O(1) lookup, first-insert-wins = same as `position`; built on first rerank, search-only callers pay nothing). Also fixed the same O(N) scan in `InMemoryVectorIndex`'s by-doc-id hit scorer. Bit-identical (19 in-memory + 15 sync_searcher tests). End-to-end hybrid before→after (same `sync_int8_fetch` bench, only the rerank changed; both arms drop by a constant ~3.7 ms, isolating the rerank): int8 5498.6→1755.0 µs **3.13×**, exact 6449.6→2637.5 µs **2.45×** | `sync_int8_fetch` | 5498.6 µs | 1755.0 µs | **0.319 (~3.13×)** | KEEP (BlackThrush) |
| 2026-06-26 | frankensearch-fusion | **int8 fast-tier `mult` 10 → 5** — the wired `INT8_FAST_TIER_MULT` was 10, the *slowest* int8 mult (int8_two_pass bench: mult=5 862 µs vs mult=10 1187 µs at 100k, both recall=1.0). Since `fetch` is already a candidate over-fetch (`k·candidate_multiplier`), a 10× int8 multiplier is wasteful; mult=5 keeps the candidate set lossless (recall=1.0 validated at 100k) with a higher pass-1 cutoff (fewer inserts) + smaller pass-2 rescore. Lossless (15 sync_searcher tests pass). Cumulative sync hybrid vs the original exact + linear-rerank: **4.38×** | `sync_int8_fetch` int8 arm | 1755.0 µs | 1472.9 µs | **0.839 (~1.19×)** | KEEP (BlackThrush) |
| 2026-06-27 | frankensearch-index | **FSVI int8 two-pass for standalone large-N vector search** — new `VectorIndex::search_top_k_int8_two_pass`: a parallel int8 pass-1 over all main records (a lazily-built corpus-quantized i8 slab from the contiguous F16 region; tombstone-aware flag check + the same cutoff fast-path as the exact scan) keeps the top `k·mult` by approximate score, then an exact f16 rescore of just those candidates selects the final top-k. The file-backed twin of the in-memory two-pass; falls back to the exact `search_top_k` for WAL/non-F16 indexes (never silently degraded). **Lossless: recall@10 = 1.0000 at mult=2/3/5/10 on 100k clustered**, and bit-identical to `search_top_k` when the candidate set retains every record — verified by `int8_two_pass_keep_all_matches_exact` + 53 search tests. NOT wired into the BOLD hybrid (that gap is not vector-bound — see docs/NEGATIVE_EVIDENCE.md); targets pure vector-search latency. A/B (100k file-backed FSVI, in-process, same index): exact flat vs int8 mult=5 | `fsvi_int8_two_pass` | 14349.0 µs | 7408.2 µs | **0.516 (~1.94×)** | KEEP (BlackThrush) |
| 2026-06-27 | frankensearch-index | **FSVI 4-bit (16-level) two-pass — fastest lossless vector-search primitive** — new `VectorIndex::search_top_k_4bit_two_pass` + a fused SIMD nibble kernel `dot_packed_4bit` (load 16 packed bytes → `i16x16`, extract signed nibbles via arithmetic `(x<<12)>>12`/`(x<<8)>>12`, low·low + high·high). Packs each vector to signed 4-bit nibbles (`dim/2` bytes/vector — **half the int8 slab**), parallel pass-1 + exact f16 rescore of the top `k·mult`. **Lossless: recall@10 = 1.0000 at mult=5 on 100k clustered** (0.96 at mult=2); kernel verified against a scalar reference (`dot_packed_4bit_matches_scalar`) + bit-identical to `search_top_k` under keep-all (`four_bit_two_pass_keep_all_matches_exact`). The nibble-unpack compute offsets some of the 2× bandwidth saving, so the net vs int8 is modest, but 4-bit@mult5 is the **fastest lossless** option (mult=10 is slower → mult=5 is the sweet spot). Falls back to exact for WAL/non-F16; not wired into BOLD (not vector-bound). A/B (100k file-backed FSVI, in-process, same index; flat 2132.4 µs, int8 mult=5 888.2 µs): exact flat vs 4-bit mult=5 → **2.56×**; vs int8 mult=5 → **1.07×** | `fsvi_4bit_two_pass` | 888.2 µs | 831.4 µs | **0.936 (1.07× vs int8; 2.56× vs flat)** | KEEP (BlackThrush) |
| 2026-06-27 | frankensearch-index | **`dot_packed_4bit` kernel: vertical accumulation (1.07× → 1.36× vs int8)** — the 4-bit pass-1 is **compute-bound** (the parallel scan is not bandwidth-limited), and the kernel did a horizontal `prod.reduce_add()` **per 16-byte chunk** (12 reduces/vector at dim=384) — the dominant cost. Replaced with a single `i16x16` vertical accumulator (`acc += s_low·q_low + s_high·q_high`), flushed to the i32 sum every 16 chunks before any lane can exceed `i16` (16·98=1568/lane; 16-lane reduce 25088 < 32767). **Bit-identical** (scalar-match + keep-all tests pass; recall@10 = 1.0000 at mult=5 unchanged) — a pure speedup. A/B (100k file-backed FSVI, in-process, same run): 4-bit mult=5 vs int8 mult=5 | `fsvi_4bit_two_pass` | 1034.5 µs | 762.3 µs | **0.737 (1.36× vs int8; 3.22× vs flat)** | KEEP (BlackThrush) |
| 2026-06-27 | frankensearch-index | **In-memory 4-bit two-pass (in-memory twin of the FSVI 4-bit)** — `InMemoryVectorIndex::search_top_k_4bit_two_pass`: packed signed-4-bit slab (`dim/2` bytes/vector, half the int8 slab) + the fused `dot_packed_4bit` pass-1 (parallel, cutoff fast-path) + exact f16 rescore — the in-memory backend twin of the FSVI 4-bit, reusing the verified vertical-accumulation kernel. **Lossless: recall@10 = 1.0000 at mult=5** (0.98 at mult=2), 10k clustered; bit-identical under keep-all (`four_bit_two_pass_keep_all_matches_exact`). A/B (10k in-memory, in-process, same run; flat 352.9 µs, int8 mult=5 160.2 µs): 4-bit mult=5 vs int8 mult=5 → **1.40×**; vs flat → **3.09×** — the fastest lossless in-memory vector-search primitive. (Wiring it into the `SyncTwoTierSearcher` fast tier, which currently uses the int8 two-pass, is the follow-up: needs a `_filtered` variant.) | `int8_two_pass` 4-bit arm | 160.2 µs | 114.2 µs | **0.713 (1.40× vs int8; 3.09× vs flat)** | KEEP (BlackThrush) |
| 2026-06-27 | frankensearch-fusion | **`SyncTwoTierSearcher` fast tier wired int8 → 4-bit two-pass (deployment)** — `search_fast_hits` default path now routes to `search_top_k_4bit_two_pass_filtered` (added a `_filtered` 4-bit variant with the same doc_id-hash pre-screen as int8) instead of the int8 two-pass — deploying the fastest lossless primitive into the sync production hybrid. The fast tier is a reranked candidate generator, so the lossless 4-bit candidate set yields **identical fused top-k**: 6 sync_searcher lib + 9 integration tests pass unchanged. End-to-end same-run A/B (100k in-memory two-tier, no lexical): exact-fetch hybrid vs 4-bit-fetch hybrid → **2.36×** (up from the int8 wiring's 1.79× vs exact; the fast-tier fetch itself is 1.40× faster than int8, measured). Explicit `SearchParams` still uses the exact scan. | `sync_int8_fetch` | 2880.2 µs | 1219.8 µs | **0.424 (2.36× vs exact)** | KEEP (BlackThrush) |
| 2026-06-27 | frankensearch-fusion | **`SyncTwoTierSearcher` 4-bit fast-tier multiplier 5 → 3** — the default sync fast tier already over-fetches candidates before quality rerank/RRF, so the extra 4-bit pass-2 rescore from `mult=5` was wasteful. Tightened the candidate multiplier to 3 and added a deterministic clustered conformance test that compares the default 4-bit path against explicit exact scan top-k IDs for 12 queries. Remote same-worker proof on `ovh-a`, target dir `/data/projects/.rch-targets/frankensearch-cod-b`: baseline source restored to `FAST_TIER_MULT=5`, then candidate `FAST_TIER_MULT=3`, same command `cargo bench -p frankensearch-fusion --profile release --bench sync_int8_fetch -- --sample-size 10 --warm-up-time 1 --measurement-time 2`. Exact arm stayed neutral (2.3088 ms → 2.3120 ms). | `sync_int8_fetch/fast_fetch_4bit` | 769.95 µs | 639.75 µs | **0.831 (~1.20×)** | KEEP (BlackThrush) |
| 2026-06-27 | frankensearch-index | **Prepared 4-bit query lanes for repeated scan dots** — `prepare_4bit_query` decodes the query nibbles once into signed SIMD lane vectors, and `dot_4bit_prepared` reuses those lanes for every stored vector. This lands the alien-graveyard/vectorized-execution lever as a pure register-reuse win on the already-shipped 4-bit scan path: old path decoded both stored and query nibbles per dot; new path decodes only stored nibbles inside the hot loop. Behavior preserved by `dot_packed_4bit_matches_scalar` now checking both packed and prepared kernels, including the `0x99` signed-nibble extreme; the prepared loop also truncates to the shorter stored/query chunk stream instead of panicking on public mismatched slices. Remote RCH worker `hz2`, target dir `/data/projects/.rch-targets/frankensearch-cod-b`, command `cargo bench -p frankensearch-index --profile release --bench dot_product fourbit_prepared -- --sample-size 10 --warm-up-time 1 --measurement-time 3`. | `dot/dim256/fourbit_prepared/10000`; `dot/dim384/fourbit_prepared/10000` | 459.40 µs; 567.38 µs | 389.81 µs; 509.84 µs | **0.849 (~1.18×); 0.899 (~1.11×)** | KEEP (Codex) |

**Lever (ParsedQuery no-negation fast path):** `ParsedQuery::parse` runs per search query (the
searcher parses for `-term`/`NOT "phrase"` negations). The committed parser always materialized a
`Vec<char>` and re-collected each word via `chars[a..b].iter().collect()`. Negation syntax requires
one of `-` `"` `\`; with none present (the common query) there are no negations, so the positive
part is just the whitespace-normalized input. The new fast path returns it directly
(`split_whitespace` + `push_str` into one buffer), skipping the char materialization. Byte-identical
(the full parser collects the same whitespace-split words and `join(" ")`s them; 42 parsed_query
tests green). Measured on a plain multi-word query (`parsed_query`): 503.4 ns → 109.6 ns, **0.218
(~4.59×)**. Queries that *do* use negation syntax still take the exact char-based path.

**Lever (ParsedQuery NOT-keyword allocation elision):** after the no-negation fast path, queries
that contain `"` or `-` still use the char parser. Its `matches_not_keyword` helper previously built
a fresh three-character `String` for every token-boundary check, then called
`eq_ignore_ascii_case("NOT")`. Replaced that with direct ASCII `N|n`, `O|o`, `T|t` checks while
leaving the surrounding boundary/whitespace rules unchanged. This preserves behavior exactly for
the accepted keyword (`eq_ignore_ascii_case` is ASCII-only here) and removes transient allocation
from negation-capable queries. Measured with the existing per-crate bench extended to include a
replica of the old helper:

```bash
AGENT_NAME=BlackThrush \
RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
  rch exec -- cargo bench -p frankensearch-core --profile release \
    --bench parsed_query -- --sample-size 10 --warm-up-time 1 --measurement-time 2
```

The requested literal `cargo bench --release` form was tried first and Cargo rejected it with
`unexpected argument '--release'`; the successful command used Cargo's release profile spelling
above. Criterion estimates for `parsed_query/not_phrase`: old mean **746.73 ns**, new mean
**668.39 ns**, ratio **0.895 (~1.12×)**. Original-comparator ratio is N/A because this is an
internal query-parser primitive, recorded in `docs/NEGATIVE_EVIDENCE.md`.

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-core --bench parsed_query
```

**Lever (SimHash table-driven vote):** building on the branchless vote, `apply_hash_votes` now
processes the window hash **one byte at a time**: each byte (0..256) indexes a compile-time
`VOTE_TABLE: [[i32;8];256]` (`2*((b>>k)&1)-1`) for its 8 ±1 votes, added to the matching 8 bit
counters as a slice — the compiler vectorizes the 8-wide `i32` add (SSE2 `i32x4`). This replaces the
64 per-bit `shift/mask/mul/sub` of the branchless form. Bit-identical (byte `j`'s bit `k` is hash bit
`8j+k`; 20 fingerprint tests + 895 core lib tests green). Measured (`simhash_votes`): branchless
15.810 µs → table 10.930 µs, **0.691 (~1.45×)**, i.e. ~0.60× vs the original conditional. The table
is built `const` (zero runtime/init cost).

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-core --bench fingerprint
```

**Lever (SimHash branchless vote):** `Fingerprint::compute` builds a 64-bit SimHash over 3-token
shingles (per document, for incremental re-embedding decisions). The inner `apply_hash_votes`
accumulated ±1 into 64 bit counters per window via `if (bit set) { +1 } else { -1 }` — a
data-dependent branch on effectively-random hash bits (~50% misprediction). Replaced with the
branchless `*weight += 2*((hash>>bit)&1) - 1`. Bit-identical (`semantic_simhash` unchanged; 20
fingerprint tests green incl. `identical_text_produces_identical_fingerprint`,
`large_document_computes_without_overflow`). Measured over a ~300-token doc's shingle sweep
(`simhash_votes`): 18.480 µs → 16.073 µs, **0.870 (~1.15×)** — confirming LLVM did *not* already
emit a branchless form (measured rather than assumed). Determinism-safe: same vote, same final
`weight > 0` test.

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-core --bench fingerprint
```

**Lever (nfc_normalize → Cow):** `canonicalize`/`canonicalize_query` begin with NFC normalization.
ASCII text is already NFC, so the prior code returned `text.to_owned()` — a whole-document copy —
which then fed `strip_markdown_and_code` (which allocates its own buffer) and the trim path. Since
those stages only need a `&str`, `nfc_normalize` now returns `Cow<str>`: `Cow::Borrowed(text)` for
ASCII (zero copy), `Cow::Owned` for non-ASCII (the `nfc()` collect, unchanged). Byte-identical (the
`&str` view is identical; 34 canonicalize + 895 core lib tests green incl.
`nfc_normalize_ascii_fast_path_matches_reference`). Measured borrow vs `to_owned` on a 2.25 KB ascii
doc (`nfc_ascii_copy`): 39.45 ns → 1.17 ns, **0.030 (~33×)**. Stacks on the earlier NFC ASCII
fast-path (which had already replaced the `nfc()` state machine with `to_owned`); this removes the
remaining copy on the dominant indexing path's entry.

**Lever (strip_markdown_line slow-path guards):** the inline-markdown slow path previously ran the
**entire** transform chain — `replace("**")`, `replace("__")`, `replace('*')`,
`strip_italic_underscores`, `replace('`')`, `strip_markdown_links` — unconditionally, allocating a
whole-line copy at each step even when the relevant char was absent. A `snake_case` line (trigger =
`_` only) thus paid four no-op allocating passes (`**`, `*`, `` ` ``, links). Now one byte scan sets
`has_star`/`has_underscore`/`has_backtick`/`has_bracket` and each transform is guarded by its
trigger char (same order → byte-identical; skipping an absent-trigger transform is the same as the
no-op copy it produced). 34 canonicalize tests green (bold/italic, links, backticks, nested,
headings). Measured on a snake_case line (`strip_markdown_slowpath`, `strip_italic_underscores`
omitted from the A/B since it runs in both): 307.9 ns → 48.3 ns, **0.157 (~6.37×)**. Stacks with
the single-pass `strip_italic_underscores` lever for the dominant code-search line shape.

**Lever (strip_italic_underscores single pass):** this runs on the inline-markdown path of
`strip_markdown_line`, which is hit by **every line containing `_`** — i.e. every `snake_case`
identifier line, the common case in code search (where almost all underscores are kept). The prior
implementation materialized a `Vec<char>`, a `Vec<bool>` keep-mask, and a final `collect` — three
allocations and three passes. Rewritten as a single pass that builds the output `String` directly,
tracking the previous source char and peeking the next to apply the exact same word-boundary test
(`prev`/`next` is-word). Byte-identical — proven by `strip_italic_underscores_matches_reference`
(snake_case, `_italic_`, mixed, leading/trailing, non-ASCII word chars) + 33 canonicalize tests.
Measured over 40 snake_case code lines (`strip_italic_underscores`): 9.770 µs → 2.982 µs, **0.305
(~3.28×)**.

**Lever (QueryClass::classify):** `classify` runs on every search query (adaptive lexical/semantic
budget + the lexical short-circuit gate both call it). `looks_like_identifier` rescanned the query
with `chars().any(char::is_whitespace)` up to four times and allocated a `Vec` via
`rsplitn(2,'-').collect()` for the issue-ID check. Now whitespace is computed once and all the
no-whitespace heuristics are grouped under it (so whitespace queries skip the whole block after a
single early-stopping scan), and the issue-ID check uses allocation-free `rsplit_once('-')`.
Behaviour-identical (26 query_class tests + 894 core lib tests green; grouping is safe because every
branch returns `true`). Measured over an 11-query identifier/keyword/natural-language mix
(`query_class`, real `classify` as `new` vs the prior logic replicated as `old`): 732.4 ns →
512.4 ns, **0.700 (~1.43×)**.

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-core --bench query_class
```

**Lever (strip_markdown_line → Cow):** building on the fast-path lever above, the strip pipeline
now returns `Cow<str>`: `strip_list_marker` yields borrowed slices in every branch (unordered →
`strip_prefix` slice, ordered → `&trimmed[digit_count+2..]` since the marker is single-byte ASCII,
not-a-marker → the line), and `strip_markdown_line`'s fast path returns that borrowed `Cow`. So a
plain line (no inline markdown — prose, most code comments) flows through `strip_markdown_and_code`
with **zero per-line allocation**: the bytes are copied once by the existing `result.push_str`. The
inline-markdown slow path still returns `Cow::Owned`. Byte-identical (894 core lib tests green incl.
ordered/unordered list markers, `numbers_not_list_markers_preserved`, headings, blockquotes).
Measured over the full 80-plain-line `push_str` loop (`strip_markdown_cow`: `string` = prior
per-line owned String vs `cow` = borrowed): 2.166 µs → 1.390 µs, **0.642 (~1.56×)**, on top of the
fast-path win.

**Lever (strip_markdown_line fast path):** `strip_markdown_line` runs on every line of every
document at index time, and markdown stripping is the dominant `canonicalize` cost. The fast path
(lines with no inline-markdown chars — plain prose, most code-comment text) copied the whole line
with `line.to_string()` before the header/blockquote prefix trims, then `strip_list_marker`
allocated the final owned String — two allocations per line. Since the prefix trims and
`strip_list_marker` only need `&str`, the fast path now trims the borrowed `line` directly and only
`strip_list_marker` allocates (one allocation). Byte-identical (same trims in the same order; 33
canonicalize tests green incl. headings/blockquotes/list-markers/links/bold-italic). Measured
head-to-head in one process over 80 plain lines (`strip_markdown_fastpath`, `strip_list_marker`'s
final owned-String step modeled): 2.328 µs → 1.306 µs, **0.561 (~1.78×)**.

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-core --bench canonicalize
```

**Lever (RRF fuse single lookup):** `rrf_fuse_with_graph` accumulates per-doc scores in an
`AHashMap<&str, FusedHitScratch>` across the lexical, semantic, and graph candidate lists. Each
loop did a `hits.get(doc_id)` (per-source dedup probe) **then** a `hits.entry(doc_id)`
(insert/update) — two hashes of the same key per candidate, where the `get` almost never skips
(doc ids are unique within a source). Consolidated to a single `entry` match: `Occupied` keeps the
dedup-skip guard then updates via `get_mut`, `Vacant` inserts — semantically identical (47 rrf
tests + 816 fusion lib tests green), one hash lookup instead of two. Measured head-to-head in one
process (`benches/rrf_fuse.rs`, 1000 lexical + 1000 semantic, ~50% overlap; the private
`FusedHitScratch` is replicated with an equivalently-shaped struct): 29.11 µs → 23.07 µs, **0.793
(~1.26×)**. RRF fusion is a per-query cost on the hybrid path, so this directly trims hybrid search
latency. Frankensearch pre-change before/after ratio only (`bd-ui41`).

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-fusion --bench rrf_fuse
```

**Lever (hash embedder alloc elision):** `HashEmbedder::embed_sync` runs on every document at
index time and every query (the non-semantic `fnv1a-*`/`jl-*` fast tiers, and the frankensearch
side of the BOLD hash-hybrid comparison). The committed path did two dimension-sized allocations
per embed: `tokenize` collected a `Vec<&str>`, and `l2_normalize(&embedding)` returned a freshly
allocated `Vec<f32>`. Now `tokenize` returns a lazy iterator (each embedder consumes tokens exactly
once) and the new `frankensearch_core::traits::l2_normalize_in_place` normalizes the owned
accumulator in place — so the only allocation is the accumulator itself. Bit-identical output
(`l2_normalize_in_place_matches_allocating` proves the in-place form equals `l2_normalize` across
zero/near-zero/non-finite cases; 286 embed lib tests + the existing `output_is_l2_normalized` /
`jl_output_is_l2_normalized` green). Measured head-to-head in one process (old 2-alloc vs new
1-alloc, replicated in `benches/hash_embed.rs`): FNV path **0.846 (~1.18×)**; the JL path is
compute-bound (O(tokens·dim) xorshift), so its alloc savings vanish into noise (1.018, neutral —
never a regression). Frankensearch pre-change before/after ratio only (no dominance-vs-original
claim; `bd-ui41`).

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-embed --bench hash_embed
```

**Lever (canonicalize tail):** `DefaultCanonicalizer::canonicalize` runs on every document at
index time and every query. The committed tail did two whole-document allocations — the old
`filter_low_signal` returned `text.to_string()` on the common (non-ack) path, then
`truncate_to_chars` copied again. Converting the filter to an allocation-free `is_low_signal(&str)
-> bool` predicate (early-return `String::new()` for acks) lets `canonicalize` pass its already-owned
`ws_normalized` buffer straight to truncation — one copy, not two. Output is byte-identical (acks →
`""`, normal docs → same bytes; 893/893 `frankensearch-core` lib tests green, incl. all 33
canonicalize tests). Measured head-to-head in one process (`pipeline_tail/old` = 2 copies vs
`/new` = 1 copy, 2.25 KB ascii doc): 91.4 ns → 59.4 ns, **0.649 (~1.54×)**. Frankensearch
pre-change before/after ratio only (no dominance-vs-original claim; blocked by `bd-ui41`).

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-core --bench canonicalize
```

**Lever (in-memory filtered-scan prescreen):** the FSVI on-disk scan (`search.rs`) already
pre-screens filters with a **precomputed `doc_id_hash`** via `SearchFilter::matches_doc_id_hash`,
but `InMemoryVectorIndex::scan_range` called plain `matches(doc_id)` — which for a `BitsetFilter`
**re-hashes the doc_id string every vector** (`fnv1a_hash` per element). In a selective filtered
scan the filter check is the dominant per-(excluded-)vector cost, so this was a real waste. Added a
lazy `doc_id_hashes: OnceLock<Vec<u64>>` (built on first *filtered* search with the same
`frankensearch_core::filter::fnv1a_hash` that `BitsetFilter` uses — so it's bit-correct;
exact/unfiltered callers pay nothing) and switched the scan to `matches_doc_id_hash(precomputed)`
with a fallback to `matches()` for filters that can't decide by hash (e.g. metadata filters).
Behaviour-identical (17 in-memory + 15 filter tests + new `search_with_bitset_filter_uses_precomputed_hash_path`
+ 357 index lib serial all green). Measured `matches` re-hash vs `matches_doc_id_hash` precomputed
over 10k `BitsetFilter` checks (`benches/filter_prescreen.rs`): 183.5 µs → 88.4 µs, **0.482
(~2.08×)** — the per-vector filter-check cost the in-memory filtered scan now saves, bringing it to
parity with the FSVI scan.

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-core --bench filter_prescreen
```

**Lever (bd-gfzk):** `dot_product_f16_bytes_f32` / `dot_product_f16_f32` — the **default
quantization** path (f16 FSVI, `search.rs:288,346,441`). The f16 paths were decode-bound: the
per-element scalar `f16::to_f32()` dominated (confirmed in `docs/NEGATIVE_EVIDENCE.md` — that's
why accumulator unrolling failed here). Replaced the 8× scalar decode with a **branchless SIMD
widen** (Giesen magic-multiply over `wide::u32x8`/`f32x8` + pure-integer inf/nan fixup) →
**~2.5–3.2× faster** on the dominant path. The widen is **bit-exact** to `f16::to_f32()` for all
65 536 f16 values (exhaustive test `simd_f16_widen_is_bit_exact`), and the single accumulator /
lane assignment is unchanged, so every dot score is **bit-identical** on finite data → no
determinism/golden risk (all 19 simd + full index lib tests green). Measured head-to-head in one
process (`f16_*_new` SIMD vs `f16_*_old` scalar). No dominance-vs-original claim (blocked by
`bd-ui41`); pre-change before/after ratio only.

_Refinement (same lever):_ the byte path also replaces the 8 scalar `from_le_bytes` + stack
round-trip with a single little-endian SIMD load (`[u8;16]`→`u16x8`→`u32x8` zero-extend, BE
fallback retained). Still bit-exact (`simd_f16_bytes_load_matches_scalar`). The in-process
`f16_bytes_new/old` ratio improved 0.345→0.317 (dim256) / 0.317→0.305 (dim384), ~4–8% on top of
the SIMD widen. Modest and not cleanly isolated from worker noise, but kept because one SIMD
load strictly dominates 8 scalar loads (worst case neutral, never a regression).

**Lever:** `dot_product_f32_bytes_f32` (used by f32-quantized FSVI indexes, `search.rs:307,372,492`).
The old kernel decoded f32s from an open-ended `&stored_bytes[off..]` slice, which the
compiler could not prove a fixed width for and so did not vectorize the `from_le_bytes`
decode. Rewriting the decode over fixed `[u8; 32]` sub-blocks (+ 4 independent f32x8
accumulators) lets it vectorize → **~2.7–3.0× faster**. Measured head-to-head in one
process via `benches/dot_product.rs` (`f32_bytes_new` vs `f32_bytes_old`), so the ratio is
immune to which rch worker the run landed on. f16 paths were tried with the same restructure
but **regressed** (decode is scalar-bound there) and were reverted — see
`docs/NEGATIVE_EVIDENCE.md`. No dominance-vs-original claim (blocked by `bd-ui41`); this is a
frankensearch pre-change before/after ratio only.

`BlueGull` confirmation command:
```bash
RCH_WORKER=vmi1149989 \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
  rch exec -- cargo bench -p frankensearch-index --profile release --bench dot_product
```
Same worker conformance gate:
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
  rch exec -- cargo test -p frankensearch-index simd --profile release
```
Result: 18 relevant SIMD tests passed; RCH reported `remote vmi1149989`.

## Foundational primitives & lever validations (not yet product wins)

These are measured de-risking results for levers that are not yet wired into the search
path. They are **not** search speedups — they tell us whether a bigger build is worth it.

### int8 dot kernel (`dot_i8_i8`) — validates the `bd-b5wl` ADC pass-1 premise

Added a tested, exported symmetric int8 dot primitive (`crates/frankensearch-index/src/simd.rs`,
SIMD via `wide::i16x8::mul_widen`; exhaustive-ish correctness test incl. the i32 overflow
worst case). Benched head-to-head against the optimized f16 dot **in the same process/run**:

| Workload | `i8_dot` (int8) | `f16_bytes_new` (optimized f16) | int8/f16 ratio |
|----------|-----------------|---------------------------------|----------------|
| `dot/dim256` (n=10k) | 368 µs | 1.113 ms | **0.331** (~3.0×) |
| `dot/dim384` (n=10k) | 505 µs | 1.622 ms | **0.311** (~3.2×) |

**Conclusion:** scoring all N with int8 is ~3× faster than the current exact f16 scan (½ the
bytes + integer `mul_widen` MAC, no f16 decode). This **validates building the int8 ADC two-pass
scan** (`bd-b5wl`): a pass-1 int8 sweep over all N + an exact f16 rescore of only `k·mult`
candidates should net ~2.5–3× on the vector scan. **Caveat:** this is a kernel-throughput ratio,
**not** a search speedup — the product win requires the sidecar + two-pass wiring + a recall@10
gate (it's approximate). The primitive is unused by the search path until that lands.

**Recall gate measured (`int8_two_pass_recall_at_10`):** int8 pass-1 (top `k·mult`) + exact f16
rescore recovers the true f16 top-10 with **recall@10 = 1.0000** at `mult=20` (top-200 of 3000
random L2-normalized vectors, averaged over 25 queries). So at a modest multiplier the two-pass
is **lossless** here. Both ADC gates now pass: ~3× pass-1 speed **and** recall = 1.0. **Caveat:**
random vectors have wide angular spread; real (clustered) embeddings may need a higher `mult` —
the production wiring must re-measure recall on a real corpus and tune `mult`/quant granularity.

**End-to-end search-level ratio (`topk_exact_f16` vs `topk_int8_2pass`, k=10, mult=20, n=10k):**
full top-10 pipeline (score all N + select), measured same-process:

| dim | exact f16 top-k | int8 two-pass top-k | ratio |
|-----|-----------------|---------------------|-------|
| 256 | 1.213 ms | 469 µs | **0.387** (~2.6×) |
| 384 | 1.794 ms | 607 µs | **0.339** (~3.0×) |

So the int8 ADC two-pass is **~2.6–3.0× faster on the actual top-k search workload with recall@10
= 1.0** (mult=20, random vectors; both pipelines pay a conservative N-element sort, so production
bounded-heap selection would only widen the gap). This is the search-level proof, not just a
kernel ratio.

> **CORRECTION (see `docs/NEGATIVE_EVIDENCE.md`):** the `topk_exact_f16` baseline above is a
> *serial full-sort* pipeline. The real product `search_top_k` is **rayon-parallel + bounded-heap
> + cutoff** and is far faster, so the ~2.6–3× "search win" holds only vs a *serial* exact — NOT
> the product's parallel exact. The kernel ~3× (`33fb45b`) and recall@10 = 1.0 stand; the
> integrated-method ratios vs parallel exact are below.

**Wired into the product (`InMemoryVectorIndex::search_top_k_int8_two_pass`):** the in-memory index
precomputes an int8 slab (one corpus-wide max-abs scale, ranking-preserving) at construction and
exposes an opt-in two-pass method — a **parallel bounded-heap** int8 pass-1 (top `limit·mult` per
chunk, `PARALLEL_CHUNK_SIZE` chunking to match the exact path's core use) → exact f16 rescore with
the *same* bounded-heap selection as the exact path. **Bit-identical** to `search_top_k` whenever
pass-1 recall is 1 (`int8_two_pass_matches_exact_topk`: identical doc-ids + scores), exact
fallback if the int8 slab is absent. Existing exact paths untouched; 355/355 index lib tests green.

**Real-method head-to-head (`inmem_topk`, top-10) vs the parallel exact `search_top_k`:**

The candidate budget `limit·mult` is the selection-overhead knob. `int8_two_pass_recall_at_10`
shows recall@10 = **1.0 down to mult=2** for random (well-separated) vectors, so the original
mult=20 was 10× more candidates than needed. Re-benched at **mult=5**:

| Workload | exact `search_top_k` | `int8_two_pass` mult=5 | ratio | (mult=20 ratio) |
|----------|----------------------|------------------------|-------|-----------------|
| dim256/n10k | 328 µs | 223 µs | **0.68** | 0.72–1.09 (worker-var) |
| dim384/n10k | 397 µs | 278 µs | **0.70** | 0.66 |
| dim384/**n100k** | 2.02 ms | 1.34 ms | **0.67** | 0.75 |

At mult=5 the win is a **robust ~1.4–1.5× across 10k AND 100k** (consistent on both fast and slow
workers), no longer the worker-dependent tie seen at mult=20. Smaller candidate budget = less
top-`k·mult` selection + merge overhead, so more of the int8 kernel's 3× survives. Still
bit-identical to exact when recall=1. **Honest scope:** ~1.4–1.5× (not the 2.6–3× serial-baseline
figure); recall=1.0 is for *random* vectors — clustered real embeddings may need a higher mult, so
the production path must re-measure recall on a representative corpus. **Remaining (`bd-t8tv`):**
the mmap FSVI `search.rs` path (on-disk int8 sidecar, where exact also pays page-faults + decode —
likely a larger win); real-corpus recall + mult tuning.

**2026-06-26 — clustered-data validation (answers the "needs higher mult on real data" caveat):**
re-ran on **clustered** vectors (64 centroids + noise — the realistic embedding shape, vs the prior
uniform-random) via the new gated-free bench `benches/int8_two_pass.rs`
(`cargo bench -p frankensearch-index --bench int8_two_pass`). Result at N=10000, dim=384, k=10:

| candidate_mult | recall@10 (clustered) | latency | ratio vs flat |
|----------------|-----------------------|---------|---------------|
| flat exact | 1.000 | 309.7 µs | 1.00× |
| 2 | **1.0000** | — | — |
| 5 | **1.0000** | 166.5 µs | **0.538 (1.86×)** |
| 10 | **1.0000** | 207.2 µs | **0.669 (1.50×)** |
| 20 | **1.0000** | — | — |

So on clustered data the two-pass is **lossless (recall@10 = 1.0000) down to mult=2** — *easier* than
random vectors (clear cluster winners always land in the `k·mult` candidate set, and the exact
rescore fixes the order). It is a **lossless ~1.5–1.86× speedup over the flat parallel-SIMD exact
scan** at this scale. This is the strongest vector lever found and is the **opposite of HNSW** (which
trades recall — see `docs/NEGATIVE_EVIDENCE.md`). **Recommendation:** use `search_top_k_int8_two_pass`
(mult≈10 for safety margin) for in-memory vector search; it is strictly better than flat exact on
realistic data. Not made the unconditional default because recall=1.0 is empirical (exact iff the
true top-k ⊆ the int8 candidate set) — safe as an opt-in / large-N path, with a recall gate for
pathological corpora.

_Scale confirmation — and the win GROWS with N:_ re-ran the same clustered bench at **N=100000**
(dim=384, k=10 — the BOLD / Tantivy-comparison corpus size): recall@10 = **1.0000 at every mult
(2, 5, 10, 20)** (lossless at production scale; int8 quantization error is scale-independent, so a
modest `k·mult` candidate set still always contains the true top-10). Measured **clustered** latency
(single run): flat 2131 µs vs int8 mult=5 **862 µs (2.47×)**, mult=10 1187 µs (1.80×) — i.e. the
relative speedup **grows from ~1.86× at 10k to ~2.47× at 100k**, because the int8 pass-1's halved
bytes + no-f16-decode advantage scales with N (consistent with the `bd-t8tv` "larger win at scale"
hypothesis). (The 10k mult sweep is pass-1-bound and worker-noisy — mult 2–10 all land ~1.3–1.6×
within noise; the clean separation appears at 100k.) So `search_top_k_int8_two_pass` (mult≈10) is a
**lossless 1.86–2.47× win across 10k–100k, strongest at production scale** — the highest-value
vector lever, pending the wiring (opt-in / large-N default with an exactness gate).

_Wiring target (verified 2026-06-26, BlackThrush):_ the validated two-pass is **in-tree but unused
by callers** — `InMemoryTwoTierIndex::search` (`two_tier.rs:350,353`) and the fusion searcher call
the **exact** `search_top_k` / `search_top_k_with_params`. The integration point is there: route
large-N searches to `search_top_k_int8_two_pass(query, k, mult≈10)`. The blocker for a *default*
(rather than opt-in) switch is the exactness gate — a sound per-query int8-slab error bound (the i8
corpus-max-abs slab quantizes each component to `q=round(x/σ)`, `σ=max_abs/127`, so by
Cauchy–Schwarz `|true_dot − int8_dot| ≤ E` with `E ≈ √dim·σ`; if `β + E < θ` where `β` is the
`k·mult`-th int8 score and `θ` the k-th *exact* rescored score, the result is provably exact).
Deriving + verifying that bound (and its int8→f32 scaling) is the correctness-critical,
multi-iteration step that shouldn't be rushed into the default path.

_Dim confirmation:_ also re-ran at **dim=768** (BGE-base / OpenAI-class embeddings), N=10000:
recall@10 = **1.0000 at every mult (2, 5, 10, 20)** — the lossless property holds across the
embedding-dim range (384–768), so the lever applies to high-dim models too. (Latency ratio
expected ~the dim=384 figure since both flat and int8 pass-1 scale ~linearly with dim; the dim=768
latency estimates didn't sync back cleanly this run, but recall — the correctness question — is the
decisive one and is lossless.)

**Exactness-gate note (for the wiring):** `ScalarQuantizer` (the u8 per-dim FSVI quantizer) already
exposes the bound machinery (`max_dim_errors()` = `scale/2`; `epsilon ≤ max_scale·√dims`), but the
in-memory two-pass uses a **separate i8 corpus-max-abs slab** — a sound certified-exact gate needs
that slab's per-query error bound derived + verified (correctness-critical), so the safe wiring is a
dedicated multi-iteration step, not a 60m change.

These rows are routing evidence for future levers, not wins.

Command:
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
  rch exec -- cargo bench -p frankensearch --profile release --bench search_bench -- --sample-size 10
```

Worker: `ovh-a`; RCH reported remote completion in `608.1s`.

| Date | Crate | Workload | Measurement | Original ratio | Status |
|------|-------|----------|-------------|----------------|--------|
| 2026-06-24 | frankensearch | `dot_product_f32/128` | `12.223 ns` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `dot_product_f32/256` | `30.630 ns` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `dot_product_f32/384` | `44.346 ns` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `dot_product_f32/768` | `82.308 ns` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `hash_embedder/short_10w` | `734.48 ns` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `hash_embedder/medium_100w` | `2.3161 us` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `hash_embedder/long_1000w` | `17.225 us` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `vector_search_topk/top10/1000` | `944.07 us` | blocked by `bd-ui41` | noisy baseline |
| 2026-06-24 | frankensearch | `vector_search_topk/top10/5000` | `3.4640 ms` | blocked by `bd-ui41` | noisy baseline |
| 2026-06-24 | frankensearch | `vector_search_topk/top10/10000` | `1.6642 ms` | blocked by `bd-ui41` | noisy baseline |
| 2026-06-24 | frankensearch | `rrf_fusion/fuse/1000+1000` | `80.563 us` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `score_normalization/z_score/10000` | `43.159 us` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `index_io/write/10000` | `13.812 ms` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `index_io/open/10000` | `19.865 us` | blocked by `bd-ui41` | baseline only |

### 2026-06-25 — non-semantic hash tier lexical guard (BlackThrush)

**Lever:** when a lexical backend is present, no quality embedder is configured, and the fast
embedder is one of the shipped non-semantic hash tiers (`fnv1a-*`/`jl-*`), Phase 1 now returns
lexical results directly instead of paying hash embedding, vector scan, and RRF. This is the
correct behavior for Lucene/Tantivy/Meilisearch-class lexical comparisons: a hash vector
contributes no semantic relevance, so fusing it only burns latency. If lexical search fails
non-cancel, the old vector fallback still runs.

**BOLD-VERIFY command:**

```bash
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
  rch exec -- env FRANKENSEARCH_BOLD_VERIFY_EMIT=1 RUST_LOG=error \
    cargo bench -p frankensearch --features lexical --profile release \
    --bench search_bench bold_verify_tantivy_class \
    -- --sample-size 10 --warm-up-time 1 --measurement-time 3
```

Worker: `vmi1152480` (`[RCH] remote vmi1152480 (804.5s)`). Incumbent is
`tantivy_doc_ids`, used here as the Lucene/Tantivy/Meilisearch-class lexical proxy.

**Kept evidence:** the guard collapses the previous hash-hybrid penalty by removing useless
semantic work and wins several Tantivy-class rows outright:

| Workload | old hash-hybrid ratio | guarded ratio | guarded p95 ratio | status |
|----------|-----------------------|---------------|-------------------|--------|
| `top10/10000/high_fanout` | 18.180 | **0.978** | **0.756** | beats Tantivy-class |
| `limit_all/10000` | 2.086 | **0.873** | **0.611** | beats Tantivy-class |
| `top10/100000/natural_language` | 5.182 | **0.982** | 1.186 | p50 beats, p95 near |
| `top10/10000/zero_hit` | 54.333 | **1.000** | 3.333 | p50 parity; tail miss |
| `top10/100000/zero_hit` | 136.696 | **1.043** | **0.722** | near p50 parity; p95 win |

**Scope:** this is not a universal BOLD victory over Tantivy-class; remaining misses are recorded
in `docs/NEGATIVE_EVIDENCE.md`. It is kept because every measured row dramatically improves over
the existing hash-hybrid path, several rows beat the lexical incumbent, and the behavior is more
semantically honest for hash-only fast tier searches. Next lever should attack the residual
`ScoredResult` materialization and high-selectivity lexical overhead rather than vector/RRF.

## Local hot-path wins

### 2026-06-27 — count-free two-term plain `search_doc_ids` top-k (BlackThrush)

**Lever:** `frankensearch-lexical::TantivyIndex::search_doc_ids` now uses the top-k-only Tantivy
collector for one or two plain syntax-free terms. The prior single-token gate left the BOLD
short-keyword shape (`rust ownership`) on the counted collector, even though the API only returns
ranked IDs and BM25 scores. Fielded, phrase, wildcard, boosted, path-like, hyphenated, and 3+ term
queries stay on the counted path.

The benchmark harness also avoids `asupersync::test_utils::run_test_with_cx` so the measurement does
not install trace-level logging while Criterion is timing the hot path.

Literal requested command still fails in this checkout because Cargo rejects `bench --release`:

```bash
AGENT_NAME=BlackThrush \
RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
RUST_LOG=off \
  rch exec -- cargo bench --release -p frankensearch-lexical \
    --bench doc_ids_topk short_keyword_bold -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Measured command (RCH local fallback: `no admissible workers: insufficient_slots=3,hard_preflight=1`):

```bash
AGENT_NAME=BlackThrush \
RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
RUST_LOG=error \
  rch exec -- cargo bench -p frankensearch-lexical --profile release \
    --bench doc_ids_topk -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Criterion artifacts:
`/data/projects/.rch-targets/frankensearch-cod-b/criterion/doc_ids_topk/*/new/estimates.json`.

| Workload | Query shape | Counted median | New median | Ratio | Status |
|----------|-------------|----------------|------------|-------|--------|
| `doc_ids_topk/short_keyword_bold` | two plain terms, BOLD `rust ownership` mirror | 171.151 us | 30.900 us | **0.181 (~5.54x)** | KEEP |

**Scope:** this attacks the current BOLD short-keyword lexical collector gap but is still a local
Tantivy-wrapper hot-path win, not a Lucene/Tantivy/Meilisearch original-comparator win. The required
N/A original-comparator ratio and noisy fallback row are recorded in `docs/NEGATIVE_EVIDENCE.md`.

### 2026-06-27 — count-free simple `search_doc_ids` top-k (BlackThrush)

**Lever:** `frankensearch-lexical::TantivyIndex::search_doc_ids` now uses a top-k-only Tantivy
collector for single plain-token queries. The API returns ranked document IDs and BM25 scores, not
total match counts, so the old `Count` collector was discarded work on this narrow path. Queries
with Boolean, phrase, field, wildcard, boost, path, quote, or hyphen syntax keep the counted path.

**Measured command:**

```bash
AGENT_NAME=BlackThrush \
RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
RUST_LOG=off \
  rch exec -- cargo bench -p frankensearch-lexical --profile release \
    --bench doc_ids_topk -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

RCH executed on `hz2` (`[RCH] remote hz2 (382.8s)`). Artifact:
`/data/projects/.rch-targets/frankensearch-cod-b/criterion/doc_ids_topk/high_fanout_{counted,free}/new/estimates.json`.

| Workload | Baseline counted median | New count-free median | Ratio | Status |
|----------|-------------------------|-----------------------|-------|--------|
| `doc_ids_topk/high_fanout` | 499.086 us | 239.233 us | **0.479** | KEEP |

**Scope:** this is a local Tantivy-wrapper hot-path win, not an original-comparator BOLD win.
The required Lucene/Tantivy/Meilisearch-class caveat, N/A original-comparator ratio, and unclaimed
fallback rows are recorded in `docs/NEGATIVE_EVIDENCE.md`.

### 2026-06-27 — single-pass `QueryClass::classify` (fused case scan + `take(4)` word count) (BlackThrush)

**Lever:** `QueryClass::classify` runs on **every** search query (adaptive lexical/semantic budget +
the lexical short-circuit gate) — pure frankensearch per-query overhead Tantivy never pays, so cutting
it directly narrows the end-to-end gap vs a Tantivy/Lucene/Meilisearch-class incumbent. Two
behaviour-identical reductions over the already-grouped baseline:
1. `looks_like_identifier` collected the camel/Pascal case flags in **three** separate Unicode-aware
   `chars()` scans (`any(is_lowercase)`, `any(is_uppercase)`, `skip(1).all(is_lowercase)`); fused into
   **one** char pass (flags are order-independent, so the boolean result is unchanged).
2. The word count only needs the `<= 3` boundary, so `split_whitespace().take(4).count()` stops after
   the 4th word instead of scanning every word of a long natural-language query.

All 26 `frankensearch-core::query_class` tests stay green, including `classify_is_trim_invariant`
(proptest) and every classification case — the change is bit-identical.

**Measured command (per-crate, in-process A/B — host-independent ratio):**

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME AGENT_NAME=BlackThrush \
  rch exec -- cargo bench -p frankensearch-core --profile release \
    --bench query_class -- --measurement-time 5
```

RCH worker `ovh-a`. `current` = the committed grouped baseline (replicated in the bench); `new` = the
shipped `classify`. The new arm name `current` doubled as a stale-binary freshness sentinel (it
appeared in the output).

| Workload (11-query mix) | current (grouped) | new (this change) | Ratio | Status |
|-------------------------|-------------------|-------------------|-------|--------|
| `query_class` | 484.88 ns | 424.50 ns | **0.875 (~1.14×)** | KEEP |

CIs do not overlap (`current [468.2, 505.2] ns`, `new [420.0, 429.9] ns`). The same run shows the
pre-grouping `old` at 805.3 ns (cumulative ~1.9×); the isolated ratio for **this** change is **0.875**.

**Scope:** a local per-query-overhead win on pure frankensearch work (query classification). The
original-comparator (BOLD vs Tantivy/Lucene/Meilisearch) ratio is **N/A** for this isolated function —
it reduces overhead the incumbent does not have, narrowing the end-to-end gap rather than beating a
Tantivy primitive head-to-head.

### 2026-06-27 — FSVI 4-bit pass-1 uses prepared query lanes (BlackThrush)

**Lever:** `VectorIndex::search_top_k_4bit_two_pass` packed the query once, but the file-backed
pass-1 still called `dot_packed_4bit` for every stored vector; that wrapper re-prepares the query
nibble lanes on every dot. The in-memory path already used the prepared-query kernel. This change
moves the FSVI path to the same loop-invariant query decode: pack once, `prepare_4bit_query` once,
then call `dot_4bit_prepared` for each stored vector. This is the `/alien-graveyard` vectorized
execution lever in its smallest form: hoist invariant decode out of a scan morsel while preserving
the exact same candidate heap, tombstone check, and f16 rescore.

**Measured command:** the literal requested form still fails in this checkout:

```bash
AGENT_NAME=BlackThrush \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
  rch exec -- cargo bench --release -p frankensearch-index \
    --bench fsvi_4bit_two_pass -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Cargo rejects `cargo bench --release` with `unexpected argument '--release'`; the valid per-crate
release-profile measurement was:

```bash
AGENT_NAME=BlackThrush \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
  rch exec -- cargo bench -p frankensearch-index --profile release \
    --bench fsvi_4bit_two_pass fourbit_mult5 \
    -- --sample-size 10 --warm-up-time 1 --measurement-time 2
```

RCH had no admissible worker for the bench and fell back locally. To avoid the known stale-binary
pitfall with the shared target dir, the candidate source mtime was bumped and the second run
confirmed a fresh `frankensearch-index` rebuild before timing.

| Workload | Baseline (`origin/main`) | New | Ratio | Recall | Status |
|----------|--------------------------|-----|-------|--------|--------|
| `fsvi_4bit_two_pass/fourbit_mult5` | 3.6991 ms | 1.5660 ms | **0.423 (~2.36x)** | recall@10 = 1.0000 at mult=5 | KEEP |

**Scope:** this is a local file-backed vector-search primitive win against the immediately prior
frankensearch FSVI 4-bit path. The original-comparator ratio vs Lucene/Tantivy/Meilisearch is
**N/A** for this isolated vector pass-1; the comparator caveat is recorded in
`docs/NEGATIVE_EVIDENCE.md`.
### 2026-06-27 — JL hash embedder xorshift ILP (interleave 4 token chains) (Cobaltmoth)

**Lever:** `HashEmbedder::embed_jl` (the `jl-*` fast tier) is the compute-bound embedder —
O(tokens·dim) xorshift64. A single token's chain is **latency-bound**: each step's three
*dependent* shift→xor ops must retire before the next, so one chain cannot saturate the CPU's
shift/ALU ports. (This ledger already noted the JL path is "compute-bound xorshift" with alloc
savings lost in noise; the only prior JL lever — branchless sign — regressed because LLVM already
cmov's it, see `docs/NEGATIVE_EVIDENCE.md`.) Each token seeds an **independent** chain, so
`embed_jl` now advances `JL_LANES = 4` token chains together per dimension (`jl_accumulate_lanes`),
exposing instruction-level parallelism that hides the single-chain latency; the (< 4) tail drains
one chain at a time via the original loop.

**Bit-identical:** the per-dimension accumulator only ever holds an exact integer-valued `f32`
(a sum of ±1 contributions, `|value| ≤ token count ≪ 2^24`), so f32 addition is exact and
reordering the lane contributions cannot change a single output bit. Proven by
`jl_ilp_matches_scalar_reference_bit_identical` (dims 256/384, two seeds) +
`jl_ilp_matches_scalar_across_token_counts` (token counts 0..=9 straddling the lane boundary); all
33 `hash_embedder` tests GREEN.

**Measured command (per-crate, in-process A/B — host-independent ratio):**

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME AGENT_NAME=Cobaltmoth \
  rch exec -- cargo bench -p frankensearch-embed --bench hash_embed -- hash_embed_jl_ilp
```

`scalar` = the committed single-chain inner loop (replicated in the bench); `ilp2`/`ilp4` = the
2-/4-lane interleaved variants. ~100-word doc, dim 384. Medians:

| Workload | scalar | ilp2 | ilp4 | Ratio (ilp4/scalar) | Status |
|----------|--------|------|------|---------------------|--------|
| `hash_embed_jl_ilp` (dim384) | 102.88 µs | 69.81 µs | **50.19 µs** | **0.488 (~2.05×)** | KEEP |

CIs do not overlap (`scalar [100.99, 105.49] µs`, `ilp4 [49.70, 50.89] µs`). Independently
corroborated by a concurrent run on a different worker/target dir (`scalar` 98.0 µs → `ilp4`
51.2 µs, **~1.92×**). `ilp4` ships; `ilp2` (~1.47×) is the runner-up the bench keeps for the curve.

**Scope:** a local hot-path win on the `jl-*` non-semantic fast tier (the slowest embedder).
Original-comparator (BOLD vs Tantivy/Lucene/Meilisearch) ratio is **N/A** — this reduces
frankensearch-internal embedding compute the incumbents do not have; it narrows the end-to-end gap
on jl-tier searches rather than beating a Tantivy primitive head-to-head.

### 2026-06-27 — MMR re-rank: incremental running-max + hoisted norms (Cobaltmoth)

**Lever:** `mmr_rerank` (diversity re-ranking) greedily picks `k` of `n` candidates; its inner
sweep did two redundant things, both pure frankensearch per-rerank overhead:
1. **O(k²·n) → O(k·n) cosine evals.** Each round recomputed `max(sim(i, j) for j in selected)` from
   scratch over the whole growing `selected` set. Now a per-candidate running max
   (`max_sim_to_selected`) is updated once per selection, so every `sim(i, selected_doc)` is computed
   exactly once. `f64::max` is associative ⇒ the running max equals the per-round fold bit-for-bit.
2. **3 reductions → 1 per pair.** `cosine_sim` re-derived both vectors' L2 norms on every pair even
   though an embedding's norm is loop-invariant. Norms are hoisted once per candidate (`root_norms`),
   leaving only the dot reduction per pair (`cosine_sim_pre`). Bit-identical for uniform-dimension
   embeddings (the MMR contract); ragged dims fall back to per-pair `cosine_sim`.

**Bit-identical:** the dot loop and accumulation order are unchanged, `root_norms[i]·root_norms[j]`
reproduces the prior `norm_a.sqrt()·norm_b.sqrt()`, and the running max reproduces the per-round fold
— the selected index list is unchanged. Proven by `incremental_norm_hoist_matches_bruteforce`
(dims 8/64/384 × n 3/16/60 × k 1/5/n vs a brute-force reference) + all 32 `mmr` tests GREEN.

**Measured command (per-crate, in-process A/B — host-independent ratio):**

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME AGENT_NAME=Cobaltmoth \
  rch exec -- cargo bench -p frankensearch-fusion --bench mmr_rerank
```

`old` = per-round full recompute + per-pair norms (replicated); `new` = running-max + hoisted norms.
dim 384. Medians:

| Workload (n cand, k results, dim384) | old | new | Ratio | Status |
|--------------------------------------|-----|-----|-------|--------|
| `mmr_rerank/n100_k20` | 5.387 ms | 581.9 µs | **0.108 (~9.3×)** | KEEP |
| `mmr_rerank/n200_k50` | 65.10 ms | 2.800 ms | **0.043 (~23.3×)** | KEEP |

The gain grows with `k` (the O(k²·n)→O(k·n) factor compounding with the 3→1 reduction hoist).

**Scope:** a local hot-path win on the diversity-rerank (MMR) path — pure frankensearch compute the
Tantivy/Lucene/Meilisearch lexical incumbents never perform. Original-comparator ratio is **N/A**
(recorded in `docs/NEGATIVE_EVIDENCE.md`); it narrows end-to-end latency when diversity rerank is
enabled rather than beating a comparator primitive head-to-head.

### 2026-06-27 — graph-rank PageRank: dense-index power iteration (Cobaltmoth)

**Lever:** `GraphRanker::rank_phase1` (query-biased PageRank, `graph` feature) ran its power
iteration over a `HashMap<GraphDocId=String, f64>` **rebuilt every iteration**, with a `String`
doc_id **clone on every teleport and every edge relaxation** (`next.entry(doc_id.clone())`) — and
because `add_edge` inserts both endpoints, every key those `entry()` calls touched was already
present, so the clones were pure dead work — plus a per-edge weight-finiteness check and `f64::from`
widen redone on every one of the (≤20) iterations. Replaced with a one-time dense index (every node
is an `adjacency()` key, so node set = `0..n`) + CSR edge arrays + two reused `Vec<f64>` buffers
(swapped per iteration). Per iteration this drops a HashMap allocation, all String clones, and a hash
probe per edge → array indexing; the weight filter + widen are hoisted into the one-time CSR build.

**Equivalent ranking:** PageRank converges to the same fixed point within `tolerance`, and the prior
`std::HashMap` accumulation order was already run-to-run non-deterministic, so no exact value was ever
pinned. Verified by `dense_rank_matches_reference_ranking` (40-node random graph, 5 seeds) which
asserts the dense result's doc_id **order matches a deterministic `BTreeMap` reference of the original
algorithm** with scores within 1e-4, plus the existing propagation/empty tests, all GREEN under
`--features graph`.

**Measured command (per-crate, in-process A/B; replicas of old HashMap vs new dense):**

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME AGENT_NAME=Cobaltmoth \
  rch exec -- cargo bench -p frankensearch-fusion --bench graph_rank
```

`old` = HashMap rebuilt per iteration + clone-keyed `entry`; `new` = dense `Vec<f64>` + CSR. 10 seeds,
limit 50, 20 iterations. Medians:

| Workload (n nodes, out-degree) | old | new | Ratio | Status |
|--------------------------------|-----|-----|-------|--------|
| `graph_rank/n500_deg6` | 2161.5 µs | 181.9 µs | **0.084 (~11.9×)** | KEEP |
| `graph_rank/n2000_deg8` | 11.661 ms | 982.7 µs | **0.084 (~11.9×)** | KEEP |

**Scope:** a local hot-path win on the optional `graph`-feature PageRank stage — pure frankensearch
compute the Tantivy/Lucene/Meilisearch lexical incumbents never perform, and off the default build
(feature-gated). Original-comparator ratio is **N/A** (recorded in `docs/NEGATIVE_EVIDENCE.md`); it
makes the graph-rank feature ~12× cheaper when enabled, not a head-to-head comparator win.

### 2026-06-27 — fsfs file-classification byte-sniff: vectorizable u64 histograms (Cobaltmoth)

**Lever:** `SniffFeatures::from_bytes` (`frankensearch-fsfs`) scans a capped content probe of **every
file** during classification (binary/text detection), counting null / non-printable / high-bit bytes.
The loop used per-byte `u32::saturating_add` for each of the three counters — a saturating add is *not*
a plain add, so it **blocks auto-vectorization** of this per-file scan. Switched to branchless `u64`
accumulators (`count += u64::from(cond)`) saturate-cast to `u32` at the end, letting LLVM emit three
SIMD histograms.

**Bit-identical:** a `u64` counter cannot overflow for any real `&[u8]` (len ≤ usize ≤ u64), and the
final `u32::try_from(..).unwrap_or(u32::MAX)` reproduces the exact `u32::MAX` saturation the old
per-byte `saturating_add` produced. Proven by `sniff_features_vectorized_matches_scalar_reference`
(empty / all-null / all-high-bit / printable / mixed / 4099-byte / full 0..=255 vs a scalar
`saturating_add` reference) + all `file_classification` tests GREEN (14 passed).

**Measured command (per-crate, in-process A/B; replicas of old saturating vs new u64):**

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME AGENT_NAME=Cobaltmoth \
  rch exec -- cargo bench -p frankensearch-fsfs --bench sniff_features
```

`old` = per-byte `u32::saturating_add`; `new` = branchless `u64` + saturate-cast. Realistic text-ish
probe (~80% printable ASCII). Medians (gain scales with probe size — vectorization amortizes setup):

| Probe size | old | new | Ratio | Status |
|------------|-----|-----|-------|--------|
| `sniff_features/probe_4096` | 3692 ns | 2669 ns | **0.723 (~1.4×)** | KEEP |
| `sniff_features/probe_16384` | 21114 ns | 10719 ns | **0.508 (~2.0×)** | KEEP |
| `sniff_features/probe_65536` | 100341 ns | 42559 ns | **0.424 (~2.4×)** | KEEP |

**Scope:** a local indexing-throughput win on the per-file classification scan — pure frankensearch
file-ingest compute. Original-comparator ratio vs Lucene/Tantivy/Meilisearch is **N/A** (this is the
file-system scan layer, not a query-time comparator primitive); recorded in `docs/NEGATIVE_EVIDENCE.md`.
(fsfs is buildable again — the earlier fsqlite blocker was a stale lock; see `4c816a7`.)

### 2026-06-27 — fsfs `count_lexical_tokens` ASCII byte fast path (Cobaltmoth)

**Lever:** `count_lexical_tokens` (`frankensearch-fsfs::lexical_pipeline`) runs **per chunk per
document** at index time (`LexicalChunkPolicy::chunk_text` calls it for every chunk) and is
allocation-free — so its cost *is* the `for ch in text.chars()` UTF-8-decode loop + per-char
`is_token_char`. For **ASCII** text (the common case for code/docs) it now takes a byte-loop fast path
(`for &b in text.as_bytes()` + `is_token_byte`), skipping the `chars()` decoder; non-ASCII text falls
back to the Unicode `chars()` path unchanged.

**Bit-identical:** for an ASCII byte `b`, `is_token_byte(b) == is_token_char(b as char)`
(`char::is_alphanumeric` == `u8::is_ascii_alphanumeric` on ASCII; same `_-./:` punctuation set), and the
token-boundary state machine is unchanged. Proven by `count_lexical_tokens_ascii_fastpath_matches_chars`
(ASCII + non-ASCII: café/CJK/emoji → fallback, compared to a pure `chars()` reference) +
`count_lexical_tokens_matches_tokenize_lexical` + all 25 `lexical_pipeline` tests GREEN.

**Measured command (per-crate, in-process A/B; `chars` = decode loop, `bytes` = ASCII fast path):**

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME AGENT_NAME=Cobaltmoth \
  rch exec -- cargo bench -p frankensearch-fsfs --bench lexical_count
```

Realistic ASCII code/doc chunk. Medians:

| Chunk size | chars (old) | bytes (new) | Ratio | Status |
|------------|-------------|-------------|-------|--------|
| `lexical_count/ascii_1024` | 2675 ns | 1054 ns | **0.394 (~2.54×)** | KEEP |
| `lexical_count/ascii_4096` | 6235 ns | 3264 ns | **0.524 (~1.91×)** | KEEP |
| `lexical_count/ascii_16384` | 21386 ns | 11581 ns | **0.542 (~1.85×)** | KEEP |

**Scope:** a local indexing-throughput win on per-chunk token counting (fsfs lexical pipeline) — pure
frankensearch file-ingest compute. Original-comparator ratio vs Lucene/Tantivy/Meilisearch is **N/A**
(file-system scan layer, not a query comparator); caveat in `docs/NEGATIVE_EVIDENCE.md`.

### 2026-06-27 - fsfs code-structure tokenization ASCII byte fast path (BlackThrush)

**Lever:** `code_structure_sidecar::tokenize` extracts lowercase ASCII alphanumeric tokens for the
code-structure sidecar. The old path always used
`value.chars().flat_map(char::to_lowercase)`, which builds a `ToLowercase` iterator for every Unicode
scalar before discarding non-ASCII separators. For ASCII code and symbol text, the new path uses
`value.is_ascii()` plus a byte loop with `to_ascii_lowercase`; non-ASCII input keeps the prior Unicode
path unchanged.

**Bit-identical:** for ASCII input, `char::to_lowercase` yields exactly the ASCII lowercase character,
and the token-boundary rule is still `is_ascii_alphanumeric`. Non-ASCII input falls back to the old
Unicode loop, including cases like the Kelvin sign that lowercase to ASCII. Proven by
`tokenize_ascii_fastpath_matches_unicode_reference` over empty, code-like ASCII, mixed punctuation,
Latin-1/Greek, CJK, and `\u{212A}` inputs.

**Measured command (per-crate, local fallback through `rch exec`; in-process A/B old vs new):**

```bash
AGENT_NAME=BlackThrush \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
  rch exec -- cargo bench -p frankensearch-fsfs --profile release \
    --bench code_tokenize -- --sample-size 10 --warm-up-time 1 --measurement-time 2
```

`old` = Unicode `chars().flat_map(char::to_lowercase)` tokenizer replica; `new` = ASCII byte fast
path. Realistic code/symbol text. Medians:

| Workload | old | new | Ratio | Status |
|----------|-----|-----|-------|--------|
| `code_tokenize/ascii_256` | 5011.625 ns | 4976.335 ns | **0.993** | tie/no-claim |
| `code_tokenize/ascii_1024` | 13.090 us | 8.374 us | **0.640 (~1.56x)** | KEEP |
| `code_tokenize/ascii_4096` | 95.735 us | 52.135 us | **0.545 (~1.84x)** | KEEP |

**Scope:** a local fsfs indexing/sidecar tokenization win for medium and larger code snippets.
Original-comparator ratio vs Lucene/Tantivy/Meilisearch is **N/A** because those incumbents do not
run frankensearch's fsfs code-structure sidecar tokenizer. The 256-byte row is deliberately recorded
as a tie in `docs/NEGATIVE_EVIDENCE.md`; do not claim a small-string win from this lever.

### 2026-06-28 — sync rank-change telemetry: drop redundant VectorHit clones (Cobaltmoth)

**Lever:** `compute_rank_changes_for_scored` runs **unconditionally on every sync hybrid query**
(`SyncTwoTierSearcher`, for `metrics.rank_changes`) — pure frankensearch per-query overhead the
Tantivy/Lucene/Meilisearch incumbents never pay. It built **two throwaway `Vec<VectorHit>`** by
**cloning every `doc_id`** out of the initial + refined `ScoredResult` slices, solely to feed
`build_borrowed_rank_map` — which only reads `doc_id` (rank = enumerate index; it ignores
`VectorHit::index`/`score`). Now the doc_id→rank maps are built **directly** from the `ScoredResult`
slices (`entry(hit.doc_id.as_str()).or_insert(rank)`), dropping 2 `Vec` allocations + 2·N `String`
clones per query.

**Bit-identical:** same first-occurrence-wins `HashMap<&str, usize>` content as `build_borrowed_rank_map`
(the discarded `index`/`score` never affected the rank map), so `compute_rank_changes_with_maps` returns
the identical `RankChanges`. All 6 `sync_searcher` lib tests GREEN.

**Measured command (per-crate, in-process A/B; `clone` = build via `Vec<VectorHit>`, `borrow` = direct):**

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME AGENT_NAME=Cobaltmoth \
  rch exec -- cargo bench -p frankensearch-fusion --bench rank_map
```

N = result count (initial + refined, partial overlap). Medians:

| Workload | clone (old) | borrow (new) | Ratio | Status |
|----------|-------------|--------------|-------|--------|
| `rank_map/n20` | 1543 ns | 778 ns | **0.504 (~1.98×)** | KEEP |
| `rank_map/n60` | 5233 ns | 2316 ns | **0.443 (~2.26×)** | KEEP |
| `rank_map/n200` | 17517 ns | 7835 ns | **0.447 (~2.24×)** | KEEP |

**Scope:** a per-query-overhead win on the default sync hybrid path (rank-change telemetry), narrowing
frankensearch's own per-query cost vs the incumbent. Original-comparator ratio is **N/A** (rank-change
metrics have no comparator counterpart); caveat in `docs/NEGATIVE_EVIDENCE.md`.

### 2026-06-28 — sync vector path: borrow doc_ids in fast/quality score maps (Cobaltmoth)

**Lever:** the no-lexical branch of `SyncTwoTierSearcher` builds two per-query `HashMap`s
(`fast_scores`, `quality_scores`) from the fast + quality candidate hits, then passes them to
`vector_hits_to_scored_results`, which only ever **`.get()`-looks-them-up by `&str`**. The maps were
keyed on **owned `String`** (`hit.doc_id.clone()` per candidate) for no reason. Now they key on
`&str` borrowed straight from `fast_hits`/`quality_hits` (which outlive the call), dropping a
per-candidate `String` clone in both map builds; the dedup `HashSet` in `vector_hits_to_scored_results`
also gets a `with_capacity` hint.

**Bit-identical:** `HashMap<&str, f32>` with the same entries, looked up by the same
`hit.doc_id.as_str()` keys → identical `fast_score`/`quality_score` resolution and identical
`ScoredResult`s. All 6 `sync_searcher` lib tests GREEN.

**Measured command (per-crate, in-process A/B; `clone` = `HashMap<String>`, `borrow` = `HashMap<&str>`):**

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME AGENT_NAME=Cobaltmoth \
  rch exec -- cargo bench -p frankensearch-fusion --bench score_map
```

N = fast + quality candidate count. Medians:

| Workload | clone (old) | borrow (new) | Ratio | Status |
|----------|-------------|--------------|-------|--------|
| `score_map/n30` | 2883 ns | 1215 ns | **0.422 (~2.37×)** | KEEP |
| `score_map/n90` | 9572 ns | 3457 ns | **0.361 (~2.77×)** | KEEP |
| `score_map/n300` | 31859 ns | 11491 ns | **0.361 (~2.77×)** | KEEP |

**Scope:** a per-query-overhead win on the **no-lexical (pure-vector) sync hybrid path** (the lexical
branch uses `rrf_fuse`, unaffected). Pure frankensearch result-assembly compute; original-comparator
ratio is **N/A** (caveat in `docs/NEGATIVE_EVIDENCE.md`). Second clone-elision in the sync
result-assembly vein (after `rank_map`, `f8f645e`).

### 2026-06-28 — sync lexical path: move doc_ids out of the rrf_fuse result (Cobaltmoth)

**Lever:** the lexical (hybrid/BOLD) branch of `SyncTwoTierSearcher` calls
`fused_hits_to_scored_results(&rrf_fuse(...), k)` for both the initial and refined phases.
`fused_hits_to_scored_results` took `&[FusedHit]` and **cloned** each `doc_id` into the `ScoredResult`,
even though the `rrf_fuse` result is a **fresh temporary** dropped immediately after. Changed it to take
`Vec<FusedHit>` by value and `into_iter().map(|hit| ScoredResult { doc_id: hit.doc_id, .. })` —
**moving** each `doc_id` (a pointer copy) instead of cloning. Both call sites now pass the `rrf_fuse(...)`
result by value.

**Bit-identical:** the moved `String` is the exact value that was previously cloned; every other field is
copied as before, so the `ScoredResult`s are unchanged. All 6 `sync_searcher` lib tests GREEN.

**Measured command (per-crate; in-process A/B with `iter_batched` so both arms pay the same input-clone
setup; `clone` = `&[FusedHit]`+clone, `move` = `Vec<FusedHit>`+move):**

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME AGENT_NAME=Cobaltmoth \
  rch exec -- cargo bench -p frankensearch-fusion --bench fused_materialize
```

N = fused candidate count (k = N, materialize all). Medians:

| Workload | clone (old) | move (new) | Ratio | Status |
|----------|-------------|------------|-------|--------|
| `fused_materialize/n20` | 482 ns | 62 ns | **0.129 (~7.76×)** | KEEP |
| `fused_materialize/n60` | 1969 ns | 107 ns | **0.054 (~18.41×)** | KEEP |
| `fused_materialize/n200` | 5970 ns | 277 ns | **0.046 (~21.55×)** | KEEP |

The ratio is large because the move eliminates **both** the per-result `String` clone (N allocs) **and**
the subsequent drop of the temporary's `doc_id`s (N frees); the new path is allocation-free. Absolute
times are small — this materialization is one step of the query — but it is bit-identical and on the
**lexical/BOLD hybrid sync path** (both phases). Original-comparator ratio is **N/A** (frankensearch
result-assembly); third clone-elision in the sync vein (after `rank_map` `f8f645e`, `score_map`
`dc86170`).

### 2026-06-28 — sync vector refined path: move blended doc_ids into final results (BlackThrush)

**Lever:** the no-lexical refined branch of `SyncTwoTierSearcher` receives an owned `blended`
`Vec<VectorHit>` from `blend_two_tier`, then immediately converted it through the generic borrowed
`vector_hits_to_scored_results(&blended, ...)` helper. That paid a `HashSet` first-occurrence dedup and
one `String` clone per final row even though `blend_two_tier` already produces one row per merged doc id
from a doc-id keyed map. The no-lexical refined branch now consumes that owned vector with
`unique_vector_hits_to_scored_results_owned`, moves `hit.doc_id` into `ScoredResult`, and leaves the
generic borrowed helper in place for raw vector-hit inputs that may still need deduplication.

**Bit-identical:** `blend_two_tier` output is unique by construction; the new helper preserves row order
and every score/source/index field, looking up `fast_score`/`quality_score` by the same `&str` keys before
moving the `String`. The lexical branch remains on `rrf_fuse` + `fused_hits_to_scored_results`.

**Measured command (per-crate, release profile; `clone_dedup` = old borrowed converter, `move_unique` =
owned blended converter):**

```bash
AGENT_NAME=BlackThrush \
RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a-vector-materialize \
RUST_LOG=off \
  rch exec -- cargo bench -p frankensearch-fusion --profile release \
    --bench vector_materialize -- --sample-size 10 --warm-up-time 1 --measurement-time 2
```

RCH had no admissible worker and fell back locally; the run completed with exit 0. N = final blended
result count (k = N). Medians:

| Workload | clone+dedup (old) | move+unique (new) | Ratio | Status |
|----------|-------------------|-------------------|-------|--------|
| `vector_materialize/n20` | 2.7685 us | 1.4859 us | **0.537 (~1.86×)** | KEEP |
| `vector_materialize/n60` | 8.3990 us | 6.0815 us | **0.724 (~1.38×)** | KEEP |
| `vector_materialize/n200` | 35.454 us | 14.391 us | **0.406 (~2.46×)** | KEEP |

**Scope:** no-lexical (pure-vector) sync refined result materialization. This is frankensearch's own
result-assembly overhead; original-comparator ratio vs Lucene/Tantivy/Meilisearch is **N/A** (caveat in
`docs/NEGATIVE_EVIDENCE.md`).

### 2026-06-28 — lexical id materialization: numeric u64 fast-field + ordinal table is 2.56–8.47× over docstore (validated lever) (BlackThrush)

**Lever (validated, bench-level; production wiring scoped as the immediate follow-up).** `TantivyIndex::
search_doc_ids` materializes each hit's `doc_id` via `collect_id_hits → load_doc → searcher.doc(addr)`,
which **decompresses the entire stored document** (id + content + title + metadata_json — all `STORED`)
just to read the `id` field. This is the BOLD `limit_all` / large-fetch materialization gap. Two prior
attacks were REJECTED: the **str-FAST-field** id (`docs/NEGATIVE_EVIDENCE.md` 2026-06-27, **2.65–18.3×
slower** — `StrColumn::ord_to_str` does a dictionary SSTable seek per hit) and the **lazy doc_id cache**
(~0-gain at the realistic ~30-candidate fetch). Their shared route-next — a **numeric `u64` fast field
carrying a dense insertion ordinal** (a flat bit-packed column, **no dictionary**) plus an external
append-only `ordinal → doc_id` table (`Vec<String>`, O(1) index) — is now **confirmed a large win**: it
skips BOTH the docstore decompress AND the dictionary seek, reading `ord = u64_column.first(local_doc)`
and cloning `table[ord]` once.

**Measured (per-crate A/B, fresh in-RAM Tantivy, N=20k, `content` `STORED` so the docstore arm pays a
realistic decompress like production; `docstore` = `searcher.doc(addr)` → read id (current path),
`numeric_ff` = u64 fast-field → `table[ord]`; doc_ids asserted bit-identical before timing; 60 samples):**

| Workload | docstore (now) | numeric_ff | Ratio | Speedup |
|----------|----------------|------------|-------|---------|
| `id_materialize/k30`   | 12.340 µs | 4.8263 µs | **0.391** | ~2.56× |
| `id_materialize/k100`  | 45.184 µs | 9.5752 µs | **0.212** | ~4.72× |
| `id_materialize/k300`  | 136.60 µs | 16.505 µs | **0.121** | ~8.28× |
| `id_materialize/k1000` | 431.81 µs | 51.009 µs | **0.118** | ~8.47× |

```bash
AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc RUST_LOG=off \
  rch exec -- cargo bench -p frankensearch-lexical --profile release \
  --bench id_materialize_numeric_ff -- --sample-size 60 --warm-up-time 1 --measurement-time 3
```

Artifact: kept bench `crates/frankensearch-lexical/benches/id_materialize_numeric_ff.rs`. Conformance:
81/81 lexical lib tests GREEN; the bench asserts the two paths return identical `doc_id`s.

**Why this is a real comparator angle (not just internal):** the incumbent Lucene/Tantivy-class proxy
also materializes ids out of the stored document (`tantivy_only_search` ultimately reads ids the same
docstore way); a numeric-ff materialization makes frankensearch's id read **2.56–8.47× cheaper than the
docstore path the comparator pays**, so it directly closes the materialization-dominated `limit_all`
gap. **Wins even at k30** — the regime where the doc_id-cache was ~0-gain — so it helps the BOLD top10
over-fetch too, not only large-limit.

**Production wiring (scoped follow-up, NOT in this commit):** add `ord: u64 FAST` to `build_schema`,
build the append-only `ordinal → doc_id` table during `index_documents`/`upsert` (rebuild on `open` for
persisted indexes), and route `collect_id_hits` through the fast field with a **docstore fallback** for
pre-`ord` indexes (so old on-disk indexes stay correct). Delete/merge safety: ordinals are monotonic and
never reused (a deleted doc's ordinal simply never appears in results; Tantivy carries fast fields
through merges), so the append-only table is delete/merge-correct; it grows by one `String` per
doc-version, rebuilt in O(total docs) on open. That correctness surface needs its own conformance pass,
hence wiring is deferred to keep this commit GREEN and reversible.

### 2026-06-28 — lexical id materialization WIRED into production: `search_doc_ids` up to 6.32× end-to-end (BlackThrush)

**Lever (the scoped follow-up above, now SHIPPED).** Wired the validated numeric-fast-field
materialization into `TantivyIndex`: `build_schema` adds `ord: u64 FAST | STORED`; `assign_ord`
stamps a dense monotonic ordinal on each document under the writer lock and appends its `doc_id` to an
append-only `RwLock<Vec<String>>` ord-table; `collect_id_hits` reads the `ord` column **lazily per
touched segment** and resolves `table[ord]`, with a **per-hit docstore fallback** for any ordinal it
can't resolve (pre-`ord` docs, reopened-but-not-rebuilt tables, poisoned lock). `SchemaFields::ord` is
resolved by name in `from_index`, so pre-`ord` on-disk indexes are `None` and use docstore unchanged.

**Conformance:** 81/81 lexical tests GREEN — incl. `search_doc_ids_matches_counted_baseline`,
`search_doc_ids_returns_ranked_identifiers` (ground-truth ids), upsert/delete, multi-segment
force-merge, and CJK/Korean/Japanese roundtrips. The A/B bench asserts `search_doc_ids` (fast) ==
`search_doc_ids_via_docstore` (forced docstore) at k=1000 before timing.

**Measured (per-crate same-binary A/B, in-memory Tantivy N=100k, high-fanout query "search"; `docstore`
= forced `searcher.doc(addr)` decompress, `fast` = wired ord path; medians):**

| Workload | docstore | fast | Ratio | Status |
|----------|----------|------|-------|--------|
| `search_doc_ids_materialize/k10`   | 611.35 µs | 602.90 µs | **0.986** (tie — no top10 regression) | KEEP |
| `search_doc_ids_materialize/k100`  | 633.21 µs | 614.02 µs | **0.970 (~1.03×)** | KEEP |
| `search_doc_ids_materialize/k1000` | 4407.6 µs | 697.22 µs | **0.158 (~6.32×)** | KEEP |

```bash
AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc RUST_LOG=off \
  rch exec -- cargo bench -p frankensearch-lexical --profile release \
  --bench search_doc_ids_materialize -- --sample-size 40 --warm-up-time 1 --measurement-time 2
```

Note: an initial **eager** variant (open every segment's column upfront) was ~1.2× *slower* at k10
(materialization is a tiny slice of BM25 search there, and top-k hits touch only 1–2 of many segments);
switching to **lazy per-touched-segment** column loading removed that and left k10 a tie. Kept benches:
`search_doc_ids_materialize` (end-to-end) + `id_materialize_numeric_ff` (isolated primitive).

**Scope vs comparator:** the incumbent Lucene/Tantivy-class proxy materializes ids out of the stored
document the same way, so this makes frankensearch's `search_doc_ids` **6.32× cheaper than the docstore
path the comparator pays** on the materialization-dominated large-limit (`limit_all`) row — directly
closing that documented gap — while top10 is an unchanged tie. On-disk **reopened** indexes currently
fall back to docstore until the ord-table is rebuilt on open (correct, just unaccelerated); that rebuild
is the one remaining follow-up. **(DONE — see the 2026-06-28 sidecar entry below.)**

### 2026-06-28 — on-`open` ord-table restore (sidecar): reopened on-disk indexes realize the fast path, 2.29× at k1000 (BlackThrush)

**Lever (the "one remaining follow-up" above, now SHIPPED).** A reopened on-disk `TantivyIndex` started
with an empty `ord_table` and fell back to docstore id-materialization. Now `commit` persists the table
to an `ord_table.json` sidecar (atomic temp-file + rename, best-effort: write errors are logged and the
in-memory fast path is unaffected), and `from_index` loads it on `open`. A stale-short sidecar (e.g. a
persist that failed on the last commit) is repaired with an **O(segments)** guard — `Column::max_value`
(columnar metadata, no per-doc scan) gives the highest existing ordinal, the table is padded to cover
it so the next assigned ordinal can't collide, and `collect_id_hits` treats an empty padded slot as a
miss → docstore fallback (safe even if a real `doc_id` is `""`). Pre-`ord` on-disk indexes have no
sidecar / no `ord` field → unchanged docstore path.

**Conformance:** 82/82 lexical tests GREEN, incl. a new `reopen_on_disk_restores_fast_id_materialization`
(create on-disk → index → commit → drop → `open` → asserts the sidecar exists and the reopened index
returns **byte-identical ranked `doc_id`s**) and the existing `reopen_preserves_documents`. The A/B
bench also asserts reopened fast == docstore ids at k=1000 before timing.

**Measured (per-crate same-binary A/B, on-disk index N=20k built→committed→dropped→reopened, query
"document"; `docstore` = forced `searcher.doc`, `fast` = sidecar-restored ord path; medians):**

| Workload | docstore | fast | Ratio | Status |
|----------|----------|------|-------|--------|
| `reopen_id_materialize/k10`   | 136.96 µs | 130.37 µs | **0.952** (tie — no regression) | KEEP |
| `reopen_id_materialize/k1000` | 741.83 µs | 323.24 µs | **0.436 (~2.29×)** | KEEP |

```bash
AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc RUST_LOG=off \
  cargo bench -p frankensearch-lexical --profile release \
  --bench reopen_id_materialize -- --sample-size 40 --warm-up-time 1 --measurement-time 2
```

**Scope vs comparator:** the docstore baseline is exactly what a Lucene/Tantivy-class incumbent pays to
read ids from the stored document, so a **reopened** frankensearch index now materializes ids ~2.29×
cheaper than that incumbent on the large-fetch row (the in-memory path's 6.32× is higher because that
bench's 100k high-fanout docstore arm is heavier). The fast path is now reached on **both** fresh and
reopened on-disk indexes. Sidecar is re-serialized in full per `commit` (fine for batch-commit; a
delta/binary format is a future optimization for commit-per-doc streaming workloads). Kept bench:
`reopen_id_materialize`.

### 2026-06-28 — async hybrid lexical-metadata re-attach borrows `&Value` (3.55–7.20× on the map step) (BlackThrush)

**Lever.** The async `TwoTierSearcher`'s `fused_hits_to_scored_results` (`searcher.rs`) re-attaches
lexical metadata to the fused results via a `doc_id → metadata` map, but only the `fused` (top-k) hits
read it. The old map was `AHashMap<&str, serde_json::Value>` and **cloned every lexical candidate's
metadata** into it — including the many candidates dropped from the top-k. Switched the map to
`AHashMap<&str, &serde_json::Value>` (borrow) and deferred the `clone` to the per-winner lookup
(`.get().copied().cloned()`), so a deep/large metadata `Value` for a candidate that misses the top-k is
**never cloned**. `lexical_results` outlives the function, so the borrows are valid. Bit-identical
output — the `fused_hits_to_scored_results_preserves_lexical_metadata` guard test and the bench's
`assert_eq` both confirm identical `Option<Value>` per winner. 817/817 fusion lib tests GREEN.

**Measured (per-crate isolated A/B, N=60 lexical candidates → k=10 winners; `clone_all` = old
clone-every-candidate map, `borrow` = borrow + clone-only-winners; medians):**

| Workload | clone_all | borrow | Ratio | Status |
|----------|-----------|--------|-------|--------|
| `lexical_metadata_map/small` (1-field metadata)  | 9.1124 µs | 2.5641 µs | **0.281 (~3.55×)** | KEEP |
| `lexical_metadata_map/large` (nested ~10-field)  | 64.788 µs | 8.9993 µs | **0.139 (~7.20×)** | KEEP |

```bash
AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc RUST_LOG=off \
  cargo bench -p frankensearch-fusion --profile release \
  --bench lexical_metadata_map -- --sample-size 60 --warm-up-time 1 --measurement-time 2
```

**Scope:** this is frankensearch's own async hybrid result-assembly bookkeeping (the metadata
re-attachment is unique to the real `TwoTierSearcher`, not the simplified BOLD comparator harness), so
the original-comparator ratio vs Lucene/Tantivy/Meilisearch is **N/A**. The win scales with metadata
size and candidate-overfetch (`candidates − k` avoided `Value` clones); it is a real per-query reduction
for metadata-bearing workloads on the production async path. Unlike the sync phase-clone elision
(2026-06-28, sub-noise — `ScoredResult` struct copies), this elides heap-allocated `serde_json::Value`
clones, hence the large ratio. Kept bench: `lexical_metadata_map`.

### 2026-06-28 — async MMR rerank reorder moves the pool instead of cloning it twice (25–75× on the reorder step) (BlackThrush)

**Lever.** When MMR diversity reranking is enabled, the async `TwoTierSearcher` reorders the top `pool`
fused results by the `mmr_rerank` permutation. The old code cloned the pool into a `head` vec, then
**cloned each element again** while emitting it in MMR order — `2×pool` full `ScoredResult` clones, each
carrying a `doc_id` `String` + a metadata `Value`. Since `order` is a **distinct permutation of
`0..pool`** (mmr_rerank never repeats a candidate — guarded by `mmr::tests::no_duplicates_in_output`),
the head can be reordered by **moving** each element into its MMR slot: `split_off` the tail, collect the
head into `Vec<Option<ScoredResult>>`, and `Option::take` each index in `order` (the tail is moved
through untouched). **Zero `ScoredResult` clones.** Bit-identical output (only a permutation; payloads
unchanged). 817 fusion lib + 32 `mmr` tests GREEN.

**Measured (per-crate isolated A/B, pool=30 + tail=10; `clone_reorder` = old 2×pool clone,
`move_reorder` = new split_off + `Option::take`; medians):**

| Workload | clone_reorder | move_reorder | Ratio | Status |
|----------|---------------|--------------|-------|--------|
| `mmr_reorder/small` (1-field metadata) | 13.842 µs | 542.62 ns | **0.039 (~25.5×)** | KEEP |
| `mmr_reorder/large` (nested ~10-field)  | 69.682 µs | 925.89 ns | **0.013 (~75.3×)** | KEEP |

```bash
AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc RUST_LOG=off \
  cargo bench -p frankensearch-fusion --profile release \
  --bench mmr_reorder -- --sample-size 60 --warm-up-time 1 --measurement-time 2
```

**Scope:** frankensearch's own async result-assembly; original-comparator ratio vs
Lucene/Tantivy/Meilisearch is **N/A** (MMR diversity rerank has no incumbent counterpart, and it is a
**conditional** path — only runs when `mmr_config.enabled`). When active it saves the full ~14–70 µs
reorder cost per query (scaling with metadata size). Second async-`searcher.rs` clone-elision win in a
row (after `lexical_metadata_map`), confirming the async hybrid path was materially less mined than the
sync searcher. Kept bench: `mmr_reorder`.

### 2026-06-28 — runtime-dispatched AVX2 `dot_i8_i8`: 2.26–2.56× over the portable `wide` kernel (BlackThrush)

**Lever (overturns the 2026-06-25 "AVX2 can't be a code lever" finding).** That finding measured a
global `+avx2` build at ~1.5–2.5× but rejected it as un-landable (a default `+avx2` `SIGILL`s on older
hosts; `wide` only selects AVX2 via *compile-time* features, so it can't dispatch at runtime) and flagged
the only code path — a hand-written runtime-dispatched intrinsic kernel — as "large, risky." **That
route-next is now done.** `dot_i8_i8` (the int8 ADC two-pass **pass-1** kernel — scans every corpus
vector) now runtime-dispatches via `std::is_x86_feature_detected!("avx2")` to a hand-written AVX2 kernel
(`#[target_feature(enable = "avx2")]`, 256-bit `vpmaddwd` over sign-extended i16 lanes, two accumulators,
horizontal sum), falling back to the existing portable `wide` kernel (`dot_i8_i8_generic`) on
non-x86_64 / pre-AVX2 hosts. `#[allow(unsafe_code)]` scoped to the kernel (crate is `deny(unsafe_code)`;
mmap already uses the same opt-in).

**Bit-identical:** integer addition is associative, so the 256-bit reduction equals the scalar/`wide`
sum exactly — new test `simd::tests::avx2_dot_matches_generic` asserts equality across 15 dim shapes
(0/1/7/8/31/32/33/63/64/65/100/256/384/511/512). Conformance: **364/364** `frankensearch-index` lib
tests GREEN run serially (incl. all int8 two-pass + recall gates). *(Note: the WAL/compaction tests are
pre-existing flakes under parallel `cargo test` — shared-tempdir interference, failing set varies
run-to-run, none SIMD-related; all pass with `--test-threads=1`.)*

**Measured (per-crate, AVX2 worker `hz2`; sum of 10 000 i8 dots; `i8_dot` = runtime dispatch → AVX2,
`i8_dot_generic` = portable `wide`-SIMD fallback):**

| Workload | i8_dot_generic (`wide`) | i8_dot (AVX2) | Ratio | Status |
|----------|-------------------------|---------------|-------|--------|
| `dot/dim256/i8` (10k vectors) | 617.17 µs | 240.91 µs | **0.390 (~2.56×)** | KEEP |
| `dot/dim384/i8` (10k vectors) | 766.66 µs | 339.42 µs | **0.443 (~2.26×)** | KEEP |

```bash
AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc RUST_LOG=off \
  rch exec -- cargo bench -p frankensearch-index --profile release \
  --bench dot_product -- "dot/dim(256|384)/i8_dot"
```

**Scope:** the int8 pass-1 vector scan is frankensearch's own vector tier (no Lucene/Tantivy
counterpart), so the original-comparator ratio is **N/A** — but this ~2.3–2.6× directly cuts the int8
two-pass scan cost on every AVX2 host (most modern x86 + the rch workers), with a safe identical-result
fallback elsewhere. First **explicit-SIMD-intrinsic** win in the tree (prior dot wins were portable
`wide` + ILP). Route-next: the same runtime-dispatch pattern applies to the 4-bit pass-1
(`dot_4bit_prepared`, the wired default — harder: nibble unpack) and the f16 pass-2 rescore
(`dot_product_f16_*` via F16C — but f32 accumulation reorders, so NOT bit-identical there). Kept bench
arm: `i8_dot_generic`.

### 2026-06-28 — runtime-dispatched AVX2 `dot_4bit_prepared`: 1.19–1.27× on the WIRED-DEFAULT pass-1 kernel (BlackThrush)

**Lever (the int8 route-next, applied to the default path).** `dot_4bit_prepared` is the **wired default**
sync-hybrid pass-1 kernel (`in_memory.rs` `search_top_k_4bit_two_pass_filtered` → it scans every corpus
vector). Same runtime-AVX2-dispatch pattern as `dot_i8_i8` (`bce9bc8`): a hand-written
`#[target_feature(enable="avx2")]` kernel — load 16 packed bytes, `_mm256_cvtepi8_epi16`, arithmetic-shift
out the low/high nibbles (`slli`/`srai`), 256-bit `_mm256_mullo_epi16` against the prepared query lanes,
accumulate in i16 (flush every 16 chunks before overflow), reduce via `_mm256_madd_epi16` — falling back
to the portable `wide` kernel (`dot_4bit_prepared_generic`) on non-AVX2/non-x86 hosts.

**Bit-identical:** integer/in-range (per-dim products ≤ 64, flush keeps lanes < i16::MAX), so the 256-bit
reduction equals the `wide` sum exactly — new test `simd::tests::avx2_dot4bit_matches_generic` asserts
equality across 12 packed-length shapes (full 16-byte chunks + sub-chunk tails). Conformance: **365/365**
index lib tests GREEN serial. *(WAL/compaction tests flake under parallel `cargo test` — pre-existing,
none SIMD-related; pass with `--test-threads=1`.)*

**Measured (per-crate, AVX2 worker `hz2`; sum of 10 000 4-bit dots; `fourbit_prepared_new` = runtime
dispatch → AVX2, `fourbit_prepared_generic` = portable `wide` fallback):**

| Workload | generic (`wide`) | AVX2 | Ratio | Status |
|----------|------------------|------|-------|--------|
| `dot/dim256/fourbit_prepared` (10k vectors) | 392.30 µs | 329.11 µs | **0.839 (~1.19×)** | KEEP |
| `dot/dim384/fourbit_prepared` (10k vectors) | 519.74 µs | 410.59 µs | **0.790 (~1.27×)** | KEEP |

```bash
AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc RUST_LOG=off \
  rch exec -- cargo bench -p frankensearch-index --profile release \
  --bench dot_product -- "dot/dim(256|384)/fourbit_prepared"
```

**Scope:** original-comparator ratio **N/A** (frankensearch's own vector tier). Smaller than the int8
2.5× — the 4-bit nibble-unpack (`slli`/`srai`/`mullo` per chunk) is more compute-bound, so the 256-bit
width helps less than the memory-bound int8 dot — but it's the **default** path, so every AVX2-host
default vector search now gets a ~1.2× faster pass-1 scan, safe-fallback elsewhere. Both int8 and 4-bit
pass-1 dots are now AVX2-dispatched. Route-next: only the f16 pass-2 rescore (F16C) remains, and that
needs a recall gate (f32 reorder is not bit-identical). Kept bench arm: `fourbit_prepared_generic`.

### 2026-06-28 — runtime-dispatched AVX2+F16C `dot_product_f16_bytes_f32`: 3.88–3.99× (BOLD vector-tier kernel) (BlackThrush)

**Lever (the f16 route-next — and it IS bit-identical after all).** `dot_product_f16_bytes_f32` is the
exact f16 scan kernel — it backs `VectorIndex::search_top_k`, i.e. **the BOLD comparator's vector tier**
(`frankensearch_hash_hybrid_search` → `fixture.vector.search_top_k`), plus the two-pass f16 rescore. The
kernel is **decode-bound**: the portable `wide` path decodes each f16→f32 in software. The new
`#[target_feature(enable="avx2,f16c")]` kernel uses `vcvtph2ps` (`_mm256_cvtph_ps`) to decode 8 f16 in
one instruction, then runtime-dispatches via `is_x86_feature_detected!("avx2") && …("f16c")`, falling
back to the `wide` kernel (`dot_product_f16_bytes_f32_generic`) elsewhere.

**Bit-identical (the prior note feared an f32-reorder; avoided it):** f16→f32 is **exact** (f32 has more
mantissa bits), so F16C yields the same f32 values as `widen8_f16_bytes`; the kernel then keeps the
**same** separate-mul+add accumulation and routes the final reduction **through `wide::f32x8::reduce_add`**
(the `__m256` accumulator is stored to `[f32;8]` → `f32x8`), so the reduction order is byte-for-byte the
generic path. New test `simd::tests::avx2_f16dot_matches_generic` asserts `to_bits` equality across 12
dim shapes (embeddings are finite; NaN payloads — the only legitimate divergence — never occur).
Conformance: **366/366** index lib tests GREEN serial. *(WAL tests flake under parallel `cargo test`;
pass with `--test-threads=1`.)*

**Measured (per-crate, AVX2+F16C worker `ovh-a`; sum of 10 000 f16 dots; `f16_bytes_new` = runtime
dispatch → AVX2+F16C, `f16_bytes_generic` = portable `wide` software-decode fallback):**

| Workload | generic (`wide`) | AVX2+F16C | Ratio | Status |
|----------|------------------|-----------|-------|--------|
| `dot/dim256/f16_bytes` (10k vectors) | 1.0825 ms | 271.48 µs | **0.251 (~3.99×)** | KEEP |
| `dot/dim384/f16_bytes` (10k vectors) | 1.5735 ms | 405.63 µs | **0.258 (~3.88×)** | KEEP |

```bash
AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc RUST_LOG=off \
  rch exec -- cargo bench -p frankensearch-index --profile release \
  --bench dot_product -- "dot/dim(256|384)/f16_bytes"
```

**Scope:** the **largest** of the three SIMD-dispatch wins (int8 2.5×, 4-bit 1.2×, f16 3.9×) because the
f16 dot is decode-bound and F16C is a pure hardware win. Original-comparator ratio vs Lucene/Tantivy is
**N/A** (vector tier has no incumbent counterpart), but this is the **BOLD vector-tier kernel**, so the
non-short-circuit BOLD rows' vector search is now ~3.9× cheaper on AVX2+F16C hosts (directly trims
frankensearch's hybrid-over-Tantivy overhead), safe-fallback elsewhere. All three integer/f16 dot
kernels are now AVX2-dispatched; the f16 fear of "not bit-identical" was wrong (reduce-through-`wide`
fixes it). Kept bench arm: `f16_bytes_generic`.

### 2026-06-28 — runtime-dispatched AVX2+F16C `dot_product_f16_f32` (slice): 3.64–3.81× on the in-memory rescore (BlackThrush)

**Lever (the f16-slice sibling of the bytes win).** `dot_product_f16_f32` (the `&[f16]`-slice kernel) is
the wired `InMemoryVectorIndex` **pass-2 exact rescore + quality scoring** kernel (`in_memory.rs` 468 /
562 / 658 / 741 / 859 — every two-pass query rescores its candidates with it). Same F16C dispatch as the
bytes variant (`7239d58`): `_mm256_cvtph_ps` hardware-decodes 8 f16, same separate-mul+add accumulation,
final reduce **through `wide::f32x8::reduce_add`**, and — matching this kernel's tail — a separate
mul+add scalar tail (the bytes variant used `mul_add`; matched per-kernel). **Bit-identical** (f16→f32
exact; `simd::tests::avx2_f16slicedot_matches_generic` asserts `to_bits` equality across 12 dim shapes).
Conformance: **367/367** index lib tests GREEN serial.

**Measured (per-crate, AVX2+F16C worker `hz2`; sum of 10 000 f16-slice dots):**

| Workload | generic (`wide`) | AVX2+F16C | Ratio | Status |
|----------|------------------|-----------|-------|--------|
| `dot/dim256/f16_slice` (10k vectors) | 1.6404 ms | 430.08 µs | **0.262 (~3.81×)** | KEEP |
| `dot/dim384/f16_slice` (10k vectors) | 2.3499 ms | 645.05 µs | **0.275 (~3.64×)** | KEEP |

```bash
AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc RUST_LOG=off \
  rch exec -- cargo bench -p frankensearch-index --profile release \
  --bench dot_product -- "dot/dim(256|384)/f16_slice"
```

**Scope:** original-comparator ratio **N/A** (frankensearch's own vector tier), but this is the wired
in-memory two-pass pass-2 rescore kernel, so every two-pass query's rescore is now ~3.7× cheaper on
AVX2+F16C hosts, safe-fallback elsewhere. With this, **all four hot dot kernels** (int8 2.5×, 4-bit
1.2×, f16-bytes 3.9×, f16-slice 3.7×) are AVX2-dispatched. The dot-kernel SIMD vein is now essentially
mined; remaining are the lower-traffic exact-f32 paths (`dot_product_f32_bytes_f32` / `dot_packed_4bit`).
Kept bench arm: `f16_slice_generic`.

### 2026-06-28 — runtime-dispatched AVX2 `dot_product_f32_bytes_f32`: 1.37–1.46× (completes the dot SIMD arc) (BlackThrush)

**Lever (the last dot kernel).** `dot_product_f32_bytes_f32` is the exact f32-quantization scan kernel
(`Quantization::F32` indexes + MRL rescore — non-default; F16 is the default). Same runtime-AVX2-dispatch
pattern: a hand-written `#[target_feature(enable="avx2")]` kernel mirroring the `wide` kernel's **4
accumulators**, `(acc0+acc1)+(acc2+acc3)` reduction, 8-chunk tail (reduced **through `wide::f32x8::reduce_add`**),
and `mul_add` scalar tail; falls back to `dot_product_f32_bytes_f32_generic` on non-AVX2/non-x86. f32 LE
bytes are native on x86, so an unaligned `loadu_ps` of the stored bytes gives the same lanes as
`decode8_f32` → **bit-identical** (`simd::tests::avx2_f32dot_matches_generic`, `to_bits` equality across
13 dim shapes). Conformance: **368/368** index lib tests GREEN serial.

**Measured (per-crate, AVX2 worker `hz2`; sum of 10 000 f32 dots; `f32_bytes_new` = runtime dispatch →
AVX2, `f32_bytes_generic` = portable `wide` fallback):**

| Workload | generic (`wide`) | AVX2 | Ratio | Status |
|----------|------------------|------|-------|--------|
| `dot/dim256/f32_bytes` (10k vectors) | 415.74 µs | 285.36 µs | **0.686 (~1.46×)** | KEEP |
| `dot/dim384/f32_bytes` (10k vectors) | 790.82 µs | 575.87 µs | **0.728 (~1.37×)** | KEEP |

```bash
AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc RUST_LOG=off \
  rch exec -- cargo bench -p frankensearch-index --profile release \
  --bench dot_product -- "dot/dim(256|384)/f32_bytes"
```

**Scope:** original-comparator ratio **N/A** (frankensearch's own vector tier). The **smallest** of the
five SIMD-dispatch wins — f32 has no decode win (it's already f32) and the `wide` generic already uses 4
f32x8 accumulators, so the gain is purely 256-bit width over the SSE2-default 2×128. Non-default path
(F32-quant / MRL), so low traffic, but a clean bit-identical win that **completes the dot-kernel SIMD
arc**: int8 (2.5×), 4-bit (1.2×), f16-bytes (3.9×), f16-slice (3.7×), f32-bytes (1.4×) — **all five hot
dot kernels are now runtime-AVX2-dispatched** with a property test + safe fallback each. Remaining:
`dot_product_f32_f32` (the `&[f16]`-less f32 slice, MRL — trivial copy of this) and the integer
`dot_packed_4bit` (non-prepared); both very low traffic. Kept bench arm: `f32_bytes_generic`.

### 2026-06-28 — runtime-dispatched AVX2 `dot_product_f32_f32` (slice): ~1.6× (MRL path) (BlackThrush)

**Lever (the f32-slice sibling of the f32-bytes win — final f32 kernel).** `dot_product_f32_f32` is the
`&[f32]`·`&[f32]` dot behind the MRL (Matryoshka truncated-embedding) rescore — non-default, low traffic.
Same runtime-AVX2 dispatch: a `#[target_feature(enable="avx2")]` kernel mirroring the `wide` kernel's 4
accumulators, `(acc0+acc1)+(acc2+acc3)` reduction (through `wide::f32x8::reduce_add`), 8-chunk tail, and
**separate-mul+add** scalar tail (matched per-kernel — this one is NOT `mul_add`, unlike f32-bytes);
falls back to `dot_product_f32_f32_generic` (the former `dot_product_f32_f32_unchecked`) elsewhere.
**Bit-identical** (`simd::tests::avx2_f32slicedot_matches_generic`, `to_bits` across 13 dim shapes).
Conformance: **369/369** index lib tests GREEN serial.

**Measured (per-crate, AVX2 worker `hz2`; sum of 10 000 f32-slice dots):**

| Workload | generic (`wide`) | AVX2 | Ratio | Status |
|----------|------------------|------|-------|--------|
| `dot/dim256/f32_slice` (10k vectors) | 770.63 µs | 477.82 µs | **0.620 (~1.61×)** | KEEP |

(dim384 same direction, noisier on a contended worker: new ≈ 953 µs.)

**Scope:** original-comparator ratio **N/A**; non-default MRL path, low traffic, modest 256-bit-width-only
win (no f32 decode). Completes the dot-kernel SIMD coverage: **six** kernels now runtime-AVX2-dispatched
(int8, 4-bit-prepared, f16-bytes, f16-slice, f32-bytes, f32-slice), each bit-identical + safe fallback.
The only dot left is the non-prepared integer `dot_packed_4bit` (lowest traffic). Kept bench arm:
`f32_slice_generic`.

### 2026-06-28 — runtime-dispatched AVX2+F16C int8 slab QUANTIZE: 5.2–5.9× (a DIFFERENT primitive) (BlackThrush)

**Lever (off the dot vein — the quantizer, not a dot).** `quantize_i8_slab` (the lazy int8 ADC slab
build) was the next-biggest un-mined **compute** kernel: it decodes the whole f16 slab to f32 **twice**
(once for the corpus max-abs, once to quantize) with software `f16::to_f32`, plus a per-element
`f32::round` + clamp — fully decode-bound and scalar. New `simd::quantize_f16_slab_to_i8` runtime-
dispatches (`is_x86_feature_detected!("avx2") && …("f16c")`) to a hand-written AVX2+F16C kernel:
`vcvtph2ps` decodes 8 f16/instruction; pass-1 is a vector `max` over `|x|`; pass-2 is `×scale` →
round-half-away (emulated `trunc(v + copysign(0.5, v))`, exact for `|v| ≤ 127`) → clamp (`min`/`max`) →
`cvttps_epi32` → `i8`. Falls back to the scalar kernel on non-AVX2+F16C. `in_memory.rs::quantize_i8_slab`
now delegates to it.

**Bit-identical:** `max` is exact/associative (vector max == scalar fold), the round emulation matches
`f32::round` exactly, clamp/cast unchanged — `simd::tests::avx2_quantize_i8_matches_generic` asserts the
`Vec<i8>` is byte-for-byte equal across lengths (incl. sub-8 tails + the zero-vector `max_abs==0` edge).
Conformance: **370/370** index lib tests GREEN serial.

**Measured (per-crate, AVX2+F16C; dim 384 f16 slab → i8; `generic` = scalar, `dispatch` = AVX2+F16C):**

| Workload | generic | dispatch | Ratio | Status |
|----------|---------|----------|-------|--------|
| `quantize_i8_slab/10000` (10k×384) | 25.85 ms | 4.94 ms | **0.191 (~5.2×)** | KEEP |
| `quantize_i8_slab/50000` (50k×384) | 157.65 ms | 26.61 ms | **0.169 (~5.9×)** | KEEP |

```bash
AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc RUST_LOG=off \
  cargo bench -p frankensearch-index --profile release --bench quantize_slab
```

**Scope:** original-comparator ratio **N/A** (frankensearch's own int8 ADC tier; no Tantivy counterpart).
This is **index-build / cold-start latency** — the slab is `OnceLock`-cached, so amortized for a static
index but recurs on every rebuild/refresh: a 50k×384 int8 slab build drops **157.65 ms → 26.61 ms**
(~131 ms/build saved; scales linearly with corpus). The first *non-dot* SIMD win — a genuinely different
primitive (quantizer), and the largest single ratio besides the f16 dots, because it's pure F16C
decode-bound. Route-next: `pack_4bit_slab` (the **wired-default** 4-bit slab build, same decode-bound
shape but with nibble packing) is the natural sibling. Kept bench: `quantize_slab`.
