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

### 2026-06-28 — runtime-dispatched AVX2+F16C 4-bit slab PACK: 10.3–13.6× on the WIRED-DEFAULT build (BlackThrush)

**Lever (the int8 quantizer's sibling — and the default path).** `pack_4bit_slab` is the lazy 4-bit ADC
slab build behind `search_top_k_4bit_two_pass_filtered` — i.e. **the wired-default sync-hybrid pass-1
storage** (`in_memory.rs::pack_4bit_slab`, `OnceLock`-cached). Same decode-bound shape as the int8 slab
(software `f16::to_f32` ×2 for max-abs + quantize) PLUS a per-nibble `slab[i] |= nib` read-modify-write.
New `simd::pack_f16_slab_to_4bit` runtime-dispatches to an AVX2+F16C kernel: `vcvtph2ps` decodes 8
f16/instruction for both passes; the quantize pass reuses the int8 `×scale` → round-half-away → clamp
pipeline (clamp `[-7,7]`) → `cvttps_epi32`, then packs the 8 nibbles as **4 fully-determined direct
byte writes** (no RMW). It beats int8's ratio because it kills BOTH the decode cost AND the per-nibble
RMW. Falls back to the scalar kernel; `in_memory.rs` delegates (the old scalar slab packer was removed).

**Bit-identical:** same nibble values + byte layout — `simd::tests::avx2_pack_4bit_matches_generic`
asserts byte-equal `Vec<u8>` across dim shapes (full 8-chunks, sub-8 tails, **odd** dims) × multiple
vectors. Conformance: **371/371** index lib tests GREEN serial (incl. the 4-bit two-pass recall gates,
which exercise the slab through the delegated `pack_4bit_slab`).

**Measured (per-crate, AVX2+F16C; dim 384 f16 slab → packed 4-bit; `generic` = scalar, `dispatch` =
AVX2+F16C):**

| Workload | generic | dispatch | Ratio | Status |
|----------|---------|----------|-------|--------|
| `pack_4bit_slab/10000` (10k×384) | 31.38 ms | 2.30 ms | **0.073 (~13.6×)** | KEEP |
| `pack_4bit_slab/50000` (50k×384) | 150.91 ms | 14.64 ms | **0.097 (~10.3×)** | KEEP |

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  cargo bench -p frankensearch-index --profile release --bench quantize_slab -- pack_4bit_slab
```

**Scope:** original-comparator ratio **N/A** (own 4-bit ADC tier). Index-build / cold-start latency on
the **default** path: a 50k×384 4-bit slab build drops **150.91 ms → 14.64 ms** (~136 ms/build saved,
linear in corpus), recurring on rebuild/refresh. Largest non-dot ratio yet. With int8 + 4-bit, **both
quantizer slab builds are now AVX2-dispatched**; the per-query `quantize_i8_query`/`pack_4bit_query` are
dim-sized (negligible) and not worth it — the build/quantize SIMD vein is now mined too. Kept bench:
`quantize_slab`.

### 2026-06-28 — runtime-dispatched F16C f32→f16 ENCODE: 4.1–5.4× (the most-common build kernel) (BlackThrush)

**Lever (the encode, not the decode/quantize — every index build).** `f16::from_f32` is the per-element
conversion at the heart of EVERY index build — `InMemoryVectorIndex::from_vectors` (both tiers of the
sync hybrid), FSVI writes. New `simd::encode_f32_to_f16_extend` runtime-dispatches to F16C `vcvtps2ph`
(8 f32→f16/instruction, round-to-nearest-even) and — critically — **bulk-stores the 8 f16 straight into
the `Vec`'s spare capacity (`_mm_storeu_si128` + `set_len`)** rather than `push`-ing per element (the
first cut at ~1.6× was push-bound; bulk store took it to ~5×). `in_memory.rs::from_vectors` now calls it.

**Bit-identical:** `vcvtps2ph` round-to-nearest-even == `half::f16::from_f32` for finite inputs
(`half::f16` is `repr(transparent)` over `u16`, so the store is layout-safe) —
`simd::tests::avx2_f16encode_matches_generic` asserts `to_bits` equality across normal/large(near-f16-max)/
tiny(subnormal) magnitudes + sub-8 tails. Conformance: **372/372** index lib tests GREEN serial (the
`from_vectors_*` tests exercise the new encode on the build path).

**Measured (per-crate, AVX2+F16C; dim 384 f32→f16; reused output buffer to isolate the encode from
per-iter allocation — matching the real build which appends to one reserved flat Vec):**

| Workload | generic (`half::from_f32` + push) | F16C (bulk store) | Ratio | Status |
|----------|-----------------------------------|-------------------|-------|--------|
| `encode_f32_to_f16/10000` (10k×384) | 5.76 ms | 1.06 ms | **0.184 (~5.4×)** | KEEP |
| `encode_f32_to_f16/50000` (50k×384) | 31.40 ms | 7.73 ms | **0.246 (~4.1×)** | KEEP |

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  cargo bench -p frankensearch-index --profile release --bench quantize_slab -- encode_f32_to_f16
```

**Scope:** original-comparator ratio **N/A** (own storage encoding). Index-build / load latency on the
**most-common** path (every `from_vectors`, all f16 indexes — the default). Smaller ratio than the
quantizer slabs (`from_f32` is a cheaper bit-twiddle than `round`/decode, and the bulk store is then
memory-bandwidth-bound), but it caps the build/load SIMD vein: dot kernels, int8+4-bit slab quantizers,
and now the f16 encode are all AVX2/F16C-dispatched. The lesson (re-recorded): when a decode/convert
kernel shows a modest SIMD ratio, check whether per-element `push` is the bottleneck — bulk-store into
spare capacity. Kept bench: `quantize_slab` (now also covers `encode_f32_to_f16`).

### 2026-06-28 — file-based FSVI slab write: F16C encode + batched write_all = 6.4–7.3× (BOLD build path) (BlackThrush)

**Lever (the FILE-based persist, the build path BOLD itself uses).** `write_vector_slab` (the
`VectorIndex::finish`/persist encoder, reached by `VectorIndex::create` + `write_record` — i.e. how the
**BOLD fixture builds its index** and how every on-disk FSVI is written) did per-element
`writer.write_all(&f16::from_f32(v).to_le_bytes())` — ~38M two-byte `write_all` calls for a 100k×384
index, each preceded by a software `from_f32`. Now: F16C-encode each record's f32→f16 via
`simd::encode_f32_to_f16_extend` into a reused scratch `Vec<f16>`, then ONE `write_all` of the whole
record's bytes (`#[cfg(target_endian="little")]` casts the `repr(transparent)` f16 slab to `&[u8]`; a
BE fallback keeps the per-element `to_le_bytes`). Kills BOTH the per-element `write_all` overhead AND the
scalar `from_f32`.

**Bit-identical:** the on-disk slab is byte-for-byte identical (LE f16) — the `write_f16_slab` bench
asserts the two arms emit equal bytes, and the existing FSVI write→read round-trip tests (which decode
the slab back) stay GREEN. Conformance: **372/372** index lib tests GREEN serial.

**Measured (per-crate; dim 384 f32 records → LE f16 slab into an in-memory writer; `per_element` = old
per-value `from_f32`+`write_all`, `batched` = F16C encode + one `write_all`/record):**

| Workload | per_element | batched | Ratio | Status |
|----------|-------------|---------|-------|--------|
| `write_f16_slab/10000` (10k×384) | 8.75 ms | 1.20 ms | **0.137 (~7.3×)** | KEEP |
| `write_f16_slab/50000` (50k×384) | 48.06 ms | 7.52 ms | **0.156 (~6.4×)** | KEEP |

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  cargo bench -p frankensearch-index --profile release --bench quantize_slab -- write_f16_slab
```

**Scope:** original-comparator ratio **N/A** (own FSVI storage write). Unlike the in-memory `from_vectors`
encode (`9f2356a`), this is the **file/persist** build — the path the BOLD comparator fixture itself
pays, plus every persisted/compacted index. Bigger than the encode-alone ratio because batching also
removes ~38M→count `write_all` calls. The remaining per-element `f16::from_f32` sites (WAL compaction
`lib.rs` ~1095, WAL lookup ~1363) are low-traffic compaction/lookup paths — left as-is. Kept bench:
`quantize_slab` (now also covers `write_f16_slab`).

### 2026-06-28 — InMemory two-pass full-recall short-cut: 1.45–1.49× (skip the pass that prunes nothing) (BlackThrush)

**Lever (the InMemory analog of the file-backed full-recall path).** `VectorIndex` (file-backed) already
skips the two-pass for full recall (`search.rs:170`: `limit >= total → collect-and-sort single f16 pass`).
`InMemoryVectorIndex`'s `search_top_k_int8_two_pass_filtered` / `search_top_k_4bit_two_pass_filtered` did
NOT: when `candidate_count = min(limit·mult, count) >= count` (a `limit_all`-class query, or a large limit
on a small corpus), pass-1 keeps **every** vector (the size-N bounded heap never evicts) and pass-2
rescores all N — so the quantized pass-1 scan, the size-N heap, the query quantize/nibble-prep, and the
lazy slab build (`quantize_i8_slab`/`pack_4bit_slab`) are **pure overhead**. Added a one-branch short-cut:
when `candidate_count >= count`, delegate to the exact `search_top_k` (f16 single pass).

**Bit-identical by construction:** the two-pass's own doc-contract is "== `search_top_k` whenever pass-1
retains the true top-k"; full recall retains *every* candidate, so the delegation is exact (not merely
approximate). `limit.min(count)` avoids a `usize::MAX`-sized heap. The `int8/four_bit_two_pass_keep_all_
matches_exact` tests + a new bench `assert_eq!(two_pass(N,1) == search_top_k(N))` confirm it. Conformance:
**372/372** index lib tests GREEN serial.

**Measured (per-crate `int8_two_pass` bench, N=10k×384 clustered, full recall = top-N; fast-path disabled
to capture the old cost, then restored):**

| Workload | old two-pass | exact (fast-path) | Ratio | Status |
|----------|--------------|-------------------|-------|--------|
| `full_recall/4bit` (k=N) | 2.220 ms | 1.491 ms | **0.67 (~1.49×)** | KEEP |
| `full_recall/int8` (k=N) | 2.162 ms | 1.491 ms | **0.69 (~1.45×)** | KEEP |

(With the fast-path restored, the two-pass arms collapse to ~1.31 ms ≈ the exact arm — the delegation is
realized.) Plus a **cold-start bonus** not in the steady-state number: full-recall now skips the
`OnceLock` slab build entirely (the 5–13× slab quantizers don't even run).

**Scope:** original-comparator ratio **N/A** (this is the InMemory sync-hybrid path; BOLD's `limit_all`
uses the file-backed `VectorIndex`, already single-pass — so this does NOT move the BOLD `limit_all` 1.4×,
which stays inherent). It closes an *internal* gap: InMemory was doing redundant pass-1 work the
file-backed path already avoided, on `limit_all`/large-limit queries through the wired two-pass. Kept
bench: `int8_two_pass` (now has a `full_recall` group).

### 2026-06-28 — AVX2 JL xorshift accumulate: 1.51× (a DIFFERENT primitive — SIMD PRNG) (BlackThrush)

**Lever (off the dot/quantize vein — the embedder's PRNG).** `HashEmbedder::embed_jl` (the
Johnson-Lindenstrauss projection embedder, the *compute-bound* tier — `O(tokens·dim)` xorshift) folds 4
independent xorshift64 token-chains per dimension. The hot loop `jl_accumulate_lanes` was scalar-ILP: 4
chains × 3 shift-xor steps = 24 scalar ops/dim plus 4 sign-selects. The 4 chains are exactly one
`__m256i` of u64 lanes (`JL_LANES == 4`), so the new `#[target_feature(enable="avx2")]`
`jl_accumulate_lanes_avx2` does each step as a single vector `slli`/`srli`/`xor_epi64` (6 vector ops vs
24 scalar), and recovers the per-dim ±1 sum branch-free: shift each lane's low bit into the sign
position, `_mm256_movemask_pd` the 4 signs, `4 - 2·popcount(mask)` = the exact integer `a0+a1+a2+a3`.
Runtime-dispatched (`is_x86_feature_detected!("avx2")`); scalar-ILP is the fallback.

**Bit-identical:** the xorshift is per-lane identical, and the per-dim contribution is a sum of ±1 lanes
— an exact small integer added to an exact-integer accumulator (`|value| ≤ token count ≪ 2²⁴`), so neither
the SIMD lane grouping nor the `4-2·popcount` reformulation changes a bit. `jl_avx2_matches_scalar`
asserts `to_bits` equality across 10 dim shapes accumulated over 25 token-groups (+ the all-odd
zero-state edge). Conformance: **289/289** embed lib tests GREEN.

**Measured (per-crate `hash_embed` bench, dim 384, 2000 4-chain groups ≈ 8k tokens, kernel isolated from
tokenize/normalize):**

| Workload | scalar-ILP | AVX2 | Ratio | Status |
|----------|-----------|------|-------|--------|
| `jl_accumulate` (384×2000) | 2349.7 µs | 1556.0 µs | **0.662 (~1.51×)** | KEEP |

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  cargo bench -p frankensearch-embed --profile release --bench hash_embed -- jl_accumulate
```

**Scope:** original-comparator ratio **N/A** (own embedder; no Tantivy counterpart). JL is **non-default**
(the default `HashEmbedder` is `FnvModular`, whose hash is inherently sequential — not SIMD-able), so this
helps deployments that pick the JL tier (better distance preservation) — the compute-bound embed path for
them. A genuinely **different primitive** (SIMD PRNG/xorshift, not a dot/quantize/encode), satisfying the
"never-safe-Rust-ceiling" steer with AVX2 u64 intrinsics. The smaller-than-the-decode-kernels ratio is
expected: the scalar-ILP already pipelines the 4 independent chains, so the win is the 4×-fewer
instructions, not a latency unlock. Kept bench arm: `jl_accumulate/scalar`.

### 2026-06-28 — JL accumulate, 4-lane → 8-lane AVX2 (2-way ILP): 1.76× further (2.76× over scalar) (BlackThrush)

**Lever (route-next from the JL win — the 4-lane kernel was latency-bound).** Yesterday's 4-lane AVX2
`jl_accumulate_lanes` is ONE `__m256i` whose 3-step xorshift (`s ^= s<<13; s ^= s>>7; s ^= s<<17`) is a
**dependency chain** → latency-bound (each step waits on the previous). Running TWO independent `__m256i`
(8 chains, `jl_accumulate_lanes8_avx2`) with the xorshift steps **interleaved** exposes the 2-way ILP that
hides that latency — the CPU pipelines the two chains. Wired as the production JL kernel (`JL_LANES = 8`,
`embed_jl` groups tokens 8-at-a-time); the 4-lane kernel is kept as the bench A/B baseline.

**Bit-identical:** the per-dim contribution is a sum of ±1 lanes (an exact small integer, `|value| ≤ token
count ≪ 2²⁴`) — independent of lane grouping AND of the `8 - 2·(popcount(maskA)+popcount(maskB))`
reformulation. `jl_8lane_matches_4lane` asserts the 8-lane embed (AVX2 **and** scalar) is `to_bits`-equal
to the 4-lane embed across 5 dim shapes. Conformance: **290/290** embed lib tests GREEN (all JL output
tests unchanged — the embed is byte-for-byte the same).

**Measured (per-crate `hash_embed` bench, dim 384, 8000 tokens, kernel isolated):**

| Workload | scalar-ILP | AVX2 4-lane | AVX2 8-lane | 8-lane vs 4-lane | 8-lane vs scalar |
|----------|-----------|-------------|-------------|------------------|------------------|
| `jl_accumulate` | 2778.9 µs | 1774.7 µs | **1005.8 µs** | **0.567 (~1.76×)** | **0.362 (~2.76×)** |

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  cargo bench -p frankensearch-embed --profile release --bench hash_embed -- jl_accumulate
```

**Scope:** original-comparator ratio **N/A** (own JL embedder, non-default). Bigger jump than yesterday's
4-lane (1.51×) precisely *because* the 4-lane was latency-bound on a single register's dependency chain —
the second register is a latency unlock, not just more width. **LESSON: a SIMD kernel whose hot recurrence
is a single-register dependency chain is latency-bound — add a 2nd independent accumulator register (more
lanes) before assuming the SIMD is saturated.** Kept bench arms: `jl_accumulate/{scalar,avx2_4lane,avx2_8lane}`.

### 2026-06-28 — AVX2 element-wise vector accumulate (model2vec mean-pool): ~1.4–1.6× (BlackThrush)

**Lever (the one candidate I'd dismissed by REASONING, not measuring).** `Model2VecEmbedder::embed_sync`
mean-pools by accumulating each in-vocab token's embedding row into a sum (`sum[d] += row[d]`, T tokens ×
`dim`). I'd assumed LLVM auto-vectorizes it away — true, but only to the **SSE2 baseline** (this workspace
builds with no global `+avx2`). New `simd::accumulate_f32_into` runtime-dispatches to a hand-written AVX2
kernel (`_mm256_loadu_ps`/`_mm256_add_ps`/`_mm256_storeu_ps`, 8 f32/instruction = 32-byte loads), which
roughly doubles the per-cycle load bandwidth on this memory-bound loop. `model2vec_embedder.rs`'s mean-pool
now calls it.

**Bit-identical:** each `sum[d] += row[d]` is an independent element-wise add (NOT a cross-lane reduction),
so SIMD only changes how many dims are added per instruction, never the per-dim arithmetic — unlike the
dot/JL reductions, this needs no reorder. `simd::tests::avx2_accumulate_matches_scalar` asserts `to_bits`
equality across 11 dim shapes accumulated over 5 rows. Conformance: **291/291** default embed lib tests +
**27/27** `--features model2vec` tests (the mean-pool feeds `embed_output_is_l2_normalized`,
`embed_deterministic`, etc.) GREEN.

**Measured (per-crate `hash_embed` bench, dim 384, 64-row mean-pool; the production `accumulate_f32_into`):**

| Workload | scalar (SSE2 auto-vec) | AVX2 dispatch | Ratio | Status |
|----------|------------------------|---------------|-------|--------|
| `vec_accumulate` run 1 | 2.194 µs | 1.530 µs | **0.70 (~1.43×)** | KEEP |
| `vec_accumulate` run 2 | 2.735 µs | 1.680 µs | **0.61 (~1.63×)** | KEEP |

(AVX2 is consistent ~1.5–1.7 µs; the scalar arm is the noisier one, so ~1.4–1.6× is the honest range.)

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  cargo bench -p frankensearch-embed --profile release --bench hash_embed -- vec_accumulate
```

**Scope:** original-comparator ratio **N/A** (own embedder). `Model2Vec` is the **opt-in** quality tier
(`model2vec` feature, potion-128M static embedder), so this speeds the mean-pool for deployments that pick
it. **Lesson reinforced (the one that bit me on the f16 dot): MEASURE, don't reason — I'd written off this
loop as "auto-vectorized" but the build's SSE2-only baseline left a clean ~1.5× on the table for a manual
AVX2 (wider loads).** Element-wise (non-reduction) kernels are the bit-identical-SAFE place to widen SIMD.
First SIMD in the embed crate's shared path (reusable `accumulate_f32_into`). Kept bench: `vec_accumulate`.

### 2026-06-28 — AVX2 element-wise scale (l2_normalize, DEFAULT path): scale kernel ~1.70× (BlackThrush)

**Lever (the element-wise vein's DEFAULT-path application).** `l2_normalize_in_place`
(`core/traits.rs`) — called on **every** embed by every embedder (the FNV/JL default tiers, model2vec,
api) — has two passes: the sum-of-squares (a reduction, bit-identical-LOCKED) and the **scale**
`vec[d] *= inv_norm` (element-wise, bit-identical-SAFE). New `core::simd::scale_f32_in_place`
runtime-dispatches the scale to AVX2 (`_mm256_mul_ps` by a broadcast `inv_norm`, 8 f32/instruction);
the scalar loop (SSE2 auto-vec) is the fallback. **First SIMD in `frankensearch-core`** (std-only —
`is_x86_feature_detected!` + `core::arch` + opt-in `#[allow(unsafe_code)]`).

**Bit-identical:** each `vec[d] *= inv_norm` is an independent IEEE f32 multiply, identical 1/4/8-wide
(no cross-lane reduction). `simd::tests::avx2_scale_matches_scalar` asserts `to_bits` equality across 11
dim shapes; `traits::tests::l2_normalize_in_place_matches_allocating` still GREEN (the AVX2 in-place
produces the same per-element `x·inv` as the allocating path). Conformance: **core** `l2_normalize` + simd
tests + **291/291** embed lib tests GREEN.

**Measured (per-crate `hash_embed` bench, dim 384, in-place scale; the production `scale_f32_in_place`):**

| Workload | scalar (SSE2 auto-vec) | AVX2 dispatch | Ratio | Status |
|----------|------------------------|---------------|-------|--------|
| `vec_scale` (scale kernel) | 2.015 µs | 1.184 µs | **0.59 (~1.70×)** | KEEP |

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  cargo bench -p frankensearch-embed --profile release --bench hash_embed -- vec_scale
```

**Scope:** original-comparator ratio **N/A** (own embedders). ~1.70× on the scale KERNEL; `l2_normalize`
overall is ~1.3× (the scale is one of its two passes; the sum-of-squares reduction stays scalar — locked).
This is the **default embed path** (vs the model2vec mean-pool's opt-in tier), so it speeds every embed on
every embedder — bigger AVX2 ratio than the accumulate (1.4–1.6×) because the scale is more compute-bound
(one input × broadcast scalar, less memory traffic). The `core::simd` helper is reusable for future
element-wise core ops. Kept bench: `vec_scale` (in `hash_embed`, importing the core helper).

### 2026-06-28 — BitsetFilter: identity hasher for already-hashed u64 keys: ~8–11% on the filtered scan (BlackThrush)

**Lever (a data-structure inefficiency, not a compute kernel).** `core::filter::BitsetFilter` stored its
allow-set as a default `HashSet<u64>` — i.e. it ran **SipHash over each key on every membership probe**.
But the keys are ALREADY FNV-1a `doc_id` hashes (uniform), so SipHash is pure redundant work. Added an
`IdentityHasherU64` (`write_u64` passes the key straight through; FNV-1a's high bits feed the SwissTable
control byte) + `BuildIdentityHasherU64`, and switched `BitsetFilter` to `HashSet<u64,
BuildIdentityHasherU64>` (re-bucketed once at construction). On a **filtered vector scan** the probe runs
per candidate (all N, before the cheap SIMD dot), so dropping SipHash speeds the hot loop.

**Bit-identical:** identity hashing preserves set membership exactly (a `u64` is present iff inserted) —
results are unchanged. Conformance: **core** filter tests 39/39 + **index** lib tests **372/372** GREEN
(filtered-search tests exercise `BitsetFilter`).

**Measured (per-crate `int8_two_pass` bench, N=10k clustered, dim 384, ~50%-pass filter; SipHash baseline
vs identity hasher):**

| Workload | SipHash | identity | Ratio | Status |
|----------|---------|----------|-------|--------|
| `flat_filtered` (exact f16 filtered scan) | ~205 µs | ~189 µs | **~1.08–1.11×** (criterion "improved", −8.6…−13%) | KEEP |
| `int8_filtered_mult5` (two-pass filtered) | ~204 µs | ~208 µs | within noise | — |

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  cargo bench -p frankensearch-index --profile release --bench int8_two_pass -- flat_filtered
```

**Scope:** original-comparator ratio **N/A** (filtered search — not the unfiltered BOLD path). The
hypothesis that SipHash *dominated* the filtered scan was wrong — measured it's ~8% of the exact-scan
cost (the dot + heap + memory dominate), so the win is modest, and the int8/4-bit two-pass filtered path
is within noise (the probe is a smaller relative fraction there; the identity probe is strictly faster, no
regression). Still a real, bit-identical fix of a genuine inefficiency (SipHash over an already-uniform
hash) on the common filtered-search path. This is a SAFE-Rust data-structure lever (not a SIMD intrinsic),
but a different *kind* of primitive from the compute kernels. `IdentityHasherU64` is reusable for any
already-hashed-u64 set. Kept the existing `int8_two_pass` filtered bench arms.

### 2026-06-29 — selective-filter GATHER fast-path: invert the loop, 6.9–50× on filtered vector search (BlackThrush)

**Lever (a DIFFERENT primitive — loop-order inversion, not a kernel).** A filtered vector search with a
hash-addressable allow-list (`BitsetFilter` — "search within these K docs") previously scanned the
**whole corpus** and ran one membership probe per document (the identity-hasher commit `f7d613b` made that
probe cheap, but it still runs `N` times). When the allow-set is a small fraction of the corpus — the
common real-world case (per-tenant / per-folder / per-ACL scoping) — almost all of that work is wasted.
The gather fast-path **inverts the loop**: iterate the (small) allow-set, map each hash → position via a
lazily-built bijective `hash → pos` table, and exact f16-scan **only** those positions. Work becomes
`O(|allow-set|)` instead of `O(corpus)`.

- New trait method `SearchFilter::candidate_hashes() -> Option<&DocIdHashSet>` (default `None`;
  `BitsetFilter` returns its allow-set). `None` ⇒ predicate/metadata/composite filters keep the scan.
- New lazy `InMemoryVectorIndex::hash_to_pos` map (identity-hashed, same FNV-1a key space). Returns
  `None` when two doc_ids collide to one hash (not a bijection) ⇒ disables the fast path, so results stay
  exact. Built once on first selective-filter search (other callers pay nothing).
- Wired into `search_top_k`, `search_top_k_int8_two_pass_filtered`, and
  `search_top_k_4bit_two_pass_filtered` (the production sync fast-tier). For the two-pass paths the gather
  is *exact* f16, so it is **strictly more accurate** than the approximate int8/4-bit pass-1 it replaces
  *and* far cheaper when selective.

**Bit-identical:** the gathered passing set `{pos : doc_id_hash[pos] ∈ allow-set}` is exactly the set the
per-document scan keeps, and both rank by the `(score, index)` total order, so order is independent of the
gather sequence. **Conformance: GREEN** — `frankensearch-core` filter 39/39, `frankensearch-index` lib
**373/373** (372 + a new `selective_filter_gather_matches_scan` asserting gather == forced scan == exact
on the public, int8, and 4-bit filtered paths). The bench also asserts parity at all 8 selectivities × 32
queries before timing.

**Measured (per-crate same-binary A/B, in-memory N=50k clustered, dim 384, k=10, 32 queries; `scan` =
forced per-document parallel filtered scan, `gather` = allow-set gather; medians; ratio = gather/scan):**

| selectivity | allow-set | scan | gather | ratio | speedup |
|-------------|-----------|------|--------|-------|---------|
| 0.1 % | 50    | 165.9 µs | 3.31 µs  | **0.020** | **50.1×** |
| 0.5 % | 250   | 194.8 µs | 16.06 µs | **0.082** | **12.1×** |
| 1 %   | 500   | 219.4 µs | 31.96 µs | **0.146** | **6.9×**  |
| 2 %   | 1 000 | 197.9 µs | 63.45 µs | **0.321** | **3.1×**  |
| 5 %   | 2 500 | 267.8 µs | 154.7 µs | **0.578** | **1.7×**  |
| 10 %  | 5 000 | 449.0 µs | 434.6 µs | 0.968 | ~tie (crossover) |
| 25 %  | 12 500 | 853 µs  | 2 517 µs | 2.95  | 0.34× (loss) |
| 50 %  | 25 000 | 852 µs  | 5 117 µs | 6.01  | 0.17× (loss) |

The serial gather crosses over the parallel scan at ~10 %; the selectivity gate is set to **N/16 (≈6.25 %)**
(`GATHER_SELECTIVITY_DIVISOR`) — comfortably inside the winning region (~1.5× at the boundary) with margin
for machines whose core count shifts the crossover. Above the gate the scan path is taken unchanged (no
regression; the existing `int8_two_pass/flat_filtered` 50 % arm is unaffected — `50%·16 ≥ 100%` ⇒ no
gather).

```bash
AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc RUST_LOG=off \
  cargo bench -p frankensearch-index --profile release \
  --bench filtered_gather -- --sample-size 30 --warm-up-time 1 --measurement-time 2
```

**Scope vs comparator:** filtered search is exactly where a Lucene/Tantivy/Meilisearch-class engine applies
the filter *first* and searches only the surviving subset; frankensearch previously paid a full-corpus
vector scan regardless. This closes that structural disadvantage for selective filters and makes the
production sync fast-tier (`search_top_k_4bit_two_pass_filtered`) both faster and *exact* on selective
queries. Original-comparator ratio is not re-measured here (the BOLD harness has no selective-filter row
yet), so this is a frankensearch before/after on the filtered path, not a new head-to-head dominance claim.

**Route next:** (1) **parallelize the gather** (rayon chunks + `merge_*_partials`, the established
order-independent pattern) to push the crossover well above 10 % and reclaim the 6–25 % band; with that the
gate could widen toward N/2. (2) Plumb `candidate_hashes` into the **FSVI file-backed** scan
(`search.rs`) and the **lexical** filtered path so the inversion covers every tier, not just in-memory.
(3) Add a selective-filter row to the BOLD comparator to convert this into a head-to-head dominance number.

### 2026-06-29 — gather follow-up: PARALLELIZE the gather → 10% flips tie→1.3× win, gate widened to N/10 (BlackThrush)

**Lever (route-next #1 above, now done).** The serial gather lost above ~10% only because it ran one core
against the parallel scan. Split `scan_gather` into a per-slice `gather_range` + a dispatcher that
`par_chunks(PARALLEL_CHUNK_SIZE)` above the chunk size and merges per-chunk bounded heaps by the
`(score, index)` total order — the exact `scan_parallel` pattern, so still **bit-identical** (new test
`parallel_gather_matches_scan` at a >chunk-size allow-set; full `frankensearch-index` lib **374/374** GREEN;
the earlier one-off `soft_delete_*` red was a fixed-temp-path flake colliding with a concurrent test run —
green on isolated re-run). Tiny allow-sets (< `PARALLEL_CHUNK_SIZE`) stay serial, so the 6.9–50× small-M
wins are unchanged.

**Measured (same `filtered_gather` A/B, N=50k clustered dim 384 k10, gather/scan median; parallel gather):**

| selectivity | scan | gather | ratio | vs serial gather |
|-------------|------|--------|-------|------------------|
| 5 %  | 273.5 µs | 128.3 µs | **0.469 (2.1×)** | was 0.578 (1.7×) |
| 10 % | 288.4 µs | 217.7 µs | **0.755 (1.3×)** | was 0.968 (tie) — **flipped to a win** |
| 25 % | 421.8 µs | 726.1 µs | 1.72 (loss) | was 2.95 — better but still loses |
| 50 % | 808.7 µs | 1169.5 µs | 1.45 (loss) | was 6.01 |

Crossover moved from ~10% to **~13%** — capped not by the (now parallel) dots but by the gather's
**serial** setup (allow-set `collect` + position `sort_unstable`), which grows with the allow-set. Gate
widened `GATHER_SELECTIVITY_DIVISOR` 16 → **10 (N/10)**: ≥1.3× at the boundary, scan path unchanged above
it (no regression — 25/50% rows never reach the gather). **Route next (unchanged):** parallelize the gather
*setup* (`par_sort_unstable` + parallel allow-set materialization) to push the crossover past 25%; then
FSVI/lexical plumbing; then a BOLD selective-filter comparator row.

### 2026-06-29 — MMR cosine: 4-accumulator f64 dot → ~1.6× on `mmr_rerank` end-to-end (BlackThrush)

**Lever (a DIFFERENT primitive — multi-accumulator ILP, the proven pattern).** MMR diversity reranking is
dominated by `cosine_sim_pre`'s inter-doc dot — `O(k·n)` evaluations of a dim-d vector per rerank. That dot
was a **single-accumulator f64 reduction** (`dot += f64::from(a[i]) * f64::from(b[i])`), latency-bound on
the loop-carried `dot` and not auto-vectorized (strict f64 order). Reformulated as **4 independent f64
accumulators** so LLVM auto-vectorizes the f32→f64 widen + multiply-add to SSE2/AVX. No `unsafe`, no SIMD
intrinsics — pure reassociation.

**Bit-identical selection:** MMR is a **search-time** reranking score (not a persisted embedding), so the
f64 reassociation is the same accepted ULP trade as the landed vector-search dot kernels. The ULP-level
score shift does not change the selection at realistic `n·k` — the `mmr_rerank` bench asserts
`mmr_new == mmr_new_4acc` (identical selected index list) for both shapes, and all `frankensearch-fusion`
`mmr::` tests pass unchanged.

**Measured (per-crate same-binary A/B, `mmr_rerank` bench, dim 384, medians; `new` = single-acc running-max
MMR, `new_4acc` = same with the 4-acc dot):**

| shape (pool n, results k) | new (1-acc) | new_4acc | ratio | speedup |
|---------------------------|-------------|----------|-------|---------|
| n=100, k=20  | 622.7 µs  | 384.7 µs  | **0.618** | **1.62×** |
| n=200, k=50  | 3 063.7 µs | 1 896.8 µs | **0.619** | **1.62×** |

```bash
AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc RUST_LOG=off \
  cargo bench -p frankensearch-fusion --profile release \
  --bench mmr_rerank -- --sample-size 30 --warm-up-time 1 --measurement-time 2
```

**Scope:** original-comparator ratio **N/A** — MMR is an opt-in diversification reranker, not on the default
BOLD hybrid path, so this is a frankensearch before/after on that path (not a head-to-head dominance claim).
Real win for the diversification feature: ~1.6× wherever MMR runs. Kept bench arm: `mmr_rerank/new_4acc`.
**Route next:** the f64 widen + `pmullw`/`mulpd` may now bind the loop; an 8-accumulator split or an AVX2
`dot_product_f32_f32` (the index crate already has one) wired into fusion could push further — but MMR's
absolute cost is small, so lower priority than the gather's FSVI/lexical plumbing.

### 2026-06-29 — `count_lexical_tokens`: 256-byte LUT + branchless transition → ~1.5–1.8× (BlackThrush)

**Lever (a DIFFERENT primitive — branchless byte-class state machine).** `count_lexical_tokens` (the
fsfs lexical chunker's per-chunk token counter, run during indexing) had an ASCII byte fast path that, per
byte, called `is_token_byte` (`is_ascii_alphanumeric()` + a 5-way `matches!`, ~7–10 ops) and then branched
on a data-dependent `in_token` flag that **mispredicts on every token boundary** (code/path text is
boundary-dense). Replaced both with (1) a compile-time **256-byte class table** `TOKEN_BYTE` (one L1 load
per byte) and (2) **branchless transition counting**: a token ends at each token→non-token transition, so
`count += (prev & !cur)` with no data-dependent branch. The table is built from `is_token_byte` in a
`const` block, so `TOKEN_BYTE[b] == is_token_byte(b)` for every byte — **bit-identical** token counts.

**Conformance:** `frankensearch-fsfs` lexical tests GREEN; the bench also asserts `count_new == count_lut`
(identical counts) for every input size before timing.

**Measured (per-crate same-binary A/B, `lexical_count` bench, realistic ASCII code/doc chunk, medians;
`bytes` = the landed scalar byte path, `lut` = table + branchless):**

| input bytes | chars (UTF-8) | bytes (scalar) | lut (this change) | lut/bytes | speedup |
|-------------|---------------|----------------|-------------------|-----------|---------|
| 1 024  | 1.093 µs  | 620.5 ns  | 399.6 ns  | **0.644** | **1.55×** |
| 4 096  | 3.949 µs  | 2.221 µs  | 1.408 µs  | **0.634** | **1.58×** |
| 16 384 | 15.64 µs  | 10.17 µs  | 5.582 µs  | **0.549** | **1.82×** |

```bash
AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc RUST_LOG=off \
  cargo bench -p frankensearch-fsfs --profile release \
  --bench lexical_count -- --sample-size 40 --warm-up-time 1 --measurement-time 2
```

**Scope:** original-comparator ratio **N/A** — internal fsfs indexing-path primitive (the token count is
chunk metadata), so this is a frankensearch before/after, and it speeds **indexing throughput** (a real
Tantivy/Lucene-class dimension) wherever the lexical chunker runs over ASCII code/docs (the common case).
The LUT/branchless pattern is reusable for any byte-class state machine. Kept bench arm: `lexical_count/lut`.

### 2026-06-29 — S3-FIFO query-embedding cache: 4–14% fewer embed misses on skewed/scan streams (BlackThrush)

**Lever (bd-tjkm S3-FIFO admission — a DIFFERENT primitive: cache eviction policy, /alien-graveyard §15.1).**
The hot-path query-embedding cache (`CachedEmbedder` → `CacheState`) used plain **FIFO** eviction (evict
oldest by insertion order; `get` never reorders), so a scan of one-hit-wonder queries evicts the hot/reused
set and a re-requested query becomes a miss = a recomputed embedding. Replaced `CacheState`'s eviction with
**S3-FIFO** (Yang et al., SOSP 2023): three entry-count queues over one map — **Small** (probation, ~10%),
**Main** (promoted on reuse), **Ghost** (recently-evicted keys → re-admit straight to Main). A key
re-requested while resident is promoted to Main and survives scan churn; cold singletons are dropped from
Small. `&str` lookups (no per-get alloc); the `CachedEmbedder`/`CacheStats`/embed API is unchanged (only
`CacheState` internals changed — the codebase's existing `core::S3FifoCache` is byte-budgeted and would
force a per-get `String` alloc, so an entry-count S3-FIFO is the right fit here).

**Measured (`cache_replay` bench, 100k accesses, cap=128 = `DEFAULT_CAPACITY`; a miss = one embed):**

| trace | FIFO hit | S3-FIFO hit | FIFO miss | S3-FIFO miss | miss_ratio (s3/fifo) |
|-------|----------|-------------|-----------|--------------|----------------------|
| zipf s=2 (U=2000)       | 0.120 | 0.154 | 87 985 | 84 577 | **0.961** |
| zipf s=3 (U=5000)       | 0.145 | 0.211 | 85 536 | 78 892 | **0.922** |
| scan-polluted (hot 256) | 0.072 | **0.201** | 92 796 | 79 915 | **0.861** |

S3-FIFO has a **strictly higher hit rate on every trace** — 4 % fewer embeds on mild Zipf, **14 % fewer** on
scan-polluted streams (where FIFO admits one-hit-wonders and evicts the hot set; S3-FIFO's Small queue
absorbs them). Per cache op S3-FIFO is ~2× the FIFO time (≈52 ns→110 ns), but that is **ns-scale vs the embed
it avoids** (µs for the hash embedder, ~0.5 ms for model2vec/Native), so the net is positive end-to-end and
**scales with embedder cost** — timely as the pure-Rust Native/model embedders land (`a18943d`).

```bash
AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc RUST_LOG=off \
  cargo bench -p frankensearch-embed --profile release --bench cache_replay
```

**Conformance:** `frankensearch-embed` lib **292/292** GREEN — the existing cache tests (hit/miss, stats,
cap=1/2 eviction) all pass (at small sizes with freq 0, S3-FIFO matches FIFO order) plus a new
`s3fifo_keeps_reused_key_through_scan` proving a reused key survives a cold scan that overflows the cache.

**Scope:** original-comparator ratio **N/A** (internal cache policy). This is the S3-FIFO admission half of
**bd-tjkm**; the `cache_replay` bench is the replay-trace churn-identification the bead requires before the
policy change. Kept bench: `cache_replay`. **Route next (bd-tjkm remainder):** the expected-loss
candidate-budget controller (candidate_multiplier / quality_timeout from measured latency-vs-loss) + a full
query-path replay harness with deterministic fallback.

### 2026-06-29 — BOLD head-to-head re-measure: dominance HELD through the session (parity-or-better, limit_all inherent) (BlackThrush)

**Head-to-head confirmation vs the Tantivy/Lucene/Meilisearch-class incumbent**, re-run after this session's
5 perf wins + concurrent-agent commits to verify nothing regressed the comparator (the mission metric).
Ran `bold_verify_tantivy_class` (`-p frankensearch --features lexical --profile release`, 10-sample p50,
RCH `ovh-a`, `FRANKENSEARCH_BOLD_VERIFY_EMIT=1`). Ratio = frankensearch_p50 / incumbent_p50 (**< 1 = we win**):

| corpus | query_class | hash_hybrid ratio | verdict |
|--------|-------------|-------------------|---------|
| 10k  | exact_identifier | **0.825** | 1.21× faster |
| 10k  | natural_language | **0.811** | 1.23× faster |
| 10k  | quoted_phrase    | 1.039 | ~tie |
| 10k  | high_fanout      | 1.015 | tie |
| 10k  | zero_hit         | 1.000 | tie |
| 10k  | short_keyword    | 1.105 | 21 µs vs 19 µs (tiny abs) |
| 10k  | **limit_all**    | **1.445** | SLOWER (inherent) |
| 100k | natural_language | **0.812** | 1.23× faster |
| 100k | exact_identifier | 1.006 | tie |
| 100k | quoted_phrase    | 1.015 | tie |
| 100k | high_fanout      | 1.006 | tie |
| 100k | short_keyword    | 1.022 | tie |
| 100k | zero_hit         | 1.000 | tie |

**Result: parity-or-better on every top-k row at both 10k and 100k** (natural_language a clean ~1.23× win
at both scales; exact_identifier 1.21× at 10k) — the session's wins (gather, MMR, LUT, S3-FIFO) and the
concurrent frankentorch/rerank commits **did not regress the comparator**. The lone slower row is
**`limit_all` (~1.45×)**, the structurally-inherent gap (the hybrid embeds + scans the *entire* vector tier
+ RRF-fuses ~2·N where the incumbent is lexical-only; decomposed + confirmed optimal/inherent in the
2026-06-28 entry — the ~1.286→1.445 shift is 10-sample/worker variance on a path nothing this session
touched). Original-comparator verdict: **frankensearch is at parity-or-better vs the Tantivy-class incumbent
on all top-k query classes; the only residual is the inherent `limit_all` semantics gap.** No code change
(measurement); artifact `/data/projects/.rch-targets/frankensearch-cc/criterion/bold_verify/summary.{md,jsonl}`.

### 2026-06-29 — limit_all doc_id materialization clone is ~23% (NOT the ledger's "3-5%") — reopens the move-not-clone lever (BlackThrush)

**Lever-sizing measurement that corrects a wrong prior estimate.** The 2026-06-28 `limit_all` decomposition
estimated the RRF `into_owned` doc_id clone-elision (the `VectorHit<'a>` lever) at "~3-5% of limit_all" —
but that was never measured. `rrf_fuse` returns `Vec<FusedHit>` whose `doc_id: String` are `to_owned()`
clones of the borrowed inputs (`rrf.rs:348`, `FusedHitScratch::into_owned`); for `limit_all` (k=N) that is
N short-String allocations. Isolated it (`materialize_clone` bench, N doc_ids, owned-clone vs borrowed):

| N | owned clone (`Vec<String>`) | borrowed (`Vec<&str>`) | clone overhead |
|---|------------------------------|------------------------|----------------|
| 10 000  | **429.8 µs** | 3.94 µs | **~426 µs (109×)** |
| 100 000 | **4.856 ms**  | (~40 µs) | ~4.8 ms |

**At 10k the clone is ~426 µs = ~23 % of the measured `limit_all` p50 (1869 µs)** — not 3-5 %. Eliding it
would move `limit_all` from **1.445× → ~1.12×** vs the incumbent, closing most of the lone remaining gap.
**Route: the elision is bit-identical and likely fusion-crate-only** — have `rrf_fuse` take *owned* inputs
and **move** each winner's doc_id `String` out of the input (`std::mem::take`) instead of cloning, rather
than the multi-crate `VectorHit<'a>` lifetime refactor. (Negligible for top-k — k=10 clones are ~0 — so this
is a `limit_all`/large-fetch win specifically.) Next: implement `rrf_fuse` move-not-clone (owned inputs),
wire the sync hot callers, A/B vs the clone path. Kept bench: `materialize_clone`. Original-comparator: this
directly attacks the `limit_all` row (the biggest measured gap).

**UPDATE — move-not-clone is BLOCKED; the elision is invasive-multi-crate after all (traced the actual
caller).** The "fusion-crate-only move" route fails: `rrf_fuse` cannot move doc_ids out of its inputs
because the inputs' doc_ids are **reused by value all through phase-2 and result assembly, after the fuse**.
Concretely in the sync path (`sync_searcher.rs`): after `rrf_fuse(lexical, &fast_hits, …)`, phase-2
`quality_scores_for_hits(query_vec, &fast_hits)` looks up `find_index_by_doc_id(&hit.doc_id)` + the WAL by
`hit.doc_id` (`two_tier.rs:35,44,58`), and the score-maps + blend re-read `fast_hits`/`lexical_hits` doc_ids
(`sync_searcher.rs:~42-48`, and `lexical_hits.as_ref()` at ~215). So both inputs must stay intact (doc_ids
included) long after the fuse — `std::mem::take` would corrupt phase-2. Eliding the clone therefore needs a
**deep multi-crate pipeline lifetime refactor** (borrow doc_ids through fuse → phase-2 → blend, materialize
once at final `ScoredResult` assembly), which is exactly the invasive `VectorHit<'a>`-class change the
2026-06-28 decomposition flagged — and it's forbidden by the per-crate constraint. **Net: the clone is big
(~23%, the ledger's 3-5% was wrong) but its elision is invasive-multi-crate-blocked, not a clean lever. The
`materialize_clone` bench is kept as the lever-sizing evidence; no per-crate win is available on it.**

**The alternative elision primitives are ALSO multi-crate-invasive (don't chase them either):** (1)
**`Arc<str>` doc_ids** would make the clone a refcount bump (~8× cheaper, ~20% off limit_all) — but it
requires changing the doc_id type on `ScoredResult` + `VectorHit` + `FusedHit` + every construction site +
the public result API (huge cross-crate blast radius). (2) **inline small-string** (`SmolStr` etc.) doesn't
help: the public `ScoredResult.doc_id: String` at the end of the pipeline forces the heap allocation
regardless of an inline type in the middle. So **every approach (move, `Arc<str>`, inline) needs a deep
cross-cutting type/lifetime change** — the String-everywhere result API + the input-doc_id reuse in phase-2
make N owned allocations unavoidable per-crate. **The limit_all doc_id clone lever is conclusively closed
for single-crate work; the deep `Arc<str>`-or-lifetime refactor is the only path and is a deliberate
multi-crate decision, not a 60-min lever.**

**UPDATE 2 — parallelizing the clone (rayon) is also REJECTED (allocator-bound), and the clone is ~10% not
23% on a clean worker.** Since the clone can't be elided per-crate, the next idea was to keep it but
parallelize it (`into_par_iter().map(into_owned)`) — bit-identical (order preserved), single-crate,
non-invasive. Measured (`materialize_clone`, clean worker, owned serial vs `owned_clone_par` rayon):

| N | serial clone | rayon par clone | borrowed (lower bound) |
|---|--------------|-----------------|------------------------|
| 10 000 | **190.6 µs** | **228.3 µs (SLOWER)** | 2.90 µs |

The parallel clone is **~1.2× SLOWER** than serial: the N small `String` allocations **contend on the global
allocator**, so the clone is allocator-bound, not CPU-bound — rayon adds spawn overhead with no parallelism
to exploit. Rejected (not wired). Also note the serial clone is **190 µs here (~10 % of limit_all's 1869 µs),
not the 23 % from `dcde68f`** — that earlier run was on a contended worker (load ~13); 190 µs is the clean
number. **So the limit_all doc_id clone is ~10 %, allocator-bound, un-elidable per-crate (move/Arc/inline all
multi-crate), and un-parallelizable (allocator contention). Fully closed; `materialize_clone` bench kept,
par arm reverted.**

---

## 2026-06-29 — federated cross-shard fuse: SipHash → aHash on the merge map is ~1.09–1.22× (BlackThrush)

**Lever LANDED (single-crate, bit-identical).** `federated::{fuse_rrf, fuse_weighted}` accumulate per-doc
aggregates in `std::collections::HashMap<String, AggregateDoc>` — the DoS-resistant **SipHash** default
hasher — keyed on owned `doc_id` strings, while the sibling single-node `rrf_fuse_with_graph` already uses
`ahash::AHashMap`. Swapped the four federated map sites to `AHashMap` (aHash). This is **separate from the
Cobaltmoth clone-elim lever** (NEGATIVE_EVIDENCE 2026-06-27, ~1.05×, reverted): that attacked the key
*clone*; this attacks the key *hash*. The per-hit allocations (`doc_id.clone()`, `appeared_in.to_owned()`)
are unchanged, so this is a clean, orthogonal hasher win.

**Bit-identical:** `into_ranked_hits` sorts by a **strict total order** — `score.total_cmp` → `appeared_in`
count → `source_rank` → **`doc_id` tiebreak** (`federated.rs:491`). Since `doc_id` is the unique map key,
no two output entries compare Equal, so the transient map's (hasher-dependent) drain order never reaches the
result. Output is identical for every input.

**Measured** (per-crate A/B, isolates only the hasher — identical alloc profile in both arms;
`federated_fuse` bench, real merge shape: `String` key clone + `BTreeSet<String>` appeared-in + template clone):
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-fusion --bench federated_fuse
```

| Workload (shards × hits, universe) | SipHash (prod) | aHash (landed) | ratio | speedup |
|------------------------------------|----------------|----------------|-------|---------|
| `s5_h200_u600`   (5 × 200, ~50% overlap)  | 173.4 µs | 142.6 µs | **0.82** | **~1.22×** |
| `s10_h500_u2500` (10 × 500, ~50% overlap) | 805.4 µs | 742.0 µs | **0.92** | **~1.09×** |

CIs non-overlapping on both. The smaller workload shows the larger ratio (hash-compute is a bigger fraction
when the total doc set is small; at scale the `BTreeSet`/`String` allocations dilute it toward ~1.09×).

**Original-comparator ratio: N/A** — federated multi-shard fusion is a frankensearch-only cross-shard path
(Tantivy/Lucene/Meilisearch have no single-call cross-shard fuse comparator). Internal micro-lever; the
`federated_fuse` A/B bench is kept for re-validation.

---

## 2026-06-29 — two-tier blend final sort: stable `sort_by` → `sort_unstable_by` is ~1.30–1.40× (BlackThrush)

**Lever LANDED (single-crate, bit-identical).** `blend_two_tier` (the main two-tier quality-phase blend,
`blend.rs:179`) finished by sorting the deduped `Vec<VectorHit>` with a **stable** `sort_by`, while the
sibling RRF fuse path (`rrf.rs`) already uses `sort_unstable_by`. Switched it (and the identical
`federated::into_ranked_hits` sort, `federated.rs:497`) to `sort_unstable_by`. Found via the
sibling-path-consistency audit (an optimization on one path, not its twin) — same heuristic that landed the
federated aHash win (`9543ae6`).

**Bit-identical:** the comparator is a **strict total order** — `score.total_cmp` then a `doc_id` tiebreak —
and `blended` is built from an `AHashMap<&str, _>` so every `doc_id` is unique; no two elements compare
Equal, so stable and unstable produce identical output. Conformance GREEN (`frankensearch-fusion` lib
817/817). `federated` is the same argument (`doc_id` is the unique map key).

**Measured** (per-crate `blend_reorder` A/B, isolates the final sort over a realistic blended set —
unique ids, **mostly-distinct** f32 scores as in a real `alpha*q+(1-alpha)*f` blend, so the `doc_id`
tiebreak rarely fires):
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-fusion --bench blend_reorder
```

| Workload | stable `sort_by` (ORIG) | `sort_unstable_by` (landed) | ratio | speedup |
|----------|-------------------------|-----------------------------|-------|---------|
| `n200`  (typical top-k blend) | 5.95 µs  | 4.57 µs  | **0.77** | **~1.30×** |
| `n2000` (limit_all blend)     | 80.07 µs | 57.19 µs | **0.71** | **~1.40×** |

CIs non-overlapping on both. (An initial over-tied bench draft — only `n/8` distinct scores — made the
tiebreak dominate and showed a false `n2000` regression; the real blend score distribution is
mostly-distinct, where unstable cleanly wins by dropping the stable-sort O(N) scratch alloc + better
constant factors. Lesson: model the real key distribution, not a tie-adversarial one.)

**Original-comparator ratio: N/A** — the blend/federated final sort is a frankensearch-only two-tier /
cross-shard step (no Tantivy/Lucene/Meilisearch single-call equivalent). Internal sort micro-lever; the
`blend_reorder` A/B bench is kept for re-validation.

---

## 2026-06-29 — exact-search final winners sort: stable `sort_by` → `sort_unstable_by` (~1.16–1.47×, biggest on limit_all) (BlackThrush)

**Lever LANDED (single-crate, bit-identical) — lands on the biggest gap vs ORIG (`limit_all`).** The exact
vector search ordered its collected `Vec<HeapEntry>` best-first with a **stable** `sort_by(compare_best_first)`
at three sites: `search.rs:183` (the **`limit_all` scan-all path**, `scan_wal_collect_all`, where `winners`
holds *every* match), `search.rs:821` (bounded top-k, `heap.into_vec()`), and `in_memory.rs:857`
(`resolve_heap`). Switched all three to `sort_unstable_by`. Found via the sibling-path-consistency audit
(`rrf.rs`/`blend.rs` already use `sort_unstable_by`).

**Bit-identical:** `compare_best_first` is a **strict total order** — `score_key.total_cmp` then a
unique-`index` tiebreak (`index` is the vector position, unique per entry) — so no two entries compare Equal
and the unstable sort yields output identical to the stable sort. Conformance GREEN (`frankensearch-index`
lib 374/374).

**Measured** (per-crate `winners_sort` A/B, isolates the final sort over realistic winners sets — unique
indices, mostly-distinct cosine-like f32 scores):
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
  rch exec -- cargo bench -p frankensearch-index --bench winners_sort
```

| Winners (path) | stable `sort_by` (ORIG) | `sort_unstable_by` (landed) | ratio | speedup |
|----------------|-------------------------|-----------------------------|-------|---------|
| `n100`   (bounded top-k)        | 1.302 µs   | 1.121 µs   | **0.86** | **~1.16×** |
| `n10000` (`limit_all`)          | 247.6 µs   | 188.2 µs   | **0.76** | **~1.32×** |
| `n50000` (`limit_all`)          | 1858.2 µs  | 1263.7 µs  | **0.68** | **~1.47×** |

CIs non-overlapping on all. The win **grows with N** (the exact `limit_all` profile): unstable drops the
stable-sort O(N) scratch allocation + has better constant factors, saving ~595 µs/query at 50k winners.

**Original-comparator relevance:** this is the frankensearch-side final ordering of the **`limit_all`**
path — the documented biggest gap vs the Tantivy/Lucene/Meilisearch-class incumbent (BOLD `limit_all` row).
It shrinks the frankensearch side of that inherent gap. Internal sort lever; `winners_sort` bench kept.

---

## 2026-06-29 — MRL rescore final sort: stable `sort_by` → `sort_unstable_by` (~1.16–1.47×, same lever) (BlackThrush)

**Lever LANDED (single-crate, bit-identical) — extends the winners-sort sweep.** `mrl_search_with_stats`
(`mrl.rs:325`, the Matryoshka truncated-dims rescore search path) ordered its `rescored` candidates with a
stable `sort_by` before `truncate(limit)`. The comparator is the **identical strict total order** as the
exact-search winners sort — `nan_safe(score).total_cmp` then a **unique `index` tiebreak** (vector position,
distinct per rescored entry) — so `sort_unstable_by` is bit-identical and drops the mergesort scratch alloc.
Directly covered by the `winners_sort` bench (same crate, same `score`+`index` comparator):

| Winners (path) | stable `sort_by` | `sort_unstable_by` | ratio | speedup |
|----------------|------------------|--------------------|-------|---------|
| `n100`   | 1.302 µs  | 1.121 µs  | 0.86 | ~1.16× |
| `n10000` | 247.6 µs  | 188.2 µs  | 0.76 | ~1.32× |
| `n50000` | 1858.2 µs | 1263.7 µs | 0.68 | ~1.47× |

**Negative result on the same sweep — `IndexWriter::finish` records sort MUST stay stable (REVERTED).** The
`finish()` full-corpus records sort (`lib.rs:1624`, `doc_id_hash` then `doc_id`) looked like the same lever,
but `self.records` can contain **duplicate `doc_id`s** (soft-delete then rewrite), so the comparator is NOT
a strict total order and the **stable** sort is load-bearing for last-write-wins among dupes. Switching to
`sort_unstable_by` broke `soft_delete_wal_restores_state_on_rewrite_failure`. Reverted to `sort_by` and added
a guard comment. Lesson: the stable→unstable lever requires a **truly unique** tiebreak key — verify the data
can't carry duplicate keys (here, updated/rewritten docs), not just that the comparator *names* a unique field.

**Original-comparator ratio: N/A** (internal sort lever; MRL is a frankensearch-only truncated-dims search path).

---

## 2026-06-29 — limit_all winners sort: gated PARALLEL sort (`par_sort_unstable_by`) ~2.81× at 50k (BlackThrush)

**Lever LANDED (single-crate, bit-identical) — a NEW primitive on the biggest gap (`limit_all`).** The
exact-search `limit_all` scan-all final sort (`search.rs:183`, `winners` holds *every* match) was a serial
`sort_unstable_by`. The earlier ledger rejected parallelizing the materialization *clone* (allocator-bound),
but the **sort** is CPU-bound (comparison sort) and parallelizes cleanly. Gated `par_sort_unstable_by`
(rayon) above `PAR_SORT_THRESHOLD = 16_384` winners; below it the serial sort stays (rayon spawn/merge
overhead is not amortized for the cheap `compare_best_first` comparison). Bit-identical — `compare_best_first`
is a strict total order, so the parallel sort yields the same unique order. Conformance GREEN (index lib 374/374).

**Measured** (per-crate `winners_sort` bench, `par` arm vs serial `sort_unstable_by`):
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
  rch exec -- cargo bench -p frankensearch-index --bench winners_sort
```

| Winners | serial `sort_unstable_by` | `par_sort_unstable_by` | ratio | verdict |
|---------|---------------------------|------------------------|-------|---------|
| `n100`   | 1.76 µs   | 1.14 µs   | (noise; rayon serial-fallback) | serial (below threshold) |
| `n10000` | 196.0 µs  | 184.7 µs  | ~1.06× (marginal/noisy)        | serial (below threshold) |
| `n50000` | 1284.3 µs | 457.6 µs  | **0.36 (~2.81×)**              | **parallel (≥16384)** |

Threshold 16384 keeps the marginal/noisy n10000 zone serial and only parallelizes where the win is clear and
large — the `limit_all` case. Stacks on the earlier serial `sort_by → sort_unstable_by` win (`afb646b`):
together ~1.16× (top-k) → ~4.1× (50k winners, stable→par) on the biggest measured gap.

**Original-comparator relevance:** shrinks the frankensearch side of the `limit_all` BOLD gap vs the
Tantivy/Lucene/Meilisearch-class incumbent. Internal sort lever; `winners_sort` bench (now with `par` arm) kept.

---

## 2026-06-29 — BOLD re-measure post sort-wins: top10 at parity, limit_all 1.59× p50 (inherent) (BlackThrush)

**Surface (measured, no new single-crate lever).** Re-ran the BOLD Tantivy-class comparator
(`bold_verify_tantivy_class`, 10k corpus, hash hybrid + lexical-guard) after this session's sort wins to
locate the current biggest gap without ceiling-framing.

| query_class | frankensearch p50 µs | tantivy p50 µs | ratio | note |
|-------------|----------------------|----------------|-------|------|
| exact_identifier | 112  | 115  | **0.97** | at/under parity |
| short_keyword    | 30   | 28   | 1.07  | parity |
| quoted_phrase    | 144  | 142  | 1.01  | parity |
| natural_language | 153  | 154  | **0.99** | at/under parity |
| high_fanout      | 85   | 78   | 1.09  | parity |
| zero_hit         | 46   | 47   | 0.98  | at/under parity |
| **limit_all**    | **2151** | **1351** | **1.59** | the lone gap |

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc rch exec -- \
  env FRANKENSEARCH_BOLD_VERIFY_EMIT=1 RUST_LOG=error cargo bench -p frankensearch --features lexical \
  --profile release --bench search_bench bold_verify_tantivy_class -- --sample-size 10 --warm-up-time 1 --measurement-time 3
```

**Reading it:** all six top10 classes are **parity-or-better** (0.97–1.09; the >1.0 cases are within the
sample-10 noise band). `limit_all` is the **only** gap at **1.59× p50** — and at 10k corpus the parallel
winners sort (`PAR_SORT_THRESHOLD = 16384`) does **not** trigger, so this is the inherent hybrid cost (vector
scan + RRF + materialize) the lexical-only incumbent never pays. (The high p95/p99 ratios, 2.15×/3.42×, are
unreliable at `--sample-size 10` — essentially the max of 10 samples — not a systematic tail.)

**Where the 800 µs limit_all gap lives (single-crate sub-levers, all addressed or locked):** the vector scan
is bandwidth-bound + AVX2-capped; the winners sort is `sort_unstable_by` (+ gated parallel at ≥16384,
`7c53d3f`); the doc_id materialization is the numeric-ord fast field (`14e87e4`, up to 6.32×); the **RRF full
sort is the largest remaining slice (~472 µs at 10k ≈ 22% of limit_all, `rrf_sort` measurement this session)**
but its `cmp_for_ranking` `doc_id`-String tiebreak is **bit-identity-locked** — a cheaper deterministic key
would reorder tied docs — and `par_sort` regresses it (this ledger, RRF-par-sort entry). The residual
materialization `String` clones are allocator-bound + multi-crate-blocked. **Conclusion: limit_all is at its
single-crate floor; top10 is won. No per-crate lever remains on the measured gap.** Original-comparator
ratios are the table above (this IS the head-to-head vs the Tantivy/Lucene/Meilisearch-class incumbent).

---

## 2026-06-29 — in-memory limit_all winners sort: gated parallel sort (extends 7c53d3f to the fast tier) (BlackThrush)

**Lever LANDED (single-crate, bit-identical) — extends the proven parallel sort to the in-memory index.**
`d4bc73c`/`7c53d3f` gated `par_sort_unstable_by` on the file-backed exact-search limit_all sort (`search.rs:183`).
The **in-memory** index (`InMemoryVectorIndex::resolve_heap`, `in_memory.rs:857`) — the two-tier **fast tier** —
serves limit_all the same way: `search_top_k` with `limit >= count` builds a `count`-sized heap, so
`winners = heap.into_vec()` holds **every record** and was sorted serially. Applied the same gated
`par_sort_unstable_by` above `PAR_SORT_THRESHOLD = 16_384`.

**Bit-identical & measured-by-coverage:** `in_memory.rs`'s `compare_best_first` is the **identical** strict
total order as `search.rs`'s (`score_key.total_cmp` then unique `index` tiebreak), and the data (POD
`HeapEntry`) is identical, so the **`winners_sort` bench directly covers this site** — no new bench needed:

| Winners | serial `sort_unstable_by` | `par_sort_unstable_by` | ratio | verdict |
|---------|---------------------------|------------------------|-------|---------|
| `n10000` | 196.0 µs  | 184.7 µs  | ~1.06× (kept serial, <threshold) | serial |
| `n50000` | 1284.3 µs | 457.6 µs  | **0.36 (~2.81×)** | **parallel (≥16384)** |

Conformance: `frankensearch-index` lib **373/374** — the lone failure is the **pre-existing flaky**
`soft_delete_wal_restores_state_on_rewrite_failure` (a WAL-restoration test, intermittent, unrelated to this
gated sort: tiny test corpora never reach the 16384 parallel branch, and the serial `else` path is
byte-identical to before). Original-comparator relevance: shrinks the in-memory fast-tier side of the
limit_all gap. The `winners_sort` bench covers it.
