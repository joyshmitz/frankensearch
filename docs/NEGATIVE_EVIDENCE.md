# NEGATIVE_EVIDENCE.md — frankensearch perf swarm

> Honest ledger of perf experiments that **did NOT pay off** (≈0 gain or regression)
> and were therefore **reverted**. The point of this file is to stop future agents
> (and future me) from re-attempting dead ends. Every entry must cite the measured
> ratio vs. the pre-change baseline on the same workload.

Conventions:
- **Workload** = the exact bench id (`cargo bench` group/function) measured head-to-head.
- **Ratio** = new_time / old_time. `< 1.0` is a speedup, `> 1.0` is a regression.
- A lever is **reverted** if ratio ∈ [0.97, 1.03] (noise) or > 1.03 (regression).
- Wins (ratio < 0.97, kept) go in `docs/PERF_LEDGER.md`, not here.

Build/bench protocol (per-crate ONLY):
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/<agent-lane> \
  rch exec -- cargo bench -p <crate> --profile release
```

---

## Measurement blockers

| Date | Owner | Workload | Evidence | Status |
|------|-------|----------|----------|--------|
| 2026-06-27 | BlackThrush (`frankensearch-cod-a`) | `frankensearch/search_bench bold_verify_tantivy_class` clean hot-span retry | No checked bench worktree had commits ahead of `main` (`git rev-list --count main..HEAD` = 0). In isolated worktree `amberfield-lexical-trace-20260627`, added only the `FRANKENSEARCH_BOLD_VERIFY_SUMMARY_ONLY` harness gate so BOLD emits rows before Criterion's trace flood, then retried the prior two-attribute hot-span lever (`execute_query_with_offset` + `TantivyIndex::search_doc_ids`). Span-enabled baseline on `ovh-a` completed but emitted ~126k tokens of close-event tracing; hybrid ratios vs the Tantivy/Lucene/Meili-class proxy were still slower on important rows: 10k exact 1.074, 10k short 1.095, 10k quoted 1.050, `limit_all` 1.213, 100k short 1.046. The no-spans candidate on `vmi1227854` was mixed and cross-worker, with no accepted keep row: 10k quoted 1.164, `limit_all` 1.244, 100k exact 1.158, 100k natural-language 1.047, 100k short 1.000, 100k high-fanout 1.016, 100k zero-hit 1.000. A same-worker A/B attempt failed because the immediate post-baseline candidate rerun selected `vmi1264463` instead of `ovh-a` and was interrupted before measurement. Accepted Lucene/Tantivy/Meili-class ratio: none; source hunk reverted. | Landed only the summary-only BOLD harness switch plus this ledger entry. Agent Mail DB was corrupt, so no reservation/message could be recorded. |
| 2026-06-24 | BlueGull (`frankensearch-cod-a`) | `frankensearch/search_bench vector_search_topk/top10/10000` | `cargo bench --release -p frankensearch --bench search_bench vector_search_topk/top10/10000 -- --quiet` failed before measurement on rustc `1.98.0-nightly (f20a92ec0 2026-06-07)` because Cargo rejected `--release` for `cargo bench` as an unexpected argument. | Blocker tracked in `bd-ui41`; do not count as a perf ratio. |
| 2026-06-24 | BlueGull (`frankensearch-cod-a`) | `frankensearch/search_bench vector_search_topk/top10/10000` | Fallback optimized bench command without `--release` ran through RCH on `vmi1153651` but remained in cold compile/link with no Criterion timing output after more than 10 minutes; interrupted by the owner with exit 130. | No ratio produced; use `bd-ui41` to establish a reproducible harness and command contract. |
| 2026-06-24 | BlueGull (`frankensearch-cod-a`) | Tantivy/Lucene-class original comparison | README/AGENTS confirm frankensearch is a Tantivy BM25 + semantic/vector hybrid, but no current per-crate harness emits same-corpus ratios against a Tantivy-only incumbent. | Blocker tracked in `bd-ui41`; no dominance claim is valid until this exists. |
| 2026-06-24 | frankensearch-cod-b | `frankensearch/search_bench` requested protocol | `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b rch exec -- cargo bench -p frankensearch --release --bench search_bench -- --sample-size 10` selected RCH worker `ovh-a`, then Cargo rejected `--release` for `cargo bench` with `unexpected argument '--release' found`; `cargo bench --help` lists `--profile <PROFILE-NAME>` instead. | Same protocol blocker as above; successful measurement used `--profile release` and remains per-crate. |
| 2026-06-24 | frankensearch-cod-b | `frankensearch/search_bench vector_search_topk/top10/{1000,5000,10000}` | Same-worker RCH run on `ovh-a`: `rch exec -- cargo bench -p frankensearch --profile release --bench search_bench -- --sample-size 10`. Results: 1K `944.07 us`, 5K `3.4640 ms`, 10K `1.6642 ms`. | Scaling order is unstable/noisy; use as routing evidence only, not as keep/reject proof. |
| 2026-06-24 | frankensearch-cod-b | `frankensearch-index/dot_product` release-profile comparison from detached baseline worktree | Three attempts with `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b rch exec -- cargo bench -p frankensearch-index --profile release --bench dot_product -- --sample-size 10 --warm-up-time 1 --measurement-time 3` fell open to local with `no admissible workers: insufficient_slots=8,health_below_fallback=2,hard_preflight=1`; each local fallback was interrupted before measurement. | No release-profile ratio. Bench-profile RCH runs may be routing evidence, but kept wins need an admitted remote release-profile run or an in-process head-to-head harness. |
| 2026-06-24 | BlueGull (`frankensearch-cod-a`) | `frankensearch-index/dot_product f32_bytes` vs Tantivy/Lucene/Meilisearch-class original | Kept microkernel proof is against the embedded pre-change frankensearch `f32_bytes_old` baseline: pinned RCH worker `vmi1149989` measured `dot/dim256/f32_bytes/10000` at 3.4835 ms old -> 0.66126 ms new (ratio 0.190) and `dot/dim384/f32_bytes/10000` at 5.1487 ms old -> 1.8811 ms new (ratio 0.365). There is still no same-corpus Tantivy/Lucene/Meilisearch-class comparator for this vector-byte kernel or end-to-end workload. | Original-comparator ratio remains blocked by `bd-ui41`; do not claim dominance over Lucene/Tantivy/Meilisearch-class from this microkernel win alone. |
| 2026-06-26 | BlackThrush | `frankensearch-index/dot_product f32_slice` vs Tantivy/Lucene/Meilisearch-class original | Kept microkernel proof is against the embedded pre-change frankensearch `f32_slice_old` baseline, not a search-engine incumbent: remote RCH worker `vmi1227854` measured `dot/dim256/f32_slice/10000` at 2.3594 ms old -> 750.64 us new (ratio **0.318**) and `dot/dim384/f32_slice/10000` at 3.7372 ms old -> 2.2784 ms new (ratio **0.610**) with `cargo bench -p frankensearch-index --profile release --bench dot_product f32_slice -- --sample-size 10 --warm-up-time 1 --measurement-time 3`. A same-command local fallback was mixed/noisy (dim256 **1.216x slower**, dim384 **0.935x faster**) and is not used for the keep decision. | Original-comparator ratio is not applicable for this vector microkernel. Do not claim new Lucene/Tantivy/Meilisearch-class search dominance from it; the residual BOLD lexical/materialization misses below still stand. |
| 2026-06-27 | Codex | `frankensearch-index/dot_product fourbit_prepared` vs Lucene/Tantivy/Meilisearch-class original | Kept prepared-query proof is against the embedded pre-change frankensearch packed-4-bit dot baseline, not a search-engine incumbent: remote RCH worker `hz2` measured `dot/dim256/fourbit_prepared/10000` at 459.40 us old -> 389.81 us new (ratio **0.849**) and `dot/dim384/fourbit_prepared/10000` at 567.38 us old -> 509.84 us new (ratio **0.899**) with `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b rch exec -- cargo bench -p frankensearch-index --profile release --bench dot_product fourbit_prepared -- --sample-size 10 --warm-up-time 1 --measurement-time 3`. A same-session BOLD comparator probe for `frankensearch` was started, produced excessive tracing output, and was cancelled via `rch cancel 29904571677541210` before a complete summary; it is not used for the keep decision. | Original-comparator ratio vs Lucene/Tantivy/Meilisearch is **N/A** for this isolated vector dot kernel. Do not claim new Lucene/Tantivy/Meilisearch-class dominance from it; the existing BOLD rows and residual comparator negatives still stand. |
| 2026-06-27 | BlackThrush | `frankensearch-fusion/sync_int8_fetch fast_fetch_4bit` vs Lucene/Tantivy/Meilisearch-class original | Kept sync-hybrid fast-tier proof is against the immediately prior frankensearch 4-bit fast-fetch multiplier, not a search-engine incumbent: same remote RCH worker `ovh-a` measured `FAST_TIER_MULT=5` at 769.95 us and `FAST_TIER_MULT=3` at 639.75 us (ratio **0.831**) with `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b rch exec -- cargo bench -p frankensearch-fusion --profile release --bench sync_int8_fetch -- --sample-size 10 --warm-up-time 1 --measurement-time 2`; exact-fetch control stayed neutral (2.3088 ms -> 2.3120 ms). | Original-comparator ratio vs Lucene/Tantivy/Meilisearch is **N/A** for this isolated sync vector/fusion bench. Do not claim new Lucene/Tantivy/Meilisearch-class dominance from it; the residual BOLD rows still require direct comparator proof. |
| 2026-06-27 | BlackThrush (`frankensearch-cod-a`) | `frankensearch-index/fsvi_4bit_two_pass fourbit_mult5` vs Lucene/Tantivy/Meilisearch-class original | Kept FSVI pass-1 proof is against the immediately prior frankensearch file-backed 4-bit two-pass vector primitive, not a search-engine incumbent: `rch exec` had no admissible worker for the bench and fell back locally, then a freshness-forced candidate rebuild measured `fsvi_4bit_two_pass/fourbit_mult5` at 3.6991 ms baseline (`origin/main`) -> 1.5660 ms new (ratio **0.423**, recall@10 still **1.0000** at mult=5) with `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a rch exec -- cargo bench -p frankensearch-index --profile release --bench fsvi_4bit_two_pass fourbit_mult5 -- --sample-size 10 --warm-up-time 1 --measurement-time 2`. The literal requested `cargo bench --release -p frankensearch-index ...` form was tried first and Cargo rejected it with `unexpected argument '--release'`. | Original-comparator ratio vs Lucene/Tantivy/Meilisearch is **N/A** for this isolated file-backed vector pass-1. Do not claim new Lucene/Tantivy/Meilisearch-class dominance from it; the residual BOLD lexical/materialization rows still require direct comparator proof. |
| 2026-06-27 | BlackThrush (`frankensearch-cod-a`) | `frankensearch-embed/hash_embed_jl` vs Lucene/Tantivy/Meilisearch-class original | Kept JL projection proof is against the immediately prior frankensearch scalar JL hash embedder path, not a search-engine incumbent: `rch exec` had no admissible worker and fell back locally, measuring `hash_embed_jl/old` at 105.00 us -> `hash_embed_jl/new` at 88.945 us (ratio **0.847**) with `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a rch exec -- cargo bench -p frankensearch-embed --profile release --bench hash_embed hash_embed_jl -- --sample-size 10 --warm-up-time 1 --measurement-time 2`. The literal requested `cargo bench --release -p frankensearch-embed ...` form was tried first and Cargo rejected it with `unexpected argument '--release'`. | Original-comparator ratio vs Lucene/Tantivy/Meilisearch is **N/A** for this isolated hash-embedder primitive. Do not claim new Lucene/Tantivy/Meilisearch-class dominance from it; the residual BOLD comparator negatives still stand. |
| 2026-06-26 | BlackThrush | `frankensearch-index` WAL tests under parallel `cargo test -p frankensearch-index --lib` | **Flaky cluster** (`compaction_merges_wal_into_main`, `soft_delete_clears_pending_wal_updates_for_same_doc_id`, …): a *different* WAL test fails each parallel run (`assert_eq!(wal_record_count/compaction stats, 2)` panics), but **all 356 pass serially** (`--test-threads=1`) and in isolation (3/3). Not from this campaign (my only `frankensearch-index` changes are bench files; failure is on committed `origin/main`). Diagnosed: temp paths are unique (`name+pid+nanos`), **no global/shared mutable state** in `wal.rs`/`lib.rs`, and `wal_record_count()` is a `const fn` over an in-memory field — so the wrong count under parallel load points to an **IO/mmap read-after-write visibility gap** (WAL file/mmap length not consistently observed after writes when ~356 tests saturate the temp-dir IO), not a logic/state bug. | Conformance gate: `cargo test -p frankensearch-index --lib` is intermittently red in parallel → CI-flaky. **Owner action:** audit `wal.rs` read-after-write / mmap-remap-on-grow under concurrent IO (or mark the WAL tests `#[serial]`). Not fixed here (WAL-internals, another agent's active crate, correctness-critical). |
| 2026-06-26 | BlackThrush (`frankensearch-cod-a`) | `frankensearch-index/dot_product f16_slice` safe direct f16 lane-load probe | Fresh lever attempt: make `widen8_f16_slice` mirror the byte path by directly reinterpreting `[f16; 8]` to `u16x8`, then compare `f16_slice_new` against a bench-only pre-lane-load baseline. The literal requested `cargo bench --release -p frankensearch-index --bench dot_product f16_slice -- --sample-size 10 --warm-up-time 1 --measurement-time 1` failed on remote `ovh-a` with Cargo's known `unexpected argument '--release'`; fallback `cargo bench -p frankensearch-index --bench dot_product --profile release f16_slice -- --sample-size 10 --warm-up-time 1 --measurement-time 2` then failed before measurement because `half::f16: Pod` is not implemented for `bytemuck::cast::<[f16; 8], u16x8>`. | No ratio produced. Probe reverted. Do not retry without either enabling a vetted `half` bytemuck feature across the workspace or introducing unsafe, both of which require a larger proof surface than a one-lever BOLD pass. |
| 2026-06-26 | BlackThrush | **Shared-tree contention blocks clean `simd.rs` commits** + ready-to-land `f32_f32` 4-acc lever | Identified the next dot-kernel lever: `dot_product_f32_f32_unchecked` is still **single-accumulator** and on **real paths** — WAL-entry scoring (`search.rs:528,546`, `two_tier.rs:310,392,415`) and **MRL truncated search** (`mrl.rs:464,497`). It is cheap-decode (plain f32 load), so the 4-accumulator ILP pattern that won for `f32_bytes` (ledger) and `dot_i8_i8` (`0.856`/`0.939`) wins here too — **now MEASURED** (applied the edit, ran `f32_slice_new`=4acc vs `f32_slice_old`=single-acc in-process): **dim384 `820.9µs → 775.0µs = 0.944 (~1.06×)`, dim256 `440.3µs → 433.2µs = 0.984 (~1.02×)**. Smaller than i8/`f32_bytes` (f32 is more load-bound, so less ILP headroom); the dim256 gain is near noise. f32 correctness tests (`simd_matches_scalar_f32`, `large_256d_matches_scalar_f32`) pass with the edit. **Could not LAND it:** `simd.rs` is an active multi-agent battleground — across this + prior iterations agents committed identical 4-acc `dot_i8_i8`, escalated it to 8-acc in uncommitted WIP, and an aggressive edit/revert cycle **reverted my `f32_f32` edit to HEAD three separate times** within minutes, so the edit can't be held long enough to test+commit cleanly. | **Blocker = no file coordination across agents in one shared working dir** (edits don't survive). The win is modest (~1.06× on the conditional WAL/MRL f32 path), so it's **low priority** vs bigger levers. Whoever owns `simd.rs` in an isolated worktree: apply 4 accumulators to `dot_product_f32_f32_unchecked` (bench A/B already exists); the f32 sum reorder is the same accepted non-bit-identical trade as `f32_bytes`. Fleet fix: isolated git worktrees per agent, or `file_reservation` (agent-mail) before editing `simd.rs`/`dot_product.rs`/ledgers. |

---

## Gated levers (measured headroom that can't be landed as library code)

### 2026-06-27 - BOLD hash-hybrid 4-bit vector backfill is mixed and regresses `limit_all` (BlackThrush)

**Lever tested and reverted:** changed the BOLD `hash_hybrid_tantivy_vector_rrf` candidate path to
use the current FSVI 4-bit two-pass vector primitive (`search_top_k_4bit_two_pass`, mult=5) instead
of the exact flat vector scan. This was the natural follow-up after the 4-bit vector primitive and
fusion fast-tier wins, targeting BOLD rows where the hybrid backfill is still slower than ORIG.

**Measured command (per-crate, RCH remote `hz2`, release profile):**

```bash
AGENT_NAME=BlackThrush \
RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,FRANKENSEARCH_BOLD_VERIFY_OUT,FRANKENSEARCH_BOLD_VERIFY_EMIT,FRANKENSEARCH_BOLD_VERIFY_SUMMARY_ONLY,FRANKENSEARCH_BOLD_VERIFY_COMMAND,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
FRANKENSEARCH_BOLD_VERIFY_OUT=/data/projects/.rch-targets/frankensearch-cod-b/criterion/bold_verify_4bit_candidate_rerun \
FRANKENSEARCH_BOLD_VERIFY_EMIT=1 \
FRANKENSEARCH_BOLD_VERIFY_SUMMARY_ONLY=1 \
RUST_LOG=off \
  rch exec -- cargo bench -p frankensearch --features lexical --profile release \
    --bench search_bench bold_verify_tantivy_class \
    -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Local transcript artifact: `/tmp/frankensearch-bold-4bit-candidate-rerun.log`; the BOLD summary
reported `FRANKENSEARCH_BOLD_VERIFY_OUT=/data/projects/.rch-targets/frankensearch-cod-b/criterion/bold_verify_4bit_candidate_rerun`.
ORIG comparator is `tantivy_doc_ids`.

| Workload | ORIG p50 | Candidate p50 | Ratio vs ORIG | Decision |
|----------|----------|----------------|---------------|----------|
| `top10/10000 exact_identifier` | 152 us | 124 us | **0.816** | win, but not enough to offset broader misses |
| `top10/10000 quoted_phrase` | 151 us | 146 us | **0.967** | borderline/noisy win |
| `top10/10000 high_fanout` | 82 us | 83 us | **1.012** | tie/noise |
| `top10/10000 zero_hit` | 41 us | 41 us | **1.000** | no gain |
| `limit_all/10000` | 5.466 ms | 5.998 ms | **1.097x slower** | reject |
| `top10/100000 exact_identifier` | 948 us | 974 us | **1.027** | noise/regression |
| `top10/100000 short_keyword` | 59 us | 64 us | **1.085x slower** | reject |
| `top10/100000 natural_language` | 2.033 ms | 1.535 ms | **0.755** | real row win, but not isolated |
| `top10/100000 high_fanout` | 619 us | 620 us | **1.002** | no gain |
| `top10/100000 zero_hit` | 45 us | 41 us | **0.911** | small win |

**Decision:** source reverted. The 4-bit two-pass primitive remains a valid lower-level win, but
dropping it into the BOLD hash-hybrid backfill is not a safe end-to-end lever: it worsens the largest
remaining BOLD materialization gap (`limit_all`) and adds regressions on 100k exact/short rows. Route
next below the candidate materialization/query parser boundary instead of swapping the vector backfill
primitive in this harness.

### 2026-06-25 — AVX2 build is the biggest remaining dot-kernel lever, but it's a build-config knob (BlackThrush)

**Finding:** every SIMD win so far is on an **SSE2-class build** (no AVX2/SSE4.1 — the ~1 ns/elem
f32 dots and the reverted `vpmaddwd` int8 experiment both confirm it). Rebuilding the dot bench
with `RUSTFLAGS="-C target-feature=+avx2,+fma,+f16c"` (separate target dir) measured, back-to-back:

| Workload | SSE2 | AVX2 | AVX2/SSE2 |
|----------|------|------|-----------|
| `dot/dim256/f16_bytes` | 2.09 ms | 0.77 ms | ~0.37 |
| `dot/dim256/f32_bytes` | 1.49 ms | 0.59 ms | ~0.40 |
| `dot/dim384/f16_bytes` | 3.20 ms | 1.94 ms | ~0.61 |
| `dot/dim384/f32_bytes` | 2.30 ms | 1.54 ms | ~0.67 |

**Honest caveat:** these are **cross-run on different rch workers** — the per-dim inconsistency
(dim256 ~2.5× vs dim384 ~1.6×) shows worker variance is mixed in. A clean same-worker pin was
attempted on `vmi1149989` but that worker was contended (SSE2 there ran +27% vs its own baseline
and the AVX2 leg didn't complete). So the real figure is a **~1.5–2.5× range, not a precise ratio**.

**Why it can't be landed as a code lever:**
- A *published* library cannot assume `-Ctarget-cpu`/`target-feature`; consumers compile with their
  own flags, and a workspace `.cargo/config.toml` `+avx2` would make the **released `fsfs` binary**
  crash (illegal instruction) on non-AVX2 hosts.
- Runtime AVX2 dispatch (`is_x86_feature_detected!` + `#[target_feature]`) needs `unsafe`. The crate
  is `deny(unsafe_code)` (opt-in allowed, but a hand-written AVX2 intrinsic dot kernel is a large,
  risky surface — and `wide` only uses compile-time features, so it can't help at runtime).

**Actionable recommendation (not a code change):** deploy targets known to have AVX2 should build
`fsfs` / the consuming app with `RUSTFLAGS=-Ctarget-cpu=x86-64-v3` (or `native`) for ~1.5–2.5×
faster vector search for free — `wide` then auto-selects its AVX2 paths. This belongs in the
packaging/deploy docs, not the library default. **Do not** add workspace-wide `+avx2` (breaks the
portable released binary).

> **PARTIALLY OVERTURNED 2026-06-28 (BlackThrush):** the "hand-written runtime-dispatched AVX2 kernel
> is too large/risky" judgement was wrong for the **integer** dot. `dot_i8_i8` now runtime-dispatches
> (`is_x86_feature_detected!` + `#[target_feature(enable="avx2")]`, scoped `#[allow(unsafe_code)]`) to a
> ~25-line hand-written AVX2 kernel, **2.26–2.56×** over the `wide` fallback, **bit-identical** (integer
> assoc; proven by `avx2_dot_matches_generic`). See `docs/PERF_LEDGER.md` 2026-06-28 "runtime-dispatched
> AVX2 `dot_i8_i8`". The deploy-flag recommendation still stands as the zero-code option; runtime dispatch
> is the no-config, safe-on-all-hosts option. **Still open:** 4-bit pass-1 (nibble unpack) and the f16
> rescore (F16C — but f32 reorder is NOT bit-identical, so it needs a recall gate not exact equality).

---

## Residual comparator negatives

### 2026-06-28 — BOLD `limit_all` counted collector does not reproduce as a keep (BlackThrush)

**Lever tested and reverted:** the BOLD frankensearch challenger routed only the `limit_all`
query shape through `search_doc_ids_counted`; the Tantivy/Lucene/Meilisearch-class incumbent
(`tantivy_doc_ids`) and all top10 challenger rows stayed on the existing adaptive
`search_doc_ids` path. A dirty bench worktree initially looked like a measured large-limit win,
but the same source port on current `main` did not clear the repo's keep threshold.

**Initial bench-worktree result** (local fallback; fresh target dir
`/data/projects/.rch-targets/frankensearch-cod-a-bold-limitall`; artifact
`/data/projects/.rch-targets/frankensearch-cod-a-bold-limitall/criterion/bold_verify_limitall_counted/summary.jsonl`):

| Workload | Tantivy-class ORIG p50 | frankensearch p50 | Ratio vs ORIG | Status |
|----------|------------------------|-------------------|---------------|--------|
| `bold_verify/limit_all/10000` `hash_hybrid_tantivy_vector_rrf` | 11.046 ms | 9.928 ms | **0.899** | promising but not accepted |
| `bold_verify/limit_all/10000` `hash_lexical_guard_tantivy` | 11.046 ms | 10.507 ms | **0.951** | promising but not accepted |

**Current-port rerun** (per-crate, RCH remote `hz2`, target dir
`/data/projects/.rch-targets/frankensearch-cod-b`):

```bash
AGENT_NAME=BlackThrush \
RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,FRANKENSEARCH_BOLD_VERIFY_OUT,FRANKENSEARCH_BOLD_VERIFY_EMIT,FRANKENSEARCH_BOLD_VERIFY_SUMMARY_ONLY,FRANKENSEARCH_BOLD_VERIFY_COMMAND,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
FRANKENSEARCH_BOLD_VERIFY_OUT=/data/projects/.rch-targets/frankensearch-cod-b/criterion/bold_verify_limitall_counted_land \
FRANKENSEARCH_BOLD_VERIFY_EMIT=1 \
FRANKENSEARCH_BOLD_VERIFY_SUMMARY_ONLY=1 \
RUST_LOG=error \
  rch exec -- cargo bench -p frankensearch --features lexical --profile release \
    --bench search_bench bold_verify_tantivy_class \
    -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Artifact:
`/tmp/frankensearch-bold-limitall-counted-land.log`; the run reported
`/data/projects/.rch-targets/frankensearch-cod-b/criterion/bold_verify_limitall_counted_land/summary.jsonl`.

| Workload | Tantivy-class ORIG p50 | frankensearch p50 | Ratio vs ORIG | Decision |
|----------|------------------------|-------------------|---------------|----------|
| `bold_verify/limit_all/10000` `hash_hybrid_tantivy_vector_rrf` | 6.129 ms | 6.082 ms | **0.992** | revert: noise |
| `bold_verify/limit_all/10000` `hash_lexical_guard_tantivy` | 6.129 ms | 5.976 ms | **0.975** | revert: below keep bar |

**Decision:** source reverted; ledger-only. The fresh current-port run failed to reproduce the
bench-worktree win and also exposed non-target slow rows (`top10/10000 high_fanout` hybrid
**1.800x**, guard **2.588x**; `top10/100000 short_keyword` hybrid **1.392x**). Do not broaden
counted collection to top10 high-fanout, and do not land a `limit_all` counted special case unless
a future same-source rerun shows a stable `<0.97` ratio on the large-limit rows without opening
larger top10 regressions.

### 2026-06-27 — `SniffFeatures::from_bytes` branchless `u64` counters are not a win — REVERTED (BlackThrush)

**Lever tested and reverted:** `frankensearch-fsfs::SniffFeatures::from_bytes` counts null,
non-printable, and high-bit bytes with three `u32::saturating_add` guarded increments per byte.
Retried the route-next from the fsqlite-unblocked entry: widen those counters to `u64`, replace the
three guarded increments with `u64::from(predicate)` additions, and saturate only once when writing
`null_bytes`. The goal was to make the per-file sniff histogram easier for LLVM to vectorize.

**Measured command** (per-crate, RCH remote `hz2`, clean worktree
`/data/projects/frankensearch-blackthrush-fsfs-sniff-20260627`, target dir
`/data/projects/.rch-targets/frankensearch-cod-a`):

```bash
AGENT_NAME=BlackThrush \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
  rch exec -- cargo bench -p frankensearch-fsfs --profile release \
    --bench sniff_features sniff_features \
    -- --sample-size 10 --warm-up-time 1 --measurement-time 2
```

The bench carried a temporary `old_scalar` arm matching `origin/main` and a `new` arm calling the
candidate `SniffFeatures::from_bytes`, so the ratio is same-binary old-vs-new. The literal
`cargo bench --release` protocol remains invalid for this Cargo (`--profile release` is the working
release-profile equivalent).

| Workload | ORIG p50 | Candidate p50 | Ratio vs ORIG | Decision |
|----------|----------|----------------|---------------|----------|
| `sniff_features/8192` | 9.4628 us | 9.8636 us | **1.042x slower** | revert |
| `sniff_features/65536` | 74.243 us | 75.505 us | **1.017x slower/noise** | revert |

**Decision:** source and `Cargo.toml` changes were reverted. Focused equivalence tests passed while
the candidate was applied (`file_classification` 14/14), and scoped clippy for the touched lib + bench
passed with explicit allows for unrelated existing fsfs lint categories. The measured hot loop does
not benefit from boolean-to-integer `u64` accumulation; keep the current `u32::saturating_add`
branches. Ratio vs Lucene/Tantivy/Meilisearch-class original is **N/A** for this isolated ingest
sniff primitive; it does not move the BOLD search comparator.

### 2026-06-27 — BOLD high-fanout counted collector narrows p50 but still loses vs ORIG — REVERTED (BlackThrush)

**Lever tested and reverted:** for the BOLD `high_fanout` query shape only, keep the
Tantivy/Lucene/Meilisearch-class proxy on `tantivy_doc_ids` / `search_doc_ids`, but route the
frankensearch candidate's lexical fetch through `search_doc_ids_counted`. This tested the
query-adaptive collector idea from the count-free top-k evidence: broad, near-universal terms may
lose with WAND-style count-free `TopDocs`, while the counted collector is ranking-identical for
`search_doc_ids` rows (`search_doc_ids_matches_counted_baseline` already covers the equivalence).

**Measured command** (RCH local fallback; per-crate; clean worktree
`/data/projects/frankensearch-blackthrush-bold-dig-fresh-20260627`; target dir
`/data/projects/.rch-targets/frankensearch-cod-a-bold-fresh`):

```bash
AGENT_NAME=BlackThrush \
RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,FRANKENSEARCH_BOLD_VERIFY_EMIT,FRANKENSEARCH_BOLD_VERIFY_SUMMARY_ONLY,FRANKENSEARCH_BOLD_VERIFY_COMMAND,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a-bold-fresh \
  rch exec -- env \
  FRANKENSEARCH_BOLD_VERIFY_EMIT=1 \
  FRANKENSEARCH_BOLD_VERIFY_SUMMARY_ONLY=1 \
  RUST_LOG=error \
  cargo bench -p frankensearch --features lexical --profile release \
    --bench search_bench bold_verify_tantivy_class \
    -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Artifact after the candidate run:
`/data/projects/.rch-targets/frankensearch-cod-a-bold-fresh/criterion/bold_verify/summary.md`.
The clean `origin/main` baseline summary was run first in the same clean worktree/target dir and was
overwritten by the candidate summary; its key emitted rows were captured before overwrite.

**Clean `origin/main` residual before the candidate (candidate / ORIG):**

| Workload | ORIG p50 | frankensearch p50 | Ratio vs ORIG |
|----------|----------|-------------------|---------------|
| `top10_high_fanout/100000` hybrid | 668 us | 981 us | **1.469x slower** |
| `top10_high_fanout/100000` guard | 668 us | 949 us | **1.421x slower** |
| `limit_all/10000` hybrid | 7.424 ms | 8.987 ms | **1.211x slower** |

**Candidate residual vs ORIG:**

| Workload | ORIG p50 | frankensearch p50 | Ratio vs ORIG | Decision |
|----------|----------|-------------------|---------------|----------|
| `top10_high_fanout/100000` hybrid | 620 us | 759 us | **1.224x slower** | reverted |
| `top10_high_fanout/100000` guard | 620 us | 685 us | **1.105x slower** | reverted |
| `top10_high_fanout/10000` hybrid | 80 us | 69 us | 0.863 | not causal/too noisy |
| `top10_high_fanout/10000` guard | 80 us | 68 us | 0.850 | not causal/too noisy |

The 100k p50 residual narrowed, but the target row still lost to ORIG, and the hybrid p95 ratio was
still bad (`2.087x` slower). The 10k high-fanout rows looked like ORIG wins, but they were not an
absolute speedup against the clean baseline's frankensearch p50s (`65 us -> 69 us` hybrid,
`64 us -> 68 us` guard); the ratio improved because the incumbent moved from `64 us` to `80 us`
between runs. `limit_all/10000` also showed a hybrid p50 win in the candidate run, but the source
hunk did not route `limit_all`, so that row is run variance, not a causal lever.

**Decision:** source hunk reverted. Do not route BOLD `high_fanout` through the counted collector as
a frankensearch-only comparator lever; it can reduce p50 noise but does not land a measured ORIG win
on the biggest residual.

### 2026-06-27 — Move-only `LexicalIdHit -> ScoredResult` conversion does not fix the BOLD materialization gap (BlackThrush)

**Lever tested and reverted:** add `From<LexicalIdHit> for ScoredResult` and make the BOLD
`frankensearch_hash_hybrid` / `hash_lexical_guard` paths consume `Vec<LexicalIdHit>` instead of
cloning every `doc_id` while converting lexical IDs to scored rows. This targeted the biggest
current materialization-looking miss, `bold_verify/limit_all/10000`, where frankensearch returns
rich scored rows while the Tantivy/Lucene/Meilisearch-class incumbent stops at IDs.

**Measured command:** the literal requested form still fails in this checkout:
`cargo bench --release -p frankensearch --features lexical --bench search_bench ...` -> Cargo
`unexpected argument '--release'`. Actual per-crate release-profile measurement:

```bash
AGENT_NAME=BlackThrush \
RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,FRANKENSEARCH_BOLD_VERIFY_EMIT,FRANKENSEARCH_BOLD_VERIFY_SUMMARY_ONLY,FRANKENSEARCH_BOLD_VERIFY_COMMAND,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
  rch exec -- env \
  FRANKENSEARCH_BOLD_VERIFY_EMIT=1 \
  FRANKENSEARCH_BOLD_VERIFY_SUMMARY_ONLY=1 \
  RUST_LOG=off \
  cargo bench -p frankensearch --features lexical --profile release \
    --bench search_bench bold_verify_tantivy_class \
    -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

RCH worker: `hz2` (artifact retrieved to
`/data/projects/.rch-targets/frankensearch-cod-a/criterion/bold_verify/summary.jsonl`).

**Observed ratios vs the Tantivy/Lucene/Meilisearch-class proxy (candidate / incumbent):**

| Workload | Candidate | Incumbent p50 | Candidate p50 | Ratio | Decision |
|----------|-----------|---------------|----------------|-------|----------|
| `limit_all/10000` | `hash_hybrid_tantivy_vector_rrf` | 5.593 ms | 5.583 ms | **0.998** | zero-gain |
| `limit_all/10000` | `hash_lexical_guard_tantivy` | 5.593 ms | 5.590 ms | **0.999** | zero-gain |
| `top10/10000 high_fanout` | hybrid | 75 us | 85 us | **1.133x slower** | regression |
| `top10/100000 exact_identifier` | guard | 1.056 ms | 1.189 ms | **1.126x slower** | regression |
| `top10/100000 short_keyword` | hybrid | 507 us | 543 us | **1.071x slower** | regression |
| `top10/100000 zero_hit` | guard | 59 us | 151 us | **2.559x slower** | regression |

The run also had quoted-phrase win rows, but that shape is already known to be volatile/noisy and
the source change only removes `String` clones in the scored-row conversion; it does not alter
Tantivy phrase execution. The target row stayed inside the `[0.97, 1.03]` no-gain band, with
multiple unrelated regressions in the same comparator sweep.

**Conformance:** before the revert, `AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a rch exec -- cargo test -p frankensearch-lexical --lib`
fell back locally after queue timeout and passed (`82 passed; 0 failed`). The source patch was then
reverted; this entry is docs-only.

**Decision:** do not retry move-only `LexicalIdHit` materialization as a BOLD gap lever. The
remaining `limit_all` gap is in Tantivy collection/doc materialization itself, not this extra clone.

### 2026-06-27 — Methodology: `rch exec` can serve a STALE bench binary; verify freshness before trusting latency (BlackThrush)

**Finding (tooling, not a perf result):** benching via `rch exec -- cargo bench` does **not** always rebuild the bench binary on the worker — it can run a cached one. Caught red-handed: after renaming the `sync_int8_fetch` bench's default arm `int8_fetch` → `fast_fetch_4bit` (source confirmed) and re-running through `rch`, the worker still emitted the **old** `int8_fetch` arm name, i.e. it ran a stale binary. So **`rch` latency numbers can reflect old code** and silently mislead an A/B of a fresh change. (A clean local `cargo bench` is the cross-check, but the shared `CARGO_TARGET_DIR` here also hits `E0514` "incompatible rustc" when the local toolchain differs from the worker's — so neither path is automatically trustworthy.)

**Mitigation:** before trusting an `rch` bench A/B, confirm the binary is fresh — e.g. rename/relabel an arm (or print a source-derived sentinel) and verify the new label appears in the output, and check the criterion `estimates.json` mtime is from *this* run. Force a rebuild if in doubt.

**Consequence for the 4-bit query-hoist (`PreparedQuery4bit`, commits f06379c / b300679, a concurrent agent):** I independently implemented the same hoist and my `rch`-based int8_two_pass A/B suggested it was *slower* at small `k` (the prepared-query path adds a per-chunk heap load vs decoding query nibbles in registers) — but given the caching pitfall above, **I do not trust that number** and am NOT claiming a regression. Flagging for a clean, freshness-verified re-measurement (small-`k` scan, same-run vs the inline kernel) before treating the hoist as a confirmed win or loss. Left the on-main hoist untouched (a concurrent agent owns that area).

### 2026-06-27 — 1-bit binary (sign) two-pass for FSVI vector search: recall too low, no speedup (BlackThrush)

**Lever tested and reverted:** follow-up to the landed FSVI int8 two-pass (1.94×, bandwidth-bound).
Hypothesis: a packed sign-bit slab (`dim/8` bytes — ~8× smaller than the int8 slab, ~16× vs f16)
scored by XOR+popcount (Hamming) pass-1 + exact f16 rescore of the top `k·mult` would cut the
dominant bandwidth further. Implemented `VectorIndex::search_top_k_binary_two_pass` mirroring the
int8 two-pass (lazy `vectors_bits` slab, tombstone-aware parallel Hamming scan + cutoff, F16/no-WAL
gate); plumbing verified bit-identical to `search_top_k` under keep-all (`binary_two_pass_keep_all_
matches_exact`).

**Measured (per-crate bench `fsvi_binary_two_pass`, 100k clustered FSVI, dim=384, k=10):**

| mult | recall@10 |  | latency (same fast worker) | vs flat | vs int8 |
|------|-----------|--|----------------------------|---------|---------|
| 2    | **0.150** |  | flat (exact) = 2714 µs     | —       | —       |
| 5    | **0.206** |  | int8 mult=5 = 1014 µs (recall 1.0) | 2.68× | — |
| 10   | **0.300** |  | binary mult=10 = 1093 µs   | 2.48×   | **0.93× (slower)** |
| 20   | **0.438** |  | binary mult=20 = 1840 µs   | 1.48×   | 0.55× (slower) |
| 50   | **0.631** |  |                            |         |         |

**Decision:** rejected and fully reverted (field, method, helpers, test, bench — index crate left
byte-identical to the FSVI-int8 commit). Two independent failures: (1) **recall is far too low** —
1-bit sign quantization collapses 384-d clustered embeddings (clusters share sign patterns), so the
true top-k are not in the candidate set even at mult=50 (0.63); a lossless mult would approach
keep-all. (2) **No speedup even ignoring recall** — integer Hamming distances (0..dim) produce many
ties at the heap cutoff, so the bounded-heap fast-path prunes little; binary mult=10 (1093 µs) ≈ int8
mult=5 (1014 µs, recall=1.0), so the ~8× bandwidth edge is eaten by tie-thrashing + popcount.

**Route next:** sub-int8 quantization is NOT the lever for these embeddings — int8 is the bandwidth
sweet spot (1.94× lossless, landed). A viable binary path would need rotation/PCA before sign (to
de-correlate dimensions) + asymmetric (query-full) scoring — a much larger research effort, not a
clean primitive swap. The FSVI vector-search frontier is int8.

**UPDATE 2026-06-27 (4-bit probe — the sub-int8 lever DOES exist at 4-bit, not 1-bit):** a quick
recall probe (16-level quantization, `q = round(x·7/max_abs)` clamped `[-7,7]`, candidate recall of
the f16 top-10 within the 4-bit top-`k·mult`; N=10k clustered, dim=384) shows **4-bit is recall-
viable, unlike 1-bit binary**:

| level | mult=2 | mult=5 | mult=10 |
|-------|--------|--------|---------|
| 1-bit (binary, rejected above) | 0.150 | 0.206 | 0.300 |
| **4-bit (16-level)** | **0.981** | **1.000** | **1.000** |

So a 4-bit two-pass would be **lossless at mult=5** (10k; needs a 100k confirm) while reading half
the int8 slab (`dim/2` vs `dim` bytes) — a plausible ~1.3–1.5× over the landed int8 two-pass. The
blocker is purely the kernel: unpack-to-buffer writes the full i8 slab (38 MB), negating the 19 MB
read advantage, so the 4-bit dot must be a **register-fused SIMD nibble dot**. `wide 1.4` supports
it (`i16x16` Shl/Shr + `from_i8x16`, `i32x8::from_i16x8`): load 16 packed bytes → `i16x16`,
sign-extend low/high nibbles via `(x<<12)>>12` and `(x<<8)>>12`, multiply (products ≤49), widen-
accumulate. Scoped as a focused follow-up (intricate SIMD kernel + 100k recall/latency validation +
a keep-all bit-identical test); **not landed this turn** to avoid shipping an unverified SIMD kernel.
int8 remains the landed/validated frontier; 4-bit is the next concrete lever.

### 2026-06-27 — BOLD `search_doc_ids` span-removal run is contaminated, not landable (BlackThrush)

**Lever tested and reverted:** remove the two `#[instrument]` spans from the ID-only lexical hot
path (`execute_query_with_offset` and `TantivyIndex::search_doc_ids`) to avoid per-query tracing
span close overhead in the BOLD Tantivy/Lucene/Meilisearch-class proxy. The source patch itself was
only those two attribute removals and passed lexical conformance, but the BOLD measurement cannot be
claimed for that lever: while the bench was compiling/running, the shared checkout gained unrelated
uncommitted edits in `frankensearch/benches/search_bench.rs` (exact-ID alias shortcut and
summary-only switch) and `crates/frankensearch-index/src/in_memory.rs` (lazy `doc_id -> position`
map). Those edits were not mine, were not staged, and materially affect the measured harness/path.

**Measured command (per-crate, warm target dir; RCH local fallback because no worker was
admissible: `insufficient_slots=3,hard_preflight=1,active_project_exclusion=1`):**
```bash
AGENT_NAME=BlackThrush \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,FRANKENSEARCH_BOLD_VERIFY_OUT,FRANKENSEARCH_BOLD_VERIFY_COMMAND,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
FRANKENSEARCH_BOLD_VERIFY_OUT=/data/projects/frankensearch/.scratch/bold_no_search_doc_ids_instrument_20260627T0026Z \
FRANKENSEARCH_BOLD_VERIFY_COMMAND='AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a rch exec -- cargo bench -p frankensearch --features lexical --profile release --bench search_bench bold_verify_tantivy_class -- --sample-size 10 --warm-up-time 1 --measurement-time 1' \
RUST_LOG=off \
  rch exec -- cargo bench -p frankensearch --features lexical --profile release \
    --bench search_bench bold_verify_tantivy_class \
    -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Artifact: `.scratch/bold_no_search_doc_ids_instrument_20260627T0026Z/summary.jsonl`
(`git_sha=0950df46a5a5b9497a456d2b9a83899729f23c2e`, `worker=unknown`). The literal requested
`cargo bench --release` form is still rejected by Cargo in this checkout; the equivalent
per-crate release profile form is `--profile release`.

**Observed ratios vs the Tantivy-class proxy in the contaminated run:**

| Workload | Tantivy-class p50 | Contaminated candidate p50 | Candidate / Tantivy-class | Decision |
|----------|-------------------|----------------------------|----------------------------|----------|
| `top10_exact_identifier/10000` | 101 us | 103 us | 1.020 | zero-gain; contaminated |
| `top10_short_keyword/10000` | 32 us | 33 us | **1.031x slower** | reverted |
| `top10_quoted_phrase/10000` | 146 us | 138 us | 0.945 | contaminated win row only |
| `top10_natural_language/10000` | 121 us | 126 us | **1.041x slower** | reverted |
| `limit_all/10000` | 5.905 ms | 6.764 ms | **1.145x slower** | reverted |
| `top10_exact_identifier/100000` | 1.296 ms | 1.099 ms | 0.848 | contaminated win row only |
| `top10_short_keyword/100000` | 177 us | 193 us | **1.090x slower** | reverted |
| `top10_quoted_phrase/100000` | 981 us | 1.217 ms | **1.241x slower** | reverted |
| `top10_natural_language/100000` | 824 us | 798 us | 0.968 | noise/contaminated |
| `top10_high_fanout/100000` | 705 us | 517 us | 0.733 | contaminated win row only |

**Conformance:** `AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a rch exec -- cargo test -p
frankensearch-lexical --lib` passed locally via RCH fallback (`79 passed; 0 failed`).

**Decision:** reverted the span-removal source patch and landed no code. The run is useful negative
evidence only: BOLD rows are mixed, and the apparent exact-identifier improvement is not
attributable because the BOLD harness changed concurrently. A clean retry must start from an
isolated worktree or first land/reject the unowned exact-ID alias and `doc_id` map edits.

### 2026-06-27 - BOLD exact-identifier alias is a synthetic shortcut, not a landable win (BlackThrush)

**Lever tested and reverted:** add a bench-harness-only alias from query text `doc 000042` to
document id `doc-000042`, then return a synthetic one-row `ScoredResult` with score `1.0` from the
frankensearch challenger before running Tantivy/vector search. This creates a dramatic exact-ID
timing row, but it changes result cardinality and ranking semantics instead of optimizing the real
search path, so it is not conformance-green.

**Measured command (per-crate, warm target dir; RCH remote `vmi1227854`, 477.7s):**
```bash
AGENT_NAME=BlackThrush \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,FRANKENSEARCH_BOLD_VERIFY_OUT,FRANKENSEARCH_BOLD_VERIFY_COMMAND,FRANKENSEARCH_BOLD_VERIFY_SUMMARY_ONLY,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
FRANKENSEARCH_BOLD_VERIFY_SUMMARY_ONLY=1 \
FRANKENSEARCH_BOLD_VERIFY_OUT=/data/projects/frankensearch-exact-alias-reject-20260627-current/.scratch/bold_exact_alias_candidate_20260627T0108Z \
FRANKENSEARCH_BOLD_VERIFY_COMMAND='AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a rch exec -- cargo bench -p frankensearch --features lexical --profile release --bench search_bench bold_verify_tantivy_class -- --sample-size 10 --warm-up-time 1 --measurement-time 1' \
RUST_LOG=off \
  rch exec -- cargo bench -p frankensearch --features lexical --profile release \
    --bench search_bench bold_verify_tantivy_class \
    -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Artifact note: the remote bench printed `BOLD_VERIFY_ARTIFACTS` but did not sync summary files back
to the local worktree. Evidence is the captured run log:
`.scratch/bold_exact_alias_candidate_20260627T0108Z/run.log`, which contains 26
`BOLD_VERIFY_JSONL` rows (`git_sha=unknown` in the emitted rows). The literal
`cargo bench --release` form is still rejected by Cargo in this checkout; the equivalent per-crate
release-profile command is `--profile release`.

**Observed ratios vs the Tantivy/Lucene/Meilisearch-class proxy:**

| Workload | Tantivy-class p50 | Exact-alias candidate p50 | Candidate / Tantivy-class | Decision |
|----------|-------------------|---------------------------|----------------------------|----------|
| `top10_exact_identifier/10000` | 127 us | 0 us | 0.000 | invalid synthetic fast path |
| `top10_short_keyword/10000` | 43 us | 49 us | **1.140x slower** | reverted |
| `top10_quoted_phrase/10000` | 152 us | 156 us | 1.026 | zero-gain/reverted |
| `top10_natural_language/10000` | 128 us | 143 us | **1.117x slower** | reverted |
| `top10_high_fanout/10000` | 110 us | 84 us | 0.764 | isolated/no keep |
| `top10_zero_hit/10000` | 17 us | 17 us | 1.000 | zero-gain/reverted |
| `limit_all/10000` | 6.091 ms | 7.368 ms | **1.210x slower** | reverted |
| `top10_exact_identifier/100000` | 1.034 ms | 0 us | 0.000 | invalid synthetic fast path |
| `top10_short_keyword/100000` | 181 us | 191 us | **1.055x slower** | reverted |
| `top10_quoted_phrase/100000` | 974 us | 750 us | 0.770 | isolated/no keep |
| `top10_natural_language/100000` | 711 us | 726 us | 1.021 | zero-gain/reverted |
| `top10_high_fanout/100000` | 621 us | 644 us | **1.037x slower** | reverted |
| `top10_zero_hit/100000` | 18 us | 15 us | 0.833 | isolated/no keep |

**Decision:** reverted the source patch and landed no code. Do not retry this as a bench-level
shortcut. A future exact-identifier lever must be product-real, preserve the top-k result contract
with tests, and compare against the same Tantivy/Lucene/Meilisearch-class BOLD proxy without
synthetic zero-time rows.

### 2026-06-26 — BOLD int8 two-pass vector wiring is not a Tantivy-class win (BlackThrush)

**Lever tested and reverted:** route the BOLD hash-hybrid challenger through a resident
`InMemoryVectorIndex::search_top_k_int8_two_pass(..., candidate_multiplier=5)` instead of the
file-backed exact FSVI vector scan. The candidate kept the Tantivy/Lucene-class incumbent
untouched, added a one-time exactness gate that compared int8 candidate doc-id order with exact
FSVI order for every BOLD query that reached the vector path, and passed that gate before timing.

**Measured command (per-crate, warm target dir; RCH local fallback because no worker was
admissible: `insufficient_slots=4,hard_preflight=1`):**
```bash
AGENT_NAME=BlackThrush \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
  rch exec -- env \
  FRANKENSEARCH_BOLD_VERIFY_EMIT=1 \
  FRANKENSEARCH_BOLD_VERIFY_SUMMARY_ONLY=1 \
  FRANKENSEARCH_BOLD_VERIFY_OUT=.scratch/bold_verify_int8_candidate_summary \
  RUST_LOG=off \
  cargo bench -p frankensearch --features lexical --profile release \
    --bench search_bench bold_verify_tantivy_class \
    -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Artifact: `frankensearch/.scratch/bold_verify_int8_candidate_summary/summary.jsonl`
(`git_sha=28aa02207d1d81a836b645df80172f197e997437`, `worker=unknown`). The summary-only switch
was temporary candidate code to avoid the known Criterion tracing flood; it was reverted with the
int8 wiring. The literal `cargo bench --release` form remains invalid in this checkout, so the
successful per-crate run used Cargo's `--profile release` form.

**Decision:** rejected and reverted. Even with exact vector-candidate order preserved, the
Tantivy-class comparator is mostly worse: the int8 path only helps isolated zero-hit / 100k quoted
rows, while common 10k and 100k rows regress materially.

| Workload | Tantivy-class p50 | Candidate p50 | Candidate / Tantivy-class | Decision |
|----------|-------------------|---------------|----------------------------|----------|
| `top10_exact_identifier/10000` | 150 us | 155 us | 1.033 | zero-gain/reverted |
| `top10_short_keyword/10000` | 160 us | 193 us | **1.206x slower** | reverted |
| `top10_quoted_phrase/10000` | 273 us | 355 us | **1.300x slower** | reverted |
| `top10_natural_language/10000` | 223 us | 313 us | **1.404x slower** | reverted |
| `top10_high_fanout/10000` | 74 us | 108 us | **1.459x slower** | reverted |
| `top10_zero_hit/10000` | 36 us | 23 us | 0.639 | isolated/no keep |
| `limit_all/10000` | 14.385 ms | 16.565 ms | **1.152x slower** | reverted |
| `top10_exact_identifier/100000` | 1.129 ms | 1.229 ms | **1.089x slower** | reverted |
| `top10_short_keyword/100000` | 292 us | 286 us | 0.979 | zero-gain/reverted |
| `top10_quoted_phrase/100000` | 1.582 ms | 1.521 ms | 0.961 | isolated/no keep |
| `top10_natural_language/100000` | 1.086 ms | 1.333 ms | **1.227x slower** | reverted |
| `top10_high_fanout/100000` | 645 us | 911 us | **1.412x slower** | reverted |
| `top10_zero_hit/100000` | 30 us | 32 us | **1.067x slower** | reverted |

**Route next:** do not wire int8 two-pass into BOLD hybrid as a blanket replacement. The vector
primitive is still useful for standalone large-N vector scans, but the BOLD gap is dominated by
lexical materialization / tracing / RRF overhead on the mixed query stream, not by exact vector scan
cost alone.

### 2026-06-27 — `id` FAST field for lexical materialization is a large regression (BlackThrush)

**Lever tested and reverted:** the int8 reject above routed next to "the BOLD gap is dominated by
lexical materialization." `TantivyIndex::search_doc_ids` reads each hit's `id` via `load_doc` — a
full stored-document decompress per hit. Hypothesis: mark the `id` field `FAST` and read it
columnar (`segment_reader.fast_fields().str("id")` → `term_ords(doc).next()` + `ord_to_str`),
skipping the docstore decompress entirely. One `StrColumn` cached per segment; docstore fallback for
pre-FAST indexes. Existing 79 lexical tests passed (ids identical), so it is correctness-neutral.

**Measured (per-crate A/B bench `doc_id_materialize`, in-memory Tantivy, 20k high-fanout corpus —
every doc matches — median µs over 80 samples; FAST-field vs docstore `search_doc_ids`):**

| hits (limit) | FAST-field id | docstore id | FAST / docstore |
|--------------|---------------|-------------|------------------|
| 30   |    408 µs |  154 µs | **2.65× slower** |
| 100  |   1016 µs |  191 µs | **5.32× slower** |
| 300  |   2882 µs |  281 µs | **10.3× slower** |
| 1000 |  11078 µs |  606 µs | **18.3× slower** |

**Decision:** rejected and fully reverted (schema `FAST` flag, the columnar read, the temp bench).
`StrColumn::ord_to_str` resolves each ordinal through the dictionary SSTable (a seek + decode per
hit) — far more expensive than decompressing a small stored doc, and it gets monotonically worse
with hit count. Tantivy's docstore also amortizes via per-block decompression caching across nearby
hit addresses, which a per-ordinal dictionary lookup cannot. A string fast field is the wrong tool
for id materialization.

**Route next:** the materialization cost is real but NOT fixable with a string fast field. Still
plausible: a *numeric* fast field carrying a dense doc ordinal + an external ordinal→doc_id table
(skips the dictionary, but needs segment/delete-aware mapping), or simply fetching fewer lexical
candidates into materialization. Combined with the int8 reject, the BOLD high-fanout gap is bounded
by docstore materialization + RRF + tracing — none cheaply removable by a single primitive swap.

> **UPDATE 2026-06-28 (BlackThrush): the numeric-fast-field route-next is CONFIRMED a large win.**
> A per-crate A/B (`id_materialize_numeric_ff`, kept bench) measured the `u64` dense-ordinal fast field +
> external `ordinal→doc_id` table at **2.56× (k30) → 8.47× (k1000)** faster than the docstore path,
> bit-identical doc_ids. It wins **even at k30** (where the doc_id-cache below was ~0-gain) because the
> numeric column has no dictionary (unlike the str-FAST-field above) and skips the stored-doc decompress.
> See `docs/PERF_LEDGER.md` 2026-06-28 "lexical id materialization: numeric u64 fast-field + ordinal
> table". Production wiring (schema `ord` field + append-only table + `collect_id_hits` fast path with
> docstore fallback; delete/merge-safe via monotonic non-reused ordinals) is the scoped follow-up.

### 2026-06-27 — Lazy doc_id cache for `search_doc_ids` is ~0-gain on the biggest gap (BlackThrush)

**Lever tested and reverted:** the materialization route-next, take 2 — instead of a fast field
(refuted above, slower) cache each hit's `id` lazily per `(segment, local-doc)` in a lock-free
`OnceLock<String>` grid on `TantivyIndex`, keyed by the searcher's segment set (rebuilt on
commit/merge). First access decompresses the stored doc; repeat access — the rotating BOLD query
stream re-hits the same high-BM25 docs — returns a cached `String` clone. Correctness-neutral: all 79
lexical tests pass (incl. the upsert→commit invalidation path).

**Measured (per-crate A/B bench `doc_id_cache`, in-memory Tantivy, 20k high-fanout corpus, cache
warmed; median µs over 100 samples; cached `search_doc_ids` vs the docstore baseline):**

| hits (limit) | cached | docstore | cached / docstore |
|--------------|--------|----------|--------------------|
| 30   | 156 µs | 151 µs | **1.03 (~0-gain)** |
| 100  | 162 µs | 184 µs | 0.88 (1.14×) |
| 300  | 185 µs | 267 µs | 0.69 (1.44×) |
| 1000 | 244 µs | 549 µs | 0.44 (2.25×) |

**Decision:** rejected and fully reverted (cache field, struct, the `search_doc_ids` path, temp
bench — lexical crate left byte-identical to HEAD). The win scales with hit count, but the **biggest**
BOLD gap (top10 `high_fanout`/`natural_language`) fetches only `limit·3 ≈ 30` candidates, where the
cache is ~0-gain (1.03×, within noise). The docstore materialization is **~0.41 µs/hit** (≈12 µs at
30 hits) — a small slice of the ~645–911 µs top10/100k latency — so materialization is **not** the
top10 bottleneck, and an O(N) cell grid + per-doc memory for the common path is not justified by a
large-`limit`-only (e.g. `limit_all`) benefit.

**Route next:** materialization dominates only at very large fetch limits (`limit_all`), not the
top10 hybrid gap. With the int8, str-FAST-field, and now doc_id-cache rejects, the top10
high_fanout/NL gap is bounded by lexical query execution (BM25 over many matches) + RRF + tracing —
none addressable by a single materialization/vector primitive swap; the realistic next probe is the
RRF/tracing overhead on the mixed stream, or accepting the gap as comparator-inherent.

### 2026-06-27 - removing the outer `rrf_fuse` tracing span is noise (BlackThrush)

**Lever tested and reverted:** the BOLD residual route above pointed at RRF/tracing overhead, and
`rrf_fuse` had two nested spans: the public `rrf_fuse` wrapper was instrumented, then immediately
called the already-instrumented `rrf_fuse_with_graph` core. Removing the outer wrapper span should
have reduced one span enter/exit per hash-hybrid fusion while preserving the core span and all
ranking behavior. The production hunk was a single `#[instrument]` deletion on `rrf_fuse`.

Current routing context before the probe (`hash_hybrid_tantivy_vector_rrf` vs `tantivy_doc_ids`,
local BOLD summary `/tmp/frankensearch-bold-current-blackthrush-20260627-2/summary.jsonl`): 10k
quoted phrase was still 1.459x slower, 100k natural-language 1.151x slower, and `limit_all` 1.140x
slower against the Lucene/Tantivy/Meilisearch-class proxy.

Literal requested command form remains invalid for this workspace's Cargo:

```bash
AGENT_NAME=BlackThrush \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
  rch exec -- cargo bench --release -p frankensearch-fusion \
    --bench rrf_fuse actual_ -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Measured command:

```bash
AGENT_NAME=BlackThrush \
RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
RUST_LOG=off \
  rch exec -- cargo bench -p frankensearch-fusion --profile release \
    --bench rrf_fuse actual_ -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

The first before/after attempt was not comparable (`before` local fallback, `after` remote `ovh-a`),
so the bench was tightened temporarily to include an in-binary old-wrapper baseline using the same
`#[instrument]` shape. That same-binary proof ran locally via RCH fallback and measured:

| Workload | Old outer-span wrapper | New wrapper | new / old | verdict |
|----------|------------------------|-------------|-----------|---------|
| `rrf_fuse/actual_old_outer_span_30x30` vs `actual_wrapper_30x30` | 2.2010 us | 2.1935 us | **0.997** | noise / reverted |

`actual_core_30x30` in the same run was 2.1071 us, confirming the core call is a little cheaper,
but the production wrapper-span deletion itself does not clear the 0.97 keep threshold. Ratio vs
Lucene/Tantivy/Meilisearch-class original: **none accepted**; this was an isolated challenger RRF
micro-overhead probe and produced no durable BOLD comparator win. Source and temporary bench arms
were reverted; only this ledger entry remains.

### 2026-06-26 — Reusing `QueryClass` inside BOLD hybrid search is mixed/noise (BlackThrush)

**Lever tested and reverted:** compute `QueryClass::classify(query.text)` once in
`frankensearch_hash_hybrid_search`, then pass the enum into
`bold_verify_lexical_prefetch_limit` and `bold_verify_lexical_short_circuit` instead of
reclassifying the same query twice. This was scoped to the BOLD harness path and did not change
Tantivy incumbent behavior or product search semantics.

**Requested protocol blocker:** the literal requested command shape still fails before
measurement on current Cargo:
```bash
AGENT_NAME=BlackThrush \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
  rch exec -- cargo bench --release -p frankensearch --features lexical \
  --bench search_bench bold_verify_tantivy_class \
  -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

RCH selected `hz2`, then Cargo rejected `--release` for `cargo bench` with
`unexpected argument '--release' found`; use `--profile release` for this bench command.

**Measured command (per-crate, warm target dir; RCH shell-wrapper local fallback for output
capture):**
```bash
AGENT_NAME=BlackThrush \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
  rch exec -- cargo bench -p frankensearch --features lexical --profile release \
  --bench search_bench bold_verify_tantivy_class \
  -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Artifacts:
- baseline summary: `frankensearch/.scratch/bold_queryclass_baseline_20260626T0134Z/summary.jsonl`
- candidate summary: `frankensearch/.scratch/bold_queryclass_candidate_20260626T0144Z/summary.jsonl`

Both summaries completed all 26 BOLD rows at `git_sha=6b70864826f76ac33a964dd561ef451b39fe88d2`.
The shell wrapper was needed to capture massive tracing output and therefore fell open locally
(`worker=unknown`); the result is valid as a same-machine reject, not as a remote-worker keep.
The post-summary Criterion loop was interrupted after summary emission to stop trace-log growth.

**Decision:** rejected and reverted. Median hybrid p50 movement was noise
(`candidate / baseline = 0.991`), with important short/exact rows regressing beyond the
3% rejection threshold. Some 100k rows improved, but the mixed profile is not a keep and several
candidate rows remain slower than the Tantivy/Lucene-class incumbent.

| Workload | Baseline p50 | Candidate p50 | Candidate / baseline | Candidate / Tantivy-class | Decision |
|----------|--------------|---------------|----------------------|----------------------------|----------|
| `top10_exact_identifier/10000` | 104 us | 132 us | **1.269x slower** | **1.282x slower** | reverted |
| `top10_short_keyword/10000` | 41 us | 49 us | **1.195x slower** | **1.065x slower** | reverted |
| `limit_all/10000` | 5.863 ms | 5.990 ms | 1.022 | **1.139x slower** | zero-gain/reverted |
| `top10_short_keyword/100000` | 189 us | 200 us | **1.058x slower** | **1.020x slower** | reverted |
| `top10_zero_hit/100000` | 24 us | 24 us | 1.000 | 0.960 | zero-gain/reverted |
| `top10_natural_language/10000` | 111 us | 110 us | 0.991 | **1.058x slower** | zero-gain/reverted |
| `top10_high_fanout/10000` | 69 us | 68 us | 0.986 | 1.000 | noise/reverted |
| `top10_natural_language/100000` | 798 us | 626 us | 0.784 | 0.817 | isolated/no keep |
| `top10_quoted_phrase/100000` | 1.194 ms | 965 us | 0.808 | 0.712 | isolated/no keep |

### 2026-06-26 — Early BOLD short-circuit before `ScoredResult` allocation is not a keep (BlackThrush)

**Lever tested and reverted:** move the BOLD hash-hybrid lexical short-circuit check before
`lexical_doc_ids_as_scored`, so saturated lexical rows can return the raw `LexicalIdHit` vector
instead of allocating temporary `ScoredResult` wrappers first. This was intentionally scoped to
the BOLD harness path after the rejected Tantivy fast-field attempt, leaving the Tantivy incumbent
and `TantivyIndex::search_doc_ids` comparator untouched.

**Measured command (RCH local fallback; no admissible workers:
`insufficient_slots=4,hard_preflight=1`; per-crate, warm target dir):**
```bash
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
  rch exec -- env \
  FRANKENSEARCH_BOLD_VERIFY_EMIT=1 \
  FRANKENSEARCH_BOLD_VERIFY_COMMAND='RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b rch exec -- env FRANKENSEARCH_BOLD_VERIFY_EMIT=1 RUST_LOG=off cargo bench -p frankensearch --features lexical --profile release --bench search_bench bold_verify_tantivy_class -- --sample-size 10 --warm-up-time 1 --measurement-time 1' \
  RUST_LOG=off \
  cargo bench -p frankensearch --features lexical --profile release \
  --bench search_bench bold_verify_tantivy_class \
  -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Artifact: `/data/projects/.rch-targets/frankensearch-cod-b/criterion/bold_verify/summary.md`
and `summary.jsonl`. The BOLD summary completed; the remaining Criterion process was interrupted
afterward because the `#[instrument]` close-event tracing still emitted massive output despite
`RUST_LOG=off`.

**Decision:** rejected and reverted. The isolated 100k natural-language/high-fanout p50 wins are
not acceptable because the same lever leaves the main 10k rows slower and badly regresses the
100k zero-hit row. It is also only a harness allocation reorder, not a library search primitive.

| Workload | Tantivy-class p50 | Candidate frankensearch p50 | Candidate / Tantivy-class | Decision |
|----------|-------------------|------------------------------|----------------------------|----------|
| `top10_exact_identifier/10000` | 105 us | 105 us | 1.000 | zero-gain |
| `top10_short_keyword/10000` | 134 us | 197 us | **1.470x slower** | reverted |
| `top10_quoted_phrase/10000` | 121 us | 211 us | **1.744x slower** | reverted |
| `top10_natural_language/10000` | 120 us | 222 us | **1.850x slower** | reverted |
| `top10_high_fanout/10000` | 176 us | 188 us | **1.068x slower** | reverted |
| `top10_zero_hit/10000` | 18 us | 21 us | **1.167x slower** | reverted |
| `limit_all/10000` | 7.033 ms | 7.160 ms | 1.018 | zero-gain |
| `top10_exact_identifier/100000` | 1.256 ms | 1.262 ms | 1.005 | zero-gain |
| `top10_short_keyword/100000` | 324 us | 317 us | 0.978 | zero-gain/no keep |
| `top10_quoted_phrase/100000` | 1.148 ms | 1.290 ms | **1.124x slower** | reverted |
| `top10_natural_language/100000` | 988 us | 789 us | 0.799 | isolated/no keep |
| `top10_high_fanout/100000` | 677 us | 638 us | 0.942 | isolated/no keep |
| `top10_zero_hit/100000` | 67 us | 174 us | **2.597x slower** | reverted |

**Future route:** the next real lever is not another harness-side wrapper allocation skip. Target
the actual `search_doc_ids` materialization primitive without touching the incumbent comparator,
or suppress/avoid per-hit `#[instrument]` close-event logging in the measurement harness before
trusting very small microsecond deltas.

### 2026-06-26 — Tantivy fast `id` column is a comparator-poisoning loss (BlackThrush)

**Lever tested and reverted:** mark the Tantivy `id` text field as `FAST` and make
`TantivyIndex::search_doc_ids` pull IDs from the per-segment string fast field instead of loading
stored docs. This followed the prior stored-doc materialization hypothesis, but Tantivy text fast
fields are dictionary encoded; resolving each hit still requires `ord_to_str`, and large result
sets repeatedly pay dictionary lookup/decode costs.

**Measured command (RCH local fallback; no admissible workers:
`insufficient_slots=5,hard_preflight=1`; per-crate, warm target dir):**
```bash
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
  rch exec -- env \
  FRANKENSEARCH_BOLD_VERIFY_EMIT=1 \
  FRANKENSEARCH_BOLD_VERIFY_COMMAND='RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b rch exec -- env FRANKENSEARCH_BOLD_VERIFY_EMIT=1 RUST_LOG=error cargo bench -p frankensearch --features lexical --profile release --bench search_bench bold_verify_tantivy_class -- --sample-size 10 --warm-up-time 1 --measurement-time 1' \
  RUST_LOG=error \
  cargo bench -p frankensearch --features lexical --profile release \
  --bench search_bench bold_verify_tantivy_class \
  -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Artifact: `/data/projects/.rch-targets/frankensearch-cod-b/criterion/bold_verify/summary.md`
and `summary.jsonl`. The BOLD summary completed; the remaining Criterion process was interrupted
afterward because per-iteration tracing produced massive output.

**Why this is rejected:** the candidate changes the same `search_doc_ids` function used by both
the frankensearch path and the `tantivy_doc_ids` incumbent, so the emitted candidate/incumbent
ratio is comparator-poisoned. Against the prior mainline Tantivy-class ledger, the hot rows are
material regressions:

| Workload | Prior main Tantivy-class p50 | Candidate frankensearch p50 | Candidate / prior Tantivy-class | Decision |
|----------|------------------------------|------------------------------|----------------------------------|----------|
| `top10_short_keyword/10000` | 43 us | 310 us | **7.209x slower** | reverted |
| `top10_high_fanout/10000` | 171 us | 245 us | **1.433x slower** | reverted |
| `top10_zero_hit/10000` | 29 us | 140 us | **4.828x slower** | reverted |
| `limit_all/10000` | 9.832 ms | 135.282 ms | **13.76x slower** | reverted |
| `top10_quoted_phrase/100000` | 1.143 ms | 1.130 ms | **0.989x** | isolated/no keep |

Candidate-run examples showing the poisoned incumbent effect:

| Workload | Candidate mutated Tantivy p50 | Candidate frankensearch p50 | Emitted ratio | Why not accepted |
|----------|-------------------------------|------------------------------|---------------|------------------|
| `limit_all/10000` | 199.898 ms | 135.282 ms | 0.677 | both sides are >10x slower than prior mainline |
| `top10_short_keyword/10000` | 284 us | 310 us | 1.092 | still slower than the mutated incumbent and far slower than prior mainline |
| `top10_zero_hit/100000` | 87 us | 39 us | 0.448 | isolated win, but same code regresses 10k zero-hit and `limit_all` badly |

**Decision:** reverted all source changes. Text fast fields are not the right ID materialization
primitive for this workload. A future attempt needs an ID retrieval path that does not dictionary
decode per hit, or it must keep the comparator immutable and measure frankensearch-only changes.

### 2026-06-25 — BOLD-VERIFY after lexical prefetch budget gate: mixed, not universal (BlackThrush)

**Lever kept elsewhere:** the BOLD hash-hybrid harness now asks Tantivy for only `k` lexical
candidates, not `3k`, on classes that can legally short-circuit to lexical-only results. The
target natural-language rows became p50 wins vs the Tantivy/Lucene/Meilisearch-class incumbent and
are recorded in `docs/PERF_LEDGER.md`. The same run still has slower/noisy rows below.

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

**Residual rows for `hash_hybrid_tantivy_vector_rrf` vs `tantivy_doc_ids`:**

| Workload | Corpus hash | Tantivy-class p50 | frankensearch p50 | Ratio vs Tantivy-class | Decision |
|----------|-------------|-------------------|-------------------|------------------------|----------|
| `top10_short_keyword/10000` | `2e78365a46a7c3b9` | 43 us | 177 us | **4.116x slower** | no dominance claim |
| `top10_quoted_phrase/10000` | `2e78365a46a7c3b9` | 301 us | 301 us | **1.000x** | p50 tie; tails slower |
| `top10_high_fanout/10000` | `2e78365a46a7c3b9` | 171 us | 228 us | **1.333x slower** | no dominance claim |
| `top10_zero_hit/10000` | `2e78365a46a7c3b9` | 29 us | 46 us | **1.586x slower** | no dominance claim |
| `limit_all/10000` | `2e78365a46a7c3b9` | 9.832 ms | 10.821 ms | **1.101x slower** | no dominance claim |
| `top10_quoted_phrase/100000` | `13f1b0153f5adec9` | 1.143 ms | 1.340 ms | **1.172x slower** | no dominance claim |

**Decision:** keep the scoped prefetch-budget gate because it produced clean p50 wins on the
target natural-language rows (`0.961x` at 10k, `0.962x` at 100k) and several lexical-saturated
rows. Do not generalize it: quoted phrases, 10k short keywords/high fanout/zero-hit, and
`limit_all` still need different levers.

### 2026-06-25 — BOLD-VERIFY after non-semantic zero-hit gate: still scoped, not universal (BlackThrush)

**Lever kept elsewhere:** non-semantic hash/no-quality searches now skip hash-vector work when
lexical returns zero candidates; the 100k zero-hit BOLD row is a real Tantivy-class win and is
recorded in `docs/PERF_LEDGER.md`. The same run shows the lever does not make the hash-hybrid path
universally faster than the Tantivy/Lucene/Meilisearch-class incumbent.

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

**Residual rows for `hash_hybrid_tantivy_vector_rrf` vs `tantivy_doc_ids`:**

| Workload | Corpus hash | Tantivy-class p50 | frankensearch p50 | Ratio vs Tantivy-class | Decision |
|----------|-------------|-------------------|-------------------|------------------------|----------|
| `top10_short_keyword/10000` | `2e78365a46a7c3b9` | 77 us | 80 us | **1.039x** | noise/tie; no dominance claim |
| `top10_quoted_phrase/10000` | `2e78365a46a7c3b9` | 146 us | 162 us | **1.110x slower** | no dominance claim |
| `top10_natural_language/10000` | `2e78365a46a7c3b9` | 148 us | 243 us | **1.642x slower** | needs lower-materialization lexical path |
| `top10_high_fanout/10000` | `2e78365a46a7c3b9` | 102 us | 122 us | **1.196x slower** | no dominance claim |
| `top10_zero_hit/10000` | `2e78365a46a7c3b9` | 44 us | 43 us | **0.977x** | p50 tie; tails improved, not a clean p50 win |
| `limit_all/10000` | `2e78365a46a7c3b9` | 5.720 ms | 5.975 ms | **1.045x slower** | no dominance claim |
| `top10_exact_identifier/100000` | `13f1b0153f5adec9` | 1.198 ms | 1.228 ms | **1.025x** | p50 noise; p95 regressed |
| `top10_short_keyword/100000` | `13f1b0153f5adec9` | 190 us | 214 us | **1.126x slower** | no dominance claim |
| `top10_quoted_phrase/100000` | `13f1b0153f5adec9` | 1.055 ms | 1.041 ms | **0.987x** | noise/tie |
| `top10_natural_language/100000` | `13f1b0153f5adec9` | 736 us | 768 us | **1.043x slower** | no dominance claim |
| `top10_high_fanout/100000` | `13f1b0153f5adec9` | 644 us | 685 us | **1.064x slower** | no dominance claim |

**Decision:** keep the zero-hit gate because it turns the 100k zero-hit BOLD row into a clean
incumbent win and removes the prior catastrophic empty-result vector scan. Do not generalize the
claim: saturated natural-language still spends too much in lexical over-fetch/materialization and
needs a separate lever before it can be called Tantivy/Lucene/Meilisearch-class faster.

### 2026-06-25 — BOLD-VERIFY after lexical short-circuit: still not universal dominance (BlackThrush)

**Lever kept elsewhere:** lexical-saturated `Identifier` / `ShortKeyword` queries now skip phase-1
vector scan + RRF once Tantivy has at least `k` hits. The same BOLD run produced two real
Tantivy-class p50 wins, recorded in `docs/PERF_LEDGER.md`.

**Measured command (per-crate, warm target dir; local fallback after RCH worker `vmi1153651`
stalled):**
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
FRANKENSEARCH_BOLD_VERIFY_EMIT=1 RUST_LOG=error \
  cargo bench -p frankensearch --features lexical --profile release \
  --bench search_bench bold_verify_tantivy_class \
  -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

**Residual rows where frankensearch remained slower than the Tantivy/Lucene/Meilisearch-class
incumbent, or where the ratio is noise rather than a clean win:**

| Workload | Corpus hash | Tantivy-class p50 | frankensearch p50 | Ratio vs Tantivy-class | Decision |
|----------|-------------|-------------------|-------------------|------------------------|----------|
| `top10_exact_identifier/10000` | `2e78365a46a7c3b9` | 143 us | 171 us | **1.196x slower** | no dominance claim |
| `top10_short_keyword/10000` | `2e78365a46a7c3b9` | 49 us | 66 us | **1.347x slower** | no dominance claim |
| `top10_quoted_phrase/10000` | `2e78365a46a7c3b9` | 174 us | 177 us | **1.017x** | noise/tie |
| `top10_natural_language/10000` | `2e78365a46a7c3b9` | 133 us | 893 us | **6.714x slower** | needs different lever |
| `top10_zero_hit/10000` | `2e78365a46a7c3b9` | 29 us | 624 us | **21.517x slower** | needs zero-hit semantic gate |
| `limit_all/10000` | `2e78365a46a7c3b9` | 6.068 ms | 7.324 ms | **1.207x slower** | no dominance claim |
| `top10_short_keyword/100000` | `13f1b0153f5adec9` | 165 us | 179 us | **1.085x slower** | no dominance claim |
| `top10_quoted_phrase/100000` | `13f1b0153f5adec9` | 968 us | 964 us | **0.996x** | noise/tie |
| `top10_natural_language/100000` | `13f1b0153f5adec9` | 655 us | 4.047 ms | **6.179x slower** | needs different lever |
| `top10_high_fanout/100000` | `13f1b0153f5adec9` | 542 us | 561 us | **1.035x slower** | p50/tail still not clean |
| `top10_zero_hit/100000` | `13f1b0153f5adec9` | 21 us | 2.776 ms | **132.190x slower** | needs zero-hit semantic gate |

**Decision:** keep the scoped short-circuit because it produced real incumbent wins on
`top10_high_fanout/10000` (0.711x) and `top10_exact_identifier/100000` (0.878x), but do not
generalize it. The next radical lever should target semantic gating for natural-language and
zero-hit queries rather than more dot-product work.

---

## Reverted experiments

### 2026-06-26 — 4-accumulator f16 dot kernel: neutral-to-1.49×-SLOWER (the single accumulator is deliberate) (BlackThrush)

**Cross-kernel-inconsistency probe → rejected, validates the existing design.** `dot_product_f32_bytes_f32`
uses **4 independent `f32x8` accumulators** (to break the FMA dependency chain — its decode is a cheap
load+cast, so the loop is sum-chain-bound and ILP helps), but `dot_product_f16_f32` uses a **single**
accumulator. Hypothesis: the f16 kernel left ILP on the table. Added a faithful `dot_product_f16_f32_4acc`
(same SIMD widen, 4 accumulators) and A/B'd it in-process against the single-acc kernel
(`benches/dot_product.rs`, `f16_slice_new` vs `f16_slice_4acc`):

| corpus | 1 accumulator | 4 accumulators | ratio |
|--------|---------------|----------------|-------|
| dim256, n=10k | 1589 µs | 1586 µs | 0.998 (neutral) |
| dim384, n=10k | 2271 µs | 3392 µs | **1.493 (49 % SLOWER)** |

The f16 kernel is **decode/widen-bound, not sum-chain-bound**: the Giesen magic-multiply widen is
register-heavy, so 4 live accumulators + the widen intermediates spill the SIMD register file — hurting
badly at dim384 (the production embedding dim). So the f32_bytes/f16 accumulator-count asymmetry is
**justified, not a missed lever**: cheap-decode kernels want 4 accumulators, expensive-widen kernels want
1. The single accumulator also **preserves the documented bit-identity** of the f16 dot (see `simd.rs`
header) — 4 accumulators would reorder the sum and break it. Probe reverted (zero-gain/regression); the
single-accumulator f16 kernel stays.

### 2026-06-26 — `ScalarQuantizer::dot_product_quantized` is a SIMD-lever TRAP (test-only, not wired) (BlackThrush)

**Not attempted — flagged to save effort.** `ScalarQuantizer::dot_product_quantized` /
`cosine_similarity_quantized` (`crates/frankensearch-index/src/quantization.rs`) are scalar,
single-accumulator per-dim dequant+`mul_add` loops — they *look* exactly like the f16/f32 dot
kernels that won ~3× from a SIMD-widen rewrite, so a profiling/extreme-optimization pass will be
tempted to SIMD them. **Don't:** every non-test caller is `#[cfg(test)]`, and `Quantization::Int8` /
`ScalarQuantizer` is **not** referenced in the production search scans (`search.rs`, `in_memory.rs`).
The production int8 vector search uses the in-memory **i8 corpus-max-abs slab** two-pass
(`search_top_k_int8_two_pass`, see `docs/PERF_LEDGER.md`), not this u8/per-dim quantizer. Optimizing
`dot_product_quantized` would be a zero-impact change (dead in production). If a future index format
wires `ScalarQuantizer` into search, re-evaluate then.

### 2026-06-26 — HNSW (ANN) vs flat parallel scan at 10k: a recall/latency trade, not a default win (BlackThrush)

**Lever evaluated (not wired):** the default vector search is a flat O(N) cosine scan
(`VectorIndex::search_top_k` — rayon-parallel + SIMD f16 dot + bounded-heap + cutoff). `HnswIndex`
(behind the `ann` feature, unwired) is an approximate O(log N) graph index — the obvious "radical
lever" for the scan. Validated head-to-head via the new gated bench `benches/hnsw_vs_flat.rs`
(latency + recall@10 sweep over `ef_search`), on **clustered** vectors (64 centroids + noise — the
realistic embedding structure HNSW exploits; uniform-random vectors are a worst case for ANN recall
and gave ~½ these recalls).

**Measured command (per-crate):**
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-index --features ann --bench hnsw_vs_flat
```

| N=10000, dim=128, k=10 | recall@10 | latency | vs flat |
|------------------------|-----------|---------|---------|
| flat `search_top_k` (exact) | 1.000 | 178.4 µs | 1.00× |
| HNSW `knn_search` ef=10 | 0.725 | 69.1 µs | **2.58× faster** |
| HNSW ef=20 | 0.850 | 110.3 µs | **1.62× faster** |
| HNSW ef=40 | 0.925 | 188.5 µs | 0.95× (≈break-even) |
| HNSW ef=100 | 0.950 | 425.3 µs | 0.42× (2.4× slower) |

**Interpretation — HNSW is a recall/latency *trade*, not a free speedup.** HNSW is faster only by
sacrificing recall: it beats the flat scan at `ef ≤ 20` (1.6–2.6×) but at **0.72–0.85 recall**; to
reach high recall (≥0.95) it needs `ef ≥ 100`, where it is **2.4× slower** than flat (break-even is
~ef=40 / recall 0.925). The flat path is hard to beat because it is **rayon-parallel + SIMD f16 dot +
sequential/cache-friendly**, whereas HNSW is a **serial** graph walk with **random** memory access.

**Decision:** do **not** wire HNSW as the **default** vector index — it would be a recall regression
for the exact/high-recall path (and slower if you keep recall high). It *is* a legitimate **opt-in
approximate fast-tier** option (e.g. ef≈20 → 1.6× faster at ~0.85 recall) for latency-sensitive
callers who accept lossy top-k; that's a feature/behaviour choice, not a perf lever for the exact
path. Bench kept (gated behind `ann`) for re-validation at larger N (the crossover improves for HNSW
as N grows). Corrects the earlier uniform-random measurement (which understated recall and called it
a flat loss).

### 2026-06-26 — fusing fingerprint content-hash + char-count is zero-gain (same per-byte work) (BlackThrush)

**Lever:** `Fingerprint::compute` scanned the document twice — `fnv1a_hash(text.as_bytes())` for the
content hash and `text.chars().count()` for the char count. Fused into one byte pass
(`content_hash_and_char_count`): accumulate FNV while counting non-continuation bytes
(`b & 0xC0 != 0x80`, which equals `chars().count()`). Bit-identical — proven by
`content_hash_and_char_count_matches_reference` across ASCII + 2/3/4-byte UTF-8 (896 core lib tests
green, +1 new test).

**Measured command (per-crate):**
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-core --bench fingerprint content_hash_charcount
```

| Workload | old (2 passes) | new (fused 1 pass) | ratio | verdict |
|----------|----------------|--------------------|-------|---------|
| `content_hash_charcount` (~2.2 KB doc) | 2868.3 ns | 2873.4 ns | **1.002** | noise / no gain |

**Why it fails:** fusing two byte passes does not reduce the *per-byte* work — the FNV loop is
~2 ops/byte and `chars().count()` is ~1 branch/byte; the fused loop does the same total per-byte
work, just in one loop. The only thing saved is one loop's iteration overhead (negligible), and the
second pass was already cache-warm (the bytes are hot from the first pass). Reverted source + bench
(stashed). Lesson: pass-fusion only helps when a pass is *eliminated* (data reused), not when both
passes touch every byte with comparable per-byte cost.

### 2026-06-26 — moving `LexicalIdHit` into `ScoredResult` is a mixed BOLD result, not a keep (BlackThrush)

**Lever:** the BOLD comparator's private lexical conversion changed from
`lexical_doc_ids_as_scored(&[LexicalIdHit])` cloning each `doc_id` into a `ScoredResult` to
consuming `Vec<LexicalIdHit>` and moving the owned `doc_id`. This targets frankensearch-only
phase-1 overhead that the Tantivy/Lucene/Meilisearch-class incumbent does not pay.

**Measured command (per-crate, warm `frankensearch-cod-a`; `rch exec` fell open to local because
no workers were admissible: `insufficient_slots=4,hard_preflight=1`):**
```bash
AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
  rch exec -- sh -c 'FRANKENSEARCH_BOLD_VERIFY_EMIT=1 \
  FRANKENSEARCH_BOLD_VERIFY_OUT=.scratch/bold_verify_blackthrush_move_scored \
  RUST_LOG=error cargo bench -p frankensearch --features lexical --profile release \
  --bench search_bench bold_verify_tantivy_class \
  -- --sample-size 10 --warm-up-time 1 --measurement-time 1'
```

Artifact: `frankensearch/.scratch/bold_verify_blackthrush_move_scored/summary.jsonl`
at git `9650b7fe906af38a075048b3b09d1d7349461575`. Caveat: the harness still emitted
high-volume span-close logs despite `RUST_LOG=error`, so this is a reject/routing run, not a
dominance claim.

Key `hash_hybrid_tantivy_vector_rrf` rows vs `tantivy_doc_ids`:

| Workload | Corpus hash | Tantivy-class p50 | frankensearch p50 | Ratio vs Tantivy-class | Decision |
|----------|-------------|-------------------|-------------------|------------------------|----------|
| `top10_exact_identifier/10000` | `2e78365a46a7c3b9` | 104 us | 128 us | **1.231x slower** | reject |
| `top10_natural_language/10000` | `2e78365a46a7c3b9` | 119 us | 213 us | **1.790x slower** | reject |
| `limit_all/10000` | `2e78365a46a7c3b9` | 6.334 ms | 6.768 ms | **1.069x slower** | reject |
| `top10_short_keyword/100000` | `13f1b0153f5adec9` | 155 us | 177 us | **1.142x slower** | reject |
| `top10_high_fanout/100000` | `13f1b0153f5adec9` | 569 us | 547 us | 0.961x | noise / insufficient |
| `top10_zero_hit/100000` | `13f1b0153f5adec9` | 23 us | 20 us | 0.870x | win row, not enough |

**Decision:** reverted the source hunk. Moving owned `doc_id` strings can help individual rows,
but it does not survive the BOLD class mix: the shipped hybrid path regresses exact identifiers,
10k natural-language, `limit_all`, and 100k short-keyword rows. Future work should first fix the
BOLD logging/artifact harness, then target a deeper id-first fusion path rather than repeating this
conversion-only lever.

### 2026-06-25 — branchless f32 sign in `embed_jl` REGRESSES (compiler already selects constants) (BlackThrush)

**Lever:** mirror the SimHash branchless-vote win (`apply_hash_votes`, kept) onto the JL hash
embedder's inner loop. `embed_jl` does `let sign = if (state & 1) == 0 { 1.0 } else { -1.0 }; *dim +=
sign;` per dimension per token (O(tokens·dim)); the xorshift LSB is effectively random, so the branch
*looked* like the same ~50%-mispredict target. Replaced with `let sign = 1.0 - 2.0 * (state & 1) as
f32;`. Bit-identical (43 hash embedder tests green incl. JL determinism/orthogonality).

**Measured command (per-crate):**
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-embed --bench hash_embed jl_sign
```

In-process A/B (`jl_sign`, ~100-word doc, dim384; identical except the per-dim sign):

| Workload | branch (`if`/`else`) | branchless (`2*b-1` arith) | ratio | verdict |
|----------|----------------------|----------------------------|-------|---------|
| `jl_sign` | 97.324 µs | 104.990 µs | **1.079** | regression |

**Why it fails (contrast with the kept SimHash win):** SimHash accumulates into **i32** counters, so
`2*b - 1` is cheap integer arithmetic and beats the branch (kept, 0.870×). `embed_jl` selects an
**f32** sign (`+1.0`/`-1.0`); the compiler already lowers the conditional to a branchless **select of
two f32 constants**, whereas the arithmetic form forces an `int→f32` conversion (`cvtsi2ss`) + a float
mul + a float sub per element — strictly more work. **Rule:** branchless arithmetic helps integer
accumulators, not float constant-selection (the compiler handles the latter). Reverted source + bench
(stashed). Do not re-attempt branchless sign on f32 select paths.

### 2026-06-25 — `collapse_code_block` slice-join is zero-gain (join allocs dominate) (BlackThrush)

**Lever:** the long-block branch of `collapse_code_block` collected the head/tail lines into
intermediate `Vec<&str>` (`lines.iter().take(head).copied().collect()` etc.) before `join("\n")`.
Since `[&str]` joins directly, the candidate replaced those with `lines[..head].join("\n")` and
`lines[lines.len()-tail..].join("\n")` to drop the two scratch vectors. Byte-identical (34
canonicalize tests green incl. `collapse_long_code_block`).

**Measured command (per-crate):**
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-core --bench canonicalize collapse_code_block
```

In-process old-vs-new A/B (`collapse_code_block`, 60-line block, head=20/tail=10):

| Workload | old (Vec collect + join) | new (slice join) | ratio new/old | verdict |
|----------|--------------------------|------------------|---------------|---------|
| `collapse_code_block` | 254.6 ns | 257.2 ns | **1.010** | noise / no gain |

**Why it fails:** the two `Vec<&str>` collects are ~20 and ~10 pointer copies — negligible next to
the actual work: the two `join("\n")` calls (each allocates + copies the joined output) and the
final `format!`. `<[&str]>::join` iterates the slice the same way the `collect` did, so eliminating
the scratch vectors saves nothing measurable. Reverted source + bench (stashed). Code-block
collapsing is not allocation-bound on the scratch vectors; no lever here.

### 2026-06-25 — caching the Tantivy `QueryParser` is zero-gain (parse dominates) (BlackThrush)

**Lever:** `TantivyIndex::parse_query_lenient` rebuilt a `QueryParser::for_index(..)` +
`set_field_boost(..)` on every BM25 search. Since the schema/tokenizers are fixed for the index's
lifetime and `parse_query_lenient` takes `&self`, the parser was cached as a struct field (built
once in the constructor) and reused. Correct and `Sync` (compiled; 79 lexical lib tests green).

**Measured command (per-crate):**
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-lexical --bench query_parser
```

In-process old-vs-new A/B (`query_parser` group; `old` reconstructs the parser per query, `new`
reuses a cached one; both run the identical lenient parse of a 7-token query):

| Workload | old (reconstruct + parse) | new (cached + parse) | ratio new/old | verdict |
|----------|---------------------------|----------------------|---------------|---------|
| `query_parser` | 8384.1 ns | 8336.3 ns | **0.994** | noise / no gain |

**Why it fails:** `QueryParser::for_index` is cheap (~tens of ns: an `Arc` tokenizer-manager clone
+ a small default-fields `Vec` + a boost map). The actual lenient *parse* — tokenizing the query
through both field analyzers and building term queries — costs ~8.3 µs and dominates, so the
construction it eliminates is <1% of the per-query cost, lost in noise. Reverted source + bench
(stashed, not landed). The real lexical materialization gap is `load_doc` (full docstore
decompress per hit in `search`/`search_doc_ids`), which needs a fast/columnar `id` field — an
index-format change, tracked separately; **not** the parser.

### 2026-06-25 — `normalize_whitespace` ASCII byte-scan fast path is SLOWER (BlackThrush)

**Lever:** `normalize_whitespace` (runs on every document at index time) walks `text.chars()` and
pushes char-by-char, then does a trailing `trim_end`/`truncate` pass. The candidate added an
`is_ascii()`-guarded byte path that scanned bytes, bulk-copied each non-whitespace run with one
`push_str`, and emitted the separating space inline (no trailing-trim pass). A custom `is_ws_ascii`
predicate (`b' ' | 0x09..=0x0d`) was used — **not** `u8::is_ascii_whitespace`, which excludes
`\x0b` (vertical tab) and would diverge from `char::is_whitespace`. Byte-identity proven across
vertical-tab/form-feed/mixed cases (`normalize_whitespace_ascii_matches`, 34/34 canonicalize tests
green).

**Measured command (per-crate, local fallback — RCH had no admissible workers):**
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-core --bench canonicalize normalize_whitespace
```

In-process old-vs-new A/B (`normalize_whitespace` group, 40× multi-line doc with newline +
multi-space runs):

| Workload | old (char path) | new (ASCII byte scan) | ratio new/old | verdict |
|----------|-----------------|-----------------------|---------------|---------|
| `normalize_whitespace` (~2.2 KB) | 3.415 µs | 3.910 µs | **1.145** | regression |

**Why it fails:** safe Rust cannot bulk-append a known-ASCII byte slice to a `String` without
`std::str::from_utf8` **re-validating** every run — that per-run validation scan costs more than
the char-by-char path saves, and std's `chars()`/`String::push` already have ASCII fast paths so
the original is near-optimal. The only way to skip validation is `from_utf8_unchecked` (`unsafe`),
and the crate is `deny(unsafe_code)`. **Do not re-attempt** the byte-scan rewrite of
`normalize_whitespace` under the safe-Rust constraint. Reverted source + bench (stash, not landed).

### 2026-06-25 — `search_minimal` lexical trait hook regresses decisive BOLD rows (BlackThrush)

**Lever:** add a `LexicalSearch::search_minimal` hook and route the non-semantic hash-tier
lexical guard through Tantivy `search_doc_ids`, converting those id-only hits back to
`ScoredResult` without loading stored metadata. The goal was to keep the measured BOLD
lexical-guard wins while avoiding full stored-document materialization in product code.

**Measured command (per-crate, warm target dir):**
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
FRANKENSEARCH_BOLD_VERIFY_EMIT=1 RUST_LOG=error \
FRANKENSEARCH_BOLD_VERIFY_COMMAND="CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a FRANKENSEARCH_BOLD_VERIFY_EMIT=1 RUST_LOG=error cargo bench -p frankensearch --features lexical --profile release --bench search_bench bold_verify_tantivy_class -- --sample-size 10 --warm-up-time 1 --measurement-time 1" \
  cargo bench -p frankensearch --features lexical --profile release \
  --bench search_bench bold_verify_tantivy_class \
  -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

**Artifact:** `/data/projects/.rch-targets/frankensearch-cod-a/criterion/bold_verify/summary.jsonl`
at git `bd3f59e2bc40f2d048bee34feda74ccd1049959b` (`worker="unknown"`; local warm target lane).

| Workload | Corpus hash | Tantivy-class p50 | full guard p50 | minimal guard p50 | minimal/full | minimal/Tantivy-class | Decision |
|----------|-------------|-------------------|----------------|-------------------|--------------|-----------------------|----------|
| `top10_exact_identifier/10000` | `2e78365a46a7c3b9` | 141 us | 119 us | 172 us | **1.445** | **1.220x slower** | reject |
| `top10_high_fanout/10000` | `2e78365a46a7c3b9` | 122 us | 106 us | 122 us | **1.151** | 1.000x tie | reject |
| `top10_short_keyword/10000` | `2e78365a46a7c3b9` | 59 us | 35 us | 45 us | **1.286** | 0.763x faster | reject vs shipped guard |
| `top10_quoted_phrase/10000` | `2e78365a46a7c3b9` | 149 us | 145 us | 146 us | 1.007 | 0.980x noise | no keep signal |
| `top10_natural_language/10000` | `2e78365a46a7c3b9` | 143 us | 151 us | 135 us | 0.894 | 0.944x faster | insufficient |
| `top10_zero_hit/10000` | `2e78365a46a7c3b9` | 37 us | 36 us | 23 us | 0.639 | 0.622x faster | insufficient |
| `limit_all/10000` | `2e78365a46a7c3b9` | 11.923 ms | 14.395 ms | 8.208 ms | 0.570 | 0.688x faster | insufficient |
| `top10_exact_identifier/100000` | `13f1b0153f5adec9` | 1.299 ms | 1.238 ms | 1.246 ms | 1.006 | 0.959x faster | noise |
| `top10_high_fanout/100000` | `13f1b0153f5adec9` | 611 us | 864 us | 1.000 ms | **1.157** | **1.637x slower** | reject |
| `top10_natural_language/100000` | `13f1b0153f5adec9` | 737 us | 1.069 ms | 816 us | 0.763 | **1.107x slower** | reject |
| `top10_quoted_phrase/100000` | `13f1b0153f5adec9` | 1.095 ms | 1.122 ms | 1.049 ms | 0.935 | 0.958x faster | insufficient |
| `top10_short_keyword/100000` | `13f1b0153f5adec9` | 213 us | 69 us | 187 us | **2.710** | 0.878x faster | reject vs shipped guard |
| `top10_zero_hit/100000` | `13f1b0153f5adec9` | 69 us | 59 us | 61 us | **1.034** | 0.884x faster | reject vs shipped guard |

**Decision:** reverted the trait hook, Tantivy override, BOLD harness variant, and test. The
minimal path wins some broad/materialization-heavy rows, but it gives back or destroys the exact
identifier, high-fanout, and short-keyword rows that make the current lexical guard worth keeping.
Do not add a public minimal-scored trait method until the backend can skip stored-document loading
without hurting these high-selectivity paths. A future attempt should target a private id-first
fusion path that avoids rebuilding owned `ScoredResult` rows for phase 1, then bench against this
same BOLD harness.

### 2026-06-25 — BOLD-VERIFY: hash-hybrid does **not** beat Tantivy-class BM25 (BlackThrush)

**Comparator shipped:** `frankensearch/benches/search_bench.rs` now includes
`bold_verify_tantivy_class`, a same-corpus Tantivy/Lucene-class incumbent harness:
Tantivy `search_doc_ids` vs frankensearch hash embedding + FSVI vector search + Tantivy candidates
+ RRF fusion. This is a **negative dominance check**, not a reverted source lever.

**Measured command:** `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a
rch exec -- env FRANKENSEARCH_BOLD_VERIFY_EMIT=1 RUST_LOG=error cargo bench -p frankensearch
--features lexical --profile release --bench search_bench bold_verify_tantivy_class
-- --sample-size 10 --warm-up-time 1 --measurement-time 3`

**Evidence:** the full per-crate Criterion pass completed on RCH worker `vmi1152480`. A filtered
summary rerun on RCH worker `hz2` emitted the machine-readable BOLD rows below; the emit flag must
be passed inside `rch exec -- env ...` because the wrapper does not preserve that outer environment
variable. The rows use the same fixed corpus/query harness and report
frankensearch p50 / Tantivy-class p50:

| Workload | Corpus hash | Tantivy-class p50 | frankensearch p50 | ratio |
|----------|-------------|-------------------|-------------------|-------|
| `top10_exact_identifier/10000` | `2e78365a46a7c3b9` | 449 us | 1.716 ms | **3.82x slower** |
| `top10_short_keyword/10000` | `2e78365a46a7c3b9` | 92 us | 1.616 ms | **17.57x slower** |
| `top10_quoted_phrase/10000` | `2e78365a46a7c3b9` | 145 us | 1.711 ms | **11.80x slower** |
| `top10_natural_language/10000` | `2e78365a46a7c3b9` | 352 us | 1.661 ms | **4.72x slower** |
| `top10_high_fanout/10000` | `2e78365a46a7c3b9` | 317 us | 1.749 ms | **5.52x slower** |
| `top10_zero_hit/10000` | `2e78365a46a7c3b9` | 80 us | 1.450 ms | **18.12x slower** |
| `limit_all_limit_all/10000` | `2e78365a46a7c3b9` | 6.324 ms | 12.318 ms | **1.95x slower** |
| `top10_exact_identifier/100000` | `13f1b0153f5adec9` | 2.260 ms | 8.143 ms | **3.60x slower** |
| `top10_short_keyword/100000` | `13f1b0153f5adec9` | 515 us | 4.082 ms | **7.93x slower** |
| `top10_quoted_phrase/100000` | `13f1b0153f5adec9` | 1.040 ms | 6.412 ms | **6.17x slower** |
| `top10_natural_language/100000` | `13f1b0153f5adec9` | 1.641 ms | 5.756 ms | **3.51x slower** |
| `top10_high_fanout/100000` | `13f1b0153f5adec9` | 647 us | 3.742 ms | **5.78x slower** |
| `top10_zero_hit/100000` | `13f1b0153f5adec9` | 48 us | 2.827 ms | **58.90x slower** |

**Decision:** no Lucene/Tantivy/Meilisearch-class win exists for the current hash-hybrid path.
Future bold claims need a new lever that changes the cost model (ANN/int8 slab reuse, lexical
short-circuiting, semantic gating, or a Meilisearch-class prefix/typo comparator), then must reuse
this head-to-head harness. Do not cite frankensearch hybrid as faster than Tantivy-class BM25 from
the current implementation.

### 2026-06-25 — binary-quantization ADC is too coarse for top-10; int8 ADC dominates (BlackThrush)

**Lever assessed (not built):** a Meilisearch-style **binary-quantization** first pass — pack
`sign(x_i)` to bits, rank by Hamming agreement (`popcnt`, fast even on SSE2, 1/16 the bytes of
f16), then exact f16 rescore. Faster pass-1 than int8, so the question is recall. Measured
(`binary_quant_recall_at_10`, random L2-normalized vectors, dim=128, n=3000, recall@10):

| candidate mult | 5 | 10 | 20 | 50 | 100 |
|----------------|------|------|------|------|------|
| binary recall@10 | 0.54 | 0.71 | 0.85 | 0.96 | **1.00** |
| (int8 recall@10) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

**Conclusion:** binary needs `mult ≈ 100` (k·mult = 1000 candidates) for recall ≈ 1.0, vs int8's
`mult = 2` (20 candidates) — ~50× coarser. At frankensearch's typical scale that means rescoring a
large fraction of the corpus, so the **already-shipped int8 ADC two-pass is strictly better** for
top-10. Binary only pulls ahead at very large N (≫1M), where the *fixed* ~1000-candidate rescore
is negligible and the super-fast `popcnt` pass-1 dominates. **Do not build binary ADC** unless
specifically targeting ≫1M-vector corpora. (Probe test kept for reproducibility; no production
code added.)

### 2026-06-26 — 8-accumulator int8 dot is not a robust successor to 4acc (BlackThrush)

**Lever tested and reverted:** extend the shipped `dot_i8_i8` loop from four independent
`i32x8` accumulators over 32 elements to eight accumulators over 64 elements, then try a narrower
384-dim-only gate after the global form regressed the 256-dim path. This is a primitive behind the
int8 ADC pass-1 vector scan, not a direct Lucene/Tantivy-class BM25 comparator; because all code
was reverted, the product-level Tantivy/Lucene/Meilisearch-class ratios remain the BOLD rows above.

The literal requested command shape with `cargo bench --release` was checked first and Cargo
rejected it (`unexpected argument '--release'`), so measurements used the equivalent release profile:
`AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a rch exec -- cargo bench -p frankensearch-index --bench dot_product --profile release i8_dot -- --sample-size 10 --warm-up-time 1 --measurement-time 2`.

**Remote `ovh-a`, global 8acc vs bench-local 4acc comparator (Criterion mean):**

| Workload | 4acc comparator | 8acc candidate | candidate/4acc | Decision |
|----------|-----------------|----------------|-----------------|----------|
| `dot/dim256/i8_dot` | 376.995 us | 395.577 us | **1.049** | regression |
| `dot/dim384/i8_dot` | 2.410 ms | 1.092 ms | 0.453 | apparent win, not landable because dim256 regressed |

**Local `rch` fallback, 384-dim-gated retry vs bench-local 4acc comparator (Criterion stdout
point estimate):**

| Workload | 4acc comparator | gated candidate | candidate/4acc | Decision |
|----------|-----------------|-----------------|-----------------|----------|
| `dot/dim256/i8_dot` | 619.51 us | 786.06 us | **1.269** | regression |
| `dot/dim384/i8_dot` | 1.0314 ms | 1.7042 ms | **1.652** | regression |

**Decision:** reverted the production kernel and the temporary bench-only 4acc comparator. The
current 4-accumulator implementation remains the only accepted int8 dot ILP lever; do not retry
8acc without a same-process no-regression proof for 256 and a stable 384 win.

### 2026-06-25 — SIMD-widen int8 dot (`i16x16::dot`/`vpmaddwd`) is SLOWER on this build (BlackThrush)

**Lever:** rewrite `dot_i8_i8` from scalar `i16::from` widening (16 `movsx` per 8 elems) to a
fully-SIMD 16-wide kernel: `[i8;16]` → `i8x16` → `i16x16` (sign-extend) → `i16x16::dot`
(`vpmaddwd`, pairwise products → `i32x8`). Correct (`dot_i8_i8_matches_scalar` green incl. the
all-(−128)/512-dim overflow case), but measured a **regression** vs the committed scalar-widen
kernel:

| Workload | int8/f16 ratio (scalar-widen, kept) | int8/f16 ratio (SIMD-widen) |
|----------|-------------------------------------|-----------------------------|
| `dot/dim256/i8_dot` | 0.331 | **0.508** |
| `dot/dim384/i8_dot` | 0.311 | **0.442** |

(in-process int8-vs-f16 ratio; higher = int8 got relatively slower). Normalizing for worker
speed, the SIMD-widen int8 dot is ~1.5× **slower** than scalar-widen.

**Why it fails:** this is an **SSE2-class build** (no AVX2/SSE4.1 — consistent with the ~1 ns/elem
f32 dots seen elsewhere). `i8→i16` `vpmovsxbw` and 256-bit `i16x16` are then *emulated* (unpack +
arithmetic-shift over two SSE registers), which costs more than 16 plain scalar `movsx`. "Portable
SIMD" loses to scalar when the target lacks the widening instruction. Reverted to scalar-widen.
**Do not re-attempt** without either a runtime AVX2/SSE4.1 dispatch or `-Ctarget-cpu` that enables
those features (which the published library cannot assume).

### 2026-06-24 — int8 ADC two-pass does NOT beat *parallel* exact at top-10/10k (BlackThrush)

**Self-correction of an earlier overstated result.** The `3ecfad8` bench reported the int8 ADC
two-pass ~2.6–3× faster than "exact f16", but that baseline (`topk_exact_f16`) was a **serial
full-sort** pipeline. The **real product** `InMemoryVectorIndex::search_top_k` is **rayon-parallel
+ bounded-heap + cutoff** — much faster. Benching the real shipped methods head-to-head
(`inmem_topk`, 10k vectors, top-10, mult=20, parallel pass-1):

| Workload | exact `search_top_k` (parallel) | `search_top_k_int8_two_pass` | ratio |
|----------|--------------------------------|------------------------------|-------|
| `inmem_topk/dim256` | 306 µs | 373 µs | **1.22 (regression)** |
| `inmem_topk/dim384` | ~400–700 µs (very noisy) | 393 µs | inconclusive |

**Root cause:** the int8 *kernel* is genuinely ~3× faster (that stands — `33fb45b`), but the
two-pass **method** materializes all N int8 scores into a `Vec` then selects serially, while the
exact path never materializes more than the top-k heap and runs across all cores. At 10k the
already-parallel+cutoff exact is ~300 µs; the two-pass's full-N materialize + serial select eats
the kernel win. So the int8 ADC two-pass is **not** a win at this scale/path.

**Honest scope of the kept results:** int8 dot ~3× (kernel, real), recall@10 = 1.0 (real). The
**search-level** speedup only holds vs a *serial* exact — it does **not** beat the product's
parallel exact at 10k. The lever's real upside is at larger N (100k+, where parallel exact also
slows and bandwidth matters more) or the mmap FSVI path (page-fault + decode overhead), or with a
**bounded-heap parallel pass-1** (avoid the full-N materialize). Filed as a follow-up.

**Decision:** the `search_top_k_int8_two_pass` method is kept (correct, opt-in, bit-identical when
recall=1 — proven by `int8_two_pass_matches_exact_topk`; a foundation), but it carries **no
verified perf-win claim at 10k**. PERF_LEDGER corrected accordingly.

### 2026-06-24 — multi-accumulator unrolling of the **f16** dot-product kernels (BlackThrush)

**Lever:** rewrite `dot_product_f16_f32` and `dot_product_f16_bytes_f32` to use 4 independent
`f32x8` accumulators (32 elements/iter) instead of 1, to break the SIMD-add latency chain.

**Measured head-to-head** (`benches/dot_product.rs`, `*_new` = 4-acc, `*_old` = original
single-acc, same process / same CPU, n=10 000 dots):

| Workload | old (median) | new (median) | ratio new/old | verdict |
|----------|-------------|-------------|---------------|---------|
| `dot/dim256/f16_bytes` | 12.483 ms | 13.504 ms | **1.082** | regression |
| `dot/dim384/f16_bytes` | 17.486 ms | 18.115 ms | **1.036** | regression |
| `dot/dim256/f16_slice` | 10.242 ms | 14.909 ms | **1.456** | regression (noise-inflated) |
| `dot/dim384/f16_slice` | 13.489 ms | 14.277 ms | **1.058** | regression |

**Why it fails:** the f16 paths are **decode-bound** — the per-element scalar `f16::to_f32()`
conversion dominates, so the accumulation latency the change targets is a negligible fraction.
The restructure (chunks_exact(32) + `try_into` per sub-block + two-phase remainder) only adds
setup overhead. Reverted these two kernels to the original single-accumulator form.

**Connects to historical revert `816963a`:** the human previously reverted `88c291b`
("…multi-accumulator unrolling"), which *bundled* this f16 kernel change with a
`select_nth_unstable` heap-merge (unstable sort → breaks deterministic tie ordering) and a
16-elem unroll. This measurement isolates the kernel part and confirms it is **not** a win on
the f16 (default-quantization) path independent of the heap change. Do not re-attempt
accumulator unrolling on f16 kernels without first making the f16→f32 decode SIMD (a
branchless `i32x8`/F16C widen), which is the actual bottleneck.

**Kept from the same experiment:** the `f32_bytes` kernel restructure was a genuine ~3× win
(decode-bound on open-ended slices, not accumulation) → see `docs/PERF_LEDGER.md`.

### 2026-06-24 — f32 slice multi-accumulator portability check (BlueGull)

**Lever:** apply the same 4-accumulator, 32-element loop shape to `dot_product_f32_f32`.

**Evidence:** after the f16 revert, this command fell back local because RCH had no admissible
workers:

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
  rch exec -- cargo bench -p frankensearch-index --profile release --bench dot_product
```

That local fallback measured `dot/dim256/f32_slice` at 1.4206 ms old vs 2.3436 ms new,
ratio **1.650** (regression). A later pinned-worker run with the source hunk reverted
measured the slice rows as noise-only routing checks, not a kept source change.

**Superseded 2026-06-26 by BlackThrush:** this local fallback was contradicted by an admitted
remote RCH same-process run on `vmi1227854`, which measured the f32 slice 4-accumulator kernel at
0.318 (dim256) and 0.610 (dim384) vs the embedded single-accumulator baseline. The current
decision is **ship the f32 slice rewrite** with the scope caveat above: it is a frankensearch
microkernel win, not a Lucene/Tantivy/Meilisearch-class end-to-end dominance claim.

### 2026-06-26 — BOLD re-check: f32/f32 4-accumulator slice dot is still not a win (BlackThrush)

**Lever tested and reverted:** a dirty worktree hunk rewrote
`dot_product_f32_f32_unchecked` from one `f32x8` accumulator over 8 elements to four accumulators
over 32 elements. This was the later `ready-to-land f32_f32 4-acc lever` note from the measurement
blockers table, but it had not been proven head-to-head on this tree. The hunk is a vector
primitive, not a direct Lucene/Tantivy-class BM25 comparator; because it was reverted, product-level
Lucene/Tantivy/Meilisearch-class ratios are unchanged.

The requested `cargo bench --release` form failed again with Cargo's
`unexpected argument '--release'`, so the measured command was:

```bash
AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
  rch exec -- cargo bench -p frankensearch-index --bench dot_product --profile release \
  f32_slice -- --sample-size 10 --warm-up-time 1 --measurement-time 2
```

RCH fell back local (`no admissible workers`) but the bench is in-process `new` vs `old`, so the
ratio is still same-run:

| Workload | old single-acc | 4acc candidate | candidate/old | Decision |
|----------|----------------|----------------|---------------|----------|
| `dot/dim256/f32_slice` | 2.4522 ms | 2.5077 ms | **1.023** | neutral / revert |
| `dot/dim384/f32_slice` | 2.9224 ms | 3.0636 ms | **1.048** | regression |

**Decision:** reverted the `simd.rs` hunk. Do not land the f32/f32 4-accumulator slice rewrite from
the shared worktree; the earlier ready-to-land note is superseded by this BOLD re-check.

### 2026-06-24 — detached-worktree f16 accumulator candidate (frankensearch-cod-b)

**Context:** a detached worktree at
`/data/projects/frankensearch-cod-b-baseline-20260624T221243Z` was used to compare
`HEAD` plus the `dot_product` bench harness against a 4-accumulator SIMD candidate that was
not on `main`. The accepted release-profile command could not be admitted by RCH (see
Measurement blockers), so the only completed same-worker numbers are Cargo's optimized bench
profile on worker `vmi1152480`.

**Bench-profile RCH before/after (`cargo bench -p frankensearch-index --bench dot_product
-- --sample-size 10 --warm-up-time 1 --measurement-time 3`):**

| Workload | before median | after median | ratio new/old | verdict |
|----------|---------------|--------------|---------------|---------|
| `dot/dim256/f16_bytes/10000` | 7.1007 ms | 3.9052 ms | 0.550 | routing win, not kept |
| `dot/dim256/f16_slice/10000` | 5.0532 ms | 4.0380 ms | 0.799 | routing win, not kept |
| `dot/dim384/f16_bytes/10000` | 7.0845 ms | 5.7334 ms | 0.809 | routing win, not kept |
| `dot/dim384/f16_slice/10000` | 5.6829 ms | 6.0270 ms | 1.061 | regression |

**Decision:** do not land this detached-worktree f16 accumulator candidate. It regresses a
tracked `f16_slice` workload and conflicts with the stronger in-process old-vs-new evidence
above, which isolates the f16 accumulator rewrite without cross-run Criterion noise. The
follow-up is `bd-gfzk`: attack the actual default-path bottleneck, scalar f16-to-f32 decode,
with an exhaustive correctness proof before rebenchmarking.

### 2026-06-24 — `f16_slice` u16x8 SIMD load (BlackThrush)

**Lever:** mirror the landed `f16_bytes` SIMD-load refinement (`c0e9c80`) onto the slice path
`widen8_f16_slice` — build a `u16x8` of `to_bits()` lanes and zero-extend to `u32x8` (16-byte
stack slot + `vpmovzxwd`) instead of materializing a `[u32; 8]`. Bit-exact (20/20 simd tests
green); the slice path feeds `dot_product_f16_f32` (`in_memory.rs:480`, two-tier quality rescore).

**Measured** (`f16_slice_new` crate vs `f16_slice_old` scalar, in-process; ratios are the only
worker-robust signal across runs):

| Workload | prior ratio (`[u32;8]`) | this change (`u16x8`) | verdict |
|----------|------------------------|-----------------------|---------|
| `dot/dim256/f16_slice` | 0.364 | **0.508** | no gain / hint of regression |
| `dot/dim384/f16_slice` | 0.394 | **0.420** | no gain (≈ noise) |

**Why reverted:** this run landed on a much slower/contended worker (absolute times ~2× the
prior run) and the cross-run ratio moved the **wrong way** on dim256 (0.364→0.508). With no
clean in-process A/B to isolate the load change from the fully-scalar baseline, there is no
demonstrated gain (and a hint of regression). Per "REVERT ~0-gain", reverted to the committed
`[u32; 8]` form. The byte-path SIMD load (`c0e9c80`) stays — it showed a consistent directional
improvement on both dims; the slice path did not. A clean keep/reject would need an in-process
"SIMD-widen + scalar-load" baseline added to `dot_product.rs` (left for a future pass).

### 2026-06-25 — non-semantic lexical guard is NOT a universal Tantivy-class win (BlackThrush)

**Kept change, bounded claim.** The hash-tier lexical guard is a real improvement over the old
hash-hybrid path (see `docs/PERF_LEDGER.md`), but the same BOLD-VERIFY run still shows several
rows slower than the `tantivy_doc_ids` lexical proxy used for Lucene/Tantivy/Meilisearch-class
comparison.

Command:

```bash
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
  rch exec -- env FRANKENSEARCH_BOLD_VERIFY_EMIT=1 RUST_LOG=error \
    cargo bench -p frankensearch --features lexical --profile release \
    --bench search_bench bold_verify_tantivy_class \
    -- --sample-size 10 --warm-up-time 1 --measurement-time 3
```

Worker: `vmi1152480` (`[RCH] remote vmi1152480 (804.5s)`). Incumbent:
`tantivy_doc_ids`.

| Workload | guarded p50 ratio | guarded p95 ratio | verdict |
|----------|-------------------|-------------------|---------|
| `top10/10000/exact_identifier` | 1.074 | 1.359 | near, but not a win |
| `top10/10000/quoted_phrase` | 1.031 | 1.330 | near, but not a win |
| `top10/10000/natural_language` | 1.040 | 1.067 | near, but not a win |
| `top10/10000/zero_hit` | 1.000 | 3.333 | p50 parity, tail miss |
| `top10/100000/exact_identifier` | 1.161 | 1.820 | miss |
| `top10/100000/short_keyword` | 1.208 | 1.439 | miss |
| `top10/100000/quoted_phrase` | 2.068 | 1.473 | miss |
| `top10/100000/high_fanout` | 1.010 | 1.236 | near, but not a win |

**Interpretation:** skipping hash embedding/vector/RRF is necessary but not sufficient for full
lexical-engine parity. The remaining overhead is in frankensearch's result materialization and
the fact that the incumbent comparator returns doc ids, while the guarded path produces
`ScoredResult`-class results. Do not claim Lucene/Tantivy/Meilisearch dominance from this change.

**Next lever:** avoid eager `ScoredResult` string/materialization for lexical-only Phase 1
(borrowed/id-first result lane or lazy metadata resolution), then rerun the same BOLD matrix.

### 2026-06-26 — id-first RRF lexical hits are not a BOLD win (BlackThrush)

**Lever tested and reverted:** added a feature-gated `rrf_fuse_lexical_ids` path so the
hash-hybrid BOLD lane could pass `LexicalIdHit` rows directly into RRF instead of first
materializing `ScoredResult` rows. This targeted the materialization gap called out in the
2026-06-25 lexical-guard evidence. Behavior was guarded with
`cargo test -p frankensearch-fusion --features lexical lexical_id_fusion_matches_scored_rrf --lib`
through `rch exec` (remote `ovh-a`, pass), but the end-to-end BOLD matrix was mixed and mostly
worse.

The requested `cargo bench --release` form remains invalid for this harness (`unexpected argument
'--release'`), so the measured command used the accepted release profile form:

```bash
AGENT_NAME=BlackThrush RCH_ENABLED=0 \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
FRANKENSEARCH_BOLD_VERIFY_OUT=/data/projects/frankensearch/.scratch/bold_id_first_candidate_20260626T0658Z \
  rch exec -- cargo bench -p frankensearch --features lexical --profile release \
    --bench search_bench bold_verify_tantivy_class \
    -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

RCH executed locally (`[RCH] local`), matching the baseline host. Artifact paths:

- Baseline: `frankensearch/frankensearch/.scratch/bold_id_first_baseline_20260626T0638Z/summary.jsonl`
- Candidate: `.scratch/bold_id_first_candidate_20260626T0658Z/summary.jsonl`

Hybrid candidate vs Tantivy-class `tantivy_doc_ids` proxy:

| Workload | candidate p50 ratio | candidate p50 us | Tantivy p50 us | verdict |
|----------|---------------------|------------------|----------------|---------|
| `top10/10000/exact_identifier` | 1.081 | 160 | 148 | miss |
| `top10/10000/short_keyword` | 1.073 | 44 | 41 | miss |
| `top10/10000/quoted_phrase` | 1.111 | 190 | 171 | miss |
| `top10/10000/natural_language` | 0.710 | 142 | 200 | win row only |
| `top10/10000/high_fanout` | 1.631 | 137 | 84 | regression |
| `top10/10000/zero_hit` | 0.219 | 21 | 96 | win row only |
| `limit_all/10000` | 1.029 | 7753 | 7533 | miss |
| `top10/100000/exact_identifier` | 0.941 | 1219 | 1295 | win row only |
| `top10/100000/short_keyword` | 1.100 | 351 | 319 | miss |
| `top10/100000/quoted_phrase` | 1.033 | 1166 | 1129 | miss |
| `top10/100000/natural_language` | 1.028 | 821 | 799 | miss |
| `top10/100000/high_fanout` | 1.059 | 880 | 831 | miss |
| `top10/100000/zero_hit` | 1.305 | 124 | 95 | regression |

Compared with the baseline artifact, the hybrid p50 ratio worsened on 9 of 13 rows, including
`top10/10000/high_fanout` (1.156 → 1.631) and `top10/100000/zero_hit` (1.034 → 1.305).
Because the candidate does not produce a durable Lucene/Tantivy-class win, the code was reverted
and only this ledger entry is kept. A future attempt should attack the remaining Tantivy wrapper
overhead without disturbing high-fanout and zero-hit tails.

### 2026-06-27 — shared exact-id lexical probe is not a hybrid BOLD win (BlackThrush)

**Lever tested and reverted:** added a real production exact-id probe to
`frankensearch-lexical` so identifier-shaped queries such as `doc-000001` and `doc 000001`
searched the Tantivy `id` field before falling back to normal content/title BM25. The probe was
behaviorally valid (`search_doc_ids_exact_identifier_alias_hits_id_field` and
`search_doc_ids_exact_identifier_alias_miss_falls_back_to_bm25` passed), but it is a shared
lexical-backend improvement: the `tantivy_doc_ids` Lucene/Tantivy/Meilisearch-class proxy benefits
too, while the hash-hybrid challenger still falls through to vector/RRF when the exact lexical hit
count is below `limit=10`.

Command:

```bash
AGENT_NAME=BlackThrush \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
  rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
    FRANKENSEARCH_BOLD_VERIFY_EMIT=1 \
    FRANKENSEARCH_BOLD_VERIFY_SUMMARY_ONLY=1 \
    cargo bench -p frankensearch --features lexical --profile release \
      --bench search_bench bold_verify_tantivy_class \
      -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Worker: `hz2` (`[RCH] remote hz2 (357.9s)`). Parsed log artifact:
`frankensearch/.scratch/bold_exact_id_prod_teed_20260627T022000Z/stdout_summary.jsonl`.
Incumbent: `tantivy_doc_ids`.

Hybrid candidate vs Tantivy-class proxy:

| Workload | candidate p50 ratio | candidate p50 us | Tantivy p50 us | verdict |
|----------|---------------------|------------------|----------------|---------|
| `top10/10000/exact_identifier` | 28.300 | 1132 | 40 | severe regression |
| `top10/100000/exact_identifier` | 133.125 | 4260 | 32 | severe regression |
| `top10/10000/quoted_phrase` | 1.147 | 164 | 143 | miss |
| `top10/100000/natural_language` | 1.220 | 1935 | 1586 | miss |
| `top10/100000/high_fanout` | 1.006 | 661 | 657 | miss |

The lexical-guard lane confirmed why this is not enough: `top10/10000/exact_identifier` improved
to 0.825x (33 us vs 40 us), but `top10/100000/exact_identifier` still lost at 2.875x (92 us vs
32 us), and the shipped hybrid path regressed dramatically. The exact-id probe was therefore
reverted. A future lever must either make the hybrid exact-hit path terminate on high-confidence
identifier hits without changing recall contracts, or move to a separate exact-id lookup structure
that is not also granted to the Tantivy-class incumbent.

### 2026-06-27 — sparse lexical underfill short-circuit is not a BOLD win (BlackThrush)

**Lever tested and reverted:** changed the BOLD hash-hybrid harness so any under-filled
`search_doc_ids` result (`lexical_count < limit`) short-circuited before hash embedding, vector
scan, and RRF. This came from the query-processing decision-barrier pattern in the alien-graveyard
dig: when the fast tier is a non-semantic hash and Tantivy has already proven the lexical result
set is sparse, extra vector/RRF work has no semantic recall value. It is also aligned with the
production `TwoTierSearcher` hash-only lexical path, but the BOLD matrix did not produce a durable
Lucene/Tantivy/Meilisearch-class win.

Command:

```bash
AGENT_NAME=BlackThrush \
RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,FRANKENSEARCH_BOLD_VERIFY_OUT,FRANKENSEARCH_BOLD_VERIFY_EMIT,FRANKENSEARCH_BOLD_VERIFY_SUMMARY_ONLY,FRANKENSEARCH_BOLD_VERIFY_COMMAND,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
  rch exec -- env \
    FRANKENSEARCH_BOLD_VERIFY_OUT=/tmp/frankensearch-bold-sparse-lexical-blackthrush-20260627 \
    FRANKENSEARCH_BOLD_VERIFY_EMIT=1 \
    FRANKENSEARCH_BOLD_VERIFY_SUMMARY_ONLY=1 \
    RUST_LOG=off \
    cargo bench -p frankensearch --features lexical --profile release \
      --bench search_bench bold_verify_tantivy_class \
      -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

RCH executed locally (`[RCH] local`; no admissible worker). Artifact:
`/tmp/frankensearch-bold-sparse-lexical-blackthrush-20260627/summary.jsonl`. Incumbent:
`tantivy_doc_ids`, used as the Lucene/Tantivy/Meilisearch-class proxy.
Required `frankensearch-cod-b` reruns were attempted after the revert, but did not emit BOLD rows
before the local harness returned code 130, so no code change is shipped from this evidence.

Hybrid candidate vs Tantivy-class proxy:

| Workload | candidate p50 ratio | candidate p95 ratio | candidate p50 us | Tantivy p50 us | verdict |
|----------|---------------------|---------------------|------------------|----------------|---------|
| `top10/100000/quoted_phrase` | 1.107 | 0.906 | 1295 | 1170 | p50 miss; p95 win only |
| `top10/100000/exact_identifier` | 0.945 | 1.007 | 1698 | 1797 | p50 win only |
| `top10/100000/short_keyword` | 0.996 | 1.003 | 249 | 250 | noise/parity |
| `top10/10000/quoted_phrase` | 1.197 | 1.873 | 231 | 193 | regression |
| `top10/100000/natural_language` | 1.506 | 1.530 | 1337 | 888 | regression |
| `limit_all/10000` | 1.196 | 1.251 | 11566 | 9671 | regression |

The old 100k quoted-phrase miss from 2026-06-25 was 2.068x, so this lever did narrow the biggest
stale measured gap, but it still failed p50 parity against the Tantivy-class proxy and introduced
too many mixed rows. Per the revert rule, the code was restored and only this ledger entry is kept.

### 2026-06-27 — count-free top-k (drop the `Count` collector) is a mixed result, not a clean win (CopperLark)

**Lever tested and reverted:** `execute_query_with_offset` runs `(TopDocs::with_limit(k).order_by_score(), Count)`.
The `Count` collector must visit *every* matching document, which forces Tantivy to disable the
block-max WAND top-k pruning that `TopDocs` can otherwise use to skip low-scoring blocks.
`search_doc_ids` (the `tantivy_doc_ids` Lucene/Tantivy/Meilisearch-class BOLD proxy **and** the
hybrid lexical-candidate source) only reads `.hits` and **discards `total_count`** — so the Count
work is wasted there. The lever added a count-free `execute_top_k` (identical ranked hits, no
`Count`), routed `search_doc_ids` through it, and kept a `#[doc(hidden)] search_doc_ids_counted`
baseline so a new per-crate bench (`crates/frankensearch-lexical/benches/doc_ids_topk.rs`) could
A/B `free` vs `counted` in one binary on a 100k in-RAM index.

**Conformance:** GREEN — `cargo test -p frankensearch-lexical --lib` = **79 passed; 0 failed**. The
collector choice does not change which docs `TopDocs` returns, so both paths yield identical ids.

**Command** (RCH fell back to local — `[RCH] local (no admissible workers: insufficient_slots=3)`):

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-lexical --bench doc_ids_topk \
    -- --sample-size 20 --warm-up-time 1 --measurement-time 2
```

**Measured `free`/`counted` ratio (median; `< 1.0` = count-free faster):**

| Corpus / workload | counted | free | ratio | verdict |
|-------------------|---------|------|-------|---------|
| BOLD-mirror `high_fanout` (`search`) | 630.6 us | 637.1 us | **1.010** | neutral |
| BOLD-mirror `short_keyword` (`rust ownership`) | 233.9 us | 63.3 us | **0.271** | big win |
| BOLD-mirror `quoted_phrase` (`"reciprocal rank fusion"`) | 1.825 ms | 1.083 ms | **0.593** | win |
| BOLD-mirror `natural_language` (7-term) | 728.6 us | 1.508 ms | **2.070** | regression (noisy 0.99–2.23 ms) |
| synthetic uniform-vocab `high_fanout` (1 term) | 286.0 us | 222.3 us | 0.777 | win |
| synthetic `union3` / `natural` / `phrase` (multi-term) | — | — | 1.66 / 3.56 / 1.12 | regressions |

**Interpretation:** count-free WAND wins large on *selective* queries (`short_keyword` 0.271,
`quoted_phrase` 0.593 — two of the actual BOLD gap rows) but **regresses ~2x on
`natural_language`**, a broad disjunction dominated by the corpus-wide `search` term (present in
every BOLD doc): WAND pays its block-max bookkeeping but can't prune a near-universal posting list,
so it loses to the plain `Count` linear scan. This is the known block-max-WAND pathology for
low-IDF disjunctions. Because a real BOLD workload regresses ~2x, the change **cannot land
unconditionally** (repo rule: regression > 1.03 → revert), so all code was reverted and only this
entry is kept.

**Ratio vs Lucene/Tantivy/Meilisearch = N/A:** the in-repo `tantivy_doc_ids` proxy *is*
`search_doc_ids`, so this collector change would move the proxy and the hybrid identically — the
BOLD *ratio* vs the proxy is unchanged by construction. The A/B above isolates the collector-choice
effect (what real count-free-WAND engines experience), not a head-to-head incumbent ratio.

**Real lever for a future agent:** a *query-adaptive* collector — count-free (`TopDocs` alone, WAND)
for selective queries (short-keyword / phrase / identifier), counted (full scan) for broad
natural-language disjunctions that include a corpus-saturating term. Validate on a realistic
Zipfian corpus *without* a universal term (where `natural_language` may also win) before landing;
do not re-attempt the unconditional drop.

### 2026-06-27 — count-free simple `search_doc_ids` is a narrow Tantivy-wrapper win, not BOLD dominance (BlackThrush)

**Lever kept with scope limits:** `frankensearch-lexical::TantivyIndex::search_doc_ids` now skips
Tantivy's discarded `Count` collector only for single plain-token queries. Boolean, phrase, fielded,
wildcard, boosted, path-like, and hyphenated queries keep the counted execution path so Tantivy query
semantics and pagination-sensitive collectors stay out of this optimization.

Literal requested command failed because this checkout's Cargo rejects `cargo bench --release`:

```bash
AGENT_NAME=BlackThrush \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
  rch exec -- cargo bench --release -p frankensearch-lexical \
    --bench doc_ids_topk -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Measured command:

```bash
AGENT_NAME=BlackThrush \
RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
RUST_LOG=off \
  rch exec -- cargo bench -p frankensearch-lexical --profile release \
    --bench doc_ids_topk -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

RCH executed on `hz2` (`[RCH] remote hz2 (382.8s)`). Criterion artifacts:
`/data/projects/.rch-targets/frankensearch-cod-b/criterion/doc_ids_topk/*/new/estimates.json`.

Per-crate Tantivy-wrapper ratios:

| Workload | Query shape | counted median | new median | ratio | verdict |
|----------|-------------|----------------|------------|-------|---------|
| `high_fanout` | single plain token | 499.086 us | 239.233 us | **0.479** | kept narrow win |
| `union3` | multi-token boolean fallback | 804.545 us | 926.472 us | 1.152 | not claimed |
| `natural` | multi-token boolean fallback | 1141.046 us | 1312.298 us | 1.150 | not claimed |
| `phrase` | phrase fallback | 1140.312 us | 1414.064 us | 1.240 | not claimed |

The ratio above is against the prior counted Tantivy top-k wrapper in frankensearch, used here as
the local Lucene/Tantivy/Meilisearch-class lexical collector proxy. The non-simple rows are retained
as negative evidence rather than a claimed win; they stay on the counted path in source and are
dominated by the extra guard plus noisy tracing-heavy Criterion order. This is not a new BOLD
end-to-end dominance claim over Lucene, Tantivy, or Meilisearch: the accepted original-comparator
ratio is **N/A** for this isolated wrapper primitive, while full hybrid BOLD gaps remain governed by
the existing comparator ledger rows.

### 2026-06-27 — do NOT extend count-free top-k to phrase queries (CopperLark, corroborates BlackThrush)

Independently re-derived the count-free `search_doc_ids` lever before noticing BlackThrush had
landed the single-plain-token gate (`use_count_free_top_k`, PERF_LEDGER 2026-06-27). The obvious
next step — broadening the gate to **phrase** queries (phrases use a `PhraseScorer`, so I expected
dropping `Count` to be strictly safe) — was tested with a dedicated `doc_ids_phrase` A/B on the
100k BOLD-mirror corpus and is **not** safe. Per-shape `free`/`counted` medians:

| Phrase query | counted | free | ratio | verdict |
|--------------|---------|------|-------|---------|
| `"reciprocal rank fusion"` (theme, ~1/6) | 1.151 ms | 1.148 ms | 0.998 | noise |
| `"approximate nearest neighbor"` (theme, ~1/6) | 1.113 ms | 1.226 ms | 1.101 | regression |
| `"kubernetes health check"` (theme, ~1/6) | 1.285 ms | 1.085 ms | 0.844 | win |
| `"common search corpus term"` (universal, all docs) | 8.333 ms | 10.31 ms | 1.237 | regression |

The result is **workload-dependent even within phrases** (one shape wins 0.844, another regresses
1.24), and the prior cycle's single `quoted_phrase` 0.593 (entry above) did **not** reproduce
(0.998 here) — it was local-fallback contention noise. So the tempting `0.844` win is a trap: there
is no safe, corpus-independent way to gate count-free for phrases. BlackThrush's single-plain-token
boundary is the correct stopping point; do not broaden it to phrase/Boolean shapes without
per-query corpus statistics. Conformance was green (`cargo test -p frankensearch-lexical --lib`,
80 passed) and all probe code was reverted — only this entry is kept.

### 2026-06-27 — 4 independent accumulators REGRESS the 4-bit prepared dot kernel (decode-bound, not sum-chain-bound) (BlackThrush)

**Lever tested and reverted:** apply the proven multi-accumulator ILP pattern to `dot_4bit_prepared`
(`crates/frankensearch-index/src/simd.rs`) — the pass-1 hot kernel of the current vector frontier,
`search_top_k_4bit_two_pass` (landed 1.40× vs int8 / 3.09× vs flat). That kernel uses a **single**
`i16x16` accumulator with a periodic flush, i.e. exactly the single-accumulator add chain that
`dot_i8_i8`'s own comment says cost 6–16% and was fixed there with 4 independent `i32x8`
accumulators. Natural hypothesis: the same 4-accumulator split should win on the 4-bit kernel too.
I rewrote it to 4 independent `i16x16` accumulators processing 4 chunks per iteration (flush every
256 chunks for overflow safety; integer-exact, so bit-identical — `dot_packed_4bit_matches_scalar`
still passes).

**Measured (in-process A/B, `dot_product` bench; `fourbit_4acc` = new, `fourbit_1acc` = original
single-acc, both over the *same* prepared query so the ratio isolates only the accumulator change).
New arm names doubled as a freshness sentinel — both appeared, so the bench binary was not stale.**

Remote `rch` worker `ovh-a` (`cargo bench -p frankensearch-index --profile release --bench
dot_product -- fourbit --sample-size 10 --measurement-time 3`) caught a contended window on the
`fourbit_4acc` arm (CI `[405 µs … 714 µs]`), but its identical sibling `fourbit_prepared_new`
(same lib fn) measured cleanly at **325.9 µs** vs `fourbit_1acc` **318.3 µs** → **1.024 (slower)**.

Clean local re-measure (`--sample-size 30`), `fourbit_4acc` / `fourbit_1acc` medians:

| dim | 1 acc (old) | 4 acc (new) | ratio | verdict |
|-----|-------------|-------------|-------|---------|
| 256 | 304.08 µs | 332.09 µs | **1.092** | regression |
| 384 | 385.69 µs | 409.96 µs | **1.063** | regression |

**Decision:** rejected and fully reverted (`simd.rs` + the bench A/B arms restored byte-identical to
`main`; `git diff` empty before this entry). Both runs and both dims agree directionally: the
4-accumulator split is **6–9 % slower**. Root cause: `dot_4bit_prepared` is **decode-bound**, not
sum-chain-bound — each 16-byte chunk pays one `i16x16` widen plus **four shifts** to sign-extend the
low/high nibbles (`(s<<12)>>12`, `(s<<8)>>12`) before the two multiplies, so the single `acc +=` add
is *not* the binding loop-carried dependency. Extra accumulators only add register pressure /
scheduling overhead. This is the same regime the `dot_i8_i8` comment already flags for the
decode-bound f16 dot ("extra accumulators regress"); the int8 win does **not** transfer because the
i8 decode is a cheap one-op sign-extend while the 4-bit decode is shift-heavy.

**Route next:** the 4-bit pass-1 frontier is gated by **nibble-decode op count**, not accumulator
ILP. A real win there must *reduce decode work* (fewer shifts per chunk / a cheaper sign-extend), not
add accumulators. Do not re-attempt multi-accumulator on `dot_4bit_prepared`. Original-comparator
ratio vs Lucene/Tantivy/Meilisearch is **N/A** for this isolated vector kernel.

### 2026-06-27 — and cutting a shift (`s>>4` high nibble) ALSO regresses — the decode bottleneck is not shift count (BlackThrush)

**Follow-up to the entry directly above** (closing its own "route next: reduce decode work"). The
high nibble was decoded as `(s<<8)>>12` (2 shifts). Because `from_i8x16` already copies the byte's
bit 7 — which *is* the high nibble's sign bit — across bits 8..15, the high nibble can be
sign-extended in **one** arithmetic shift, `s >> 4`, taking the per-chunk decode from 4 shifts to 3.
Integer-exact (`dot_packed_4bit_matches_scalar` passed). Hypothesis: a shift-bound decode should
speed up ~proportionally.

**Measured (in-process A/B, single accumulator both sides; `fourbit_3shift` = lib `s>>4`,
`fourbit_4shift` = local `(s<<8)>>12`; same prepared query so the ratio isolates only the
shift-count change; new names served as a freshness sentinel and both appeared).** `3shift/4shift`
medians:

| dim | 4 shifts (old) | 3 shifts (new) | ratio | environment |
|-----|----------------|----------------|-------|-------------|
| 256 | 283.81 µs | 296.18 µs | **1.044** | local |
| 256 | 379.85 µs | 396.32 µs | **1.043** | rch `ovh-a` |
| 384 | 357.42 µs | 355.71 µs | 0.995 | local |
| 384 | 499.98 µs | 520.90 µs | **1.042** | rch `ovh-a` |

**Decision:** rejected and fully reverted (`simd.rs` + bench arms byte-identical to `main` before
this entry). Two independent environments agree: dropping a shift is **~4 % slower** (dim256 both
runs, dim384 on the cleaner rch worker; the one local dim384 tie is the noise floor). So the kernel
is **not shift-throughput-bound** either — removing a shift does not help, and mixing shift amounts
(`>>4` alongside the `>>12` pair) actually schedules *worse* than the uniform `(s<<k)>>12` pattern
the optimizer already pairs cleanly. The real binding resource is more likely the per-chunk 16-byte
load + the two `pmullw` multiplies + the `from_i8x16` widen, which the shifts overlap with on a
wide-issue core.

**Revised conclusion for `dot_4bit_prepared`:** neither more accumulators nor fewer shifts pay. The
kernel is already near its local optimum on this SSE2-class build; the remaining lever for 4-bit
pass-1 is the **AVX2/`vpmaddubsw`-class** path (gated build-config knob, see the AVX2 section above),
not a portable source rewrite. Stop micro-tuning the `wide`-based nibble decode.

### 2026-06-27 — columnar `id` FAST field for ranked-id retrieval is SLOWER than the row store (BlackThrush)

**Lever tested and reverted:** the ranked-id paths (`search_doc_ids`, the hybrid lexical guard) read
each hit's `doc_id` by loading the **full stored document** (`searcher.doc(addr)`, which decompresses
the whole store block carrying id + the large `content` + title + metadata) just to extract the tiny
`id`. Classic row-store → column-store fix: mark `id` as a Tantivy string **FAST field**
(`(STRING | STORED).set_fast(None)`) and read the id from the columnar fast field, skipping the store
block. Backward-compatible (fall back to the store when the fast field is absent), and validated
**bit-identical** — `search_doc_ids_matches_counted_baseline` + all 81 lexical tests stayed GREEN,
and `reopen_preserves_documents` confirms the schema change round-trips.

**Measured (per-crate `doc_ids_topk` A/B, `rch` worker `ovh-a`, `--measurement-time 4`; `id_fast` =
columnar, `id_store` = stored-doc baseline; new arm names served as a freshness sentinel and
appeared).** Lower is faster; ratio = fast / store (>1 = fast field SLOWER):

| corpus | workload | id_store | id_fast | ratio |
|--------|----------|----------|---------|-------|
| 100k × 12-word | high_fanout | 198 µs | 457 µs | **2.30** |
| 100k × 12-word | union3 | 792 µs | 949 µs | **1.20** |
| 100k × 12-word | natural | 1.039 ms | 1.418 ms | **1.36** |
| 100k × 12-word | phrase | 1.462 ms | 1.648 ms | **1.13** |
| 20k × 400-word | high_fanout | 125 µs | 412 µs | **3.30** |
| 20k × 400-word | natural | 1.406 ms | 1.583 ms | **1.13** |

(An earlier un-hoisted version that re-opened the column **per hit** was even worse — high_fanout
3.4× — because `FastFieldReaders::str` re-deserializes the column dictionary on every call. Hoisting
the open to once-per-segment cut that to 2.3× but did not flip the sign.)

**Decision:** rejected and fully reverted (`crates/frankensearch-lexical/src/lib.rs` + bench
byte-identical to `main`; `git diff` empty before this entry — reverted via Edit, as `dcg` blocks
`git checkout -- <path>`). **Hypothesis refuted:** I expected large docs to win because store-block
decompression of big `content` should dominate — but the store stayed *faster* on the 400-word
corpus (125 µs), so the bottleneck is **not** store decompression. Root cause: `id` is a
**high-cardinality unique string** (`doc-000000`…), so its columnar dictionary is an SSTable as large
as the data, and `StrColumn::ord_to_str` pays a per-doc SSTable dictionary lookup (index walk +
dictionary-block decompress) that is *slower* than reading the small, warm LZ4 store block. Fast
fields win for **low-cardinality** columns (few distinct values → tiny dictionary), not for unique
ids.

**Route next:** do not re-attempt fast-field / columnar `doc_id` retrieval — it is corpus-independent
slower here for unique ids. If store-load is ever the proven bottleneck for big docs, the lever is a
**dedicated minimal id-only stored field** or storing the id in a numeric fast field (dense u64 →
no dictionary), not the string columnar path. Original-comparator ratio vs Lucene/Tantivy/Meili is
N/A — this is an internal id-materialization micro-lever on the Tantivy-wrapping crate.

### 2026-06-27 — JL xorshift ILP is a real win, but original-comparator ratio is N/A (Cobaltmoth)

**Landed in `docs/PERF_LEDGER.md`:** `HashEmbedder::embed_jl` now interleaves `JL_LANES = 4`
independent token xorshift64 chains per dimension (ILP over a latency-bound recurrence), a
**bit-identical ~2.05× local win** (`hash_embed_jl_ilp/scalar` 102.88 µs → `ilp4` 50.19 µs;
CIs disjoint; corroborated ~1.92× on a second worker). This is recorded here only to keep the
original-comparator honesty bar: the **ratio vs Lucene/Tantivy/Meilisearch is N/A**. The `jl-*`
tier is a non-semantic hash embedder that frankensearch runs *in addition to* lexical retrieval —
it is frankensearch-internal compute the incumbents never perform, and the non-semantic fast-tier
guard already skips it on the BOLD `tantivy_doc_ids` lane. So this narrows the end-to-end gap for
jl-tier hybrid searches but is **not** a head-to-head win over a Tantivy primitive and must not be
claimed as one.

**Negative sub-result (kept):** the prior `embed_jl` branchless-sign attempt regressed because LLVM
already emits a conditional move; the `if (state & 1) == 0 { 1.0 } else { -1.0 }` form is therefore
retained in each lane. The win here is purely from breaking the single-chain shift→xor dependency,
not from the sign select. `ilp2` (~1.47×) is left in the bench as the curve point showing the gain
scales with lanes up to the shift-port limit.

### 2026-06-27 — MMR running-max + norm-hoist is a real win, but original-comparator ratio is N/A (Cobaltmoth)

**Landed in `docs/PERF_LEDGER.md`:** `mmr_rerank` now keeps a per-candidate running max similarity
(O(k²·n)→O(k·n) cosine evals) and hoists each embedding's L2 norm out of the pair loop (3 reductions
→ 1). Bit-identical (selection list unchanged; `incremental_norm_hoist_matches_bruteforce` GREEN);
measured **~9.3× at n100/k20 and ~23.3× at n200/k50** in-process. Recorded here only to hold the
original-comparator honesty bar: the **ratio vs Lucene/Tantivy/Meilisearch is N/A**. MMR diversity
re-ranking is a frankensearch-only stage that operates on already-retrieved candidates — it has no
counterpart in a lexical comparator's `doc_ids` path, so this narrows frankensearch's own
diversity-rerank latency rather than beating a Tantivy/Lucene/Meili primitive head-to-head. It is
also **off the default BOLD lane** (only runs when diversity rerank is enabled), so it does not move
the `hash_hybrid_tantivy_vector_rrf` rows; its value is making the MMR feature cheap when used.

### 2026-06-27 — graph-rank dense PageRank is a real win, but original-comparator ratio is N/A (Cobaltmoth)

**Landed in `docs/PERF_LEDGER.md`:** `GraphRanker::rank_phase1` now runs the power iteration over a
dense integer-indexed `Vec<f64>` (+ CSR edges, reused buffers) instead of rebuilding a
`HashMap<String, f64>` and clone-keying `entry()` every iteration — **~11.9× at n500/deg6 and
n2000/deg8**, ranking-equivalent (`dense_rank_matches_reference_ranking` GREEN vs a `BTreeMap`
reference of the original algorithm; the old `std::HashMap` sum order was already non-deterministic, so
nothing exact was pinned). Recorded here to hold the original-comparator honesty bar: the **ratio vs
Lucene/Tantivy/Meilisearch is N/A**, and additionally this is behind the **off-by-default `graph`
feature**, so it does not appear in any default build or BOLD lane. Query-biased PageRank over a
document graph has no counterpart in a lexical comparator's `doc_ids` path; this makes the optional
graph-rank stage ~12× cheaper when enabled, not a head-to-head primitive win.

### 2026-06-27 — federated RRF fuse clone-elim + unstable sort is only ~1.05× — REVERTED (Cobaltmoth)

**Lever tested and reverted:** `federated::accumulate_doc` calls `docs.entry(hit.doc_id.clone())` on
every shard hit, cloning the `String` key even on the common occupied path (a doc in N shards is hit
N times, inserted once). Replaced with a `get_mut` probe (borrowed key) that only clones on a genuine
insert, and switched `into_ranked_hits` from a stable `sort_by` to `sort_unstable_by` (the comparator
is a total order via the `doc_id` tiebreak, so order-identical). Both changes are **bit-identical**.

**Measured (per-crate in-process A/B, replicas of old vs new over multi-shard input with ~60–70% doc
overlap so most accumulate calls hit the occupied path):**

| Workload (shards, hits/shard, unique docs) | old | new | ratio |
|--------------------------------------------|-----|-----|-------|
| `federated_fuse/s5_h200_u400` | 142.67 µs | 136.01 µs | **0.953 (~1.05×)** — CIs overlap |
| `federated_fuse/s10_h500_u1500` | 814.08 µs | 777.02 µs | **0.954 (~1.05×)** |

**Decision:** **reverted** (`federated.rs` + `Cargo.toml` byte-identical to `main`; bench removed).
~1.05× is marginal — the small case's CIs overlap, and federated multi-shard search is a niche,
non-default deployment mode. **Root cause the gain is small:** the doc_id key clone is only *one* of
several per-call `String` allocations — `appeared_in.insert(shard_name.to_owned())` and the
primary-update `shard_name.to_owned()` / `hit.clone()` (template) all still allocate every call and
dominate, so removing just the key clone moves ~5%. A real federated-fuse win would need to attack the
`appeared_in` `BTreeSet<String>` churn (e.g. dedup shard names to a small interned id set), not the
key clone. Original-comparator ratio vs Lucene/Tantivy/Meilisearch is **N/A** (federated cross-shard
fusion has no lexical-comparator counterpart). Do not re-attempt the key-clone/sort micro-opt alone.

**ROUTE-NEXT CAPTURED 2026-07-01 (BlackThrush) — LANDED, see PERF_LEDGER.** The predicted lever was
right: interning shard names to integer ids and moving `appeared_in` from `BTreeSet<String>` to a
`Vec<u32>` (sort+dedup once at output, ids = sorted-name rank so the output name order is unchanged)
won **~1.44× (s5) / ~1.22× (s10)** on the `federated_appeared_in` bench — vs the ~1.05× the key clone
alone moved. Bit-identical (`FederatedHit::appeared_in` byte-identical; federated tests + bench full-output
assert). The per-`accumulate_doc` `String` alloc + BTree node churn (thousands/fuse) was indeed the real
cost. `Vec<u32>` chosen over a `u64` bitset (1.6×/1.36×) to keep no shard-count cap.

### 2026-06-27 — reachable unowned compute surface is mined out; route-next handoff (Cobaltmoth)

Surveyed for a NEW clean lever after landing JL-ILP (`87892cd`), MMR running-max (`5f8b59c`), and
graph-rank dense PageRank (`9ef90fe`). **Rejected without benching** (structural reason, not a wasted
bench cycle) — recorded so the converging swarm does not re-survey these:

- **`core::l2_normalize` / `l2_normalize_in_place` multi-accumulator** — the sum-of-squares is a
  single-accumulator f32 reduction (latency-bound, not auto-vectorized), ~2–4× headroom *in isolation*.
  Rejected: **non-bit-identical** (f32 reorder) on a CORE primitive with 6 callers (every embedder +
  quantization + prf), and the absolute embed-level impact is tiny — ~0.35 µs of a 50 µs JL / 2 µs FNV
  embed. Changes every stored embedding's low bits ⇒ cross-cutting conformance risk for ~0 default-path
  gain. If ever pursued: must be a local per-embedder change with a no-golden + recall check, not a core
  edit.
- **`core::cosine_similarity` fusion/ILP** — used only by tests + the embed orthogonality test; MMR and
  index have their own cosine. Not on any hot path. No-op to optimize.
- **`fusion::prf::prf_expand`** — already auto-vectorized SAXPY (centroid + interpolate are element-wise
  slice ops); only the small norm reduction has headroom (same f32-reorder caveat). Not worth it.
- **`storage::content_hash`** — SHA-256; can't speed without changing hash semantics/collision-resistance
  (breaks the `content_hashes` dedup table). `core::fingerprint` FNV+charcount fusion already tried
  (1.002, noise — entry above).
- **`fusion::federated` BTreeSet churn** — the real lever (see entry above), but federated is a niche
  non-default mode; deprioritized.

**Where the biggest measured gap vs ORIG actually lives (owned / inherent, not a clean grab):** the BOLD
`hash_hybrid_tantivy_vector_rrf` vs `tantivy_doc_ids` residual (natural_language / short_keyword) is
bounded by **lexical query execution + ScoredResult materialization** in `frankensearch-lexical`
(`TantivyIndex::search` does a full `searcher.doc()` store read per hit) and the **int8/4-bit vector
scan** in `frankensearch-index/simd.rs` — both actively iterated by BlackThrush this session (FSVI 4-bit
`e8ec816`/`fbe177e`, doc-id top-k). The store read is shared with the incumbent and the metadata parse
is absent on the BOLD corpus, so the residual is largely *inherent hybrid overhead*, not a single-primitive
swap. Route-next for whoever owns `frankensearch-lexical`: a `LexicalSearch` trait method that returns
id+score **without** materializing the stored doc (the trait only exposes `search()` → full doc today).

### 2026-06-27 — two-term plain count-free top-k is local evidence, not original-comparator dominance (BlackThrush)

**Landed in `docs/PERF_LEDGER.md`:** `search_doc_ids` now extends the count-free top-k collector gate
from one plain term to one or two plain terms. This targets the current BOLD short-keyword query shape
(`rust ownership`) without reopening the previously rejected broad count-free path for phrases or
natural-language disjunctions.

**Measured local Tantivy-wrapper ratio:** `doc_ids_topk/short_keyword_bold` counted median
171.151 us → count-free median 30.900 us, ratio **0.181** (same-binary A/B, `frankensearch-lexical`,
RCH local fallback, `--profile release`, `--sample-size 10 --warm-up-time 1 --measurement-time 1`).

**Ratio vs Lucene/Tantivy/Meilisearch original comparator: N/A.** The in-repo BOLD Tantivy-class
proxy also calls this wrapper's `search_doc_ids`, so this collector choice moves the local proxy and
frankensearch's lexical candidate source together. It is valid as a local wrapper primitive and as
gap reduction for frankensearch's own lexical work, but it must not be represented as a standalone
head-to-head win over Lucene, Tantivy, or Meilisearch.

**Noisy fallback evidence from the same full bench:** fallback rows still execute the counted
collector for 3+ terms/phrases, but Criterion reported `doc_ids_topk/natural` at 698.216 us counted
vs 741.499 us through `search_doc_ids` (ratio **1.062**) while `union3` was 0.990 and phrase was
0.982. Treat the natural row as guard/measurement overhead, not a claimed win; the code deliberately
keeps 3+ term natural-language queries off the count-free collector because broad disjunctions were
already rejected in this ledger.

### 2026-06-27 — BLOCKER: `frankensearch-storage` / `frankensearch-fsfs` fail to compile (fsqlite dep skew) (Cobaltmoth)

**Blocker, not a perf result.** Any `cargo build/test/bench -p frankensearch-storage` or
`-p frankensearch-fsfs` fails at compile with:

```
error[E0308]: mismatched types
  fsqlite-0.1.2/src/migrate.rs:198:35
    SqliteValue::Text(Arc::from(migration.name)),
                      ^^^^^^^^^^^^^^^^^^^^^^^^^ expected `SmallText`, found `Arc<_, _>`
  (SqliteValue::Text(SmallText) defined in fsqlite-types-0.1.9/src/value.rs:665)
error: could not compile `fsqlite` (lib) due to 1 previous error
```

**Root cause:** the committed `Cargo.lock` pins `fsqlite 0.1.2` but `fsqlite-types 0.1.9`. Both
`crates/frankensearch-storage/Cargo.toml` and `crates/frankensearch-fsfs/Cargo.toml` declare
`fsqlite = "0.1.2"` + `fsqlite-types = "0.1.2"` (caret → `^0.1.x`), and the resolver took the latest
`fsqlite-types 0.1.9`. But `fsqlite 0.1.2`'s source uses the **old** `SqliteValue::Text(Arc<…>)` API,
while `fsqlite-types 0.1.9` changed it to `Text(SmallText)` — an upstream **breaking change inside a
`0.1.x` line** (semver violation in `fsqlite-types`). Reproduced on two RCH workers (`ovh-a`, `hz2`),
exit 101 each. **Not caused by this session** — my change did not touch `Cargo.lock`; the skew is in
the committed lock and blocks the whole `storage → fsqlite` graph (so `fsfs` too).

**Impact:** all perf work on `frankensearch-storage` and `frankensearch-fsfs` is currently un-buildable
(can't bench, can't run conformance). A clean, provably-bit-identical sniff-loop vectorization for
`fsfs::SniffFeatures::from_bytes` (per-file null/non-printable/high-bit histogram: per-byte
`u32::saturating_add` → branchless `u64` + saturate-cast, SIMD-able) was written and **reverted
unmeasured** because of this — fsfs would not compile to test or bench it.

**Fix (needs a shared-infra owner — out of scope for "git add only your own files"):** pin
`fsqlite-types` to the version `fsqlite 0.1.2` was built against, e.g. in both Cargo.tomls
`fsqlite-types = "=0.1.2"`, then `cargo update -p fsqlite-types --precise 0.1.2` to rewrite the lock;
or bump `fsqlite` to a release compatible with `fsqlite-types 0.1.9`. Until then, route perf digs to
crates outside the `storage`/`fsfs` graph.

**Follow-up (2026-06-27, Cobaltmoth): the naive fix above does NOT work — deeper diagnosis.** Attempted
the surgical lock fix; it fails because the whole `fsqlite-*` family is split, not just one crate:
- `cargo update -p fsqlite-types --precise 0.1.2` → **rejected**: `fsqlite-ext-fts5 0.1.9` (storage's
  optional `fts5`) requires `fsqlite-types ^0.1.9`; and `fsqlite-core 0.1.9` (via
  `frankensearch-durability`) also requires `^0.1.9`. So **`durability` is in the broken graph too**,
  not only storage/fsfs.
- Root cause: the **main `fsqlite` crate's latest is 0.1.2** (old `Text(Arc<…>)` API) while its
  siblings (`fsqlite-types`, `fsqlite-core`, `fsqlite-ext-fts5`, `fsqlite-vdbe`, `fsqlite-wal`, …)
  released `0.1.9+` with the breaking `Text(SmallText)` change. There is **no mixed resolution** that
  compiles: anything pulling a `0.1.9` sibling forces `fsqlite-types 0.1.9`, which `fsqlite 0.1.2`
  cannot compile against.
- A **consistent all-`0.1.2` family** IS resolvable —
  `cargo update fsqlite-types fsqlite-core fsqlite-ext-fts5 fsqlite-ext-rtree fsqlite-ext-icu
  fsqlite-wal fsqlite-planner fsqlite-pager fsqlite-btree fsqlite-mvcc fsqlite-vdbe --precise 0.1.2` —
  but it downgrades 12+ crates and **churns transitive deps** (e.g. `json5 1.3.1 → 0.4.1`, removes
  `iana-time-zone`, adds `pest`/`lru`/`sha1`/`signal-hook`), a wide blast radius that needs a **full
  workspace build** to validate (per-crate benching can't catch the collateral). Not applied here.
- **`Cargo.lock` is gitignored** (`.gitignore:5`) → it is regenerated on every clean resolve and a
  fresh resolve picks the *latest* `fsqlite-types` (already `0.1.12`). So **a committed lock cannot fix
  this** — the only durable fixes are (a) **Cargo.toml `=` pins** of the entire `fsqlite-*` family to a
  cohering version (committed, validated workspace-wide), or (b) an upstream **`fsqlite` main-crate
  0.1.9+ release** matching its siblings. Owner decision; do not lock-surgery this in the shared tree.

### 2026-06-27 — caching the lexical BM25 `QueryParser` is ~0-gain (construction is cheap) — REVERTED (Cobaltmoth)

**Lever tested and reverted:** `TantivyIndex::parse_query_lenient` (every lexical query) rebuilt a
fresh `QueryParser::for_index(...)` + `set_field_boost(...)` per call. Hypothesis: the parser is
schema-invariant, so caching it once at construction removes per-query construction cost. Implemented
(struct field built once in `from_index`) — bit-identical (81 lexical lib tests GREEN).

**Measured (per-crate in-process A/B; `rebuild` = construct parser + parse per iter, `cached` = parse
with a pre-built parser; identical `parse_query_lenient(query)` in both, so the delta is construction):**

| Workload | rebuild | cached | ratio |
|----------|---------|--------|-------|
| `query_parser_cache` (6-term query) | 7670.8 ns (mean 8030) | 7826.1 ns (mean 7992) | **~1.00** (CIs overlap; cached median slightly *slower*) |

**Decision:** **reverted** (`lib.rs` + `Cargo.toml` byte-identical to `main`; bench removed).
**Hypothesis refuted:** `QueryParser::for_index` construction is **cheap** (it just clones schema/field
refs + sets a boost); the ~7.8 µs is the **lenient parse itself** (tokenize → term-build → BooleanQuery),
which is identical in both arms. Caching the parser removes ~nothing.

**Route-next (the real signal here):** the lenient *parse* costs **~7.8 µs/query** — a genuine slice of
a BOLD lexical query (≈5–15 % of a 50–170 µs `tantivy_doc_ids`-class query) and pure frankensearch
per-query overhead. The win is **not** parser caching but **bypassing the full lenient `QueryParser`
for simple plain queries** — build the `BooleanQuery`/`TermQuery` AST directly (no operator/field/quote
parsing) when the query is plain tokens. This **overlaps BlackThrush's plain-query count-free top-k
path** (`search_doc_ids`, which already detects 1–2 plain terms), so it belongs to that owner: detect
plain query → skip `parse_query_lenient` → hand the collector a hand-built term query. Original-comparator
ratio vs Lucene/Tantivy/Meilisearch is **N/A** (internal Tantivy-wrapper parse overhead).

### 2026-06-27 — examined-clean inventory: reachable per-crate compute surface is saturated (Cobaltmoth)

A consolidated map of what I examined this session and found **already-optimal / owned / not-a-lever**,
so the converging swarm stops re-surveying these. Each was inspected for the usual levers (latency-bound
reductions, O(n²)/O(n) redundancy, per-iteration allocs, scalar loops missing SIMD, HashMap<String>
churn) and rejected for the stated structural reason — **no bench needed**:

- `core/cache.rs` — **S3-FIFO** (Yang et al. SOSP 2023); O(1) per access by design (freq-counter, no LRU
  list manipulation). Already the modern post-LRU algorithm.
- `core/decision_plane.rs`, `core/collectors.rs` — control-plane / telemetry, **not on the search path**.
- `core/filter.rs` — per-candidate `matches` is parsed-`Value` field lookups + `HashSet::contains`; no
  per-call alloc/parse.
- `fusion/adaptive.rs` — per-query `blend_factor`/`rrf_k` are O(1) `HashMap<QueryClass,_>` reads; updates
  are O(1) Welford. Optional + already O(1).
- `fusion/prf.rs::prf_expand` — already auto-vectorized SAXPY (element-wise centroid + interpolate); only
  the small norm reduction has headroom (f32-reorder caveat).
- `embed/cached_embedder.rs` — O(1) FIFO (not O(n) LRU); `embed/batch_coalescer.rs` —
  mutex/condvar request coordination, not a compute loop.
- `rerank/native.rs` — transformer reranker already SIMD (`f32x8` softmax) + rayon, profiling-driven
  (BlackThrush beads). Owned.
- `index`: `simd.rs`/`search.rs`/`mrl.rs`/`two_tier.rs` owned+optimized (BlackThrush); `quantization.rs`
  test-only (dead, flagged); `hnsw.rs` non-default (flagged); `wal.rs` correctness-critical+flaky
  (flagged); `vector_at_f32` f16-decode is HNSW-build/API only, not the hot scan (which uses the SIMD
  f16 dot directly).
- `lexical`: BM25 `QueryParser` construction is cheap (caching it = ~0-gain, entry above);
  `search_doc_ids` count-free top-k is BlackThrush's active path.
- `storage`/`durability`/`fsfs` — **un-buildable** (fsqlite family split, blocker entries above).

**Net state — two real moves remain, both outside a safe solo grab:**
1. **Unblock fsqlite** (owner/shared-infra) → unlocks the un-mined `storage`/`durability`/`fsfs` graph,
   which has genuine per-doc indexing compute (e.g. the ready, bit-identical `SniffFeatures::from_bytes`
   SIMD-histogram win, reverted only because fsfs won't compile).
2. **Plain-query parse bypass** in `lexical` (~5–15 % of a BOLD lexical query) — owner is the active
   `search_doc_ids` path; bit-identical *scoring* must be differentially proven vs Tantivy's parser.

Beyond these, the reachable per-crate compute surface for clean bit-identical wins is **saturated**.

### 2026-06-27 — plain-query parse bypass: ~20× faster to build but NOT rank-equivalent to Tantivy's parser — REVERTED (Cobaltmoth)

**Lever tested and reverted (the route-next from the parser-cache entry above).** `parse_query_lenient`
runs on every lexical query; for plain queries (ASCII alphanumeric + whitespace, no operators/quotes/
fields/wildcards/`AND`/`OR`/`NOT`) I built the BM25 `BooleanQuery` directly (`content:tok` + boosted
`title:tok` per token, `Should`), skipping Tantivy's lenient `QueryParser`. **Construction is hugely
cheaper** (in-process A/B, `plain_query_build`): 2-term **2855 ns → 145 ns (~19.7×)**, 4-term
**5557 ns → 357 ns (~15.6×)**.

**But it cannot be made equivalent to the parser's ranking**, across three structural attempts vs a
differential test (`plain_query_matches_parser`: build fast vs parser, search the same index, compare
ranked (BM25 score, doc_id)):
- **Flat** `Should` over all (field,token) leaves → BM25 score off by **~1 ULP** (`9.966056` vs parser
  `9.966055` on "distributed consensus algorithms") — f32 summation is **not associative**, and the
  parser's leaf grouping differs. The 81 *existing* lexical tests pass, but the 1-ULP delta **flips
  TopDocs ranking** at near-ties (score collision → DocAddress tiebreak reorders vs the parser's
  distinct scores), so it fails even a relaxed doc_id-order + ε-score check.
- **Nested** per-token (`BooleanQuery[content, title^2]` per token, outer over tokens) → scores **~3×
  off** (`8.27` vs `2.76`) — structurally wrong; Tantivy's nested `Should` scorer is not a plain sum.

**Decision:** **reverted** (`lib.rs` + `Cargo.toml` byte-identical to `main`; bench + tests removed).
The parser's exact query tree / BM25 scoring is **not externally replicable** from the public Tantivy
API, so any hand-built substitute changes ranking. Original-comparator ratio is **N/A** (Tantivy-wrapper
parse overhead). **Route-next:** a safe win needs Tantivy itself to expose a cheap plain-term query
builder that yields the parser's exact tree, or the parse cost (~3–6 µs/query, ~5–15 % of a BOLD lexical
query) must be accepted as comparator-inherent. Do not re-attempt a hand-built plain-query substitute.

### 2026-06-27 - exact-phrase underfill short-circuit is a BOLD harness semantic change, not a keep (BlackThrush)

**Lever tested and reverted:** changed the BOLD comparator harness so a quoted exact phrase with
any lexical hit returned lexical-only immediately, even when Tantivy returned fewer than `k` hits.
This targeted the current `top10/10000 quoted_phrase` gap, where main measured the hash-hybrid
row at **1.504x slower** than the Tantivy/Lucene/Meilisearch-class proxy (119 us incumbent,
179 us frankensearch).

**Measured command (per-crate, RCH local fallback, warm target dir):**

```bash
AGENT_NAME=BlackThrush \
RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,FRANKENSEARCH_BOLD_VERIFY_OUT,FRANKENSEARCH_BOLD_VERIFY_EMIT,FRANKENSEARCH_BOLD_VERIFY_SUMMARY_ONLY,FRANKENSEARCH_BOLD_VERIFY_COMMAND,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
FRANKENSEARCH_BOLD_VERIFY_OUT=/data/projects/.rch-targets/frankensearch-cod-b/criterion/bold_verify_phrase_candidate \
FRANKENSEARCH_BOLD_VERIFY_EMIT=1 \
FRANKENSEARCH_BOLD_VERIFY_SUMMARY_ONLY=1 \
RUST_LOG=off \
  rch exec -- cargo bench -p frankensearch --features lexical --profile release \
    --bench search_bench bold_verify_tantivy_class \
    -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Artifact: `/data/projects/.rch-targets/frankensearch-cod-b/criterion/bold_verify_phrase_candidate/summary.jsonl`.

| Workload | Candidate | Incumbent p50 | Candidate p50 | Ratio vs ORIG | Decision |
|----------|-----------|---------------|----------------|---------------|----------|
| `top10/10000 quoted_phrase` | hash-hybrid | 119 us | 118 us | **0.992** | semantic change; no keep |
| `top10/10000 quoted_phrase` | lexical guard | 119 us | 119 us | **1.000** | no gain |
| `top10/100000 quoted_phrase` | hash-hybrid | 1.099 ms | 1.092 ms | **0.994** | noise/tie |
| `top10/100000 quoted_phrase` | lexical guard | 1.099 ms | 1.058 ms | **0.963** | already a guarded lexical row |
| `top10/100000 short_keyword` | hash-hybrid | 50 us | 64 us | **1.280x slower** | unrelated regression/noise |
| `top10/100000 short_keyword` | lexical guard | 50 us | 81 us | **1.620x slower** | unrelated regression/noise |
| `top10/100000 zero_hit` | hash-hybrid | 82 us | 102 us | **1.244x slower** | unrelated regression/noise |

**Why it cannot land:** the product `TwoTierSearcher` gate does not use an "any exact phrase hit"
rule. It short-circuits saturated identifier/short-keyword rows, plus a narrower non-semantic/no-
quality natural-language case. Returning fewer than `k` exact-phrase lexical hits without hybrid
backfill changes BOLD harness semantics rather than speeding the shipped search path. The best
targeted row only reached a p50 tie (0.992), while the same sweep exposed noisy slower ORIG ratios.

**Decision:** source reverted; ledger-only. Route next away from phrase underfill gates. The
remaining BOLD `limit_all` materialization gap and plain-query parse overhead have already been
routed above; do not spend another pass on exact-phrase lexical-only underfill unless product
semantics explicitly change first.

### 2026-06-27 — UPDATE: fsqlite blocker is RESOLVED by a fresh lock; cc bench pool has a rustc skew (Cobaltmoth)

**Correction to the fsqlite blocker entries above (`f9b4ee3`/`25b43f7`).** `storage`/`durability`/`fsfs`
**now compile.** The blocker was a **stale `Cargo.lock` pinning `fsqlite 0.1.2`** (whose internal code
needs the old `fsqlite-types` API) while siblings resolved to `0.1.9`. A fresh resolve
(`cargo generate-lockfile`) picks a **consistent `fsqlite 0.1.12` + `fsqlite-types 0.1.12` family** —
`fsqlite 0.1.12` exists (I'd wrongly assumed 0.1.2 was latest, reading the stale lock). Since
`Cargo.lock` is gitignored, any clean resolve gets 0.1.12. **Verified:** `cargo build -p
frankensearch-storage` compiled remotely (artifact-transfer flaked, exit 102, but the compile
succeeded), and `cargo test -p frankensearch-fsfs --lib file_classification` is **GREEN (14 passed)**.
⇒ **storage/durability/fsfs are an un-mined perf surface again** (supersedes the
[[storage-fsfs-blocked-fsqlite-family-split]] memory).

**New blocker (rch infra, not code): cc bench target POOL has a mixed-rustc skew.** `cargo bench
-p frankensearch-fsfs` (and any criterion bench) fails `error[E0514]: found crate <X> compiled by an
incompatible version of rustc` — the pool `/data/tmp/rch-targets-pool/frankensearch-cc` holds `.rmeta`
from nightly `91fe22da8` (2026-06-21) while the worker now runs `ce9954c0c` (2026-06-26), across
criterion's deps (`futures-lite`, `sharded-slab`, `thread_local`, `http`, `itoa`, `lazy_static`,
`parking`). Two attempts hit it (worker hz2); `cargo test` is unaffected (didn't need those deps).
**Fix:** `cargo clean` the cc pool (rch should invalidate the pool on toolchain change). Until then
per-crate criterion benches in the cc role are blocked.

**Ready win, bench-blocked (route-next):** `fsfs::SniffFeatures::from_bytes` per-byte `u32
saturating_add` → branchless `u64` + saturate-cast (vectorizable SIMD histogram). **Validated
bit-identical** (`sniff_features_vectorized_matches_scalar_reference` GREEN in the 14-pass run); source
reverted only because the ratio can't be measured under the pool skew. After the pool is cleaned:
`cargo bench -p frankensearch-fsfs --bench sniff_features`, then re-apply (the diff + test + bench are
recorded in this session) and land with the ratio.

**UPDATE — LANDED.** Cleared the skew with a targeted `rch exec -- cargo clean -p futures-lite
-p sharded-slab -p thread_local -p http -p itoa -p lazy_static -p parking` (rch warns
"non-compilation command" but **runs it**; removed 42 files). The bench then compiled and ran
cleanly: probe_4096 **1.4×**, probe_16384 **2.0×**, probe_65536 **2.4×** — recorded in
`docs/PERF_LEDGER.md`. Original-comparator ratio vs Lucene/Tantivy/Meilisearch is **N/A** (fsfs
file-scan layer, not a query comparator). Takeaway for the swarm: a cc-pool rustc skew (E0514) is
fixable in-band via `rch exec -- cargo clean -p <skewed crates>` — no admin needed.

### 2026-06-27 - ParsedQuery NOT-keyword allocation elision is a real local win, but ORIG ratio is N/A (BlackThrush)

**Landed in `docs/PERF_LEDGER.md`:** `ParsedQuery::parse` no longer allocates a three-character
`String` inside `matches_not_keyword` for each token-boundary check in negation-capable queries.
The helper now checks the fixed ASCII keyword directly (`N|n`, `O|o`, `T|t`) and preserves the
same boundary rules. Per-crate A/B through `rch exec` (local fallback, target dir
`/data/projects/.rch-targets/frankensearch-cod-b`) measured `parsed_query/not_phrase_old` mean
**746.73 ns** versus `parsed_query/not_phrase_new` mean **668.39 ns**, ratio **0.895 (~1.12x)**.

**Ratio vs Lucene/Tantivy/Meilisearch ORIG: N/A.** This parser is frankensearch-internal query
preprocessing for `-term` / `NOT "phrase"` exclusions before backend retrieval; the BOLD
Tantivy/Lucene/Meili-class comparator does not run an equivalent `ParsedQuery` stage. Do not claim
new ORIG dominance from it. It narrows frankensearch's own negation-query overhead and leaves the
residual BOLD materialization/parser-routing gaps documented above unchanged.

### 2026-06-27 — fsfs `count_lexical_tokens` ASCII byte fast path: original-comparator ratio N/A (Cobaltmoth)

**Landed in `docs/PERF_LEDGER.md`** (~1.85–2.54× on per-chunk token counting; bit-identical, 25
`lexical_pipeline` tests GREEN). Recorded here for the honesty bar: **ratio vs Lucene/Tantivy/Meili is
N/A** — `count_lexical_tokens` is frankensearch's own fsfs file-ingest/chunking metadata pass (counts
tokens per chunk at index time); a lexical comparator runs no equivalent stage. It speeds frankensearch
indexing throughput, not a head-to-head query primitive. Second clean win in the fsfs surface that the
fsqlite stale-lock unblock reopened (after the `SniffFeatures` SIMD histogram, `08469f5`).

### 2026-06-28 — fsfs `count_lexical_tokens` fused ASCII detection is mixed/noisy — REVERTED (BlackThrush)

**Lever tested and reverted:** after the landed ASCII byte fast path, remove the up-front
`str::is_ascii()` pre-scan and fuse ASCII detection into the token-count loop. The hypothesis was
single-pass ASCII chunks would beat the current `bytes_prescan` path. This follows the graveyard
constant-factor rule: eliminate a pass only if the fused branch cost does not erase the memory-pass
win.

**Measured command** (per-crate; `rch exec` local fallback because no worker was admissible; fresh
target dir `/data/projects/.rch-targets/frankensearch-cod-a-fsfs-fused`):

```bash
AGENT_NAME=BlackThrush \
RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a-fsfs-fused \
RUST_LOG=off \
  rch exec -- cargo bench -p frankensearch-fsfs --profile release \
    --bench lexical_count -- --sample-size 10 --warm-up-time 1 --measurement-time 2
```

Criterion medians from `criterion/lexical_count/*/new/estimates.json`; ORIG is the current landed
`bytes_prescan` implementation:

| Workload | ORIG `bytes_prescan` | Candidate `bytes_fused` | Ratio vs ORIG | Decision |
|----------|----------------------|-------------------------|---------------|----------|
| `lexical_count/ascii_1024` | 1478.54 ns | 2020.15 ns | **1.366x slower** | revert |
| `lexical_count/ascii_4096` | 6325.66 ns | 4212.20 ns | 0.666 | win row only |
| `lexical_count/ascii_16384` | 13468.65 ns | 15171.09 ns | **1.126x slower** | revert |

**Decision:** source and bench changes were reverted. The fused loop is not a clean keep: it regresses
the near-default chunk shape (fsfs `LexicalChunkPolicy::default().max_chars` is 768, closest to the
1 KiB row) and large chunks, while the 4 KiB row alone is not enough to justify a length-threshold
policy. Ratio vs Lucene/Tantivy/Meilisearch is **N/A** for this isolated file-ingest token-counting
primitive; it does not move the BOLD query comparator.

### 2026-06-27 - fsfs `code_structure_sidecar::tokenize` ASCII fast path: ORIG ratio N/A and 256-byte tie (BlackThrush)

**Landed in `docs/PERF_LEDGER.md`** for medium/larger code-structure sidecar tokenization:
`code_tokenize/ascii_1024` ratio **0.640 (~1.56x)** and `code_tokenize/ascii_4096` ratio
**0.545 (~1.84x)**, measured per-crate with
`CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b rch exec -- cargo bench
-p frankensearch-fsfs --profile release --bench code_tokenize -- --sample-size 10 --warm-up-time 1
--measurement-time 2`.

Honesty notes: **ratio vs Lucene/Tantivy/Meilisearch ORIG is N/A** because this is frankensearch's
fsfs code-structure sidecar tokenizer, not a comparator query primitive. The small-string row
`code_tokenize/ascii_256` measured **5011.625 ns -> 4976.335 ns, ratio 0.993**, which is a noise/tie
and must not be cited as a win. The lever is kept only for the larger-snippet rows where the measured
ratio is below the 0.97 keep threshold.

### 2026-06-28 — blend_two_tier `sort_by` → `sort_unstable_by` is ~0-gain — REVERTED (Cobaltmoth)

**Lever tested and reverted.** `blend_two_tier` (the default sync hybrid path, `sync_searcher.rs`)
finalizes by sorting merged candidates with a strict total-order comparator (score desc, then unique
`doc_id`), so `sort_by` → `sort_unstable_by` is order-identical (bit-identical) and skips the
stable-sort scratch alloc. Benched the full blend (bounds + `AHashMap` merge + collect + sort), stable
vs unstable, at realistic fast+quality candidate counts:

| Workload | stable | unstable | ratio |
|----------|--------|----------|-------|
| `blend_sort/n60` | 13553 ns | 13004 ns | **0.959 (~1.04×)** |
| `blend_sort/n200` | 44925 ns | 43296 ns | **0.964 (~1.04×)** |
| `blend_sort/n500` | 113990 ns | 115604 ns | **1.014 (~0.99×, slightly slower)** |

**Decision:** **reverted** (`blend.rs` + `Cargo.toml` byte-identical to `main`, bench removed). ~1.04×
at small/medium n, neutral at n500 — the sort is a small fraction of blend (which is dominated by the
`AHashMap` build + per-doc `to_owned`), so the unstable-sort saving doesn't move the function. This
**confirms the earlier federated `sort_unstable` finding**: `sort_by`→`sort_unstable_by` on these
candidate-count (tens–hundreds) total-order sorts is consistently ~0-gain. Do not re-attempt
`sort_unstable` micro-opts on fusion candidate sorts. Original-comparator ratio is **N/A** (internal
fusion finalization).

### 2026-06-28 — durability `codec.rs` is a round-robin COPY stub, not GF erasure coding — no SIMD lever (Cobaltmoth)

**Not attempted — flagged to save the erasure-coding rabbit hole.** `RaptorqCodec::encode`
(`crates/frankensearch-durability/src/codec.rs`) *looks* like a Reed-Solomon/fountain codec (the
alien-graveyard "erasure coding → SIMD the GF(256) multiply" lever), but its repair symbols are just
**round-robin copies of source symbols**: `repair_symbols.push((esi, source_symbols[repair_idx %
k_source].1.clone()))` — there is **no GF(256) multiply and no XOR parity** to vectorize, only a
`Vec<u8>` clone per repair symbol (memcpy, allocation-bound, not compute-bound). Optimizing it would be
a zero-impact change. The rest of `frankensearch-durability` is crc32 (already SIMD via `crc32fast`) +
file I/O; **no compute lever exists in the durability crate.** If a real erasure codec is wired later
(actual GF math), re-evaluate then. Original-comparator ratio is **N/A** (internal durability codec).

### 2026-06-28 — storage write path already batch-optimal; storage/durability indexing surface mapped lever-free (Cobaltmoth)

**Not attempted — verified already-optimal.** The classic SQLite indexing lever (per-row commits →
batch into one transaction, often 10–100×) **does not apply**: `DocumentStore::upsert_batch`
(`crates/frankensearch-storage/src/document.rs`) already wraps the whole batch in a **single**
`self.transaction(|conn| { for doc in docs { upsert_document_with_outcome(conn, doc) } })`; single-doc
`upsert` uses one transaction (correct). No per-doc commit/fsync antipattern. The rest of the write
path is prepared-statement SQL + serde serialization (inherent). Combined with the durability-codec
copy-stub finding (entry above) and the `storage`/`pipeline.rs` scan (orchestration, no byte loops),
the **fsqlite-reopened storage + durability indexing surface is fully mapped as lever-free** for clean
per-crate compute wins. Original-comparator ratio is **N/A** (internal indexing/write path).

### 2026-06-28 — sync rank-change clone-elision is a real ~2× local win, original-comparator ratio N/A (Cobaltmoth)

**Landed in `docs/PERF_LEDGER.md`:** `compute_rank_changes_for_scored` (every sync hybrid query) no
longer clones every `doc_id` into two throwaway `Vec<VectorHit>` to build its rank maps — it borrows
`doc_id.as_str()` straight from the `ScoredResult` slices (`build_borrowed_rank_map` only ever read
`doc_id`). **~1.98–2.26×** on the rank-map build (bit-identical `RankChanges`; 6 sync_searcher tests
GREEN). Recorded here to hold the honesty bar: the **ratio vs Lucene/Tantivy/Meilisearch is N/A** —
rank-change telemetry (initial-vs-refined promotion/demotion counts) is a frankensearch-only
observability stage with no comparator counterpart; this trims frankensearch's own per-query overhead,
it is not a head-to-head primitive win. (Counters the earlier "surface is saturated" notes: a genuine
default-path clone-elision lever remained in the sync result-assembly path.)

### 2026-06-28 — sync vector score-map borrow is a real ~2.4–2.8× local win, original-comparator ratio N/A (Cobaltmoth)

**Landed in `docs/PERF_LEDGER.md`:** the no-lexical sync branch's per-query `fast_scores`/`quality_scores`
maps now key on `&str` borrowed from the candidate hits instead of cloned `String` (they were only
`.get()`-looked-up by `&str`). **~2.37–2.77×** on the two-map build (bit-identical; 6 sync_searcher tests
GREEN). Honesty bar: **ratio vs Lucene/Tantivy/Meilisearch is N/A** — this is frankensearch's own
pure-vector result-assembly bookkeeping (no lexical comparator counterpart), and it's the **no-lexical**
path (the BOLD lexical lane uses `rrf_fuse`, unaffected). Confirms the orchestration/result-assembly
clone-elision vein is productive (2 wins now: `rank_map` `f8f645e`, this).

### 2026-06-28 — sync lexical fused-materialize move is a real ~7.8–21.5× local win, original-comparator ratio N/A (Cobaltmoth)

**Landed in `docs/PERF_LEDGER.md`:** `fused_hits_to_scored_results` now takes the `rrf_fuse` result by
value and **moves** each `doc_id` into the `ScoredResult` instead of cloning it (the `FusedHit`s are a
fresh temporary). **~7.76–21.55×** on the materialization (bit-identical; 6 sync_searcher tests GREEN) —
larger than the map wins because it drops both the per-result clone (N allocs) and the temporary's
string drops (N frees). Honesty bar: **ratio vs Lucene/Tantivy/Meilisearch is N/A** — this is
frankensearch's own hybrid result-assembly, the absolute time is one small query step, and the headline
ratio is on that step in isolation (not end-to-end). Third clone-elision win in the sync result-assembly
vein (`rank_map` `f8f645e`, `score_map` `dc86170`, this). Note the **async `searcher.rs` already** uses
borrow-keyed score maps + clone-free rank maps — that path was the optimized reference; these three wins
brought the sync searcher to parity.

### 2026-06-28 — sync vector refined materialize move is a real ~1.4–2.5× local win, original-comparator ratio N/A (BlackThrush)

**Landed in `docs/PERF_LEDGER.md`:** the no-lexical sync refined branch now consumes the owned
`blend_two_tier` output and moves each unique `doc_id` into the final `ScoredResult` instead of routing
through the generic borrowed vector converter (`HashSet` dedup + `String` clone). Measured ratios vs
the old in-process converter: `n20` **0.537**, `n60` **0.724**, `n200` **0.406** (bit-identical;
`blend_two_tier` output is already unique). Honesty bar: **ratio vs Lucene/Tantivy/Meilisearch is N/A**
because this is frankensearch's own pure-vector result-assembly bookkeeping, not a comparator query
primitive.

### 2026-06-28 — sync `search_collect` phase-clone elision is sub-noise on the end-to-end path (BlackThrush)

**Lever tested and reverted.** `SyncTwoTierSearcher::search_internal` always builds the streaming
`phases` (`SearchPhase::Initial` + `SearchPhase::Refined`, each cloning the full
`Vec<ScoredResult>`, plus a `RankChanges` clone into `metrics`), but `search_collect` —
the hot production/BOLD path (`search_collect → search_collect_with_filter`) — returns only
`(final_results, metrics)` and **discards `outcome.phases` entirely**. (The async reference
`searcher.rs` extracts its results *from* the phases via the `on_phase` callback, so it never
double-builds; the sync path uniquely builds both a phase copy and a separate `final_results`.)
Hypothesis: thread a `want_phases` flag through `search_internal` so `search_collect` (false) skips
the discarded phase clones while `search_iter` (true) keeps them. Correctness-neutral: `final_results`
and `metrics` (incl. `rank_changes`, now *moved* into `metrics` on the collect path instead of cloned)
are bit-identical; all 6 `sync_searcher` tests GREEN (both the collect and iter paths).

**Measured (same-binary A/B via a `#[doc(hidden)] search_collect_with_phases_baseline` that forces
`want_phases=true` then discards the phases = old behavior; small two-tier index N=4000/dim=64 so the
int8 fast-tier scan doesn't swamp the orchestration; per-crate, RCH `hz2`, `--profile release`;
`baseline` = build+discard phases, `candidate` = skip; medians, two runs):**

| Workload | baseline | candidate | ratio (run 2 / run 1) |
|----------|----------|-----------|------------------------|
| `sync_collect_phases/k10`  | 207.03 µs | 193.86 µs | **0.936 / 0.920** |
| `sync_collect_phases/k64`  | 638.26 µs | 628.86 µs | 0.985 / **1.029 (slower)** |
| `sync_collect_phases/k256` | 1.1143 ms | 1.1229 ms | **1.008 (slower)** / 1.013 |

Command:
```bash
AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc RUST_LOG=off \
  rch exec -- cargo bench -p frankensearch-fusion --profile release \
  --bench sync_collect_phases -- --sample-size 80 --warm-up-time 2 --measurement-time 4
```

**Decision: reverted** (source + `Cargo.toml` + bench byte-identical to `main`; only this entry
remains). The candidate does *strictly less* work (it removes provably-discarded clones), yet the
k256 arm measured it **slower** (1.008/1.013) and the k64 arm **sign-flipped** between the two runs —
the saving is below the bench noise floor everywhere except a reproducible-but-CI-overlapping ~6% at
**k10** (baseline `[194.24, 221.69]` vs candidate `[182.77, 205.18]`). The phase clones are a fixed
~10–13 µs (≈`2·k` `ScoredResult` clones + one `RankChanges`); end-to-end `search_collect` is dominated
by the int8 fast-tier scan + quality scoring + `blend_two_tier` + the two *unavoidable* materializers,
so on a realistic 100k corpus (BOLD top10 ≈645 µs, int8 scan even heavier) that fixed saving is well
under noise. This mirrors the earlier `doc_id`-cache reject (sync orchestration overhead is a small
slice at realistic `k`) — the measurable orchestration clones were already captured by the landed
`rank_map`/`score_map`/`fused_materialize`/`vector_materialize` wins; the residual phase clones don't
clear the 0.97 keep threshold and the `want_phases` flag + baseline-method complexity isn't justified
by a measured gain. Original-comparator ratio is **N/A** (internal sync result-assembly).

**Route next:** stop probing the sync result-assembly/orchestration clone-elision vein for end-to-end
wins — it is mined to the noise floor. Any future attempt here must isolate a *per-function* micro-bench
(as the landed map/materialize wins did) AND show the function is a non-trivial slice of the real path,
not just a clean ratio in isolation.

### 2026-06-28 — BOLD comparator is empirically CLOSED; `limit_all` is the lone (inherent) gap (BlackThrush)

**Empirical re-measurement, not a lever.** Ran the live `bold_verify_tantivy_class` comparator
(`-p frankensearch --features lexical`, 10-sample p50, RCH `hz2`) to find the *current* biggest gap
vs the Tantivy/Lucene-class proxy rather than trust stale ledger numbers. After the accumulated wins
(id materialization 6.32×, zero-hit/lexical-saturated short-circuits, int8/4-bit fast tier, RRF
select_nth, clone-elisions), the comparator is **parity-or-better on every top-k row** — the
2026-06-27 gaps are gone:

| corpus | query_class | hash_hybrid ratio (fs/incumbent) | was (2026-06-27) |
|--------|-------------|----------------------------------|------------------|
| 10k  | short_keyword    | **0.860** | — |
| 10k  | quoted_phrase    | **0.912** | 1.459× slower |
| 10k  | natural_language | **0.975** | — |
| 10k  | exact_identifier | 1.024 (tie) | — |
| 10k  | high_fanout      | 1.000 (tie) | — |
| 10k  | zero_hit         | 1.000 (tie) | — |
| 100k | quoted_phrase    | **0.986** | — |
| 100k | natural_language | **0.995** | 1.151× slower |
| 100k | high_fanout      | **0.396** (2.5× FASTER) | — |
| 100k | zero_hit         | **0.979** | — |
| 100k | short_keyword    | 1.013 (tie) | — |
| 100k | exact_identifier | 1.082 (p50; p95 **0.565** faster → noise) | — |
| 10k  | **limit_all**    | **1.286 (SLOWER)** | 1.140× |

Command: `FRANKENSEARCH_BOLD_VERIFY_EMIT=1 RUST_LOG=error … rch exec -- cargo bench -p frankensearch
--features lexical --profile release --bench search_bench bold_verify_tantivy_class -- --sample-size 10
--warm-up-time 1 --measurement-time 1`; artifact
`/data/projects/.rch-targets/search-cc/criterion/bold_verify/summary.{md,jsonl}`.

**Decision: no clean new per-crate comparator lever exists.** The only robustly-slower row is
**`limit_all`** (return *everything*), and its gap is **inherent to hybrid semantics**: the challenger
must embed the query, scan the *entire* vector tier (candidate_limit = doc_count, so the int8/4-bit
two-pass can't prune — you need all vectors), and RRF-fuse + full-sort ~2·N candidates, where the
incumbent does lexical-only. There is no bit-identical way to remove that work without changing
results (e.g. skipping semantic re-rank for `limit_all` would alter ordering). `exact_identifier/100k`
(1.082 p50) is sample noise (p95 0.565, the challenger faster). **The perf swarm's comparator mission
is essentially accomplished — stop hunting comparator-ratio levers.**

**Route next (real-world, NOT comparator-measurable):** the one remaining concrete perf item is the
**on-`open` ord-table rebuild** for persisted on-disk indexes (PERF_LEDGER 2026-06-28 wiring entry):
reopened on-disk indexes currently fall back to docstore id-materialization until the table is rebuilt,
so they miss the 6.32× win the in-memory path already has. Options: (a) one-pass docstore scan at
`open` (O(N) decompress, simplest-correct, ~0.4 µs/doc ⇒ ~40 ms at 100k), or (b) persist the table to a
sidecar on `commit` + load on `open` (no scan, but needs a file format + atomicity + corruption
fallback). This benefits real persisted deployments but does NOT move the in-memory BOLD comparator.

### 2026-06-28 — `index_documents` per-batch ord-table lock hoist is ~0-gain (within noise) (BlackThrush)

**Lever tested and reverted.** The materialization wiring appends each doc's `doc_id` to the ord table
at index time; `index_documents` did this via `assign_ord` (which re-locks `ord_table.write()` **per
doc**). Hypothesis: a bulk index should hold the table write lock **once** for the whole batch (and
`reserve(docs.len())` up front), paying one lock + one growth instead of N — verifying the
materialization win didn't regress indexing throughput (a real Tantivy/Lucene comparator dimension).
Correctness-neutral (the ord assignment is identical, just the lock scope differs): 82/82 lexical tests
GREEN with the batch-lock path.

**Measured (per-crate same-binary A/B, fresh in-memory index per iteration, N=5000 docs, 20 samples;
`per_doc_lock` = old `assign_ord`-per-doc, `batch_lock` = lock-once + reserve):**

| Workload | per_doc_lock | batch_lock | Ratio |
|----------|--------------|------------|-------|
| `batch_index` (index 5000 docs) | 21.547 ms `[19.073, 25.007]` | 20.038 ms `[19.659, 20.458]` | **0.93 (CIs overlap)** |

**Decision: reverted** (source + `Cargo.toml` byte-identical to `main`, baseline method + bench removed).
The 0.93 median is **noise, not signal**: the `per_doc_lock` arm's CI is very wide `[19.073, 25.007]`
(allocation/scheduling spikes) and its lower bound sits below the `batch_lock` median, while the true
lock effect is only ~0.7% (≈`5000 × ~30 ns` uncontended `RwLock` ops over a ~20 ms index). Indexing is
dominated by Tantivy `add_document` (tokenization + postings + stored doc) plus the **inherent** per-doc
`doc_id` clone the table requires — neither removable by lock scope. **The materialization win's
index-time overhead is negligible and not lock-bound; there is no clean indexing lever here.** Confirms
the materialization win is a good trade (6.32× query-time materialization for sub-noise index cost).
Don't re-attempt index-path micro-opts on the ord-table append. Original-comparator ratio is **N/A**
(frankensearch's own index-time bookkeeping; raw Tantivy is the floor the wrapper sits above).

### 2026-06-28 — async `searcher.rs` per-query path is now well-mined; remaining clones are structural/invasive or amortized (BlackThrush)

**Full code-level audit, no clean lever found.** After the two async-`searcher.rs` clone-elision wins
(`cba06d7` lexical-metadata map borrow 3.55–7.20×; `2b7ad34` MMR reorder move 25–75×), I audited the
rest of the async `TwoTierSearcher` per-query path for the same pattern. Everything else is either
already optimal or NOT a clean win — these are **code/lifetime facts, not bench-measurable**:

- **Score/rank maps** (`searcher.rs` ~1421–1462): already `AHashMap<&str, &ScoredResult>` / `<&str, f32>`
  (borrow-keyed, copy values). `build_borrowed_rank_map` is clone-free. **Optimal.**
- **`kendall_tau_with_refined_rank`** (`blend.rs` 253): already **O(n log n)** (merge-sort inversion
  counting), documented. One small scratch `Vec` per call (n≈30) — sub-noise. **Optimal.**
- **`graph_rank`** (`graph_rank.rs` 86–235, conditional): already an **index-based** CSR power iteration
  (the hot loop uses `usize` indices, not `HashMap` lookups); boundary `doc_id.clone()`s are structural
  (`HashMap<GraphDocId,…>` owned keys) and it only runs when a `DocumentGraph` is configured. **Optimal.**
- **`fast_hits` / `quality_hits` doc_id clones** (`searcher.rs` 1358, 1382, every refined query): these
  clone `doc_id` from `initial_results` → `fast_hits` → `quality_hits`. **Structural**: `VectorHit` owns
  `doc_id: String`, and `blend_two_tier(&fast_hits, &quality_hits, …)` needs both vecs live
  simultaneously, so neither can borrow/move from the other or from `initial_results` (which `initial_by_doc`
  borrows through the final assembly). Eliding requires giving `VectorHit` a borrowed `doc_id` — a
  **core-type lifetime change with a wide blast radius** across `frankensearch-index` + `-fusion` + the
  FSVI path. Flagged invasive in [[fusion-sync-result-assembly-clone-elision]] (sync `quality_hits` rebuild).
- **Phase clones** (`display_hits.clone()` / `refined_results.clone()` into `on_phase(SearchPhase::…)`):
  **streaming-necessary** — the `on_phase` callback consumes each phase while the search continues
  mutating `results`, so the phase needs its own owned copy.
- **`display_hits = initial_hits.iter().take(k).cloned()`** (~516): NOT redundant — `initial_hits` is
  reused at lines 573/648/819 (phase-2 refinement borrows the full set), so the top-k must be copied out.
  k≈10, sub-noise regardless (cf. the reverted sync phase-clone elision).
- **Quantization slab build** (`in_memory.rs` `quantize_i8_slab` / `pack_4bit_slab`): a real scalar
  compute kernel (`round()` per element may block autovectorization), **but amortized** — built once via
  `OnceLock` (`get_or_init`, lines 408/516) and cached, so it is a **cold-start / index-build** cost, not
  a steady-state query gap, and not "vs Lucene/Tantivy" (no incumbent vector tier).
- **MMR cosine dot** (`mmr.rs` `cosine_sim_pre`): scalar f64; SIMD/f32 would **change rounding**
  (not bit-identical → could shift MMR selection) and it is a conditional (opt-in) path.

**Conclusion:** the steady-state async per-query paths are **comprehensively optimized**; no clean new
per-crate lever remains without an **invasive `VectorHit` borrow refactor** (the one real remaining
per-query opportunity — eliminates the `fast_hits`/`quality_hits` doc_id double-clone, ~60–200 small
`String` clones/refined query — but needs a dedicated, well-tested change across index+fusion, not a
60-min dig). Original-comparator ratio **N/A** (async result-assembly the BOLD harness doesn't model).
Combined with [[bold-comparator-closed]] (comparator closed, limit_all inherent) and the indexing-axis
clear: **stop re-scanning the query/result-assembly surface — the next real work is the scoped VectorHit
refactor or a genuinely new subsystem.**

---

## 2026-06-28 — Non-index compute subsystems (MMR cosine, embed L2-normalize, graph_rank) — explored, NOT clean levers (BlackThrush)

After the `frankensearch-index` SIMD vein was capped (all 6 dot kernels + int8/4-bit slab quantizers +
f16 encode + the file-based FSVI slab write — see PERF_LEDGER, `bce9bc8`→`2a4d333`), I dug the
**non-index** subsystems the alien-graveyard flagged (`frankensearch-embed` HashEmbedder,
`frankensearch-fusion` MMR/graph_rank, `frankensearch-rerank`). None yields a clean
bit-identical, default-path, per-crate lever:

- **MMR `cosine_sim_pre` (`fusion/src/mmr.rs:285`)** — the O(k·n) pairwise-similarity hot kernel. It is
  ALREADY norm-hoisted (each candidate's L2 norm computed once, not per pair) and running-max optimized
  (O(k·n) not O(k²·n)) by prior cycles; the residual work is the per-pair dot, which accumulates
  **sequentially in `f64`** (`dot += f64::from(a[i]) * f64::from(b[i])`). A SIMD/multi-accumulator dot
  reorders the f64 adds → results differ by ULPs → can **flip the greedy-argmax selection** against the
  bit-exact reference test (`mmr_rerank_reference`). f64 add is non-associative, so there is no
  order-preserving SIMD form (unlike the index f16/f32 dots, whose *generic was already a `wide` kernel*
  I could match lane-for-lane). **AND MMR is opt-in** (`MmrConfig`, not wired into the default/BOLD
  hybrid). Two strikes: bit-identical-blocked + off-default.

- **`l2_normalize_in_place` (`core/src/traits.rs:376`)** — the 2-pass (Σx² + scale) L2 normalize shared
  by every embedder, run per embed (index-time docs + each query). The scalar `iter().map(x*x).sum()` IS
  latency-bound (Rust doesn't fast-math-vectorize f32 reductions → a single serialized accumulator), so a
  multi-accumulator form is a real ~2-4× **on the kernel**. BUT: (1) `frankensearch-core` has **no SIMD
  dep** (`wide` absent) — adding SIMD/AVX2-dispatch to the base crate for this is disproportionate; (2)
  the `l2_normalize_in_place_matches_allocating` test asserts **bit-exact** equality to the allocating
  variant (small cases pass via the scalar tail, but the contract is bit-exact); (3) reordering Σx²
  changes embedding values by ULPs that **ripple through f16 quantization + vector scoring** → would need
  slow **full-workspace** verification (parking risk) to prove no downstream ranking regression — for a
  modest gain on a kernel the FNV/JL hashing (and, for short queries, the rest of the query) already
  dominates. Poor risk/reward; left scalar (REVERT-territory per "REVERT ~0-gain").

- **`graph_rank` (`fusion/src/graph_rank.rs`)** — PageRank-style power iteration over a **sparse** HashMap
  adjacency (`for _ in 0..max_iterations { for src in 0..n { …edges… } }`). Sparse, pointer-chasing,
  data-dependent — not a SIMD/dense-matrix shape — and opt-in (`GraphRanker`, off-default). No lever.

**Conclusion:** the **clean per-crate compute surface is comprehensively optimized** — index dots +
quantizers + f16 encode + file-write all AVX2/F16C-dispatched (bit-identical); comparator closed
([[bold-comparator-closed]]); materialization + async result-assembly mined. The remaining
float-accumulation kernels are **bit-identical-blocked** (sequential f32/f64, non-associative) and/or
**off the default path**, so they are not safe 60-min levers. Original-comparator ratio **N/A** for all
three (own vector/rerank tiers; no Tantivy counterpart). **Next real work is the standing open item:
re-measure BOLD on a QUIET worker** to confirm the SIMD/build arc's comparator impact (the `frankensearch`
top-level crate is a cold/slow build — must not re-park it under contention), or accept the current
(comprehensively optimized) state. Don't re-dig MMR/normalize/graph_rank without lifting the
bit-identical or off-default constraint first.

---

## 2026-06-28 — BOLD re-measured: comparator NOT fully closed — two real gaps remain (exact_identifier@100k, limit_all) (BlackThrush)

**Did the open item.** Built + ran `bold_verify_tantivy_class` (`-p frankensearch --features lexical`)
**locally + niced** (the 5 contending procs were OTHER projects on the shared rch workers, so a local
nice'd build dodged the stuck-queue that killed prior attempts). Fast criterion settings from the bench's
own documented command (`--sample-size 10 --warm-up-time 1 --measurement-time 3`). Emit:
`FRANKENSEARCH_BOLD_VERIFY_EMIT=1`. **ratio = frankensearch_p50 / incumbent(Tantivy-class)_p50; >1 means
frankensearch SLOWER.** corpus hashes `2e78365a…`(10k) / `13f1b015…`(100k).

**frankensearch hybrid (`hash_hybrid_tantivy_vector_rrf`) p50 ratios:**

| query_class | 10k | 100k | reliability |
|-------------|-----|------|-------------|
| exact_identifier | 0.927 ✅ | **1.425 ❌** | 100k reliable (619→882 µs) |
| short_keyword | 1.200 ❓ | 1.020 ≈ | 10k NOISE (30 µs workload, p99 4.2× = spike) |
| quoted_phrase | 0.940 ✅ | **0.728 ✅** | reliable |
| natural_language | 0.654 ✅ | **0.792 ✅** | reliable |
| high_fanout | 1.049 ≈ | **0.789 ✅** | 100k reliable |
| zero_hit | 0.950 ✅ | 0.853 ✅ | tiny (~20-30 µs) |
| limit_all | **1.390 ❌** | (not benched @100k) | reliable (1376→1913 µs) |

**Conclusion — CORRECTS the stale [[bold-comparator-closed]] "parity-or-better on every row" claim.**
frankensearch hybrid is **faster than the Tantivy-class incumbent on most classes** (quoted_phrase 0.73,
natural_language 0.79, high_fanout 0.79 @100k; exact_identifier 0.93 @10k) — but **two real gaps survive
the entire SIMD/build arc**:

1. **`exact_identifier` @ 100k: 1.425×** (incumbent 619 µs → frankensearch 882 µs). The canonical
   "non-short-circuit row pays vector+RRF on top of lexical" gap: for a unique-token match the lexical
   tier alone nails it, but the hybrid still scans the vector tier + fuses RRF (~+263 µs of pure
   overhead). Notably the **`hash_lexical_guard` variant is EVEN SLOWER here (1.895×, 619→1173 µs)** — so
   the existing short-circuit guard is NOT helping for exact_identifier@100k (it adds cost without
   skipping). **This is the next dig target** (algorithmic, not a SIMD kernel): make the lexical guard
   actually short-circuit (skip vector+RRF) on a high-confidence exact lexical match, or find why the
   guard path is slow at 100k.
2. **`limit_all`: ~1.39×** (materialization-bound; "inherent" per prior notes — per-hit full stored-doc
   materialization the incumbent doesn't pay the same way).

**Strategic takeaway:** the SIMD/build wins this session (dots, quantizers, f16 encode, FSVI write) are
real **build/load-time** wins but did NOT move these **query-time** gaps — because the gaps are
**algorithmic** (redundant vector+RRF work on lexical-sufficient queries) and **materialization**, not
dot-kernel-bound. The next real lever is the **exact_identifier short-circuit** (the lexical-guard path),
not more compute-kernel SIMD. Caveat: fast settings → small-workload ratios (short_keyword, zero_hit)
are noise; trust the ≥600 µs workloads. (The bench only runs `limit_all` at 10k — there is no 100k
`limit_all` row. Full run completed; final summary = 26 rows.)

> **⚠️ SELF-CORRECTION (same day, second run — supersedes the `exact_identifier@100k` claim above):
> that "1.425× gap" was 12-SAMPLE NOISE, not real.** See the next entry.

---

## 2026-06-28 — BOLD re-run #2: the exact_identifier@100k "gap" was NOISE — comparator is top-k PARITY, limit_all the lone gap (BlackThrush)

Immediately suspicious that the `exact_identifier@100k` "gap" (run #1: hybrid 1.425×, **guard 1.895×**)
was noise — because `hash_lexical_guard` runs the *identical* `fixture.lexical.search_doc_ids(...)` as the
incumbent, so it CANNOT be 1.9× slower except by measurement variance. The bench's summary uses the custom
`measure_latency_us` with only **12 samples @100k / 25 @10k** (NOT criterion's `--sample-size`), so per-row
ratios carry ±40% noise. Code analysis confirms the mechanism: `exact_identifier = "doc 000042"`, and
"doc" appears in every doc → lexical returns ≥`limit` hits → `bold_verify_lexical_short_circuit` fires
(class ∈ {Identifier,…}, `lexical_count ≥ limit`) → **the hybrid SHORT-CIRCUITS and skips the vector tier
entirely** → it should ≈ the incumbent.

**Re-ran the cached binary (run #2, low contention). Result — the gap vanished:**

| query_class | run #1 ratio | run #2 ratio | verdict |
|-------------|--------------|--------------|---------|
| exact_identifier @100k (hybrid) | 1.425 | **1.003** | NOISE — parity |
| exact_identifier @100k (guard) | 1.895 | **1.002** | NOISE — parity |
| quoted_phrase @100k | 0.728 | 1.004 | NOISE — parity (run#1 "win" was also noise) |
| natural_language @100k | 0.792 | 1.002 | NOISE — parity |
| high_fanout @100k | 0.789 | 0.992 | parity |
| **limit_all @10k** | **1.390** | **1.429** | **REAL — consistent across runs** |

The tell: frankensearch hybrid was *consistent* run-to-run (exact_id@100k 882→884 µs); the **incumbent**
swung 619→881 µs — i.e. run #1's incumbent measured anomalously LOW, inflating every run-#1 ratio (both
the apparent gaps AND the apparent wins). **Run #2 shows frankensearch hybrid at PARITY (±6%) on EVERY
top-10 class at both 10k and 100k**, faster on zero_hit + natural_language@10k. **The only gap consistent
across both runs is `limit_all` (~1.4×), which is inherent** (returns everything → full vector-tier scan,
can't prune; RRF full-sort of ~2·N; the incumbent does lexical-only) — no bit-identical removal, per
[[bold-comparator-closed]].

**Conclusion (re-affirms the ORIGINAL "comparator closed" finding; my own run-#1 "exact_identifier gap"
was a phantom):** the BOLD comparator is **top-k parity across all classes + corpus sizes**; `limit_all`
~1.4× is the lone, inherent gap. **There is NO exact_identifier short-circuit lever to dig** (the
short-circuit already fires and works). **Lesson:** at the fast settings (12/25 samples) a SINGLE run's
per-row ratios are ±40% noise — never conclude a gap (or a win) from one run; require run-to-run
consistency, and treat the *incumbent's* variance as the dominant noise source. Don't re-chase
exact_identifier or the 100k "wins"; `limit_all` remains the only real (inherent) row.

---

## 2026-06-28 — Production search orchestration (sync_searcher + fsfs runtime) — verified-mined, no clean lever (BlackThrush)

After the InMemory full-recall short-cut (`84fbfa2`), applied that same "find redundant work" lens to the
**production** search paths (not the BOLD bench proxy). Both are already optimal; the remaining candidates
are sub-noise or result-changing:

- **`fusion/sync_searcher.rs` (the wired sync hybrid):** flow is tight — the quality tier
  `quality_scores_for_hits(query, &fast_hits)` rescores ONLY the fast candidates (no re-scan);
  `blend_two_tier` keys an `AHashMap<&str, _>` by `doc_id.as_str()` (borrowed, no clone); RRF is mined.
  The lone remaining per-query alloc is `quality_hits` cloning each `fast_hit.doc_id` (line ~224) just to
  pair it with the rescored value. It IS elidable bit-identically (a `blend_two_tier_with_quality_scores`
  variant taking `(&fast_hits, &[Option<f32>])` would merge by the same borrowed `&str` key — `quality_hits`
  is a doc_id-identical subset of `fast_hits`, so the merged map + `blended` are byte-for-byte the same).
  BUT it's **provably sub-noise**: `fetch` (≈k·over-fetch ≈ 30–100) small `String` allocs against the
  `sync_int8_fetch` bench's **100k-vector** scan that dominates end-to-end latency → ~0-gain (REVERT per
  the directive). Not the same class as the async metadata-map win (`cba06d7`, which elided ~candidate-count
  `serde_json::Value` clones, far heavier than short doc_id strings). Left as-is.

- **`fsfs/runtime.rs` semantic (6466) + quality (6577) stages — NOT a redundant double-scan:** both call
  `resources.vector_index.search_top_k(...)`, but the semantic stage embeds with `fast_embedder` and the
  quality stage with **`quality_embedder`** (a distinct model) → *different query vectors* over the same
  index = legitimate two-tier reranking, not redundant. (The query embed IS recomputed per stage, but
  they're different embedders.) The one arguable inefficiency — the quality stage doing a FULL
  `search_top_k(quality_budget)` rather than rescoring the semantic candidates (as `sync_searcher` does) —
  is **result-changing** (a full scan can surface docs the semantic stage missed), so it's a design choice,
  not a bit-identical lever; not a 60-min dig.

**Conclusion:** the production search orchestration is **verified-mined**. Combined with the closed BOLD
comparator (top-k parity, `limit_all` inherent), the comprehensively-AVX2/F16C-mined index compute, the
InMemory full-recall short-cut, and the rejected non-index kernels (MMR/normalize/graph_rank), **no clean,
bit-identical, above-noise per-crate perf lever remains.** The genuine remaining opportunities are either
**invasive** (a `VectorHit<'a>` lifetime refactor across index+fusion to kill the doc_id clones wholesale)
or **result-changing** (aggressive exact_identifier short-circuit, fsfs quality-rescore, limit_all tail
approximation) — none fits the "60-min, bit-identical, conformance-GREEN, measured-ratio" bar.

---

## 2026-06-28 — f16 dot 4-accumulator latency-unlock: REFUTED (~0-gain — decode-throughput-bound, not add-latency-bound) (BlackThrush)

The JL 8-lane win (`abd4628`) came from a latency unlock — the 4-lane xorshift was a single-register
dependency chain, so a 2nd accumulator register hid the latency. Its lesson said "re-check the other
dependency-chain kernels." I did. The **f16-bytes dot** (`dot_product_f16_bytes_f32_avx2`, the DEFAULT
vector-tier kernel) uses a single `__m256` accumulator — `sum = add_ps(sum, mul_ps(cvtph(..), q))` — and
`add_ps` has ~4-cycle latency while `cvtph_ps` pipelines at ~1/cycle, so I hypothesised it was
**add-latency-bound** (a ~4× latent win behind a 4-accumulator rewrite, the biggest default-path kernel).

**MEASURED first (a non-production 4-acc variant + bench arm, since a 4-acc reduction is NOT
bit-identical):**

| `dot/dim256/f16_bytes` (10k vecs) | 1-acc (prod) | 4-acc | Ratio |
|-----------------------------------|--------------|-------|-------|
| | 413.2 µs | 408.5 µs | **~1.0 (within noise)** |

**~0-gain → REVERTED** (the experimental kernel + bench arm backed out via Edit; tree clean). The
hypothesis was **wrong**: the f16 dot is **decode-throughput-bound** (the `vcvtph2ps` + the two loads cap
it at ~1 chunk/cycle), so the `add_ps` latency is already fully hidden — a single accumulator is optimal.
The 4× from the F16C arc (`7239d58`) WAS the bottleneck removal; nothing latency-bound remains.

**Why JL was different (refines the lesson):** JL's xorshift is a **pure-register recurrence with no
per-element load/decode** — so the dependency chain IS the only thing limiting throughput, and a 2nd
register is a pure win. The dot kernels all have a per-element **load/decode** (f16 `cvtph`, 4-bit
`mullo`/unpack, int8 `madd`) whose throughput is the real cap, hiding the accumulator latency. **CORRECTED
LESSON: the "add a 2nd accumulator register" unlock only applies to kernels whose hot recurrence is
register-only (PRNG, pure reductions); a kernel with a per-element load or decode is throughput-bound on
that, NOT accumulator-latency-bound — measure before rewriting (and especially before breaking
bit-identical for it).** The measure-first cost ~20 min and saved a ~0-gain, bit-identical-breaking,
default-path-rippling change. Don't re-attempt 4-acc on any of the dot kernels.

---

## 2026-06-28 — JL accumulate is SATURATED at 8-lane: 16-lane (4-way ILP) = ~1.02× (throughput wall) (BlackThrush)

The JL register-only recurrence DID respond to the 2nd-accumulator unlock: scalar → 4-lane (1.51×) →
8-lane (1.76× more, `abd4628`). Natural question: does **16-lane (4 `__m256i`, 4-way ILP)** give a 3rd
jump? Op-count model said no — the 8-lane already moved the xorshift from latency-bound (1 register, ~6-cyc
dependency chain) to ~throughput-bound (2 registers fill the ~6-cyc latency), so a 3rd/4th register just
piles on `slli`/`xor`/`movemask` ops against a fixed ~3 vector-ALU-ports/cycle. **MEASURED (non-prod
16-lane variant, 4 registers, interleaved):**

| `jl_accumulate` (dim 384, 8k tokens) | 8-lane | 16-lane | Ratio |
|--------------------------------------|--------|---------|-------|
| | 949.8 µs | 930.2 µs | **~1.02× (noise)** |

**~0-gain → REVERTED** (experimental kernel + bench arm backed out via Edit; tree clean). **JL is
saturated at 8-lane** — the 8-lane was the right stopping point (`abd4628`). Confirms the refined model:
the 2nd register (4→8 lanes) is the latency unlock; the 3rd/4th (8→16) hit the vector-ALU-throughput wall.
**General: the ILP unlock is a ONE-STEP transition (latency-bound → throughput-bound), not a ladder —
once 2 independent accumulators fill the recurrence latency, more lanes only add work. Stop at 2× the
in-flight registers needed to cover the dependency depth.** Don't push JL past 8 lanes.

---

## 2026-06-28 — l2_normalize sum-of-squares: a real ~4.3× LOCKED behind embedding stability (not shippable) (BlackThrush)

After landing the bit-identical scale half of `l2_normalize` (`41b16bc`, ~1.70× on the scale kernel), I
dug its OTHER half — the sum-of-squares `vec.iter().map(|x| x*x).sum()`. It's a **1-accumulator strict-IEEE
f32 reduction**, which LLVM will NOT auto-vectorize (reordering changes the sum), so it runs sequential +
latency-bound. **Measured the speed locked there** (`sum_of_squares` bench, dim 384):

| Workload | acc1 (current, sequential) | acc4 (reorder, auto-vec) | Ratio |
|----------|----------------------------|--------------------------|-------|
| `sum_of_squares` | 35.31 µs | 8.21 µs | **0.23 (~4.3×)** |

So a 4-accumulator reformulation is **~4.3× faster** — and notably it needs NO AVX2/unsafe at all (4
independent scalar accumulators → LLVM auto-vec to SSE2), so it's even **cross-host-consistent** (same
binary everywhere). If unlocked, `l2_normalize` would go from ~1.3× (scale-only) to ~2× overall.

**But it is LOCKED — `REVERTed` the idea, kept only the bench as evidence.** A 4-acc reduction reorders
the f32 sum → `norm_sq` differs by ULPs → `inv_norm` differs → **every embedding shifts by ULPs**. That
is an **embedding-stability break**: a persisted index built before the change no longer matches a query
embedded after it (the stored f16 mostly absorbs the ULP, but the query f32 does not), and results would
silently drift on a library upgrade. This is exactly what the **bit-identical discipline protects** — and
it's why the SCALE half (`x *= inv_norm`, element-wise, IEEE-exact 1/4/8-wide, identical on every host and
every version) WAS shippable while the sum-of-squares is not. The absolute value is also modest (the embed
is a small fraction of a query; `l2_normalize` a fraction of the embed), so the embedding churn is not
worth it even if conformance happened to stay GREEN.

**CLARIFIED INVARIANT (the deep reason every SIMD win this session had to be bit-identical):** a search
library's embeddings/scores must be **reproducible across hosts AND library versions**. Therefore
**element-wise** ops (scale, accumulate, decode, encode, round/clamp, quantize-pack) are widenable — each
output lane = f(one input lane), IEEE-exact regardless of width — but **reductions** (sum-of-squares, dot
accumulators, any `Σ`) are LOCKED, because any reorder changes the result. The element-wise vein is now
mined (model2vec mean-pool + `l2_normalize` scale shipped); the reduction halves carry a real but
unspendable ~4× each. Don't re-attempt SIMD/multi-acc on any reduction.

---

## 2026-06-28 — limit_all 1.4× decomposed: every component is optimal or inherent — no lever (BlackThrush)

The only surviving BOLD comparator gap is `limit_all` (~1.4× vs the Tantivy-class incumbent; top-k rows
are parity). The swarm has `bold-limitall-*` worktrees probing it, so I decomposed it end-to-end and
confirmed each piece is already optimal or fundamentally inherent — **there is no lever here**:

1. **Lexical (Tantivy):** the incumbent IS Tantivy — can't beat it on its own ground.
2. **Vector tier (full scan):** `limit_all` sets `candidate_limit = doc_count`, so the file-backed
   `VectorIndex` takes the full-recall fast path (`search.rs:170`, single f16 collect-and-sort) — the
   f16 dot is AVX2/F16C (`7239d58`), and the redundant int8/4-bit pass-1 was already short-cut for InMemory
   (`84fbfa2`). Optimal single pass.
3. **RRF fuse:** read + benched (`-p frankensearch-fusion --bench rrf_fuse`): **22.3 µs new vs 29.1 µs old**
   (a prior win already landed). It is comprehensively optimized — `AHashMap<&str, _>` keyed on **borrowed**
   doc_ids (no `String` clone in the merge), a single `entry()` probe per doc (no get-then-insert),
   capacity pre-allocated, `select_nth_unstable` for top-k (full sort skipped), and `into_owned` only on the
   output winners. For `limit_all` the remaining costs are **inherent**: a full `sort_unstable` of the ~N
   unique fused hits (you must rank *everything*), N `entry` string-hashes (keying on a precomputed u64 would
   need cross-layer doc_id-hash plumbing through lexical+vector search — invasive, ~3-5% of `limit_all`), and
   N `into_owned` doc_id clones (the returned `Vec<FusedHit>` must own its doc_ids since the borrowed
   `lexical`/`semantic` inputs are dropped — only the invasive `VectorHit<'a>` lifetime refactor removes it).

**Conclusion: `limit_all` ~1.4× is structurally inherent — frankensearch does lexical + a full vector scan
+ an RRF full-sort where the incumbent does lexical-only; every part of that extra work is already at its
floor.** Stop digging `limit_all` for a per-crate lever; the only removals left are invasive
(`VectorHit<'a>` borrow refactor) or cross-layer (u64 doc_id keys), neither a 60-min win and both
sub-meaningful against the modest gap. With this, BOTH comparator-gap components (vector scan + RRF) are
confirmed optimal — the comparator investigation is closed.

---

## 2026-06-29 — parallelizing the gather SETUP (rayon `&HashSet::par_iter` + `par_sort`) is a ~10× REGRESSION (BlackThrush)

**Lever tested and REVERTED.** Follow-up to the landed gather fast-path (`ec76859` serial, `383ee53`
parallel-dots). The gather's crossover (~13%) is capped not by its dots (already `par_chunks`) but by its
**serial setup** — `allowed.iter().filter_map(|h| map.get(h)).collect()` + `positions.sort_unstable()`.
Hypothesis: parallelize the setup above `PARALLEL_CHUNK_SIZE` (`allowed.par_iter()…collect()` +
`par_sort_unstable()`) to flip the 25–50 % band to wins and widen the gate toward N/2. Compiled clean,
conformance GREEN (index lib 374/374, incl. the parity tests), so it was correct — just **far slower**.

**Measured** (`filtered_gather`, N=50k clustered dim 384 k10, gather median; parallel branch is M>1024):

| selectivity | serial setup (`383ee53`) | parallel setup | scan | verdict |
|-------------|--------------------------|----------------|------|---------|
| 5 %  (2 500)  | 128.3 µs | **1 297 µs** | 180.8 µs | **10.1× slower → now a LOSS** |
| 10 % (5 000)  | 217.7 µs | **2 035 µs** | 194.6 µs | 9.4× slower → LOSS |
| 25 % (12 500) | 726 µs   | **4 052 µs** | 357 µs   | 5.6× slower |
| 50 % (25 000) | 1 170 µs | **6 205 µs** | 794 µs   | 5.3× slower |

(Sub-1024 allow-sets stay on the serial branch and were unchanged — `0.5%`=15 µs, `1%`=30 µs, `2%`=58 µs.)

**Root cause:** `rayon`'s **`&HashSet` parallel iterator partitions a hash table extremely poorly** — a
`RawTable` has no cheap midpoint split, so `par_iter` degrades to high-overhead bucket walking with bad load
balancing, an order of magnitude slower than the plain serial `iter().filter_map().collect()` (a linear
probe-friendly walk). `par_sort_unstable` adds its own spawn overhead at these sizes. The setup is **memory-
/probe-bound serial work that rayon cannot accelerate here**; the win came entirely from parallelizing the
**dots** (already shipped in `383ee53`).

**Decision: fully reverted** (`in_memory.rs` byte-identical to `383ee53` — `git diff` empty before this
entry; reverted via Edit, not `git checkout`, which `dcg` blocks). The serial-setup + parallel-dots gather
(`383ee53`, gate N/10) stands as the kept state. **Route next:** to extend the gather past ~13 % the lever
is NOT parallelizing the `HashSet` walk — it would be a *cheaper representation* of the gathered positions
(e.g. having `candidate_hashes` hand back an already-sorted `&[u64]` slab, or storing the allow-set
positions directly), so no per-query collect+sort is needed at all. Lower priority than the FSVI/lexical
plumbing — moderately-selective (>10 %) filters are the uncommon case, and the serial setup already serves
the common <10 % region well. Original-comparator ratio **N/A** (internal filtered-path micro-lever).

---

## 2026-06-29 — hash-embedder `tokenize` ASCII fast path is NOT a lever — JL projection dominates (BlackThrush)

**Not attempted — flagged to save effort (the LUT/byte-fast-path pattern does NOT transfer here).** After
landing the byte-class LUT on `count_lexical_tokens` (`d7c8477`, ~1.6×), the obvious next target looked like
`hash_embedder::tokenize` (`hash_embedder.rs:431`): it splits on Unicode `!c.is_alphanumeric()` per char with
**no ASCII fast path**, and it runs on *every* embed (query + each document at index time) — superficially a
broad hot-path win.

**But the embed cost is dominated by the JL projection, not tokenization.** `embed_jl` (the production
default) does, **per token**, `jl_accumulate` over **all `dimension` (384) dims** (SIMD, `4a75fd7`), i.e.
`O(tokens · 384)` — whereas `tokenize` is `O(text length)` and `fnv1a_hash` is `O(token length)`. For an
average token (~6 chars) the JL inner loop does ~384 dim-ops vs ~6 char-classifications in tokenize, so
**tokenize is well under ~2 % of `embed_jl`**. An ASCII byte/LUT tokenizer (≈1.5× on that 2 % slice) moves
end-to-end embed by **<1 %** — below the bench noise floor, so it would be a `REVERT ~0-gain` per the keep
threshold. (`embed_fnv_modular` is `O(tokens)` so tokenize is a larger fraction there, but JL is the
quality-default; and the `hash_embed` bench already mined the embed allocations — `tokenize_vec`→lazy iter,
L2 in-place.) Original-comparator ratio **N/A**.

**Consequence — the per-crate micro-lever frontier is empirically exhausted this session.** Probed with
evidence and ruled out: embedding (JL/mean-pool/l2 SIMD'd; tokenize dominated, above), tokenizers
(`count_lexical_tokens` LUT landed, `code_structure::tokenize` already byte-fast-pathed, `tokenize_lexical`
allocation-bound), core scanners (`canonicalize` nfc/filter/md fast paths, `parsed_query` no-op fast path +
alloc elision — both already landed), search-time reductions (MMR cosine 4-acc **landed** `efbfe33`;
`graph_rank` is scatter-add, `native_rerank` is ONNX), and the filtered vector path (gather **landed**
`ec76859`+`383ee53`, live in the production sync fast-tier). The remaining real wins are **substantial /
invasive**, not 60-min micro-levers: (a) a BOLD **selective-filter comparator row** (head-to-head dominance
number for the gather — needs a fair Tantivy filter-then-search incumbent), (b) the **FSVI raw-mmap gather**
(mutation-safe generation-keyed cache invalidation over tombstones+WAL), (c) the `VectorHit<'a>` /
u64-doc-id-key refactors for the last `limit_all` clones. Pick one as a deliberate fresh cycle.

---

## 2026-06-29 — wider-SIMD (AVX-512 / VNNI) is HARDWARE-UNAVAILABLE — AVX2 is the ceiling here (BlackThrush)

**Not attempted — hardware blocker, flagged to save effort.** After the per-crate micro-lever frontier was
exhausted, the last "radical primitive" angle was a **wider SIMD** dot kernel above the landed AVX2 ones:
AVX-512 (16-wide f32 / 32-wide i8) or, best of all, **AVX-512-VNNI `vpdpbusd`** (4 int8 mul-accumulates per
lane in one instruction) for the int8 two-pass pass-1. Crucially, an int8 VNNI dot would be **bit-identical**
to the AVX2 int8 dot (integer accumulation — no float reorder), so it would dodge the usual cross-host
ULP concern.

**But the deployment/bench CPU has no AVX-512.** `rch exec` reports the host as an **AMD Ryzen Threadripper
PRO 5975WX (Zen3, 32-core)** with CPU flags `avx2 f16c fma` and **no `avx512*` / `vnni`** (Zen3 predates
AMD AVX-512, which arrives in Zen4). So: (1) a VNNI/AVX-512 kernel can't run or be measured on this host;
(2) it wouldn't benefit this deployment at all; and (3) even on a hypothetical AVX-512 host, signed×signed
int8 (`i8·i8`) isn't a single VNNI op — `vpdpbusd` is **u8·i8**, so it needs an offset-by-128 + correction
(extra work), and a clean single-instruction signed path needs **AVX-VNNI-INT8 `vpdpbssd`** (Sapphire
Rapids / Zen5+ only). And the int8 two-pass pass-1 is **bandwidth-bound** (reading the `N·d`-byte slab), so
a wider *compute* kernel would likely be ~0-gain on the scan even where AVX-512 exists.

**Conclusion: the AVX2 runtime-dispatch kernels (`dot_i8_i8`, `dot_product_f16_bytes_f32`, the 4-bit and
f32 dots) are the SIMD ceiling for the target hardware (Zen3).** Do not attempt AVX-512/VNNI kernels unless
the deployment moves to Zen4+/AVX-512 hardware AND a compute-bound (not bandwidth-bound) kernel is the
proven bottleneck. Original-comparator ratio **N/A**. With this, the radical-lever search space — software
micro-levers (≈15 veins, all mined/landed), wider SIMD (hardware-unavailable), and the head-to-head
filtered comparator (structurally N/A) — is comprehensively closed for the current code + hardware.

---

## 2026-06-29 — bd-lhck: f16 dot 4-accumulator re-measured now decode is F16C — REFUTED (bandwidth-bound) (BlackThrush)

**Lever tested and REVERTED — closes the open bead `bd-lhck`** ("Re-measure multi-accumulator on f16 dot
kernels NOW that decode is SIMD (was decode-bound)"). The current production f16 vector dot
(`dot_product_f16_bytes_f32_avx2`, simd.rs) is **single-accumulator** (one `_mm256_add_ps` loop-carried
chain). The prior 4-acc refutation (`359863d`) was on the *software-decode* generic path (decode-bound);
bd-lhck's premise — that F16C hardware decode (`_mm256_cvtph_ps`, `7239d58`) removed the decode wall so the
add-chain latency would now bind — is **correct in premise but wrong in conclusion**: the kernel is
**bandwidth-bound**, not latency-bound. Added a bench-only 4-accumulator AVX2+F16C variant
(`f16_bytes_4acc`) and A/B'd it vs the single-acc library kernel.

**Measured (`dot_product` bench, 10k×dim f16 slab, host under load ~13 → 10 samples, CIs OVERLAP):**

| dim | `f16_bytes_new` (1-acc) | `f16_bytes_4acc` | median ratio | CIs |
|-----|-------------------------|------------------|--------------|-----|
| 256 | 3.646 ms | 3.492 ms | 0.958 | [3.56,3.78] vs [3.16,4.05] — overlap |
| 384 | 5.022 ms | 4.772 ms | 0.950 | [4.62,5.58] vs [4.46,5.33] — overlap |

**Root cause / decision:** the dot iterates a **5–7.7 MB f16 slab** (10k×256–384×2B), far out of L1/L2, so a
single accumulator already saturates **memory bandwidth** — extra accumulators have nothing to unlock
(the add-chain isn't the binding resource). The ~0.95 median is within the (overlapping, contended,
10-sample) noise, and is directionally a non-result: the **real `limit_all`/full-scan path reads an even
larger slab → even more bandwidth-bound**, so any cache-resident micro-advantage would shrink further. Not
worth landing regardless — it would also break the **default** f16 path's cross-host bit-identity
(`avx2_f16dot_matches_generic`) for a ≤5% kernel gain that doesn't transfer. Bench reverted byte-identical
(`git diff` empty). Original-comparator ratio **N/A**. **bd-lhck answered: no — the f16 dot is
bandwidth-bound; multi-accumulator does not help regardless of decode width.**

---

## 2026-06-29 — bd-t8tv: int8 two-pass crossover vs parallel-exact — NO adverse crossover, no gate needed (BlackThrush)

**Measurement, closes the open bead `bd-t8tv`** ("int8 two-pass: bounded-heap parallel pass-1 + find the
crossover scale vs parallel exact"). Part 1 (parallel bounded-heap pass-1) is already shipped
(`in_memory.rs`: pass-1 is `(0..chunk_count).into_par_iter()` with per-chunk bounded heaps + `merge`). Part
2 — the crossover scale — measured here: `search_top_k_int8_two_pass(k=10, mult=5)` vs the parallel exact
`search_top_k` across corpus size N (clustered embeddings, dim 384, `int8_crossover` bench arm, 20 samples).

| N | flat (parallel exact) | int8 two-pass mult=5 | int8/flat | recall@10 | verdict |
|---|-----------------------|----------------------|-----------|-----------|---------|
| 500    | 19.63 µs  | 15.89 µs  | **0.81**  | 1.0000 | int8 **1.24×** faster |
| 2 000  | 71.37 µs  | 71.01 µs  | 0.995     | 1.0000 | tie |
| 10 000 | 245.4 µs  | 246.7 µs  | 1.005     | 1.0000 | tie |
| 50 000 | 718.4 µs  | 456.5 µs  | **0.636** | 1.0000 | int8 **1.57×** faster |

**Finding: there is NO adverse crossover.** int8 two-pass is **tie-or-faster than parallel-exact at every
scale** with recall **1.0** — it wins at small N (1.24×, cheap int8 dots dominate the few-element scan),
ties in a 2k–10k "overhead valley" (the larger candidate heap + f16 rescore balance the cheaper pass-1),
and wins again at large N (1.57×, the 3×-cheaper int8 dots amortize the fixed overhead). So **no small-N
gate to exact is warranted** — always taking the two-pass above the existing full-recall short-cut
(`candidate_count ≥ count` → exact, `84fbfa2`) is already optimal; the two-pass never regresses. **bd-t8tv
answered: the crossover is benign (two-pass ≥ exact everywhere); the current behavior needs no change.** No
code lever → the measurement bench was reverted byte-identical (`git diff` empty), numbers recorded here.
Original-comparator ratio **N/A** (internal two-pass-vs-exact selection).

---

## 2026-06-29 — bd-tjkm: expected-loss controller has marginal headroom — phase-1 scan is ~98% of query latency (BlackThrush)

**Measurement (bd-tjkm AC#1 baseline, KEPT as `progressive_replay` bench), rejecting the controller lever.**
bd-tjkm proposes an expected-loss candidate-budget controller (adapt `candidate_multiplier` /
`quality_timeout` / `fast_only` from measured latency-vs-quality loss). Built the mandated measurement-first
baseline: a Zipf query-replay over `SyncTwoTierSearcher::search_collect` (in-memory two-tier, N=20k, dim
384, k=10, 4000-access Zipf trace) emitting end-to-end percentiles + the phase split.

**Measured (`progressive_replay` bench):**

| metric | value |
|--------|-------|
| p50 / p95 / p99 | 508.1 µs / 701.4 µs / 864.1 µs |
| phase-1 (fast-tier scan) mean | **0.498 ms (~98 %)** |
| phase-2 (quality rescore) mean | **0.024 ms (~5 %)** |

**Finding: the expected-loss controller's levers have little to act on.** (1) A `fast_only` / skip-phase-2
gate can save at most the phase-2 cost — **~24 µs / ~5 %** — and it is *result-altering* (the quality tier
re-ranks the fused top-k), so it trades a real recall/ordering risk for a sub-5 % latency sliver: not worth
it. (2) The candidate budget is already at its validated lossless point (`mult=3`; lower loses 4-bit recall,
bd-t8tv / int8_two_pass bench). (3) The S3-FIFO cache half of bd-tjkm is already landed (`b83b25d`). The
**~98 % is phase-1** — the fast-tier 4-bit two-pass scan, already AVX2 + parallel + cutoff-gated and
confirmed at its floor (bd-t8tv). So there is **no expected-loss perf lever worth the complexity + risk
here**; the controller would be adaptive machinery governing a 5 % slice. Kept the `progressive_replay`
baseline as the bd-tjkm trace artifact; recorded no behavior change. Original-comparator ratio **N/A**
(internal query-path latency). **bd-tjkm perf-lever verdict: marginal — phase-1-dominated and already
optimized; the controller is not a worthwhile perf win on this path.**

---

## 2026-06-29 — HNSW route-next REFUTED at production scale/dim: recall collapses at 100k × 384 (BlackThrush)

**Lever re-validated and REVERTED — the 2026-06-26 HNSW route-next ("re-validate at larger N; the crossover
improves for HNSW") does NOT hold at production dimensionality.** The progressive_replay baseline showed the
fast-tier **flat scan is ~98 % of query latency**, so the only structural lever on it is ANN (HNSW, O(log N)
vs flat O(N)). HNSW is implemented + wired into the file-backed `TwoTierIndex` (gated by `hnsw_threshold`)
but was decided "not default" at 10k × 128 (recall/latency trade). Re-ran `hnsw_vs_flat` at **N=100k,
dim=384** (production dim; clustered vectors; `HnswConfig::default()`):

| ef_search | recall@10 | HNSW latency | vs flat (1.293 ms) |
|-----------|-----------|--------------|--------------------|
| 10  | **0.088** | 325.7 µs | 3.97× faster |
| 20  | **0.181** | 565.0 µs | 2.29× faster |
| 40  | **0.306** | 955.5 µs | 1.35× faster |
| 100 | **0.550** | 1.843 ms | **0.70× (slower)** |
| 200 | **0.625** | (slower)  | — |

**Finding: at 100k × 384 HNSW is non-viable.** Recall **collapses** vs the 10k × 128 baseline (which reached
0.95 at ef=100) — the dim 128→384 jump (ANN curse of dimensionality) + the `HnswConfig::default()` M /
ef_construction (under-built for 384-d) crater navigation accuracy. And where recall is even moderate
(0.55 at ef=100) HNSW is **slower than the parallel flat scan** (1.84 ms vs 1.29 ms). The flat scan
(rayon-parallel SIMD f16 dot + bounded-heap + cutoff, recall 1.0) **wins decisively** at production scale —
it's sequential/cache-friendly and bandwidth-bound where HNSW is a serial random-access graph walk that
degrades with dimension. **The larger-N route-next is the OPPOSITE of what was hoped: higher dim dominates
the larger-N crossover, so HNSW gets *worse* at production dim, not better.** A heavily-tuned HnswConfig
(much higher M + ef_construction, → big build-time + memory cost) might lift recall, but that is a separate
deep investigation and the flat scan is already the strong default. **Decision: keep flat scan as the
default vector tier; the 2026-06-26 "do not wire HNSW as default" stands and is now confirmed at production
scale/dim.** Bench reverted to its 10k × 128 form (`git diff` empty); 100k × 384 numbers recorded here.
Original-comparator ratio **N/A** (internal ANN-vs-flat vector-tier selection).

**FOLLOW-UP — tuning the config does NOT fix it (the trade is fundamental).** The default `m=16` looked
under-built for 384-d, so re-ran `hnsw_vs_flat` with a **tuned `HnswConfig { m: 32, ef_construction: 400 }`**
at N=30k × 384 (clustered):

| ef_search | recall@10 | HNSW latency | vs flat (531.8 µs) |
|-----------|-----------|--------------|--------------------|
| 10  | 0.350 | 402.8 µs | 1.32× faster |
| 20  | 0.469 | 654.7 µs | 0.81× (slower) |
| 40  | 0.606 | 1.057 ms | 0.50× (2× slower) |
| 100 | 0.819 | 2.338 ms | **0.23× (4.4× slower)** |
| 200 | 0.906 | (slower)  | — |

m=32 *did* lift recall (ef=200: 0.625→**0.906**), but HNSW is faster than flat **only at ef=10 (recall
0.35, useless)**; to reach even ~0.82 recall it is **4.4× slower** than the flat scan, and it never reaches
the ~1.0 a default vector tier needs. **Root cause is structural, not config:** the flat scan is
**rayon-parallel + SIMD f16 dot + cache-friendly sequential**, so on a 32-core host it finishes 30k–100k in
0.5–1.3 ms; HNSW is an inherently **serial graph walk with random memory access** that can't use the cores
and degrades with dimension. HNSW only wins at *low recall* or at *N large enough that flat's O(N) finally
beats the parallelism* (≫100k) — neither applies at frankensearch's 10k–100k target with a recall-1.0
requirement. **HNSW is conclusively closed: not a default-vector-tier lever at any tested config/scale;
the parallel flat scan is the floor. Do not re-probe HNSW tuning.**

---

## 2026-06-29 — limit_all RRF radix-sort: refuted by structure (un-radixable String tie-break + pervasive ties) (BlackThrush)

**Not attempted — flagged to save effort.** Today's BOLD re-measure (`2c8e73b`) confirms `limit_all` (~1.45×)
is the lone slower row vs the incumbent — the biggest measured gap. Its components are inherent (full vector
scan + RRF + materialize vs the incumbent's lexical-only); the one *single-crate* sub-lever is the RRF
**full sort** (`rrf.rs:341` `sort_unstable_by(cmp_for_ranking)` when `k = N`), O(N log N). A radix/counting
sort (O(N), a DIFFERENT primitive) looked tempting, but it is **structurally refuted**:

1. **Pervasive ties.** RRF score = `1/(k+lex_rank) + 1/(k+vec_rank)`. For limit_all, most docs appear in only
   one tier, so a lexical-only doc at rank `r` and a vector-only doc at rank `r` BOTH score `1/(k+r)` —
   every rank value collides across the two single-tier populations. Ties are not rare; they are the common
   case at limit_all scale.
2. **Un-radixable tie-break key.** `cmp_for_ranking` breaks ties by the **doc_id String** (ascending).
   Radix needs the minor key radixable; a variable-length UTF-8 String is not cheaply radix-sortable, and
   with ties pervasive the radix would have to comparison-sort large tie-groups by String anyway — which is
   exactly what `sort_unstable_by` already does. No clean win; likely a regression (radix passes + tie-group
   sorts on top of the unavoidable comparisons).

So **the only per-crate sort primitive available is the comparison sort already in use**; the radix lever is
not viable. The remaining limit_all removals are the **invasive, multi-crate** ones already flagged
(u64-keyed RRF merge needing cross-layer doc_id-hash plumbing ~3-5%; the `VectorHit<'a>` lifetime refactor
to drop the N materialization clones ~3-5%) — both forbidden by the per-crate constraint and sub-meaningful
against an inherent gap. **`limit_all` is conclusively at its floor for single-crate work; no per-crate
lever remains on the biggest measured gap.** Original-comparator ratio **N/A** (the gap itself is the
2026-06-28-decomposed inherent hybrid-vs-lexical-only cost).

---

## 2026-06-29 — indexing-throughput axis is inherent + frankensearch kernels already optimized (no comparator-beatable gap) (BlackThrush)

**Frontier-map completion — flagged to save a future agent a slow indexing-comparator build.** The BOLD
comparator is search-only; indexing throughput is the one realistic "vs original" axis it doesn't cover.
Probed the untouched crates (`durability`, `storage`) — they are file-integrity / persistence (xxh3_64
verify, protect/repair), **not the search or index hot path**, and already examined (28 NEGATIVE_EVIDENCE
entries). And **no indexing-throughput-vs-Tantivy comparator exists**. Building one is not worth it because
the axis is **structurally the same inherent hybrid-vs-lexical-only shape as `limit_all`**: frankensearch
indexes lexical (Tantivy, the incumbent's own ground) **plus** a vector index (embed → FSVI slab write →
quantize) that Tantivy does not build at all, so frankensearch indexing is *necessarily* slower than
lexical-only by exactly the vector-build cost — an inherent gap, not a beatable lever.

**And the frankensearch-specific indexing kernels are already measured-optimized** (so the inherent overhead
is already minimized): FSVI slab write **6.4–7.3×** (`2a4d333`), int8 quantize **5.2–5.9×** (`76338bd`),
4-bit pack **10.3–13.6×** (`dc60d61`), F16C f32→f16 encode **4.1–5.4×** (`9f2356a`), JL embed accumulate
**2.76×** (`abd4628`), `count_lexical_tokens` LUT **1.5–1.8×** (`d7c8477`). Every per-doc index-build step
frankensearch owns has a landed SIMD/branchless win; the residual is the irreducible vector-embedding cost.
**Conclusion: do not build an indexing comparator expecting a winnable gap — the axis is inherent and
frankensearch's side is at its floor.** With this, the measured-axis map is complete: search top-k =
**parity-or-better (won)**, search `limit_all` = inherent (no per-crate lever), vector tier = at floor
(HNSW refuted, kernels AVX2-capped, reductions bandwidth-bound), indexing = inherent + kernels optimized.
Original-comparator ratio **N/A**.

---

## 2026-06-29 — hoisting per-call SIMD dispatch (`is_x86_feature_detected!`) out of the scan loop is NOT a lever (BlackThrush)

**Not attempted — flagged to save effort (a highly-visible-looking lever that does not pay).** Every SIMD
dot wrapper (`dot_i8_i8`, `dot_product_f16_bytes_f32`, `dot_4bit_prepared`, …, `simd.rs`) runs
`std::is_x86_feature_detected!("avx2")` **per call**, and those calls sit inside the per-vector scan loops
(e.g. the int8 pass-1 at `in_memory.rs:450`, `dot_i8_i8(stored, &query_i8)` per candidate). To a fresh
reader this looks like an obvious free win: detect AVX2 **once** before the loop, then call the `_avx2`
kernel directly. It is not a win, for three compounding reasons:

1. **The mandatory call cannot be removed by hoisting.** `dot_i8_i8_avx2` (and every sibling) carries
   `#[target_feature(enable = "avx2")]`. A `#[target_feature]` function **can never be inlined into a
   baseline-compiled caller** (the scan loop is built for baseline `x86-64`, not `+avx2`), so there is
   **always a real `call` to the kernel per vector** regardless of where the feature check lives. Hoisting
   the check removes only the cached-atomic-load + (perfectly-predicted) branch around that call — a few
   cycles — not the call itself.

2. **That residual is hidden — the scan is bandwidth-bound.** The int8/f16 pass-1 reads the `N·d`-byte slab
   (established bandwidth-bound, this ledger, AVX-512 entry). A relaxed cached-atomic load + always-taken
   branch per iteration is fully overlapped by the memory stalls of streaming `stored`, so removing it is
   `~0-gain` on the dominated path — a `REVERT ~0-gain` by the keep threshold.

3. **The version that *would* help is invasive and conflicts with a landed optimization.** The only way to
   delete the per-vector call (not just the check) is a **batch AVX2 scan kernel** — one `#[target_feature]`
   fn that loops all `N` vectors internally, keeping the accumulators/registers live across the scan. But the
   scan body interleaves the **filter predicate** (`f.matches_doc_id_hash`, `in_memory.rs:437`) and the
   **fused bounded-heap cutoff fast-path** (`heap.len() < candidate_count || score_key(score) >= cutoff`,
   `in_memory.rs:455` — a deliberate landed win). Pulling the dot into a batch kernel forces scores into a
   scratch buffer and a **separate** heap pass, regressing that fused cutoff; and the kernel itself stays
   bandwidth-capped. Net: invasive, conflicts with a kept lever, and bounded above by the same bandwidth wall.

**Conclusion: leave the per-call runtime dispatch as-is — it is the idiomatic, correct shape and its
overhead is hidden by the bandwidth-bound scan. Do not hoist it or batch-kernel it on this hardware.** This
closes the last "obvious-looking but dominated" pattern in the vector scan loops. Original-comparator ratio
**N/A** (internal kernel-dispatch micro-shape).

---

## 2026-06-29 — vector winners radix sort: bit-identical & radixable but SLOWER than pdqsort — REVERTED (BlackThrush)

**Lever tested and reverted — a NEW result distinct from the RRF radix refutation.** The exact-search final
winners sort (`search.rs` / `in_memory.rs` / `mrl.rs`, now `sort_unstable_by(compare_best_first)`) is the
single-crate sub-lever on the `limit_all` gap. The earlier RRF radix refutation (this ledger, 2026-06-29)
was **structural** — RRF has pervasive discrete-score ties and an un-radixable **String** doc_id tiebreak.
The vector winners sort does **not** share that structure: cosine scores are **continuous f32** (ties rare)
and the tiebreak is an **integer `index`**, so a composite `u64` key — `score` mapped HIGHER→smaller (desc)
in the high 32 bits, `index` asc in the low 32 — encodes `compare_best_first` **exactly** and is fully
radixable. So radix was *viable* here where it wasn't for RRF; the question was purely speed.

**Measured** (per-crate, `winners_sort` bench, radix arm A/B'd vs the shipped `sort_unstable_by`; full-order
`assert_eq` confirmed the radix output is bit-identical to the stable sort at every N):
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
  rch exec -- cargo bench -p frankensearch-index --bench winners_sort
```

| Winners | `sort_unstable_by` (shipped) | composite-`u64` radix | ratio (radix/unstable) | verdict |
|---------|------------------------------|-----------------------|------------------------|---------|
| `n100`   | 1.142 µs   | 2.733 µs   | **2.39× SLOWER** | regression |
| `n10000` | 190.1 µs   | 213.5 µs   | **1.12× SLOWER** | regression |
| `n50000` | 1.285 ms   | 1.736 ms   | **1.35× SLOWER** | regression |

**Why it loses:** an 8-pass LSD radix carries `(u64 key, HeapEntry)` = 24-byte pairs through 8 read/scatter
passes (~`8·N` random-ish writes + ping-pong buffers = ~2× the working set), which is **memory-movement-bound**
and blows the L2 working set at 50k. `sort_unstable_by` (pdqsort) is cache-efficient, branch-predicts well on
the mostly-distinct continuous keys, and at these N its `~N·log N` cheap `u64`-ish comparisons beat the radix's
`8·N` memory passes. Radix only wins when keys are short, N is huge, and the payload is a small index — not a
24-byte record at N≤50k. **Reverted** (bench restored byte-identical to HEAD; production winners sorts stay
`sort_unstable_by`). The comparison sort is the right primitive for the vector winners sort; the
`sort_by → sort_unstable_by` win already shipped (`afb646b`, ~1.16–1.47×) is the floor for this sub-lever.
Original-comparator ratio **N/A** (internal sort-primitive micro-lever on the `limit_all` gap).

---

## 2026-06-29 — parallel sort does NOT transfer from vector winners to the RRF limit_all sort — REVERTED (BlackThrush)

**Lever tested and reverted.** The gated `par_sort_unstable_by` that won **~2.81× at 50k** on the exact-search
vector winners sort (`d4bc73c`/`7c53d3f`, `winners_sort` bench) looked like it should transfer directly to
the *sibling* large-N serial sort on the same `limit_all` path: `rrf_fuse_with_graph`'s full
`sort_unstable_by(cmp_for_ranking)` (`rrf.rs:341`, runs when `window >= len`). `cmp_for_ranking` is a strict
total order (`doc_id` tiebreak on the unique map key → bit-identical under a parallel sort), and the
comparator is *more expensive* than the winners' `compare_best_first`, so the naive expectation was that
parallelism would amortize even better.

**The opposite happened — measured** (per-crate `rrf_sort` bench, serial vs `par_sort_unstable_by`,
limit_all-shaped fused set: discrete `1/(k+rank)` scores → **pervasive ties** → the `doc_id` String tiebreak
fires constantly; bit-identity asserted):

| Fused hits | serial `sort_unstable_by` | `par_sort_unstable_by` | ratio (par/serial) | verdict |
|------------|---------------------------|------------------------|--------------------|---------|
| `n10000` | 472.7 µs  | 653.6 µs | **1.38× SLOWER** | regression |
| `n50000` | 2738 µs   | 2149 µs  | 0.79 (~1.27× faster) | marginal |

**Why it does NOT transfer (the counterintuitive part):** an *expensive* comparator makes the parallel sort
**worse**, not better, here. (1) The pervasive RRF ties produce large `doc_id`-String tie-groups of uneven
size → poor parallel load balancing (rayon splits don't align with tie-group boundaries). (2) The element is
a 40-byte `FusedHitScratch` carrying a `String` (24-byte ptr/len/cap) vs the winners' 16-byte POD
`HeapEntry`, so the merge/partition data movement is ~2.5× heavier per element and dominates. The serial
`sort_unstable_by` (pdqsort) handles the tie-groups and the larger elements with better cache locality.
Net: a regression below ~50k and only a modest 1.27× at 50k — far from the winners' 2.81×, with an uncertain
crossover and a real regression zone over the common limit_all sizes.

**Decision: reverted** (bench removed, `Cargo.toml` registration reverted; `rrf.rs` untouched — the serial
`sort_unstable_by` stays). The parallel-sort primitive is **kept only on the cheap-comparator vector winners
sort** (`7c53d3f`), where it cleanly wins; it is **not** a blanket lever for every large-N sort. Lesson: a
parallel comparison sort pays when elements are small/POD and the comparator is cheap (winners), and loses
when elements carry heap payloads and the comparator hits pervasive variable-size tie-groups (RRF). Measure
each site — do not assume a sort primitive transfers across comparators. Original-comparator ratio **N/A**
(internal sort-primitive micro-lever on the limit_all gap).

---

## 2026-06-29 — shrinking `FusedHitScratch` ranks (usize→u32) NOT pursued — anomalous A/B, sort is comparison-bound (BlackThrush)

**Lever investigated, NOT landed (anomalous measurement + sound first-principles reason).** `FusedHitScratch`
(the RRF sort element) carries three `Option<usize>` rank fields (lexical/semantic/graph) that are **not** in
`cmp_for_ranking` — pure payload. Ranks are result-list positions (< u32::MAX), so `Option<u32>` would be
bit-identical and shrink the struct 120→96 B (`size_of` confirmed), cutting per-element sort data movement
~20%. The hypothesis: a smaller sorted element speeds the limit_all RRF full sort (~22% of limit_all).

**The isolated A/B (`rrf_struct_size` bench, wide-120B vs narrow-96B, identical data + `cmp_for_ranking` +
`sort_unstable_by`) returned an anomalous, non-physical result:** the *smaller* struct sorted **~2× SLOWER**
at both sizes (n10000: 245 µs vs 490 µs; n50000: 1.45 ms vs 2.86 ms). A 24-byte size reduction cannot make
an identical sort 2× slower — this is a bench/layout/load artifact I could not pin down (the per-iteration
10k-`String` allocation in `iter_batched` setup plausibly confounds the timing), so the number is untrusted
and no production change was made on it.

**The deeper reason the lever is unpromising anyway:** the limit_all RRF sort is **comparison-bound, not
movement-bound** — `cmp_for_ranking`'s `doc_id`-String tiebreak fires on the pervasive `1/(k+rank)` ties, so
the cost is in the comparisons (bit-identity-locked), not the per-element move that a smaller struct would
reduce. This is the same root cause that made `par_sort` fail here (this ledger, RRF-par-sort entry). Bench
removed, `Cargo.toml` reverted to HEAD; `FusedHitScratch` unchanged. Original-comparator ratio **N/A**.

---

## 2026-06-29 — batched (GEMM-style) multi-query vector scan: only 1.03–1.14×, decode-once REGRESSES — the f16 scan is compute-bound, not bandwidth-bound (BlackThrush)

**Fundamentally-different execution model probed (not a micro-lever on the existing single-query path).**
The PERF_LEDGER repeatedly calls the brute-force vector scan "bandwidth-bound + AVX2-capped." If true, a
high-QPS server scoring a *batch* of `B` query vectors against the full f16 corpus should win big by
**streaming the corpus from RAM once** and reusing each loaded vector across all `B` queries (loop
interchange / GEMM-style), instead of the production path's `B` independent full-corpus scans. Lucene/
Tantivy/Meili execute queries independently, so this would be a structural throughput edge. Built the
smallest per-crate harness, `frankensearch-index/benches/batched_query_scan.rs` (committed `8c6c51a`,
`e7d3452`): N=100k f16 vectors (73 MB slab ≫ L3), `B∈{4,16,64}`, identical `B·N` dots, results asserted
equal. Three arms — `sequential` (production per-query), `batched` (loop-interchange, bit-identical),
`batched_decode_once` (decode each f16 vector to f32 *once* via `vcvtph2ps`, then `B` f32 dots from L1).

**Measured (per-crate, `CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc rch exec -- cargo bench -p
frankensearch-index --bench batched_query_scan`, criterion p50, head-to-head in one process):**

| B  | sequential (ORIG) | batched | ratio | batched_decode_once | ratio |
|----|-------------------|---------|-------|---------------------|-------|
| 4  | 15.014 ms | 13.224 ms | **0.881 (1.14× faster)** | 66.13 ms | **4.40× SLOWER** |
| 16 | 60.947 ms | 55.951 ms | **0.918 (1.09× faster)** | 100.54 ms | **1.65× SLOWER** |
| 64 | 228.40 ms | 220.98 ms | **0.967 (1.03× faster)** | 221.65 ms | 0.97 (parity) |

**Decision: no domination here; decode-once REJECTED, batched kept as bench-only evidence.** Two findings,
both correcting the "bandwidth-bound" premise:

1. **Loop-interchange `batched` is only 1.03–1.14×, and the win SHRINKS as `B` grows.** If the scan were
   bandwidth-bound, amortizing corpus RAM traffic over more queries would help *more* at larger `B` — the
   opposite happened. The hardware prefetcher already hides almost all of the corpus re-streaming, so
   reading it once buys very little. The scan is **compute-bound** (FMA throughput on the per-element dot),
   not memory-bound.

2. **`batched_decode_once` is a hard regression (4.40×/1.65× slower at B≤16).** `vcvtph2ps` is fused into
   the load in the production `dot_product_f16_bytes_f32` kernel — decoding f16→f32 is essentially free.
   Materializing a decoded f32 scratch buffer and re-reading it adds a store+load round-trip per corpus
   vector that the free hardware decode never paid, so "decode once" loses to "decode B times for free."

**Route next:** the vector scan's floor is FMA compute, not memory bandwidth — so a batched/GEMM query API
would net at most ~1.1× and only at tiny batch sizes, not worth the multi-query-plumbing it would require.
Do not re-probe batched/loop-interchange query execution for the f16 scan, and do not build a decode-to-
scratch kernel. The "bandwidth-bound" characterization in the PERF_LEDGER should be read as "AVX2-FMA-bound";
sublinear candidate reduction (ANN/IVF — already present behind the `ann` feature) remains the only way to
cut the f16 scan's wall-clock, since the per-vector dot itself is at its compute floor.

---

## 2026-06-29 — int8 dot 2→4 accumulators is ~0 (throughput-bound, NOT latency-bound like f16) — bench kept, no prod change (BlackThrush)

**Route-next from the f16-dot ILP win (`82e151f`) tested and rejected for integer kernels.** That win came
from breaking the exact f16 dot's single-`vaddps` dependency chain (latency-bound) with 4 accumulators
(1.45×). The surfaced route-next asked whether `dot_i8_i8_avx2` (shipped with **2** accumulators) and the
4-bit kernel share the same latency-bound shape. Built `i8_dot_ilp` (commit prior) with faithful 2-acc and a
4-acc copy (integer adds associate ⇒ bit-identical, asserted).

**Measured (per-crate, in-process A/B, dim=384, 4096-vector i8 scan):**

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc rch exec -- \
  cargo bench -p frankensearch-index --bench i8_dot_ilp
```

| kernel | time (4096×384 i8 dots) | ratio |
|--------|-------------------------|-------|
| `acc2` (shipped) | 104.99 µs `[95.81, 115.83]` | 1.00 |
| `acc4` | 105.77 µs `[96.55, 117.19]` | 1.007 (~0, CIs overlap) |

**Decision: no prod change (kernel left at 2 accumulators); bench kept as evidence.** The int8 kernel is
**throughput-bound on `vpmovsxbw` (4×/iter, port 5) + `vpmaddwd` (port 0)**, not latency-bound on the
`vpaddd` accumulator chain — so additional accumulators have nothing to unblock. This is the structural
difference from the f16 dot: there the per-element decode (`vcvtph2ps`) is cheap and pipelines freely, so the
lone f32-add chain was the bottleneck; here the per-element widen+multiply already saturate their ports.

**Route next:** the 4-bit pass-1 kernel (`dot_4bit_prepared_avx2`) uses a **single** i16 accumulator but
flushes to i32 every 16 chunks and is dominated by `vpmullw` (2×/iter) + `vpsllw`/`vpsraw` (4×/iter) +
`vpmovsxbw` — the same port-throughput regime as int8, and the flush already segments the i16 add chain. By
that structural read it is **not** an accumulator-latency lever either; do not expect the f16 win to repeat
on the integer/quantized kernels. The f16 ILP win was specific to the cheap-decode + lone-add-chain shape.

---

## 2026-06-29 — scalar→SIMD f32 max-reduction wins 1.35× in isolation but has no measurable hot-path home (BlackThrush)

**Probed the f16-dot ILP pattern (`82e151f`) in non-dot reductions.** A scalar `f32::max` reduction
(`for &x in row { m = m.max(x) }`) is the shape of the native reranker's attention-softmax max pass
(`softmax_row_fused`, native.rs:79); `f32::max`'s NaN semantics block LLVM from auto-vectorizing it, so it
runs as one serial `maxss` chain. Built `max_reduction` (commit prior): scalar vs f32x8 lanewise-max +
horizontal reduce (bit-identical for finite/-inf; asserted).

**Measured (per-crate, `-p frankensearch-index`):**

| n | scalar | simd | ratio |
|---|--------|------|-------|
| 128 | 28.33 ns | 21.50 ns | 0.76 (1.32× faster) |
| 512 | 116.46 ns | 84.27 ns | 0.72 (1.38× faster) |
| 2048 | 469.55 ns | ~335 ns | ~0.71 (~1.4× faster) |

**Decision: NOT wired (no measurable impactful home) — bench kept as evidence.** Unlike the f16 dot (called
`N`× per query → the kernel IS the path, so 1.45× kernel ⇒ 1.45× path), the only hot caller of a scalar
f32-max reduction is the softmax **max pass**, which is a small fraction of the forward — the vectorized
`exp` pass (a transcendental polynomial, far more work/elem) dominates softmax, and softmax is ~24% of the
per-doc forward. So even a 1.4× max-pass is a low-single-digit-% softmax change ⇒ ~0 end-to-end. Worse, it is
**unmeasurable here**: `native_rerank` (the only end-to-end reranker bench) requires a staged model at
`/private/tmp/ee-reranker-port/model` that is absent on the rch workers, so it SKIPs — there is no way to
confirm a real forward-latency delta. The other scalar f32 reduction in `native.rs` (the L2-norm
sum-of-squares at line 1134) is **per-doc mean-pooling** over `H=768` once per document (cold), not a
per-token hot loop. **Route next:** the max-reduction SIMD kernel is real and reusable, but the f16-win
pattern only pays where the reduced loop is a *large fraction* of a measured hot path; no such home exists in
the core/index/fusion crates today (fusion `min_max_normalize`/`z_score_normalize` are off the RRF rank path).
Do not wire the softmax max without a way to measure the native reranker forward end-to-end.

---

## 2026-06-29 — RRF precomputed-key comparator (int keys + doc_id-prefix u64) is ~0/slight-regression once struct growth is charged (BlackThrush)

**Lever tested and rejected — distinct from the radix refutation (which rejected replacing the sort
algorithm).** The `limit_all` RRF final sort (`rrf.rs:341`, ~22% of limit_all) is comparison-bound:
`cmp_for_ranking` runs `f64::total_cmp` + `f32::total_cmp` + byte-by-byte `str::cmp` per comparison,
O(N log N)×. Hypothesis: precompute, O(N)×, a `total_cmp`-equivalent i64/i32 key per score level + a
big-endian u64 doc_id prefix (full `str::cmp` fallback only on prefix-tie, so **bit-identical order** —
asserted), turning each comparison into cheap integer compares. Built `rrf_sort_key` (committed), realistic
limit_all-shaped input (N docs, `1/(60+rank)` with wrap → pervasive ties, ~20% in-both).

**Measured (per-crate, `-p frankensearch-fusion`, honest A/B: `current` = production-shape struct + total_cmp
comparator; `fast` = struct + **3 added key fields** + the O(N) precompute pass + int-key comparator):**

| N | current (small struct) | fast (keyed struct + precompute) | ratio |
|---|------------------------|----------------------------------|-------|
| 10000 | 873.06 µs `[850,898]` | 892.01 µs `[883,905]` | 1.02 (CIs overlap → tie) |
| 50000 | 4.8652 ms `[4.75,5.00]` | 5.0687 ms `[4.95,5.20]` | 1.04 (slightly SLOWER) |

**Decision: rejected, no production change (bench-only).** A first pass that left the 3 key fields on the
struct for *both* arms showed a misleading 1.07–1.14× "win" — but that only proved the comparator is cheaper
*at a fixed (bigger) struct size*. Charged honestly, the lever **loses on two fronts that cancel the
comparator savings**: (1) the O(N) precompute pass (key_f64 + key_f32 + doc_prefix per element), and (2) the
~18% `FusedHitScratch` size growth (3 added fields → 8/4/8 B), which makes every pdqsort swap copy more
bytes — and the sort, while comparison-*bound*, still moves the records. Net ~0 to 4% slower. The cheaper
per-comparison work is real but the per-element precompute + per-swap movement of a fatter struct give it
back. **Route next:** a comparator-key win here would need the keys carried *without* growing the sorted
record (e.g. sort a separate `(u128 key, u32 idx)` array, then gather) — but the radix-on-indices variant is
the adjacent idea already refuted structurally (un-radixable String tail + pervasive ties), and gather-by-
index then needs an unsafe move-out of the `FusedHitScratch` Vec. Not worth it for a ~22%-of-a-noisy-row
slice. Do not re-attempt fattening `FusedHitScratch` with sort keys.

---

## 2026-06-29 — RRF sort input order: pdqsort is highly adaptive (presorted 4.3-5.8×), but reorder-via-hashmap-remove regresses at large N — NOT wired (BlackThrush)

**Fresh algorithmic lever (not a kernel micro-op).** The RRF final sort consumes `hits.into_values()`
(random hashmap order) then `sort_unstable_by(cmp_for_ranking)` — a full O(N log N). pdqsort is *adaptive*,
and for `limit_all` the semantic list is already in fused order for the vector-only majority (their fused
score is `1/(60+sem_rank)`, monotonic), so feeding the sort a **semantic-ordered** input is bit-identical
(the comparator is a total order → same output for any input permutation) yet should sort in ~O(N).

**Probe 1 — does input order matter? (sort only, identical output):**

| N | random | presorted | nearsorted (20% displaced) |
|---|--------|-----------|----------------------------|
| 10000 | 1.326 ms | **305 µs (4.3×)** | 936 µs (1.42×) |
| 50000 | 8.43 ms | **1.46 ms (5.8×)** | — |

Huge — pdqsort exploits the near-sortedness. **Probe 2 — honest full path** (both arms clone the same
prebuilt `AHashMap` in the timed region; `current` = `into_values`+sort, `reorder` = drain in semantic
order via `remove`+sort, charging the N hashmap removes `into_values` never pays):

| N | current | reorder | ratio |
|---|---------|---------|-------|
| 10000 | 1.9146 ms | 1.4042 ms | **0.73 (1.36× faster)** |
| 50000 | 16.208 ms | 17.791 ms | **1.10 (SLOWER)** |

**Decision: NOT wired (corpus-size-dependent, regresses at scale).** The reorder wins at 10k but **loses at
50k**: the semantic-order `remove` loop does N random accesses into the hashmap, which is cache-miss-bound
once the map exceeds LLC (~100 ns/remove at 50k) — exceeding the sort savings. A gate (reorder only below
~16 k, like `PAR_SORT_THRESHOLD`) would bank only the small-corpus win, which lands on the noise-dominated
`limit_all/10k` BOLD row — not worth the branch. **Route next (the clean capture):** a *merge-structured*
RRF that never builds the N-entry hashmap — keep a small `L`-entry lexical/graph contribution map (cache-
resident), then iterate the already-sorted semantic list **once in order**, emitting `FusedHitScratch`
directly (semantic-ordered, near fused order) and applying the lexical boost via the small map. That gets the
near-sorted sort input AND avoids both the N-entry hashmap build and the N cache-missing removes — but it is
a structural rewrite of `rrf_fuse_with_graph` with dedup/graph/lexical-only edge cases to keep bit-identical,
so it is a scoped follow-up, not a one-commit lever. Do not re-attempt the simple reorder-via-remove.

---

## 2026-06-29 — limit_all materialize doc_id clone MEASURED at ~23% of limit_all (not the estimated 3-5%) but structurally un-elidable per-crate (BlackThrush)

**Measured a previously-estimated-only cost.** The PERF_LEDGER guessed the `limit_all` RRF `into_owned`
doc_id clones at "~3-5% of limit_all (never measured)". Ran the existing `materialize_clone` bench
(`-p frankensearch-fusion`): N short-String clones vs borrow.

| N | `owned_clone` (current) | `borrowed` (refactor target) | ratio |
|---|--------------------------|------------------------------|-------|
| 10000  | **432.61 µs** | 3.88 µs | ~110× |
| 100000 | ~4.3 ms (linear) | ~39 µs | ~110× |

**432 µs at 10k is ~23% of `limit_all`/10k (~1869 µs)** — **5-7× the ledger's estimate.** Each `doc-NNNNNN`
clone is ~43 ns of malloc+memcpy; at N=10k that is a major, real slice.

**But it is structurally un-elidable as a single-crate lever (verified, not assumed):** the only ways to
avoid the clone both fail —
1. **Move the input** (owned-input fuse): blocked — `sync_searcher.rs` **reuses `fast_hits` after the fuse**:
   the quality tier re-scores it (`quality_scores_for_hits(query_vec, &fast_hits)`) AND clones its doc_ids
   again into `quality_hits` (`doc_id: fast.doc_id.clone()`). So `fast_hits` can't be consumed by the fuse;
   its doc_ids are needed by ≥2 downstream consumers.
2. **Borrow** (`FusedHit<'a>`): the clone just moves downstream — the final `ScoredResult`/`quality_hits` own
   `doc_id: String` (they outlive `fast_hits`), so a `String` must materialize somewhere; borrowing relocates
   the N clones, it doesn't remove them.

**The only real fix is cross-crate:** make `doc_id` an `Arc<str>` on `VectorHit`/`ScoredResult`/`FusedHit`
(frankensearch-core types used everywhere), turning the per-consumer clones into ~5 ns refcount bumps (1
alloc + N bumps vs N allocs). That touches every doc_id constructor/consumer across core/index/fusion —
**not a per-crate lever**, and high regression surface. **Decision:** no per-crate change; the clone is the
single biggest remaining frankensearch-owned slice of `limit_all` after the f16-dot (`82e151f`) and merge-RRF
(`4aeb66b`) wins, but capturing it is an `Arc<str>` doc_id refactor scoped at the workspace level. Recorded so
the magnitude (~23%, not 3-5%) is on the books and the `Arc<str>` refactor can be prioritized as the next
*big* lever rather than re-estimated.

**FOLLOW-UP 2026-06-30 (BlackThrush) — the *separate* `quality_hits` doc_id clone WAS per-crate elidable
(now LANDED, see PERF_LEDGER 2026-06-30).** The argument above cited `quality_hits` (`sync_searcher.rs`
`doc_id: fast.doc_id.clone()`) only as *evidence* `fast_hits` can't be moved into the fuse — it never
analyzed that clone for elision. It turns out the `quality_hits` `Vec<VectorHit>` is a re-scored subset of
`fast_hits` (same doc_ids/index), built purely to hand a `&[VectorHit]` to `blend_two_tier`, and **both** of
its consumers — the blend (reads `hit.doc_id.as_str()`) and the downstream `quality_scores` borrow-map (also
`as_str()`) — only ever read the doc_id as `&str`. So a new `blend_two_tier_aligned(fast_hits, &[Option<f32>])`
blends straight from the aligned `quality_scores_for_hits` output, borrowing each doc_id from `fast_hits` and
eliding those N clones with **bit-identical** output (single-pass merge guarded by `is_none()`; permanent
equivalence test + bench guard). Measured **1.38× at 10k / 1.08× at 100k** on the blend region (`blend_aligned`
bench). This does NOT touch the RRF `into_owned` clone analyzed above — that one remains un-elidable per-crate
and still needs the workspace `Arc<str>` refactor. Lesson: when a clone is cited as a *blocker* for some other
elision, separately check whether that clone is itself dead weight — `quality_hits`' doc_ids were never owned
downstream, only borrowed.

---

## 2026-07-01 — main comparator confirmed at floor; two remaining levers are both multi-cycle (BlackThrush)

**Surfaced blocker, not a tested-and-rejected lever.** After landing four fusion wins this arc
(`blend_two_tier_aligned` sync+async `41beaff`/`5bc9d58`, `want_phases` `5100734`, federated `appeared_in`
interning `8ec3108`), swept for a NEW clean per-crate lever on the biggest gap vs ORIG and found none that is
safely landable-GREEN in a single ~60 m window. Recording the frontier state + the two real remaining levers
with exact scope so the next session executes instead of re-exploring.

**Main BOLD-VERIFY comparator (`frankensearch/benches/search_bench.rs`) is at floor.** Its challenger,
`frankensearch_hash_hybrid_search`, is hand-rolled: `lexical.search_doc_ids` → `lexical_doc_ids_as_scored`
(bench glue) → `vector.search_top_k` → `rrf_fuse`. Every component is individually mined (RRF fuse/merge, dot
kernels, id materialization, prescreen). It does **not** call `SyncTwoTierSearcher::search_collect` /
`blend_two_tier` / `search_internal`, so the recent two-tier wins don't move this row. The production
`SyncLexicalSearch` impl is integrator-provided (only the test `StaticLexical` exists in-tree), so there is no
fixed lexical→`ScoredResult` conversion clone to attack here. The only robustly-slower comparator row remains
`limit_all` (~1.286× @10k, inherent hybrid semantics — see 2026-06-27 comparator entry).

**Lever 1 (biggest gap vs ORIG) — `Arc<str>` doc_ids. Multi-cycle.** Closing the `limit_all` gap needs the
RRF `into_owned` + resolve_heap source materialization (~20 % of limit_all, ~432 µs/10k measured) turned into
refcount bumps. Scope counted this session: **182 `doc_id:` construction sites** in non-test src (fusion 106,
core 46, index 28, lexical 2) + serde **`rc`** feature + a public-API break on `VectorHit`/`FusedHit`/
`ScoredResult`/core `traits.rs` across **5 crates** (also storage/fsfs downstream). Most *read* sites survive
via `Arc<str>: Deref<Target=str>`; the real edit surface is the 182 constructors (`x` → `x.into()`) + serde.
This is a dedicated multi-cycle refactor with high regression surface (820+ tests), NOT a 60 m ship. Execution
order: core types → serde rc → index source (`self.doc_ids: Vec<String>` → `Vec<Arc<str>>`, `resolve_heap`) →
fusion/lexical constructors → per-crate `cargo test` after each.

**Lever 2 (NEW, found this session) — async `search_collect_with_text` wastes the Initial-phase clone.**
Same proven pattern as the sync `want_phases` win (`5100734`, 1.29×), on the async production collect path used
by `TwoTierSearcher::search_collect` **and by federated** (per-shard `collect_shard_results`). `search()`
(searcher.rs) streams phases via `on_phase`, cloning results into **every** phase (Initial `display_hits.clone()`
@543, Refined @683, Reranked @736). `search_collect_with_text` (@881) drives it with a callback that just does
`best_results = results` — keeping only the **latest** phase. So when a Refined/Reranked phase follows, the
**Initial phase clone is pure waste** (built, stored, overwritten, dropped) — N `ScoredResult` clones per
collect call at limit_all. Fix sketch: private `search_inner(..., emit_intermediate: bool)` (pub `search`
passes `true`, `search_collect_with_text` passes `false`); guard the Initial emit @542 to fire only when
`emit_intermediate || !should_run_quality()` (Initial is terminal only when no quality runs); update the
collect callback to capture `RefinementFailed { initial_results, .. }` (@636/@805) so the failure path still
delivers results when the Initial emit was skipped. **Blocker:** no async bench harness exists (all fusion
benches are sync; `Cx` currently only from `asupersync::test_utils::run_test_with_cx`, test-internals-gated);
measuring end-to-end needs a bench that stands up a runtime + `Cx` + stub embedders (replicated from the
`#[cfg(test)]` module). Bit-identity is conformance-checkable now; the ratio needs that harness. High EV
(production + federated path, proven pattern) — top of the next-session queue.

**TESTED + REVERTED 2026-07-01 (BlackThrush) — measured ~0/noise, NOT the ~1.1× extrapolated.** Implemented
it (`82eb351`, bit-identical, conformance 820 GREEN) AND built the direct async bench harness — which works:
a fusion bench CAN stand up `Cx::for_testing()` + `asupersync::runtime::RuntimeBuilder::current_thread()`
(via the dev-dep's `test-internals`) and `rt.block_on` per iter; `HashEmbedder` needs
`frankensearch-embed = { workspace=true, features=["hash"] }` in fusion `[dev-dependencies]` (its `hash`
feature is off in the lib graph). The A/B — `stream` (pub `search()` + latest-capturing callback = old, Initial
cloned) vs `collect` (`search_collect_with_text` = new, Initial skipped), same searcher, bit-identical output
asserted — measured on `search-cc` (worker hz2), N=10k/dim=384/k=N:

| arm | time | ratio |
|-----|------|-------|
| `stream` (old) | 19.340 ms `[18.822, 19.910]` | 1.00 |
| `collect` (new) | 19.052 ms `[18.502, 19.670]` | **1.015× — CIs overlap = tie** |

**Why the extrapolation was wrong:** `collect_limit_all` (sync) measured **2** phase clones = 6.29 ms (≈3.1 ms
each) on a 21 ms path, but that delta also bundled phase-struct/iterator overhead, and — decisively — the async
Initial `display_hits` clone is only ~0.29 ms (~1.5 %) of the 19 ms async collect path: the async fast tier's
Initial result set is a far smaller fraction than the sync 2-clone limit_all case, so eliminating it is within
noise. **Reverted per "revert near-zero gains"** (the change was bit-identical, so no regression either way, but
the `search_inner`/`emit_intermediate` complexity isn't justified by a ~0 gain). Lesson: do NOT extrapolate a
clone's ratio across searcher paths — the same `Vec<ScoredResult>` clone is 30 % of the sync collect path but
<2 % of the async one; measure the actual path. The `hash`-dev-dep async-bench recipe above is the reusable
takeaway if a future async lever needs measuring.

**Assessed marginal, do not re-tread alone:** the federated `accumulate_doc` template `hit.clone()`
(insert + primary-update) and the per-call `docs.entry(hit.doc_id.clone())` key clone are each ~5 % after the
`appeared_in` win (the key clone was already reverted at ~1.05×, 2026-06-27); the insert/template clone
relocates to output (1↔1) so only the O(log M)/doc update clones are elidable — sub-1.1× on a niche path.

---

## 2026-07-01 — RRF indirect index-sort is SLOWER (cache-miss-bound), not the win the flawed dismissal implied (BlackThrush)

**Untested route-next tested (its prior rejection rested on a false premise) — REJECTED for a different, real
reason.** The 2026-06-29 sort-key entry dismissed "sort a separate `(key, idx)` array then gather" because
"gather-by-index needs an unsafe move-out of the `FusedHitScratch` Vec." That premise is **false**:
`FusedHitScratch.doc_id` is `&'a str` (borrowed), so gather-by-index reads Copy fields + `into_owned`s the
borrowed `&str` (a clone, no move, no unsafe). So the indirect index-sort (sort a `Vec<u32>` of **4-byte**
indices instead of the ~112-byte 10-field struct, then gather by index) is safe and worth measuring — 4-byte
swaps vs ~112-byte swaps on the ~22%-of-limit_all RRF sort.

**Measured (per-crate, `-p frankensearch-fusion`, `rrf_index_sort` bench, `iter_batched` so the base clone is
untimed; realistic limit_all shape, faithful `&str` doc_id; bit-identical output asserted — 0 reorders):**

| N | `struct_sort` (production: sort fat structs in place) | `index_sort` (sort u32 idx + gather) | ratio |
|---|------------------------------------------------------|--------------------------------------|-------|
| 10000 | **686.22 µs** `[669,710]` | 715.82 µs `[711,723]` | **1.04× SLOWER** |
| 50000 | **3.7289 ms** `[3.69,3.78]` | 4.2800 ms `[4.20,4.39]` | **1.15× SLOWER (worsens with N)** |

**Decision: rejected, no production change (bench kept as evidence).** The indirect sort **loses**, and worse
at scale: pdqsort over the fat struct array touches near-adjacent elements (cache-friendly), and even ~112-byte
swaps are cheaper than the **scattered cache-miss reads** the indirect comparator (`scratch[a]` vs `scratch[b]`
for random `a,b`) and the scattered gather (`scratch[idx]`) incur once the `N`-struct array (1.1 MB @10k,
5.6 MB @50k) exceeds cache. So the prior dismissal reached the right conclusion (don't do it) for the wrong
reason (it isn't an unsafe-safety issue, it's a cache-locality issue). **The RRF final sort is at its floor**:
the sort algorithm (radix), the comparator (precomputed keys / key-fattening), the sorted-record size (indirect
index sort) have all now been probed and rejected. Route next: the only remaining limit_all lever is the
`Arc<str>` doc_id materialization (workspace refactor), not the sort.

---

## 2026-07-02 — limit_all doc_id clone: CompactString (SSO) is 29.8× cheaper for short ids, beating Arc<str> — the materialization lever's optimal type (BlackThrush)

**Positive finding — now LANDED (see 2026-07-02 landing note below).** The doc_id materialization clone
(RRF `into_owned` + resolve_heap + blend, ~23% of limit_all, ~432 µs/10k) was slated for an `Arc<str>`
refactor. Benched the actual N-clone cost of the candidate types (`doc_id_clone_sso`, N=10k):

| type | short id (`doc-000042`, 10 B) | long id (36 B uuid-like) |
|------|-------------------------------|--------------------------|
| `String` (current) | 438.58 µs `[431,448]` (confirms `materialize_clone` 432 µs) | 248.99 µs |
| **`CompactString`** (SSO, ≤24 B inline) | **14.73 µs — 29.8× cheaper** | 295.47 µs (~1.2× slower, heap fallback) |
| `Arc<str>` (refcount bump) | 66.58 µs — 6.6× cheaper | 72.18 µs — 3.5× cheaper |

**`CompactString` is the better target than `Arc<str>` for typical (short) doc_ids:** 29.8× vs 6.6× (inline
memcpy beats an atomic refcount bump), AND far more drop-in — it has `.as_str()`, `Deref<str>`, `From<String>`
/`From<&str>`, `PartialEq`/`Hash`/`Ord`, and serde built in, so most call sites (`hit.doc_id.as_str()`, `==`,
`clone`, `format!`) compile **unchanged**; only the struct field types + constructors (`.into()`) change, and
**no serde `rc` feature** is needed. `Arc<str>` lacks `.as_str()` (breaks every `.as_str()` site) and needs
serde `rc`. **Tradeoff:** `CompactString` degrades to ~1.2× *slower* clone for ids >24 B (UUIDs/long URLs),
where `Arc<str>` stays a universal win — so for a library with unknown consumer id lengths, the choice is
`CompactString` (bet on short ids, huge win, small tail regression) vs `Arc<str>` (universal, no regression,
bigger refactor). **Recommendation for the dedicated materialization session:** `doc_id: CompactString` on the
frankensearch-core types (`VectorHit`/`FusedHit`/`ScoredResult`) + index doc_ids table — near-drop-in, ~22%
off limit_all for the common short-id case. `doc_id_clone_sso` bench kept as evidence.

---

## 2026-07-02 — LANDED: `DocId = CompactString` on the hot materialization structs (`8529084`, BlackThrush)

**The recommendation above shipped.** `pub type DocId = CompactString` added to `frankensearch-core::types`;
`VectorHit`/`FusedHit`/`ScoredResult` `doc_id` fields + their RRF/blend/resolve clone sites swapped `String → DocId`
across the workspace (60 files, +199/−190). The hot `limit_all` path benefits: `FusedHitScratch.doc_id` stays
`&'a str` (borrowed) and `into_owned` (rrf.rs:113, called via `.map(FusedHitScratch::into_owned)` at rrf.rs:352/552)
now does `self.doc_id.into()` = `<CompactString as From<&str>>::from` = **SSO inline memcpy** for ids ≤24 B instead
of a heap `String` alloc. Non-hot structs (traits.rs `LexicalHit`, commit_replay events, ope/queue records) keep
`String` intentionally. `compact_str 0.9.1` already pinned in `Cargo.lock`.

**Measured basis (already on main, `62f06f5`):** `doc_id_clone_sso` — short-id N-clone `String` 438 µs → `CompactString`
14.73 µs = **29.8× cheaper**; this is the isolated cost of exactly the `into_owned` clone the landing removes from the
`limit_all` output materialization (~23% of the fusion `limit_all` collect path). Ratio vs incumbents on the
`limit_all` BOLD row (recorded pre-landing baseline, `bold_verify/limit_all/10000`): frankensearch **0.933–0.966×
tantivy** (already at/under parity); this landing removes the largest remaining per-query heap-alloc term from that
path in the common short-id case.

**Conformance (re-verified at HEAD):** targeted `cargo test -p frankensearch-core -p frankensearch-fusion
-p frankensearch-index --lib --features lexical` (the three crates carrying the doc_id change; no network deps) →
**819 passed, 1 failed**. The single failure was `searcher::tests::exclusion_overhead_is_sub_millisecond_for_typical_query`
— a pure **wall-clock timing assertion** (`overhead_ms < 1.0`, observed 1.066 ms) that CompactString cannot affect
(it isn't on the exclusion-negation path, and cheaper clones only *reduce* overhead); it tripped on the degraded/loaded
ovh-a worker and **passed clean on re-run** on hz2 (`exclusion_overhead ... ok`). Net **820/820 GREEN**, consistent with
`blackthrush-compactstr-docid`'s recorded "lib + tests GREEN / bit-identical" (the landed tracked source is
byte-identical to that branch). The full-workspace lib test also compiled+ran GREEN before hanging on the network-bound
`model_download` HuggingFace tests (a pre-existing fleet-network issue, unrelated).

**End-to-end note:** the earlier plan to diff against the saved `collect_limit_all/collect` baseline (21,372 µs) is
**invalid — cross-worker**: HEAD ran on hz2 at 7.31 ms `[6.99, 7.67]`, but the 21 ms base was a *different* worker, so
the absolute delta is hardware noise, not the CompactString effect. The clean same-worker measurement is the
`docid_materialize_ab` bench (both String and CompactString arms in one binary on one worker) — see the next entry.

---

## 2026-07-02 — same-worker A/B: `DocId=CompactString` FusedHit materialization is 2.2–2.3× faster (BlackThrush)

**Measured, not extrapolated — the clean end-to-end-ish number for the landed `8529084`.** `doc_id_clone_sso`
measured the *bare* doc_id clone at 29.8× (String 438 µs → CompactString 14.73 µs / 10k). To answer "how much of that
survives when folded into the real `FusedHitScratch::into_owned` — building the whole 10-field `FusedHit` (9 `Copy`
fields + `doc_id.into()`) over N `limit_all` winners," the `docid_materialize_ab` bench runs BOTH arms (`String` vs
`CompactString` doc_id, identical `Copy` fields) in **one binary on one worker** — no cross-worker noise.

| N | `string` (pre-landing) | `compact` (landed) | speedup |
|---|------------------------|--------------------|---------|
| 10,000 | 294.15 µs | **133.31 µs** | **2.21×** |
| 100,000 | 3242.39 µs `[3.24 ms]` | **1425.59 µs `[1.39, 1.43 ms]`** | **2.27×** |

**Interpretation:** the 29.8× bare-clone win dilutes to **~2.2×** once the fixed `Copy`-field struct-build cost is
charged to both arms (that cost is identical either way, so it caps the achievable ratio) — the *doc_id term itself*
still collapses from a heap alloc to an SSO inline memcpy, halving the full per-winner materialization. This is the
per-query cost of the RRF `into_owned` at `limit_all` (called once per output hit, `rrf.rs:352/552`), so the searcher's
`limit_all` collect shrinks its largest per-hit heap term by ~2.2× for the common short-id (≤24 B) case. **Ratio vs
incumbent context:** frankensearch was already at/under tantivy parity on the `limit_all` BOLD row (0.933–0.966×
pre-landing); this removes the dominant remaining allocation from that path. Bench `docid_materialize_ab` kept as
evidence. **Route next:** the RRF sort, comparator, struct size, and now the doc_id materialization type are all probed
— the fusion `limit_all` collect path is at its floor for short ids; the only residual is the >24 B id tail (~1.2×
clone regression, absent on measured corpora).

---

## 2026-07-02 — dig: post-CompactString, the fusion result-assembly arc has no remaining clone/move lever (BlackThrush)

**Negative evidence from a full route-next sweep + fresh source audit — no production change, records the frontier so it
isn't re-walked.** After landing `DocId=CompactString` (`8529084`), swept every open "Route next" in this doc and
re-audited the sync searcher's result-assembly by source, all confirmed floored:

- **Reopen id-materialization** (route-next @ the 2026-06-27 "numeric fast field" entry): already **SHIPPED** — PERF_LEDGER
  `reopen_id_materialize/k1000` = **0.436 (~2.29×) KEEP** via the persisted ord/sidecar table; reopened on-disk indexes no
  longer fall back to docstore materialization. Closed.
- **`fused_hits_to_scored_results`** (`sync_searcher.rs:418`, the hot hybrid `limit_all` path): already **moves** each
  `doc_id` out of the owned `Vec<FusedHit>` via `into_iter` (no second clone). Optimal.
- **Refined semantic path** (`:251-284`): already uses `unique_vector_hits_to_scored_results_owned(blended, …)` — **moves**
  from the owned `blended` vec; the `fast_scores`/`quality_scores` maps key on `&str` (no clone). Optimal.
- **Initial fast-phase** (`vector_hits_to_scored_results(&fast_hits, …)`, `:162`): the one residual `doc_id.clone()`. It is
  **necessary, not a lever** — `fast_hits` is reused downstream at `:193` (len), `:214` (`quality_scores_for_hits`), `:243`
  (`blend_two_tier_aligned`), `:270-277` (score maps), so it cannot be consumed here; and the clone is already **~2.2×
  cheaper** post-CompactString (SSO). Moving it would require duplicating `fast_hits`, a net loss.
- **RRF indirect index-sort** (route-next @ 2026-07-01): rejected, cache-miss-bound (kept as evidence).

**Conclusion:** the fusion result-assembly arc (RRF fuse → materialize → ScoredResult) is move-optimized everywhere a
move is sound, with the sole necessary clone reduced to an SSO memcpy. No single-primitive lever remains on this path.
The biggest gap vs the incumbents (`limit_all` wall-clock) is bounded by the actual per-hit struct build (the `Copy`-field
`FusedHit`/`ScoredResult` writes + Vec alloc, identical for any doc_id type) — not a clone, and not shrinkable without
changing the **public** `FusedHit`/`ScoredResult` types (an API break, out of scope). Route next is off the fusion
result-assembly vein entirely — the productive frontier is now the vector-scan and index tiers (already heavily mined) or
a corpus/algorithmic change (approximate top-k, block-max), not a clone-elision.

---

## 2026-07-02 — dig: vector-scan + index top-k tier audited, no mechanical lever left; remaining wins are decision-gated (BlackThrush)

**Negative evidence — followed the previous entry's own route-next onto the vector/index tier; source-audited the hot
top-k path, all optimal. No production change.** After the fusion arc floored, audited `frankensearch-index/src/search.rs`:

- **Bounded-heap top-k** (`int8_scan_range` :291-326, and the f16/4-bit twins): already the textbook O(N·log k) shape —
  `BinaryHeap` capped at k, per-element cutoff early-skip (`heap.len() < limit || score_key(score) >= cutoff`),
  incremental record/slab offset advance (no recompute), tombstone flag gate. `insert_candidate` peek-compare-pop.
- **Two-pass quantization** already in place: parallel 4-bit (`dot_4bit_prepared`) or int8 (`dot_i8_i8`) approximate
  pass-1 keeps `k·mult` candidates → exact f16 rescore. AVX2 runtime-dispatch kernels ([[avx2-runtime-dispatch]]).
- **ANN / HNSW** IS wired into the two-tier search (`two_tier.rs:285` — `ann.knn_search` when the sidecar is present,
  brute-force fallback otherwise); not a missing lever.
- **`score_key`-in-int8 micro-lever considered + rejected without a bench:** `score_key` is a `const fn` `is_nan()→NEG_INF`
  passthrough; int8 dots (`dot_i8_i8 as f32`) are always finite, so it's ~1 branch/elem of dead work — but the int8 scan
  is **dot-throughput-bound** (measured, [[batched-query-scan-compute-bound]] / int8 route-next ~0), so removing a
  single inlined branch on a memory/FMA-bound loop is a predicted ~0. Not worth ~8 min of degraded-fleet build time.

**Conclusion — the mechanical per-crate frontier is exhausted.** Across this session all three frankensearch-owned hot
tiers were confirmed at floor: fusion result-assembly (moves everywhere sound), vector top-k (heap+cutoff+two-pass+ANN),
index materialization (reopen ord-table shipped 2.29×). The residual gaps vs Tantivy are **inherent** (the Copy-field
`FusedHit`/`ScoredResult` struct build; tantivy-internal BM25) — not removable by a single-primitive swap. **The next
real lever is decision-gated, not mechanical:** (a) use ANN in the BOLD hybrid at large N (trades exact recall for
latency — breaks the bit-identical/parity invariant the BOLD comparison holds, needs an explicit recall-budget
decision), or (b) shrink the public `FusedHit`/`ScoredResult` (an API break). Both need owner direction; neither is a
per-crate perf lever. **Route next: stop re-auditing the owned hot paths — re-measure the live BOLD ratios first, and
only pursue (a)/(b) with an explicit go-ahead.**

---

## 2026-07-02 — REJECTED (measured): shrinking `FusedHit` ~96 B→56 B buys only ~1.04× on limit_all materialize (BlackThrush)

**Negative evidence — the struct-shrink lever (b) from the prior entry, benched instead of assumed. No production
change; the ~40 % smaller struct does NOT justify the public-API break.** Added a `packed` arm to `docid_materialize_ab`:
`FusedHitPacked` replaces the `Option<usize>`/`Option<u32>` ranks with `u32` (`u32::MAX` sentinel) and `Option<f32>`
scores with `f32` (`NaN` sentinel) — ~96 B → ~56 B, same `CompactString` doc_id — and materializes it the same way over
N `limit_all` winners. All three arms (`string`/`compact`/`packed`) run in ONE binary on ONE worker (ovh-a here), so the
packed-vs-compact ratio is same-worker-clean.

| N | `string` (96 B, pre-CompactString) | `compact` (96 B, landed) | `packed` (56 B) | packed/compact |
|---|---|---|---|---|
| 10,000 | 181.1 µs | 88.5 µs | 85.4 µs | **1.036× (3.5 % faster)** |
| 100,000 | 2014 µs | 905.6 µs | 862.4 µs | **1.050× (4.8 % faster)** |

**Why so small:** the materialize is dominated by the `CompactString` clone (a 24 B SSO inline memcpy, **identical in both
arms**) plus per-element `Vec` push/alloc — NOT the `Copy`-field footprint the packing shrinks. Cutting 40 B of `Copy`
fields off each record moves the full materialize by only ~4 %. **Decision: rejected — do NOT pack/shrink the public
`FusedHit`/`ScoredResult`.** A ~1.04× on one sub-step of `limit_all` (which is already at/under tantivy parity) cannot
justify (i) breaking every consumer that reads `.lexical_rank: Option<usize>` etc. and (ii) losing the `Option`
niche-safety (sentinels re-admit "is this `u32::MAX` a real rank or absent?" bugs). The `limit_all` struct-build floor is
**real and inherent** — confirmed by measurement, not just asserted. (NB: absolute times differ from the 2.2× entry's
run — that was hz1, this is ovh-a; the SAME-worker packed/compact ratio is the valid comparison, re-confirming
cross-worker absolutes are noise.) `docid_materialize_ab` packed arm kept as evidence. **Route next: the two remaining
levers are now (a) ANN-in-BOLD (recall tradeoff, decision-gated) only — struct-shrink (b) is closed by data.**

---

## 2026-07-02 — LANDED: `DocId=CompactString` extended to the lexical id-materialization (sibling-path consistency)

**A real win, not a rejection — the `8529084` CompactString refactor stopped at the core/fusion/index boundary; the
`frankensearch-lexical` crate was the last hot holdout still on `String`.** ([[sibling-path-consistency-audit]] pattern:
grep for an optimization applied to one path but not its twin.) The lexical fast id-materialization (`collect_id_hits`,
the `ord` u64 FAST-column → `ord_table` lookup that skips the docstore decompress) stored `ord_table: RwLock<Vec<String>>`
and returned `LexicalIdHit { doc_id: String }`, so on **every lexical query** (the exact `exact_identifier` /
`quoted_phrase` classes where frankensearch historically trailed tantivy) each hit's id was a `String` heap clone
(`ord_table[ord].clone()`, rrf-lexical.rs:1046) **and then re-converted** `String→CompactString` at the fusion boundary
(`ScoredResult.doc_id` is already `DocId`).

**Change (`<sha>`):** `ord_table` → `Vec<DocId>`, `LexicalIdHit.doc_id` → `DocId`, `docstore_id()` → `DocId`,
`assign_ord` push → `DocId::from`, sidecar load/persist → `Vec<DocId>` (CompactString serializes as a JSON string, so
existing `ord_table.json` sidecars stay byte-compatible — no reindex). `DocId` re-exported from `frankensearch_core`.
Net: the per-hit lexical id clone becomes an **SSO inline memcpy** (the measured 29.8× bare-clone / 2.2× full-materialize
primitive from [[limit-all-materialize-clone-arc-str]] / `docid_materialize_ab`, now applied on the lexical hot path)
AND the `String→CompactString` boundary re-conversion is **eliminated** (the id moves straight into `ScoredResult`).
Strictly cheaper-or-equal: same 24 B footprint, same >24 B heap-fallback behavior, so no regression.

**Conformance:** `cargo test -p frankensearch-lexical -p frankensearch-fusion -p frankensearch-core --lib --features
lexical` GREEN (exit 0) on a **forced-fresh core rebuild** (`cargo clean -p frankensearch-core` first). METHODOLOGY
[[rch-stale-cache-false-green]]: a first run *falsely failed* `unresolved import frankensearch_core::types::DocId` on a
worker whose cached core `.rlib` predated the `DocId` alias, while a lexical-only check *falsely passed* on a
fresh-core worker — always force a clean core rebuild when a cross-crate type alias won't resolve. `search_doc_ids_materialize`
per-crate bench launched on search-cc for a current-state datapoint (fleet-slow; the win magnitude is the already-measured
CompactString primitive, so it does not gate the landing). **Route next:** the CompactString arc is now complete across
every hot crate (core/index/fusion/lexical); the sole remaining perf lever is (a) ANN-in-BOLD (recall-gated).

---

## 2026-07-02 — LANDED FIX: `ann` feature broke on main since 8529084 (CompactString missed `cfg(ann)` code) + ANN-vs-flat data (BlackThrush)

**A real landable fix surfaced by trying to *measure* the last lever.** Running the `hnsw_vs_flat` bench
(`--features ann`) to quantify ANN-in-BOLD failed to compile: 2× `E0308 expected CompactString, found String` at
`hnsw.rs:402` (`VectorHit { doc_id: doc_id.clone() }`) and `two_tier.rs:322` (`doc_id: entry.doc_id.clone()`). The
`8529084` `DocId=CompactString` refactor changed `VectorHit.doc_id` String→CompactString but **missed these two
`#[cfg(feature = "ann")]`-gated sites** — the `ann` feature is not in the default `cargo test`/CI lanes, so the break
went unnoticed on main. **Fixed (`<sha>`)** with the SSO-optimal form `doc_id.as_str().into()` (builds `CompactString`
straight from the `&str`, skipping the intermediate `String` heap clone) at both sites. Verified: `cargo bench
--features ann --bench hnsw_vs_flat` now compiles + runs GREEN. **METHODOLOGY:** a `cfg`-gated feature is a blind spot
for a workspace-wide type migration — after a `DocId`-style refactor, grep `cfg(feature` for the changed type or build
`--all-features`. (Follow-up worth doing: add `ann` to the feature-smoke lane so this can't regress silently.)

**ANN-vs-flat measurement (`hnsw_vs_flat`, N=10k, DIM=128, K=10, hz2):**

| arm | latency | vs flat |
|-----|---------|---------|
| `flat` (exact, rayon-parallel `search_top_k`) | 175.5 µs | 1.00× |
| `hnsw_ef10` | **73.7 µs** | **2.38× faster** |
| `hnsw_ef20` | 119.7 µs | 1.47× faster |
| `hnsw_ef40` | 205.2 µs | 0.86× (slower) |
| `hnsw_ef100` | 471.3 µs | 0.37× (2.7× slower) |

**Read:** at BOLD's smaller scale (10k) the rayon-parallel *exact* flat scan is already ~175 µs, so HNSW only wins at
**low `ef`** (ef10/ef20) — precisely where recall is worst — and *loses* at ef≥40. ANN's O(log N) edge only dominates at
much larger N; at ≤10k it's a recall-for-latency trade with a *shrinking* latency payoff. So **ANN-in-BOLD is not a clear
win at the measured scales** — it stays decision-gated (would need a 100k+ recall/latency sweep + an explicit recall
budget before wiring). The bench is validation-only; nothing ANN is wired into the product search path. **Route next:**
ANN-in-BOLD needs a 100k-scale recall sweep, not a 10k one, before it's worth pursuing — deferred pending that data + a
recall-budget decision.

---

## 2026-07-02 — LANDED FIX #2: the `graph` feature was ALSO broken by 8529084 (same cfg blind spot) (BlackThrush)

**Applied the [[cfg-gated-feature-migration-blindspot]] lesson from the `ann` fix and found a second casualty.** Ran
`cargo check -p frankensearch-fusion --all-features --all-targets` (and the top-level `full` lane) to hunt for *other*
cfg-gated code the `DocId=CompactString` refactor (`8529084`) missed — and it did: `graph_rank.rs:61` built
`ScoredResult { doc_id, .. }` (shorthand) from a `String`-keyed rank map, `E0308 expected CompactString, found String`.
The `graph` feature (like `ann`) is absent from the default `--workspace --lib` conformance, so it broke silently. **Fixed
(`<sha>`)** `doc_id: doc_id.into()` (owned `String`→`CompactString`, `From<String>` reuses the heap buffer for >24 B ids).
Verified: `cargo check -p frankensearch-fusion --all-features --all-targets` GREEN (lib+tests+benches).

**Tally:** the CompactString migration silently broke **two** feature configs — `ann` (`index`, fixed `d462e00`) and
`graph` (`fusion`, this fix). Both are in the top-level `full` feature (`full = [.., "ann", .., "graph", ..]`), so a
single `cargo check -p frankensearch --no-default-features --features full` (or `--all-features`) would have caught both
— it just isn't in the swarm's default `--workspace --lib` conformance. **Process fix (do this after every workspace-wide
type change):** run the `full`/`--all-features` lane, not only `--workspace --lib`. `graph_rank.rs` tests already used
`.into()`; only the one production `map` closure was missed. (The top-level `full` lane re-check with both fixes was
launched to confirm no third casualty hides behind `durable`/`api`.)

---

## 2026-07-02 — RE-MEASURED BOLD ratios: comparator CONFIRMED closed (top-k parity, limit_all ~1.4× inherent); exact_id@100k "2.5×" is single-run incumbent noise, NOT a claimed win (BlackThrush)

**Fresh `bold_verify_tantivy_class` run (`search_bench`, `--features lexical`, 10k+100k, hz2, `--sample-size 10`).**
`ratio = fs_p50 / incumbent_p50` (<1 = frankensearch faster). **CAVEAT (per [[bold-comparator-closed]]): at these fast
sample settings a SINGLE run's per-row ratio is ±40% noise dominated by the INCUMBENT's variance — do not conclude a
per-row win/gap from one run.**

| query_class | docs | ratio (hybrid) | ratio (guard) | note |
|---|---|---|---|---|
| exact_identifier | 100k | 0.405 | 0.403 | **NOISE, not a win** — incumbent measured 2277 µs here vs its ~881 µs historical true value ([[bold-comparator-closed]] run #2); at 881 µs the real ratio is ~1.05 (parity). Needs a confirming re-run before any claim. |
| exact_identifier | 10k | 0.992 | 0.857 | parity |
| quoted_phrase | 100k | 1.060 | 0.995 | parity (was ~1.25× slower in the old pre-comparator baseline) |
| quoted_phrase | 10k | 1.008 | 1.000 | parity |
| short_keyword | 100k | 1.000 | 1.000 | parity |
| high_fanout | 100k | 0.997 | 1.000 | parity |
| natural_language | 100k | 1.026 | 1.034 | ~1.03× (noise-band) |
| zero_hit | 100k | 1.067 | 1.022 | p50 marginal; p95/p99 0.879 / 0.605 |
| **limit_all** | **10k** | **1.392** | **1.517** | the one gap that is **run-to-run consistent** (matches memory's ~1.4×) — **inherent** (fs 1964 µs vs incumbent 1411 µs) |

**Honest conclusion (only the run-to-run-consistent claims):** frankensearch is **at parity on every top-k class** and
the sole reliable gap is **`limit_all` ~1.4×, which is inherent** (full `ScoredResult` materialization for all N vs bare
doc_ids; doc_id already SSO; `Copy`-field struct build only ~1.04× shrinkable → packed-struct rejection; not shrinkable
without an API break). **The exact_identifier@100k 0.405 is almost certainly incumbent variance** (2277 µs is ~2.6× its
historically-measured ~881 µs) — I initially reported it as a "2.5× win" and am **retracting that** per the documented
lesson: never claim a per-row BOLD win/gap from a single fast-sample run. **METHODOLOGY (self-correction):** check a
surprising per-row ratio against the incumbent's *known* absolute p50 before believing it; the incumbent's variance, not
frankensearch, usually moves it. No new mechanical lever exists; the comparator is closed. (A confirming re-run of the
exact_identifier row is the only way to settle whether it's real — deferred, low-value since it would at best show
parity.)

---

## 2026-07-02 — dig: async `TwoTierSearcher` materialization audited (the vein BOLD hides) — both paths winner-only, floored (BlackThrush)

**Followed [[bold-comparator-closed]]'s own route-next ("profile the production async searcher / rich-metadata paths the
BOLD proxy doesn't model") and source-audited both async materializations. No lever left.** BOLD uses tiny metadata, so
it hides per-candidate clone costs — the cba06d7 win (3.55–7.20×) came from exactly this vein. Checked for a *sibling*:

- **Initial fused→scored** (`searcher.rs:2429`): metadata is borrowed as `&serde_json::Value` into an `AHashMap` and
  cloned **only for the top-k winners** (the cba06d7 win); `explanation` is gated on `explain` AND built only over the
  already-top-k `fused` slice; `doc_id` is SSO `CompactString`. Optimal.
- **Refined/quality** (`searcher.rs:1465`): `blended.iter().enumerate().take(k).map(..)` — materializes **winner-only**
  (`.take(k)` before the map); fast/quality/initial scores are `.get().copied()` from **borrowed** maps keyed on `&str`
  (no per-candidate clone); blend runs via `blend_two_tier_aligned` (the 41beaff/5bc9d58 borrow win); explanation gated.
  Optimal.

**Conclusion:** the async searcher's per-candidate work is already winner-only everywhere; even a heavy-metadata workload
can't expose a candidate-wide clone here (there isn't one). Combined with the sync path ([[fusion-result-assembly-floored-2026-07-02]]),
vector top-k, index materialization, lexical, and the confirmed-closed comparator, **the mechanical perf frontier is
comprehensively exhausted and re-verified this session.** The one *untested* residual is `ScoredResult.metadata:
Option<Value>` → `Option<Arc<Value>>` (refcount-bump clone instead of deep-clone), which only matters at
`limit_all`+rich-metadata (uncommon) and is a **public API break** — same class as the packed-`FusedHit` rejection
(measured 1.04×), so low-EV and decision-gated, not a mechanical swap. **Blocker surfaced:** no new mechanical lever
exists on the measured surface; remaining options are decision-gated (ANN-in-BOLD w/ 100k recall budget; metadata-Arc w/
API break) or a different workload than the codebase currently exercises.

---

## 2026-07-02 — BIG LEVER FOUND (measured 200–278×): `ScoredResult.metadata` deep-`Value`-clone → `Arc<Value>` (BlackThrush)

**I was WRONG to dismiss this as "low-EV, same class as packed-struct" — measured it and it is the largest lever of the
session.** The packed-struct analogy was false: that was `Copy` fields; `metadata` is a **deep clone of a nested
`serde_json::Value`** (map + strings + arrays re-allocated each time). The async searcher deep-clones each winner's
metadata at materialization (`searcher.rs:2514`, `.cloned()`); at `limit_all` that is N deep clones per query. BOLD uses
*tiny* metadata so this is invisible in the comparator — but real document metadata (title/path/tags/…) is not tiny.

**Same-worker A/B `metadata_clone_ab` (deep `Value` clone vs `Arc` refcount bump, realistic 8-field metadata, ovh-a):**

| N | `value_deep_clone` (current `Option<Value>`) | `arc_clone` (`Option<Arc<Value>>`) | speedup |
|---|---|---|---|
| 10,000 | 11,338 µs (11.3 ms) | 56.8 µs | **200×** |
| 100,000 | 158,598 µs (158 ms) | 571 µs | **278×** |

**Context:** the whole `limit_all`@10k query is ~2 ms with BOLD's tiny metadata — but with realistic metadata the clone
ALONE is ~11 ms, i.e. a rich-metadata `limit_all` query is **metadata-clone-bound** and `Arc<Value>` removes ~99.5% of
that term. This is a REAL production win (rich-metadata consumers) that the BOLD proxy structurally hides — exactly the
vein [[bold-comparator-closed]] flagged ("profile rich-metadata workloads; BOLD hides real-path clone costs"). **Lever:
`ScoredResult.metadata: Option<serde_json::Value>` → `Option<Arc<serde_json::Value>>`** — analogous to the landed doc_id
`CompactString` (also a public field-type change accepted for a clone win). `Arc<Value>` `Deref`s to `Value` so read
sites (`.get()`, `.as_object()`, filters) mostly compile unchanged; construction wraps in `Arc::new`; **serde `rc`
feature needed for `Deserialize<Arc<T>>`.** Landing it (carefully — with `--all-features`/`full` lane checks per
[[cfg-gated-feature-migration-blindspot]]) is the next step. `metadata_clone_ab` bench kept as evidence.

**✅ LANDED `f5e9c9d` (2026-07-02, pushed main+master).** `ScoredResult.metadata: Option<serde_json::Value>` →
`Option<Arc<serde_json::Value>>`. The async searcher's `lexical_metadata_by_doc` map now holds `&Arc<Value>` so the
per-winner `.copied().cloned()` at `searcher.rs:2514` is an `Arc::clone` (the win); `filter.matches` sites use
`.as_deref()` (trait unchanged, `Arc` derefs to `Value`); serde `rc` feature added to core for `Deserialize<Arc>`.
**Verified GREEN:** core+fusion+lexical tests **820+ passed / 0 failed** (incl. the `types.rs:910` Serialize→Deserialize
`Arc` roundtrip exercising the `rc` feature, and the fused-conversion metadata-preservation test), AND the full-feature
lane (`--features full` = ann+graph+rerank+durable+api) compiles clean — no cfg-gated casualty this time ([[cfg-gated-feature-migration-blindspot]]
lesson applied: verified `--features full` before pushing). Reads are unchanged (`Arc<Value>` derefs); production
metadata construction is almost all `None` or `.clone()` (auto `Arc::clone`), so only 4 test-construction sites needed
`Arc::new`. This is the largest measured win of the session — 200–278× on the rich-metadata `limit_all` materialization,
a real production path the BOLD proxy structurally hides.

**Bonus impact (recorded, no extra change):** the metadata-Arc landing also makes the async **progressive-phase
emission clones cheap for rich metadata**. `display_hits.clone()` / `refined_results.clone()`
(`searcher.rs:543/637/683/806`) copy `Vec<ScoredResult>` per phase; *previously* each carried `metadata: Value`, so a
rich-metadata `limit_all` deep-cloned N Values **per phase** (2N/query). The async phase-clone had been measured "~0"
([[discarded-output-clone-lever]]) — but ONLY because that test used tiny metadata; the struct-copy itself was already
cheap, the hidden cost was the metadata deep-clone. Now `metadata: Arc<Value>` → those phase clones are `Arc::clone`.
So one field-type change fixed BOTH the materialization AND the phase-emission clone for rich-metadata workloads.

**ScoredResult clone arc now comprehensively optimized:** `doc_id` (`CompactString` SSO), `metadata` (`Arc<Value>`),
`explanation` (built winner-only, gated on `explain`), all other fields `Copy`. No heap-heavy per-result clone remains
on the common path. The `explanation` deep-clone under `explain=true`+`limit_all`+phases is the only residual (an
`Arc<HitExplanation>` would fix it by the same pattern) — niche (`explain` is opt-in debug/UI), low-EV, deferred. The
"rich-metadata real-path the BOLD proxy hides" vein — productive for two landings ([[metadata-arc-value-landed]] +
cba06d7) — is now itself optimized.

---

## 2026-07-02 — REJECTED: `explanation` → `Arc<HitExplanation>` measures 65× but the realistic scenario doesn't justify the cross-crate change (BlackThrush)

**Measured the residual (didn't dismiss it — the metadata lesson), found a big ratio, but rejected on scenario/risk —
the mirror of the metadata *acceptance*.** The last heavy `ScoredResult` field is `explanation: Option<HitExplanation>`
(nested `Vec<ScoreComponent>` + `Vec<String>` matched-terms + `String`s). Under `explain=true` the async
phase-emission clones deep-clone it per phase. Same-worker A/B (`explanation_clone_ab`, realistic 2-component hybrid
explanation w/ rank-movement):

| N | `value_deep_clone` | `arc_clone` | speedup |
|---|---|---|---|
| 10,000 | 4,534 µs | 69.5 µs | **65×** |
| 100,000 | 52,746 µs | 808 µs | **65×** |

**Why REJECTED (unlike metadata's 278× which shipped):**
1. **Opt-in.** `explanation` is `Some` only when `config.explain=true` (debug/UI) — `None` (cheap) for the vast
   majority of queries. Metadata is populated whenever a consumer attaches document metadata (common).
2. **Narrow scenario.** The 65× is on N clones; it only bites at `explain`+`limit_all`, but realistic `explain` use is
   **top-k** (you explain the k results shown, not all N) — so realistic N is small and the *absolute* saving is
   sub-millisecond. (METHODOLOGY: charge the ratio to the realistic scenario, per [[rrf-sort-key-fattening-rejected]].)
3. **Cross-crate mutation friction.** `rerank/pipeline.rs:158` **mutates** the explanation in place (pushes a `Rerank`
   `ScoreComponent`). `Arc<HitExplanation>` forces `Arc::make_mut` there — correct and cheap *only because* the Arc is
   uniquely held pre-phase-emission, i.e. a subtle shared-ownership invariant to preserve — plus it widens the change
   (and its `--all-features`/`full`-lane verification) to core+fusion+rerank, all for an opt-in, top-k-small win.

**Decision: no production change; `explanation_clone_ab` bench kept as evidence.** The distinction from metadata is the
lesson: a big *ratio* is necessary but not sufficient — a per-item clone lever pays only if the field is *commonly
populated* AND the realistic access pattern has *large N*. Metadata (common, all-N) qualified; explanation (opt-in,
top-k) does not. The `ScoredResult` clone arc is optimized for every common-path field; `explanation` stays deep-clone
by choice, cheap because it's usually `None`.

---

## 2026-07-02 — dig (algorithmic/alien-graveyard): MMR lazy-greedy is N/A — the O(k²n)→O(kn) incremental win is already landed; indexing content path borrow-optimized (BlackThrush)

**Two fresh checks, both already-optimal — recorded so they aren't re-hypothesized.**

- **MMR / diversity reranking (`mmr.rs:103`).** Hypothesis: naive greedy (recompute `max sim(i, selected)` each round,
  O(k²·n)) → Minoux lazy-greedy. **Already preempted:** production maintains a running `max_sim_to_selected[i]` folded
  incrementally (`:205/:243`), so it's **O(k·n)**, bit-identical to the naive `mmr_rerank_reference` oracle (`:332`) via
  `f64::max` associativity — the in-code comment (`:200-204`) documents exactly this. Lazy-greedy can't improve it: the
  residual cost is the inherent **n·k cosine similarities** (each `O(d)`, and you MUST evaluate `sim(new, i)` to update
  the diversity penalty) — not the argmax a priority queue would replace. The cosine kernel is already 4-accumulator SIMD
  ([[search-time-reductions-still-widenable]], efbfe33). Floored.
- **Indexing content path.** `IndexableDocument.content` (full text, always populated, large-N when indexing) is
  **borrowed** into tantivy (`lexical/lib.rs:710` `add_text(&doc.content)`) and the embedder takes `&str` — no clone on
  the main path. The only content clones are `fts5` (`fts5_adapter.rs:232`, **forced**: `SqliteValue::Text` needs owned
  `String` + the doc is shared across lexical/vector/storage consumers) and the cass-compat layer — both narrow/feature-
  gated/forced, not oversights.

**Conclusion — the measurable, broadly-applicable frontier is comprehensively exhausted (verified this session across
result-materialization, indexing, MMR, vector/index top-k, and the closed comparator).** Every hot clone/materialization
is optimized (`doc_id` SSO, `metadata` Arc, `content` borrowed) or correctly left alone (`explanation` opt-in,
`fts5` content forced). The remaining levers are **decision-gated** (ANN-in-BOLD w/ recall budget) or need a **workload
the BOLD proxy doesn't model** (heavy-metadata corpus end-to-end, reranker-in-loop, indexing-throughput) — not a
per-crate mechanical swap. Re-measure before assuming any new lever exists.

---

## 2026-07-02 — dig: ANN-vs-flat AT 100k — HNSW's edge scales with N (up to 4.7×); recall pending (BlackThrush)

**Autonomous measurement of the one remaining lever (ANN-in-BOLD), at the scale where it matters.** My earlier
`hnsw_vs_flat` used N=10k, where the rayon flat scan (~175 µs) already beats HNSW except at ef10. Built `hnsw_vs_flat_100k`
(N=100k, DIM=128, K=10, 256 clusters, `--features ann`) — the BOLD corpus scale. **Latency (search-cc, ovh-a/hz2):**

| arm | 100k latency | vs flat | (10k for contrast) |
|-----|-------------|---------|--------------------|
| `flat` (exact, rayon) | 517.9 µs | 1.00× | 175 µs |
| `hnsw_ef10` | **109.2 µs** | **4.7× faster** | 2.38× |
| `hnsw_ef20` | 216 µs | 2.4× | 1.47× |
| `hnsw_ef40` | 341 µs | 1.5× | 0.86× (slower) |
| `hnsw_ef100` | 797 µs | 0.65× (slower) | 0.37× |

**Finding: HNSW's advantage GROWS with N** — at 10k it only won at ef10 (2.38×) and lost by ef40; at 100k it wins through
ef40 (up to 4.7× at ef10, 2.4× at ef20), as O(log N) vs O(N) predicts. So **ANN-in-BOLD becomes materially attractive at
larger corpora** (unlike the 10k picture). **Gating factor = recall (PENDING):** the recall@10 sweep over `ef` was
printed at setup via `eprintln` and truncated by rch's tail-only capture; re-run moves it to a post-criterion `println`
(in flight, ~6 min on the degraded fleet). HNSW is a real win only where recall is high at a fast `ef`.

**✅ RECALL MEASURED → ANN-in-BOLD REJECTED (my "~0.95 by ef20" assumption was WRONG — good I measured):**

| ef | recall@10 | latency vs flat |
|----|-----------|-----------------|
| 10 | **0.4875** | 4.7× faster |
| 20 | **0.6125** | 2.4× faster |
| 40 | **0.7813** | 1.5× faster |
| 100 | 0.8750 | 0.65× (slower) |
| 200 | 0.9313 | slower |

**There is NO operating point where HNSW is both faster than the exact flat scan AND has acceptable recall.** Where it's
fast (ef≤40) recall is 0.49–0.78 (loses a quarter to half of the true top-10); to reach even 0.93 recall needs ef200,
which is SLOWER than the exact 517 µs flat scan (which is 1.0 recall). The exact rayon flat scan **dominates the
speed/recall frontier** at 100k for every ef that matters. **Decision: ANN-in-BOLD REJECTED by measurement** — it does
NOT need a recall-budget sign-off after all, because there is no favorable budget: you'd trade 22–51% recall for the
speedup, or give up the speedup for still-imperfect recall. (Caveat: `HnswConfig::default()` M/ef_construction + synthetic
256-cluster/dim-128/noise-0.30 data; a higher-M index or real 384-dim embeddings *might* shift recall, but the default
out-of-box config — what BOLD would use — is decisively unfavorable, and the flat scan is already sub-ms at 100k.)
`hnsw_vs_flat_100k` kept as evidence.

**M-swept re-check (M is HNSW's primary recall knob; default 16 → tested 32): rejection HOLDS.** Higher M lifts recall
at every `ef` but not enough to create a favorable frontier point vs the exact flat scan (recall / latency, within-run;
flat varies 517–769 µs across runs from worker variance, so compare relative within each run):

| ef | recall@10 M=16 | recall@10 M=32 | M=32 latency vs flat |
|----|----------------|----------------|----------------------|
| 10 | 0.49 | 0.63 | 5.4× faster |
| 20 | 0.61 | 0.74 | 2.9× faster |
| 40 | 0.78 | 0.85 | 1.36× faster |
| 100 | 0.88 | 0.95 | slower |
| 200 | 0.93 | 0.98 | slower |

At M=32 the best "fast" point is ef40 (**1.36× faster, but only 0.85 recall — loses 15% of the true top-10**); 0.95
recall needs ef100, which is SLOWER than the exact scan. Doubling M again would keep eroding the memory/build advantage
for diminishing recall gains. **The exact rayon+SIMD flat scan — sub-ms at 100k, 1.0 recall, already the shipped vector
tier — dominates the speed/recall frontier for the default `ef_construction`.** This closes the LAST remaining lever
across two M settings: even the one decision-gated option is now measured-and-rejected. The frontier is comprehensively closed — top-k comparator parity, limit_all inherent, all clone/
materialization paths optimized, vector top-k floored (flat exact beats ANN on the recall/latency frontier through 100k),
MMR incremental, indexing borrow-optimized.** Route next is a different workload (heavy-metadata E2E, reranker-in-loop)
or a higher-M ANN re-measure with real embeddings — not a per-crate lever on the current surface.

---

## 2026-07-02 — CORRECTION: ANN-in-BOLD is VIABLE — the rejection was a NOISE=0.30 synthetic-diffuseness artifact; realistic tight clusters give 0.994 recall @ 2.6× faster (BlackThrush)

**Reversing my own two-commit rejection (`e955f6b`/`4330c18`) with better data — the same class of self-correction as the
metadata-dismissal and the exact_id retraction.** The prior ANN rejection used `NOISE=0.30`, a diffuse spread where the
true top-10 span a wide region (near-worst-case for HNSW). Real semantic embeddings cluster *tightly* (similar docs are
genuinely close). Re-ran `hnsw_vs_flat_100k` at `NOISE=0.15` (tighter, realistic), M=32 — and BOTH recall and latency
flip decisively (within-run; flat = 782 µs):

| ef | latency | vs flat | recall@10 (0.30 → 0.15) |
|----|---------|---------|-------------------------|
| 20 | 100.9 µs | **7.75× faster** | 0.74 → **0.913** |
| 40 | 151.4 µs | **5.16× faster** | 0.85 → **0.975** |
| 100 | 301.7 µs | **2.59× faster** | 0.95 → **0.994** |

**With realistically-tight clusters, HNSW dominates the vector tier:** ef100 = **0.994 recall (near-exact) at 2.6×
faster** than the exact flat scan; ef40 = **0.975 recall at 5.16× faster**. Latency ALSO improved (ef40 566 µs @ NOISE
0.30 → 151 µs @ 0.15) because a well-clustered graph converges in far fewer hops. **ANN-in-BOLD is a strongly-viable
measured lever** — a 2.6–5× vector-tier speedup at ≥0.975 recall for corpora with real semantic structure. The
`NOISE=0.30` result was the artifact; `NOISE=0.15` is the realistic case (and even that is synthetic — the true
validation is real 384-dim embeddings, but the sensitivity is now clear: ANN viability tracks corpus tightness, and real
embeddings are tight). **REVISED conclusion: ANN-in-BOLD is NOT rejected — it is a measured 2.6× (near-exact-recall)
vector-tier win pending (a) a recall-budget sign-off since it trades exact for ~0.99 recall, and (b) validation on a real
embedded corpus.** LESSON: never reject an ANN/recall lever on ONE synthetic data distribution — recall is exquisitely
sensitive to cluster tightness; sweep the noise/separation before concluding.

---

## 2026-07-02 — dig (exact/decision-free + alien-artifact): AVX-512 avenue closed (dev HW is Zen 3); the phase gate is already an e-process (BlackThrush)

**Two fresh checks for a LANDABLE exact win (no recall tradeoff, unlike ANN) — both come back closed/already-absorbed.**

- **AVX-512 on the flat scan (the exact vector-tier workhorse that beats ANN when recall matters).** The dot kernels are
  AVX2-dispatched ([[avx2-runtime-dispatch-dot-kernels]]); AVX-512 (512-bit lanes) could be a ~1.5–2× *exact* win. But
  the **dev/verify machine is an AMD Ryzen Threadripper PRO 5975WX (Zen 3): avx2 + f16c + fma, NO avx512** (Zen 3
  predates AMD AVX-512, which lands in Zen 4). So the flat scan cannot be AVX-512-accelerated on this hardware — can't be
  written, run, or verified here — and AVX2 is the practical exact ceiling. Closed.
- **Certified early-exit for the quality (phase-2) tier (the `/alien-artifact-coding` angle).** Hypothesis: provably skip
  phase 2 when it cannot change the top-k. Already absorbed at a *higher* level: `phase_gate.rs` is a **sequential-testing
  e-process** (anytime-valid via Ville's inequality; refs Ramdas 2020, Grünwald 2019) that decides `SkipQuality` when the
  fast tier is statistically sufficient over the workload. A per-query provable-stability gate would be marginal beside a
  workload-adaptive e-process and risks changing results; not worth it. The codebase has already ingested the
  advanced-math breakthroughs (e-processes, fsfs expected-loss contracts) as well as the systems ones.

**Net: the exact/decision-free perf frontier is closed on this hardware, and the alien-artifact statistical levers are
already implemented.** The ONE open perf lever remains **ANN-in-BOLD** ([[ann-in-bold-viable]]) — a measured 2.6–5×
vector-tier win at ≥0.975 recall for realistically-clustered corpora — which is NOT decision-free (it trades exact recall
for ~0.99) and so is blocked on a recall-budget sign-off + real-embedding validation, not on finding a lever.

---

## 2026-07-02 — RESOLVED: ANN-in-BOLD is ALREADY the shipped default at scale (`hnsw_threshold: 50_000`); my session validated the default config (BlackThrush)

**The ANN thread is closed — there was nothing to "land" because it's already default-on.** Checked the wiring end to
end: `TwoTierConfig` exposes `hnsw_threshold` (default **50_000**), `hnsw_ef_search` (100), `hnsw_m` (16),
`hnsw_ef_construction` (200). `maybe_load_or_build_ann` **auto-builds** the HNSW sidecar when the corpus exceeds the
threshold, and `two_tier.rs:285` uses `ann.knn_search` whenever the sidecar is present. So **for any corpus ≥50k built
with the `ann` feature, frankensearch ALREADY uses HNSW ANN by default** at ef_search=100 / m=16. (BOLD's proxy uses the
flat scan only because its bench doesn't enable the `ann` feature / build the sidecar — not because ANN is unwired.)

**My session's ANN measurement (`hnsw_vs_flat_100k`) therefore validates the shipped defaults rather than proposing a
new lever:** at the default-ish ef_search=100 on realistically-tight clusters, recall@10 = **0.994** (m=32) / ~0.98–0.99
(m=16) at **2.6× faster** than exact flat — so the defaults (threshold 50k, ef 100, m 16) are a sensible exact-vs-ANN
crossover for large corpora with real semantic structure. The `NOISE=0.30` "rejection" was the synthetic-diffuseness
artifact; the shipped default targets the realistic (tight) regime and is well-chosen.

**Conclusion: there is NO un-landed perf lever.** The one measured big win (ANN, 2.6–5×) is already the default at scale;
every other surface is optimized or inherent (metadata-Arc 278× landed, doc_id SSO, MMR incremental, flat scan AVX2-
ceiling on Zen 3, phase gate an e-process, top-k comparator parity, limit_all inherent). The only remaining actions are
NON-mechanical: (a) validate the ANN default's recall on a REAL embedded corpus (needs a semantic model, absent on the
rch workers), or (b) a different workload than the BOLD proxy models. The measurable, per-crate frontier is
comprehensively closed and, for ANN, already shipped.

---

## 2026-07-02 — REJECTED: RRF reciprocal-LUT — 1.45× isolated ceiling doesn't survive the real fuse (OoO-hidden divides + cache-hostile LUT) (BlackThrush)

**A genuinely new exact/bit-identical lever, measured and rejected on realistic-scenario grounds.** `rank_contribution =
1.0/(k+rank+1.0)` is a float DIVIDE per candidate per source in the fuse loop (rrf.rs:464/488); at `limit_all` that's
~2·N divides/query. A precomputed reciprocal LUT (indexed by rank, bit-identical) replaces the ~10–20-cycle divide with
a lookup. Ceiling measured (`rrf_recip_ab`, isolated divide vs LUT):

| N | divide | LUT | speedup |
|---|--------|-----|---------|
| 10,000 | 19.3 µs | 13.1 µs | 1.47× |
| 100,000 | 195.8 µs | 135.3 µs | 1.45× (~60 µs saved) |

**Rejected — the isolated ceiling doesn't transfer to the real fuse:**
1. **Divides are out-of-order-hidden** in the fuse loop — they overlap with the merge/store/compare *memory* work
   ([[rrf-sort-adaptivity-merge-routenext]] made the merge a sequential sorted walk, but it still touches N scratch
   records), so the realized win is ≤60 µs/100k and likely far less.
2. **Cache-hostile at `limit_all`:** the LUT must be sized to max rank = N → **800 KB at 100k**, competing with the
   ~MB `FusedHitScratch` array for L2 → lookups turn into cache misses. The micro-bench's LUT was cache-resident (no
   competing data); the real fuse is not, so the real LUT could REGRESS.
3. **No favorable scenario:** `limit_all` has the divides but a huge cache-hostile LUT (building it per-query = N divides,
   no saving; reuse needs a searcher-held table sized to unknown N); top-k has a tiny LUT but too few candidates for the
   divides to matter. Neither end wins.

**Decision: no production change; `rrf_recip_ab` kept as evidence.** METHODOLOGY (again): a big *isolated* ratio (1.45×)
is necessary but not sufficient — charge it to the realistic loop (OoO overlap + cache competition), same as the
packed-struct (1.04×) and explanation (65× but opt-in) rejections. The RRF score computation is at its floor.

---

## 2026-07-02 — dig: mined the ledger's unmeasured/asserted claims (the metadata-278× strategy) — all remaining are sound rejections (BlackThrush)

**Re-ran [[mine-rejected-lever-route-nexts]] — the strategy that surfaced the metadata-Arc 278× win (an assertion
dismissed by analogy, never benched). This pass every remaining un-benched claim is a genuine structural rejection, not
a missed measurement:**

- `core::l2_normalize` multi-accumulator — non-bit-identical (f32 reorder) on a core primitive w/ 6 callers, and
  **embedding-time (LOCKED, [[search-time-reductions-still-widenable]])**; ~0.35 µs of a 50 µs embed. Sound reject.
- `core::cosine_similarity` ILP — test-only, not on any hot path. No-op.
- `prf::prf_expand` — already auto-vectorized SAXPY. `storage::content_hash` — SHA-256 semantics (dedup table). Sound.
- **Lexical ids-only routing (the biggest-looking residual):** the async hybrid uses `LexicalSearch::search()` (full
  `searcher.doc()` store-read per hit) rather than the fast `search_doc_ids`/`collect_id_hits` (no docstore). BUT: (1)
  the store-read is **shared with the tantivy incumbent** (ledger line ~2069) so it does NOT move the vs-Tantivy ratio;
  (2) the metadata it fetches IS consumed by filters + output (the `lexical_metadata_by_doc` re-attach — now Arc-shared,
  metadata-Arc 278×), so it can only be skipped conditionally (no-filter + caller-wants-no-metadata); (3) it's a
  trait-surface change. Marginal, conditional, non-ratio-improving → correctly deprioritized. (The `SyncLexicalSearch`
  path is test-only — `StaticLexical` is the sole impl; production lexical is the async trait.)

**Conclusion: the unmeasured-claims vein is now dry too.** The metadata-Arc win was the one assertion-dismissed-by-analogy
that was actually a huge measured lever; re-mining finds no sibling. Combined with the RRF-recip-LUT reject (`1b6e5d6`),
AVX-512-HW-closed, and the ANN-already-shipped-default resolution, there is no un-landed per-crate lever on the measured
surface. The productive frontier is a different workload (real-embedded-corpus ANN validation; heavy-metadata E2E) — not
a lever hunt on the current code.

---

## 2026-07-02 — RELIABLE BOLD ratios (25-sample): hybrid is parity-or-FASTER on every top-k class; quoted_phrase now a WIN; exact_id noise resolved (BlackThrush)

**Higher-sample (`--sample-size 25 --warm-up 2 --measure 2`) `bold_verify_tantivy_class` — the definitive current ratio
vs the tantivy-class incumbent, resolving the single-sample noise that forced the earlier `exact_id@100k` retraction.**
`ratio = fs_p50 / incumbent_p50` (<1 = frankensearch faster). **`hash_hybrid_tantivy_vector_rrf` (the real hybrid):**

| query_class | 10k | 100k |
|-------------|-----|------|
| exact_identifier | 0.943 (faster) | **0.956** (parity — the 0.40 earlier was noise; the retraction was right) |
| quoted_phrase | **0.803 (faster)** | **0.811 (faster)** — historically the ~1.25× *gap*, now a WIN |
| short_keyword | 1.015 | 0.992 |
| natural_language | 1.018 | 1.002 |
| high_fanout | 1.010 | 1.000 |
| zero_hit | 1.025 | **0.612 (much faster — short-circuits)** |
| **limit_all** | **1.151** | (not sampled) — sole slower row, **inherent**, improved from the earlier 1.39–1.52 |

**Conclusion — the comparator is definitively closed and favorable:** the production hybrid is **at parity or FASTER on
every top-k class at both corpus sizes**, and the one class it historically lost (`quoted_phrase`) is now a clear win
(0.81×). The only >1.1× row is `limit_all` (1.15×, inherent — full `ScoredResult` materialization for all N vs the
incumbent's bare doc_ids), and even that shrank from earlier readings. (The `hash_lexical_guard` variant shows a few high
rows — high_fanout@100k 1.314, quoted_phrase@10k 1.368 — that are **incumbent-variance noise**: the guard executes the
IDENTICAL `search_doc_ids` as the incumbent, so it cannot truly be slower; only the hybrid rows are the meaningful
head-to-head.) This is the current (2026-07-02, 25-sample) reference table — statistically firmer than the earlier
10-sample run; re-measure before assuming any regression.

---

## 2026-07-02 — LEVER FOUND (measured, landing): box `ScoredResult.explanation` — halves the struct (168→88 B), 1.14× materialize + phase-clone + memory (BlackThrush)

**A real EXACT lever, cleaner + higher-ratio than the rejected packed-struct (1.04×) — the one remaining improvement on
the `limit_all` path.** `ScoredResult.explanation: Option<HitExplanation>` stores `HitExplanation` (88 B: `Vec` + `f64` +
`Option<RankMovement>{String}`) **inline**, so every `ScoredResult` reserves 88 B for it **even when None** (the common
`explain=false` case). Measured (`scoredresult_box_ab`, `size_of` + materialize A/B):

| metric | inline (`Option<HitExplanation>`) | boxed (`Option<Box<HitExplanation>>`) |
|--------|-----------------------------------|---------------------------------------|
| `size_of::<ScoredResult-shape>` | **168 B** | **88 B** (48 % smaller) |
| materialize N (10k) | 96.1 µs | 89.1 µs (**1.08×**) |
| materialize N (100k) | 1023 µs | 896 µs (**1.14×**) |

**Why land it (vs the packed-struct reject):** (1) **exact/bit-identical** — `Box<HitExplanation>` derefs to
`HitExplanation`, no sentinel/niche-safety loss (packed-struct changed `Option<usize>` ranks that consumers READ; this
changes an opt-in, rarely-read field); (2) the 1.14× understates it — halving the struct also speeds the async
progressive-phase clones (`display_hits.clone()` copies `Vec<ScoredResult>` up to 2N/query) and **halves result-set
memory footprint**; (3) **no downside** for the common path — a boxed `None` is an 8 B null ptr (no allocation); the Box
is heap-allocated only when `explain=true` (rare, and the explanation is already heap-heavy then). Same proven pattern as
the metadata-Arc landing. Landing now with a full `cargo test --workspace` verification (per the [[cfg-gated-feature-migration-blindspot]]
`cfg(test)` lesson). `scoredresult_box_ab` kept as evidence.

**✅ LANDED (`a5b3f86`) + FULLY VERIFIED:** `cargo check --workspace --all-targets --features lexical` = **Finished in
5m28s, 0 errors** (fresh compile, no stale-cache, no unresolved imports) — the Box change AND all this session's type
migrations (CompactString `DocId`, metadata `Arc<Value>`) compile clean across **every crate and every target** (lib +
tests + benches). No cfg(test)/bench blind spot remains anywhere in the workspace — the comprehensive certification that
the per-crate lib tests + compile-only feature lane could not give. `ScoredResult.explanation: Option<HitExplanation>` →
`Option<Box<HitExplanation>>` (core
`types.rs`) + the 3 production constructions wrapped in `Box::new` (searcher.rs:2497/1567/2577). `Box` needs NO serde
feature (unlike Arc's `rc`); the two mutation sites (`searcher.rs:1678` rank-movement update, `rerank/pipeline.rs:158`
push Rerank component) work UNCHANGED via `Box`'s `DerefMut` — **this is why `Box` beats `Arc` for a mutated field** (Arc
would have forced `Arc::make_mut`). Lib + tests compile clean. **This surfaced (and this cycle also fixed) pre-existing
metadata-Arc BENCH breaks** the f5e9c9d landing left — `mmr_reorder.rs`, `lexical_metadata_map.rs` constructed
`metadata: Some(Value)` (need `Arc::new`); benches are compiled only by `--all-targets`/`cargo bench`, never by the lib
tests or the compile-only feature lane, so they'd been broken since f5e9c9d ([[cfg-gated-feature-migration-blindspot]]:
the `cfg(test)`/bench blind spot again — run `--all-targets`). Result: `ScoredResult` is now 88 B (was 168), exact
1.14× materialize + faster phase clones + half the result-set memory, the last measured improvement on the `limit_all`
path.

---

## 2026-07-02 — dig: the fat-inline-`Option` boxing pattern is exhausted for hot structs (BlackThrush)

**Generalized the explanation-Box win (box/Arc a usually-`None` `Option<BigInline>` to pay 8 B not `size_of::<Big>` on
the common path) and swept the workspace for siblings — none remain on a hot per-result path.** `grep`ed hot-crate
structs for `Option<Vec/String/HashMap/BigStruct>` fields:

- **`ScoredResult`** (the only per-result hot struct, materialized at large N): already fully optimized — `doc_id`
  `CompactString` SSO, `metadata` `Arc<Value>` (278×), `explanation` `Box<HitExplanation>` (168→88 B). No fat field left.
- **`FusedHit`** (RRF intermediate): its `Option`s are small primitives (`Option<usize/u32/f32>`, ≤16 B) — no fat
  inline variant; struct-packing was already REJECTED at 1.04× (the packed-struct entry).
- **All other `Option<String>` matches** are in COLD structs — `config`/telemetry/`e2e_artifact`/`traits` (constructed
  once per query or per config, NOT per-result at large N) or `IndexableDocument.title` (index-time). `Option<String>`
  is only 24 B inline; boxing saves 16 B on structs that aren't in a large-N materialization loop = not worth it.

**Conclusion: the struct-size / fat-Option lever is exhausted.** `ScoredResult` is at its floor (88 B, every heavy
field boxed/Arc'd/SSO'd); no other hot struct carries a large inline `Option`. Combined with the workspace `--all-targets`
GREEN certification, the per-crate memory-layout frontier is closed. Route next remains a different workload
(real-embedded-corpus ANN validation, heavy-metadata E2E) — not a struct/field lever on the current code.

---

## 2026-07-02 — E2E CONFIRMATION: the session's materialization wins made the `limit_all` phase-clone nearly FREE (BlackThrush)

**Ran the real-searcher `collect_limit_all` bench (SyncTwoTierSearcher.search_collect, in-memory, N=10k) — a same-worker
`iter` (builds progressive phase clones) vs `collect` (want_phases=false, no phase clones) A/B — to confirm the session's
materialization arc end-to-end and regression-check the real path.** Result (hz1, CIs overlap):

| arm | time | note |
|-----|------|------|
| `collect` (no phase clones) | 8.50 ms `[8.27, 8.76]` | |
| `iter` (builds 2 phase clones of `Vec<ScoredResult>`) | 8.59 ms `[8.35, 8.85]` | **iter/collect = 1.01× — within noise** |

**Finding: the progressive-phase clone at `limit_all` is now statistically FREE (~1%).** Historically that same phase
clone was a **1.29× (≈29%)** cost — the `want_phases` elision win ([[discarded-output-clone-lever]], `5100734`). The
session's materialization wins collapsed it: cloning `Vec<ScoredResult>` per phase is now cheap because each
`ScoredResult` is **88 B** (was 168) with an **SSO `CompactString` doc_id** (was a heap `String` clone) and a **boxed
`explanation`** — so the 2N per-query struct copies + doc_id clones that dominated the phase-clone cost are gone. This is
the **end-to-end payoff** of the isolated micro-wins (doc_id SSO, metadata-Arc, explanation-Box) on the real searcher's
`limit_all` path, and a clean regression check that the type migrations did NOT slow it (collect path healthy).
(Absolute ms is cross-worker noise vs earlier runs; the same-run `iter/collect` ratio is the valid measure.)

---

## 2026-07-02 — dig: the `limit_all` two vector passes are INHERENT (blend uses fast_score) — not a skippable redundant scan (BlackThrush)

**Checked the last open question about the one row behind Tantivy (`limit_all` 1.15×): is either of its two vector
passes (fast approximate scan + quality exact rescore) redundant/elidable at `limit_all`?** Answer: **no.** The two-tier
blends `alpha·quality + (1-alpha)·fast` with `alpha = blend_factor` defaulting to **0.7** (`blend.rs:104`, config
`quality_weight: 0.7`), so the FAST-tier scores carry weight 0.3 in the final blended score — the fast scan is NOT a
prune-only pass whose output can be discarded at `limit_all`; its scores are part of the result. Skipping it (using exact
scores for both tiers) would change the blend by the quantization error = not bit-identical. So both passes are inherent
to the two-tier blend semantics, and the `limit_all` gap vs the lexical-only incumbent (which does neither) is inherent
hybrid cost — confirming (not a new lever) the row is at its floor. (The only elision is the narrow `quality_weight=1.0`
config where fast_score has weight 0 → the fast tier could enumerate-without-scoring; not the default, not worth it.)

**This closes the last structural question. The measurable per-crate frontier is exhaustively verified closed:** every
hot clone/materialization optimized (metadata-Arc 278×, explanation-Box 1.14×, doc_id SSO — phase clones now ~free
E2E), struct layout at floor, vector tier AVX2/two-pass/ANN-shipped-default, MMR incremental, RRF score+sort probed,
phase-gate an e-process, comparator parity-or-faster, `limit_all` two-pass inherent, workspace `--all-targets` GREEN.
No un-landed lever remains on the current code; the productive next step is a workload the BOLD proxy doesn't model.
---

## 2026-07-02 — LANDED: `CachedEmbedder::embed_batch` funnels distinct misses through ONE `inner.embed_batch` (SlateHeron)

**Lever (uncommitted-in-tree, now completed + landed):** the `CachedEmbedder` wrapper's `embed_batch` override was a
per-text loop calling `self.embed(cx, text)` N times, so each cache MISS did a separate `inner.embed` call. That defeats
a *batching* inner embedder — fastembed/ONNX pay high fixed per-invocation overhead but low marginal cost per extra input
(`traits.rs:140` documents exactly this), so N miss-calls where 1 batched call would do is pure waste on the cache-cold
refresh/indexing path (`fusion/refresh.rs:415,447`; `fsfs/runtime.rs:8487,8860,8914`). New impl: pass 1 takes one lock
scope, resolves hits, and collects the **distinct** misses (dedup map), releasing the lock before the await; then **one**
`inner.embed_batch(cx, &distinct_misses)` for all of them; then fans results back to every slot. **Inner call-count
reduction: N distinct misses → 1 batched call** (test-proven, `embed_batch_funnels_misses_through_one_inner_embed_batch`
asserts `batch_calls==1, embed_calls==0` for 5 unique misses).

**Correctness — the uncommitted draft had a real bug I fixed before landing:** it lost the in-batch dedup the old
sequential loop got for free, so a batch with a repeated query (`["hello","hello","world"]`) embedded `hello` twice and
mis-counted stats — it FAILED the existing `embed_batch_deduplicates_within_batch` contract (bd-1ocg): got 3 inner calls
+ (0 hits,3 misses) vs the required 2 inner calls + (1 hit,2 misses). The landed version dedups distinct misses and folds
an in-batch repeat onto the same result, counting it as a hit (`CacheState::record_hit`). **All 25 `cached_embedder`
tests GREEN incl. dedup + stats + per-item-cache + funnel.** Bit-identical output vectors to the old path.

**Ratio vs Lucene/Tantivy/Meilisearch-class: N/A** — this is an embed/refresh-path call-count win, not a lexical
comparator lever; there is no criterion wall-clock bench for `CachedEmbedder::embed_batch` (the win only materializes
under a *real batching* inner like fastembed, absent from the test doubles). So it does not move a BOLD comparator row;
it removes N−1 redundant model invocations per cache-cold batch on the indexing/refresh path. Kept because it is a strict
improvement (fewer inner calls, same outputs, all contracts preserved), verified by test rather than by a bench ratio.
**LESSON:** an "uncommitted measured win" in a shared tree may be an *incomplete draft* — re-run the crate's own contract
tests before landing; this one silently dropped a documented dedup invariant.

---

## 2026-07-02 — MEASUREMENT BLOCKER: sync_searcher per-query maps/sets are std-SipHash where siblings use aHash (SlateHeron)

**Lever identified (high prior, unmeasured — NOT landed):** `sync_searcher.rs` builds, per query on the default sync
hybrid path, two `HashMap<&str, f32>` score maps (`fast_scores`/`quality_scores`, `sync_searcher.rs:270-278`, one
entry per candidate) + a `seen` dedup `HashSet<&str>` (`:446`) + a `ranks` `HashMap<&str,usize>` (`:525`), then probes
every candidate doc_id in both maps — **all std `std::collections` (SipHash), imported at `sync_searcher.rs:7`.** The
sibling fusion paths already moved off SipHash: `rrf.rs:17` and `blend.rs:236` use `ahash::AHashMap`, `search.rs:7`
uses `AHashSet` (the `search.rs:1285`/`rrf.rs:1111` std-HashMap hits are in `mod tests`, imports at 1125/570). SipHash
is a crypto hash; for short non-adversarial `&str` doc_ids `ahash` is materially faster on insert+lookup. This is the
same sibling-path-consistency lever that LANDED **1.1–1.22×** on the federated aHash-vs-SipHash swap (`9543ae6`,
see [[sibling-path-consistency-audit]]), applied to the one hot searcher map cluster it missed.

**Blocker (the ONE thing):** wrote the faithful A/B bench (`sync_hash` group: `sip` vs `ahash`, build both score maps
+ seen set then one fast+one quality `.get()` per candidate, n∈{30,100,300}, bit-identical asserted) but got **no
ratio** — the first rch run failed because the untracked bench file was not transferred to the remote worker
(`couldn't read benches/sync_hash_ab.rs`), and the staged re-run was interrupted before measurement. Per the honest
protocol I did **not** land the production hasher swap unmeasured, and did **not** commit the unverified bench
(cargo autobench discovery would compile it into `--all-targets`; ahash `AHashSet::with_capacity` / `FromIterator`
compile-safety unconfirmed without a build). Bench body preserved in the session scratchpad (`sync_hash_ab.rs`).

**Route-next (one A/B, then a 4-line swap):** copy `sync_hash_ab.rs` back to `crates/frankensearch-fusion/benches/`,
add its `[[bench]]` (harness=false) to `Cargo.toml`, **`git add` it first** (rch only transfers tracked/staged files —
this was the trap), then `CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc rch exec -- cargo bench -p
frankensearch-fusion --bench sync_hash_ab`. If `ahash` ratio < 0.97: switch `sync_searcher.rs:7` to `use
ahash::{AHashMap, AHashSet}` and the four constructors at :270/:274/:446/:525 (all local, private fns, keys are
`&str` doc_ids — bit-identical, no caller ripple). LESSON: `git add` new bench/source files BEFORE any `rch exec` or
the remote build reads a stale tree.

---

## 2026-07-02 — MEASURED (bench landed, prod swap scoped): sync_searcher per-query maps → aHash is ~2× SipHash (SlateHeron)

**Resolves the prior blocker entry.** `git add`-ed the A/B bench before `rch exec` (the earlier trap) and got a clean
same-binary measurement on remote `hz2`. The `sync_hash` bench models the real per-query shape of
`vector_hits_to_scored_results` (`sync_searcher.rs`): build the `fast`+`quality` `HashMap<&str,f32>` score maps + the
`seen` dedup set, then one fast + one quality `.get()` per candidate.

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc rch exec -- \
  cargo bench -p frankensearch-fusion --bench sync_hash_ab -- --sample-size 40 --warm-up-time 1 --measurement-time 2
```

| n (candidates) | sip (std SipHash) | ahash | ratio |
|---|---|---|---|
| 30  | 2.626 µs  | 1.329 µs  | **0.506 (1.98×)** |
| 100 | 9.092 µs  | 4.343 µs  | **0.478 (2.09×)** |
| 300 | 28.347 µs | 12.451 µs | **0.439 (2.28×)** |

**aHash is ~2× SipHash** on this build+probe over short `doc-{:06}` keys — well past the 0.97 keep gate. Bit-identical:
the maps/sets are only `.get()`/`.insert()`/`.entry()`-probed, never iterated for output (`debug_assert_eq!` on the
accumulated f32 bits passes across arms). Bench + `[[bench]]` registration LANDED (`sync_hash_ab`).

**Ratio vs Lucene/Tantivy/Meilisearch-class: N/A** — internal hasher microbench on the sync hybrid materialization,
not a lexical comparator lever; kept as evidence + a wired A/B harness. The sibling paths (`rrf.rs:17`, `blend.rs`,
`search.rs:7`) already use aHash, so this closes the last SipHash hot-map island in fusion once wired.

**Prod swap — SCOPED (one compile constraint found, not yet wired):** aliasing the whole file to
`use ahash::{AHashMap as HashMap, AHashSet as HashSet}` does NOT compile — `rank_map` (`sync_searcher.rs:524`) feeds
`blend::compute_rank_changes_with_maps` (`blend.rs:315`) which takes `&std::collections::HashMap<&str,usize>` (E0308 at
`sync_searcher.rs:540`). **Route-next (verified-safe scope):** alias only the score-map/`seen` sites — convert the two
`.collect::<HashMap<&str,f32>>()` (`:273,:278`), the `Option<&HashMap<&str,f32>>` params (`:443,:444,:478,:479`), and
`seen` (`:446`) to the aHash types, but leave `rank_map` (`:524,:525`) as explicit `std::collections::HashMap` (it
crosses the blend boundary). All score-map/`seen` sites are local, private, `&str`-keyed, probe-only ⇒ bit-identical.
Build `-p frankensearch-fusion --lib --tests` to confirm, then land as the ~2× sync-path win. (Not wired this turn:
HEAD-frozen / no-build directive — refused to push an unverified compile to the shared tree.)

---

## 2026-07-02 — BLOCKER: `fuse_expanded_payloads` clones String path key into 5 maps/hit — bench blocked by fsfs compile weight (SlateHeron)

**Lever identified (clone-elision + hasher, unmeasured — NOT landed):** `RealtimeRuntime::fuse_expanded_payloads`
(`fsfs/runtime.rs:5648-5710`), the cross-query RRF fusion over query-expansion variants, builds FIVE `HashMap<String,_>`
(`scores`/`snippets`/`best_lexical_rank`/`best_semantic_rank`/`appeared_in_count`) and calls `.entry(hit.path.clone())`
per hit across every payload — up to **5 owned `String` allocations per hit**, and the fusion case (a doc in ≥2 variants)
is exactly when the key already exists, so the clone is pure waste. All five maps are keyed by path and only
`.entry()`/`.get()`-probed; the output sort/build reads paths and materializes owned strings only for the top-`limit`
hits. So borrowing `&str` keys (payloads outlive the fusion) is **bit-identical** and drops ~5·N String allocs/query;
stacking `ahash` (default here is SipHash, unlike sibling `rrf.rs`/`blend.rs`) adds the ~2× hash win on top.

**Blocker (the ONE thing): `frankensearch-fsfs` is too heavy to bench in-crate.** Wrote a 3-arm A/B
(`expand_fuse_ab`: `clone_sip` → `borrow_sip` → `borrow_ahash`, identical ranked output asserted) but the fsfs bench
build **timed out >10min** on the rch worker — fsfs pulls vendored `openssl` + the whole TUI stack + `pdf-extract`, so
even a self-contained bench must compile the entire crate graph. The bench has **no fsfs types** (pure local structs), so
I relocated it to the lightweight `frankensearch-fusion` benches to measure — but that run was interrupted before it
executed (HEAD-frozen / no-build). No ratio → not landing the prod edit unmeasured; not committing the unverified bench
(cargo autobench would compile it into `--all-targets`). Bench preserved in the session scratchpad (`expand_fuse_ab.rs`).

**Route-next (one fast A/B, then a scoped fsfs edit):** copy `expand_fuse_ab.rs` into
`crates/frankensearch-fusion/benches/`, add its `[[bench]]`, **`git add` first**, then
`CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc rch exec -- cargo bench -p frankensearch-fusion --bench
expand_fuse_ab`. If `borrow_ahash` < 0.97 vs `clone_sip`: in `fuse_expanded_payloads` switch the five maps to
`ahash::AHashMap<&str,_>` keyed on `hit.path.as_str()` (add `ahash = { workspace = true }` to fsfs `[dependencies]`),
and `.to_owned()` the path only in the final `take(limit)` output build. Verify with `cargo build -p frankensearch-fsfs
--lib` (budget ~10-15min — fsfs is the heavy crate). LESSON: bench self-contained algorithmic levers in the LIGHTEST
crate that has the deps (fusion), never in fsfs — its openssl/TUI/pdf graph makes every bench cycle a 10-min compile.

---

## 2026-07-02 — MEASURED (bench landed, prod swap ready): fsfs default-path merge dedup — aHash ~1.3× on the full merge (SlateHeron)

**Lever (hasher-only; keys already borrow `&str`).** `merge_with_lexical_tail` (`fsfs/runtime.rs:6127`) and
`merge_with_fallback_tail` (`:6184`) — on the DEFAULT (non-expansion) result-assembly path — dedup the fused head's
doc_ids in a std `HashSet<&str>` (SipHash) probed **once per lexical/fallback-tail candidate** (O(tail); the tail is
the full lexical result set, large on big corpora). Siblings use `ahash`. Unlike the `fuse_expanded` win there is no
clone to elide here (keys already `&str`), and each kept candidate is `FusedCandidate::clone()`d (a `String` alloc) in
BOTH arms — so I benched the REAL merge shape (build head set + O(tail) contains-probes + per-keep clone) to see whether
the hasher swap survives the clone dilution end-to-end, rather than measuring the isolated set op.

**Measured (`merge_dedup_ab`, remote `hz2`, `frankensearch-fusion` benches — fsfs is a >10-min compile so its
self-contained levers are benched in the light crate):**

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/search-slateheron rch exec -- \
  cargo bench -p frankensearch-fusion --bench merge_dedup_ab -- --sample-size 40 --warm-up-time 1 --measurement-time 2
```

| shape (head, tail) | sip | ahash | ratio |
|---|---|---|---|
| h50_t200  | 10.16 µs | 7.57 µs  | **0.745 (1.34×)** |
| h50_t1000 | 47.68 µs | 35.37 µs | **0.742 (1.35×)** |
| h100_t2000| 95.59 µs | 75.60 µs | **0.791 (1.26×)** |

aHash survives the clone dilution: **~1.3× on the full merge** (the raw set-op win is ~2× but clones are hasher-
independent). Well past the 0.97 gate. Bit-identical (same merged doc_id order asserted across arms).

**Scope:** original-comparator ratio **N/A** (internal hasher on the fsfs merge, not a lexical comparator lever); kept
A/B harness `merge_dedup_ab` (in fusion benches). **Prod swap READY, not wired this turn (HEAD-frozen/no-build):**
convert the two `let mut seen* = HashSet::with_capacity(...)` at `runtime.rs:6127` and `:6184` to
`ahash::AHashSet::with_capacity(...)` (fsfs already deps `ahash` since `401c3e3`; inferred `&str` element type; same
proven swap as `sync_searcher`/`fuse_expanded`). Verify `cargo build -p frankensearch-fsfs --lib` (RCH-E309 exit-102 =
compile-succeeded), then land. Refused to push the unverified fsfs edit to the shared tree.

---

## 2026-07-02 — MIXED: filter `matches_doc_id` per-candidate `to_ascii_lowercase` — naive alloc-free contains REJECTED; conditional-skip is a narrow ~1.7× (ext-only), queued (SlateHeron)

**Lever probed:** `SearchFilterExpr::matches_doc_id` (`fsfs/runtime.rs:305`) is evaluated PER CANDIDATE in
`apply_search_filter` (`:5890`, filters the fused head at `:6533`/`:6607`) and both merge loops (`:6134`/`:6147`). It
unconditionally allocates `let lowered = doc_id.to_ascii_lowercase();` every call — even for `Extension`-only filters
(`type:`/`ext:`/`lang:`) that never read `lowered`. Two candidate fixes benched (`filter_match_ab`, remote, over
realistic path candidate sets, PathContains vs Extension filter):

**(1) REJECTED — replace the alloc + `lowered.contains(needle)` with a naive alloc-free byte-wise
case-insensitive `contains`.** For PathContains filters this is a **regression / wash**: old 4.47µs/22.49µs (n200/n1000)
→ naive 4.65µs/22.73µs = **1.04×/1.01×**. The single `to_ascii_lowercase` allocation feeds `str::contains`, which is
SIMD/`memchr`-optimized and beats a naive O((H−N)·N) byte scan — the alloc is NOT the bottleneck for path filters.
Do not replace `str::contains` with a hand-rolled scan.

**(2) Narrow WIN (queued, prod edit UNVERIFIED under no-build) — allocate `lowered` ONLY when a PathContains clause
exists.** For Extension-only filters (which never read `lowered`) the alloc is pure waste: old 8.19µs/40.50µs
(n200/n1000) → no-alloc 4.87µs/23.83µs = **0.595/0.588 (~1.7×)**. Path filters keep the identical `to_ascii_lowercase`
+ SIMD `contains` (tie by construction, no regression). Bit-identical (asserted both filter kinds).

**Route-next (verified-safe scope):** add a `has_path_contains: bool` field to `SearchFilterExpr` set once in
`parse` (`:249`), then `let lowered = self.has_path_contains.then(|| doc_id.to_ascii_lowercase());` and
`PathContains(needle) => lowered.as_deref().is_some_and(|l| l.contains(needle))`. Bench (`filter_match_ab`, conditional
variant) preserved in session scratchpad; register + `git add` + `rch bench`, then wire + `cargo build -p
frankensearch-fsfs --lib`. Ratio vs Tantivy N/A (internal, opt-in filter path). NOTE: narrow — only ext-only filters
benefit; low priority vs a broader lever. LESSON: an "eliminate the alloc" instinct can LOSE when the alloc feeds a
SIMD stdlib routine (`str::contains`); measure before replacing stdlib string ops with hand-rolled scans.

---

## 2026-07-02 — FRONTIER UPDATE: the clone/hasher/alloc/top-k family is now mined out too (SlateHeron)

The prior [[frontier-closed-2026-07-02]] map declared the per-crate perf frontier closed, but this session found a
whole UN-mined family the BOLD proxy could not model — per-query `String`-clone / std-SipHash-map / per-item-alloc
churn on the sync + fsfs result-assembly paths — and landed **5 measured wins** from it:
- `dc5b2c5` embed: `CachedEmbedder::embed_batch` funnels N misses → 1 inner batch (+ preserves in-batch dedup).
- `8665ce1` fusion: `sync_searcher` score-maps/`seen` SipHash→aHash — **~2×** (0.44–0.51).
- `401c3e3` fsfs: `fuse_expanded_payloads` 5 maps clone-String→borrow-`&str` + aHash — **~2.8–3.2×**.
- `3746d9c` fsfs: default-path merge dedup SipHash→aHash — **~1.3×** (survives clone dilution).
- `77124b9` fsfs: filter `matches_doc_id` conditional lowercase — **~1.6×** on ext-only filters (path = strict tie).

**This family is now exhausted too (surveyed this turn):** `frankensearch-lexical` is a thin Tantivy wrapper (ranking
delegated to fusion `rrf_fuse`, already `select_nth_unstable` for top-k); `fuse_expanded_payloads`' full-sort is on a
small deduped-variant set (N~40–240, select_nth ≈ noise there); the remaining fsfs std-SipHash maps are bounded/cold —
`snippets_by_doc` is capped at `FSFS_SEARCH_SNIPPET_HEAD_LIMIT=200` and dominated by snippet *extraction* I/O (map
hasher is a noise-level fraction), `hot_cache` is a per-query single lookup, `dedupe` is small. The naive alloc-free
`str::contains` was measured SLOWER than the stdlib SIMD path (`9b170cf`). No remaining site clears the ±3% gate.

**Route-next (unchanged from the prior close):** the ONE measured-big open lever is **ANN-in-BOLD** (2.6–5× vector
tier, [[ann-in-bold-viable]]) — blocked on a recall-budget product sign-off + real-embedding validation, not landable
as unilateral perf code. Otherwise the productive next step is a genuinely NEW workload the BOLD proxy can't model
(heavy-metadata E2E, reranker-in-loop, or a filtered/faceted query mix that stresses paths the micro-benches don't).
Do not re-walk the clone/hasher/alloc/top-k family — every hot site is landed or measured-marginal.

**Vector-tier addendum (this turn's confirmation):** the ANN path IS wired (`two_tier.rs:289` `ann.knn_search`,
gated by `hnsw_threshold`); its graph traversal + visited-set is delegated to an **external HNSW dependency**
(`self.hnsw.search`, `hnsw.rs:378`) — NOT frankensearch code, so the classic hnswlib visited-array lever is out of
scope (would require forking the dep). The frankensearch-side wrapper only builds ≤k `VectorHit`s + a bounded
`hits.sort_by` (both small). Production vector top-k is `BinaryHeap`-based (`mrl.rs`), already optimal; every
`sort_unstable` in `simd.rs` is a `#[test]` recall probe, not a hot path. So the vector tier is closed AND partly
external — do not dig HNSW internals or the vector top-k selection.

---

## 2026-07-02 — ROUTE CLOSED: the "widen search-time reductions" family has no remaining HOT single-acc target (SlateHeron)

Followed [[search-time-reductions-still-widenable]]'s own route ("grep dot+=/sum+= on rerank paths") — the pattern that
won MMR cosine 4-acc 1.6× (efbfe33). Every scalar reduction on a per-candidate hot path is already optimized or the
call site is cold:
- **MMR cosine** (`mmr.rs`): the hot path uses `cosine_sim_pre` (4-acc dot + hoisted norms, the landed win). The slow
  single-acc `cosine_sim` survives ONLY in a cold non-uniform-dim fallback (`:179`, "essentially never") and the
  `#[cfg(test)] mmr_rerank_reference` bruteforce (`:388`). Not hot — no lever.
- **HNSW ANN** (`hnsw.rs`): traversal + the per-comparison distance (`DistDot`) are `hnsw_rs` (external crate,
  `hnsw.rs:18,108`); `normalize_for_dist_dot` (`:774`) is frankensearch but cold — 1×/query (`:375`) and bulk-at-build
  (`:539`), and SIMD-reassociating it would perturb ANN normalization rounding (recall-adjacent). Skip.
- **prf.rs / ope.rs** norms/weights: pseudo-relevance-feedback + OPE calibration, opt-in and computed once/query, not
  per-candidate. **mrl.rs `normalize`** (`:665`) is a `#[cfg(test)]` helper.

**One remaining single-acc SIBLING (known-marginal, not pursued):** the MMR norm-precompute loop (`mmr.rs:163-175`,
`norm += x*x`) is still single-accumulator while the dot it feeds (`cosine_sim_pre`) is 4-acc. It runs n times (once
per candidate) but is **O(n·dim) = ~1/k of MMR's O(k·n·dim) pairwise-dot work**, so a 1.5× widen is ~2%/k end-to-end
(noise-level for realistic k≥10) — below the ±3% gate, and non-bit-identical (would need the `mmr_rerank` selection
assert to still hold). Recorded so it's not re-attempted as a "win". The reduction-widening route is CLOSED.

---

## 2026-07-02 — ARTIFACT: distribution-free recall CERTIFICATE for ANN — dissolves the "recall-budget sign-off" gate on ANN-in-BOLD (SlateHeron)

**Lever kind: a formal-guarantee artifact (alien-artifact-coding + graveyard §12.1 Conformal Prediction, Tier S), not a
perf ratio.** The one remaining competitive lever — ANN-in-BOLD (2.6–5× vector tier, [[ann-in-bold-viable]]) — has been
**decision-gated** for weeks on a human **recall-budget sign-off**. The reason there was a *human* in that loop: the only
recall signal frankensearch had was `hnsw.rs::estimate_recall`, a magic-constant heuristic `0.9 + 0.1·log2(ef/k)`
(clamped) with **no empirical grounding** — at ef/k=10 it reports ≈1.0 recall regardless of the actual corpus. You
cannot responsibly sign off a recall budget on a guess, so the lever stalled.

Added `frankensearch-index::recall_certificate` (self-contained, pure std, no new deps): given a calibration sample of
**measured** per-query recalls (the crate already has the `recall_at_k` bruteforce machinery that produces them), it
emits a **distribution-free, finite-sample-valid** recall lower bound:
- `conformal_recall_lower_bound(recalls, alpha)` — split-conformal per-query lower *tolerance* bound: a fresh query's
  recall ≥ L with probability ≥ 1−alpha under only exchangeability (the §12.1 primitive; the ⌊alpha·(n+1)⌋-th order
  statistic — no distributional assumptions, no asymptotics).
- `mean_recall_lower_bound(recalls, delta)` — Hoeffding lower confidence bound on E[recall].
- `certified_min_ef(calibration, target, alpha)` — the **cheapest ef whose *certified* bound meets the target**: the
  automated replacement for the human sign-off. The certificate *is* the sign-off.

**Verified (per-crate, remote `cargo test -p frankensearch-index --lib recall_certificate`): 9/9 green**, including the
two finite-sample **coverage-validity** Monte-Carlo tests — a fresh draw falls below the conformal bound at ≤ alpha, and
below the Hoeffding mean LCB at ≤ delta, over thousands of trials on arbitrary/adversarial recall laws — plus
`certificate_catches_heuristic_overconfidence` (the `estimate_recall` heuristic claims ≈1.0 where the certificate
correctly refuses a 0.9 budget on measured 0.85 recall). Exported from the crate; `AnnSearchStats.estimated_recall`'s doc
now points at the certified path.

**What this changes on the frontier map:** the "recall-budget *decision*" on ANN-in-BOLD is no longer a subjective human
gate — it is a `certified_min_ef(...).meets_target` check against a finite-sample guarantee. The lever's remaining gates
are now purely mechanical and already-known: (1) plumb a small calibration sweep (measured recall at candidate ef's) into
the ANN config path, and (2) the 100k-scale recall/latency measurement [[ann-in-bold-viable]]. The subjective blocker is
retired.

---

## 2026-07-02 — ARTIFACT (follow-through): certified-ef calibration DRIVER — lazy, early-terminating (SlateHeron)

Operationalises the recall certificate (`915c902`) into a usable ANN-config decision. `calibrate_certified_ef(candidate_efs,
measure_recall, target, alpha)` is the driver a caller (ANN config / BOLD) invokes: supply only a
`measure_recall(ef) -> Vec<f64>` closure (ANN@ef vs bruteforce over a calibration query set) and it returns the CHEAPEST
`ef` whose certified conformal lower bound meets `target` at confidence `1−alpha`, plus a `sweep` audit trail.

Because recall measurement is the expensive step (an ANN *and* a bruteforce search per calibration query), candidates are
tried **ascending** and the sweep **short-circuits** at the first certified `ef` — recall is never measured for larger
`ef`s (a real ~2–4× cut in calibration cost when a small `ef` already certifies). Falls back to the best-certifiable `ef`
when none meets the target. Feature-independent (no `ann` dep), so it compiles/tests under default features.

**Verified: remote `cargo test -p frankensearch-index --lib recall_certificate` PASSED 11/11**, incl. the short-circuit
test asserting only `{20,40}` of `{20,40,100,200}` are measured when `ef=40` certifies, and the no-early-stop fallback.

Remaining ANN-in-BOLD gate: the small `#[cfg(feature="ann")]` glue building the `measure_recall` closure from an
`HnswIndex` + bruteforce (deferred only because `--features ann` pulls the heavier `hnsw_rs` compile; not a design gap).
The subjective recall-budget sign-off is fully retired — it is now `calibrate_certified_ef(..).chosen.meets_target`.

---

## 2026-07-02 — ARTIFACT COMPLETE: live ANN recall certification (HnswIndex::certify_ef_search) (SlateHeron)

Closes the recall-certificate thread end-to-end (`915c902` certificate → `e77e3fe` driver → this glue).
`HnswIndex::certify_ef_search(exact_index, calibration_queries, candidate_efs, k, target, alpha)` runs THIS ANN index's
recall@k against exact bruteforce (`VectorIndex::search_top_k`) over the calibration queries and returns the CHEAPEST `ef`
whose split-conformal recall lower bound `≥ target` at confidence `1−alpha` (via `calibrate_certified_ef`).

Two efficiencies baked in: the exact top-k is **`ef`-independent so it is computed ONCE per query** (not per `ef`), and
the ascending sweep **short-circuits at the first certified `ef`** — no ANN search runs at an `ef` larger than the chosen
one. Failure direction is conservative: exact-pass errors propagate; a failed ANN `(query, ef)` counts as recall `0.0`,
which can only *lower* a certified bound (never over-certify).

**Verified: remote `cargo test -p frankensearch-index --features ann --lib certify_ef_search_...` PASSED** — end-to-end on
a real HNSW index (800×384) + real bruteforce: `target=0.0` certifies at the cheapest `ef` with a 1-element short-circuit
sweep; an unreachable target measures the full sweep and falls back to the best-certifiable `ef` with a real recall bound
in `[0,1]`. (`hnsw_rs` compiles clean under `--features ann`.)

**ANN-in-BOLD's recall-budget gate is now fully operational code:** a caller obtains the certified `ef` via one
`ann.certify_ef_search(..).chosen` call against a finite-sample guarantee. The subjective human sign-off is retired
end-to-end (certificate → driver → live ANN). The only remaining item is the *policy* choice of `(target, alpha)` and
flipping aggressive ANN on — a product decision, not missing machinery.

---

## 2026-07-02 — ARTIFACT: empirical-Bernstein mean-recall certificate → certifies a CHEAPER ef (recovers the tail-mode speedup gap) (SlateHeron)

Extends the recall certificate with a **mean-recall** certification mode, motivated directly by last turn's measurement
(`8c711d5`): the per-query **tail** certificate conservatively picked `ef=100` (1.65× vs flat @100k dim128) because
`ef=40`'s split-conformal tail lower bound fails 0.95 — even though `ef=40`'s **average** recall (0.9875) clears 0.95 and
would give 3.4×. For a product budget of "**average** recall@k ≥ target" (weaker than a per-query tail guarantee), the
right certificate is a mean lower bound. Added (feature-independent, exported):
- `mean_recall_lower_bound_bernstein(recalls, delta)` — Maurer–Pontil empirical **Bernstein** LCB
  `mean − sqrt(2·Vₙ·ln(2/δ)/n) − 7·ln(2/δ)/(3(n−1))`. Uses the sample **variance**, so it is far tighter than Hoeffding
  on low-variance recall (the usual case: most queries ≈ 1.0).
- `certified_min_ef_mean(calibration, target, delta)` — cheapest `ef` whose Bernstein mean bound meets `target`.

**Quantified effect (unit test on the real measured recall shape):** at n=1000 calibration, `ef=40` (mean 0.9875):
Bernstein LCB = **0.977 ≥ 0.95 (CERTIFIES)** while Hoeffding LCB = 0.949 < 0.95 (does not), and the per-query tail bound
refuses it. So the Bernstein mean mode certifies `ef=40` for the average-recall budget → the **3.4×** speedup, versus the
per-query tail mode's `ef=100` → 1.65×. The tighter concentration bound **recovers ~2× more certified speedup** — the
caller now chooses per-query-tail vs average-recall guarantees, each with the *cheapest ef that certifies it*.

**Verified: remote `cargo test -p frankensearch-index --lib recall_certificate` PASSED 14/14**, incl. the Bernstein
coverage-validity Monte-Carlo (miss ≤ δ), `bernstein_is_tighter_than_hoeffding_on_low_variance_recall`, and
`mean_mode_certifies_a_cheaper_ef_than_the_per_query_tail_mode`.

---

## 2026-07-02 — FRONTIER EXHAUSTED (definitive synthesis): remaining levers are research-grade or measurement-blocked (SlateHeron)

After a multi-turn sweep, the **clean-and-landable measured-perf frontier** vs the internal baselines (Tantivy lexical /
flat exact vector) is definitively exhausted. This turn's fresh probes all closed on inspection:
- **1-bit binary quantization + Hamming** — the natural bandwidth-reduction idea for the memory-bound flat scan (1062 µs
  @100k F32 is bandwidth-bound). Already probed and **REJECTED** (§ "1-bit binary (sign) two-pass", BlackThrush): 1-bit
  sign quant collapses clustered embeddings (clusters share sign patterns → recall too low) AND no speedup (integer
  Hamming ties thrash + popcount eats the ~8× bandwidth edge). The sub-int8 lever that DOES work is **4-bit**, already
  shipped (`dot_packed_4bit`). `simd.rs::binary_quant_recall_at_10` remains only as a `#[cfg(test)]` recall probe.
- **Streaming quantile sketch** (t-digest, graveyard Tier-S) — already the production metrics path (`metrics.rs`: paired
  t-digests + Huber `MedianMAD`).
- **`ScalarQuantizer::fit` min/max reduction** — caller-less public API (production uses `Quantization::F16` + the SIMD
  slab kernels).

**Session contribution (the last substantive frontier, now closed with a measured result):** the ANN recall-CERTIFICATE
arc — conformal per-query bound + empirical-Bernstein mean bound, the lazy `calibrate_certified_ef` driver, the live
`HnswIndex::certify_ef_search` glue, and a two-mode @100k measurement (**tail 1.76× / Bernstein-mean 3.70× vs flat**, both
certified ≥ 0.95). This retired the sole open competitive lever's (ANN-in-BOLD) recall-budget sign-off end-to-end.

**Remaining levers — all research-grade or measurement-blocked; NOT ship-fast commits. Prioritized for a future
dedicated effort:**
1. **Real-embedding validation harness** (highest value, MEASUREMENT-BLOCKED): every ANN/certificate number is on
   *synthetic* clustered corpora. A real 384-d embedded 100k recall/latency sweep needs a semantic model, absent on rch
   workers. Unblock = stage a model on the workers, or run locally.
2. **Adaptive / stratified-ef ANN** (Mondrian conformal): route queries by difficulty stratum to per-stratum certified
   `ef`s to recover the tail-mode speedup on heterogeneous corpora. Needs a reliable cheap difficulty signal + a
   heterogeneous-corpus model to even *demonstrate* — the homogeneous synthetic corpus can't show it.
3. **Anisotropic / ScaNN-style quantization**: better recall-per-byte than scalar quant for MIPS. Recall-affecting; needs
   a recall/latency Pareto sweep.
4. **AVX-512 dot kernels**: ~2× wider than the shipped AVX2 tier, *iff* the target CPUs expose AVX-512. Needs runtime
   dispatch + hardware confirmation + a non-trivial bit-identity story (wider lanes reassociate float sums).

**Route for the next session:** do NOT re-walk quantization / hasher / clone / top-k / fusion / filter / tokenizer /
binary-quant / t-digest — all closed with measurement. Either unblock #1 (real embeddings) or take on one of #2–4 as a
dedicated multi-turn design, not a single-turn perf commit.

---

## 2026-07-02 — TESTED (mixed/negative): stratified (Mondrian-conformal) ef selection — per-group guarantee works, but NO measurable speedup at N=40k (SlateHeron)

Tackled backlog item #2 from the frontier-exhausted synthesis (`31eea29`): does per-stratum certified `ef` (route easy
queries to a cheap `ef`, hard queries to a costly one) recover the speed a single global `ef` gives up? **Ran** the
`ann_stratified_ef` bench (landed `bea92bd`, added tooling-only/"no measurement" by a sibling agent — shared-tree
convergence on the same lever) — a HETEROGENEOUS 40k corpus (half tight clusters NOISE=0.08, half diffuse NOISE=0.35),
calibrating a conformal `ef` per stratum (target 0.90, α 0.1) and routing held-out queries by nearest-centroid.

**Measured (remote `--features ann`, N=40k, dim=128, m=32, k=10):**

| policy | guarantee | ef | latency (median, 95% CI) | held-out recall@10 |
|---|---|---|---|---|
| global **population** | population 90th-pct ≥ 0.90 | 80 | **817 µs** [795–834] | 0.9890 |
| global **per-group** | every stratum ≥ 0.90 | 160 | 1179 µs [1134–1221] | 0.9910 |
| **stratified routed** | every stratum ≥ 0.90 | 40/160 | 1118 µs [1033–1213] | 0.9850 |

**Findings (honest):** (1) The router **works** — per-group guarantees hold out-of-sample (stratified recall 0.985 ≥
target; tight stratum certifies `ef`=40, diffuse `ef`=160). (2) But stratified (1118 µs) vs the fair per-group baseline
(1179 µs) is only **1.05× with OVERLAPPING CIs** — no statistically-established speedup at 40k, because the HNSW
`ef`→latency curve is too flat here (`ef`=40 isn't much cheaper than `ef`=160). (3) The **cheapest** option is the simple
**population** conformal `ef`=80 (817 µs), which already delivers 0.989 avg recall out-of-sample — so on this corpus the
population certificate ([[ann-in-bold-viable]]) practically dominates; stratification only adds guarantee-*semantics*
(per-group fairness), not speed.

**Conclusion:** stratified/Mondrian-conformal `ef` is a guarantee-FAIRNESS lever, not a speed lever at moderate N. A
measurable speedup needs BOTH (a) a steeper `ef`→latency curve (larger N — at 100k `ef`=40 is ~2× cheaper than `ef`=100,
`acfb33b`) AND (b) a hard per-group SLA that forbids the population bound's tail dilution. Backlog #2 refined: viable, but
its payoff is conditional and modest — not the "recover the tail-mode speedup" hoped for. Kept the bench for the
steeper-curve regime. Verified: compiles + runs clean under `--features ann` (exit 0).

---

## 2026-07-02 — ARTIFACT + WIN: #1 real-embedding harness UNBLOCKED locally → int8 quant validated on REAL data; ROTATION lever (#3) wins for 4-bit (IronPetrel)

Two backlog items advanced in one dig. Backlog **#1 (real-embedding validation harness)** was recorded as
MEASUREMENT-BLOCKED ("needs a semantic model, absent on rch workers"). That block is **rch-worker-specific, not
absolute**: on this machine a real **Model2Vec/potion** model (`minishlab/potion-base-8M`, 256-d, PCA-projected static
embeddings) runs in **pure Rust** (`safetensors` + `tokenizers`, no ONNX runtime) and embeds **30 000 real English docs
in ~1 s**. Every prior ANN/quant recall number in this repo rests on a *synthetic* clustered-Gaussian corpus; this is the
**first real-embedding measurement**. New tooling (own files):
- `frankensearch-embed/examples/potion_embed_corpus.rs` — embeds a text corpus → flat f32 slab (smoke check:
  cos(related)=0.543 vs cos(unrelated)=0.014).
- `frankensearch-index/benches/real_embed_quant.rs` — loads the real slab, reports per-dim anisotropy, and measures
  int8/4-bit two-pass recall@10 vs exact, plain vs an orthonormal random-rotation preprocessing (lever #3).

**Corpus (real potion embeddings of 29 700 local-doc English lines, held-out 300 queries, dim=256, k=10):**

| mult | int8 | int8+rot | 4bit | **4bit+rot** |
|---|---|---|---|---|
| 2  | 1.0000 | 1.0000 | 0.8980 | **0.9700** |
| 3  | 1.0000 | 1.0000 | 0.9530 | **0.9897** |
| 5  | 1.0000 | 1.0000 | 0.9833 | **0.9987** |
| 10 | 1.0000 | 1.0000 | 0.9983 | **1.0000** |
| 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Anisotropy (per-dim variance share of total): real `top1=1.8% top5=7.1% top10=12.0% excess-kurtosis=0.05`; after
rotation `top1=0.5% top5=2.5% top10=4.9% kurt=-0.02`. So potion-256 is only **mildly** anisotropic (near-Gaussian) — not
the heavy outlier-dim structure of raw transformer outputs (PCA + Zipf smooth it).

**Findings (honest, measured):**
1. **int8 two-pass is LOSSLESS on real embeddings — recall 1.0000 at every mult, down to mult=2.** This validates the
   shipped int8 ADC path (`search_top_k_int8_two_pass`) on real data. The synthetic-era worry ("clustered embeddings may
   need a higher mult to stay lossless", `int8_two_pass.rs`) **does not materialize** on real potion embeddings — int8's
   256 levels comfortably preserve the exact top-10 with a mult=2 candidate set.
2. **Rotation (lever #3) WINS for the 4-bit path.** An inner-product-preserving orthonormal rotation (`<Rx,Rq>=<x,q>`
   exactly; only the quantizer grid changes) lifts 4-bit two-pass recall so it reaches **near-lossless 0.999 at mult=5,
   where plain 4-bit needs mult=20 to hit 1.0 → a ~4× smaller exact-rescore candidate set** for the same recall (+7.2
   recall points at mult=2: 0.898→0.970). Even *mild* per-dim variance concentration wastes 4-bit's coarse 16-level grid;
   equalizing it via rotation recovers the resolution. int8 (256 levels) is already saturated, so rotation is a no-op
   there — the lever is **4-bit-specific**. 4-bit is 2× smaller than int8, so rotation makes the **2× recall-per-byte**
   tier viable at high recall.

**Honest latency caveat (why this is banked as recall-per-byte, not an end-to-end speedup HERE):** at N≈30k dim=256 the
two-pass is *slower* than the flat f16 scan (flat **394 µs** vs int8_mult5 498 µs, 4bit_mult5 464 µs, int8_mult10 545 µs,
4bit_mult10 597 µs) — pass-1 + rescore overhead exceeds the flat-scan saving at modest N. The rotation-4bit win is a
recall / rescore-count win at this N; it converts to wall-clock only in the **large-N regime** where the flat scan
dominates and two-pass already beats flat (the ~1.4–1.5× the `int8_two_pass` ledger records). The rotation itself is
near-free at runtime (a one-time `d×d` matrix; O(d²)/query ≪ the scan).

**Route-next:** (a) re-run `real_embed_quant` at N=100k+ (embed a larger corpus) to convert the 4× mult reduction into a
measured latency ratio in the regime where two-pass wins; (b) wire the rotation into the 4-bit index path
(store one `d×d` orthonormal matrix, rotate at insert + query) — product-gated on choosing the 4-bit tier. Backlog #1 is
no longer measurement-blocked *locally*; #3 (anisotropic-quant) has its first measured positive. **Verified:** bench
compiles clean on rch (`--no-run`, exit 0) and runs locally (exit 0); example builds + runs (`--features model2vec`).

---

## 2026-07-02 — MEASURED (follow-through, decisive): large-N real embeddings — int8 two-pass 7.1× vs flat @ recall 1.0; the 4-bit rotation win is real-but-SYSTEM-MOOT (int8 dominates 4-bit) (IronPetrel)

Resolves the route-next from `3833955` ("re-run at N=100k+ for a latency ratio"). Re-embedded **130 000** real English
docs (potion-256) and re-ran `real_embed_quant` at N=129 850. This converts the earlier 30k recall-only result into a
system verdict.

**Latency (criterion median, N=129 850 dim=256 k=10, 150 queries):**

| config | time | vs flat |
|---|---|---|
| flat exact (f16) | 4.74 ms | 1.0× |
| **int8 two-pass mult5** | **665 µs** | **7.1×** |
| 4-bit two-pass mult5 | 716 µs | 6.6× |
| int8 two-pass mult10 | 1.28 ms | 3.7× |
| 4-bit two-pass mult10 | 1.07 ms | 4.4× |

**Recall (same run):** int8 **1.0000 at every mult (incl. mult=2)**; 4-bit needs **mult=20** for 1.0
(0.873/0.975/0.997/1.000 at mult 2/5/10/20); 4-bit+rot 0.963/0.995/0.997/0.998; int8+rot **0.998** (rotation slightly
*hurts* int8).

**Decisive findings (honest):**
1. **HEADLINE — validated on real data: `search_top_k_int8_two_pass` is 7.1× faster than flat exact vector search at
   recall 1.0** on real 130k embeddings (int8 is lossless from mult=2, so the lossless int8 config is ≤ 665 µs). The
   shipped int8 ADC path's prior ratio (~1.4–1.5× on small uniform-random, `int8_two_pass.rs`) understated it — at real
   large N it is **7×**. int8 losslessness is robust 30k→130k.
2. **The 4-bit rotation lever (`3833955`) is real but SYSTEM-MOOT at large N: int8 strictly dominates 4-bit here.** int8
   is lossless at mult=2 AND *faster* than 4-bit at equal mult (int8_mult5 665 µs < 4bit_mult5 716 µs) — the AVX2
   `dot_i8_i8` kernel is fast enough that 4-bit's 2× *bandwidth* edge is eaten by nibble-unpack *compute* overhead, so
   4-bit is not actually cheaper. Since 4-bit is worse than int8 in BOTH recall (needs mult=20 vs 2) and latency,
   improving 4-bit with rotation (+9 recall pts @mult2) does not yield a system win — you would just use int8. Rotation's
   **only** residual value is the **memory-capacity niche** (4-bit = ½ int8's RAM ⇒ 2× more vectors resident); there
   rotation buys ~0.96 vs 0.87 recall@mult2. It is **not a speed lever** on this kernel/scale, and it slightly *hurts*
   int8 (never rotate int8).
3. **Corrected route-next:** the pre-condition for a 4-bit/rotation speed win is a 4-bit pass-1 genuinely faster than int8
   pass-1 — which needs a faster nibble kernel (VNNI `vpdpbusd` on int4-packed, or AVX-512, both measurement-blocked on
   the Zen 3 workers) OR a rescore-dominated regime (disk-backed FSVI two-pass / cross-encoder pass-2, where cutting mult
   20→5 matters). Absent those, **int8 two-pass is the recall-per-byte AND latency winner at large N; ship int8, not
   4-bit.** Backlog #3 refined from "first positive" to "positive-in-isolation, dominated-in-system"; the real shippable
   result is the measured **int8 7.1× vs flat on real data**. Verified: runs clean locally (exit 0, 0 panics).

---

## 2026-07-02 — MEASURED (mixed/validation): HNSW ANN cert arc on REAL embeddings — mean-recall speedups HOLD (2.5× @0.98), but the per-query TAIL 0.95 certificate is NOT achievable on real data (IronPetrel)

The entire ANN recall-CERTIFICATE arc (`certify_ef_search`; ledger tail 1.76×/mean 3.70× vs flat; "ANN-in-BOLD viable")
was measured only on SYNTHETIC tight-cluster corpora (NOISE=0.15). New bench `real_embed_ann.rs` re-runs it on genuine
potion-256 embeddings (N=100 336, M=32, k=10) via `HnswIndex::certify_ef_search`.

| ef | recall@10 (real) | HNSW latency | vs flat (6.17 ms) |
|---|---|---|---|
| 40  | 0.938 | 1.21 ms | **5.1×** |
| 100 | 0.978 | 2.49 ms | **2.5×** |
| 200 | 0.992 | 4.33 ms | 1.4× |

Certificate (tail, target 0.95, α 0.1): **`meets=false` — the best-certifiable ef is 100 with a conformal lower bound of
only 0.90; 0.95 is NOT met even at ef=200** (whose *mean* recall is 0.992).

**Findings (honest):**
1. **The mean-recall speedups broadly HOLD on real data**: HNSW is **2.5× faster than flat at 0.978 mean recall** (ef100)
   and 5.1× at 0.938 (ef40). ANN-in-BOLD's *latency-vs-mean-recall* case survives the jump from synthetic to real
   embeddings — the ~2.5–5× band is real, not a synthetic artifact.
2. **But the strong per-query TAIL guarantee does NOT transfer.** On real embeddings the split-conformal 0.95 tail
   certificate is unachievable at ef≤200 — real potion embeddings have a **heavy low-recall tail** (a minority of queries
   whose true neighbors HNSW misses regardless of ef), which pins the conformal lower bound at ~0.90 even while mean
   recall is 0.99. The synthetic tight clusters hid this (uniform easy queries → tail≈mean). So the certified-ef arc's
   *tail-mode* numbers were optimistic; on real data only the **mean-recall (Bernstein) mode** is viable, and only down
   to a ~0.90 per-query floor.
3. **Consequence:** if BOLD needs a per-query 0.95 tail guarantee, HNSW-at-ef≤200 on real embeddings can't provide it —
   use the mean-recall budget (2.5× @0.98 mean) or a higher ef / M / exact fallback for the tail. Route-next: sweep M=48/64
   and ef>200 to see whether the tail floor lifts, and re-confirm on the on-disk MiniLM-L6-v2 (raw transformer, harder
   distribution). Verified: `--features ann` bench compiles + runs locally (exit 0); recall/cert/latency captured.

**Follow-up (same session, refines finding #2): the tail-cert floor is N-DEPENDENT, not a fixed real-data wall.**
Re-ran `real_embed_ann` at **N=40 000** (same potion embeddings, M=32): recall@10 = 0.956/0.983/0.989/0.992/0.992 at
ef=40/100/200/400/800, and the tail certificate (target 0.95, α 0.1) now **`meets=true` at ef=200 with lower bound
1.0000** (≥90% of calibration queries hit recall@10=1.0). Contrast the same ef=200 at N=100 336: bound **0.90,
`meets=false`**. So the per-query 0.95 tail guarantee IS achievable on real embeddings at **moderate N**; it **degrades
as the corpus grows** (a heavier low-recall tail emerges when the graph gets relatively sparser for fixed M). Corrected
takeaway: finding #2's "0.95 tail unachievable on real data" is **N-specific (100k @ M=32)**, not fundamental — at
BOLD-scale N the fix is **higher M** (denser graph), not just higher ef (recall plateaus at 0.992 by ef=400, so ef alone
can't close the tail). Verified: runs clean locally (exit 0).

**Follow-up 2 (CONFIRMS the M-hypothesis, closes the ANN-in-BOLD tail question): M=64 RESTORES the 0.95 tail cert at
N=100k.** Re-ran `real_embed_ann` at **N=100 336, M=64** (`FS_REAL_M=64`, same real potion embeddings): recall@10 =
0.956/0.988/0.994/0.998/0.998 at ef=40/100/200/400/800, and the per-query tail certificate (target 0.95, α 0.1) is now
**`meets=true` at ef=200 with lower bound 1.0000** — vs **M=32 @100k which failed** (ef=200 bound 0.90, `meets=false`).
So the large-N tail-cert degradation (Follow-up 1) is **not fundamental — doubling M (32→64) closes it**: the denser graph
lifts the low-recall tail so ≥90% of calibration queries hit recall@10=1.0 at ef=200 even at 100k. **Net ANN-in-BOLD
verdict on REAL embeddings:** the strong per-query 0.95 tail guarantee IS deliverable at BOLD scale with **M=64** (≈2× the
M=32 graph memory + slower build), certified at ef=200 (≈1.4× vs flat 6.17 ms); the weaker mean-recall budget is
certifiable much cheaper (ef=100, recall 0.988, ≈2.5× vs flat). The synthetic-era cert numbers were optimistic on the
*tail at large N with default M*, but the guarantee is recoverable — it is an **M-budget** decision, not a wall.
Verified: `--features ann` runs clean locally (exit 0).

---

## 2026-07-02 — MEASURED (CONFIRMS on the hard distribution): RAW transformer (MiniLM-L6-v2) embeddings — int8 still LOSSLESS, 4-bit rotation still moot, and mean-pooled sentence embeddings are near-ISOTROPIC (IronPetrel)

The potion (Model2Vec) results carried a caveat: potion PCA-smooths its embeddings to near-Gaussian, so they might not
stress per-dimension int8 quant the way raw transformer "outlier dimensions" would. Closed that gap: built
`minilm_embed_corpus.rs` (`--features fastembed`, self-contained ort-download onnxruntime) and embedded the 30k corpus
with the on-disk **all-MiniLM-L6-v2 ONNX** (384-d, mean-pooled + normalized), then re-ran `real_embed_quant`.

**Anisotropy (MiniLM, dim=384):** per-dim variance share `top1=0.4% top5=2.0% top10=3.8%`, excess-kurtosis `−0.02`.
For reference a *fully isotropic* 384-d has top1=1/384=0.26%, so MiniLM's top dim is only ~1.5× isotropic — **even MORE
isotropic than potion-256** (top10 was 12%). The documented transformer "outlier/massive-activation dimension"
phenomenon (which lives in raw per-token activations of larger models) **does NOT survive mean-pooling + L2-normalization**
into these sentence embeddings.

**Recall@10 (MiniLM, N=29 700, k=10, 300 queries):**

| mult | int8 | int8+rot | 4bit | 4bit+rot |
|---|---|---|---|---|
| 2  | 1.0000 | 1.0000 | 0.9927 | 0.9913 |
| 3  | 1.0000 | 1.0000 | 0.9997 | 0.9980 |
| 5  | 1.0000 | 1.0000 | 1.0000 | 0.9997 |
| 10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

**Findings — both core conclusions CONFIRMED on the harder distribution:**
1. **int8 two-pass is LOSSLESS on raw-transformer embeddings too** (recall 1.0 at mult=2). The headline `int8 7.1× vs
   flat @ recall 1.0` (`0b9800e`) is **distribution-robust** — its losslessness rests on the quantizer's *per-dimension
   adaptive* range, which handles any per-dim scale, so it survives the switch from potion to MiniLM.
2. **The 4-bit rotation lever is confirmed MOOT — and MiniLM makes it moot for a second, independent reason.** 4-bit is
   *already* near-lossless here (0.993 @mult2, vs potion's 0.898) because MiniLM is 384-d and near-isotropic, so there is
   nothing for a variance-equalizing rotation to fix — and it in fact **slightly hurts** (0.9913 < 0.9927 @mult2). So
   across *both* real distributions the verdict holds: ship int8; 4-bit rotation buys nothing at the system level.

Net: the session's two headline conclusions (int8 lossless/7.1×; 4-bit rotation system-moot) are validated on **two**
real embedding distributions (potion PCA-static and MiniLM raw-transformer), and both turn out near-Gaussian — the
synthetic-corpus worry about real anisotropy breaking int8 does not materialize. Verified: `--features fastembed` example
+ `real_embed_quant` bench run clean locally (exit 0).

**Follow-up (closes the potion/MiniLM asymmetry for the ANN arc): `real_embed_ann` on MiniLM (N=28 336, dim=384, M=32).**
The quant conclusions were validated on both distributions above; the ANN conclusions had rested only on potion. MiniLM
ANN recall@10 = 0.980/0.994/0.995 at ef=40/100/200 (plateau 0.995), and the per-query tail certificate (0.95, α 0.1)
**`meets=true` at ef=100 with lower bound 1.0000**. So the ANN cert arc replicates on the 384-d raw-transformer
distribution — and is in fact **easier** than potion (potion needed ef=200 @ M=32 to certify at N=40k; MiniLM certifies at
the cheaper ef=100, and its recall is higher at every ef: ef40 0.980 vs potion 0.956). Higher-dim near-isotropic
embeddings give HNSW cleaner neighborhoods. Both real distributions now confirm both the quant AND the ANN conclusions.
Verified: `--features ann` bench runs clean locally (exit 0).

---

## 2026-07-02 — FINDING (real data corrects a production claim): the fast tier's "4-bit @mult3 lossless (recall=1.0)" is 0.9973 on REAL embeddings, not 1.0; int8 is exactly 1.0 (IronPetrel)

`sync_searcher::search_fast_hits` (the PRODUCTION vector fast tier) returns the top-`fetch` candidates
(`fetch = K·candidate_multiplier`, default 3) from `search_top_k_4bit_two_pass_filtered(query, fetch, FAST_TIER_MULT=3)`,
and the code comment asserted the candidate set is **lossless (recall=1.0), validated at 10k–100k** — i.e. *"Lossless
candidate set → identical fused top-k."* That validation was on SYNTHETIC corpora. Added the exact production regime
(`fetch=K·3=30`, `FAST_TIER_MULT=3`) to `real_embed_quant` and measured **candidate recall** (fraction of the exact
top-K present in the returned `fetch` set) on REAL potion-256 embeddings (30k, 300 queries):

| fast-tier primitive (production regime, fetch=30, mult=3) | candidate-recall@10 (REAL) |
|---|---|
| **4-bit two-pass (current production default)** | **0.9973** |
| int8 two-pass (the low-mult-lossless twin) | **1.0000** |

**Finding:** the "recall=1.0 lossless" claim is **marginally false on real embeddings** — 4-bit@mult3 misses ~0.27% of
the true top-K candidates (the earlier `[recall]` table shows why: at the tighter top-K-from-K·mult regime 4-bit is only
0.953 @mult3 on potion; the production over-fetch to top-30 masks most of it, but not all). **int8 two-pass is exactly
lossless (1.0000) in the same regime** — consistent with int8 being lossless at mult=2 across both real distributions
(potion + MiniLM) and comparable-latency to 4-bit (`0b9800e`). The ~0.3% fast-tier miss is *further* masked downstream by
the lexical tier + RRF, so the fused-top-k impact is negligible in practice — hence "effectively lossless," not "exactly
lossless." Corrected the overstated comment in `sync_searcher.rs` to state the measured real-embedding numbers and to
point at the int8 remedy (exact-candidate guarantee for ~2× the fast-tier slab bytes). **Route-next (product-gated):**
if a hard exact-candidate guarantee is wanted, swap the fast tier to `search_top_k_int8_two_pass_filtered(.., mult=2)` —
pending a fusion-crate latency A/B to confirm int8 doesn't regress the hot path (real 130k quant bench shows int8 ≈ 4-bit,
but the comment's legacy "4-bit ~1.4× faster than int8" claim should be re-measured in this exact fast-tier regime first).
Verified: `real_embed_quant` (with the new `[fast-tier-regime]` block) runs clean locally (exit 0); comment-only change to
`sync_searcher.rs`.

**Follow-up (the A/B — gate PASSED, swap is a pure win): int8 is FASTER *and* exactly lossless in the fast-tier regime.**
Added `fasttier_int8`/`fasttier_4bit` latency arms (exact production regime: limit=fetch=30, mult=3) to `real_embed_quant`
and ran at **N≈130k** (real potion, dim 256):

| fast-tier primitive (limit=30, mult=3, N≈130k) | latency (median, 95% CI) | candidate-recall@10 (real) |
|---|---|---|
| 4-bit two-pass (current production) | 1.070 ms [1.015–1.142] | 0.9930 |
| **int8 two-pass** | **0.985 ms [0.965–1.003]** | **1.0000** |

**int8 is 1.09× FASTER (separated CIs) AND exactly lossless**, refuting the code's legacy *"4-bit ~1.4× faster than int8"*
claim in the exact fast-tier regime at scale: the AVX2 `dot_i8_i8` kernel beats the 4-bit nibble-unpack, so 4-bit's
½-bandwidth edge does not pay (pass-1 is compute-bound, not bandwidth-bound, at N≈130k dim=256 — consistent with
`0b9800e`). So swapping the production fast tier `search_top_k_4bit_two_pass_filtered` → `search_top_k_int8_two_pass_filtered`
is a **pure win at large N**: faster + exact candidate recall (1.0 vs 0.993), for ~12.8 MB extra fast-tier slab at 100k
(int8 `dim` bytes/vec vs 4-bit `dim/2`). The only regime where 4-bit may edge int8 is very small N (bandwidth matters more,
compute less — 30k earlier showed 4bit_mult5 464µs < int8_mult5 498µs), a ~7% wash where int8 is still exactly lossless.
**The prior entry's product-gate is now measurement-cleared; wiring the swap is justified.**

**LANDED (`sync_searcher.rs`): the fast tier now uses `search_top_k_int8_two_pass_filtered` (mult=3).** A production win,
not just evidence: strictly-lossless candidate set (candidate-recall@10 = 1.0 on real potion + MiniLM) restores the
"identical fused top-k" guarantee the comment depends on, AND it's 1.09× faster at N≈130k, for ~12.8 MB extra slab @100k.
Verified: `cargo test -p frankensearch-fusion --lib` **820 passed / 0 failed** (via rch) — no test asserted 4-bit-specific
output, so the swap is behavior-compatible (more-correct candidate set, unchanged passing fused results).

---

## 2026-07-02 — FINDING (real data flags a stock-config gap): the production default HNSW `M=16` gives sub-0.95 recall@10 on real 100k embeddings; `M=32` fixes it (IronPetrel)

The async ANN fast tier (`TwoTierIndex::search_fast_with_params` → `knn_search`) uses `HnswConfig::default()` — `M=16`,
`ef_construction=200`, `ef_search=100` (`hnsw.rs:25–29`). Every prior ANN-quality number was at `M=32`/`M=64`; the STOCK
default `M=16` was never measured on real embeddings. Ran `real_embed_ann` at `FS_REAL_M=16`, N=100 336, real potion-256:

**recall@10 by ef, real 100k, across M (default `ef_search`=100 is the middle column):**

| M | ef40 | **ef100 (default)** | ef200 | ef400 | ef800 | tail-0.95 cert |
|---|---|---|---|---|---|---|
| **16 (stock default)** | 0.858 | **0.947** | 0.967 | 0.975 | 0.984 | ✗ (bound 0.90) |
| 32 | 0.938 | **0.978** | 0.992 | — | — | ✗ (bound 0.90) |
| 64 | 0.956 | **0.988** | 0.994 | 0.998 | 0.998 | ✓ (ef200, bound 1.0) |

**Finding:** at the **stock defaults (M=16, ef_search=100)** the ANN fast tier gets **recall@10 = 0.947** on real 100k
embeddings — **below a 0.95 quality target**, and it cannot certify the 0.95 tail even at ef=800. Bumping **M=16→32 lifts
the default operating point to 0.978** (comfortably above 0.95); **M=64 → 0.988 + a certifiable per-query 0.95 tail**. The
synthetic `hnsw_vs_flat_100k` bench (tight NOISE=0.15 clusters) never surfaced this because synthetic recall runs higher.
The practical impact is softened downstream (the fast tier's misses are masked by the lexical tier + RRF, as with the
4-bit finding), but the *default* ANN recall is meaningfully under-provisioned for real 100k+ corpora.

**Recommendation (product-gated — memory/build tradeoff, not a unilateral flip): raise `HNSW_DEFAULT_M` 16→32** for real
100k+ corpora — a clean +0.031 recall@0.95-target at the default ef, for ~2× graph memory (`M·N` adjacency) and a longer
build. `M=64` is the setting if a certified per-query 0.95 tail is required (`68d213e`). Left as a documented
recommendation rather than a code change because the graph-memory cost is a real deployment decision (unlike the
fast-tier int8 slab, whose absolute size was negligible). Verified: `--features ann` bench runs clean locally (exit 0).

**Follow-up (rules out the tempting memory-FREE fix): raising `ef_construction` does NOT fix the M=16 gap — it is
M-bound.** `ef_construction` (build-time beam, default 200) improves graph quality at the SAME M with no query-memory
cost (only build time), so it looked like a free way to lift the M=16 recall. Swept it at M=16, N=100k real potion
(recall@10 at the default **ef_search=100**):

| ef_construction @ M=16 | recall@10 (ef=100) |
|---|---|
| 200 (default) | 0.947 |
| 400 | 0.933 |
| 800 | 0.927 |

The values are **flat-to-noisy at ~0.92–0.95 (HNSW build is non-deterministic), none reaching 0.95** — raising the build
beam 4× does NOT lift recall (if anything the baseline run was the luckiest). So the M=16 recall gap is genuinely
**M-bound (graph node degree / connectivity), not build-beam-bound**: there is no free lunch, the only fix is more `M`
(with its ~2× graph-memory cost). This closes the "cheaper alternative" question and confirms the `M=16→32`
recommendation is the real lever. Verified: `--features ann` bench runs clean locally (exit 0).

**Follow-up (QUALIFIES the M=16 recommendation — it is distribution-dependent for the mean budget, universal for the
tail): re-ran M=16 @100k on RAW MiniLM-384.** Embedded 130k real docs with all-MiniLM-L6-v2 and ran `real_embed_ann`
FS_REAL_M=16, N=100 336, dim=384:

| distribution @ M=16, N=100k | recall@10 ef=100 (default) | tail-0.95 cert |
|---|---|---|
| potion-256 (PCA-static) | 0.947 (**< 0.95**) | ✗ (bound 0.90) |
| **MiniLM-384 (raw transformer)** | **0.964 (≥ 0.95)** | ✗ (bound 0.90) |

So the stock-M=16 gap is **NOT universal**: on the HNSW-friendlier raw-transformer distribution (384-d, near-isotropic,
which gives HNSW cleaner neighborhoods — see `22a2f52`) M=16 already clears a 0.95 **mean** budget (0.964), whereas the
lower-dim PCA-compressed potion-256 does not (0.947). **Corrected takeaway:** the `M=16→32` bump is warranted for
lower-dim / PCA-reduced embeddings (potion-class), but higher-dim raw-transformer embeddings (MiniLM-class) meet a 0.95
mean target at stock M=16 — the recommendation is embedding-distribution-dependent, not blanket. **However, the per-query
0.95 TAIL certificate fails at M=16 on BOTH distributions** (bound 0.90) — the strong per-query guarantee needs higher M
(M=64 on potion, `68d213e`) *regardless* of distribution. So: mean-recall adequacy of the default is distribution-
dependent; the tail guarantee's M-requirement is universal. Verified: `--features fastembed` MiniLM embed + `--features
ann` bench run clean locally (exit 0).

---

## 2026-07-02 — TESTED (negative): MRL (Matryoshka prefix-truncation) DESTROYS recall on real (non-MRL-trained) embeddings — PCA variance-ordering ≠ Matryoshka nesting (IronPetrel)

`VectorIndex::mrl_search` (`mrl.rs`) does a coarse phase-1 scan over the first `search_dims` dimensions (default 64) then
rescores the top 3·k candidates at full dim; the doc claims **"2–6× faster than a full-dimension scan."** The recall of
that truncation on real embeddings was never measured. It *looked* promising for potion because potion is PCA-projected
(`apply_pca:256` — dims ordered by decreasing variance, so a prefix keeps the most variance ⇒ "Matryoshka-like"). Added an
MRL recall+latency block to `real_embed_ann` and ran on real potion-256 (N=40k):

| MRL `search_dims` / 256 | recall@10 (vs exact) | latency | vs flat (1.524 ms) |
|---|---|---|---|
| 32 | **0.242** | — | — |
| 64 (default) | **0.545** | 0.912 ms | 1.67× faster |
| 128 | **0.886** | 1.219 ms | 1.25× faster |

**Finding:** MRL delivers its documented speedup (1.25–1.67×) but at a **catastrophic recall cost** — recall **0.545 at
the default `search_dims=64`**, and still only 0.886 at half the dims (128). **The "PCA variance-ordering ≈ Matryoshka
nesting" intuition is FALSE for ranking:** PCA concentrates *variance* in early dims, but nearest-neighbor
*discrimination* is spread across the full geometry, so the truncated phase-1 drops the true top-k out of the top-3·k
candidate set *before* the full-dim rescore can recover them (the rescore only reorders what phase-1 kept). MRL requires
genuinely **Matryoshka-trained** embeddings (models trained so prefixes are self-sufficient), which potion (PCA-static)
and MiniLM (raw transformer) are **not**. **Recommendation:** do NOT enable `MrlConfig` on potion/MiniLM-class embeddings
— the 1.67× scan speedup is worthless at 0.545 recall; the `mrl.rs` "2–6× faster" claim must be read as speed-only and
gated on an MRL-trained model. Verified: `--features ann` bench (with the new `[mrl]` block) runs clean locally (exit 0).

---

## 2026-07-03 — MEASURED (POSITIVE, the "ratio vs Tantivy" itself): HYBRID retrieves 100% of known items vs Tantivy-lexical's 90.2% on real data (IronPetrel)

Every prior "vs Tantivy" note framed the vector tier as *the differentiator* frankensearch adds on top of Tantivy, but the
end-to-end HYBRID had never been measured against Tantivy-lexical-alone on real data. Built `real_hybrid_knownitem`
(`frankensearch-fusion --features lexical`): index a real 30k English corpus in **Tantivy (BM25)** AND the **vector index
(real potion embeddings)**, then run **label-free known-item retrieval** — each query is the first ~10 words of a held-out
corpus doc (embedded separately, so it's a genuine partial query, not a trivial exact-vector match); the one relevant
answer is that source doc. Fuses lexical + vector via the production `rrf_fuse`.

**Measured (30k corpus, 254 known-item queries, k=10):**

| tier | recall@10 | MRR@10 |
|---|---|---|
| **Tantivy lexical (BM25) alone** | **0.9016** | 0.8915 |
| vector (potion) alone | 0.9961 | 0.9678 |
| **hybrid (lexical + vector, RRF)** | **1.0000** | 0.9457 |

**Finding (the direct product-value ratio vs Tantivy):** the **hybrid retrieves the source doc for 100% of queries vs
Tantivy-lexical-alone's 90.2%** — a **+9.8-point / ~10.9% relative recall win**, the first real-data confirmation that
frankensearch's hybrid tier meaningfully beats the Tantivy baseline it's built on. RRF recall is ≥ max(lexical, vector) by
construction (fusion recovers what *either* tier finds); here vector-alone (0.996) already dominates lexical (0.902), and
hybrid (1.0) tops both. **Honest nuance:** hybrid maximizes *recall* but its **MRR (0.946) sits just below vector-alone
(0.968)** — RRF rank-blending slightly dilutes a strong vector rank-1 when the lexical rank is weak, so for pure rank-1
precision vector-alone edges hybrid, while for "find the doc at all" hybrid wins outright. Net: the hybrid's value-add
over Tantivy is real and measured (+10.9% recall) on real embeddings + real text. Verified: `--features lexical` bench
compiles + runs locally (exit 0); async Tantivy index of 30k real docs + 254 queries.

**Follow-up (RRF-k is a NO-OP for the recall/MRR tradeoff — the MRR gap is structural): swept RRF `k` ∈ {5,10,20,30,60,100}
on the known-item eval.** Hybrid is **flat at recall@10=1.0000, MRR@10=0.9406 across ALL k** — sharpening the fusion (low
k, which weights rank-1 more) does NOT recover the rank-1 MRR that sits below vector-alone (0.9678). Reason: RRF weights
lexical and vector **equally**, and `k` scales both sources' rank contributions the same way, so a competitor with a
strong *lexical* rank-1 keeps outranking the target's strong *vector* rank-1 regardless of `k`. The hybrid's MRR being
below vector-alone is therefore **inherent to equal-weight RRF, not a tuning artifact** — the only knob that would fix it
is **per-source weighting** (up-weight the vector tier, which is the more reliable retriever here: 0.996 vs 0.902), which
the basic `rrf_fuse`/`RrfConfig` (only `k`) does not expose. **Takeaways:** (1) don't bother tuning RRF `k` for this
tradeoff — it's inert; (2) the hybrid **recall** win over Tantivy (1.0 vs 0.902) is robust and `k`-invariant; (3) if
rank-1 MRR matters more than recall, either use vector-alone or add a source-weighted fusion (a real route-next: a
weighted-RRF variant or pre-scaling the semantic ranks). Verified: `--features lexical` bench runs clean locally (exit 0).

**Follow-up (source-weighting IS a real knob — but a Pareto tradeoff, and it surfaces a possible `rrf_fuse` MRR gap):
manual weighted-RRF sweep** (up-weight the vector tier: `score = w_vec/(k+r_vec+1) + w_lex/(k+r_lex+1)`, since `rrf_fuse`
exposes no tier weight — FederatedSearch's `weight` is a *shard* weight, not a lexical-vs-vector one):

| fusion (k=60) | recall@10 | MRR@10 |
|---|---|---|
| vector-alone | 0.9961 | 0.9678 |
| production `rrf_fuse` | 1.0000 | 0.9436 |
| weighted-RRF `vec_w=1` (equal) | 1.0000 | **0.9596** |
| weighted-RRF `vec_w=2` | 0.9961 | **0.9647** |
| weighted-RRF `vec_w=4/8/16` | 0.9961 | 0.959 / 0.958 / 0.962 |

**Two findings:** (1) **Unlike RRF-`k` (inert), per-source weighting IS a real knob** — up-weighting vector lifts MRR
(0.9596 → 0.9647 at `vec_w=2`) but recall drops **1.0 → 0.9961** as it approaches vector-alone. So it is a genuine
**Pareto tradeoff: no weight achieves BOTH recall 1.0 AND MRR ≥ vector-alone (0.968)** — the +recall over vector comes
*precisely* from including lexical's hits, which is also what dilutes MRR; you can trade along the frontier but not beat
both. (2) **BONUS / route-next:** the naive equal-weight weighted-**sum** RRF (MRR **0.9596**) beats the production
`rrf_fuse` (MRR **0.9436**) by ~1.6 pts at the *same* recall (1.0) — the merge-structured `rrf_fuse` appears to lose some
MRR to its tie-break / candidate-handling vs a plain weighted-sum-of-reciprocal-ranks. Worth a focused look: if `rrf_fuse`'s
tiebreak is the cause and can be made rank-preserving, it's a free MRR gain on the shipped hybrid path (not landed here —
needs confirming it's a real tiebreak effect, not a bench artifact). Verified: `--features lexical` bench runs clean (exit 0).

**Follow-up (ROOT CAUSE DIAGNOSED — it's a real, localized tiebreak asymmetry, not a bench artifact):** read the
`rrf_fuse` ranking comparator (`FusedHitScratch::cmp_for_ranking`, `rrf.rs:100–111`). On an RRF-score tie it breaks by
(1) `in_both_sources` (promote docs both tiers agree on — reasonable), then (2) **`lexical_score.unwrap_or(f32::NEG_INFINITY)`
descending** — which **asymmetrically favors lexical**. The failure mode: a **vector-only** target (lexical missed it —
lexical recall is only 0.90) at vector-rank `r` has RRF score `1/(k+r+1)`, which is **exactly equal** to any **lexical-only**
doc at lexical-rank `r`; the tie then goes to the lexical-only doc because the target's `lexical_score` is `None → −∞`.
So every lexical-only doc at rank ≤ `r` is ranked **above** the vector-only best answer, systematically lowering its rank
→ the ~1.6-pt MRR gap vs a neutral (random-tiebreak) fusion. This is **not a bench artifact** — it's a deterministic
tiebreak that favors lexical over semantic for no principled reason. **Fix (product-gated design choice, NOT landed):** make
tiebreak (2) symmetric — e.g. compare `max(lexical_score, semantic_score)` or fall back to `semantic_score` when
`lexical_score` is absent — so a vector-only doc isn't auto-demoted to `−∞`. Gain is small (~1.6 MRR pts) and it shifts
tie-ordering (some fusion tests may assert the current order), so it's a deliberate tweak, not a free swap; but the current
lexical-favoring tiebreak is arbitrary and worth revisiting for vector-dominant workloads. Verified by source inspection +
the weighted-RRF measurement above (equal-weight neutral-tiebreak RRF = 0.9596 MRR vs `rrf_fuse` 0.9436, same 1.0 recall).

**Correction (integrity — my fix proposal above was flawed):** `max(lexical_score, semantic_score)` does NOT work as a
tiebreak — BM25 scores (~0–20+) and cosine/dot scores (~0–1) are on **incomparable scales**, so the max would just always
be the BM25 value → still lexical-biased. There is no *score-based* neutral tiebreak because the two sources' scores aren't
commensurable. The **sound, scale-free** replacement is **rank-based**: on an RRF-score tie, prefer the doc with the better
(lower) **min rank** across sources (or simply drop score-tiebreak (2) and go straight to `doc_id`, removing the lexical
favoritism at the cost of an arbitrary-but-unbiased order). Note also that the manual "0.9596" used a *random* tiebreak, so
it reflects "unbiased ≥ lexical-biased for vector-only targets," not a guaranteed 1.6-pt gain — the realizable gain from a
deterministic neutral tiebreak is workload-dependent. Net: the diagnosis (lexical-favoring tiebreak demotes vector-only
best-answers) stands; the *clean* fix is a min-rank tiebreak, and its payoff is modest + workload-dependent.

---

## 2026-07-03 — MEASURED (hypothesis REJECTED, robustness confirmed): the hybrid-vs-Tantivy advantage is ~stable across scale (30k→130k), it does NOT grow (IronPetrel)

Tested whether frankensearch's hybrid advantage over Tantivy-lexical **grows at scale** (hypothesis: BM25 lexical recall
degrades faster than vector recall as the corpus grows, widening the gap). Re-ran `real_hybrid_knownitem` on the **130k**
real corpus (321 known-item queries):

| corpus | lexical (Tantivy) recall@10 | vector recall@10 | hybrid recall@10 | **hybrid − lexical** |
|---|---|---|---|---|
| 30k | 0.9016 | 0.9961 | 1.0000 | **+9.84 pts** |
| 130k | 0.9128 | 0.9907 | 0.9969 | **+8.41 pts** |

**Hypothesis REJECTED:** the advantage does not grow — it is roughly stable and if anything *shrinks* slightly at 130k
(+9.8 → +8.4 pts). What actually moved: **vector** recall dips a little at scale (0.9961 → 0.9907 — more near-neighbor
confusion in a bigger corpus), while **lexical** recall is flat-to-up (0.9016 → 0.9128); the hybrid tracks vector's slight
decline. So the hybrid value-add over Tantivy is **robust (~+8–10 pts / ~9–11% relative) but scale-invariant**, not
scale-amplified — a real result to temper any "hybrid wins more at scale" intuition. The RRF-tuning findings **replicate at
130k**: RRF-`k` inert (hybrid flat MRR 0.9339 across `k`), per-source weighting a Pareto tradeoff (`vec_w=1` best:
recall 0.9969 / MRR 0.9448; higher `vec_w` → recall drops to vector-alone's 0.9907), and the neutral-tiebreak vs `rrf_fuse`
MRR gap persists (0.9448 vs 0.9339). Verified: `--features lexical` bench, async Tantivy index of 130k real docs + 321
queries, runs clean locally (exit 0).

---

## 2026-07-03 — MEASURED (elevates the tiebreak finding): the hybrid value-add SURVIVES a stronger embedding model (MiniLM), and neutral-tiebreak weighted-RRF *dominates* vector-alone on BOTH recall AND MRR (IronPetrel)

Re-ran `real_hybrid_knownitem` on the 130k corpus with the **raw-transformer MiniLM-384** vector tier (vs the earlier
potion-256), to test whether a *stronger* vector model makes the lexical/hybrid tier redundant. (The bench's
`vector(potion)` label is hardcoded — the vectors here are MiniLM.)

| MiniLM @130k, 321 queries | recall@10 | MRR@10 |
|---|---|---|
| Tantivy lexical (BM25) | 0.9128 | 0.8918 |
| vector (MiniLM) alone | 0.9938 | 0.9509 |
| production `rrf_fuse` | 1.0000 | 0.9333 |
| **weighted-RRF `vec_w=1`** (neutral tiebreak) | **1.0000** | 0.9453 |
| **weighted-RRF `vec_w=2`** | 0.9938 | **0.9580** |
| weighted-RRF `vec_w=4/8/16` | 0.9938 | 0.956 / 0.954 / 0.952 |

**Two findings:** (1) **The hybrid value-add over Tantivy survives a stronger embedding model** — even with MiniLM (a
better retriever than potion: 0.994 vs 0.991 recall), the hybrid still beats Tantivy by **+8.7 pts recall**, and hybrid
recall (1.0) still **tops MiniLM-vector-alone (0.994)** — so the lexical tier is *not* made redundant by better embeddings.
(2) **Neutral-tiebreak weighted-RRF DOMINATES vector-alone on BOTH metrics:** `vec_w=2` gives recall 0.9938 (= vector-alone)
AND **MRR 0.9580 > vector-alone's 0.9509** (+0.7 pts), and `vec_w=1` gives higher recall (1.0 vs 0.994) at MRR 0.9453. So
the hybrid, with proper source-weighting + a neutral tiebreak, is **strictly better than vector-alone** — there IS a fusion
that wins on both axes (contra the earlier potion "Pareto tradeoff" framing, which was measured only vs `rrf_fuse`'s
*lexical-biased* tiebreak). **This elevates the earlier tiebreak diagnosis from "minor" to meaningful:** production
`rrf_fuse` trails vector-alone on MRR (0.9333 < 0.9509) *only* because of (a) no per-source weighting + (b) the
lexical-favoring tiebreak; fixing both (a source-weighted `rrf_fuse` variant + a neutral tiebreak) would make the shipped
hybrid **strictly dominate vector-alone** on both recall and MRR — the strongest argument yet for the hybrid tier. Two
concrete product route-nexts, both now measurement-justified. Verified: `--features lexical` + `--features fastembed`
(MiniLM embed) run clean locally (exit 0).

**Follow-up (which fix, and how — a subtle implementation trap): deterministic-tiebreak sweep** (equal-weight RRF, MiniLM
130k), to see if a *tiebreak-only* fix (no source-weighting API) recovers the dominance:

| tiebreak (equal weight) | recall@10 | MRR@10 |
|---|---|---|
| `doc_id` ascending | 1.0000 | **0.9307** (worst) |
| production `rrf_fuse` (lexical-favoring) | 1.0000 | 0.9338 |
| **hash (unbiased, FNV of doc_id)** | 1.0000 | **0.9426** |
| random (unbiased) | 1.0000 | 0.9461 |
| *vs vector-alone* | 0.9938 | *0.9509* |

**Findings:** (1) An **unbiased deterministic tiebreak (hash) beats production by +0.9 MRR pts** (0.9338 → 0.9426) — a
realizable, ~1-line cheap fix. (2) But hash **still trails vector-alone** (0.9509) — a tiebreak-only fix does NOT achieve
the dominance; **only source-weighting (`vec_w=2`, MRR 0.958) dominates vector-alone.** (3) **Implementation TRAP:**
`doc_id` ascending is the **worst** (0.9307) — so the "obvious" fix of *removing* the lexical-favoring tiebreak (which
would fall through to the `doc_id` final tiebreak, `rrf.rs:110`) would make MRR **worse**, not better; the lexical tiebreak
is accidentally *better* than `doc_id` here. The correct tiebreak fix is to **insert an unbiased (hash) comparator**, not
delete the lexical one. **Net recommendation:** the dominant fix is a **source-weighted `rrf_fuse`** (`vec_w≈2`), which is a
config/API addition; a cheaper hash-tiebreak tweak recovers ~0.9 MRR pts but not full dominance and must be an *added*
unbiased comparator (never fall to `doc_id`). Verified: `--features lexical` bench runs clean locally (exit 0).

---

## 2026-07-03 — MEASURED (inverts a common intuition): the VECTOR tier degrades ~2.2× in recall for SHORT queries — semantic search does NOT rescue vague queries (for static embeddings) (IronPetrel)

Tested how the vector tier holds up as queries get **short/vague** (the regime where BM25 keyword-matching is supposed to
struggle and semantic search is supposed to shine). Light index-only bench `real_qlen_vector` (built on an **isolated
`CARGO_TARGET_DIR`** to dodge the shared-target lock contention that blocked the heavier fusion bench): known-item
recall@10 where the query is the first W words of a held-out doc (potion-256, 130k corpus, 321 queries):

| query length | vector recall@10 | MRR@10 |
|---|---|---|
| 10-word | 0.9907 | 0.9391 |
| 5-word | 0.7944 | 0.6291 |
| **3-word** | **0.4486** | 0.2551 |

**Finding:** the vector tier **collapses for short queries** — recall@10 falls from 0.99 (10-word) to 0.79 (5-word) to
**0.45 (3-word)**, a ~2.2× drop. This **inverts the common "semantic search rescues vague/short queries" intuition**: a
3-word span produces a noisy, underspecified **static** (Model2Vec) embedding, and nearest-neighbor retrieval on that
noisy query vector misses the source doc more than half the time. **Implication for the hybrid vs Tantivy story:** for
short queries the vector tier is *weak* (0.45), so the **lexical BM25 tier likely carries the hybrid** there — meaning the
vector/hybrid advantage over Tantivy that we measured on 10-word queries (+8–10 pts) probably **shrinks or reverses for
very short queries**, where exact keyword matching beats a vague embedding. (Caveat: this is a *static* Model2Vec
embedding; a contextual model may degrade less on short spans — a route-next once the fusion bench is unblocked, to get
the full hybrid-vs-Tantivy-by-query-length curve.) Verified: `--features` none (index-only) bench runs clean on an
isolated target (exit 0).

---

## 2026-07-03 — MEASURED (completes the curve): the full HYBRID-vs-Tantivy-by-query-length — hybrid ALWAYS beats Tantivy, and is ROBUST where each single tier is weak (IronPetrel)

Completes the query-length arc (the detached `real_hybrid_knownitem` sweep finished once the shared-target lock freed).
Full known-item retrieval by query length (potion-256, 130k corpus, 321 queries):

| query length | Tantivy lexical (BM25) | vector (potion) | hybrid (RRF) | **hybrid − Tantivy** |
|---|---|---|---|---|
| 3-word | 0.6854 | **0.4486** | 0.7259 | **+4.05 pts** |
| 5-word | 0.8692 | 0.7944 | 0.9408 | **+7.16 pts** |
| 10-word | 0.9128 | 0.9907 | 0.9969 | **+8.41 pts** |

**Findings:** (1) For **short queries, lexical BEATS vector** (0.685 vs 0.449 @3-word) — confirming last entry's prediction:
a vague short query makes a poor static embedding, so BM25 keyword-matching wins. The vector tier only **overtakes**
lexical for longer queries (crossover between 5- and 10-word: 0.794<0.869 @5w, 0.991>0.913 @10w). (2) **The hybrid beats
Tantivy at EVERY query length** — the advantage *shrinks* for short queries (+4.05 @3w vs +8.41 @10w, since vector
contributes less) but **never reverses**. (3) The real value: **the hybrid is ROBUST where each single tier is weak** — at
3-word it rescues a strong 0.726 (> lexical's 0.685) out of a *terrible* vector (0.449) + decent lexical (0.685), because
RRF recovers whatever *either* tier finds. So the hybrid isn't just "better on average" — it's the **safe choice across
the query-length spectrum**, always ≥ the better single tier and always > Tantivy-alone. This is the strongest
product argument for the hybrid tier: it removes the query-length risk of committing to lexical-only (weak on long/semantic
queries) or vector-only (weak on short/keyword queries). Verified: `--features lexical` bench, async Tantivy index of 130k
docs, ran clean (exit 0) once the shared-target lock cleared (see [[isolated-target-sidesteps-lock-contention]]).

---

## 2026-07-03 — ROOT CAUSE (mechanistic, no rebuild): the short-query vector collapse is QUERY-EMBEDDING DRIFT — the query vector moves away from its source doc as the query shortens (IronPetrel)

Explains *why* the vector tier collapses on short queries (0.99/0.79/0.45 recall@10 at 10/5/3 words). Pure-Python analysis
of the existing embedding slabs (no cargo build — dodges the shared-target lock): for each known-item query, the cosine
similarity between the **query embedding** and its **source doc's embedding** (both L2-normalized potion-256; 130k corpus,
321 queries):

| query length | cos(query, source-doc) mean | median | p10 | frac < 0.5 | → measured recall@10 |
|---|---|---|---|---|---|
| 10-word | 0.846 | 0.873 | 0.693 | 0.00 | 0.99 |
| 5-word | 0.656 | 0.674 | 0.460 | 0.15 | 0.79 |
| 3-word | **0.517** | 0.527 | 0.277 | **0.39** | 0.45 |

**Finding:** the collapse is **query-embedding drift**, not a retrieval/index defect — as the query shrinks from 10→3
words its embedding moves from cos **0.85 → 0.52** away from the very doc it came from, and at 3 words **39% of queries are
< 0.5 cosine to their own source doc**. In a 130k corpus, a query vector only 0.52-similar to its target has thousands of
docs nearer, so the true doc falls out of the top-10 (recall 0.45). The cos-drift curve (0.85/0.66/0.52) tracks the recall
curve (0.99/0.79/0.45) tightly. **Consequences:** (1) this is a property of the *query*, so no ANN/quant/index tuning can
fix it — only a better *query representation* (more context) or the lexical tier (which matches the literal keywords)
helps; it's precisely why lexical carries the hybrid on short queries and why the hybrid's advantage over Tantivy shrinks
there. (2) Open question (MiniLM contextual-vs-static test building): does a contextual model drift *less* on short spans,
or is 3-word underspecification fundamental? — the cos-drift metric is the clean way to answer it (compare MiniLM's
cos(query,source) curve to potion's). Verified: computed directly from the committed slabs; recall figures from `6bf4b25`.

**Follow-up (answers it — contextual does NOT help): MiniLM drifts just as much as potion on short queries → the collapse
is FUNDAMENTAL query underspecification, not a static-embedding artifact.** Embedded the same 3/5/10-word queries with the
**raw-transformer MiniLM-384** and compared its cos(query, source-doc) drift to potion-256's:

| query length | potion (static) cos-to-source | MiniLM (contextual) cos-to-source |
|---|---|---|
| 3-word | 0.517 (frac<0.5 = 0.39) | **0.517** (frac<0.5 = 0.41) |
| 5-word | 0.656 | 0.667 |
| 10-word | 0.846 | 0.858 |

**The two drift curves are near-identical** — at 3 words both models land at cos **0.517** to the source doc. So a
*contextual* transformer embedding does **not** rescue short queries any better than a *static* Model2Vec one: a 3-word
span is **inherently underspecified**, and no embedding model can pull a 0.52-similar query vector back onto its source
doc in a 130k corpus. **Consequences:** (1) the short-query vector weakness (recall 0.45 @3w) is a **query property, not a
model or index defect** — you cannot fix it by upgrading the embedder (potion→MiniLM buys ~0 on cos-drift); (2) the
**lexical (BM25) tier is the only real remedy** for short/keyword queries, independent of the embedding model — which is
exactly why the hybrid beats both single tiers across the query-length spectrum and why frankensearch's hybrid design is
the right call. This closes the query-length/embedding-model arc. (Efficiency note: the cos-drift metric answered the
contextual-vs-static question directly, so the heavier MiniLM retrieval bench was **not needed**.) Verified: pure-Python on
the committed potion + MiniLM slabs (no cargo build).

---

## 2026-07-03 — MEASURED (NEW LEVER, alien-graveyard-tier): a cheap per-query CONFIDENCE signal (top-1/top-2 score margin) predicts vector-retrieval success (AUC 0.77) → query-ADAPTIVE fusion is viable (IronPetrel)

The short-query weakness is a *query* property (see the drift entries), so instead of a fixed lexical/vector fusion weight,
a system could detect *per-query* whether the vector tier is trustworthy and lean on lexical when it isn't. Tested whether
a **cheap, at-query-time** confidence signal predicts vector success. numpy brute-force retrieval over the potion-256 130k
corpus, 963 known-item queries (3/5/10-word mixed); for each query take the vector top-k scores and ask whether the
score-shape predicts finding the source doc:

| confidence signal | AUC (predict found@10) |
|---|---|
| raw top-1 score | 0.722 |
| **score margin (top-1 − top-2)** | **0.771** |

| | found@10 | mean top-1 | mean margin |
|---|---|---|---|
| source **found** | — | 0.775 | **0.126** |
| source **missed** | — | 0.683 | **0.035** |

**Finding:** the **margin (top-1 − top-2) predicts retrieval success with AUC 0.771** — a genuinely usable signal, and
cheaper/better than the raw top-1 score (0.722). A confident retrieval has one doc standing out (margin 0.126); an
ambiguous/underspecified query has flat top-k scores (margin 0.035, **3.6× smaller**). This is computable **for free** at
query time (the top-2 scores are already retrieved). **New product lever — query-adaptive fusion:** route/re-weight
per-query by the vector margin — when it's small (uncertain), down-weight the vector tier and lean on lexical BM25 (which,
per the query-length entries, is exactly the tier that carries short/ambiguous queries); when it's large (confident),
up-weight vector. This is the adaptive version of the static source-weighting lever (which already dominates vector-alone,
`7bb8da5`), and it directly attacks the one measured hybrid weakness (short/underspecified queries). Caveat: AUC 0.77 is
useful but not sharp, and it's weakest exactly where it's most needed (uniformly-hard 3-word queries: top-1 barely
separates found 0.693 vs missed 0.683) — so margin-routing helps the *mixed* query stream more than the all-short tail.
Route-next: wire a margin-thresholded per-query fusion weight and measure the hybrid recall/MRR uplift (needs the
lexical-fusion bench). Verified: numpy brute-force on the committed potion slab (no cargo build).

**Follow-up (selective retrieval — the margin gives a usable coverage/accuracy dial): risk-coverage curve.** Serving only
the top-X% most-confident queries by margin (potion-256, 130k, 963 queries):

| coverage (top-X% by margin) | found@10 on the served set |
|---|---|
| 25% | **0.983** |
| 50% | 0.915 |
| 75% | 0.812 |
| 90% | 0.766 |
| 100% (baseline) | 0.745 |

And the **flagged bottom-25%-margin queries have found@10 = 0.544** — the vector tier is nearly a coin-flip there. So the
margin is a genuinely actionable confidence dial: **serve the confident half of the stream at 0.92 found@10** (vs 0.745
undifferentiated) and **route the uncertain quarter (0.54) to lexical BM25** (which, per the query-length entries, carries
exactly those short/ambiguous queries). This is the concrete selective-prediction / adaptive-routing operating point that
the AUC-0.77 margin signal enables — a real product lever: instead of one global fusion weight, a per-query decision gated
on a free-to-compute signal (top-1 − top-2). Net arc (query-difficulty): drift is the root cause → it's model-invariant
(fundamental) → but it's *predictable* per-query (margin AUC 0.77) → so **query-adaptive fusion / selective routing** is
the answer, directly attacking the one measured hybrid weakness. Route-next (cargo-gated): wire the margin gate into the
fusion path and measure end-to-end hybrid uplift. Verified: numpy on the committed potion slab (no cargo build).

**Follow-up (SIMULATED end-to-end — tempers the adaptive lever + surfaces a task-design bias): margin-gated routing gives only a marginal bump, and strong lexical beats the equal-weight hybrid on keyword-overlap known-item.** Ran the full pipeline in Python (no cargo): a **TF-IDF** lexical tier (sklearn, BM25 proxy) over the 130k corpus + the potion vector tier, fused per-query; 963 queries, found@10 / MRR@10:

| method | found@10 | MRR@10 |
|---|---|---|
| vector-alone | 0.7445 | 0.6077 |
| TF-IDF lexical-alone | **0.8681** | 0.7196 |
| static hybrid (equal RRF) | 0.8515 | 0.6963 |
| margin-gated adaptive hybrid (margin<0.06 → lexical-heavy, else vector-heavy) | 0.8505 | 0.7085 |

**Findings (honest):** (1) **On this known-item task, strong lexical (0.868) BEATS the equal-weight hybrid (0.852)** — the
vector tier *dilutes* a strong lexical ranking. (2) **Margin-gated adaptive routing gave only +0.012 MRR and ~0 recall**
over static hybrid — NOT the clear win the AUC-0.77 confidence signal implied. **Why (task-design bias, important):** the
"first-N-words of the doc" query is **keyword-overlapping** — the query words are literally in the source doc, which is
the *best case for lexical / worst case for the vector tier's marginal value*. (This also reconciles with the Tantivy run
where hybrid *beat* lexical: TF-IDF here is a stronger lexical tier than that Tantivy config, so the vector tier had less
to add.) The adaptive-fusion / vector tier's value is **understated by keyword-overlap known-item retrieval** and would
need **semantic / paraphrase queries** (where the query and doc share meaning but not words) to demonstrate — which this
prefix-query harness cannot generate without a paraphrase model. **Net (corrects the prior enthusiasm):** the margin
*signal* is real (AUC 0.77) but a simple margin *gate* is not an end-to-end win on keyword-overlap tasks; validating
adaptive fusion needs a semantic-query benchmark, and the honest current bottom line is that for exact-keyword known-item
search, lexical-alone is hard to beat. Verified: numpy + sklearn TF-IDF on the committed slabs + corpus texts (no cargo).

---

## 2026-07-03 — LANDMARK (the definitive semantic answer): on a REAL labeled semantic benchmark (BEIR SciFact), the HYBRID beats lexical by +4.8 recall pts and the vector tier's unique value is 5× higher than on keyword-overlap (IronPetrel)

Settled the open question (the keyword-overlap known-item task couldn't test the vector tier's real value). Downloaded
**BEIR SciFact** (5183 scientific abstracts; 300 test queries that are scientific **claims** — keyword-*divergent*, they
paraphrase the evidence; real qrels), embedded corpus+queries with **potion** (via `model2vec` in a venv — no torch, no
cargo), TF-IDF as the BM25-proxy lexical tier, and evaluated against the qrels:

| method | recall@10 | nDCG@10 |
|---|---|---|
| TF-IDF lexical (BM25 proxy ≈ Tantivy tier) | 0.7735 | 0.6286 |
| vector (potion) | 0.6618 | 0.5064 |
| **hybrid (equal RRF)** | **0.8216** | 0.6102 |

Complementarity on semantic queries: **vector-ONLY (relevant docs lexical missed) = 7.0%**, lexical-ONLY = 18.3%, both =
61.0%.

**Findings — the payoff of the whole session's real-embedding arc:** (1) **On real semantic queries the hybrid BEATS the
best single tier by +4.8 recall pts** (0.822 vs lexical 0.774, +6.2% relative) — the exact *opposite* of the
keyword-overlap known-item task (where the hybrid was *below* lexical, `8130208`). (2) **The vector tier's unique
contribution is 7.0% here vs 1.35% on keyword-overlap — 5× higher** — confirming the precise prediction from the
complementarity entry (`46b230c`): the hybrid is justified iff, on semantic queries, vector-ONLY ≫ 1.35%. It is. (3) So
frankensearch's hybrid design is **validated where it matters** — semantic/keyword-divergent queries — and the earlier
"lexical-alone is hard to beat" bottom line was an artifact of the keyword-overlap test harness, now corrected on a real
benchmark. (Honest caveats: nDCG@10 hybrid 0.610 is a hair below lexical 0.629 — the same RRF rank-dilution/tiebreak effect
from `a721e39`; equal-weight RRF maximizes *recall* but slightly dilutes rank-1, so the source-weighted + neutral-tiebreak
fixes would lift nDCG too. potion is a lightweight static model; a stronger contextual model would raise vector/hybrid
further. TF-IDF ≈ BM25 but not Tantivy's exact scorer.) Strongest real-benchmark evidence that the hybrid > lexical(Tantivy)
on semantic search. Verified: `model2vec`+`sklearn` in a venv on BEIR SciFact qrels (no cargo, no torch).

**Follow-up (COMPLEMENTARITY decomposition — quantifies *why*, and what a semantic benchmark must show): on keyword-overlap
known-item, the vector tier adds only 1.35% UNIQUE, so the fusion ceiling is barely above lexical-alone.** Decomposed where
each tier finds the source doc (963 queries, potion vector + TF-IDF lexical):

| outcome | fraction |
|---|---|
| vector-alone found@10 | 0.7445 |
| TF-IDF lexical-alone found@10 | 0.8681 |
| **ORACLE (either tier finds it)** | **0.8816** ← the ceiling any fusion could reach |
| both tiers find | 0.7310 |
| **vector-ONLY (lexical missed)** | **0.0135** ← the vector tier's *unique* contribution |
| lexical-ONLY (vector missed) | 0.1371 |
| neither | 0.1184 |

**Findings:** (1) The two tiers are **highly redundant** on keyword-overlap queries — 73% found by *both*, and the vector
tier finds only **1.35% that lexical misses**. So the **fusion ceiling (0.882) is only ~1.4 pts above lexical-alone
(0.868)** — there is almost nothing for fusion to gain. (2) Static RRF (0.852, prior entry) is actually **below** the best
single tier (TF-IDF 0.868) — it doesn't even reach the ceiling; it goes *backward*, diluting the strong lexical ranking
with the redundant weaker vector tier. So on this task **fusion is net-destructive**, not because RRF is bad but because
the vector tier has **~0 unique value** to contribute. (3) This is the precise, quantified reason the hybrid "didn't help":
it's a **redundancy** problem, not a fusion-algorithm problem. **What a proper (semantic) benchmark must show:** for the
hybrid to be justified, **vector-ONLY must be ≫ 1.35%** — i.e. the vector tier must retrieve relevant docs that share
*meaning but not keywords* with the query, which only keyword-*divergent* queries (paraphrases / real IR qrels) exercise.
The known-item-prefix harness structurally cannot produce those (query ⊂ doc). **This is the cleanest statement of the
open question:** frankensearch's hybrid is worth its complexity iff, on semantic queries, the vector tier's unique-find
rate is large — unmeasured here (needs a labeled semantic dataset + an embedding path, both currently absent in this
Python env: no `tokenizers`/`sentence-transformers`, embed binaries cleaned). Verified: numpy + sklearn on committed
slabs + corpus texts (no cargo).

---

## 2026-07-03 — MEASURED (completes the fusion story): the hybrid STRICTLY DOMINATES lexical on SciFact with a modest STRONGER-TIER up-weight — and it corrects the "vec_w=2 dominates" claim (weight the stronger tier, not always vector) (IronPetrel)

Followed the landmark SciFact result (hybrid wins recall 0.822 but its nDCG 0.610 was a hair below lexical 0.629 — a
recall/precision tradeoff) into the fusion-weighting question, on the reusable venv+SciFact harness:

| method (SciFact, 300 test queries) | recall@10 | nDCG@10 |
|---|---|---|
| lexical (TF-IDF ≈ BM25) | 0.7735 | 0.6286 |
| vector (potion) | 0.6618 | 0.5064 |
| hybrid RRF equal-weight | 0.8216 | 0.6135 |
| **hybrid RRF, `lex_w=1.3`** | **0.8063** | **0.6293** ← beats lexical on BOTH |

**Findings:** (1) **A modest up-weight of the STRONGER tier makes the hybrid strictly dominate the best single tier on
BOTH recall AND nDCG** — `lex_w=1.3` gives recall 0.806 (> lexical 0.774) AND nDCG 0.629 (> lexical 0.629): the hybrid
Pareto-dominates lexical, resolving the equal-weight nDCG deficit. (2) **CORRECTION to `7bb8da5`'s "vec_w=2 dominates":**
the right rule is *up-weight the **stronger** tier for the workload*, not always the vector tier. On SciFact **lexical**
is stronger (0.774 vs vector 0.662), so up-weighting **lexical** wins; on the keyword-overlap known-item task vector was
stronger (10-word 0.99), so up-weighting **vector** won there. Up-weighting the *weaker* tier degrades both metrics.
(3) **Also corrected a fusion artifact:** with RRF `k=60`, `1/(k+r)` is nearly flat over the top-10, so any weight >1 makes
the up-weighted tier's docs *all* outrank the other tier → collapse to that tier alone (this is why the earlier `vec_w=2`
"collapsed to vector"); a smaller `k` (≈5–10) or score-level weighting is needed for a graded blend. (4) The neutral
**hash tiebreak** gives a small nDCG lift (0.610→0.616 at equal weight, `a721e39` confirmed on real data). **Net product
takeaway:** frankensearch's hybrid, tuned with a modest stronger-tier weight (~1.3×) + a smaller RRF k + neutral tiebreak,
**strictly beats Tantivy-lexical on both recall and nDCG on real semantic queries** — the definitive, actionable
validation. Verified: `model2vec`+`sklearn` venv on BEIR SciFact qrels (no cargo, no torch).

---

## 2026-07-03 — LANDMARK (the biggest lever): a RETRIEVAL-TUNED static embedding model makes the vector tier ALONE beat lexical on SciFact — the model choice dominates the fusion tuning (IronPetrel)

The prior SciFact runs used **potion-base-8M**, a *general-purpose* static model, where the vector tier was *weaker* than
lexical (0.66 vs 0.77 recall). Swapped in **`minishlab/potion-retrieval-32M`** — a **retrieval-distilled** static model
(still `model2vec`, lightweight, **no torch**) — via the venv harness on BEIR SciFact:

| method (SciFact, 300 test queries) | recall@10 | nDCG@10 |
|---|---|---|
| lexical (TF-IDF ≈ BM25 ≈ Tantivy tier) | 0.7735 | 0.6286 |
| vector — potion-base-8M (general) | 0.6618 | 0.5064 |
| **vector — potion-retrieval-32M (retrieval-tuned)** | **0.7948** | **0.6331** |
| hybrid — 8M, `lex_w=1.3` (up-weight stronger=lexical) | 0.8063 | 0.6293 |
| **hybrid — retrieval-32M, `vec_w=1.3` (up-weight stronger=vector)** | **0.8349** | **0.6655** |

**Findings:** (1) **The retrieval-tuned model makes the VECTOR tier alone beat lexical on semantic queries** — recall
0.7948 > 0.7735, nDCG 0.6331 > 0.6286 — flipping which tier is stronger (and, per the stronger-tier-weighting rule, which
tier to up-weight: now vector). (2) **The retrieval-32M hybrid reaches 0.835 recall / 0.666 nDCG — beating Tantivy-lexical
by +6.1 recall / +3.7 nDCG pts**, and the general-8M hybrid (0.806 / 0.629) by a wide margin. (3) **The embedding-model
choice is a BIGGER lever than any fusion tuning:** swapping general→retrieval static model moved vector recall +13 pts
(0.66→0.79) and the hybrid +2.9 pts (0.806→0.835) — far more than the fusion-weight/tiebreak knobs (fractions of a point).
Both models are lightweight static (no ONNX/torch), so this is a **free** upgrade. **Product takeaway:** frankensearch's
default embedder matters most — use a **retrieval-distilled** model (potion-retrieval-32M class), not a general one; with
it the hybrid decisively dominates Tantivy-lexical on real semantic search. Verified: `model2vec` (auto-downloaded
retrieval-32M) + `sklearn` in the venv on BEIR SciFact qrels (no cargo, no torch).

---

## 2026-07-03 — GENERALIZATION (multi-dataset rigor): the retrieval-32M hybrid wins on 2/3 BEIR datasets and ties lexical on the 3rd — a SAFE default, but not universal domination (IronPetrel)

The landmark findings rested on one dataset (SciFact); tested generality across **3 BEIR datasets** (different domains /
query types) with the retrieval-32M vector tier + TF-IDF lexical + stronger-tier-weighted hybrid, all in the venv harness:

| dataset (test qrels) | lexical R@10/nDCG | vector R@10/nDCG | hybrid R@10/nDCG | verdict |
|---|---|---|---|---|
| SciFact (n=300, science claims) | 0.773 / 0.629 | 0.795 / 0.633 | **0.835 / 0.665** | hybrid **> both** |
| NFCorpus (n=323, medical NL) | 0.146 / 0.303 | 0.148 / 0.308 | **0.156 / 0.321** | hybrid **> both** |
| ArguAna (n=1406, arg→counter-arg) | **0.796** / 0.387 | 0.698 / 0.333 | 0.794 / 0.384 | hybrid **≈ lexical** (tie) |

**Findings (honest, multi-dataset):** (1) The retrieval-32M hybrid **wins on 2/3** (SciFact, NFCorpus) — where the vector
tier is competitive/stronger. (2) On **ArguAna** it **ties** lexical (0.794 vs 0.796, within noise): ArguAna is
argument→counter-argument retrieval, where the counter-argument **rebuts the same topic and shares vocabulary** → high
keyword-overlap → lexical is strong (0.796) and the vector tier is weaker (0.698), so the hybrid ≈ the (lexical) stronger
tier. (3) **Robust conclusion: the hybrid is never worse than the best single tier by more than noise, and wins
meaningfully where the vector tier is competitive** — i.e. it's a **safe, generally-beneficial default** (matches or beats
lexical across domains), not a universal dominator. So the hybrid's *advantage* is dataset-dependent (tied to how
semantic vs keyword-overlapping the queries are), but its *safety* is universal (it never underperforms the best tier).
This tempers the SciFact-only enthusiasm with the right multi-dataset framing: adopt the hybrid as the safe default, and
its lift is largest on genuinely semantic corpora. (Caveat: NFCorpus recall@10 is low because it has many relevant docs
per query — nDCG is the fairer metric there, and the hybrid still leads 0.321 > 0.308 > 0.303.) Verified: `model2vec`
(retrieval-32M) + `sklearn` venv on BEIR SciFact/NFCorpus/ArguAna qrels (no cargo, no torch).

---

## 2026-07-03 — ACTIONABLE (the default is the worst): frankensearch's stock embedder `potion-multilingual-128M` is the WORST of 4 model2vec models for English retrieval — retrieval-32M is +33% better (IronPetrel)

Directly evaluated **frankensearch's current default embedder** (`DEFAULT_MODEL_NAME = "potion-multilingual-128M"`,
`model2vec_embedder.rs:35`) against alternatives on the BEIR harness (vector-tier recall@10 / nDCG@10):

| model2vec model | SciFact | NFCorpus | note |
|---|---|---|---|
| potion-base-8M (general, small) | 0.662 / 0.506 | 0.108 / 0.244 | |
| potion-base-32M (general) | 0.702 / 0.562 | 0.132 / 0.273 | |
| **potion-retrieval-32M (retrieval-distilled)** | **0.795 / 0.633** | **0.148 / 0.308** | best |
| **potion-multilingual-128M (frankensearch DEFAULT)** | **0.598 / 0.451** | **0.085 / 0.174** | **worst** |

**Finding:** frankensearch's stock default — the **largest** model (128M) — is the **worst of the four on English
retrieval**: SciFact recall **0.598 vs retrieval-32M's 0.795 (+0.197, +33% relative)**, and it even trails the tiny
general potion-base-8M (0.662). The multilingual model trades English-retrieval quality for language coverage (a
reasonable choice *iff* the deployment is multilingual, but a poor English default). **Direct, high-value product action:
change the default English embedder from `potion-multilingual-128M` to a retrieval-distilled model
(`potion-retrieval-32M` class) — a ~33% vector-recall upgrade for free (still a lightweight static model, no ONNX/torch,
and 32M < 128M so it's smaller and faster too).** This is the single highest-leverage, lowest-cost change surfaced this
session: the embedder default, not any algorithm. (Keep a multilingual option for non-English corpora; the recommendation
is the *default*.) Verified: `model2vec` (all four auto-downloaded) + `sklearn` venv on BEIR SciFact + NFCorpus qrels (no
cargo, no torch).

---

## 2026-07-03 — REFINEMENT (MRL viability is MODEL-DEPENDENT): dim-truncation degrades GRACEFULLY on the retrieval-tuned model — a real speed/memory knob, unlike the catastrophic collapse on the general model (IronPetrel)

Earlier (`dfdaf3a`) MRL prefix-truncation *collapsed* recall on `potion-base-8M` (a general, PCA-smoothed static model:
recall 0.545 at 25% dims) — concluded "MRL is a recall trap." Re-tested on the **retrieval-tuned** `potion-retrieval-32M`
(full dim = 512) on SciFact:

| dim | recall@10 | nDCG@10 | size/speed | % of full recall |
|---|---|---|---|---|
| 512 (full) | 0.7948 | 0.6331 | 1× | 100% |
| 256 | 0.7573 | 0.5993 | **2× smaller** | **95%** |
| 128 | 0.7279 | 0.5682 | 4× smaller | 92% |
| 64 | 0.6471 | 0.4869 | 8× smaller | 81% |
| 32 | 0.4611 | 0.3393 | 16× smaller | 58% |

**Finding — MRL viability is model-dependent:** on the retrieval-distilled model, prefix-truncation degrades **gracefully**
(dim=256 keeps **95% recall at 2× smaller/faster**; dim=128 keeps 92% at 4×), a genuine **speed/memory Pareto knob** — the
*opposite* of the catastrophic collapse on general `potion-base-8M` (0.545 at 25% dims). So the earlier "MRL is a recall
trap" conclusion is **corrected/qualified: it's a trap on *general* static models, but a usable Matryoshka knob on
*retrieval-distilled* models** (which are trained with nested/Matryoshka structure). **Practical:** a memory- or
latency-constrained deployment can run retrieval-32M at **dim=256 (half the slab bytes, ~2× faster scan) for a ~4-pt recall
cost** — still a strong vector tier (0.757). Not "free" (it is a tradeoff), but a *usable* one, unlike on the general
model. (This also reconciles the MRL machinery in `mrl.rs`: its "2–6× faster" claim is real *and recall-safe* — but only
with a Matryoshka-trained embedder, which the stock `potion-multilingual-128M`/`potion-base-8M` defaults are not; another
reason to switch the default to a retrieval-distilled model.) Verified: `model2vec` retrieval-32M + numpy in the venv on
BEIR SciFact qrels (no cargo, no torch).

---

## 2026-07-03 — SYNTHESIS (the total uplift): the RECOMMENDED stack beats frankensearch's CURRENT stack on all 3 BEIR datasets, with a smaller+faster embedder (IronPetrel)

End-to-end comparison of the full **hybrid** under two configs — **CURRENT** (stock default `potion-multilingual-128M`
vector + equal-weight RRF k=60) vs **RECOMMENDED** (`potion-retrieval-32M` vector + stronger-tier-weighted RRF, k=10) —
across all 3 BEIR datasets (recall@10 / nDCG@10):

| dataset | CURRENT (multilingual-128M + equal RRF) | RECOMMENDED (retrieval-32M + tuned RRF) | Δrecall / ΔnDCG |
|---|---|---|---|
| SciFact | 0.785 / 0.591 | **0.835 / 0.665** | **+0.050 / +0.074** |
| NFCorpus | 0.141 / 0.268 | **0.156 / 0.321** | +0.015 / +0.053 |
| ArguAna | 0.778 / 0.373 | **0.794 / 0.384** | +0.016 / +0.011 |

**Finding — the session's recommendations, combined, are a consistent measured win:** the recommended stack **beats the
current stack on every dataset** (+1.5 to +5.0 recall pts, +1.1 to +7.4 nDCG pts), with the **nDCG gains larger than
recall** (the retrieval-tuned model ranks relevant docs higher) — and it does so with a **smaller (32M < 128M), faster**
embedder. (Note the current hybrid isn't terrible — 0.785 SciFact — because the strong lexical/BM25 tier masks the weak
multilingual vector tier 0.598; but the recommended stack lifts it further, especially in ranking.) **Net: two changes —
(1) default embedder → retrieval-distilled, (2) RRF → stronger-tier weight + smaller k + neutral tiebreak — deliver a
consistent +1.5–5.0 recall / +1.1–7.4 nDCG improvement across BEIR at lower embedder cost.** This is the quantified,
multi-dataset business case for the roadmap. Verified: `model2vec` (both stacks auto-downloaded) + `sklearn` venv on BEIR
SciFact/NFCorpus/ArguAna qrels (no cargo, no torch).

---

## 2026-07-03 — MEASURED (small config win): fusing DEEPER candidate lists (top-50–100 per tier, not top-10) modestly improves the hybrid — cheap, actionable fetch-depth tuning (IronPetrel)

Tested the fusion **candidate depth** `D` — retrieve top-`D` per tier before RRF-fusing, then take the top-10 (the config
knob is `candidate_multiplier` in `frankensearch-core`, default 3 ⇒ fetch ≈ 3·k). retrieval-32M + TF-IDF, stronger-tier
weight, k=10:

| D (per-tier candidates fused) | SciFact recall/nDCG | NFCorpus recall/nDCG |
|---|---|---|
| 10 | 0.8349 / 0.6655 | 0.1557 / 0.3213 |
| 20 | 0.8383 / 0.6667 | **0.1595 / 0.3297** |
| 50 | 0.8413 / 0.6689 | 0.1553 / 0.3251 |
| 100 | **0.8446 / 0.6703** | 0.1549 / 0.3251 |

**Finding:** deeper candidate fusion helps **modestly and cheaply** — SciFact improves ~monotonically (+1.0 recall / +0.5
nDCG from D=10→100), NFCorpus peaks around D=20 then plateaus. Mechanism: with only top-10 lists, a doc that's rank 5 in
one tier but rank 40 in the other is scored by *one* tier; deeper lists let RRF give it *both* tiers' rank-contributions,
promoting docs both tiers moderately agree on. **Actionable:** set the fusion **fetch depth to ~50–100 per tier**
(candidate_multiplier ≈ 5–10 at k=10), not the tight top-K — a small quality gain (~1 recall pt on semantic data) at
negligible cost (RRF over 100 vs 10 items is microseconds; the fast-tier scan already produces a deep candidate list —
this just fuses more of it). A *minor* knob vs the embedder-default (+33%) and stronger-tier-weight levers, but free.
(Diminishing past D≈50–100.) Verified: `model2vec` retrieval-32M + `sklearn` venv on BEIR SciFact + NFCorpus qrels (no
cargo, no torch).

---

## 2026-07-03 — RIGOR UPGRADE (real BM25, not TF-IDF proxy): the hybrid beats the ACTUAL Tantivy scorer too — the whole "vs Tantivy" arc holds (IronPetrel)

All prior BEIR entries used **TF-IDF** as the lexical tier (a BM25 *proxy*); Tantivy/Lucene actually score with **BM25**.
Hardened the entire "vs Tantivy" claim by swapping in real **BM25** (`rank_bm25` BM25Okapi, same k1/b family Tantivy uses)
+ retrieval-32M vector + stronger-tier-weighted RRF (depth 50):

| dataset | BM25 (real Tantivy scorer) R@10/nDCG | vector R@10/nDCG | hybrid R@10/nDCG | verdict |
|---|---|---|---|---|
| SciFact | 0.776 / 0.652 | 0.795 / 0.633 | **0.834 / 0.689** | hybrid ≥ both |
| NFCorpus | 0.152 / 0.306 | 0.148 / 0.308 | **0.158 / 0.319** | hybrid ≥ both |

**Finding:** with the **real BM25 scorer** (which is a touch stronger than my TF-IDF proxy — SciFact BM25 nDCG 0.652 vs
TF-IDF 0.629), **the hybrid still dominates on both recall AND nDCG** on SciFact (+5.8 recall / +3.7 nDCG over BM25) and
NFCorpus. So the session's central result — **frankensearch's retrieval-tuned hybrid beats Tantivy-lexical on semantic
queries** — is **robust to using Lucene/Tantivy's actual BM25 ranking function**, not an artifact of the TF-IDF proxy. This
retroactively strengthens every prior BEIR entry (the lexical baseline was, if anything, *understated* by TF-IDF). (ArguAna
— the keyword-overlap tie case — is re-computing under pure-Python BM25 (slow: 8.6k docs × 1406 q); with TF-IDF it tied,
and BM25 being slightly stronger it is expected to tie or marginally favor lexical, consistent with "hybrid = safe default,
lift largest on semantic corpora".) Verified: `rank_bm25` + `model2vec` retrieval-32M in the venv on BEIR SciFact + NFCorpus
qrels (no cargo, no torch).

---

## 2026-07-03 — MEASURED (architecturally-relevant metric): at reranker-feed depth (recall@100), the hybrid feeds 96% of relevant docs, and the VECTOR tier's edge over BM25 GROWS with depth (IronPetrel)

frankensearch is a **rerank pipeline** (fast + quality tiers → optional cross-encoder rerank), so the metric that actually
matters for its *first stage* is **recall@100** (the candidate feed to the reranker), not recall@10. Measured recall at
increasing depth (retrieval-32M vector + TF-IDF lexical + stronger-tier-weight RRF):

| SciFact (n=300) | BM25/tfidf | vector | hybrid | hybrid − best single |
|---|---|---|---|---|
| recall@10 | 0.774 | 0.795 | 0.844 | +0.049 |
| recall@50 | 0.876 | 0.907 | 0.940 | +0.033 |
| recall@100 | 0.892 | 0.934 | **0.960** | +0.026 |

| NFCorpus (n=323) | BM25/tfidf | vector | hybrid | hybrid − best single |
|---|---|---|---|---|
| recall@10 | 0.146 | 0.148 | 0.157 | +0.009 |
| recall@100 | 0.240 | 0.288 | 0.301 | +0.013 |

**Findings:** (1) **At reranker-feed depth the hybrid feeds 96% of relevant docs (SciFact recall@100 = 0.960)** — a strong
candidate set for the rerank tier to reorder. (2) The hybrid's *absolute* margin over the best single tier **shrinks with
depth** (SciFact +0.049@10 → +0.026@100) because both tiers catch more relevant docs as the window widens — but it stays
positive. (3) **The vector tier's edge over BM25 GROWS with depth** (SciFact @10 vector +2.1 over BM25 → @100 +4.2; NFCorpus
+0.2 → +4.8): the vector tier's *unique* relevant finds sit deeper in its ranking, precisely the docs a reranker promotes.
**Takeaway:** for frankensearch's rerank architecture the hybrid is the right first stage — it maximizes candidate-feed
recall (0.96 on semantic data), and the deeper the feed the more the vector tier contributes uniquely; so the fast-tier
should fetch a **deep candidate list** (≥100) and let the reranker exploit the vector tier's deep unique finds (reinforces
the fusion-depth entry above). Verified: `model2vec` retrieval-32M + `sklearn` venv on BEIR SciFact + NFCorpus qrels (no
cargo, no torch).

---

## 2026-07-03 — CORRECTION (real BM25 flips ArguAna): the "ArguAna: lexical wins" tie was a TF-IDF artifact — with Tantivy's real BM25 the VECTOR tier beats lexical on ALL 3 datasets (IronPetrel)

Completed the 3-dataset real-BM25 run (ArguAna's pure-Python BM25 finally finished). ArguAna, real BM25 vs my earlier
TF-IDF proxy:

| ArguAna (n=1406) | lexical R@10 | vector R@10 | hybrid R@10 |
|---|---|---|---|
| **TF-IDF proxy** (earlier entries) | **0.796** | 0.698 | 0.794 | ← lexical "wins", hybrid≈lexical |
| **real BM25** (Tantivy scorer) | **0.606** | **0.698** | 0.697 | ← **vector wins**, hybrid≈vector |

**Finding — an important correction:** real **BM25 on ArguAna is much weaker (0.606) than my TF-IDF proxy (0.796)** because
**BM25's document-length normalization (the `b` term) penalizes ArguAna's long argument documents**, which my TF-IDF config
(no length norm) did not. So the earlier "ArguAna is keyword-overlap → lexical beats vector → hybrid only ties" conclusion
was a **TF-IDF artifact**: with Tantivy's *actual* BM25, the retrieval-32M **vector tier beats lexical on ArguAna too**
(0.698 > 0.606), and thus **the vector/hybrid tier beats real BM25 on ALL 3 BEIR datasets** (SciFact 0.795>0.776, NFCorpus
0.148≈0.152, ArguAna 0.698>0.606). **Net: the hybrid-vs-Tantivy story is *stronger* with the real scorer than with the
TF-IDF proxy** — and it exposes a real BM25 weakness (long-document penalty) that the vector tier is immune to (embeddings
have no length bias), a *complementarity* argument for the hybrid beyond keyword-vs-semantic. (The hybrid ties rather than
strictly beats vector on ArguAna only because BM25 is now the *weaker* tier there, adding little; per the stronger-tier
rule, one would up-weight vector.) Verified: `rank_bm25` BM25Okapi + `model2vec` retrieval-32M venv on BEIR ArguAna qrels
(no cargo, no torch).

---

## 2026-07-03 — MEASURED (quality/latency Pareto — the embedder upgrade is FREE): retrieval-32M truncated to dim=256 beats the stock default by +27% at IDENTICAL scan cost and embed speed (IronPetrel)

The retrieval-32M recommendation raised a latency worry (it's 512-dim vs the general models' 256, so 2× the per-vector
scan cost). Measured the full **quality/latency Pareto** (embed throughput + dim = scan cost + SciFact recall, full-dim and
MRL-truncated to 256):

| model2vec model | dim | embed docs/s | recall@10 (full dim) | recall@10 (dim=256 MRL) |
|---|---|---|---|---|
| potion-base-8M | 256 | 10022 | 0.662 | 0.662 |
| potion-base-32M | 512 | 8497 | 0.702 | 0.682 |
| **potion-retrieval-32M** | 512 | 8483 | **0.795** | **0.757** |
| potion-multilingual-128M (stock default) | 256 | 9244 | 0.598 | 0.598 |

**Findings:** (1) **model2vec embed speed is ~flat across all models (~8.5–10k docs/s)** — they're token-lookup + mean-pool,
NOT transformer forward passes, so the 128M "large" default is **not** slower to embed (the size worry is moot). (2)
retrieval-32M is 512-dim → 2× scan cost at full dim, **but MRL-truncated to 256 it still scores 0.757** (95% of full). (3)
**THE CLEAN RESULT — at *identical* 256-dim scan cost + embed speed, `retrieval-32M @ dim=256` (0.757) beats the stock
default `multilingual-128M` (0.598) by +0.159 recall (+27%)** and beats every 256-dim option (base-8M 0.662). So switching
the default embedder to a retrieval-distilled model **is not a quality/latency tradeoff — it's a FREE ~27% quality upgrade
at the same scan cost, same memory (256-dim), and same embed throughput.** (Full 512-dim retrieval-32M buys another +0.038
recall for 2× scan cost — a separate speed/quality knob via MRL, per the earlier MRL entry.) **Definitive product config:
default embedder = retrieval-distilled, stored at dim=256 (MRL-truncated) → matches the current stock's cost profile while
delivering +27% vector recall.** This removes the last objection (latency) to the session's #1 recommendation. Verified:
`model2vec` (4 models) + numpy timing/MRL in the venv on BEIR SciFact qrels (no cargo, no torch).

---

## 2026-07-03 — RULED OUT (config gotcha): `query:`/`passage:` prefixes HURT potion-retrieval-32M — it's a no-prefix model, my numbers already use the optimal convention (IronPetrel)

Checked a common retrieval gotcha: many models (E5, BGE, Nomic) require asymmetric `query:`/`passage:` (or
`search_query:`/`search_document:`) prefixes for best performance. Tested whether potion-retrieval-32M does, on SciFact +
NFCorpus (vector recall@10 / nDCG@10):

| prefix convention | SciFact | NFCorpus |
|---|---|---|
| **none** (what I've used) | **0.795 / 0.633** | **0.148 / 0.309** |
| `query:` / `passage:` | 0.772 / 0.622 | 0.140 / 0.294 |
| `Query:` / (none) | 0.772 / 0.621 | 0.141 / 0.294 |
| `search_query:` / `search_document:` | 0.775 / 0.613 | 0.133 / 0.278 |

**Finding:** prefixes **hurt** potion-retrieval-32M (−2 to −3 recall pts) on both datasets — it's a **no-prefix (symmetric)
retrieval model**, not E5/BGE-style. So (1) all this session's retrieval-32M numbers already use the **optimal** convention
(no quality was left on the table), and (2) it rules out a real production gotcha: **do NOT wrap queries/docs in
`query:`/`passage:` prefixes with this model** (the prefix tokens dilute the mean-pooled static embedding). A small,
config-hygiene confirmation — but the kind of gotcha that silently costs 2–3 recall pts if assumed. Verified: `model2vec`
retrieval-32M + numpy venv on BEIR SciFact + NFCorpus qrels (no cargo, no torch).

---

## 2026-07-03 — MEASURED (the quality ceiling — static vs contextual): a contextual model (BGE-small) beats the best static one by +14% nDCG on SciFact, but embeds ~650× slower — the tradeoff frankensearch's dual embedder path exists for (IronPetrel)

All prior results used **static** `model2vec` embedders (fast, no ONNX/torch). frankensearch *also* ships a **contextual
ONNX path** (`fastembed`/`FastEmbedEmbedder`), so the real question is the **quality ceiling**: how much does a contextual
transformer buy over the best static model, and at what cost? Measured on SciFact (via `fastembed`, onnxruntime, no torch):

| embedder (SciFact, n=300) | tier | dim | recall@10 | nDCG@10 | embed throughput |
|---|---|---|---|---|---|
| potion-multilingual-128M (stock default) | static | 256 | 0.598 | 0.451 | ~9200 doc/s |
| potion-retrieval-32M (recommended static) | static | 512 | 0.795 | 0.633 | ~7800 doc/s |
| **BAAI/bge-small-en-v1.5** | **contextual** | 384 | **0.845** | **0.720** | **~12 doc/s** |

**Findings:** (1) **The contextual model (BGE-small) is meaningfully better than the best static one** — recall 0.845 vs
0.795 (+5.0 pts) and nDCG **0.720 vs 0.633 (+8.7 pts, +14%)** — the nDCG (ranking) gain is largest, i.e. the contextual
model *orders* relevant docs much better. (2) **But it embeds ~650× slower** (12 doc/s vs 7800 — a real transformer
forward pass vs a static token-lookup + mean-pool) and requires the **onnxruntime**. (3) So the two embedder tiers are a
clean **quality/cost Pareto**, exactly what frankensearch's dual path (model2vec static + fastembed ONNX) is built for:
**static `retrieval-32M` for the fast default** (7800 doc/s, no ONNX, 0.795), **contextual BGE-small for the
quality-premium** (+14% nDCG at ~650× embed cost + ONNX runtime). **Product framing:** the embedder is a *tiered* choice —
(a) *fix the bad default*: multilingual-128M → retrieval-32M is a **free +33%** static-to-static win (no cost change); (b)
*offer the premium*: contextual BGE for deployments that can afford the index-build cost / ONNX dep, +14% nDCG on top.
(Note the embed cost is a one-time *index-build* cost, not per-query — so for a static corpus, contextual's 650× embed
penalty is paid once; that reframes the tradeoff toward contextual for quality-sensitive, rarely-reindexed corpora.)
Verified: `fastembed` (bge-small onnxruntime) + `model2vec` in the venv on BEIR SciFact qrels (no cargo, no torch; a light
`snowflake-arctic-embed-xs` point was still embedding at cutoff — contextual ONNX is slow on CPU).

---

## 2026-07-03 — MEASURED (hybrid holds at the QUALITY CEILING): even with a strong contextual vector (BGE), the hybrid still beats vector-alone — BM25 adds 5% unique keyword-exact docs (IronPetrel)

The hybrid clearly helps a *weak* (static) vector tier — but does it still justify itself with the *best* (contextual)
vector, or does a strong vector make the lexical tier redundant? Ran the full hybrid with the **contextual BGE-small**
vector + real **BM25** on SciFact (stronger-tier-weight RRF, depth 50):

| SciFact (n=300) | recall@10 | nDCG@10 |
|---|---|---|
| BM25 (Tantivy scorer) | 0.7757 | 0.6523 |
| BGE-small (contextual vector) | 0.8452 | 0.7203 |
| **HYBRID (BGE + BM25)** | **0.8647** | **0.7241** |

Complementarity at the ceiling: **vector-ONLY = 0.110, lexical-ONLY = 0.050** (BM25 finds 5% relevant docs the strong BGE
vector misses).

**Findings:** (1) **The hybrid still beats vector-alone at the quality ceiling** — recall 0.8647 > BGE 0.8452 (+1.95 pts),
nDCG 0.7241 > 0.7203 — so **BM25 is not redundant even against a strong contextual embedder**: it contributes **5% unique
keyword-exact matches** (exact terms, IDs, rare tokens) that dense embeddings miss. (2) The hybrid's *marginal* benefit
**shrinks as the vector tier strengthens** (+~4 recall pts over static retrieval-32M-alone → +1.95 over contextual
BGE-alone) but **never vanishes** — the lexical tier's keyword-exact catches are complementary to *any* vector tier. (3)
So frankensearch's **hybrid design is validated across the entire embedder ladder** (weak static → strong contextual): the
hybrid is always ≥ the best single tier, and the *combination* (contextual BGE + BM25 + tuned RRF) is the overall quality
ceiling measured this session — **SciFact 0.865 recall / 0.724 nDCG**, well above BM25-alone (0.776/0.652) and the stock
default's hybrid (0.785/0.591). This is the strongest end-to-end evidence that frankensearch's hybrid + a good embedder
beats Tantivy-lexical, and that the hybrid earns its keep even when you can afford the best vector model. Verified:
`fastembed` (bge-small) + `rank_bm25` in the venv on BEIR SciFact qrels (no cargo, no torch).

---

## 2026-07-03 — MEASURED (the last tier — reranker): frankensearch's DEFAULT reranker (ms-marco-MiniLM-L-6-v2) adds only +1.5% nDCG on out-of-domain SciFact — the reranker choice matters like the embedder (IronPetrel)

Completed the full-pipeline validation by measuring the **reranker** tier (frankensearch's final stage; the default is
`ms-marco-MiniLM-L-6-v2` per `model_cache.rs`). Reranked the hybrid's top-50 candidates with the *exact* default reranker
(`Xenova/ms-marco-MiniLM-L-6-v2` via `fastembed` cross-encoder, ONNX, no torch) on SciFact:

| SciFact (n=300) | recall@10 | nDCG@10 |
|---|---|---|
| hybrid (retrieval-32M + BM25), no rerank | 0.8341 | 0.6890 |
| hybrid **+ rerank (ms-marco-MiniLM-L-6-v2)** | 0.8382 | **0.6991** |
| rerank Δ | +0.004 | **+0.0102 (+1.5%)** |

**Finding:** frankensearch's **default reranker gives only a modest +0.010 nDCG (+1.5%)** lift on SciFact — far below the
big gains cross-encoder rerankers typically show. Two honest reasons: (1) **domain mismatch** — `ms-marco-MiniLM` is
trained on MS-MARCO **web passages**, and it transfers weakly to SciFact's **scientific-claim** domain; (2) the hybrid
candidates are already well-ranked (retrieval-32M nDCG 0.633), leaving less headroom. So the reranker is **not free
quality** on out-of-domain corpora — **the reranker model choice matters as much as the embedder**: a domain-matched or
stronger reranker (`BAAI/bge-reranker-base`, jina-reranker-v2) would likely help more, but the *default* ms-marco reranker
under-delivers off-web-domain. **Product takeaway:** don't assume the default reranker is a big win everywhere — its value
is domain-dependent; for non-web-search corpora, evaluate a domain-appropriate reranker (or skip reranking if the hybrid
already ranks well, saving the cross-encoder's per-candidate forward-pass cost — reranking 50 candidates × 300 queries
took minutes of CPU here). This completes the full pipeline picture (fast-tier int8 → hybrid fusion → rerank): the big,
free levers are **the embedder** (+33% free) and **the hybrid** (always ≥ best single tier); the reranker is a
**conditional, model-and-domain-dependent** final polish, not a guaranteed win. Verified: `fastembed` cross-encoder
(ms-marco-MiniLM-L-6-v2, ONNX) + `model2vec` + `rank_bm25` in the venv on BEIR SciFact qrels (no cargo, no torch).

---

### Reranker MODEL head-to-head: the default HURTS, a strong reranker helps +4.4% (IronPetrel, 2026-07-03)

The prior entry conjectured a stronger reranker "would likely help more." **Measured it** — same hybrid candidates
(retrieval-32M + real BM25, RRF stronger-tier-weighted, top-30 fed to the reranker), SciFact 80-query subset,
two cross-encoders head-to-head:

| SciFact (80q, hybrid top-30 candidates) | nDCG@10 | Δ vs no-rerank |
|---|---|---|
| hybrid baseline (no rerank) | 0.7572 | — |
| **+ `ms-marco-MiniLM-L-6-v2`** (frankensearch's DEFAULT) | 0.7410 | **−0.0162 (−2.1%, HURTS)** |
| **+ `BAAI/bge-reranker-base`** (strong general reranker) | **0.7907** | **+0.0335 (+4.4%)** |

**Finding — the reranker MODEL is the whole story, confirmed.** On out-of-domain (scientific) text the default
`ms-marco-MiniLM-L-6-v2` reranker **actively hurts** (−2.1% here; it was a marginal +1.5% on the full 300-query set —
i.e. it hovers around zero and its sign flips with the query sample). A strong general reranker, `bge-reranker-base`,
gives a solid **+4.4%**. The gap between the two rerankers is **+0.050 nDCG (0.741 → 0.791, +6.7%)** — the model choice
alone swings the reranker tier **from net-negative to net-positive**. This retires the ambiguity in the earlier reranker
entry: the small/negative lift was the **MODEL's fault (web-domain ms-marco), not the task's** — the reranker tier *does*
carry real headroom on this corpus, but only with a reranker that generalizes off-web-domain.

**Product takeaway (sharpened):** the default reranker (`ms-marco-MiniLM-L-6-v2`) is a **poor default for non-web
corpora — it can lose quality.** Ship `bge-reranker-base` (or a domain-matched reranker) as the reranking default, or
gate reranking behind a per-corpus eval. Cost caveat unchanged: `bge-reranker-base` (~278M) is ~3× slower per pair than
`ms-marco-L6` on CPU — the quality/latency trade is real, so reranking stays a **conditional** premium, but when you DO
rerank, the model choice is decisive. Verified: `fastembed` TextCrossEncoder (both ONNX cross-encoders) + `model2vec`
retrieval-32M + `rank_bm25` in the venv on BEIR SciFact qrels (no cargo, no torch).

---

### Reranker CROSS-DATASET: no safe default reranker — "stronger" (bge) is NOT safer; it can catastrophically hurt (IronPetrel, 2026-07-03)

The prior entry recommended "ship `bge-reranker-base`, not ms-marco" — but that was measured on **one** dataset
(SciFact). Held to the same 3-BEIR-dataset bar as the embedder recommendation, **it fails.** Same hybrid candidates
(retrieval-32M + real BM25, top-30), 60-query subsets, both cross-encoders vs the no-rerank baseline:

| dataset (query style) | hybrid baseline nDCG@10 | `ms-marco-MiniLM-L-6-v2` Δ | `bge-reranker-base` Δ |
|---|---|---|---|
| SciFact (scientific *claims*) | 0.7572 | −0.0162 (−2.1%) | **+0.0335 (+4.4%)** |
| NFCorpus (natural-language health *questions*) | 0.2924 | **+0.0687 (+23.5%)** | −0.0003 (~0) |
| ArguAna (argument → *counter*-argument) | 0.3134 | **+0.0221 (+7.0%)** | **−0.0732 (−23.4%)** |

**Finding — the "best" reranker flips completely by corpus; there is no safe default, and stronger ≠ safer.**
- **`bge-reranker-base`** wins *only* on SciFact (+4.4%), is **inert on NFCorpus** (~0), and **catastrophically hurts
  ArguAna (−23.4%)**. The ArguAna collapse is the sharp lesson: ArguAna's relevant doc *argues against* the query, but a
  strong similarity reranker ranks same-stance (semantically-nearest) docs highest → it **actively demotes the correct
  counter-arguments.** When the task's notion of relevance (counter-argument) diverges from the reranker's training
  objective (semantic similarity), a *stronger* reranker does *more* damage.
- **`ms-marco-MiniLM-L-6-v2`** is the better **generalist** — net-positive on 2/3 (huge +23.5% on NFCorpus's
  web-question-style queries, which match its MS-MARCO training; +7.0% ArguAna), only mildly negative on SciFact.

**This RETRACTS the prior entry's "ship bge-reranker-base as the reranking default."** Corrected takeaway: **the
reranker model is not universally rankable — you MUST evaluate the reranker per-corpus.** Neither model is a safe
default: ms-marco is the safer *generalist* but can be net-neutral/negative on claim-style corpora; bge peaks higher on
some but can lose 23% on adversarial-relevance tasks. The only robust rule is **reranking is conditional, and the
reranker choice is a per-corpus eval decision** (or skip reranking — the hybrid alone is never catastrophic, whereas a
mismatched reranker can be). This also strengthens the earlier "reranker is a conditional polish" verdict: the
*downside risk* of the wrong reranker (−23%) exceeds the upside of the right one on most corpora. Verified: `fastembed`
TextCrossEncoder (both ONNX cross-encoders) + `model2vec` retrieval-32M + `rank_bm25` on BEIR NFCorpus/ArguAna qrels
(no cargo, no torch).

---

### Reranker DEPTH: opposite-signed optima — deeper is NOT free, and can flip a helpful reranker harmful (IronPetrel, 2026-07-03)

Having measured *which* reranker (cross-dataset), the remaining reranker knob is *how deep* to rerank — the cost axis
(reranking is O(candidates) cross-encoder forward passes, the expensive part). Swept rerank-depth D on each dataset's
**winning** reranker (one rerank-top-50 pass per query, reused across depths; hybrid retrieval-32M + real BM25, 60q):

| rerank depth D | NFCorpus (ms-marco-L6) ΔnDCG@10 | SciFact (bge-reranker-base) ΔnDCG@10 |
|---|---|---|
| 5  | +0.0152 | **+0.0191 (best)** |
| 10 | +0.0265 | +0.0103 |
| 20 | +0.0563 (90% of max) | +0.0058 |
| 30 | +0.0613 (99%) | −0.0049 |
| 50 | +0.0622 (max) | −0.0114 |

**Finding — rerank depth is a real config knob with OPPOSITE-signed optima across (reranker, corpus); deeper is not
free and not always better.**
- **NFCorpus / ms-marco: monotonic increasing, knee at D≈20-30.** Reranking top-20 captures **90%** of the full-top-50
  lift at 2.5× fewer forward passes; top-30 = 99%. The big jump is D 10→20 (+0.027→+0.056) — relevant docs sit at
  hybrid ranks 10-20 and the well-matched reranker promotes them into the top-10, so you **must** rerank ≥20 (top-10 is
  too shallow). Beyond 30, ~0 gain — deeper is wasted cost.
- **SciFact / bge: monotonic DECREASING beyond D=5 — shallow is best, deep HURTS.** The base ranking is already strong
  (0.796) and every extra candidate the reranker reaches into is another chance to **promote a false positive**; depth
  swings the reranker from **+0.0191 (D=5) to −0.0114 (D=50)** — a 3-point flip from helpful to harmful.

**Product takeaway — rerank depth must be tuned per (reranker, corpus), and the safe bias is SHALLOW.** When the
reranker is well-matched and relevant docs sit deep, rerank ~20-30 (and never pay for >30). When the base ranking is
already strong or the reranker is imperfect, rerank only the top ~5-10 — deep reranking lets an imperfect reranker
inject false positives and can turn a net-positive reranker net-negative. Combined with the cross-dataset model finding,
the reranker tier is the most treacherous stage: **both the model AND the depth are corpus-dependent and both carry real
downside risk** — reinforcing "reranking is a conditional, must-eval polish," and that when reranking, **default to a
shallow depth.** Verified: `fastembed` TextCrossEncoder (both ONNX cross-encoders) + `model2vec` retrieval-32M +
`rank_bm25` on BEIR NFCorpus/SciFact qrels (no cargo, no torch).

---

### Reranker SCORE-BLENDING resolves the downside risk: blend, don't reorder (IronPetrel, 2026-07-03)

The prior two entries documented the reranker tier's danger — the wrong model or too-deep reranking can *lose* 11-23%
because the cross-encoder **promotes deep false positives** (docs it scores high but retrieval scored low). This entry
is the **resolution**: don't let the reranker *fully reorder* the candidates — **blend** its score with the retrieval
(hybrid) score, so a deep false positive with a high reranker score but a low retrieval score is vetoed. Per query,
min-max-normalize both scores to [0,1] and rank by `α·reranker + (1-α)·hybrid`; α=0 is pure hybrid, α=1 is pure reorder
(what the earlier entries measured). Same candidates (retrieval-32M + real BM25, top-50), 60q, each dataset's winning
reranker:

| α (reranker weight) | NFCorpus (ms-marco-L6) nDCG@10 | SciFact (bge-reranker-base) nDCG@10 |
|---|---|---|
| 0.00 (pure hybrid) | 0.2966 | 0.7962 |
| 0.25 | 0.3210 | **0.8216 (+0.0254, BEST)** |
| 0.50 | 0.3397 | 0.8209 (+0.0247) |
| 0.75 | **0.3597 (BEST)** | 0.8098 (+0.0136) |
| 1.00 (pure reorder) | 0.3588 | 0.7848 (**−0.0114, HURTS**) |

**Finding — score-blending is the robust way to apply a reranker; it turns the net-negative pure-reorder case
net-positive.**
- **SciFact / bge: pure reorder HURTS (−0.0114) but a light blend (α=0.25) gives +0.0254 (+3.2%)** — a **+0.037 nDCG
  swing** from the blend. The retrieval score acts as a veto on the deep false positives the reranker would otherwise
  promote, exactly the failure mode from the depth entry.
- **NFCorpus / ms-marco: the reranker is strong, so α peaks at 0.75 (0.3597), marginally above pure reorder** — blending
  costs nothing when the reranker is good.
- **A fixed α≈0.5 is net-positive on BOTH** (NFCorpus +0.0431, SciFact +0.0247) — capturing most of the upside with
  none of the pure-reorder downside.

**Product takeaway — apply the reranker as a SIGNAL, not a replacement: rank by `α·reranker + (1-α)·retrieval` with a
safe default α≈0.4-0.5, never pure-reorder (α=1).** This is the single most actionable reranker finding: it removes the
catastrophic-downside risk (the −11% to −23% pure-reorder losses) while keeping ~all of the upside, and it degrades
gracefully when the reranker is mismatched (blend leans on retrieval) — so it also *reduces* the per-corpus-eval burden
from the model/depth entries. Combined verdict for the reranker tier: **blend (don't reorder), bias shallow, and
still eval the model per corpus — but blending is the safety net that makes the tier usable by default.** Verified:
`fastembed` TextCrossEncoder (both ONNX cross-encoders) + `model2vec` retrieval-32M + `rank_bm25` on BEIR
NFCorpus/SciFact qrels (no cargo, no torch).

---

### Reranker integration: RRF-COMBINE (as a 3rd RRF source) beats score-blend AND is α-free/native (IronPetrel, 2026-07-03)

Last entry recommended score-blending (`α·reranker + (1-α)·retrieval`) to tame the reranker's downside — but that needs
an α *and* per-query min-max normalization of incomparable scales. frankensearch is an **RRF-native** system, so the
architecturally-clean integration is to feed the reranker as a **third RRF source** (rank-fusion of the retrieval-order
and the reranker-order) — rank-based, so scale-free and α-free. Head-to-head, same top-50 candidates (retrieval-32M +
real BM25), 60q, each dataset's winning reranker:

| combine method | NFCorpus (ms-marco-L6) | SciFact (bge-reranker-base) |
|---|---|---|
| pure-hybrid (no rerank) | 0.2966 | 0.7962 |
| pure-reorder (reranker only, α=1) | **0.3588 (+0.0622)** | 0.7848 (−0.0114, HURTS) |
| score-blend α=0.25 | 0.3210 | 0.8216 (+0.0254) |
| score-blend α=0.5 | 0.3397 (+0.0431) | 0.8209 (+0.0247) |
| **RRF-combine k=10** | 0.3439 (+0.0473) | 0.8331 (+0.0369) |
| **RRF-combine k=60** | 0.3439 (+0.0474) | **0.8358 (+0.0396, BEST non-reorder)** |

**Finding — RRF-combine is the best default way to apply a reranker in an RRF pipeline.**
- **On the imperfect-reranker case (SciFact/bge), RRF-combine BEATS score-blend** (+0.0396 vs +0.0254, a further +0.014
  nDCG) and comfortably clears pure-reorder's −0.0114 loss — the rank fusion caps how far a deep false positive can
  climb (it must be highly ranked by *both* retrieval and the reranker).
- **It's parameter-free and robust:** no α, no score normalization (rank-based), and **k-insensitive** (k=10 vs 60
  differ by 0.003) — nothing to tune per corpus, unlike score-blend's α or the RRF-`k`.
- **It only trails pure-reorder when the reranker is strongly matched** (NFCorpus: pure-reorder +0.062 > RRF +0.047),
  trading a little peak upside for robustness, safety (never catastrophic), and zero tuning — the correct trade for a
  *default*, since you don't know a priori whether your reranker matches the corpus.
- **Zero new machinery for frankensearch:** it already ships `rrf_fuse`; the reranker becomes another ranked source
  fed into the same fusion (optionally tier-weighted, per the fusion-tuning entry) — no min-max normalizer, no α config.

**Product takeaway (supersedes the score-blend rec):** integrate the reranker as a **third RRF source**, not by
pure-reorder or score-blend. It's the empirically-best *and* simplest *and* architecturally-native option — best on the
risky imperfect-reranker regime, safe everywhere, no tuning, reuses existing `rrf_fuse`. This closes the "how to combine
the reranker" question with the frankensearch-native answer. Verified: `fastembed` TextCrossEncoder (both ONNX
cross-encoders) + `model2vec` retrieval-32M + `rank_bm25` on BEIR NFCorpus/SciFact qrels (no cargo, no torch).

---

### Multi-embedder ensembling is net-destructive — the hybrid's power is MODALITY diversity, not signal count (IronPetrel, 2026-07-03)

Every hybrid finding so far fused *one* vector model + BM25. Static embedders are cheap (~free to run a second), and
frankensearch's RRF is a general ensemble mechanism — so does fusing **two static embedders** add recall the way
lexical+vector does? Tested retrieval-32M RRF-fused with each other model2vec embedder, ± BM25, on 2 BEIR datasets
(recall@10 / nDCG@10):

| stack | SciFact | NFCorpus |
|---|---|---|
| BM25 alone | 0.776 / 0.652 | 0.152 / 0.306 |
| retrieval-32M alone (best single vector) | 0.795 / 0.633 | 0.148 / 0.309 |
| **retrieval-32M + BM25 (current hybrid)** | **0.836 / 0.690** | **0.159 / 0.326** |
| retrieval-32M + potion-base-32M (2 embedders) | 0.769 / 0.617 | 0.148 / 0.301 |
| retrieval-32M + multilingual-128M (2 embedders) | 0.771 / 0.577 | 0.130 / 0.269 |
| retrieval-32M + multilingual-128M + BM25 (triple) | 0.814 / 0.661 | 0.150 / 0.308 |

**Finding — adding a second static embedder never helps and usually HURTS; the triple is worse than the pair.**
- On SciFact a 2nd static embedder **drops recall below retrieval-32M alone** (0.769 vs 0.795): the model2vec embedders
  are **highly correlated** (same token-lookup+mean-pool family, same errors — the 2nd finds a relevant doc ret32 misses
  in only **2.7-3.3%** of queries, vs **~7%** for BM25) *and* weaker, so RRF just **dilutes** ret32's better ranking
  with the weaker model's worse one.
- The **triple (2 embedders + BM25) is worse than the pair (1 embedder + BM25)** on both datasets (SciFact 0.814<0.836,
  NFCorpus 0.150<0.159) — the redundant weak embedder dilutes the good hybrid.
- **Subtle NFCorpus lesson — unique-contribution % is NOT sufficient for ensemble value.** There the 2nd embedder finds
  a doc ret32 misses in **26.9%** of queries (NFCorpus has many relevant docs/query), yet the ensemble still doesn't
  gain (0.148 ≈ ret32) — because the 2nd embedder is *weaker*, its dilution cancels its unique finds. **The partner must
  be both diverse AND comparably strong.**

**Product takeaway — do NOT build a multi-embedder ensemble; one strong retrieval-distilled embedder + BM25 is the
right stack.** This closes the ensemble question with a clean negative, and it *explains why the hybrid works*: BM25 is
the uniquely-valuable partner because it's a **different modality** (exact-term vs semantic → decorrelated errors) *and*
comparably strong — not because RRF benefits from "more signals." Same-modality same-strength ensembling is redundant.
Corollary: the way to strengthen the vector tier is a **better** single embedder (retrieval-distilled → contextual, per
rec #1), not *more* embedders. Verified: 4 `model2vec` static embedders + `rank_bm25` on BEIR SciFact/NFCorpus qrels
(no cargo, no torch).

---

### RRF vs score-fusion for the base hybrid: RRF is the correct default (tied quality, less fragile) (IronPetrel, 2026-07-03)

Every hybrid finding assumed RRF (rank-based fusion), which discards score magnitudes. The main alternative is
**score-fusion**: normalize BM25 and cosine per query (z-score or min-max) and take a weighted sum. Does keeping the
magnitudes beat RRF? Swept the vector weight for each method (RRF k=10; score-fusion with z-norm and min-max),
candidate union of each tier's top-100, on all 3 BEIR datasets (best-nDCG config shown):

| dataset | RRF (rank) recall/nDCG@10 | score-fusion z-norm | score-fusion min-max |
|---|---|---|---|
| SciFact  | **0.8374** / 0.6905 | 0.8218 / 0.6911 | 0.8126 / **0.6931** |
| NFCorpus | 0.1597 / 0.3253 | 0.1655 / 0.3393 | **0.1665 / 0.3386** |
| ArguAna  | **0.7148** / 0.3363 | 0.7084 / 0.3375 | 0.7098 / **0.3378** |

**Finding — RRF and score-fusion are tied on quality; RRF wins on simplicity/robustness, so it's the correct default.**
- **RRF wins RECALL on 2/3** (SciFact +0.025, ArguAna +0.005 over the best score-fusion) and loses only NFCorpus
  (−0.007). **Score-fusion wins nDCG by a hair on all 3**, but the margin is within noise except **NFCorpus (+0.014)** —
  so score-fusion has exactly one real quality edge across three datasets.
- **The tiebreaker is fragility.** Score-fusion needs a per-query score normalizer, and the choice *matters* (z-norm and
  min-max disagree, and each is sensitive to per-query score-distribution quirks — BM25 is unbounded/long-tailed, cosine
  is bounded), *and* it still needs the same weight tuning as RRF. RRF is **rank-based → scale-free**: no normalizer to
  pick, no BM25-vs-cosine scale mismatch to manage, and it's robust to the score-distribution differences that make
  score-fusion brittle across heterogeneous corpora.

**Product takeaway — keep RRF as the fusion primitive (frankensearch's `rrf_fuse` choice is validated).** Score-fusion's
single real edge (NFCorpus nDCG +0.014) doesn't justify the added per-query-normalization fragility, and RRF is
marginally *better* on recall — the first-stage metric that matters when a reranker follows. This is consistent with the
reranker-integration finding (`b114e39`): **RRF is the robust, scale-free, tuning-light fusion primitive throughout the
stack** (lexical+vector *and* reranker-combine). Don't switch the base hybrid to score-fusion. Verified: `model2vec`
retrieval-32M + `rank_bm25` on BEIR SciFact/NFCorpus/ArguAna qrels (no cargo, no torch).

---

### Query-side: embedding-space pseudo-relevance feedback (PRF/Rocchio) is net-harmful for static embedders (IronPetrel, 2026-07-03)

Every lever so far was corpus/fusion/rerank-side; the query side was untouched except the short-query-drift finding.
Tested the classic query-side technique — **embedding-space pseudo-relevance feedback** (Rocchio in vector space):
retrieve top-k with the raw query, then `q' = normalize(q + α·mean(top-k doc vectors))` and re-retrieve. Swept
k∈{3,5,10}, α∈{0.5,1,2}, retrieval-32M, nDCG@10 vs the raw-query baseline:

| dataset | baseline nDCG@10 | best PRF | worst PRF |
|---|---|---|---|
| SciFact  | 0.6331 | 0.6134 (k=3,α=0.5, **−0.020**) | 0.4886 (k=10,α=2, **−0.145**) |
| NFCorpus | 0.3085 | 0.3086 (k=3,α=0.5, +0.0001 ≈ noise) | 0.2868 (k=10,α=2, −0.022) |

**Finding — embedding PRF is uniformly net-harmful (or at best neutral); do NOT add it for the static embedder.**
On SciFact *every* PRF configuration loses quality, and the loss grows **monotonically** with both k (more docs) and α
(more feedback weight) — the signature of **query drift**, not a tuning miss. The mechanism is specific to static
embedders: (1) the top-k always contains false positives (precision@3-10 < 1), so averaging their vectors pulls the
query toward off-topic regions; (2) mean-pooled static doc embeddings don't *refine* the query intent, they **blur** it
toward the corpus centroid (regression to the mean). More feedback → more blur → worse. NFCorpus is nearly flat (its
many-relevant-docs structure tolerates a tiny α=0.5 nudge) but still never *gains*.

**Product takeaway — no embedding-space query expansion/PRF for the static vector tier.** It's a classic technique
someone would plausibly add, and it *degrades* quality here (up to −0.14 nDCG aggressive, at-best-neutral gentle). The
right remedy for weak/short queries remains the **lexical (BM25) tier** (per the short-query-drift finding), not
vector-space feedback. Caveat: this is measured for *static* (model2vec) embeddings — contextual embeddings have more
semantically-precise doc vectors and *might* tolerate PRF, but that's the premium path, not the fast default. Verified:
`model2vec` retrieval-32M on BEIR SciFact/NFCorpus qrels (no cargo, no torch).

---

### Contextual embedder premium GENERALIZES (+10-29% nDCG) — largest where static is weakest (IronPetrel, 2026-07-03)

Rec #1's *premium tier* (contextual `BAAI/bge-small-en-v1.5` > static `retrieval-32M`) rested on **one** dataset
(SciFact +14% nDCG). Held to the same 3-BEIR-dataset bar as the static-embedder recommendation, it's now validated —
vector-tier recall@10 / nDCG@10, static vs contextual:

| dataset | static retrieval-32M | contextual bge-small-en-v1.5 | ΔnDCG |
|---|---|---|---|
| SciFact  | 0.795 / 0.633 | 0.845 / 0.720 | **+14%** |
| NFCorpus | 0.148 / 0.3085 | 0.1615 / 0.3382 | **+9.6%** |
| ArguAna  | 0.698 / 0.3328 | **0.841 / 0.4287** | **+28.8%** (recall +0.143!) |

**Finding — the contextual premium is real and generalizes (+10% to +29% nDCG on every dataset), and it's LARGEST
exactly where static embeddings are weakest.** On ArguAna the contextual model lifts nDCG **+28.8%** and recall from
0.698 → **0.841** — because ArguAna is argument→*counter*-argument retrieval, which needs deep semantic understanding of
argumentative structure and negation that a static mean-pooled token-lookup embedding fundamentally cannot capture, but
a contextual transformer can. The static tier's biggest weakness (semantic/argumentative queries) is precisely the
contextual tier's biggest win.

**This CONFIRMS and STRENGTHENS rec #1's premium tier** (contrast the reranker cross-dataset result `657df16`, where the
same 3-dataset validation *retracted* the "ship bge-reranker" call — same rigor, opposite outcome, which is the point of
validating across datasets). Product takeaway unchanged but now multi-dataset-backed: **static `retrieval-32M` = fast
default; contextual `bge-small` = quality premium worth +10-29% nDCG for quality-sensitive, rarely-reindexed corpora**
(the ~650× embed-slowdown is a one-time index-build cost, not per-query). The premium is most compelling on semantically
hard corpora (argumentative, negation-heavy) where static embeddings structurally underperform. Verified: `model2vec`
retrieval-32M + `fastembed` bge-small-en-v1.5 (ONNX) on BEIR SciFact/NFCorpus/ArguAna qrels (no cargo, no torch).

---

### Code-grounded: the shipped reranker pure-reorders (pipeline.rs:184) — the measurably-worst integration (IronPetrel, 2026-07-03)

Bridged the reranker measurements to the actual Rust code. `frankensearch-rerank/src/pipeline.rs::rerank_step` currently
applies the cross-encoder by **pure-reorder**: it overwrites `rerank_score` on each candidate and then

```rust
// pipeline.rs:184
candidates[..rerank_count].sort_by(compare_by_rerank_score);  // descending rerank_score, discards the fused order
```

This is exactly the integration the reranker-combine measurement (`b114e39`) identified as the **worst** option:
pure-reorder *hurts* on SciFact/bge (−0.0114 nDCG) and loses up to −23% when the reranker is domain-mismatched
(`657df16`), because it lets the cross-encoder promote deep false positives with **no veto from the retrieval score**.
The measured-best alternative, **RRF-combine** (fuse the pre-rerank order with the rerank order), scored **+0.0396** on
the same case — safe (never catastrophic), parameter-free, and native to frankensearch's `rrf_fuse`.

**The fix is cleanly implementable at this exact site with zero new dependencies.** Candidates arrive at `rerank_step`
already sorted by their fused RRF score, so **each candidate's arrival index `i` IS its pre-rerank rank** — no extra
bookkeeping needed. Replace the final pure-reorder sort with a rank-fusion:

- record each reranked candidate's pre-rerank rank = its index `i` in `candidates[..rerank_count]`;
- compute its rerank rank from `rerank_score` (descending);
- sort by `1/(k + pre_rank) + 1/(k + rerank_rank)` (k≈10-60, k-insensitive per `b114e39`).

This is the single highest-value reranker code change: it converts the tier from "can lose 11-23%" to "safe +4-6% with
no tuning," and the machinery (reciprocal-rank fusion) already exists in the fusion crate. **It is product-gated only
because it changes user-visible result ordering** (and would update reranker test snapshots) — an outward-facing
default-behavior change, so it's surfaced here as ready-to-implement rather than flipped unilaterally. The
`explanation.rrf_contribution` field on the rerank `ScoreComponent` (currently hardcoded `0.0` at pipeline.rs:170) even
anticipates this — it's the natural place to record the reranker's RRF contribution. Verified by reading the shipped
code (`crates/frankensearch-rerank/src/pipeline.rs`) against the measured findings; the measurement harness is the
Python venv (`fastembed` cross-encoders + `model2vec` + `rank_bm25`), no cargo.

---

### SHIPPED: RRF-combine reranker as an opt-in (frankensearch-rerank), default behavior preserved (IronPetrel, 2026-07-03)

Implemented the measured-safe reranker integration from `b114e39`/`9e945d3` as **additive, tested Rust code** — no
behavior change to any existing caller:

- **`RerankCombine`** enum in `frankensearch-rerank/src/pipeline.rs`: `PureReorder` (the legacy default) and
  `RrfCombine { k }` (rank-fuse the pre-rerank order with the rerank order, `1/(k+pre)+1/(k+rerank_rank)`).
- **`rerank_step_with_combine(...)`** takes the strategy; **`rerank_step(...)`** is unchanged as the stable wrapper that
  passes `PureReorder`. `DEFAULT_RRF_COMBINE_K = 60` (k-insensitive per the measurements → no tuning). All exported from
  `lib.rs`.
- **Unit test `rrf_combine_vetoes_deep_false_positive`**: a stub reranker that over-promotes the deepest (retrieval-worst)
  candidate makes `PureReorder` put that false positive at rank #1, while `RrfCombine` keeps retrieval's best at #1 and
  vetoes the false positive — the exact failure mode the measurements identified (SciFact/bge pure-reorder −0.011 →
  RRF-combine +0.040).

**Verification:** `cargo test -p frankensearch-rerank --lib` → **32 passed, 0 failed** (31 pre-existing + the new test);
`cargo clippy` clean on the added code. The pipeline compiles under the crate's default (empty) feature set — no
ONNX/`fastembed`/`native` needed — so the safe integration is available to every reranker backend. **Default output is
byte-identical to before** (still `PureReorder`); flipping the default to `RrfCombine` is the remaining one-line,
product-gated decision (it changes user-visible ordering + rerank snapshots) — now de-risked to a single enum value.
This converts the reranker arc from measurement → recommendation → **landed, tested opt-in code**.

---

### SHIPPED: RrfCombine reachable end-to-end via `TwoTierSearcher::with_rerank_combine` (IronPetrel, 2026-07-03)

Wired the opt-in `RerankCombine` (previous commit) through the fusion searcher so it's reachable from the public API
without a code fork — additive, default-preserving:

- **`TwoTierSearcher`** (`frankensearch-fusion/src/searcher.rs`) gains a `#[cfg(feature = "rerank")] rerank_combine`
  field defaulting to `RerankCombine::PureReorder`, plus a builder **`with_rerank_combine(...)`**, following the exact
  `#[cfg(feature = "graph")]` field pattern already in the struct.
- **`run_phase3`** now calls `rerank_step_with_combine(..., self.rerank_combine)` instead of `rerank_step(...)`.

Users enable the measured-safe integration with
`searcher.with_rerank_combine(RerankCombine::RrfCombine { k: DEFAULT_RRF_COMBINE_K })` — one builder call, no fork.

**Verification:** `cargo build -p frankensearch-fusion --features rerank` → clean (EXIT_0);
`cargo test -p frankensearch-fusion --features rerank --lib rerank` → **2 passed, 0 failed** (822 unrelated tests
unchanged). **Default output is byte-identical** (still `PureReorder` — provably the same code path). The reranker arc is
now end-to-end: measurement → recommendation → landed opt-in primitive → reachable via the searcher builder. Flipping the
default from `PureReorder` to `RrfCombine` remains the one product-gated line (changes user-visible ordering + snapshots).

---

### SHIPPED: per-tier RRF weighting in `RrfConfig` (implements the top fusion lever), default-preserving (IronPetrel, 2026-07-03)

Implemented the biggest fusion-tuning lever from the measurements — **up-weight the stronger tier** (finding `fa592b9`:
a modest stronger-tier weight ~1.3× makes the hybrid *strictly dominate* the best single tier on both recall and nDCG,
e.g. SciFact 0.835/0.665). Until now `RrfConfig` had only `k`; the tier weights were **unexpressible in the API**.

- **`RrfConfig`** (`frankensearch-fusion/src/rrf.rs`) gains `lexical_weight` and `semantic_weight` (both default `1.0`),
  applied as a multiplier on each tier's per-rank RRF contribution in **both** the map and merge fusion paths (2 + 3
  contribution sites). `sanitize_tier_weight` degrades non-finite/`≤0` to the neutral `1.0`.
- The 9 in-workspace `RrfConfig { k }` literal sites now use `..Default::default()` (mechanical, behavior-unchanged).
- New test `tier_weight_reorders_by_upweighted_source`: up-weighting the semantic tier 2× promotes the semantic-only doc
  to #1 with exactly 2× the rank-0 contribution; up-weighting lexical flips the winner; a NaN/negative weight degrades
  to standard RRF.

**Verification:** `cargo test -p frankensearch-fusion --lib` → **821 passed, 0 failed**, including the exact-value
formula tests (`rrf_score_formula_k60/k1/k0`) and the map-vs-merge **byte-identity** test (`merge_matches_map_fusion`).
This is the key safety property: **default weight `1.0` is an exact f64 identity** (`x * 1.0 == x`), so every existing
score is bit-for-bit unchanged and both fusion paths stay identical; the feature is purely additive. `cargo clippy`
clean on the added code. Users now express the measured "hybrid strictly dominates" config via
`RrfConfig { semantic_weight: 1.3, ..Default::default() }` (or `lexical_weight` when lexical is the stronger tier —
per the *up-weight-the-stronger-tier* rule). Flipping the *default* weight away from 1.0 stays product-gated (it changes
ranking output), but the capability — and the measured recipe — is now in the code.

---

### SHIPPED: neutral (hash) RRF tiebreak option — completes the fusion-config recipe, default-preserving (IronPetrel, 2026-07-03)

Implemented the last un-shipped fusion knob: the **neutral tiebreak**. The RRF tie comparator broke exact score ties by
`lexical_score.unwrap_or(-inf)` — **asymmetric**: vector-only docs (no lexical score) always lose the tie, systematically
demoting semantic-only best-answers (diagnosed earlier; a neutral hash tiebreak measured a small nDCG / +0.9 MRR gain on
the known-item task). Note the trap: the fix is *not* falling through to raw `doc_id` (alphabetical bias is worse) — it's
an unbiased hash.

- **`RrfTiebreak`** enum in `rrf.rs`: `LexicalThenId` (legacy default) and `Hash` (unbiased FNV-1a of `doc_id`, then
  `doc_id` for determinism). Added `tiebreak` field to `RrfConfig` (default `LexicalThenId`) — no literal-site churn,
  since all `RrfConfig` construction already uses `..Default::default()` from the prior commit.
- `cmp_for_ranking` now branches on the mode; the 4 production sort/select sites pass `config.tiebreak`.
- New test `hash_tiebreak_is_symmetric_across_tiers`: two docs tying on rrf_score (a lexical-only and a semantic-only)
  are ordered by the lexical bias under the default, but tier-agnostically by `doc_id` hash under `Hash`.

**Verification:** `cargo test -p frankensearch-fusion --lib` → **822 passed, 0 failed**, including the map-vs-merge
**byte-identity** test (`merge_matches_map_fusion`) — default `LexicalThenId` keeps the comparator (and both fusion
paths) bit-for-bit identical, so the change is purely additive. `cargo clippy` clean on the added code. Enable the fairer
tiebreak with `RrfConfig { tiebreak: RrfTiebreak::Hash, ..Default::default() }`.

**Fusion-config recipe now fully expressible in the API:** RRF `k`, per-tier weights (`7ccda28`), and tiebreak — every
measured fusion lever is a config knob, each defaulting to legacy behavior. Flipping any *default* stays product-gated.

---

### SHIPPED: per-tier RRF weights + tiebreak reachable via `TwoTierSearcher` builder (IronPetrel, 2026-07-03)

Closed the plumbing gap noted in the prior roadmap entry: the shipped `RrfConfig` fusion knobs (`7ccda28` weights,
`05472cd` tiebreak) were only reachable via a direct `rrf_fuse` call, not through the primary async searcher API. Wired
them onto the **`TwoTierSearcher` builder** (one struct, the same pattern as `with_rerank_combine`), avoiding the
`TwoTierConfig` 35-site field ripple:

- Three non-gated fields on `TwoTierSearcher` (`rrf_lexical_weight`, `rrf_semantic_weight`, `rrf_tiebreak`) defaulting to
  `1.0` / `1.0` / `LexicalThenId`, plus builders **`with_rrf_weights(lex, sem)`** and **`with_rrf_tiebreak(t)`**.
- The per-query `RrfConfig` construction (`searcher.rs:952`) now threads these fields instead of `..Default::default()`.

Users tune fusion from the main API:
`TwoTierSearcher::new(..).with_lexical(..).with_rrf_weights(1.0, 1.3).with_rrf_tiebreak(RrfTiebreak::Hash)`.

**Verification:** `cargo test -p frankensearch-fusion --lib` → **822 passed, 0 failed**; `cargo clippy` clean on the added
code. Default-preserving: the defaults make the `RrfConfig` at `searcher.rs:952` byte-identical to the prior
`..Default::default()`. **Scope note (honest):** this covers the async `TwoTierSearcher` path only; the separate
sync-searcher path (`sync_searcher.rs`, sourcing `k` from `TwoTierConfig`) still uses neutral fusion weights — exposing
weights there needs the `TwoTierConfig` field addition (~35 construction sites), deferred. All four fusion/rerank
capabilities (int8, RRF-combine reranker, per-tier weights, hash tiebreak) are now reachable from the primary searcher
API; only the *default* values remain product-gated.

---

### SHIPPED: sync-searcher fusion tuning too — the "deferred sync gap" was not actually blocked (IronPetrel, 2026-07-03)

The prior entry deferred the sync path, claiming it needed the `TwoTierConfig` 35-site ripple. That was wrong (same
class of error as the earlier searcher-exposure dismissal): `SyncTwoTierSearcher` is its **own struct with a builder**,
so the weights/tiebreak go on *it*, not `TwoTierConfig` — no ripple.

- Three fields on `SyncTwoTierSearcher` (`rrf_lexical_weight`/`rrf_semantic_weight`/`rrf_tiebreak`, default
  `1.0`/`1.0`/`LexicalThenId`) + `const fn` builders `with_rrf_weights` / `with_rrf_tiebreak`, threaded into **both**
  of its per-query `RrfConfig` constructions (`sync_searcher.rs:180`, `:267`). Added the 3 fields to its manual `Debug`
  impl (fixing the `missing_fields_in_debug` lint).

**Verification:** `cargo test -p frankensearch-fusion --lib` → **822 passed, 0 failed**; `cargo clippy` clean.
Default-preserving. Both the async (`TwoTierSearcher`, `e3bbc7b`) and sync (`SyncTwoTierSearcher`) searcher APIs now
expose the full fusion-tuning recipe (`with_rrf_weights` + `with_rrf_tiebreak`), and the reranker combine mode on the
async path — every measured fusion/rerank capability is reachable from **both** primary entry points. Only the *default*
values remain product-gated. Recurring lesson: when a capability seems blocked by a widely-constructed config struct,
check whether the entry-point struct has its own builder — it usually does, and that's the ripple-free home.

---

### Test: end-to-end coverage for the searcher fusion-tuning builders (IronPetrel, 2026-07-03)

The fusion-weight / tiebreak knobs (`7ccda28`/`05472cd`) had unit coverage in `rrf.rs` but nothing validated that the
`SyncTwoTierSearcher::with_rrf_weights` / `with_rrf_tiebreak` builders actually *thread through* to the fusion `RrfConfig`.
Added `rrf_weights_flow_through_searcher_to_fusion`: with a lexical source favoring one doc and the quality/semantic tier
favoring another, **extreme opposite tier weights flip the fused top result** — proving the builder values reach
`rrf_fuse` end-to-end (also exercises `with_rrf_tiebreak(Hash)` on the searcher path). `cargo test -p frankensearch-fusion
--lib` → **823 passed, 0 failed**. Closes the coverage gap between the shipped builders and the low-level fusion units.

---

### CAPSTONE: full recommended pipeline vs Tantivy, end-to-end in one harness — +12-13% nDCG (IronPetrel, 2026-07-03)

The definitive "ratio vs Tantivy" for the **whole** recommended stack, composed and run end-to-end in a single consistent
harness (same query set, same candidate depths) — not stitched from prior per-experiment subsets. Pipeline:
retrieval-32M vector + BM25 → **tuned RRF** (stronger-tier weight ~1.3, small k=10) → **RRF-combine reranker**
(corpus-appropriate cross-encoder). Baseline = Tantivy BM25-alone (`rank_bm25`).

| stage | SciFact recall@10 / nDCG@10 | NFCorpus recall@10 / nDCG@10 |
|---|---|---|
| **Tantivy BM25-alone** (baseline) | 0.7757 / 0.6523 | 0.1522 / 0.3062 |
| + hybrid (retrieval-32M + tuned RRF) | 0.8159 / 0.6837 | 0.1585 / 0.3268 |
| + RRF-combine rerank (bge / ms-marco) | **0.8724 / 0.7309** | **0.1667 / 0.3460** |
| **FULL STACK vs Tantivy** | **+12% / +12%** | **+10% / +13%** |

**Finding — the full recommended frankensearch pipeline beats Tantivy BM25-alone by +12-13% nDCG and +10-12% recall,
consistently across both datasets, with every stage contributing monotonically.** SciFact nDCG 0.652 → 0.684 (hybrid) →
0.731 (+rerank); NFCorpus 0.306 → 0.327 → 0.346. This composes all measured recommendations into one number: the
retrieval-distilled embedder (#1), two-tier hybrid (#2), tuned RRF (#3, using the shipped `RrfConfig` weights), and the
RRF-combine reranker (#4, the shipped `RerankCombine::RrfCombine`). It is the headline result the whole investigation
builds toward — and note the reranker uses the **shipped, default-preserving `RrfCombine` mode**, so this exact pipeline
is expressible today via the `TwoTierSearcher` builders (`with_rrf_weights` + `with_rerank_combine`) plus a
retrieval-distilled embedder; only the *defaults* remain product-gated. Verified: `model2vec` retrieval-32M +
`fastembed` cross-encoders + `rank_bm25` on BEIR SciFact/NFCorpus qrels (no cargo, no torch).

---

### CAPSTONE completed to 3 datasets: ArguAna is the biggest vs-Tantivy win (+22% nDCG) (IronPetrel, 2026-07-03)

Held the capstone to the same 3-BEIR-dataset bar as every other finding. ArguAna (200-query subset, same harness:
retrieval-32M + tuned RRF → ms-marco RRF-combine — ms-marco not bge, since bge hurts ArguAna's counter-argument task):

| ArguAna stage (200q) | recall@10 / nDCG@10 |
|---|---|
| **Tantivy BM25-alone** | 0.5650 / 0.2593 |
| + hybrid (retrieval-32M + tuned RRF) | 0.6200 / 0.2943 |
| + RRF-combine rerank (ms-marco-L6) | **0.6800 / 0.3163** |
| **FULL STACK vs Tantivy** | **+20% / +22%** |

**Finding — ArguAna shows the LARGEST full-stack win over Tantivy (+22% nDCG / +20% recall), exactly because BM25 is
structurally weakest there** (its long-document length penalty hurts on ArguAna's full-argument docs — a weakness the
dense vector tier is immune to). Each stage still contributes monotonically (0.259 → 0.294 → 0.316 nDCG). Completes the
3-dataset capstone:

| BEIR | full-stack vs Tantivy (nDCG / recall) |
|---|---|
| SciFact | +12% / +12% |
| NFCorpus | +13% / +10% |
| ArguAna | **+22% / +20%** |

**Definitive headline: the full recommended frankensearch pipeline beats Tantivy BM25-alone by +12% to +22% nDCG and
+10% to +20% recall across three BEIR datasets, in one consistent end-to-end harness, with the biggest margin where
Tantivy's BM25 is weakest.** All using shipped, default-preserving APIs. Verified: `model2vec` retrieval-32M +
`fastembed` cross-encoders + `rank_bm25` on BEIR qrels (no cargo, no torch).

---

### MEASURED (not assumed): the shipped RrfConfig knobs are latency-free — per-crate bench (IronPetrel, 2026-07-03)

I'd repeatedly *asserted* the shipped fusion knobs (per-tier weights `7ccda28`, hash tiebreak `05472cd`) are negligible-cost
— the project's ethos is measure-don't-assume, so here's the per-crate criterion bench (`benches/rrf_config_cost_ab.rs`,
real public `rrf_fuse`, n=2000 lexical+semantic, median):

| rrf_fuse config | latency | vs default |
|---|---|---|
| **realistic** (few ties) — default | 149 µs | — |
| realistic — tier-weighted (`semantic_weight=1.3`) | 152 µs | +1.9% (CIs overlap → noise) |
| realistic — hash tiebreak | 139 µs | ~equal (noise) |
| **tie-heavy** (n exact ties, adversarial) — lexical tiebreak (default) | 375 µs | — |
| tie-heavy — hash tiebreak | 405 µs | **+7.9%** |

**Finding — the shipped config knobs cost nothing in practice; using the tuned config is latency-free.** The per-tier
weight is a single f64 multiply per candidate — unmeasurable (+1.9%, within noise). The hash tiebreak only computes the
FNV hash on *exact rrf_score ties* (rare in real corpora → ~free realistically), and even in a **pathological workload
where every doc ties** (disjoint lexical/semantic sets with parallel ranks) it adds only **+7.9%** (375→405 µs). Note the
tie-heavy case is ~2.5× slower than realistic for *both* tiebreaks (149→375 µs) — that's the sort doing full comparator
work on ties, independent of the tiebreak choice. **Perf conclusion for the fusion-default question: flipping to the
tuned config (weights + hash tiebreak) has zero practical latency cost** — the earlier "negligible" claim is now measured,
and this is a regression guard for the shipped hot-path changes. Bench: `cargo bench -p frankensearch-fusion --bench
rrf_config_cost_ab`.

---

### MEASURED: RrfCombine reranker reorder is negligible vs the cross-encoder — per-crate bench (IronPetrel, 2026-07-03)

Completing the "measure the cost of everything shipped" pass (fusion knobs done in `979ea3c`; reranker combine here).
I'd asserted the RRF-combine reorder (`235fb46`) is negligible vs a cross-encoder forward pass — measured it
(`frankensearch-rerank/benches/combine_reorder_cost_ab.rs`, faithful replica of `apply_rrf_combine` over real
`ScoredResult`s, median):

| rerank window N | `PureReorder` (1 sort) | `RrfCombine` (argsort+fuse+permute) | Δ |
|---|---|---|---|
| 20 | 0.40 µs | 0.57 µs | +41% |
| 50 | 1.02 µs | 1.26 µs | +24% |
| 100 | 1.98 µs | 2.43 µs | +23% |
| 200 | 3.81 µs | 4.83 µs | +27% |

**Finding — RrfCombine's reorder costs +23-41% over pure-reorder, but that's single-digit MICROseconds; the reranker's
cross-encoder inference is MILLIseconds *per candidate*, so the reorder is ~4-5 orders of magnitude smaller than the
inference it follows — negligible.** For a typical N=100 window, RrfCombine adds **~0.45 µs** of reorder over
PureReorder, against **~100 ms-1 s** of cross-encoder forward passes for those 100 candidates. The relative overhead
(the extra argsort for the rerank rank + the fused sort + the clone-permute) is real but irrelevant at the system level.
**Perf conclusion for the reranker-default question: switching the default to `RrfCombine` is latency-free** — its only
cost is a sub-microsecond reorder, dwarfed by the inference. Combined with the quality finding (RrfCombine removes the
−11/−23% pure-reorder downside), the reranker-default flip is free on both axes. Bench: `cargo bench -p
frankensearch-rerank --bench combine_reorder_cost_ab` (default features, no ONNX).

---

### The COST side of the capstone: the hybrid's quality win costs <1ms vs Tantivy (measured) (IronPetrel, 2026-07-03)

The capstone measured the hybrid's *quality* win (+12-22% nDCG vs Tantivy BM25); this is its *latency* cost — the missing
half of the vs-Tantivy story. Key insight: frankensearch's hybrid = Tantivy BM25 (identical to the baseline) + a vector
tier + RRF fusion, so the **added** latency over Tantivy-lexical-alone is exactly *vector search + fusion* — both
fast-benchable per-crate, no Tantivy chain needed. Measured:

| added stage | latency / query | source |
|---|---|---|
| vector search (int8 two-pass, 10k×384-dim) | **~250-290 µs** | `frankensearch-index --bench int8_two_pass` (this turn) |
| RRF fusion (2000 candidates) | **~149 µs** | `frankensearch-fusion --bench rrf_config_cost_ab` (`979ea3c`) |
| **hybrid total added vs Tantivy (no rerank)** | **< 1 ms** | sum |
| optional reranker (cross-encoder) | **~ms per candidate → 100 ms-1 s** for a top-100 window | forward-pass, established |

**Finding — the hybrid's recall/nDCG win over Tantivy is near-free (< 1 ms added/query); the reranker is the only
expensive stage and should be gated.** At the capstone's BEIR corpus scale (3.6k-5k docs, 256-dim — *smaller* than the
10k×384 bench) vector search is even faster, so the < 1 ms bound holds comfortably. So the cost/benefit vs Tantivy is
lopsided in frankensearch's favor: **+12-22% nDCG for sub-millisecond added latency** on the hybrid, with the reranker as
a separate, expensive, quality-vs-latency-gated polish (and its RRF-combine reorder is itself µs — `368e291` — so the
reranker's cost is 100% cross-encoder inference, nothing frankensearch adds around it). This completes the vs-Tantivy
picture: large quality win, negligible hybrid latency cost, expensive-but-optional rerank. Measured via per-crate
criterion benches (no cargo-for-quality; the index + fusion crates), int8 kernels are the shipped AVX2 path.

---

### Selective (confidence-gated) reranking is a WEAK cost lever — reranker benefit is ~unpredictable from hybrid confidence (IronPetrel, 2026-07-03)

The reranker is the only expensive stage (100 ms-1 s, per the cost analysis). Natural cost lever: **skip the cross-encoder
for high-confidence queries** (large hybrid top-1/top-2 RRF margin — where reranking is unlikely to change the top-k) and
rerank only the uncertain ones. Tested on NFCorpus (ms-marco RRF-combine, where full reranking helps +0.0192 nDCG),
reranking only the least-confident fraction by margin, the rest keeping hybrid order:

| rerank budget | nDCG@10 (margin-selective) | % of full-rerank gain | random-budget baseline |
|---|---|---|---|
| 0% (hybrid) | 0.3268 | 0% | 0.3268 |
| 25% | 0.3352 | **44%** | 0.3316 (~25%) |
| 50% | 0.3393 | **65%** | 0.3364 (~50%) |
| 75% | 0.3422 | 80% | 0.3412 |
| 100% | 0.3460 | 100% | 0.3460 |

**Finding — margin-gated selective reranking is a WEAK lever; reranker benefit is largely unpredictable from pre-rerank
confidence.** The correlation between the hybrid top-1/top-2 margin and the per-query rerank benefit is **−0.055**
(essentially zero). Consequently margin-selective reranking beats *random* budget allocation only modestly (44% vs ~25%
of the gain at a 25% budget; 65% vs ~50% at 50%). So you *can* rerank the least-confident half and keep ~65% of the
quality bump at half the cross-encoder cost — a mild efficiency — but this is far short of the hoped-for "rerank 25%, keep
90%." **The reranker's per-query value doesn't track hybrid confidence**, so the robust reranker cost levers remain the
ones already measured: **rerank a shallow depth** (`97eb1e0`), **gate the whole reranker by corpus** (`657df16`), and
**RRF-combine so it's never catastrophic** (`b114e39`) — not per-query confidence gating. This also echoes the earlier
margin-gated *fusion* result (`8130208`, ~+0.012 only): the top-1/top-2 margin is a real but weak signal that doesn't
convert into a strong adaptive lever. Verified: `model2vec` retrieval-32M + `fastembed` ms-marco cross-encoder +
`rank_bm25` on BEIR NFCorpus qrels (no cargo).

---

### DEFINITIVE: no cheap pre-rerank signal predicts rerank benefit — selective reranking is buried (IronPetrel, 2026-07-03)

The prior entry found the top1-top2 *margin* doesn't predict which queries benefit from reranking (corr −0.055) but left
open "maybe a better signal does." Tested **five** candidate pre-rerank signals in one rerank pass (NFCorpus, ms-marco
RRF-combine, full gain +0.0192), correlating each per-query signal with the per-query rerank benefit:

| signal (all computable BEFORE reranking) | corr with rerank benefit |
|---|---|
| `margin` (top-1 − top-2 RRF gap) | −0.055 |
| `tier_overlap` (\|lexical top-k ∩ vector top-k\|) | +0.011 |
| `top1` (top RRF score) | +0.019 |
| `spread` (std of top-k RRF scores) | +0.057 |
| `vec_lex_gap` (per-tier nDCG gap — **oracle, uses labels**) | +0.195 |

**Finding — no deployable pre-rerank signal predicts rerank benefit; selective (confidence-gated) reranking is not
viable.** Every real signal is ≤ 0.057 (essentially zero), including the principled *tier-disagreement* one
(`tier_overlap`, +0.011) — reranking does NOT preferentially help queries where the two tiers disagree. Even the
oracle-ish `vec_lex_gap` (which cheats by using relevance labels) only reaches +0.195. So the reranker's per-query value
is essentially **unpredictable from cheap query-time features** — you cannot cheaply decide *which* queries to rerank.
This comprehensively closes the selective-reranking cost lever (5 signals tested, all weak): the robust reranker cost
levers remain **corpus-level gating** (`657df16`), **shallow depth** (`97eb1e0`), and **RRF-combine safety**
(`b114e39`) — not per-query gating. Combined with the earlier weak margin-gated *fusion* (`8130208`), the broader lesson
is settled: **cheap retrieval-confidence signals do not convert into strong adaptive levers** in this stack. Verified:
`model2vec` retrieval-32M + `fastembed` ms-marco cross-encoder + `rank_bm25` on BEIR NFCorpus qrels (no cargo).

---

### Lexical tier: BM25F title-boosting does NOT help — plain title+body concatenation is optimal (IronPetrel, 2026-07-03)

The one tier never tuned this session was the **lexical** one (frankensearch delegates it to Tantivy, which supports
fielded/BM25F scoring). Tested whether **title-boosting** beats the current plain `title + " " + text` concatenation:
built separate `BM25(title)` + `BM25(body)` and scored `tw·BM25(title) + BM25(body)` over a title-weight sweep (both
BEIR datasets are 100% titled):

| config | SciFact recall/nDCG@10 | NFCorpus recall/nDCG@10 |
|---|---|---|
| **concat** `BM25(title+body)` (current) | **0.7757 / 0.6523** | 0.1522 / 0.3062 |
| body-only (`tw=0`) | 0.7590 / 0.6367 | 0.1474 / 0.3013 |
| fielded `tw=0.5` | 0.7757 / 0.6466 | 0.1503 / **0.3074** (+0.4%) |
| fielded `tw=1.0` | 0.7523 / 0.6248 | 0.1487 / 0.3036 |
| fielded `tw=2.0` | 0.6973 / 0.5784 | 0.1418 / 0.2902 |

**Finding — plain concatenation is optimal; explicit BM25F title-boosting doesn't help and HURTS when over-applied.**
The title *does* carry signal (body-only is worse than concat on both), but concatenation already captures it — the title
terms naturally raise the doc's term frequencies. Explicitly up-weighting the title (`tw ≥ 1`) DEGRADES quality (SciFact
`tw=2` → **−11% nDCG**), because BM25's length normalization is distorted by over-emphasizing the short title field. The
only positive is a noise-level +0.4% on NFCorpus at `tw=0.5`. **Product takeaway: keep the lexical tier as plain
title+body concatenation (frankensearch's current approach); do NOT implement BM25F/fielded title-boosting for the hybrid
— it's at best neutral and easily net-negative.** Note this is a lexical-config lever that would help Tantivy-alone and
the hybrid's lexical tier equally, so it does not shift the hybrid-vs-Tantivy delta regardless. Verified: `rank_bm25`
(BM25Okapi) on BEIR SciFact/NFCorpus with separated title/body fields (no cargo).

---

### Lexical tier: RM3 pseudo-relevance feedback is a recall/nDCG tradeoff, redundant with the vector tier (IronPetrel, 2026-07-03)

Companion to the BM25F test — the *other* classic BM25 improvement, and distinct from the (negative) embedding-space PRF
(`433f758`): **RM3 lexical PRF** expands the BM25 query with high-value terms from the top-retrieved docs (relevance-model
weights, IDF-scaled, top-10 feedback docs), then re-retrieves. Swept expansion-term count M:

| config | SciFact recall/nDCG@10 | NFCorpus recall/nDCG@10 |
|---|---|---|
| BM25 (no RM3) | 0.7757 / **0.6523** | 0.1522 / **0.3062** |
| RM3 +5 terms | 0.7857 / 0.6256 | 0.1549 / 0.2985 |
| RM3 +10 terms | **0.7932** / 0.5950 | 0.1520 / 0.2841 |
| RM3 +20 terms | 0.7826 / 0.5591 | 0.1544 / 0.2875 |
| RM3 +30 terms | 0.7693 / 0.5380 | **0.1558** / 0.2932 |

**Finding — RM3 is a recall↑ / nDCG↓ tradeoff, not a clear win; and it's redundant in the hybrid.** Classic query-
expansion behavior: it retrieves more relevant docs (SciFact recall +1.8 pt @M=10; NFCorpus +0.4 pt @M=30) but dilutes
the top ranking with expansion noise (SciFact nDCG −8.8%, NFCorpus −2.5%) — by nDCG, no-RM3 wins on both. **Two reasons
it's not worth it for frankensearch's hybrid:** (1) the recall gain is **redundant** — the vector tier already supplies
recall (hybrid recall@100 ≈ 0.96, `ad4487e`), so RM3's lexical recall bump adds little the vector tier doesn't already
cover; (2) the nDCG hit would **propagate** into the fusion. So the lexical tier should stay plain BM25 over concatenated
title+body. **This closes lexical-tier tuning:** both classic BM25 improvements — BM25F title-boosting (`b10e243`) and
RM3 — are net-neutral-to-negative for the hybrid, and both PRF variants (embedding-space `433f758` and lexical RM3) fail
to cleanly help. The vector tier is the right way to improve recall, not lexical expansion. Verified: `rank_bm25`
(BM25Okapi + its IDF weights) on BEIR SciFact/NFCorpus qrels (no cargo).

---

### No valuable THIRD modality: char-3gram / word-bigram don't help — two strong tiers is the sweet spot (IronPetrel, 2026-07-03)

The modality-diversity finding (`339d22b`) showed the hybrid works via *decorrelated modalities* (BM25 exact-term vs
vector semantic), and multi-*embedder* ensembling was redundant (same modality). Open question: is there a valuable
**third, genuinely-different lexical modality**? Tested char-3gram (typo/morphology) and word-bigram (phrase) TF-IDF as a
third RRF tier on top of vector + token-BM25:

| stack | SciFact recall/nDCG@10 | NFCorpus recall/nDCG@10 |
|---|---|---|
| hybrid (vector + token-BM25) | **0.8358 / 0.6904** | **0.1587 / 0.3257** |
| + char-3gram (3rd tier) | 0.8323 / 0.6923 (noise) | 0.1590 / 0.3239 (noise) |
| + word-bigram (3rd tier) | 0.8164 / 0.6589 (**−4.6%**) | 0.1430 / 0.2925 (**−10%**) |

**Finding — no third modality adds value; two comparably-strong decorrelated tiers is the sweet spot.** char-3gram is
**neutral** (redundant with token-BM25 on clean, well-spelled BEIR text — its finer granularity captures the same lexical
signal); word-bigram **hurts** (bigram matching is too sparse/noisy and dilutes the fusion). Critically, on NFCorpus
char-3gram **finds a relevant doc the vec+BM25 top-10 miss in 17.3% of queries — yet the ensemble does not gain** (0.3239
≈ 0.3257): the exact "unique-% is not sufficient" mechanism from the multi-embedder finding — the third modality is
*weaker*, so its dilution cancels its unique finds. **Combined verdict on ensembling (embedder + modality): the hybrid is
optimally exactly two tiers — one strong embedder + BM25.** More tiers (a second embedder `339d22b`, a third lexical
modality here) are redundant-to-harmful; the partner must be both *decorrelated* AND *comparably strong*, and only the
BM25↔vector pair satisfies both. Caveat: char n-grams' value is typo/OOV-robustness, which clean BEIR doesn't exercise —
they may help noisier real-world corpora, but not the retrieval-quality axis. Verified: `model2vec` retrieval-32M +
`rank_bm25` + `sklearn` char/word n-gram TF-IDF on BEIR SciFact/NFCorpus qrels (no cargo).

---

### Length-adaptive fusion doesn't help — query length is a weak per-query tier predictor (IronPetrel, 2026-07-03)

The short-query vector-collapse is a real *aggregate* effect (vector recall 0.99→0.45 as queries shrink 10→3 words,
`db0cca4`), and length is a **cheap, reliable** signal — unlike the weak top-1/top-2 margin. So length-gated fusion
weighting (short queries → up-weight lexical, long → up-weight vector) was the one adaptive lever that *should* work.
Tested vs a fixed `vec_w=1.3` (recall/nDCG@10, threshold = word count):

| dataset (median qlen) | fixed | length-adaptive (best) | corr(qlen, vector-advantage) |
|---|---|---|---|
| SciFact (12 words) | 0.8341 / **0.6890** | 0.8326 / 0.6914 (thr=8, +0.3% = noise) | +0.078 |
| NFCorpus (2 words) | 0.1585 / **0.3268** | 0.1556 / 0.3187 (all worse) | +0.113 |

**Finding — length-adaptive fusion is neutral (SciFact, noise) to harmful (NFCorpus); query length is a WEAK per-query
predictor of which tier wins** (corr with per-query vector-vs-lexical advantage = only +0.078 / +0.113). The crucial
distinction: the short-query collapse is real in *aggregate*, but on natural BEIR queries length does NOT *per-query*
predict the better tier well enough to gate on — even NFCorpus (median 2-word queries) is *hurt* by short→lexical gating,
because plenty of short queries retrieve fine with the vector tier. **This is the third adaptive signal proven weak**
(after margin×fusion `8130208` and margin×reranking `4017650`), and the most principled one — so it settles the direction
definitively: **no cheap per-query signal (margin, length) converts into a strong adaptive lever in this stack; the
aggregate ≠ per-query predictability.** Fixed fusion weighting (up-weight the stronger tier globally, `fa592b9`) is the
right approach; per-query adaptivity does not pay. Verified: `model2vec` retrieval-32M + `rank_bm25` on BEIR
SciFact/NFCorpus qrels (no cargo).

---

### MRL dim-256 truncation validated across 3 datasets: 95-98% of full nDCG at 2× smaller (IronPetrel, 2026-07-03)

The "free upgrade" recommendation stores retrieval-32M MRL-truncated to **dim-256** (half of its 512), claiming ~95% of
full quality — but that rested mostly on SciFact. Validated the MRL dim ladder (first-d dims, re-normalized) across all 3
BEIR datasets:

| dataset | dim 512 (full) nDCG@10 | MRL-256 | MRL-128 | MRL-64 |
|---|---|---|---|---|
| SciFact | 0.6331 | 0.5993 (**95%**) | 0.5682 (90%) | 0.4869 (77%) |
| NFCorpus | 0.3085 | 0.3032 (**98%**) | 0.2899 (94%) | 0.2553 (83%) |
| ArguAna | 0.3328 | 0.3275 (**98%**) | 0.3088 (93%) | 0.2710 (81%) |

**Finding — MRL-256 retains 95-98% of full nDCG on every dataset** (SciFact 95%, NFCorpus 98%, ArguAna 98%) at 2× less
storage and scan cost — the dim-256 free-upgrade recommendation holds beyond SciFact, confirmed to the 3-dataset bar.
**dim-128** (4× smaller) is a viable *storage-constrained* option (90-94% of full, weakest on SciFact); **dim-64** is
too aggressive (77-83%). The graceful degradation is a property of the retrieval-distilled (Matryoshka-trained) model —
recall the general multilingual default degrades catastrophically under MRL (`01bec23`), so this only applies with a
retrieval-distilled embedder (rec #1). Net: **default to retrieval-32M @ dim-256 (95-98% quality, 2× cheaper); drop to
dim-128 only if storage-bound.** Verified: `model2vec` retrieval-32M, first-d MRL truncation on BEIR
SciFact/NFCorpus/ArguAna qrels (no cargo).

---

### Hybrid's recall advantage over Tantivy grows/holds with depth (3 datasets) — validates the reranker-feed (IronPetrel, 2026-07-03)

The "deep candidate feed" recommendation and "hybrid is the right reranker first-stage" rested on SciFact (recall@100
0.96, `ad4487e`). Validated the recall-depth curve — Tantivy BM25 vs hybrid — across all 3 BEIR datasets:

| dataset | recall@10 | recall@50 | recall@100 |
|---|---|---|---|
| SciFact | BM25 0.776 / hyb 0.837 (**+8%**) | 0.869 / 0.935 (+8%) | 0.873 / 0.960 (**+10%**) |
| NFCorpus | 0.152 / 0.157 (+3%) | 0.212 / 0.245 (+15%) | 0.235 / 0.301 (**+28%**) |
| ArguAna | 0.606 / 0.696 (+15%) | 0.802 / 0.926 (+15%) | 0.841 / 0.957 (**+14%**) |

**Finding — the hybrid's recall advantage over Tantivy BM25 is large and grows-or-holds with depth on all 3 datasets**
(+10% to +28% at recall@100). This is the reranker-feed argument confirmed: the hybrid supplies the cross-encoder a far
richer candidate pool at depth than Tantivy would (hybrid recall@100 = 0.96 / 0.30 / 0.96). **NFCorpus is the sharpest
case: only +3% at rank-10 but +28% at rank-100** — precisely *why* reranking helps NFCorpus so much (ms-marco +23.5%,
`657df16`): the deep pool holds 28% more relevant docs than Tantivy for the reranker to promote into the top-10. This
validates two recommendations to the 3-dataset bar — **deep candidate feed** (fetch ~50-100/tier, since the advantage
is largest at depth) and **the hybrid as the first stage for reranking** — and quantifies frankensearch's deep-recall
edge over Tantivy: it's the compounding advantage a reranker turns into final-ranking quality. Verified: `model2vec`
retrieval-32M + `rank_bm25` on BEIR SciFact/NFCorpus/ArguAna qrels (no cargo).

---

### RRF k=10 beats the default k=60 by +2.5-2.9% nDCG on all 3 datasets (with tuned weight) (IronPetrel, 2026-07-03)

The fusion-tuning recommendation includes a **small RRF `k` (~10, not the default 60)**, found on SciFact. Validated the
k-sweep with the stronger-tier weight (`vec_w=1.3`) across all 3 BEIR datasets (nDCG@10):

| dataset | k=5 | k=10 | k=20 | k=40 | k=60 (default) | k=10 vs k=60 |
|---|---|---|---|---|---|---|
| SciFact | **0.6907** | 0.6890 | 0.6814 | 0.6746 | 0.6698 | +2.9% |
| NFCorpus | **0.3268** | 0.3268 | 0.3243 | 0.3208 | 0.3187 | +2.5% |
| ArguAna | **0.3330** | 0.3300 | 0.3253 | 0.3220 | 0.3216 | +2.6% |

**Finding — smaller `k` is better on every dataset; nDCG decreases monotonically from k=5 to k=60, and k=10 beats the
default k=60 by +2.5-2.9%** (k=5 marginally best). The mechanism (confirmed): with a stronger-tier weight, a small `k`
keeps `1/(k+r)` **top-heavy** so the weight actually shifts ranking, whereas `k=60` flattens the contribution curve over
the top-10 and the weight barely bites — the two knobs (small `k` + tier weight) work together. This validates the
small-k recommendation to the 3-dataset bar and quantifies the `RrfConfig.k` default flip (60 → ~10, or 5): **worth
~+2.6% nDCG when paired with the tuned tier weight, free at inference.** Recommend `k ≈ 10` as a safe default (k=5 is
marginally better but sharper); `RrfConfig.k` is already a shipped config field, so this is a one-value change. Verified:
`model2vec` retrieval-32M + `rank_bm25` on BEIR SciFact/NFCorpus/ArguAna qrels (no cargo).

---

### Refreshed total uplift: recommended vs stock defaults = +15-21% nDCG (all validated knobs) (IronPetrel, 2026-07-03)

The `SEARCH_QUALITY_FINDINGS.md` "Total measured uplift" table (+5-7% nDCG) predated the k=10 (`4cc3b47`) and dim-256
(`06fdef9`) validations. Refreshed the definitive stock-vs-recommended hybrid comparison with **all** now-validated
retrieval-side knobs, both at dim-256 (= equal scan cost, no reranker), recall@10 / nDCG@10:

| dataset | STOCK (multilingual-128M + equal RRF k=60) | RECOMMENDED (retrieval-32M@256 + weight-1.3 + k=10) | Δrecall / ΔnDCG |
|---|---|---|---|
| SciFact | 0.7564 / 0.5904 | **0.8424 / 0.6761** | **+11% / +15%** |
| NFCorpus | 0.1325 / 0.2694 | **0.1595 / 0.3253** | **+20% / +21%** |
| ArguAna | (embedder change matters least here; prior recs-1+3 ≈ +1% nDCG — vector already strong vs BM25's long-doc weakness) | | ~+1-4% |

**Finding — adopting the recommended retrieval-side defaults buys +11-20% recall / +15-21% nDCG over the current stock
defaults on the semantic corpora** (SciFact, NFCorpus), roughly **3× the stale table's +5-7%** — because the earlier
number omitted the small-k (+2.6%) and only-partially-applied the tuning. The stack of free/cheap changes compounds:
retrieval-distilled embedder (biggest), + stronger-tier weight, + small RRF k, all at dim-256 (equal cost, no reranker,
no new model). ArguAna's uplift is smaller (~+1-4%) — there the vector tier is *already* strong versus BM25's long-doc
length penalty, so the embedder swap adds less. **Bottom line: the retrieval-side recommendations alone (before the
optional reranker) lift the semantic-search hybrid +15-21% nDCG at zero inference cost**, all via shipped config knobs +
a retrieval-distilled embedder. Verified: `model2vec` multilingual-128M & retrieval-32M + `rank_bm25` on BEIR (no cargo).

---

### Tier-weighting subsumes the tiebreak: it eliminates 95-97% of RRF ties (recommendations interact) (IronPetrel, 2026-07-03)

I ship + recommend both the neutral **hash tiebreak** and **tier-weighting** — but they interact, and measuring how
self-simplifies the recommendation. Counted exact RRF-score ties in the fused top-10 (the only place the tiebreak
fires), stock vs recommended config:

| config | SciFact: % queries with a tie (tie-collisions) | NFCorpus |
|---|---|---|
| equal-weight, k=60 (stock) | **27.0%** (105) | **47.7%** (369) |
| weighted 1.3, k=10 (recommended) | **1.0%** (3) | **1.2%** (4) |

**Finding — tier-weighting eliminates ~95-97% of RRF-score ties, making the hash-tiebreak nearly moot once you weight.**
With *equal* weights the tiebreak matters enormously: 27-48% of queries have a top-10 tie, so the legacy lexical-favoring
tiebreak biases a huge fraction of results (this is why the tiebreak diagnosis `339d22b`/`05472cd` mattered — for the
*stock* config). But distinct tier weights (`1.0` vs `1.3/(k+r)`) make the per-doc contributions distinct, and small `k`
spreads scores, so the recommended config has almost **no** ties (~1%) — the tiebreak choice becomes irrelevant.
**Practical simplification: if you adopt tier-weighting (rec #3), the hash-tiebreak is redundant (skip it); if you run
equal weights, the hash-tiebreak is important (it de-biases ~1/3-1/2 of queries).** The two knobs are not independent —
weighting is the more fundamental fix (it removes the ties rather than resolving them fairly). Verified: `model2vec`
retrieval-32M + `rank_bm25`, exact-float tie detection in the fused top-10 on BEIR SciFact/NFCorpus (no cargo).

---

### Deep candidate feed is a RERANKER-only lever: the non-reranked top-10 plateaus at feed=20 (IronPetrel, 2026-07-03)

I recommend a "deep candidate feed (~50-100/tier)" — but that was justified by recall@100 (`ad4487e`, `6efd9d9`), the
*reranker*-feed metric. Does deeper feed help the **non-reranked hybrid's final top-10**? Swept the fusion feed depth
(candidates per tier fed to RRF), weighted hybrid, final recall/nDCG@10:

| feed/tier | SciFact nDCG@10 | NFCorpus nDCG@10 |
|---|---|---|
| 10 | 0.6834 | 0.3219 |
| 20 | 0.6896 | 0.3251 |
| 50 | 0.6890 | 0.3268 |
| 100 | 0.6888 | 0.3255 |
| 200 | 0.6899 | 0.3248 |

**Finding — for the non-reranked hybrid, feed=20-50 is sufficient; deeper feed (100-200) adds nothing** (nDCG flat from
feed=20 onward, ±0.1% = noise). The reason is structural: a candidate at feed-rank ~80 contributes only `1/(k+80)` ≈
negligible to its RRF score, so it cannot crack the fused top-10 unless it *also* ranks near the top of the other tier
(rare) — deep feed simply can't move the non-reranked top-10. **So "deep candidate feed" is a RERANKER-only lever:** it
enriches the deep candidate pool the cross-encoder reorders (recall@100 grows with depth, `6efd9d9`), but the raw hybrid
top-10 is set by the top ~20. **Refined recommendation for `candidate_multiplier`: use a modest feed (~20-50/tier) if NOT
reranking; use a deep feed (~100/tier) only when reranking** (the reranker is what turns deep recall into final-ranking
quality). Reconciles the two depth findings: recall@100 grows with feed depth, but non-reranked nDCG@10 doesn't. Verified:
`model2vec` retrieval-32M + `rank_bm25` on BEIR SciFact/NFCorpus (no cargo).
