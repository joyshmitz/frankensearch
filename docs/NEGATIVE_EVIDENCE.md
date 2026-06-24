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
| 2026-06-24 | BlueGull (`frankensearch-cod-a`) | `frankensearch/search_bench vector_search_topk/top10/10000` | `cargo bench --release -p frankensearch --bench search_bench vector_search_topk/top10/10000 -- --quiet` failed before measurement on rustc `1.98.0-nightly (f20a92ec0 2026-06-07)` because Cargo rejected `--release` for `cargo bench` as an unexpected argument. | Blocker tracked in `bd-ui41`; do not count as a perf ratio. |
| 2026-06-24 | BlueGull (`frankensearch-cod-a`) | `frankensearch/search_bench vector_search_topk/top10/10000` | Fallback optimized bench command without `--release` ran through RCH on `vmi1153651` but remained in cold compile/link with no Criterion timing output after more than 10 minutes; interrupted by the owner with exit 130. | No ratio produced; use `bd-ui41` to establish a reproducible harness and command contract. |
| 2026-06-24 | BlueGull (`frankensearch-cod-a`) | Tantivy/Lucene-class original comparison | README/AGENTS confirm frankensearch is a Tantivy BM25 + semantic/vector hybrid, but no current per-crate harness emits same-corpus ratios against a Tantivy-only incumbent. | Blocker tracked in `bd-ui41`; no dominance claim is valid until this exists. |
| 2026-06-24 | frankensearch-cod-b | `frankensearch/search_bench` requested protocol | `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b rch exec -- cargo bench -p frankensearch --release --bench search_bench -- --sample-size 10` selected RCH worker `ovh-a`, then Cargo rejected `--release` for `cargo bench` with `unexpected argument '--release' found`; `cargo bench --help` lists `--profile <PROFILE-NAME>` instead. | Same protocol blocker as above; successful measurement used `--profile release` and remains per-crate. |
| 2026-06-24 | frankensearch-cod-b | `frankensearch/search_bench vector_search_topk/top10/{1000,5000,10000}` | Same-worker RCH run on `ovh-a`: `rch exec -- cargo bench -p frankensearch --profile release --bench search_bench -- --sample-size 10`. Results: 1K `944.07 us`, 5K `3.4640 ms`, 10K `1.6642 ms`. | Scaling order is unstable/noisy; use as routing evidence only, not as keep/reject proof. |
| 2026-06-24 | frankensearch-cod-b | `frankensearch-index/dot_product` release-profile comparison from detached baseline worktree | Three attempts with `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b rch exec -- cargo bench -p frankensearch-index --profile release --bench dot_product -- --sample-size 10 --warm-up-time 1 --measurement-time 3` fell open to local with `no admissible workers: insufficient_slots=8,health_below_fallback=2,hard_preflight=1`; each local fallback was interrupted before measurement. | No release-profile ratio. Bench-profile RCH runs may be routing evidence, but kept wins need an admitted remote release-profile run or an in-process head-to-head harness. |
| 2026-06-24 | BlueGull (`frankensearch-cod-a`) | `frankensearch-index/dot_product f32_bytes` vs Tantivy/Lucene/Meilisearch-class original | Kept microkernel proof is against the embedded pre-change frankensearch `f32_bytes_old` baseline: pinned RCH worker `vmi1149989` measured `dot/dim256/f32_bytes/10000` at 3.4835 ms old -> 0.66126 ms new (ratio 0.190) and `dot/dim384/f32_bytes/10000` at 5.1487 ms old -> 1.8811 ms new (ratio 0.365). There is still no same-corpus Tantivy/Lucene/Meilisearch-class comparator for this vector-byte kernel or end-to-end workload. | Original-comparator ratio remains blocked by `bd-ui41`; do not claim dominance over Lucene/Tantivy/Meilisearch-class from this microkernel win alone. |

---

## Reverted experiments

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

**Decision:** do not ship the f32 slice accumulator rewrite. The committed lever is only
`dot_product_f32_bytes_f32`, which won on both remote `vmi1149989` and local-fallback checks.

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
