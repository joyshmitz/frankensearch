# Quill Performance Attribution Runbook (MT8 discipline)

> Profile-first discipline for every Quill optimization lever (plan §14 law 5,
> `bd-quill-e8-perf-doctrine-x4e4.3`). A kept perf change MUST name a specific
> frame ≥ 0.1 % self-time BEFORE the source touch and show it closed AFTER.
> This runbook is the local attribution procedure that makes that possible, plus
> the committed baseline attribution for the query and ingest lanes.

`rch` cannot produce symbolized self-time (`bd-e41k`): the release profile strips
symbols and the workers lack `perf`. **Attribution runs LOCALLY on the dev box;
`rch` is only for the throughput/instruction-count matrix.** Everything below was
validated end-to-end on csd (Zen-class x86, `perf 6.17`, `perf_event_paranoid=1`).

---

## 1. Build a profiling binary (local, symbolized)

The profiling target is the `segment_fanout_ab` bench (`crates/frankensearch-quill/benches/`)
— it drives both ingest (`build_index`) and query (`bench_search_sealed_forced`) over a
synthetic multi-segment corpus, sized by env.

```bash
# One-time: a shim to invoke a KNOWN-GOOD dated nightly's cargo directly.
# WHY a dated nightly, not `nightly`: the floating `nightly` toolchain on this box
# triggers a broken rustup component auto-install mid-build (a failing rust-docs
# install, "detected conflict: share/doc/rust/html") — pin a dated nightly that has
# `--check-cfg`. WHY not the rustup shim (~/.cargo/bin/cargo): the shim dispatches on
# argv[0] and rejects a non-`cargo` name ("unknown proxy name: 'qc'"), and the rch
# PreToolUse hook rewrites a `cargo`-named invocation to a remote build.
ln -sf ~/.rustup/toolchains/nightly-2026-07-06-x86_64-unknown-linux-gnu/bin/cargo /tmp/qc

# Build with symbols, no strip, and LTO OFF.
# WHY LTO off: the shipped `lto = true` release build is memory-heavy and fails/OOMs
# locally under fleet contention; non-LTO still surfaces the dominant hotspot (inlining
# differs, so treat absolute %s as directional, not exact). Isolated target dir so the
# shared build cache is untouched.
env RUSTUP_TOOLCHAIN=nightly-2026-07-06-x86_64-unknown-linux-gnu \
    CARGO_PROFILE_RELEASE_LTO=false \
    CARGO_PROFILE_RELEASE_DEBUG=1 \
    CARGO_PROFILE_RELEASE_STRIP=false \
    CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 \
    CARGO_TARGET_DIR=/tmp/quill-prof \
  /tmp/qc build --profile release -p frankensearch-quill \
    --features bench-internals --bench segment_fanout_ab
# Binary: /tmp/quill-prof/release/deps/segment_fanout_ab-<hash>
# (the hash is stable across quill-only rebuilds; do NOT capture it with `$(ls …)` —
#  `ls` is aliased to a detailed listing here, so a captured var becomes the whole
#  listing line and the run silently fails as an env error — hardcode the literal path.)
```

## 2. Record self-time (the attribution primitive)

**Use FLAT sampling** (`-F 999`, no `-g`) with `RAYON_NUM_THREADS=1`. Flat sampling
gives per-symbol self-time — which is exactly what the ≥ 0.1 % MT8 gate needs — at low
overhead. **DWARF call-graph (`--call-graph dwarf`) HANGS** on the rayon-parallel search
workload (the unwinder deadlocks); single-thread it and prefer flat.

```bash
BIN=/tmp/quill-prof/release/deps/segment_fanout_ab-<hash>   # literal path

# QUERY lane (query-decode dominates: build once, many query rounds)
perf record -F 999 -o /tmp/quill.query.perf.data -- \
  env QUILL_E49_SCALE=smoke QUILL_E49_ROUNDS=9 RAYON_NUM_THREADS=1 "$BIN"

# INGEST lane (ingest dominates: ~71k docs built, one query round)
perf record -F 999 -o /tmp/quill.ingest.perf.data -- \
  env QUILL_E49_SCALE=full QUILL_E49_ROUNDS=1 RAYON_NUM_THREADS=1 "$BIN"

# Read self-time, by symbol and by source line:
perf report --stdio --sort symbol  -i /tmp/quill.query.perf.data | grep frankensearch_quill | head -20
perf report --stdio --sort srcline -i /tmp/quill.query.perf.data | grep -E 'scribe|quiver|grimoire|argus' | head -20
```

`perf report --sort srcline` maps hot samples straight to source lines — the sharpest
tool for finding *which* line in a hot function is paying. `perf annotate --stdio -i …`
disassembles a symbol and puts a percent on each instruction: this is how you find *why*
a frame is hot (e.g. a function prologue = it is never inlined; a `push`-heavy prologue
at the top of a small function called in a tight loop is the classic inline-me signal).

### Ingest-lane caveat — test-logging contamination

`segment_fanout_ab` builds the index via `asupersync::test_utils::run_test_with_cx`, which
calls `init_test_logging()` and installs a tracing subscriber that **cannot be disabled via
`RUST_LOG`**. So the ingest profile carries ~6 % test-logging overhead
(`format_event`, `record_debug`, `TestWriter`, `sharded_slab`) that production (tracing off)
does not, and any per-document diagnostic value the accumulator computes is *used* here.
A frame that is only paid to feed a tracing span (e.g. `bytes_reserved`) is real in this
profile but pure waste in a tracing-off deployment — check whether a hot accumulator method
is called from PRODUCTION logic (e.g. `should_flush`) or only from a tracing `record`
before optimizing it. For a clean production-representative ingest profile, a bench that
does not init test-logging is needed (open follow-up).

## 3. Prove the lever (deterministic instruction count)

Wall-clock is noisy; **`perf stat -e instructions` is deterministic** (run-to-run ±0.01 %),
so even a sub-1 % change is decidable. Build TWO binaries (HEAD vs the change) to SEPARATE
target dirs and compare — the shared tracing overhead CANCELS between the two, so a core
change is measured cleanly even on the tracing-on ingest bench.

```bash
for BIN in "$OLD" "$NEW"; do
  perf stat -e instructions env QUILL_E49_SCALE=full QUILL_E49_ROUNDS=1 RAYON_NUM_THREADS=1 "$BIN"
done
```

Caveats: the local disk-pressure ballast (`sbh`) cleans stale target dirs — **measure
promptly after building**. Redirects to a `$VAR`-expanded path are blocked by `dcg`
(dynamic-truncation guard) — use literal paths in `>`/`2>` targets.

**A dependency-bound loop is the one place instruction count lies:** if the removed
instruction is overlapped with a loop-carried stall (e.g. the cursor bounds-check inside
the delta-decoded `consume_position_run`), instruction count drops but wall-clock does not.
Confirm with `perf stat -e cycles` before claiming a win on a serial loop.

## 4. The MT8 attribution loop (worked example — the vint lever, `bd-b6tc`)

1. **Report** — query-lane flat self-time: `SliceReader::read_vint` 16.3 %,
   `PositionByteReader::read_u32_vint` 11.3 % → vint decode ≈ 33 % of query self-time.
2. **First hypothesis wrong** — a 1-byte fast path alone: re-profile showed `read_vint`
   *unchanged* (~17 %). Do not ship on a hypothesis; re-profile.
3. **Annotate** — `perf annotate read_vint`: the hot instructions were the function
   *prologue* (6-callee-saved-register spill ≈ 18 % of the frame). The readers were never
   inlined, so every call in the tight `decode_entry`/`consume_position_run` loops paid it.
4. **Fix** — inline-hot / outline-cold: `#[inline]` the 1-byte fast path into callers,
   `#[cold] #[inline(never)]` the multi-byte loop.
5. **Prove** — instruction A/B: **7.300 B → 6.370 B, −16.2 %**, byte-identical, 473/473 tests.
6. **Close the frame** — re-profile: `read_vint` self-time drops and its work correctly
   redistributes into the callers (which is why total, not the single symbol, is the metric).

Same loop closed `decode_bitmap_payload` (bit-by-bit scan → `trailing_zeros`, −4.86 %) and
the ingest `TermInterner::bytes_used` recompute (O(distinct-terms) per document via
`should_flush` → O(1) running counter reusing the already-computed `added_bytes`, −12.7 %).

---

## 5. Baseline attribution — QUERY lane

`QUILL_E49_SCALE=smoke ROUNDS=9`, 1-thread, flat `-F 999`, post-`bd-2fha` build. Top
self-time frames (frankensearch_quill only):

| self-time | frame | note |
|---|---|---|
| 15.4 % | `grimoire::decode_entry` | term-dict entry decode: inlined vints + prefix-key memcpy + validation |
| 13.9 % | `quiver::consume_position_run` | phrase position decode — **dependency-bound** (delta prefix-sum) |
| 8.8 % | `SliceReader::read_vint_multibyte` | LEB128 cold path (grimoire multi-byte vints are common) |
| 7.3 % | `PositionList::positions_for_ordinal` | position lookup |
| 6.3 % | `grimoire::validate_block` | canonical-format validation (corruption defense — do not strip) |
| ~8 % | `PostingCursor::next` / `posting_ordinal` | cursor advance |
| 3.3 % | `quiver::decode_frequencies` | |
| 3.2 % | `quiver::decode_bitmap_payload` | (was 6.8 % pre-`bd-udz8`) |
| 2.4 % | `grimoire::validate_metadata_basic` | corruption defense |
| 1.2 % | `bitpack::unpack_wide_into` | SIMD posting unpack (banded dispatch, `bd-bz7q`) |

Landed query-lane levers: `bd-b6tc` (vint −16.2 %), `bd-udz8` (bitmap −4.86 %),
`bd-2fha` (position peel −0.81 %), `bd-y1ab` (packed collector key), `bd-bz7q` (banded
unpack). Remaining frames are dependency-bound, inherent LEB128, or deliberate defenses.

## 6. Baseline attribution — INGEST lane

`QUILL_E49_SCALE=full ROUNDS=1` (~71k docs), 1-thread, flat `-F 999`, post-`bd-w4j5` build.
(Carries the test-logging overhead noted in §2.) Top self-time frames:

| self-time | frame | note |
|---|---|---|
| 4.3 % | `ColumnarAccumulator::bytes_reserved` | tracing-only per-doc recompute; capacity/realloc-driven; asserted exactly by tests |
| 4.6 % | `__memmove_avx_unaligned_erms` | Vec-growth reallocs + arena/key copies + span-field formatting |
| 1.9 % | `tracing_subscriber ...::record_debug` | test-logging (absent in production) |
| 1.8 % | `tracing_subscriber ...::format_event` | test-logging |
| 1.3 % | `TermInterner::intern_accounted` | hash + compare + composite-key copy |
| 1.2 % | `ColumnarAccumulator::add_document_with_values` | per-document accumulate |
| 0.7 % | `FrankensearchTokenizer::analyze` | SWAR default tokenizer (already optimized) |

Landed ingest-lane lever: `bd-w4j5` (`TermInterner::bytes_used` O(1) running counter,
−12.7 % ingest instructions — `should_flush` recomputed it per document). `bytes_used`
is gone from this map (was 7.3 % pre-`bd-w4j5`).

## 7. Committed artifacts

Baseline flamegraphs (call-graph SVGs, `flamegraph --perfdata` over a **single-thread**
DWARF capture — `--call-graph dwarf` at `RAYON_NUM_THREADS=1`, which does not hang the way
the multi-thread rayon path does):

- `docs/perf-artifacts/quill-query-flamegraph.svg` — query lane (`smoke`, ROUNDS=3).
- `docs/perf-artifacts/quill-ingest-flamegraph.svg` — ingest lane (`full`, ROUNDS=1).

Regenerate:

```bash
perf record -F 499 --call-graph dwarf -o /tmp/quill.query.cg.perf.data -- \
  env QUILL_E49_SCALE=smoke QUILL_E49_ROUNDS=3 RAYON_NUM_THREADS=1 "$BIN"
flamegraph --perfdata /tmp/quill.query.cg.perf.data -o docs/perf-artifacts/quill-query-flamegraph.svg
# (`inferno-*` binaries are not installed on this box; use `flamegraph --perfdata`.)
```

The raw `perf.data` files are large binaries and are **not** committed; regenerate them with
§2. The attribution TABLES above (§5, §6) are the durable, reviewable text artifact — the
≥ 0.1 % self-time frames that every MT8 lever must name. The measured levers derived from
them are recorded in `docs/PERF_LEDGER.md`; rejected hypotheses in `docs/NEGATIVE_EVIDENCE.md`.
