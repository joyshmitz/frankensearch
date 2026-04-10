# Modes of Reasoning: Project Analysis Report

**Project:** frankensearch
**Date:** 2026-04-08
**Modes Used:** 10 of 80 available
**Agents:** 10 Claude Opus 4.6
**Lead Agent:** Claude Opus 4.6 (1M context)

---

## 1. Executive Summary

This multi-perspective analysis deployed 10 reasoning modes against frankensearch, a two-tier hybrid search engine for Rust combining BM25 lexical and semantic vector search via Reciprocal Rank Fusion. The 10 agents collectively produced **96 findings** across systems-level feedback loops, contract consistency, security attack surfaces, failure modes, edge cases, stakeholder perspectives, design counterfactuals, code patterns, root causes, and cognitive biases.

### Key Takeaways

1. **Missing quality scores are treated as 0.0, systematically penalizing documents without quality-tier embeddings** -- a design choice that silently degrades relevance for partially-indexed corpora. Independently identified by 5 modes (Systems-Thinking, Root-Cause, Deductive, Edge-Case, Failure-Mode). This is the single highest-impact correctness issue.

2. **`TwoTierConfig` accepts arbitrary invalid values without validation**, producing silent empty results (`candidate_multiplier=0`) or degenerate scoring (`rrf_k=0`). Identified by 6 modes. The configuration surface is the project's largest undefended boundary.

3. **The `poll_immediate` pattern in Phase 1 creates a hidden sync-only constraint on ostensibly async traits**, preventing real async embedders (API-based, daemon-backed) from working. This architectural constraint is documented only in runtime error messages, not in type signatures or trait docs.

4. **Default parameters (K=60, quality_weight=0.7) are anchored, not empirically validated.** The CMA-ES optimization run confirmed defaults rather than exploring alternatives -- the optimizer's best nDCG (0.223) came from parameters 10% different from defaults, but the final output reverted to near-defaults.

5. **Memory-mapped FSVI files lack file-level locking for mutation paths**, enabling data races between concurrent soft-deletes, compaction, and search operations.

### Overall Confidence: 0.88
High confidence in convergent findings (5+ modes agreeing on core issues). Lower confidence in meta-reasoning findings (debiasing, counterfactual) which are inherently more speculative.

---

## 2. Methodology

### Why These 10 Modes?

| # | Mode | Code | Category | Selection Rationale |
|---|------|------|----------|-------------------|
| 1 | Systems-Thinking | F7 | Causal | Feedback loops between progressive phases, circuit breakers, and degradation paths |
| 2 | Root-Cause | F5 | Causal | Trace architectural symptoms to their deepest structural causes |
| 3 | Deductive | A1 | Formal | Verify that documented contracts match implementation |
| 4 | Inductive | B1 | Ampliative | Discover recurring patterns across 11 crates revealing hidden assumptions |
| 5 | Adversarial-Review | H2 | Strategic | Stress-test input validation, mmap safety, model supply chain |
| 6 | Failure-Mode | F4 | Causal | Systematic FMEA of progressive search degradation paths |
| 7 | Edge-Case | A8 | Formal | Boundary conditions in scoring, quantization, CJK tokenization |
| 8 | Perspective-Taking | I4 | Social | Library consumers, CLI users, AI agents, enterprise evaluators |
| 9 | Counterfactual | F3 | Causal | Evaluate whether 8 key design decisions remain optimal |
| 10 | Debiasing | L2 | Meta | Detect anchoring, confirmation, NIH, and complexity biases |

### Category Coverage

| Category | Modes Selected | Coverage |
|----------|---------------|----------|
| A: Formal/Deductive | 2 | Deductive (A1), Edge-Case (A8) |
| B: Ampliative | 1 | Inductive (B1) |
| F: Causal/Systems | 4 | Systems-Thinking (F7), Root-Cause (F5), Failure-Mode (F4), Counterfactual (F3) |
| H: Strategic | 1 | Adversarial-Review (H2) |
| I: Social/Interpretive | 1 | Perspective-Taking (I4) |
| L: Meta-Reasoning | 1 | Debiasing (L2) |

### Axis Coverage

| Axis | Covered By |
|------|-----------|
| Ampliative vs Non-ampliative | Inductive (ampliative) vs Deductive, Edge-Case (non-ampliative) |
| Descriptive vs Normative | Systems-Thinking, Root-Cause (descriptive) vs Perspective-Taking (normative) |
| Single-agent vs Multi-agent | Deductive, Edge-Case (single) vs Adversarial, Perspective-Taking (multi) |
| Belief vs Action | Debiasing, Counterfactual (belief) vs Failure-Mode (action) |
| Uncertainty vs Vagueness | Failure-Mode (uncertainty/probability) vs Inductive (vague patterns) |

---

## 3. Convergent Findings (High Confidence)

### C1: Missing Quality Scores Treated as 0.0, Systematically Penalizing Partially-Indexed Documents

**Supporting modes:** Systems-Thinking, Root-Cause, Deductive, Edge-Case, Failure-Mode
**Confidence:** 0.95

When a document has no quality-tier embedding (common during progressive indexing, quality-embedder failures, or partial builds), `quality_scores_for_hits` returns 0.0 via `unwrap_or(0.0)` (`two_tier.rs:411`). The blend function then applies `alpha * 0.0 + (1-alpha) * fast_score = 0.3 * fast_score`, destroying 70% of the document's relevance signal.

**Evidence from each mode:**
- **Systems-Thinking (F3, F4):** Identified the emergent ranking inversion -- a mediocre document in both tiers outranks a highly relevant fast-only document. Traced how `index_builder.rs:357-374` silently skips failed quality embeddings at `debug` level, creating invisible quality gaps.
- **Root-Cause (F6):** Traced to the conflation of "absent" with "zero similarity" -- `Option<f32>` should distinguish `None` (no data) from `Some(0.0)` (genuinely orthogonal).
- **Deductive (F9):** Proved the asymmetry is 2.33x: a quality-only document with normalized score 0.43 beats a fast-only document with score 1.0.
- **Edge-Case:** Confirmed degenerate behavior when all documents lack quality vectors.
- **Failure-Mode (F2):** Identified that dimension mismatch between build-time and search-time embedders compounds this -- fast-tier returns garbage scores while quality-tier catches the mismatch.

**Why convergence matters:** Five independent analytical frameworks -- emergent system behavior, causal tracing, logical deduction, boundary analysis, and failure enumeration -- all identify this as a ranking correctness problem. No mode found a mitigating factor.

**Recommended action:** Propagate `Option<f32>` through the blend pipeline. When quality score is `None`, use `alpha = 0.0` (fast-only blend) for that document rather than penalizing with 0.0. Add an `incomplete_embeddings` counter to `TwoTierMetrics`.

---

### C2: TwoTierConfig Accepts Invalid Values Without Validation

**Supporting modes:** Root-Cause, Deductive, Inductive, Edge-Case, Perspective-Taking, Debiasing
**Confidence:** 0.95

`TwoTierConfig` is the central control surface with 15+ tunable fields, but has no `validate()` method. Invalid values are silently accepted:

| Invalid Config | Effect |
|---------------|--------|
| `candidate_multiplier = 0` | Zero candidates fetched, empty results with no error |
| `rrf_k = 0.0` | Rank-0 score = 1.0, extreme sensitivity (harmonic series) |
| `quality_weight = -5.0` | Negative blend weight, nonsensical scores |
| `quality_timeout_ms = 0` | Immediate timeout, silently disables quality tier |
| `FRANKENSEARCH_QUALITY_WEIGHT=0.7f` | Env parse failure silently ignored, keeps default |

**Evidence from each mode:**
- **Root-Cause (F1):** Silent env override failure -- no log, no warning, no metric for parse errors.
- **Deductive (F2):** Proved `candidate_multiplier=0` causes `saturating_mul(0)=0` in `candidate_count()`, yielding empty results.
- **Inductive (F4):** Found that newer subsystems (DurabilityConfig, CollectorConfig) have `validate()` but TwoTierConfig -- the oldest and most central -- does not.
- **Edge-Case (F2):** Found sync/async inconsistency -- `SyncTwoTierSearcher` clamps `multiplier.max(1)` but async path does not.
- **Perspective (F8):** CLI users with env var typos get no feedback, running with unintended defaults for hours.
- **Debiasing:** The lack of validation is consistent with confirmation bias in testing (tests use valid configs).

**Recommended action:** Add `TwoTierConfig::validate() -> Result<(), SearchError>` enforcing: `candidate_multiplier >= 1`, `rrf_k > 0.0`, `quality_weight` in [0.0, 1.0], `quality_timeout_ms >= 10`. Call it from `TwoTierSearcher::new()`. Emit `tracing::warn!` on env parse failures.

---

### C3: Phase 1 sync-in-async Constraint is Invisible in the Type System

**Supporting modes:** Systems-Thinking, Root-Cause, Deductive, Inductive
**Confidence:** 0.92

Phase 1 uses `rayon::join` with `poll_immediate` to parallelize fast embedding and lexical search (`searcher.rs:856-884`). Any `Embedder` implementation that returns `Pending` (i.e., performs actual async I/O) gets an opaque runtime error: "only sync-in-async embedders are supported in the rayon path."

**Evidence from each mode:**
- **Systems-Thinking (F9):** Identified the hidden system boundary -- the `Embedder` trait is async but Phase 1 structurally requires sync implementations.
- **Root-Cause (F4):** Traced to the "NO Tokio" constraint forcing `rayon::join` as the parallelism mechanism, which cannot drive async futures.
- **Deductive (F4, F8):** The `Reranker` trait contract and `Embedder` dimension contract similarly lack enforcement.
- **Inductive (F2):** The pattern of 82 `_cx: &Cx` unused parameters confirms most implementations are already synchronous.

**Why convergence matters:** The type system promises async capability, but the architecture delivers sync-only execution in the critical fast path. This prevents API-based embedders, daemon-backed embedders, or any genuinely async implementation from working in Phase 1.

**Recommended action:** Either (a) add a `SyncEmbedder` marker trait that Phase 1 requires at compile time, making the constraint visible in types, or (b) provide an async Phase 1 path using asupersync structured concurrency instead of rayon for embedders that need it.

---

### C4: Memory-Mapped File Mutation Without Locking

**Supporting modes:** Adversarial-Review, Failure-Mode, Edge-Case
**Confidence:** 0.90

FSVI files are opened with `MmapMut` (read+write) and mutated in-place for soft-delete operations (`lib.rs:468-562`) without any file-level locking (`flock`/`fcntl`). Concurrent processes or threads can corrupt in-flight reads.

**Evidence from each mode:**
- **Adversarial (F2, F7, F10):** External file truncation causes SIGBUS; concurrent soft-deletes corrupt flags; Rayon parallel reads race with mutations.
- **Failure-Mode (F5):** Disk-full during index build leaves partial files that cause parse errors rather than graceful degradation.
- **Edge-Case:** Concurrent write + search produces inconsistent flag reads.

**Recommended action:** Use `Mmap` (read-only) for search paths. Add advisory file locking for mutation. Adopt atomic write pattern (write to temp, fsync, rename) for index builder.

---

### C5: Feature Flag Combinatorial Explosion with Sparse Cross-Feature Testing

**Supporting modes:** Root-Cause, Inductive, Adversarial, Perspective-Taking
**Confidence:** 0.88

16 feature flags create 2^11+ theoretical combinations. `auto_detect.rs` alone has 133 `#[cfg(...)]` blocks (one per ~7 lines). CI does not evidence combinatorial feature testing.

**Evidence from each mode:**
- **Root-Cause (F5):** `auto_detect_with_policy` has two entirely separate implementations differing only by `download` flag.
- **Inductive (F6):** The `fsfs` binary has its own `embedded-models` feature composing embed features differently from the library.
- **Adversarial:** Attack surface varies by feature combination -- security properties of one combo don't transfer.
- **Perspective (F10):** Enterprise evaluators see 1,400+ transitive deps under `full` feature set.

**Recommended action:** Add CI matrix testing for at minimum: `default`, `semantic`, `hybrid`, `full`. Refactor `auto_detect.rs` to use a trait-based strategy pattern with cfg-guarded registration rather than inline conditional compilation.

---

### C6: Default Parameters Anchored, Not Validated

**Supporting modes:** Debiasing, Counterfactual, Deductive
**Confidence:** 0.85

K=60 and quality_weight=0.7 are presented as empirically validated, but the evidence is circular:

- CMA-ES optimization found K=60.04 (0.07% from default) because it ran on synthetic 100-doc corpus with hash embedders
- The adaptive fusion prior is literally `N(60, 10^2)` -- a Gaussian centered on 60
- Generation 5 found quality_weight=0.77 with the best nDCG (0.223) but final output reverted to ~0.7
- No test compares RRF against CombSUM, CombMNZ, or other fusion methods

**Evidence from each mode:**
- **Debiasing (F1, F9, F2):** Anchoring bias (K=60 from Cormack 2009 TREC metasearch, different domain), confirmation bias (optimizer validates defaults), quality_weight=0.7 is prior not finding.
- **Counterfactual (F3):** Acknowledged RRF is reasonable but noted the optimization couldn't improve beyond defaults.
- **Deductive (F6):** K=0 is allowed by runtime but rejected by env override -- inconsistent validation.

**Recommended action:** Run the parameter optimizer with production-quality embedders on realistic corpora. Test at least K={20, 40, 60, 80, 100} and quality_weight={0.5, 0.6, 0.7, 0.8, 0.9} with nDCG evaluation. Document results either confirming or updating defaults.

---

### C7: Phase Gate Decision is Irrecoverable, Creating Permanent Quality-Tier Lockout

**Supporting modes:** Systems-Thinking, Failure-Mode
**Confidence:** 0.90

The `PhaseGate` implements e-process sequential testing. Once it decides `SkipQuality` (`phase_gate.rs:101`), `update()` becomes a permanent no-op. Combined with the circuit breaker's trip behavior and adaptive fusion's monotone posterior, three independent mechanisms push toward fast-only mode with no recovery path.

**Evidence:**
- **Systems-Thinking (F1, F2):** Triple reinforcing loop -- circuit breaker, phase gate, and adaptive fusion all observe the same signal and lock in the same direction. Phase gate is the most severe because its decision is literally `Option<PhaseDecision>` that never resets.
- **Failure-Mode (F4):** Circuit breaker consecutive_failures counter has a TOCTOU race under concurrency.

**Recommended action:** Add a `reset()` or periodic decay mechanism to `PhaseGate`. Consider time-based expiry for the e-value accumulator to allow recovery after transient quality-tier issues.

---

## 4. Divergent Findings

### D1: Is the asupersync Decision Still Optimal?

**Position A (Counterfactual, confidence 0.45):** The structured concurrency benefits (Cx, LabRuntime, cancel-correctness) are real but the ecosystem cost is the single highest adoption friction. A Tokio compatibility layer would capture most upside of both worlds.

**Position B (Debiasing, confidence 0.80):** This is sunk-cost reasoning. asupersync is a single-author, bus-factor-1 dependency pinned to a git revision. Each new crate deepens the lock-in. The "no Tokio" policy is stated as doctrine, not as a continuously evaluated tradeoff.

**Position C (Inductive, observation):** 82 instances of `_cx: &Cx` (unused parameter) suggest most code doesn't actually use structured concurrency -- it threads Cx through as ceremony.

**Analysis:** These modes are operating at different levels -- Counterfactual evaluates the technical merits, Debiasing evaluates the decision process, Inductive observes the implementation reality. All three converge on the finding that asupersync creates friction, but disagree on severity. The technical merits of structured concurrency are real for the searcher's Phase 1/2 timeout and cancellation paths, but the 82 unused `_cx` parameters suggest the benefits are concentrated in the fusion crate while the cost is spread across all 11 crates.

**Resolution strategy:** Contextualize -- both positions are valid at different scales. For the library's internal architecture, asupersync provides genuine value. For the library's adoption surface, it creates genuine friction. A `tokio-compat` feature flag is the pragmatic synthesis.

---

### D2: Is the Complexity Justified?

**Position A (Debiasing, confidence 0.85-0.90):** The fusion crate's 27,460 lines with Bayesian adaptive fusion, conformal prediction, sequential testing, MMR, federated search, and calibration are wildly disproportionate for a v0.1 local search library with one consumer (`fsfs`). This is planning fallacy -- building for a future that hasn't proven product-market fit.

**Position B (Counterfactual, confidence 0.78):** The 11-crate structure with feature flags enables consumers to pick exactly what they need. The alternative (monolith) would be simpler to publish but worse for selective dependency. The current structure is defensible.

**Position C (Perspective, observation):** From a maintainer perspective, the 31-module fsfs crate is the real navigability concern, not the inter-crate structure.

**Analysis:** The Debiasing mode's critique is at the feature level (too many advanced features for v0.1), while Counterfactual evaluates the structural level (crate decomposition). Both can be correct simultaneously -- the crate structure is sound but overfilled with speculative features. The dead config knobs (`mrl_search_dims`, `mrl_rescore_top_k` -- Inductive F3) provide concrete evidence of unfinished feature integration.

**Resolution strategy:** Separate -- crate structure is defensible (keep), but speculative features (conformal prediction, federated search, adaptive fusion) should be documented as experimental or gated behind a separate feature flag rather than being in the default compilation path.

---

## 5. Supported Findings (2-Mode Agreement)

### S1: `TwoTierConfig::optimized()` Uses Compile-Time Path, Broken Outside Workspace
**Modes:** Root-Cause (F7), Perspective-Taking (F7)
**Confidence:** 0.90

`env!("CARGO_MANIFEST_DIR")` bakes the build machine's absolute path into the binary. When installed via `cargo install`, the path doesn't exist and `optimized()` silently returns defaults. The README documents this as a user-facing feature without noting the limitation.

### S2: Dead Config Knobs (mrl_search_dims, mrl_rescore_top_k)
**Modes:** Inductive (F3), Perspective-Taking (F2)
**Confidence:** 0.90

These fields exist in `TwoTierConfig`, are serialized/deserialized/tested, and documented -- but never read by any production code path. The MRL module has its own `MrlConfig` struct. Users setting these knobs via TOML or env vars get no effect.

### S3: Underscore Replacement in Canonicalization Destroys Snake-Case Identifiers
**Modes:** Edge-Case (F6), Edge-Case (F9 related -- mixed-script CJK)
**Confidence:** 0.85

`strip_markdown_line` replaces ALL underscores with spaces (`canonicalize.rs:185`), turning `my_function_name` into `my function name`. This degrades lexical recall for snake_case identifiers in non-code-block text -- particularly damaging for `QueryClass::Identifier` queries that lean on lexical search.

### S4: Non-Transactional Batch Ingest
**Modes:** Root-Cause (F3), Failure-Mode (implicit)
**Confidence:** 0.88

`ingest_batch` iterates individual `ingest()` calls, each with its own transaction. If document M+1 fails with `QueueFull`, documents 1..M are committed and M+2..N are never attempted. Partial ingestion creates inconsistent state with no caller feedback on which documents succeeded.

### S5: Unbounded API Response Body Accumulation
**Modes:** Adversarial (F1)
**Confidence:** 0.88 (single mode but high evidence quality)

`api_embedder.rs:212-223` accumulates the entire HTTP response body into `Vec<u8>` with no size limit. A malicious or compromised API endpoint can OOM the process.

---

## 6. Unique Insights by Mode

These findings were produced by a single mode and represent the distinctive value of that analytical lens.

### Systems-Thinking Only

**Kendall Tau as quality proxy rewards noise over accuracy (F5).** The adaptive fusion uses `tau < 0.98` as the signal that quality improved results. But tau measures rank *change*, not rank *quality*. A quality tier that randomly reshuffles results appears "successful" and increases blend weight toward quality. Accuracy (high tau) is penalized. *This creates a reinforcing loop where noise is rewarded.*

**Embedding cache is FIFO, not LRU (F7).** Under search-ahead/typeahead, earlier partial queries get evicted even if the user backspaces and retypes them. LRU would naturally retain frequently-queried terms.

### Adversarial Only

**TOCTOU in model verification cache (F3).** After legitimate verification writes a `.verified` marker, an attacker can replace model files while preserving mtimes. Subsequent loads skip SHA-256 verification. ONNX models are arbitrary computation graphs -- a tampered model could exfiltrate data.

**Regex query cache as memory amplification vector (F4).** 128-entry global `HashMap` with clear-all eviction. Adversarial wildcard patterns can produce megabyte-scale compiled DFA automata.

### Inductive Only

**SubsystemError funnel (F1).** 308 occurrences of `SearchError::SubsystemError` across 33 files. Every downstream crate tunnels structurally different errors through one opaque variant distinguished by a string label. Programmatic recovery (retry on db-busy, skip on schema-mismatch) requires fragile `Box<dyn Error>` downcasting.

**2,443 `.unwrap()` calls in production code (F9).** `index/lib.rs`: 110 unwraps, `runtime.rs`: 75 unwraps. For a library crate, panics propagating to consumers are a reliability risk.

### Failure-Mode Only

**FSVI verify() unconditionally reports `repairable: true` (F1).** It never checks whether the sidecar has enough FEC repair symbols. Repairability is only discovered when `repair()` is called and may then fail with `Unrecoverable`.

**No WAL/index format versioning or migration path (F6).** WAL uses `WAL_VERSION = 1` with hard rejection on mismatch. No migration logic. When the format changes, all existing indices become unrecoverable.

### Counterfactual Only

**Hash embedder is architecturally load-bearing (F8, confidence 0.95).** Enables zero-dependency testing, instant first-run experience, graceful degradation floor, and CI without model downloads. Its JL-projection variant provides formal distance-preservation guarantees. Removing it would break the entire testing and onboarding story.

### Debiasing Only

**Tests validate the design, never challenge it (F10).** Zero tests compare RRF against CombSUM/CombMNZ. Zero tests compare two-tier vs single-tier on latency/quality tradeoff. Zero tests evaluate whether f16 quality loss is actually <1% on target corpus types. The test suite is an echo chamber.

### Edge-Case Only

**All-NaN training dimensions produce INFINITY in dequantized vectors (F10).** `ScalarQuantizer::fit()` leaves mins at `f32::INFINITY` for all-NaN dimensions. Dequantized vectors contain `f32::INFINITY` that propagates through dot products.

**Mixed CJK/Latin tokens bypass bigram decomposition entirely (F9).** A token like "hello世界" passes through unchanged because `chars.all(is_cjk)` is false. Common in Japanese text mixing kanji with romaji.

---

## 7. Risk Assessment

| # | Risk | Severity | Likelihood | Supporting Modes | Action |
|---|------|----------|-----------|-----------------|--------|
| R1 | Silent relevance degradation from missing quality scores | Critical | High | 5 modes | Fix blend to use Option<f32> |
| R2 | Invalid config produces silent empty results | High | Medium | 6 modes | Add validate() |
| R3 | mmap data races from concurrent mutation | High | Medium | 3 modes | Add file locking |
| R4 | Sync-only constraint hidden in async API | High | Medium | 4 modes | Expose in type system |
| R5 | Phase gate permanent lockout | High | Medium | 2 modes | Add reset/decay |
| R6 | No index format migration path | Critical | Low (now) | 1 mode | Add version negotiation |
| R7 | Feature flag combination bugs | Medium | Medium | 4 modes | CI matrix testing |
| R8 | Model verification TOCTOU | Medium | Low | 1 mode | Re-verify on load |
| R9 | Non-transactional batch ingest | Medium | Medium | 2 modes | Wrap in outer transaction |
| R10 | Unbounded API response body | Medium | Low | 1 mode | Add size cap |

---

## 8. Recommendations (Prioritized)

| Priority | Recommendation | Effort | Supporting Modes | Expected Benefit |
|----------|---------------|--------|-----------------|-----------------|
| P0 | Fix blend to handle `None` quality scores without penalty | Medium | 5 | Correct ranking for partially-indexed corpora |
| P0 | Add `TwoTierConfig::validate()` with enforcement in constructor | Low | 6 | Prevent silent misconfiguration |
| P1 | Add advisory file locking for FSVI mutation paths | Medium | 3 | Prevent concurrent-access corruption |
| P1 | Make Phase 1 sync constraint visible in types or provide async path | High | 4 | Enable API-based embedders in Phase 1 |
| P1 | Add periodic reset/decay to PhaseGate | Low | 2 | Prevent permanent quality-tier lockout |
| P1 | Emit tracing::warn on env var parse failures in config | Low | 3 | Surface configuration errors |
| P2 | Add format version negotiation and migration tooling | Medium | 1 | Prevent index loss on format upgrade |
| P2 | Replace Kendall tau proxy with relevance-based signal in adaptive fusion | Medium | 1 | Stop rewarding rank noise |
| P2 | Refactor auto_detect.rs to trait-based strategy pattern | Medium | 2 | Reduce cfg combinatorial risk |
| P2 | Run parameter optimization with production embedders on realistic corpora | Medium | 3 | Validate or update K=60 and weight=0.7 |
| P3 | Add response body size cap to API embedder | Low | 1 | Prevent OOM from malicious API |
| P3 | Add CI matrix for feature flag combinations | Low | 4 | Catch combination-specific bugs |
| P3 | Remove or connect dead config knobs (mrl_*) | Low | 2 | Prevent user confusion |
| P3 | Fix optimized() to use runtime config path | Low | 2 | Make it work outside workspace |

---

## 9. New Ideas and Extensions

| Idea | Source Mode | Innovation Level | Description |
|------|-----------|-----------------|-------------|
| Per-document blend weight | Systems-Thinking | Significant | Propagate `has_quality_embedding` metadata from index to searcher; use `alpha=0.0` for missing-quality documents instead of penalizing |
| Tokio compatibility layer | Counterfactual | Significant | `tokio-compat` feature flag wrapping public API in Tokio-compatible futures, keeping asupersync internally |
| LRU embedding cache | Systems-Thinking | Incremental | Replace FIFO `CachedEmbedder` with LRU for typeahead workload patterns |
| Content-addressed model storage | Adversarial | Significant | Use SHA-256 hash as filename, eliminating TOCTOU verification gap |
| Typed subsystem errors | Inductive | Significant | Replace `SubsystemError` string-tag pattern with `StorageError`, `DurabilityError`, etc. for programmatic recovery |
| Alternative fusion comparison | Debiasing | Significant | Test suite comparing RRF vs CombSUM vs CombMNZ to validate or challenge the fusion choice |
| Mixed-script CJK tokenization | Edge-Case | Incremental | Split mixed CJK/Latin tokens before bigram decomposition for better CJK recall |

---

## 10. Assumptions Ledger

Unstated assumptions surfaced across all modes:

| Assumption | Surfaced By | Risk If Wrong |
|-----------|------------|---------------|
| Corpus size is <100K documents for typical use | Counterfactual, Debiasing | Brute-force search degrades; ANN path undertested |
| Quality embedder failures are transient | Systems-Thinking | Phase gate locks out quality permanently on persistent failures |
| All Embedder implementations are synchronous | Root-Cause, Systems-Thinking | Phase 1 fails at runtime with confusing error |
| Hash embedder provides meaningful fallback | Counterfactual | Near-random results for semantic queries without warning |
| K=60 is optimal across domains | Debiasing | Different corpus types may need very different K values |
| Single-writer access to FSVI files | Adversarial | Data races from concurrent processes |
| Model files are not tampered after verification | Adversarial | TOCTOU allows post-verification replacement |
| Nightly Rust toolchain is acceptable for consumers | Perspective | Enterprise users require stable toolchain |

---

## 11. Open Questions for Project Owner

1. **Is the quality-score-as-0.0 penalty intentional?** If so, should it be documented? If not, the fix is straightforward (C1).
2. **What is the realistic target corpus size?** Many design decisions (brute-force default, HNSW threshold, memory budget) depend on this.
3. **Is a Tokio compatibility layer planned?** This is the single highest-friction adoption barrier (D1).
4. **Has the parameter optimizer been run with production embedders?** The current results only validate defaults on synthetic data (C6).
5. **Is the phase gate's irrecoverable decision intentional or an oversight?** The e-value accumulator never resets (C7).
6. **What is the migration plan when FSVI format or WAL version changes?** Currently, existing indices become unrecoverable.

---

## 12. Confidence Matrix

| Finding | Confidence | Supporting Modes | Dissenting Modes |
|---------|-----------|-----------------|-----------------|
| C1: Quality 0.0 penalty | 0.95 | Systems, Root-Cause, Deductive, Edge, Failure | None |
| C2: Config validation gap | 0.95 | Root-Cause, Deductive, Inductive, Edge, Perspective, Debiasing | None |
| C3: Sync-in-async constraint | 0.92 | Systems, Root-Cause, Deductive, Inductive | None |
| C4: mmap mutation safety | 0.90 | Adversarial, Failure-Mode, Edge | None |
| C5: Feature flag complexity | 0.88 | Root-Cause, Inductive, Adversarial, Perspective | None |
| C6: Parameter anchoring | 0.85 | Debiasing, Counterfactual, Deductive | None (but lower confidence due to speculative nature) |
| C7: Phase gate lockout | 0.90 | Systems-Thinking, Failure-Mode | None |
| D1: asupersync tradeoff | 0.60 | All positions valid at different levels | Counterfactual vs Debiasing tension |
| D2: Complexity justification | 0.70 | Debiasing (too much) vs Counterfactual (structure ok) | Different-level disagreement |

---

## 13. Contribution Scoreboard

| Mode | Findings | Unique Findings | Evidence Quality | Calibration | Mode Fidelity | Score |
|------|----------|----------------|-----------------|-------------|---------------|-------|
| Systems-Thinking (F7) | 10 | 3 (F5 tau proxy, F7 FIFO cache, F8 conformal cascade) | 0.95 | 0.90 | 0.95 | **0.91** |
| Root-Cause (F5) | 9 | 2 (F3 non-transactional batch, F8 telemetry cloning) | 0.95 | 0.90 | 0.90 | **0.88** |
| Deductive (A1) | 10 | 2 (F1 cosine_similarity doc, F3 in_both_sources misnomer) | 0.95 | 0.95 | 0.95 | **0.87** |
| Inductive (B1) | 9 | 3 (F1 SubsystemError, F7 repro_ pattern, F8 dual Cx imports) | 0.90 | 0.85 | 0.90 | **0.85** |
| Adversarial (H2) | 10 | 4 (F1 API body, F3 TOCTOU, F4 regex cache, F5 CDN trust) | 0.90 | 0.85 | 0.95 | **0.87** |
| Failure-Mode (F4) | 10 | 3 (F1 repairable:true, F6 no migration, F8 Tantivy unprotected) | 0.90 | 0.90 | 0.95 | **0.88** |
| Edge-Case (A8) | 10 | 3 (F1 zero-width, F9 mixed CJK, F10 all-NaN quantizer) | 0.90 | 0.85 | 0.90 | **0.84** |
| Perspective (I4) | 10 | 2 (F3 stream protocol positive, F4 license rider) | 0.85 | 0.85 | 0.90 | **0.82** |
| Counterfactual (F3) | 8 | 2 (F8 hash embedder value, F6 FSVI vs vector DB) | 0.85 | 0.85 | 0.90 | **0.80** |
| Debiasing (L2) | 10 | 3 (F4 survivorship, F7 planning fallacy, F10 test echo chamber) | 0.80 | 0.85 | 0.90 | **0.79** |

**Diversity metric:** 6 of 12 categories covered. 5 of 7 axes represented. At least 2 opposing pairs (Deductive vs Inductive, Adversarial vs Perspective).

**Most productive modes:** Systems-Thinking and Failure-Mode -- their causal lens was well-suited to the progressive search architecture with its multiple feedback loops and degradation paths.

**Most unique contributions:** Adversarial-Review (4 unique findings) -- security analysis surfaces issues that no other mode considers. Debiasing (3 unique metacognitive findings) -- the anchoring and confirmation bias findings are invisible from within the design.

---

## 14. Mode Performance Notes

**Systems-Thinking** was the standout performer. The two-tier progressive architecture with circuit breakers, phase gates, and adaptive fusion is fundamentally a *system with feedback loops*, making this lens maximally productive. The reinforcing degradation loop (F1) and Kendall tau proxy problem (F5) are the kind of emergent-behavior findings only systems-thinking reveals.

**Debiasing** was the most provocative. Its findings are harder to verify (by nature -- biases are invisible to the biased) but the anchoring analysis of K=60 and quality_weight=0.7 is backed by concrete evidence from the optimization logs.

**Perspective-Taking** identified the strongest *positive* finding: the streaming protocol is exceptionally well-designed for agent consumption (versioned schema, retry directives, failure categorization). This demonstrates that the project's design rigor is high in areas where the team is most focused.

**Counterfactual** had the most nuanced output. The asupersync evaluation at confidence 0.45 ("debatable whether current decision is optimal") is honest calibration -- most modes would have collapsed to a binary recommendation.

---

## 15. Mode Selection Retrospective

**Would I choose different modes with hindsight?**

The selection was well-calibrated for this project. The causal modes (Systems-Thinking, Root-Cause, Failure-Mode, Counterfactual) were particularly productive because the progressive search architecture is fundamentally about causally connected phases.

**What I would add if I had 12 agents:**
- **Bayesian (B3):** To formally evaluate the parameter sensitivity question (is the nDCG landscape actually flat near K=60, or is the optimizer too weak?)
- **Type-Theoretic (A7):** To analyze the type-level gaps more systematically (the sync/async constraint, the unverified dimension contract, the unserializable SearchPhase)

**What I would drop:**
- None. Every mode produced at least 2 unique findings. The coverage was efficient.

---

## 16. Appendix: Provenance Index

| Finding | Source Mode(s) | Report Section | Status |
|---------|---------------|----------------|--------|
| Quality 0.0 penalty | Sys-F3, Sys-F4, RC-F6, Ded-F9, Edge-F5, FM-F2 | C1 | KERNEL |
| Config validation gap | RC-F1, Ded-F2, Ind-F4, Edge-F2, Persp-F8, Debias | C2 | KERNEL |
| Sync-in-async constraint | Sys-F9, RC-F4, Ded-F4, Ind-F2 | C3 | KERNEL |
| mmap mutation safety | Adv-F2, Adv-F7, Adv-F10, FM-F5 | C4 | KERNEL |
| Feature flag complexity | RC-F5, Ind-F6, Persp-F10 | C5 | KERNEL |
| Parameter anchoring | Debias-F1, Debias-F9, CF-F3, Ded-F6 | C6 | KERNEL |
| Phase gate lockout | Sys-F1, Sys-F2, FM-F4 | C7 | KERNEL |
| asupersync tradeoff | CF-F1, Debias-F5, Debias-F8, Ind-F2 | D1 | DISPUTED |
| Complexity justification | Debias-F3, Debias-F7, CF-F5, Persp-F6 | D2 | DISPUTED |
| optimized() compile-time path | RC-F7, Persp-F7 | S1 | SUPPORTED |
| Dead mrl_* config knobs | Ind-F3, Persp-F2 | S2 | SUPPORTED |
| Underscore destruction | Edge-F6 | S3 | SUPPORTED |
| Non-transactional batch | RC-F3 | S4 | SUPPORTED |
| Kendall tau proxy rewards noise | Sys-F5 | Unique-Systems | HYPOTHESIS |
| FIFO cache eviction | Sys-F7 | Unique-Systems | HYPOTHESIS |
| TOCTOU model verification | Adv-F3 | Unique-Adversarial | HYPOTHESIS |
| SubsystemError funnel | Ind-F1 | Unique-Inductive | HYPOTHESIS |
| FSVI repairable:true always | FM-F1 | Unique-Failure | HYPOTHESIS |
| No WAL migration path | FM-F6 | Unique-Failure | HYPOTHESIS |
| Hash embedder is load-bearing | CF-F8 | Unique-Counterfactual | HYPOTHESIS |
| Tests never challenge design | Debias-F10 | Unique-Debiasing | HYPOTHESIS |
| All-NaN quantizer corruption | Edge-F10 | Unique-Edge | HYPOTHESIS |
| Mixed CJK/Latin bypass | Edge-F9 | Unique-Edge | HYPOTHESIS |
| License rider adoption friction | Persp-F4 | Unique-Perspective | HYPOTHESIS |
| Stream protocol well-designed | Persp-F3 | Unique-Perspective | POSITIVE |
