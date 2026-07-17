# WIZARD IDEAS — CC (Claude Fable)

Dueling-idea-wizards deliverable for the Quill lexical engine plan
(`COMPREHENSIVE_PLAN_FOR_THE_QUILL_LEXICAL_ENGINE.md`, bead family `bd-quill-*`).
All ideas are additive to or corrective of the design of record; all respect the hard
constraints (zero new external deps, no unsafe beyond the mmap allowance, asupersync-only,
conformance-before-default-flip).

---

## Idea 1 — Cross-Process Writer Lock + "Readers Never GC": close the multi-process data-loss hole

**What it is.** The plan has a genuine correctness hole that is also a *regression versus the
incumbent*: tantivy 0.26.1 acquires a directory lockfile (`INDEX_WRITER_LOCK`,
`tantivy-0.26.1/src/index/index.rs:545`) on `Index::writer`, so today frankensearch gets
cross-process single-writer protection for free. Quill's plan (§11) serializes writers only with
an **in-process** asupersync `Mutex` and — worse — makes `open()` = recover, which **GCs every
unreferenced `seg-*.fslx` / `.tmp-*` file** (§11.4, bead quill-e3.3). Combine those with how this
repo actually runs (agent swarms; `fsfs search` invoked concurrently with an `fsfs index --watch`
session on the same index) and you get a concrete destruction scenario:

1. Process A (watch mode) seals `seg-X.fslx`, fsyncs it, has not yet published MANIFEST gen N+1.
2. Process B runs `fsfs search`, calls `QuillIndex::open()`, sees `seg-X` unreferenced by gen N,
   and **deletes it** per the crash-only GC doctrine.
3. Process A publishes gen N+1 referencing a file that no longer exists. The index is broken; the
   "committed docs durable" recovery property is violated by a *reader*.

A second scenario needs no seal window at all: two writers (watch session + one-shot `fsfs index`)
both publish MANIFEST via atomic rename; last writer silently drops the other's segments.

The fix is a three-part protocol, all implementable with `std` + existing deps:

1. **Read-only opens never GC and never write.** Split the API: `QuillIndex::open()` (reader:
   validate, fall back to `MANIFEST.prev`, build snapshot, *touch nothing*) vs
   `QuillIndex::open_writer()` (acquires the writer lock, then — and only then — runs recovery GC).
   GC becomes writer-exclusive by construction, which kills scenario 1 outright.
2. **Advisory writer lockfile with heartbeat + staleness takeover.** `LOCK` file created with
   `OpenOptions::create_new` (O_EXCL) containing `{pid, engine_version, created_unix, nonce}`;
   the owning writer refreshes mtime on a slow heartbeat (piggybacked on seal/commit, plus a
   region-scoped timer task). Takeover only if mtime is stale beyond a generous threshold
   (e.g. 10× heartbeat), by rename-then-verify (rename `LOCK` → `LOCK.stale-<nonce>`, re-create,
   re-verify). No flock/fcntl needed, so no new deps and no unsafe; the known imperfections of
   lockfiles are acceptable for an *advisory* lock whose failure mode is falling back to part 3.
3. **Publish-time CAS on generation.** Before renaming the new MANIFEST into place, re-read the
   current MANIFEST and verify `generation == expected`; on mismatch, fail the commit with a typed
   `SearchError` (`writer conflict: index advanced underneath us`) instead of silently clobbering.
   This is the belt-and-braces detection layer for the window a lockfile can't seal.

Additionally, keep a **GC grace window** (never delete unreferenced files younger than N seconds)
as cheap defense-in-depth for the takeover race.

**Why it wins.** This is the only idea in this document that fixes a way for Quill to *lose
committed or in-flight data in normal operation* while every gauntlet lane stays green — the crash
matrix (quill-e3.9) is single-process by construction and will never see it, and the incumbent's
lockfile means the behavior is a silent downgrade nobody will think to test for. It converts an
undefined multi-process behavior into a typed, documented contract (`LockBusy`-equivalent error,
matching the §3.1 error-taxonomy row that already acknowledges writer-lock errors exist). Cost is
tiny (~300 LOC + tests); the payoff is not shipping a search engine that eats indexes when two
agents touch the same repo.

**Implementation sketch.**
- New bead under quill-e3 (Keeper), blocking quill-e3.3 (GC) and quill-e7.1 (trait impls):
  `quill-e3.10: cross-process writer lock, reader/writer open split, publish CAS, GC grace`.
- FSLX registry note: `LOCK` and `LOCK.stale-*` join the index-directory naming schema (GC must
  recognize, never collect a live `LOCK`).
- Tests: two-process integration test (spawn a second process via `std::process::Command` in a
  `#[ignore]`-by-default lane; in-process simulation via two `QuillIndex` instances on one dir for
  the fast lane); LabRuntime kill-point coverage for "crash while holding LOCK" (staleness
  takeover works); publish-CAS conflict test; reader-open-during-seal property test asserting the
  file inventory is untouched.
- fsfs surfaces the busy case: `fsfs index` against a locked index reports the owning PID instead
  of corrupting.

**Risk/cost.** Low. Lockfile staleness heuristics are the classic weak point, but the design never
relies on the lock for correctness of *published* state (CAS + readers-never-GC carry that);
the lock only prevents wasted work and provides good errors. ~2–4 agent-days including tests.

**Confidence: 0.9.** The failure scenario is checkable by reading the plan's own §11.4 against
bead e3.3; the incumbent-regression argument is verifiable in tantivy's source; the mitigation is
standard practice (tantivy, Lucene, LMDB all do a variant of it).

---

## Idea 2 — "Skeleton-First" resequencing: land a scalar reference engine as an explicit G1a milestone, then land every leapfrog bet as a measured lever

**What it is.** A bead-graph restructure. The plan already mandates a scalar reference
implementation for *every* SIMD/pruned kernel (SIMD≡scalar bit-parity gates in e2.2, pruned≡
exhaustive in e4.4, delta≡sealed in e5.5) — but the graph is sequenced with the optimized
components on the critical path: e4.1 (query parser) is *blocked by* e1.1 (SIMD tokenizer),
e6.2 (oracle differential runner) transitively waits on collectors, parsers, and codecs in their
final form. Nothing end-to-end exists until most of E1–E4 is done; the gauntlet — the project's
single most important risk-retirement asset — has no subject until late.

Invert that: define **G1a, the scalar skeleton** — a deliberately boring, single-shard,
delta-free engine that is *semantically complete* for the §3.1 surface:

- Tokenizer: a direct port of the shipping scalar `FrankensearchTokenizer` logic (it already
  exists in `lexical/src/lib.rs:466`; porting it is hours, and it is byte-parity-correct by
  construction since it *is* the specification).
- Accumulator: the simple hash-accumulate path (which e1.7's A/B needs as a comparator anyway).
- Codecs: scalar FOR encode/decode (already required as the parity baseline).
- Query: parser + exhaustive top-k kernel only (no MaxScore/BMW — exhaustive is already the
  reference side of e4.4's differential).
- Keeper: full FSLX/MANIFEST/crash-only open (these are format contracts, not optimizations —
  they cannot be stubbed).

Then re-type the optimized kernels as what they truly are under the house doctrine: **one-lever-
per-change perf beads** against a working, gauntlet-green baseline — SIMD tokenizer (vs scalar,
ledgered), columnar-radix flush (vs hash-accumulate, the e1.7 A/B becomes the *landing* decision),
MaxScore/BMW (vs exhaustive), delta segment (vs seal-per-batch), segment-parallel fan-out.

**Why it wins.** Three compounding effects:

1. **The gauntlet gets a subject months earlier.** Oracle differentials, the 90-test behavioral
   harvest, metamorphic suites, and crash matrices all start retiring the *actual* top risk
   (rank-parity, parser corner cases — the plan's own #1 risk row) while the perf kernels are
   still being written. Conformance bugs found on a 3-KLOC skeleton are cheap; the same bugs found
   inside a SIMD radix pipeline are expensive and ambiguous (semantics bug or kernel bug?).
2. **Every leapfrog bet becomes falsifiable in isolation.** This is exactly the workflow this
   repo has proven over ~50 ledgered perf outcomes: baseline first, one lever, A/B, ledger. The
   plan pays lip service (e1.7) but the graph shape contradicts it. If columnar-radix loses on
   real corpora (the plan's own risk table flags this as plausible), the skeleton *is* the SPIMI
   fallback — a config choice, not a rewrite under deadline.
3. **Swarm parallelism improves.** Post-G1a, the lever beads (SIMD tokenizer, radix flush, BMW,
   delta, fan-out) are genuinely independent — five agents can hold five non-conflicting file
   reservations against a green baseline, instead of serializing through a half-integrated stack.

**Implementation sketch.**
- New milestone bead `quill-g1a-scalar-skeleton` (task, P1) depending on: e1.0, scalar-subset
  splits of e1.1/e1.3/e1.4, e2.1, scalar half of e2.2, e2.5, e2.6, e3.1–e3.3, e4.1–e4.3, e4.7.
- Graph edits: cut e4.1's dep on e1.1 (the parser needs the *analyzer contract*, not the SIMD
  implementation); make e6.2 (differential runner) depend on G1a instead of final Argus; re-type
  e1.1-SIMD / e1.5 / e4.4 / e5.* as levers depending on G1a; e1.7's A/B becomes the acceptance
  test for e1.5.
- G1 (plan §18) splits into G1a (skeleton green on fast gauntlet) and G1b (bets landed with
  ledger entries). QG gates unchanged — they bind at G3 as before.

**Risk/cost.** The main objection is schedule optics: the skeleton "wastes" effort on paths that
will be replaced. But ~90% of the skeleton is code the plan already requires as reference/parity
baselines; the marginal cost is wiring it end-to-end early, which the metamorphic and crash suites
need anyway. Slight risk that skeleton-era format decisions ossify — mitigated because FSLX v1 is
already fully specified in e0.2 and doesn't change under this resequencing.

**Confidence: 0.8.** The dependency-graph facts are checkable (`br show bd-quill-e4-argus-3ycz.1`
lists the e1.1 edge); the methodology claim is backed by this repo's own ledger discipline and by
the plan's own risk table.

---

## Idea 3 — Shadow-Oracle mode in fsfs: dual-run Quill against tantivy on real workloads during G2, serving incumbent results

**What it is.** An opt-in `ShadowLexicalSearch` wrapper implementing `LexicalSearch` that holds
two engines — `serving` (tantivy, pre-flip) and `shadow` (Quill) — runs every real query against
both, serves the incumbent's results untouched, and feeds divergences through the gauntlet's
canonicalizing comparator into an append-only local report (`.quill-shadow/divergences.jsonl`,
MismatchSignature-deduped exactly like quill-e6.2). Config:
`FRANKENSEARCH_LEXICAL_SHADOW={off|quill}` + a `TwoTierConfig` field; surfaced in `fsfs status`
and `fsfs doctor` ("shadow: 14,203 queries compared, 0 unclassified divergences, 3 known
ScoreEpsilon ties").

The offline gauntlet (E6) validates Quill against *corpora and queries we thought to generate*.
Shadow mode validates it against the only distribution that ultimately matters: the real indexes
(this repo, the users' monorepos, the RCH fleet's checkouts) and the real query stream (agents
hammering `fsfs search` all day, with all the weird identifiers, paths, operator soup, and
truncation-boundary queries that agents actually type). The plan's own top correctness risk —
"parser corner cases" (§20 row 1) — is precisely the risk that synthetic query-class generators
are weakest against and live traffic is strongest against.

**Why it wins.**
- **It converts daily dogfooding into conformance evidence at near-zero marginal effort.** This
  project's agents run fsfs constantly; every one of those searches becomes a free differential
  test during the entire G2 window. Weeks of shadow-clean operation across the fleet is *stronger*
  flip evidence than any fixed corpus, because it is adversarially sampled by reality.
- **Zero user risk.** Incumbent results are served byte-identically; shadow comparison runs in a
  region-scoped background task (asupersync, cancel-safe), so even a Quill panic or hang cannot
  affect the serving path (catch via task outcome; a `Panicked` outcome is itself a logged
  finding — `Outcome`'s four-valued result exists for exactly this).
- **It de-risks the flip socially, not just technically.** The flip commit (e7.6) can cite
  "N real queries, M machines, zero unclassified divergences" — the kind of evidence that makes a
  default-flip review a formality instead of a leap of faith.
- **It is nearly free to build** because every piece exists: both engines already implement
  `LexicalSearch`; the comparator is quill-e6.2's, factored into a small shared module; fsfs
  already has telemetry/artifact conventions.

**Implementation sketch.**
- New bead under quill-e7 (blocking e7.6, depending on e7.1 + e6.2): `quill-e7.8: shadow-oracle
  dual-run mode + divergence reporting`.
- Wrapper: `search_fusion_candidates` calls serving engine synchronously (preserving the
  poll_immediate contract — the wrapper's future is exactly as ready as the serving engine's),
  clones the query + budget into a bounded two-phase channel consumed by a comparison worker in
  the searcher's region; worker runs the shadow engine + comparator, appends classified artifacts.
  Backpressure policy: drop-compare (never block serving) with a dropped-count metric.
- Ingest side: shadow indexes build from the same `LiveIngestPipeline` batches via the existing
  `LexicalIndexBackend` abstraction (a `TeeBackend`); disk cost = one extra lexical index,
  bounded and documented.
- Flip criterion addition to G3 (soft gate): ≥2 weeks of shadow operation on dogfood machines
  with zero unclassified divergences.
- Post-flip, the wrapper is retained briefly with roles swapped (serve Quill, shadow tantivy) as
  a rollback tripwire, then deleted with the tantivy retirement (e9.3) — it is oracle-feature
  tooling, not a compatibility shim.

**Risk/cost.** ~2x lexical CPU + ~2x lexical index disk *on opted-in dogfood machines only*;
lexical search is single-digit milliseconds at our scale so this is invisible in the two-tier
budget. The comparison worker must never publish shadow results into fusion — enforced by type
(wrapper returns only serving results) and by an EngineIdentity assertion in artifacts. Moderate
plumbing cost (~600 LOC + tests), well inside one epic-bead.

**Confidence: 0.85.** The mechanism is standard practice for risky engine swaps (dark launches),
the seam (`Arc<dyn LexicalSearch>`) is proven by two existing implementations, and the fusion
contract analysis (`searcher.rs:1254-1290`) shows the wrapper preserves it trivially.

---

## Idea 4 — Auto-minimizing divergence shrinker: every gauntlet failure becomes a 3-doc reproducible fixture

**What it is.** A delta-debugging minimizer built into the gauntlet (quill-e6.2's runner): when a
differential fails on a generated (corpus, query) pair — which at nightly scale means "somewhere
in 1M synthetic docs, rank 7 and 8 swapped" — the harness automatically shrinks it to a minimal
reproducing case and emits it as a permanent, self-contained regression fixture.

Loop (all steps deterministic because both engines are deterministic given ingest order — plan
§1.5 guarantees this for Quill; single-writer, fixed-order indexing guarantees it for the oracle):

1. **Corpus shrink (ddmin):** split docs in half, re-index both engines on each half (+
   complements), keep any subset that still diverges; recurse. Geometric shrink means ~O(log n)
   re-index cycles, and the corpora shrink as you go, so total cost is dominated by the first few
   iterations.
2. **Query shrink:** drop boolean clauses / terms / phrase words; shorten glob patterns; keep the
   smallest still-diverging query.
3. **Document shrink:** token-level halving of surviving docs' content (bounded rounds).
4. **Emit:** a content-addressed fixture (`gauntlet/regressions/<xxh3>.json`) containing the
   minimized docs, query, both engines' full outputs, the divergence class, and — see below — the
   paired score explanations. The regression directory is run by the fast gauntlet lane forever.

Pair it with **explanation-driven auto-triage**: Quill owns its scorer, so it can emit a per-hit
BM25 factor breakdown (idf, tf, fieldnorm byte, avgdl, per-field boost contribution) essentially
for free; tantivy exposes `Query::explain` on the oracle side. The comparator diffs the *factor
vectors* of the first diverging doc and buckets the root cause automatically: idf mismatch ⇒
snapshot-stats/tombstone semantics; tf-norm mismatch ⇒ fieldnorm table or doc-length accounting;
boost mismatch ⇒ parser/field-expansion; equal factors but different order ⇒ tie-break/accumulation
order. The Divergence Register entry (e6.8) is then pre-drafted by the tool.

**Why it wins.** The gauntlet is the project's central bet (Q5), and its operational bottleneck
will not be *finding* divergences — nightly generators at 1M-doc scale will find plenty early on —
but *diagnosing* them. An unminimized nightly divergence is hours of agent time and invites
misclassification (the exact failure mode the Divergence Register exists to prevent: an
"unclassified divergence blocks the gate"). A minimized 3-doc fixture with a factor-level root
cause is minutes. For a swarm-executed project this is the difference between the gauntlet being
a force multiplier and being a queue of stale red nightlies that agents route around. It also
compounds: every bug ever found becomes a permanent fast-lane fixture, so the regression corpus
grows exactly along the engine's historical weak spots — the highest-value test distribution that
exists.

**Implementation sketch.**
- New bead under quill-e6, depending on e6.2: `quill-e6.9: divergence minimizer + explanation
  auto-triage + regression fixture registry` (~500 LOC: hand-rolled ddmin — no proptest exists in
  the dep universe and none is needed; the structure being shrunk is a Vec of docs and a query
  AST, both already serializable in the gauntlet).
- Requires one small engine hook: `QuillIndex::explain_score(docid, query) -> ScoreBreakdown`
  (also directly reusable by `fsfs explain` for lexical hits — a user-facing UX win for free).
- Budget guard: minimizer capped by wall-clock/fuel; on cap, emits the best-so-far shrink (still
  vastly better than the raw case).
- CI: regressions directory in the PR fast lane; minimizer runs only in the nightly lane.

**Risk/cost.** Low risk — it is pure dev tooling behind the dev-only gauntlet crate; worst case it
is merely unused. Cost ~3-4 agent-days. One subtlety: shrinking must hold the *divergence class*
fixed (a ScoreEpsilon tie that shrinks into a RankExact failure is a different bug) — the shrink
predicate matches on classified signature, not on "any difference".

**Confidence: 0.8.** ddmin over deterministic differential subjects is textbook and the
determinism preconditions are explicitly guaranteed by the plan; the explanation hook leans on
scoring code Quill must own anyway.

---

## Idea 5 — Fuel-metered, cancel-aware query execution: make the read path actually asupersync-native

**What it is.** The plan's bet #3 ("the runtime mismatch is real") criticizes tantivy for running
blocking work inline on async workers — but Quill's own read path, as designed, has the identical
property: fully synchronous internals wrapped in immediately-ready futures (§9.1), required by the
fusion rayon contract. A pathological query — a `Complex` glob forcing a full dictionary scan, a
high-fanout disjunction over a 1M-doc index, a phrase query over degenerate position stacks —
runs to completion no matter what, pinning a rayon thread and ignoring the `Cx` cancellation that
the constitution (§1.3) promises ("every Quill operation that blocks... honors cancellation").
Today `_cx` is unused in the incumbent's search paths; Quill should not inherit that.

Fix with two composable mechanisms that preserve "resolve without Pending" exactly:

1. **Cooperative cancellation checkpoints.** Argus kernels check the `Cx` cancellation flag (one
   relaxed atomic load) at deterministic intervals — every K posting blocks decoded, every
   dictionary prefix-block scanned during glob expansion, every segment boundary in the k-way
   merge. On cancel: return `Err(SearchError::Cancelled { phase: "lexical_query" })`. Returning an
   error is fully compatible with the poll_immediate contract (the future still completes on first
   poll — it completes with `Err`); the fusion path already propagates `Cancelled` specially
   (`searcher.rs:1304-1306` returns it rather than degrading), so the plumbing exists end-to-end.
2. **Deterministic query fuel.** A per-query work budget measured in engine-deterministic units
   (blocks decoded + dict entries scanned + positions matched), configurable via `QuillConfig`
   (default: off/∞ for conformance parity; fsfs sets a generous ceiling). Exhaustion returns a
   typed `SearchError::QueryBudgetExceeded { consumed }`. Because fuel counts work units — not
   time — behavior is bit-reproducible, LabRuntime-testable, and platform-independent, unlike a
   timeout. It is the engine-level analogue of the searcher's `quality_timeout_ms` philosophy,
   but deterministic.

**Why it wins.**
- **Robustness for the actual consumers.** fsfs TUI and agent pipelines issue adversarial queries
  routinely (agents paste code fragments as queries). Today the only defense is the two-tier
  timeouts *around* the lexical call, which cannot stop the underlying work — the thread stays
  burned. Fuel + checkpoints bound the damage at the source, protecting fusion p99 and watch-mode
  ingest (which shares the machine) from query-side tail explosions.
- **It makes a marketing claim true.** "asupersync-native" currently describes Quill's write path
  only. With checkpoints, cancellation works on every path, LabRuntime can test
  cancel-at-every-checkpoint on *queries* (the plan's §15.5 only exercises cancel points on
  ingest), and the differentiation-vs-tantivy story becomes fully honest.
- **Determinism makes it testable and conformance-safe.** Fuel is off in conformance/gauntlet
  profiles (no behavioral divergence risk); when on, exhaustion is a typed error, never a silent
  truncation — no "partial results presented as results" (which would violate the
  no-silent-divergence constitution).
- **Overhead is provably negligible and gated.** One branch + one relaxed load per 128-doc block
  is noise-level; land it with the house A/B discipline (checkpoints on/off within [0.97, 1.03]
  or the interval widens and the design is revisited).

**Implementation sketch.**
- Amendment to quill-e4.3 (cursor trait + kernel): thread a `QueryGas { remaining: u64, cx: &Cx }`
  through the evaluation loop; checkpoint helper inlined (`#[inline]`, cold path out-of-line).
- New small bead under quill-e4: `quill-e4.11: cancellation checkpoints + deterministic fuel
  (+ LabRuntime cancel-at-checkpoint suite, + overhead A/B ledger entry)`.
- Glob expansion (e4.6) and snippet generation (e4.8) take the same gas handle — dictionary scans
  are the worst-case loops.
- fsfs: map `QueryBudgetExceeded` to the existing graceful-degradation surface (lexical failure ⇒
  semantic continues — the README's degradation table already defines this row).

**Risk/cost.** Low-moderate. The one real risk is checkpoint overhead on the hot kernel — bounded
by the mandatory A/B gate and by checkpoint granularity tuning (K is a constant, not a config).
Fuel semantics must be documented as engine-internal (unit values may change across versions;
never part of the conformance contract). ~2-3 agent-days.

**Confidence: 0.7.** The mechanism is simple and the contract analysis is solid; docked because
the fuel *defaults* need empirical tuning to be useful-but-not-surprising, and the win is
robustness insurance rather than a headline number.

---

## Appendix — the other 25 ideas considered (one-liners)

1. **Group-commit durability policy** — decouple seal/fsync cadence from watch-batch cadence
   (seal on delta-budget breach OR interval OR idle), since Q3 makes visibility free and canonical
   storage makes rebuild the recovery path; kills per-batch tiny-segment churn + manifest fsync
   amplification. (Strong; narrowly missed top 5 — partially reachable via e7.2 debounce retune.)
2. **Hostile-bytes no-panic reader fuzz lane** — bit-flip/truncate/lie-in-length-fields fuzzing of
   FSLX open/read paths with a "typed error, never panic, never unbounded alloc" gate (validate
   section bounds before slicing/allocating); e2.8 covers codec roundtrips, not adversarial files.
3. **fsync/rename/write error-injection in the crash matrix** — the fsyncgate lesson: after a
   failed fsync the previous generation must remain the published truth; kill-points cover
   crashes, not I/O errors.
4. **Per-segment DocId presence filters** (binary-fuse/xor over IDHASH keys, in-memory at open) —
   keep upsert/delete probes ~O(1) as segment count grows in watch mode.
5. **Fused token hashing in the SIMD classify pass** — compute the interner probe hash while token
   bytes are in-register; the interner probe is the next hot op after classification (e8.5 seed).
6. **Hot-term postings-offset cache** keyed by (manifest generation, field, term) — agents repeat
   queries heavily; skips dict probes on repeat queries (e8.5 seed).
7. **memmap2 `Advice` hints** — SEQUENTIAL for concat-merge scans, WILLNEED on open for TERMDICT/
   BLOCKMAX, RANDOM for postings; safe API, helps QG-5/QG-9 (e8.5 seed).
8. **Interleaved A/B/A/B same-process oracle benches** — extend the bench-internals convention to
   engine-vs-engine indexing runs to cancel RCH fleet drift (amendment to e8.1).
9. **Cold-cache lanes for QG-6/QG-9** — query latency and open-time measured on evicted page cache
   too, not just warm (amendment to e0.6 manifests).
10. **Keeper telemetry contract → ops TUI** — structured events (seal/merge/compact/GC/recover-
    from-prev) with a materialization row in frankensearch-ops, so fleet dashboards show lexical
    health (segment count, tombstone debt, generation lag).
11. **`fsfs doctor` FSLX rules** — verify checksums, orphan/lock inventory, tombstone-density and
    segment-count SLO warnings, `.fec` coverage report; maps to `QuillIndex::verify(deep)`.
12. **Opt-in query harvesting for the gauntlet** — record (hashed, local-only) real fsfs query
    strings into a replay corpus so e6.1's generators are calibrated against reality.
13. **Delta-cursor conservative block-max** — maintain a running max-freq per delta chain header so
    MaxScore/BMW pruning stays enabled in mixed snapshots instead of degrading to exhaustive.
14. **Tombstone side-journal** — append-only tombstone deltas folded into MANIFEST periodically if
    per-batch manifest rewrite shows in watch-mode profiles (WAL pattern already in-tree).
15. **Content-addressed on-demand corpus generation** — never commit the 1M-doc xlarge fixture;
    commit the seeded generator + expected xxh3 of the canonical corpus bytes.
16. **Three-way FTS5 set-level cross-check** — boolean result-*set* agreement (not scores) as a
    cheap independent sanity oracle for the query algebra.
17. **Snapshot pin API** — expose epoch pinning so TUI pagination/explain operate on one stable
    snapshot across interactions (Arc-clone already gives this; make it a public contract).
18. **Canonical-order bulk rebuild mode** — sort-by-content-hash ingest option making full rebuilds
    byte-identical across machines (strengthens §1.5 determinism for cross-host artifact diffing).
19. **Merge/compaction priority throttling** — region-scoped background tasks yield to foreground
    search on low-core machines (P/E-core awareness on Apple Silicon).
20. **Tokenizer bandwidth-ceiling harness as a standing tool** — automate QG-1's honesty
    denominator (tokenize-only throughput) so every ingest bench reports %-of-ceiling.
21. **`fsfs explain` lexical BM25 breakdown** — surface Idea 4's `ScoreBreakdown` to end users
    (idf/tf/fieldnorm/boost per matched term), a UX capability tantivy never gave frankensearch.
22. **LabRuntime failure-seed registry** — nightly-discovered failing seeds append to a committed
    seed corpus replayed forever in the fast lane (parallel to the regression-fixture registry).
23. **Cross-arch CI parity smoke** — a macOS ARM lane running codec/tokenizer bit-parity + fast
    gauntlet, since Apple Silicon is a headline reference target but the fleet is x86.
24. **IDHASH robin-hood layout + load-factor contract** — pin probe-length bounds in the format
    spec with fuzz coverage, so worst-case upsert probes are bounded on adversarial DocId sets.
25. **Positions-usage telemetry** — count phrase-query incidence per index and have doctor suggest
    the documented positions-off config when a corpus never phrase-searches (informed by data,
    never silent).

---

*End of deliverable. Top-5 beads are proposed as: e3.10 (writer lock), g1a (skeleton milestone +
graph edits), e7.8 (shadow mode), e6.9 (minimizer/triage), e4.11 (fuel/cancel).*
