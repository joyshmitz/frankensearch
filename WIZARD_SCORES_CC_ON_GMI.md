# WIZARD SCORES — CC evaluating GMI's top 5

Evaluator: Wizard CC (Claude Fable). Subject: `/data/projects/frankensearch/WIZARD_IDEAS_GMI.md`.
All claims below verified against the repo, the plan (`COMPREHENSIVE_PLAN_FOR_THE_QUILL_LEXICAL_ENGINE.md`),
the `bd-quill-*` beads, and pinned crate sources where cheap.

---

## Idea 1 — Snapshot-Epoch 2D BM25 Memoization Table — **Score: 480/1000**

**Justification.** The idea is a legitimate micro-lever family, but three of its load-bearing claims
are factually wrong. (1) The size arithmetic is off 4x: 256×256×4B = **256KB per field per
snapshot**, not "64KB L1-resident" — that is L2-resident on every reference machine (Zen 3 L1d =
32KB), so the lookup competes with posting blocks for L2 and replaces a fully-pipelined f32
divide (~10-14 cyc latency, better throughput) with a likely-L1-missing load. (2) "Completely
eliminating division and multiplication" is false: idf×boost varies per term, so a multiply
remains in the loop no matter what. (3) The real baseline is not naive per-doc BM25 arithmetic:
tantivy 0.26.1 **already precomputes the 1D 256-entry norm cache per term weight**
(`tantivy-0.26.1/src/query/bm25.rs:62-68,76`, `score = weight * tf / (tf + cache[norm_id])`), and
since Quill's scoring contract mirrors tantivy (plan §8.4), Quill's baseline will carry the same
1D cache — the marginal delta of 2D-over-1D is one div+add, not "division and multiplication
stripped from the tightest loop." There is also an unnoticed conformance trap: `(w·tf)/(tf+n)`
(tantivy's op order) and `w·(tf/(tf+n))` (what a weight-independent table forces) differ in f32
rounding, so the table path cannot be bit-identical to the oracle formula — it must either define
Quill's pinned scalar reference with the table's association from day one (possible, since
RankExact is defined engine-internally and cross-engine comparison is ScoreEpsilon, §5.3) or ship
only under the `fast_scoring` relaxation (§8.4, "default off until G3 evidence"). GMI flags none
of this at "99% confidence / algorithmic slam dunk," which is exactly the self-deception §14's
five laws and the NEGATIVE_EVIDENCE discipline exist to catch. Also note freq>255 fallback
branches in the hot loop and per-field tables (per-field avgdl, STATS §10.2) multiply the footprint.

**Strongest point in favor.** Snapshot-epoch invariance of avgdl is correctly observed, and the
rebuild cost per delta-publish (~65k divides per field) is genuinely negligible — the lever is
cheap to build and trivially safe to A/B under quill-e8.5's one-lever discipline.

**Sharpest flaw.** The comparison baseline is wrong: the incumbent already has the 1D cache, so
the claimed win largely doesn't exist; what remains is a division-vs-L2-load tradeoff that this
repo's own ledger history (L1-table-vs-compute lessons) says must be measured, not assumed.

**What would change my score.** A ledgered A/B on the medium/xlarge profiles showing ≥3-5% end-to-end
QG-6 improvement over a correct 1D-cache baseline, with the bit-parity story resolved in the
Language Contract — that would move this to ~700 as a solid e8.5 lever.

---

## Idea 2 — Page-Fault Pipelining via `memmap2::Advice::WillNeed` — **Score: 580/1000**

**Justification.** Verified real: the workspace pins memmap2 0.9 (`Cargo.toml:92`), and
`Mmap::advise_range(Advice::WillNeed, offset, len)` exists as a safe API in the vendored source
(`memmap2-0.9.10/src/lib.rs:835`; `Advice::WillNeed` in `advice.rs:29`) — no new deps, no unsafe
beyond the existing mmap allowance, no conflict with the fusion poll_immediate contract (madvise
never blocks). The mechanism is sound for cold-cache tails. But it is oversold as top-5: (a) the
stated overlap window ("while Argus sets up MaxScore/BMW heaps and parses the query tree") is
mis-ordered — dictionary probes happen *after* parsing, and heap setup is sub-microsecond, so
meaningful overlap requires the probe-all-terms-then-advise-then-iterate structure (fortunately
Argus's plan phase, §9.1, permits exactly that); (b) fsfs's dominant deployment is a long-running
watch process whose index is warm, where each advise call is a pure-overhead syscall (~1-2µs/term/
segment) with no safe residency check available to gate it (memmap2 exposes no mincore); (c) the
plan's posture already accepts page-fault costs explicitly (§11.2). So this is a real but modest
cold-tail lever for QG-6-p99/QG-9, needing the standard warm-path-regression A/B gate.

**Strongest point in favor.** It attacks the one structural weakness of the mmap read path with
~20 lines of safe code, and the API availability claim checks out exactly as stated.

**Sharpest flaw.** No warm-path story: without residency information, every warm query pays the
syscalls, and warm queries are the overwhelming majority in watch-mode — the expected-value
calculation is done only on the cold branch.

**What would change my score.** Evidence from a cold/warm split bench (my own appendix proposed
the same lever plus cold-cache lanes) showing p99 wins with warm-path regression inside [0.97,
1.03]; and gating advise on extent size (skip tiny postings). That's a ~680 as an e8.5 lever.

---

## Idea 3 — Quarantine-and-Degrade Open Protocol — **Score: 320/1000**

**Justification.** The instinct (don't hold a 1M-doc corpus hostage to one bad 64MiB mini-segment)
is right, but the mechanism as designed is wrong-layered and rests on a false premise. The false
premise is the backfill claim: "missing documents will naturally backfill via watch-mode
incremental updates" — they will not. fsfs incremental change detection
(`fsfs/src/incremental_change.rs`, plan §2.4) diffs *canonical storage state* (content hashes /
mtimes), not index-vs-storage membership; a document whose file never changes again is silently
unsearchable forever. The correct recovery already exists and is cited by the plan: rebuild from
canonical storage is "already the recovery path" (§16.2, bead quill-e7.5 rebuild-on-detect), so
the right shape is: Keeper returns a typed, information-rich error (or opens read-only with an
explicit `degraded: [segment ids]` marker), and *fsfs* — which owns rebuild policy per
quill-e3.3's own note ("rebuild policy belongs to fsfs, e7.5") — quarantines the file for
forensics and triggers rebuild. Silent partial-serving inside Keeper violates the plan's
constitution §1.8 ("no silent semantic divergence") twice over: result sets silently lose
documents, and snapshot-level BM25 stats (N, avgdl, doc_freq, §8.4) silently shift every score.
It also collides with mechanics the sketch ignores: manifest tombstone sets for the quarantined
segment dangle (upsert of a doc whose prior version lived there can't tombstone it — resurrection
hazard if the segment is restored), `.quarantine` files need registration in the GC naming schema
(quill-e3.3's RULE-1-derived "GC deletes only files matching Quill's own naming schema"), and the
rename-not-delete choice is fine under RULE 1 but was evidently not checked against it. Note the
system *already* degrades gracefully at the correct seam: lexical hard-failure ⇒ semantic-only
search (README failure-modes table), which returns *honest* results rather than silently
incomplete lexical ones.

**Strongest point in favor.** It correctly identifies that strict fail-stop `open()` plus a
multi-segment format creates a new availability failure mode tantivy's consumers rarely hit, and
that *something* should exploit segment independence.

**Sharpest flaw.** The backfill premise is factually wrong for this codebase, which converts
"temporary graceful degradation" into "permanent silent data loss" — the worst outcome class for
agent consumers who trust returned result sets.

**What would change my score.** Re-scoped as: typed `IndexDegraded` error carrying segment ids +
fsfs-side quarantine-forensics-and-auto-rebuild + prominent doctor/ops surfacing, with read-only
degraded serving allowed *only* behind an explicit flag that also marks every response — that
version is a ~650 idea and a good new bead under e7.5.

---

## Idea 4 — Shadow-Traffic "Dark Launch" Gate (G2.5) — **Score: 800/1000**

**Justification.** This is the strongest of the five, and I say so with full awareness that it is
convergent with my own top-5 (independent convergence by two adversarial wizards is itself
evidence the idea is load-bearing). The core argument is exactly right: the plan's top-ranked
correctness risk (§20 row 1: parser corner cases, segment-geometry stats) is the risk class that
synthetic generators (quill-e6.1) are weakest against and real agent query traffic is strongest
against; dark-launching converts the fleet's daily fsfs usage into free, adversarially-sampled
conformance evidence before the e7.6 flip, at zero serving risk. Framing it as an explicit gate
("G2.5") between the plan's G2 and G3 is a genuinely good structural touch. Two design blemishes
keep it from the 850+ tier. First, placement: hooking the dual-run into `TwoTierSearcher` touches
the fusion crate the plan promises to leave unchanged (§13.1 "Zero changes") — the cleaner shape
is a `ShadowLexicalSearch` wrapper behind the existing `Arc<dyn LexicalSearch>` seam
(`fusion/src/searcher.rs:129`), which preserves the poll_immediate contract trivially (the
wrapper's future is exactly as ready as the serving engine's) and needs zero fusion edits.
Second, the divergence sink: writing from the search path into "frankensearch-ops SQLite" inverts
the dependency direction (ops *consumes* telemetry; fusion/fsfs must not depend on ops) — local
JSONL artifacts with MismatchSignature dedup (reusing quill-e6.2's comparator) plus ops-side
ingestion is the house-consistent shape. The doubled-indexing cost is honestly acknowledged and
correctly mitigated (opt-in, time-boxed).

**Strongest point in favor.** It is the only pre-flip evidence source sampled from the true query
and corpus distribution, and it produces Divergence Register entries (§15.6) from reality rather
than from what test authors imagined.

**Sharpest flaw.** The concrete wiring (fusion-layer placement, ops-SQLite coupling) contradicts
the plan's own seam discipline and dependency direction; as specified it would need rework in
review.

**What would change my score.** Respecifying at the LexicalSearch seam with backpressure-safe
drop-compare semantics and a panic-isolated comparison region (asupersync `Outcome::Panicked` as
a logged finding) would put this at ~860.

---

## Idea 5 — Zero-Copy SWAR Snippet Windowing — **Score: 280/1000**

**Justification.** The premise is misplaced and the mechanism, as specified, fails the plan's own
parity gate. Premise: snippets are not on the Phase-1 (<15ms `Initial`) path — the two-tier
fusion path returns id+score candidates only (`search_fusion_candidates` →
`search_doc_ids`, `lexical/src/lib.rs:1411-1435`; no snippet call exists anywhere in
`fusion/src/searcher.rs`); snippet generation runs on the separate fsfs lexical display path
(`search_with_snippets`), so "risks missing the <15ms target" attributes cost to an envelope that
never pays it. Mechanism: plan §9.5 pins snippet **output parity with tantivy's generator on
fixtures**, and tantivy's generator tokenizes the stored text with the analyzer and matches at
token granularity. Raw byte `memmem` against stored bytes changes semantics in at least two
ways GMI missed: token boundaries (query term `cat` highlights inside `concatenate`) and case
folding (analyzed terms are lowercased; stored bytes are original-case, so `Rust` is missed for
term `rust` without a case-insensitive scan). Meanwhile the one risk GMI *does* name — ASCII term
bytes matching inside a multi-byte character — is structurally impossible in UTF-8 (every
continuation byte is ≥0x80), which suggests the analysis ran on vibes rather than on the encoding.
Once you fix the semantics, you must analyze the document text anyway (the analyzer is already
SIMD per quill-e1.1), and the plan's stated design — "greedy window over token hits" with
borrowed spans and no per-token allocation (§6.1, §9.5) — already *is* the salvageable content of
this idea. What remains is "don't allocate Strings until the winning window," which is a code-review
note, not a top-5 design idea.

**Strongest point in favor.** The zero-copy discipline (byte offsets end-to-end, single UTF-8
boundary snap + one allocation at the winning window) is the right implementation posture for
quill-e4.8 and worth writing into that bead's notes.

**Sharpest flaw.** Byte-level memmem without analysis would fail the §9.5 output-parity fixtures
against the oracle on day one — the idea optimizes the generator by changing what it generates.

**What would change my score.** Reframed as "quill-e4.8 implementation guidance: analyzer-driven
token hits with byte-offset windowing, allocation only at the final window, plus a SWAR pre-scan
*only* as a candidate-rejection fast path that cannot change output" — that's a ~500 note-level
contribution.

---

## Appendix skim — notable items

- **#9 io_uring for sealing**: violates the closed dependency universe and the unsafe budget
  (io_uring needs a new crate or raw syscalls); also sealing already runs under `spawn_blocking`
  (plan §11.2), so the thread-blocking premise is weak. Should not have survived triage.
- **#15 cargo-afl/LibFuzzer** and **#16 Java Lucene cross-oracle**: both breach the zero-new-deps
  posture (§1.1) — the plan's fuzzing is deliberately hand-rolled/property-based, and a JVM
  harness is far outside the universe. The *goal* of #15 (no-panic hostile-bytes reader) is right
  and achievable in-tree.
- **#20 read-only lockless fallback for crashed writers**: quietly the best appendix item — it is
  half of the multi-process story, though it presupposes a write lock the GMI top-5 never
  proposes, and it misses the sharper hazard (reader-side GC-on-open destroying a live writer's
  unpublished segments, plan §11.4 + bead quill-e3.3).
- **#7 mmap-lifetime `&str` DocIds**: fights `DocId = CompactString` in core types
  (`core/src/types.rs`) and would leak snapshot lifetimes into the public trait — API breakage for
  marginal gain.
- **#10 query result caching** ("invalidate only when matching docs are appended"): invalidation
  condition is wrong as stated (deletes/upserts/tombstones also invalidate); needs generation-keyed
  caching to be sound.

---

## Score index

| # | GMI idea | Score |
|---|----------|-------|
| 1 | Snapshot-epoch 2D BM25 memoization table | 480 |
| 2 | Page-fault pipelining via `Advice::WillNeed` | 580 |
| 3 | Quarantine-and-degrade open protocol | 320 |
| 4 | Shadow-traffic dark launch gate (G2.5) | 800 |
| 5 | Zero-copy SWAR snippet windowing | 280 |
