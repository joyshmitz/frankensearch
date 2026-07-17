# Dueling Idea Wizards Report: Quill Lexical Engine (frankensearch)

**Date:** 2026-07-16 · **Subject:** `COMPREHENSIVE_PLAN_FOR_THE_QUILL_LEXICAL_ENGINE.md` + the `bd-quill-*` bead graph (83 beads at duel time; `quill-e0.7` already landed as `quill_contract.rs`, commit c5cd8b51).

## Executive Summary

Two wizards — **CC (Claude Fable)** and **GMI (Gemini 3.1 Pro High)** — independently studied the repo, generated 30 ideas each, winnowed to 5, adversarially cross-scored each other 0–1000, then ran reveal, rebuttal, steelman, and blind-spot phases. Of 10 primary ideas: **five consensus winners** (led by a genuine data-loss hole neither the plan nor three fresh-eyes passes had caught: no cross-process writer lock), **one independent convergence** (both wizards separately proposed shadow-oracle dark-launching — the strongest validation signal this methodology produces), **one idea killed by mutual agreement**, **two salvaged with conditions**, and **one contested score overturned by orchestrator fact-check in the defender's favor**. The blind-spot probe produced four further high-value gaps, the best being the **cross-process visibility contract** (Q3's "searchable immediately" is process-local — invisible to the flagship watch-daemon + CLI-search topology). Eleven new beads operationalize the winners.

**Substitution notice:** the requested opponent **cod (Sol 5.6 ultra)** was unavailable — hard usage limit until 2026-07-23, verified via headless probe, no API-key fallback, no caam profiles. Per the skill's fallback table the duel ran cc + agy (cross-vendor). `bd-quill-sol-reduel-0j3h` tracks an optional three-way re-run after credits reset.

## Methodology

- **Agents:** CC = Claude Fable (persistent background subagent); GMI = Gemini 3.1 Pro (High) via headless `agy --print`, stateless per phase with context relayed in-prompt.
- **Phases:** study → ideate (30→5 + appendix) → adversarial cross-score (0–1000) → reveal → rebuttal → steelman (opponent's #1) → blind-spot probe → this synthesis.
- **Grounding rule:** both wizards were instructed to verify claims against the repo before scoring/conceding; the orchestrator independently fact-checked the one disputed factual claim.
- **Artifacts:** `WIZARD_IDEAS_{CC,GMI}.md`, `WIZARD_SCORES_{CC_ON_GMI,GMI_ON_CC}.md`, `WIZARD_{REACTIONS,REBUTTAL,STEELMAN,BLINDSPOTS}_{CC,GMI}.md` (12 files, committed alongside this report).

## Score Matrix

| # | Idea | Origin | Self-rank | Opponent score | Post-reveal movement | Verdict |
|---|------|--------|-----------|----------------|----------------------|---------|
| 1 | Cross-process writer lock + readers-never-GC + publish CAS | CC | 1 | **980** | GMI steelman → 1000 w/ PID-liveness fix; CC argued its own score ~55 pts too generous (TOCTOU in specified mitigation) → merged design | **CONSENSUS #1** → bead |
| 2 | Shadow-oracle / dark-launch dual-run | **BOTH** (CC#3 ≡ GMI#4) | 3 / 4 | **920** / **800** | GMI conceded CC's placement critique (LexicalSearch seam, not fusion; JSONL not ops-SQLite); CC adopted GMI's load-shedding | **CONSENSUS #2 (convergent)** → bead |
| 3 | Auto-minimizing divergence shrinker + BM25-factor triage | CC | 4 | **890** | CC adopted GMI's preserve-original-AST refinement | **WINNER** → bead |
| 4 | Skeleton-first resequencing (G1a scalar reference engine) | CC | 2 | **850** | CC refuted the trait-thrash flaw (cursor trait pre-specified by e4.3/e2.2; accumulator seam forced by e1.7 regardless) | **WINNER** → bead |
| 5 | Fuel-metered, cancel-aware query execution | CC | 5 | **750** | CC refuted the hot-loop critique (checkpoints were specified at coarse boundaries; GMI misquoted QG-6) | **WINNER (borderline)** → bead |
| 6 | Snapshot-epoch 2D BM25 memoization table | GMI | 1 | **480** | GMI conceded all three flaws (size math 256KB not 64KB; tantivy already ships the 1D cache at `bm25.rs:62`; f32 association parity trap). CC steelman produced a 32KiB/field freq-capped best-version → **680** | **SALVAGED as lever** (entry criteria) → e8.5 backlog |
| 7 | `madvise(WillNeed)` page-fault pipelining | GMI | 2 | **580** | GMI conceded not-top-5 without cold-cache gating (API verified real in pinned memmap2 0.9) | **LEVER BACKLOG** (entry criteria) → e8.5 |
| 8 | Quarantine-and-degrade open protocol | GMI | 3 | **320** | **GMI rebuttal vindicated by orchestrator fact-check**: `IndexFreshnessAudit`/`MissingLexical`/`EnqueueReindex` exist (`incremental_change.rs:505-525`) — CC's "permanent silent data loss" premise was too strong. Nuance stands: recovery requires the audit to actually run (watch mode yes; search-only topology no) | **REVIVED with conditions** → bead |
| 9 | Zero-copy SWAR snippet windowing | GMI | 5 | **280** | GMI fully conceded ("wrong path, wrong semantics") — snippets aren't on the phase-1 latency path; byte memmem breaks analyzer parity | **KILLED** |

### Blind-spot probe (not cross-scored; orchestrator-reviewed)

| Idea | Origin | Orchestrator assessment | Disposition |
|------|--------|------------------------|-------------|
| Cross-process visibility contract (Q3 is process-local; QG-3 two-topology amendment; `max_visibility_lag_ms`; freshness surfacing) | CC | Verified structural: delta lives in process memory; `fsfs` CLI-per-query is the documented agent workflow. Real honesty gap with a user-visible failure mode | **Bead (P1)** |
| Blue-green index dirs + atomic `CURRENT` pointer (flip = ms-reversible pointer swap; RULE-1-correct retention; shadow synergy) | CC | Sound; reuses the in-tree two-slot publish discipline one level up; dissolves the in-place-rebuild tension with RULE 1 | **Bead (P2)** |
| Crash-resumable bulk builds (intermediate bulk publishes + per-doc `content_hash` in IDMAP) | CC | The IDMAP format row is cheap **now** and expensive after v1 freezes — time-critical; also gives `fsfs doctor` an index↔storage audit witness (exactly what the quarantine debate showed missing) | **Bead (P2, blocks e2.6)** |
| Query AST canonicalization/dedup | GMI | Good, cheap — **but neither wizard noticed dedup changes BM25 disjunction scores** (`A OR A` scores A twice in a sum-union; tantivy does not dedup). Must be scoped to score-neutral contexts or conformance-classified | **Bead (P2, with the scoring caveat)** |
| Thread-local allocator arenas (mimalloc/jemalloc) | GMI | **Violates the closed dependency universe as stated.** The underlying concern (global-allocator contention at high core counts) is legitimate and measurable | Folded into e8.4 as a required attribution axis |

## Killed Ideas

- **SWAR snippet windowing (GMI #5, 280):** optimizes the display path believing it was the retrieval path; byte-level matching breaks token-boundary/case-fold parity. Originator conceded fully.
- (From appendices, noted but never promoted: GMI's Elias-Fano IDMAP and interpolation search duplicate already-rejected plan decisions with registered retry conditions — evidence the plan's §4 adopt/reject table is doing its job.)

## Meta-Analysis

- **Scoring asymmetry is itself signal:** GMI averaged **878** on CC's ideas; CC averaged **492** on GMI's. CC graded with source-verification (found the incumbent's existing 1D cache, the size-math error, the fsfs re-ingest question); GMI graded more generously but its critiques were still specific enough that CC **adopted three of them** (load-shedding, AST preservation, and the SIGSTOP concern that led to the merged lock design).
- **Each wizard's systematic blind spot was the other's strength.** CC treated the engine as embedded in a multi-process agent fleet (writer lock, visibility contract, blue-green) but was wrong about the fsfs freshness audit. GMI reached for micro-architectural/OS levers (tables, madvise, SWAR) and knew the fsfs ecosystem contracts, but repeatedly missed repo-contract specifics (incumbent's cache, the fusion-seam constraint, phase boundaries).
- **The reveal produced genuine movement both ways:** GMI conceded 4/5 critiques explicitly; CC conceded the SIGSTOP hole, upgraded its own design (O_EXCL generation-file CAS + `kill(pid,0)` liveness, converging with GMI's steelman), and argued against its own interest that 980 was too generous.
- **The steelman phase paid for itself:** CC's forced defense of the idea it scored 480 produced a genuinely shippable 680-version and surfaced a time-critical contract insight (the f32 association must be pinned while the Language Contract is still unfrozen — partially mooted the same day by the parallel landing of `quill_contract.rs`, which pins op-order conventions).
- **Convergence where it matters:** both models independently proposed shadow-oracle validation — and both blind-spot lists then converged on the same theme (the plan is strong on single-process correctness, thinner on multi-process operations). That theme, not any single idea, is the duel's headline finding.

## Recommended Next Steps (operationalized as beads)

1. **`quill-duel-writer-lock` (P1, blocks the flip):** merged CC+GMI design — split `open()`/`open_writer()`, readers never GC, LOCK file with PID + `kill(pid,0)` liveness (never mtime-only takeover), O_EXCL generation-claim CAS at publish.
2. **`quill-duel-visibility-contract` (P1):** pin in-process vs cross-process visibility classes; `max_visibility_lag_ms`; freshness in `segment_stats()`/`fsfs status`; QG-3 measured in both topologies.
3. **`quill-duel-shadow-oracle` (P1, blocks the flip):** dark-launch dual-run behind the `LexicalSearch` seam, JSONL divergence artifacts stamped with manifest generation, sampling + load-shedding, opt-in.
4. **`quill-duel-skeleton-first` (P1):** G1a scalar-reference resequencing so the gauntlet gets a subject months earlier; every leapfrog bet lands as a ledgered one-lever A/B.
5. **`quill-duel-resumable-bulk` (P2, blocks e2.6):** IDMAP `content_hash` format row now, intermediate bulk publishes, resume driver, resume≡uninterrupted fixture.
6. Remaining: divergence shrinker (P2), fuel metering (P2), blue-green CURRENT pointer (P2), AST canonicalization with the score-neutrality caveat (P2), quarantine-with-audit-coupling (P3), and the e8.5/e8.4 lever-backlog updates (2D table best-version, WillNeed with cold-gating, allocator-contention attribution).

*Orchestrator editorial is confined to this section and the fact-check notes marked as such; scores and arguments above are reported as recorded in the artifact files.*
