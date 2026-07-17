# WIZARD_REACTIONS_GMI.md

## Honest Reactions to CC (Claude Fable)'s Scoring

This was a bloodbath. Fable caught me in multiple severe technical oversights. Here is my honest accounting of where Fable was right, where they were wrong, and how my own evaluations shift.

### 1. 2D BM25 Memoization Table (Score: 480)
**Fable is entirely right.** I concede all three points:
1. **Size Arithmetic:** 256 * 256 * 4 bytes is exactly 256KB per field. I simply flubbed the math claiming 64KB. Pushing 256KB into L2 instead of L1 destroys the latency argument.
2. **The Baseline:** Checking `tantivy-0.26.1/src/query/bm25.rs:62` confirms it already computes a 1D `[Score; 256]` cache for `tf / (tf + norm)`. My claim that I was stripping division from the tightest loop was fundamentally ignorant of the incumbent's existing optimization.
3. **The Bit-Parity Trap:** I missed the `f32` non-associativity rounding trap (`(w*tf)/(tf+n)` vs `w*(tf/(tf+n))`). This breaks the `ScoreEpsilon` / `RankExact` conformance contract on day one.
**Self-Correction:** My 99% confidence here was sheer hubris. It's a micro-optimization at best, fraught with regression risk. Fable's 480 is brutal but fair.

### 2. Page-Fault Pipelining via `WillNeed` (Score: 580)
**Fable is mostly right.** The lack of a warm-path gating mechanism (since `madvise` lacks residency visibility like `mincore`) means I am trading a pure syscall overhead on every warm query to save latency on cold queries. Since watch-mode queries run overwhelmingly hot, this is a net regression in expected value.
**Self-Correction:** Without a cold-cache detection heuristic, this is not a top-5 lever.

### 3. Quarantine-and-Degrade Open Protocol (Score: 320)
**Fable is FACTUALLY WRONG.** Fable claims missing documents will never backfill because fsfs only diffs canonical storage state (mtimes/hashes), making this "permanent silent data loss". Fable didn't read the `fsfs` audit contracts. I will dismantle this in my rebuttal.

### 4. Shadow-Traffic "Dark Launch" Gate (Score: 800)
**Fable is entirely right.** Their architectural critique of my design is spot on. Hooking this into `TwoTierSearcher` violates the E13 "Zero changes to fusion" constraint. Fable's placement behind the `Arc<dyn LexicalSearch>` seam is strictly superior, and writing JSONL artifacts instead of reaching directly into the ops SQLite database respects the telemetry dependency direction.

### 5. Zero-Copy SWAR Snippet Windowing (Score: 280)
**Fable is entirely right.** I completely missed the E1 timeline boundaries (snippets run on the display path, not the `<15ms` retrieval path). Even worse, my byte-level `memmem` idea completely breaks analyzer semantics—I embarrassingly forgot about case folding and token boundaries (e.g., matching `cat` inside `concatenate`).
**Self-Correction:** This was a fundamentally flawed idea that optimizes the wrong path with the wrong semantics.

### Overall Impact on My Evaluation
Fable's surgical precision in dismantling my BM25 and SWAR snippet ideas proves they are operating at a higher level of codebase familiarity regarding the incumbent and the exact evaluation pathways. However, their blind spot on the `fsfs` freshness audit reveals they are treating the search engine as an isolated component rather than a node in the broader `fsfs` ecosystem.
