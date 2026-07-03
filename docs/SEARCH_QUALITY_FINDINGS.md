# Search Quality: Measured Findings & Recommendations

A consolidated, prioritized roadmap distilled from the real-embedding + real-benchmark
investigation recorded in `NEGATIVE_EVIDENCE.md` (commits `3833955`→, IronPetrel, 2026-07-02/03).
Every number below is measured — on real embeddings, real BEIR datasets (SciFact / NFCorpus /
ArguAna) with real relevance judgments, and Tantivy's real BM25 ranking function (`rank_bm25`),
via a pure-Python `model2vec` + `sklearn` harness (no cargo, no torch). "vs Tantivy" = vs the
BM25 lexical tier frankensearch is built on.

## The verdict

frankensearch's **hybrid (lexical + vector via RRF) beats Tantivy-lexical-alone on semantic
search** — and the single biggest lever is the **embedding model**, where the current default is
measurably the worst option for English retrieval.

## Recommendations, by leverage (all measured)

### 1. Change the default embedder → a retrieval-distilled model. [biggest, ~free]
- **What:** `DEFAULT_MODEL_NAME`/`DEFAULT_HF_ID` in `crates/frankensearch-embed/src/model2vec_embedder.rs:35`
  is `potion-multilingual-128M` — **the worst of 4 model2vec models on English retrieval**
  (SciFact vector recall@10 = 0.598). A retrieval-distilled model (`minishlab/potion-retrieval-32M`
  class) scores **0.795 — +33% relative.**
- **Cost:** none — `model2vec` embed speed is ~flat across models (token-lookup + mean-pool, not a
  transformer forward pass), and `retrieval-32M` MRL-truncated to **dim=256** scores 0.757 (95% of
  full) at the **same scan cost, memory, and embed throughput** as the 256-dim default → a **free
  +27%** at identical cost. 32M < 128M, so it's smaller on disk too.
- **Caveat:** keep a multilingual option for non-English corpora; the recommendation is the
  *English default*. (This is the one genuine product decision — multilingual vs English-first.)
- **Premium tier (contextual):** frankensearch also ships a contextual ONNX path (`fastembed`).
  A contextual model (`BAAI/bge-small-en-v1.5`) scores **0.845/0.720** on SciFact — **+14% nDCG**
  over the best static — but embeds **~650× slower** (transformer forward pass vs static lookup)
  and needs the onnxruntime. Since embed cost is a one-time *index-build* cost (not per-query),
  contextual is the right premium for quality-sensitive, rarely-reindexed corpora. Tiered guidance:
  **static `retrieval-32M` = fast default; contextual BGE = quality premium.**

### 2. Ship the hybrid as the default (it's the safe choice across domains).
- Hybrid ≥ the better single tier on all 3 BEIR datasets; wins meaningfully on semantic corpora
  (SciFact hybrid 0.834 vs BM25 0.776), ties on keyword-overlap ones. **Never worse than the best
  single tier by more than noise.** The vector tier alone beats real BM25 on all 3 datasets
  (including ArguAna, where BM25's long-doc length-penalty hurts it — a weakness embeddings lack).

### 3. Tune RRF fusion: up-weight the STRONGER tier, small k, deep candidate feed.
- **Up-weight the stronger tier ~1.3×** (not always vector) + smaller RRF `k` (~10, not 60 —
  `k=60` is too flat over the top-10, so any weight >1 degenerates to that tier alone) → the hybrid
  strictly dominates the best single tier on **both** recall and nDCG (SciFact 0.835/0.665).
- **Neutral (hash) tiebreak**, not the current lexical-favoring one in `rrf.rs:100-111`
  (never fall through to `doc_id` — that's worse); small nDCG lift.
- **Deep candidate feed** (`candidate_multiplier` ⇒ fetch ~50-100/tier, not tight top-K): +~1 recall
  pt, ~free. frankensearch reranks, so the first-stage metric is **recall@100** — the hybrid feeds
  **96% of relevant docs** (SciFact recall@100 = 0.960), and the vector tier's edge over BM25
  *grows* with depth (exactly what a reranker exploits).

### 4. Reranker is a CONDITIONAL polish — there is NO safe default reranker (measured across 3 datasets).
The "best" reranker **flips completely by corpus**, and *stronger is not safer.* Same hybrid candidates,
both cross-encoders vs no-rerank, on all 3 BEIR datasets:

| dataset (query style) | `ms-marco-L6` (default) | `bge-reranker-base` (strong) |
|---|---|---|
| SciFact (scientific *claims*) | −2.1% | **+4.4%** |
| NFCorpus (health *questions*) | **+23.5%** | ~0 |
| ArguAna (arg → *counter*-arg) | **+7.0%** | **−23.4%** |

- **`bge-reranker-base` is NOT a safe default** (retracts an earlier single-dataset call): it wins only on
  SciFact, is inert on NFCorpus, and **loses 23% on ArguAna** — a strong *similarity* reranker demotes the
  *counter*-arguments ArguAna wants. When the task's relevance notion diverges from the reranker's training
  objective, a stronger reranker does *more* damage.
- **`ms-marco-L6` is the better generalist** (net-positive 2/3, +23.5% on web-question-style NFCorpus) but is
  mildly negative on claim-style SciFact.
- **Recommendation:** treat the reranker as a **per-corpus eval decision, not a default** — the downside of a
  mismatched reranker (−23%) exceeds the upside of the right one on most corpora. **The hybrid alone is never
  catastrophic; a mismatched reranker can be.** Cost adds to the case: `bge-reranker-base` (~278M) is ≈3× slower
  per pair than `ms-marco-L6` on CPU. When in doubt, **skip reranking** — or A/B a candidate reranker per corpus.
- **Rerank DEPTH is a second corpus-dependent knob — bias shallow.** How many candidates to feed the cross-encoder
  has opposite-signed optima: on NFCorpus/ms-marco reranking is monotonic-increasing (knee at D≈20-30 — captures 90%
  of the lift at top-20, never pay for >30); on SciFact/bge it's monotonic-**decreasing** past D=5 (deep reranking
  promotes false positives, flipping +0.019→−0.011). So **default to a shallow rerank depth (~top-10)** and only go
  deeper if a per-corpus eval shows the reranker is well-matched *and* relevant docs sit deep. Deeper reranking is
  neither free (linear cross-encoder cost) nor safe (an imperfect reranker injects false positives at depth).
- **BLEND the reranker score — don't let it fully reorder (this is the safety net).** Rank by
  `α·reranker + (1-α)·retrieval` (per-query min-max normalized), **not** by the reranker alone. This resolves the
  downside risk above: on SciFact/bge pure-reorder *hurts* (−0.011) but a light blend (α=0.25) gives **+0.025** — a
  +0.037 swing — because the retrieval score vetoes the deep false positives. When the reranker is strong (NFCorpus)
  blending costs nothing (peaks α=0.75, marginally over pure-reorder). **A fixed α≈0.4-0.5 is net-positive on every
  dataset tested** and degrades gracefully on a mismatched reranker (leans on retrieval) — so blending is what makes the
  reranker tier usable *by default* and shrinks the per-corpus-eval burden. Net reranker verdict: **blend (never
  pure-reorder), bias shallow, still sanity-check the model per corpus.**

### 5. int8 two-pass as the fast-tier primitive. [DONE — landed `39dd9be`]
- On real embeddings int8 two-pass is **7.1× faster than flat exact @ recall 1.0**, and it's both
  faster *and* exactly lossless vs 4-bit (the AVX2 `dot_i8_i8` kernel beats the 4-bit nibble-unpack).
  Already swapped in `sync_searcher.rs` (820/820 fusion tests green).

## Total measured uplift (recommendations 1+3, end-to-end hybrid)
| BEIR dataset | current stack (multilingual-128M + equal RRF) | recommended (retrieval-32M + tuned RRF) | Δrecall / ΔnDCG |
|---|---|---|---|
| SciFact | 0.785 / 0.591 | **0.835 / 0.665** | +5.0 / +7.4 |
| NFCorpus | 0.141 / 0.268 | **0.156 / 0.321** | +1.5 / +5.3 |
| ArguAna | 0.778 / 0.373 | **0.794 / 0.384** | +1.6 / +1.1 |

## Gotchas ruled out (measured)
- **Query prefixes** (`query:`/`passage:`): do NOT add them — potion-retrieval-32M is a no-prefix
  (symmetric) model; prefixes cost 2-3 recall pts.
- **MRL dim-truncation:** viable (graceful) only on retrieval-distilled models (95% recall @2× smaller);
  catastrophic on the general default (0.545). `mrl.rs`'s "2-6× faster" is real+recall-safe *only* with
  a Matryoshka-trained embedder — another reason for recommendation #1.
- **Short queries:** the vector tier collapses on ≤3-word queries (recall 0.45) — a *fundamental*
  query-underspecification (query-embedding drift), model-invariant; lexical BM25 carries them.

## What's NOT yet done (implementation, product-gated)
The above are measured recommendations, not code changes (except #4, landed). Wiring #1 (default
embedder + dim-256), #3 (RRF weight/k/tiebreak/depth) into the Rust config are product decisions.
The full measurement trail, with self-corrections, is in `NEGATIVE_EVIDENCE.md`.
