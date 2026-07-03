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
  A contextual model (`BAAI/bge-small-en-v1.5`) beats the best static on **all 3 BEIR datasets** —
  **+14% nDCG SciFact, +9.6% NFCorpus, +28.8% ArguAna** (ArguAna recall 0.698→0.841) — but embeds
  **~650× slower** (transformer forward pass vs static lookup) and needs the onnxruntime. Since embed
  cost is a one-time *index-build* cost (not per-query), contextual is the right premium for
  quality-sensitive, rarely-reindexed corpora. **The premium is largest exactly where static embeddings
  are weakest** (ArguAna's argument→counter-argument task, which needs contextual understanding of
  argumentative structure/negation that static mean-pooling can't capture). Tiered guidance:
  **static `retrieval-32M` = fast default; contextual BGE = quality premium (+10-29% nDCG, multi-dataset
  validated), most compelling on semantically hard / argumentative corpora.**

### 2. Ship the hybrid as the default (it's the safe choice across domains).
- Hybrid ≥ the better single tier on all 3 BEIR datasets; wins meaningfully on semantic corpora
  (SciFact hybrid 0.834 vs BM25 0.776), ties on keyword-overlap ones. **Never worse than the best
  single tier by more than noise.** The vector tier alone beats real BM25 on all 3 datasets
  (including ArguAna, where BM25's long-doc length-penalty hurts it — a weakness embeddings lack).
- **Exactly two tiers — one strong embedder + BM25. Do NOT ensemble multiple embedders.** Adding a
  second static embedder never helps and usually *hurts* (SciFact ret32+base32 0.769 < ret32 alone
  0.795; the triple 2-embedders+BM25 is worse than the pair on both datasets). The model2vec embedders
  are too correlated (a 2nd finds a doc the 1st misses in only ~3% of queries vs ~7% for BM25) *and*
  weaker, so RRF just dilutes. **The hybrid's power is MODALITY diversity (exact-term vs semantic →
  decorrelated errors), not signal count** — and the partner must be *comparably strong* (high
  unique-% alone doesn't help if the partner is weak; see NEGATIVE_EVIDENCE). To strengthen the vector
  tier, use a *better* single embedder (rec #1), not *more* embedders.

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
- **Keep RRF (rank fusion) — don't switch to score-fusion.** Measured RRF vs score-fusion (normalize
  BM25+cosine, weighted sum) across all 3 datasets: they're **tied on quality** (RRF wins recall on 2/3;
  score-fusion wins nDCG by a hair, only NFCorpus's +0.014 above noise), but RRF is **scale-free** — no
  per-query normalizer to choose (z-norm vs min-max disagree and are brittle to BM25's unbounded vs
  cosine's bounded distributions). RRF's simplicity + recall edge win. This validates `rrf_fuse` as the
  fusion primitive across the whole stack (base hybrid *and* reranker-combine).

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
- **Integrate the reranker as a THIRD RRF SOURCE — don't pure-reorder, don't even score-blend (this is the safety net,
  and it's native).** Rank by RRF-fusing the retrieval-order and the reranker-order, **not** by the reranker score
  alone. This resolves the downside risk above and beats the alternatives: on SciFact/bge pure-reorder *hurts* (−0.011),
  score-blend recovers it (+0.025), but **RRF-combine does best (+0.040)** — the rank fusion caps how far a deep false
  positive can climb (it must rank high by *both* retrieval and the reranker). It's **parameter-free** (no α, no score
  normalization, and k-insensitive: k=10≈k=60), so nothing to tune per corpus. It only trails pure-reorder when the
  reranker is strongly matched (NFCorpus +0.062 vs RRF +0.047) — the right trade for a default, since you can't know a
  priori if the reranker fits. **Zero new machinery: frankensearch already ships `rrf_fuse`** — feed the reranker as
  another ranked source (optionally tier-weighted). Net reranker verdict: **RRF-combine (never pure-reorder), bias
  shallow depth, still sanity-check the model per corpus — but RRF-combine makes the tier safe and tuning-free by
  default.** (Score-blend `α·reranker+(1-α)·retrieval`, α≈0.5, is the fallback if a non-RRF stage is combining.)

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

## Capstone: FULL pipeline (1+3+4) vs Tantivy BM25-alone (one end-to-end run)
The whole recommended stack — retrieval-32M + tuned RRF hybrid **+ RRF-combine reranker** — vs Tantivy BM25-alone,
each stage's contribution shown (recall@10 / nDCG@10). This is the headline number, using the *shipped*
`RerankCombine::RrfCombine` + `RrfConfig` weights (so it's expressible today; only defaults are product-gated):

| BEIR | Tantivy BM25 | + hybrid | + RRF-combine rerank | full vs Tantivy |
|---|---|---|---|---|
| SciFact | 0.776 / 0.652 | 0.816 / 0.684 | **0.872 / 0.731** | **+12% / +12%** |
| NFCorpus | 0.152 / 0.306 | 0.159 / 0.327 | **0.167 / 0.346** | **+10% / +13%** |
| ArguAna (200q) | 0.565 / 0.259 | 0.620 / 0.294 | **0.680 / 0.316** | **+20% / +22%** |

The full-stack win is **largest on ArguAna (+22% nDCG)** — precisely because BM25 is structurally weakest there (its
long-doc length penalty), a weakness the dense vector tier is immune to. Headline across 3 datasets: **+12% to +22%
nDCG / +10% to +20% recall vs Tantivy BM25-alone.**

### How to configure this pipeline today (shipped, default-preserving APIs)
The capstone stack is expressible now — no default changes, all opt-in builders landed this session:
```rust
use frankensearch_rerank::{RerankCombine, DEFAULT_RRF_COMBINE_K};
// #1 embedder: load a retrieval-distilled static model (potion-retrieval-32M) as the fast/quality embedder.
let searcher = TwoTierSearcher::new(index, retrieval_distilled_embedder, config) // config.rrf_k = 10.0  (#3 small k)
    .with_lexical(tantivy_bm25)                                     // #2 two-tier hybrid
    .with_rrf_weights(1.0, 1.3)                                     // #3 up-weight the STRONGER tier (~1.3×)
    .with_rrf_tiebreak(RrfTiebreak::Hash)                          // #3 neutral tiebreak
    .with_reranker(cross_encoder)                                   // #4 corpus-appropriate cross-encoder
    .with_rerank_combine(RerankCombine::RrfCombine { k: DEFAULT_RRF_COMBINE_K }); // #4 RRF-combine (never pure-reorder)
```
`SyncTwoTierSearcher` exposes the same `with_rrf_weights` / `with_rrf_tiebreak`. int8 fast-tier (#5) is already the
default. The only thing not yet default is the *choice* of these values — flipping the shipped defaults to match is the
remaining product-gated step.

## Gotchas ruled out (measured)
- **Query prefixes** (`query:`/`passage:`): do NOT add them — potion-retrieval-32M is a no-prefix
  (symmetric) model; prefixes cost 2-3 recall pts.
- **MRL dim-truncation:** viable (graceful) only on retrieval-distilled models (95% recall @2× smaller);
  catastrophic on the general default (0.545). `mrl.rs`'s "2-6× faster" is real+recall-safe *only* with
  a Matryoshka-trained embedder — another reason for recommendation #1.
- **Short queries:** the vector tier collapses on ≤3-word queries (recall 0.45) — a *fundamental*
  query-underspecification (query-embedding drift), model-invariant; lexical BM25 carries them.
- **Embedding-space PRF / query expansion (Rocchio):** do NOT add it for the static vector tier — it's
  net-harmful (SciFact: every config loses, up to −0.14 nDCG; harm grows monotonically with feedback =
  query drift). Static mean-pooled doc vectors averaged into the query blur it toward the corpus centroid;
  top-k false positives poison the centroid. Lexical BM25, not vector PRF, is the weak-query remedy.

## Implementation status

The measured levers are now **shipped as opt-in capabilities** — each default-preserving (legacy behavior
byte-for-byte unchanged), so the recipe is expressible in code with **no** product decision required:

| Lever | Status | Enable via |
|---|---|---|
| int8 fast-tier (#5) | **LIVE (default)** `39dd9be` | (default in `sync_searcher.rs`) |
| RRF-combine reranker (#4) | **Shipped, opt-in + wired** `235fb46`/`7ca8877` | `TwoTierSearcher::with_rerank_combine(RerankCombine::RrfCombine { k })` |
| Per-tier RRF weight (#3) | **Shipped, opt-in** `7ccda28` | `RrfConfig { semantic_weight: 1.3, .. }` (up-weight the *stronger* tier) |
| Neutral hash RRF tiebreak (#3) | **Shipped, opt-in** `05472cd` | `RrfConfig { tiebreak: RrfTiebreak::Hash, .. }` |
| RRF `k` (#3) | already configurable | `RrfConfig { k: 10.0, .. }` / `TwoTierConfig.rrf_k` |
| Deep candidate feed (#3) | already configurable | `candidate_multiplier` |

**Remaining work is outward-facing DEFAULT flips (product-gated).** Turning the recipe on *by default* changes
user-visible ranking output and updates test snapshots, so each needs a product sign-off — each is de-risked to a
one-to-few-line change:
- **#1 Embedder default → `potion-retrieval-32M` @ dim-256** (`model2vec_embedder.rs:35`): +33% English recall at
  equal cost; keep multilingual as an option. The single biggest lever — but a multilingual→English product call.
- **#4 Reranker default → `RrfCombine`**: removes the −11%/−23% pure-reorder downside (updates 1 rerank test).
- **#3 Fusion defaults → stronger-tier weight ~1.3, `k`≈10, `Hash` tiebreak**: makes the hybrid strictly dominate.

**Known plumbing gap:** the per-tier weights / tiebreak are reachable via `RrfConfig` (a direct `rrf_fuse` call) but
not yet through the high-level `TwoTierSearcher` builder — exposing them there is blocked on `TwoTierConfig`'s ~35
construction sites (a large field-addition ripple), so it's deferred rather than forced through churn.

The full measurement trail, with self-corrections, is in `NEGATIVE_EVIDENCE.md`.
