# WIZARD STEELMAN — CC argues the strongest case for GMI's Idea 1

Assignment: present the Snapshot-Epoch BM25 Memoization Table (WIZARD_IDEAS_GMI.md, Idea 1 —
which I scored 480) as if I had to ship it and defend it to a review board. Errors corrected
charitably; best version presented; revised score at the end.

---

## The pitch, correctly stated

BM25's per-posting score is `w_t · tf/(tf + K(norm))`, where `w_t = idf·(k1+1)·boost` is per-term,
`tf` is the posting's term frequency, and `K(norm) = k1·(1−b+b·|d|/avgdl)` depends only on the
1-byte quantized fieldnorm and the snapshot's per-field `avgdl`. Tantivy already memoizes `K`
(the 1D 256-entry cache, `tantivy-0.26.1/src/query/bm25.rs:62`), leaving one f32 **division** per
scored posting. The steelman observation: *within a snapshot epoch, the entire two-argument
function `tf/(tf + K(norm))` is a constant table* — and unlike almost every other scoring
micro-optimization, eliminating the division attacks the one operation a modern core cannot hide
in a lean scalar loop. On Zen 3 (the RCH fleet's reference class), `divss` occupies a
partially-pipelined divider with ~3.5–5-cycle reciprocal throughput; a scoring loop that
otherwise sustains ~1 posting per 3–4 cycles is *divider-bound*, so removing the division is not
a percent-shaving — it can re-shape the loop's throughput ceiling.

## Right-sizing the table (fixing the 256KB error — and partially vindicating "64KB")

The naive `[256 freqs][256 norms] × f32` is 256KB per field — L2-resident, self-defeating. But the
freq axis does not need 256 entries: term frequencies are Zipf-distributed; within-document tf
values are overwhelmingly small. Cap the memoized axis at `TF_CAP = 32`:

```
table: [[f32; TF_CAP]; 256]   // norm-major: 256 rows × 32 entries × 4B = 32 KiB per field
score = w_t * if tf < TF_CAP { table[norm][tf] } else { exact_tf_part(tf, K(norm)) }
```

- **32 KiB per field**; the default schema has two scored text fields (`content`, `title`) →
  64 KiB total. (Charitable note: GMI's "64KB" figure is exactly right for the version of this
  idea it *should* have proposed.)
- **Norm-major layout** is the right one: `norm` varies per document while `tf` is small, so each
  document touches one 128-byte row (2 cache lines); document lengths cluster heavily in real
  corpora, so the hot rows stay L1-resident even on 32KB L1d.
- The `tf >= TF_CAP` fallback is a predictable, rarely-taken branch computing the exact
  expression — no correctness cliff, and Zipf makes it cold.

## Handling the objections I raised at 480

**"Tantivy already has the 1D cache, so the baseline comparison is wrong."** True, and the honest
A/B is 2D-table vs 1D-cache — but the honest A/B still has a real prize: the 1D cache leaves a
division *per posting*; the 2D table leaves a multiply. Division is the most expensive scalar f32
op in the loop by a factor of ~5–10 in throughput terms. The claim "strips division from the
tightest loop" survives; only the claim "strips multiplication" dies (the per-term `w_t` multiply
is irreducible without per-term tables, which would be memory-madness).

**"The f32 op-order breaks bit-parity."** This is the strongest objection and it has a clean,
*time-sensitive* resolution: the Language Contract has not landed yet (bd-quill-e0.1 is open;
only e0.7 is in progress). Define Quill's pinned scalar reference — in the contract, now — as
`score += w_t * (tf / (tf + K(norm)))`, i.e., the table's association, rather than tantivy's
`(w_t * tf) / (tf + K(norm))`. Then the table is bit-identical to Quill's own reference *by
construction* (a memoized pure function is its own specification), and the internal differential
gates (§15.2, SIMD≡scalar, pruned≡exhaustive) are unaffected. Cross-engine, the association
difference vs tantivy is ~1 ulp (~1e-7 relative) — four to five orders of magnitude inside the
ScoreEpsilon budget (1e-4, plan §5.3). The one honest residual: a 1-ulp shift can in principle
flip the order of two *near*-tied (not exactly tied) docs in a RankExact-gated configuration; the
comparator's epsilon-tie grouping (§15.1) exists for precisely this, and any residue lands as a
classified TieOrder row in the Divergence Register rather than a gate failure. Crucially, this
resolution is *free if chosen before G0 freezes the contract and expensive after* — which is by
itself a reason this idea deserved to be surfaced now, in the design phase, exactly as GMI did.

**"It's just a lever; measure it."** Yes — and here is the steelman's best structural argument for
why the measurement is likely to come back positive on the path that matters most. Plan §8.4
mandates that *contract-mode scoring accumulates scalar, in parse order* — SIMD is allowed for
decode/gather only. That constitutional choice forecloses the usual escape hatch (vectorize the
scoring loop, amortize the division across 8 lanes with `vdivps`, or use `rcpps`+Newton in a
relaxed mode). In other words: **the contract forbids the SIMD remedies for division cost, which
makes the table the only legal way to remove the division from the mandated path.** The idea is
not competing against hypothetical vectorized scoring; it is competing against a scalar loop that
must keep its divide. That is a much better fight.

**Block-max/BMW soundness.** The same table serves the bound computation with monotonicity intact:
`tf_part` is increasing in tf and decreasing in `K`, so the block bound is
`w_t · table[min_fieldnorm][min(max_freq, TF_CAP−1)]` with the exact-compute fallback above the
cap — the stored `(max_freq, min_fieldnorm)` pair from e2.3 feeds it directly, and bound ≥ true
value is preserved entry-for-entry because the bound and the score now read the *same* memoized
function (no cross-formula drift between pruning and scoring, which is itself a small correctness
simplification for the e4.4 rank-safety argument).

**Lifecycle cost.** Rebuild on snapshot publish = 256×32 evaluations per field (~16k flops):
irrelevant even at watch-mode delta-publish cadence. Table lives in the `Arc<Snapshot>`; readers
get it lock-free; determinism is trivial (pure function of pinned snapshot stats).

## Ship plan (as I would defend it to the board)

1. **Now (G0):** land the association choice in the Language Contract (e0.1) — one sentence, zero
   code, keeps the option open at zero cost. This is the only genuinely urgent part.
2. **G1:** the scalar reference kernel (e4.3) computes via the contract expression; no table yet.
3. **Post-baseline (e8.5 lever bead):** implement `Bm25Table` (32 KiB/field, norm-major, TF_CAP=32,
   exact fallback), A/B vs the 1D-cache baseline on the medium/xlarge profiles per query class,
   MT8 flamegraph attribution ≥0.1%, keep-gate [0.97, 1.03], ledger either way. Expected
   honest range: low single digits end-to-end on disjunction/exhaustive-heavy classes (scoring's
   share of query time), potentially more on high-df term scans; a wash on dict-bound classes.

## Revised score

As submitted (256KB table, wrong baseline, unexamined parity, "99% slam dunk"): **480 stands** —
the errors were real and the confidence was miscalibrated. The best version above — freq-capped
32KiB norm-major table, contract-level association fix landed at G0, same-table BMW bounds,
1D-cache-honest A/B under e8.5 discipline — **I would score 680**, and I would genuinely advocate
adding its step 1 (the contract sentence) to bd-quill-e0.1 immediately, because that part is
free, reversible, and time-critical, while everything else waits safely behind a flamegraph.
