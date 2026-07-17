# WIZARD REBUTTAL — CC

Formal rebuttal to `/data/projects/frankensearch/WIZARD_SCORES_GMI_ON_CC.md`. GMI's scores on my
five ideas were high (980/850/920/890/750), so this rebuttal does two things the duel actually
needs: (A) defend the one idea whose *scoring reasoning* was most wrong even though its number was
merely adequate — Idea 5; and (B) argue, against my own interest, that GMI was too **generous** in
one place — Idea 1 at 980 — because it found a real flaw and then failed to price it. Both halves
attack the weakest parts of GMI's scoring method: it graded outcomes without auditing whether its
own flaw-findings were (a) accurate against the text it was scoring and (b) reflected in the
number it produced.

---

## A. The most wrongly-treated idea: Fuel-metered, cancel-aware queries (scored 750)

The 750 is defensible; the *reasoning* under it is not, and reasoning is what cross-scoring is
supposed to produce. Three specific failures:

**1. The "sharpest flaw" critiques a design that is not in my file.** GMI: checkpoints "inside the
tightest inner loops (e.g., MaxScore decoding or SWAR snippet windowing) introduces branch
overhead." My specification (WIZARD_IDEAS_CC.md, Idea 5, implementation sketch) places checkpoints
at deterministic *block* boundaries — "every K posting blocks decoded, every dictionary
prefix-block scanned, every segment boundary in the k-way merge" — i.e., structurally outside the
`wide` u32x8 unpack kernels, at a stated cost of "one branch + one relaxed load per 128-doc
block." Nothing in my text proposes per-element or per-iteration checks. And "SWAR snippet
windowing" is not my design at all — it is GMI's own Idea 5, imported into a critique of mine.
A scorer that attributes its own architecture to the opponent and then penalizes the hybrid has
not evaluated the submission.

**2. The gate citation is wrong.** GMI: overhead "directly threatens the sub-15ms p99 latency
target (QG-6)." Plan §14 defines QG-6 as *p50 parity (±10%), p99 ≤ tantivy per query class*. The
<15ms number is the two-tier Phase-1 delivery envelope (README / plan §2.4 throughput contract
context) — a different budget, owned by the fusion layer, of which lexical search is one
parallel arm. Conflating them inflates the apparent risk: a one-relaxed-load-per-1024-docs
checkpoint cadence is ~0.1%-class on the lexical kernel, which is itself a fraction of the
Phase-1 envelope.

**3. The improvement condition would damage the feature it conditions on.** GMI's
"what would change my score": restrict checkpoints "to coarse-grained boundaries (e.g.,
per-segment merge or dictionary prefix-block edges) to guarantee zero impact." Per-segment
granularity makes cancellation latency unbounded in exactly the adversarial case the feature
exists for: a single high-df term scanned within one segment of a 1M-doc index is on the order of
8,000 blocks (~1M postings) of uninterruptible work — the pathological-query hang, reintroduced.
Note the internal tension: GMI *accepts* dictionary prefix-block-edge checks (fine-grained, on the
glob path) while rejecting posting-block-edge checks (the same granularity class, on the scoring
path). The empirically correct arbiter was already in my spec: the mandatory checkpoint-on/off A/B
with a [0.97, 1.03] keep-gate — precisely the house discipline (§14 law 5) for resolving
"does this branch cost anything" questions. Conditioning the score on adopting a *weaker* design
instead of on the measurement gate is backwards.

If the flaw paragraph had engaged the actual design, the residual criticisms (fuel defaults need
tuning; the win is insurance, not a headline) are the ones I myself declared at confidence 0.7.
On its own stated methodology, this idea merits ~800: the mechanism is contract-compatible by
GMI's own analysis ("Returning `Err(Cancelled)` legally satisfies the `poll_immediate` contract"),
uniquely LabRuntime-testable, and the only cost question is one that a mandated gate already
answers.

---

## B. Where GMI was too generous: the writer lock at 980

I will not pretend this is against my interest gracefully — it is my best idea and I want it to
win — but the duel is scored on candor, so: **980 was too high for the idea as I submitted it,
and GMI's own review proves it.** GMI correctly identified that heartbeat/staleness advisory
locks are unsound under suspended processes (SIGSTOP — which this repo's own process-triage
tooling performs on agents as a matter of routine). That observation is not a cosmetic caveat; it
punctures the *specified mitigation*: my layer-3 backstop ("publish-time CAS on generation" via
read-manifest-then-verify-then-rename) is TOCTOU, not CAS — two SIGSTOP-resumed writers can both
read generation N and both perform the atomic rename, and the second silently clobbers the first.
So as written, my design reduced the multi-writer window but did not close it; the data-loss bug
I found in the plan had a smaller cousin surviving inside my own fix.

The repair is straightforward and I have adopted it (see WIZARD_REACTIONS_CC.md): generation
publication via `OpenOptions::create_new` on generation-named manifest files
(`MANIFEST.<gen+1>`, O_EXCL — a true filesystem CAS: exactly one racing writer succeeds; the
loser gets a typed conflict error), with "current" defined as highest-numbered checksum-valid
generation and old generations GC'd under the writer lock. With that amendment the design is
airtight against suspension races, and the lockfile demotes to an efficiency/ergonomics device.

A scorer that discovers a correctness hole in the submitted mitigation should dock for it: the
submission's value = (bug discovery: near-max) + (fix: strong but incomplete as specified). I'd
put the submitted version at ~925, and the amended version (O_EXCL generation CAS) at the 980
GMI awarded. Awarding the amended score to the unamended text is grade inflation, and it matters
beyond politeness: if the flip-era implementation bead had been written from my original text,
it would have shipped the TOCTOU.

---

## C. Two smaller scoring-reasoning defects worth recording

**Shadow mode (920):** the flaw paragraph treats query-side and ingest-side costs as one
("doubles the I/O and memory bandwidth pressure... spurious timeout errors"), but the remedy it
conditions the score on (dynamic shedding on CPU/IOps thresholds) only addresses the query side —
the dual-index ingest cost cannot be sampled or shed, because a partial shadow index is not
comparable. The correct decomposition: query-side interference is bounded by the drop-compare
bounded channel already specified; ingest-side cost is irreducible and is exactly why the mode is
opt-in and time-boxed. A score condition that cannot in principle satisfy its own concern is not
a real condition.

**Skeleton-first (850):** the sharpest-flaw (trait thrash) attributes to the skeleton a cost the
plan already owns: the cursor trait is pre-specified by quill-e4.3/e2.2 with delta (E5), BMW
(e4.4), and block-skip requirements known before any code lands, and the accumulator seam must be
abstracted anyway because quill-e1.7's A/B mandates both hash-accumulate and columnar-radix
implementations regardless of sequencing. The marginal thrash risk introduced *by resequencing*
is close to zero; the 850 was fair but the stated reason for withholding the rest was not.
