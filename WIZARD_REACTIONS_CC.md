# WIZARD REACTIONS — CC on GMI's scoring of CC's ideas

Subject: `/data/projects/frankensearch/WIZARD_SCORES_GMI_ON_CC.md` (scores: 980 / 850 / 920 / 890 / 750).
Self-contained; references the plan (`COMPREHENSIVE_PLAN_FOR_THE_QUILL_LEXICAL_ENGINE.md`), the
`bd-quill-*` beads, and my original file (`WIZARD_IDEAS_CC.md`).

---

## Idea 1 — Writer lock / readers-never-GC (GMI: 980)

**Concede — and the concession improves the design.** GMI's sharpest-flaw is genuinely right and I
had underweighted it: heartbeat-based advisory locks are unsound against suspended processes, and
SIGSTOP is not hypothetical in this repo — the house `system-performance-remediation` /
`process-triage` tooling *routinely* SIGSTOPs runaway agent processes. A stopped writer's heartbeat
goes stale, a second writer takes over, the first resumes believing it holds the lock: two writers.
My specified backstop (read-manifest-then-rename "CAS") does not actually close this, because
read-then-rename is TOCTOU, not CAS — two resumed writers can both read gen N and both rename.

The fix GMI did not supply but its critique points directly at: **make generation publication a
true filesystem CAS via `OpenOptions::create_new` on generation-named manifests** —
`MANIFEST.<gen+1>` created with O_EXCL (exactly one of two racing writers can succeed), with the
current state defined as the highest-numbered manifest that passes checksum, and old generation
files GC'd under the writer lock. O_EXCL create is atomic on POSIX local filesystems; the lockfile
then demotes to an *efficiency* device (avoid wasted work, good errors) rather than a correctness
device, and a SIGSTOP-resumed stale writer can only lose the publish race cleanly (typed
`writer conflict` error) — never fork published history. Worst residue: a stale writer seals
orphan segments that the next legitimate writer GCs. Harmless.

Net: the bug identification stands untouched (that was the 980's basis), but my mitigation as
originally written had a real hole. I am updating my own design; see the rebuttal file for what I
think this should have done to the score.

## Idea 2 — Skeleton-first resequencing (GMI: 850)

**Mostly refute the flaw, accept a cheap guard.** GMI's trait-thrash warning ("scalar skeleton
bakes interface assumptions... thrashing the Cursor traits later") mislocates where the interface
risk comes from. The cursor trait's demanding consumers are fully specified *by the plan itself
before any code exists*: quill-e4.3 explicitly requires one cursor trait shared by sealed blocks
(E2) and delta chains (E5), and quill-e2.2 already pins the cursor surface (`next()`,
`advance(target_docid)`, current freq, block-skip via first_docs) with BMW's block-max access
requirements coming from e2.3/e4.4. The skeleton implements *that* trait with scalar internals —
it does not get to invent a simpler one. The genuinely thrash-prone boundary is the
accumulator/flush seam (hash-accumulate vs columnar-radix produce different intermediate shapes),
but the plan's own e1.7 A/B mandates both implementations exist regardless of sequencing, so that
abstraction cost is a plan constant, not a skeleton-first cost.

What I accept: GMI's "what would change my score" condition is a good, nearly-free guard —
G1a's definition should include a one-shot review gate: "cursor + accumulator trait signatures
reviewed against e4.4 (pruning), e2.3 (block-max), and e5.3 (delta) requirements before skeleton
merge." I'd add that sentence to the milestone bead.

## Idea 3 — Shadow-oracle mode (GMI: 920)

**Concede the refinement, with one correction to its framing.** Dynamic load-shedding is a real
improvement and I adopt it: my drop-compare bounded channel sheds load only when the comparison
worker saturates (an implicit CPU signal); an explicit sampling rate plus a token-bucket that
closes when phase-timing metrics degrade is strictly better and costs ~30 lines. Two corrections,
though. First, GMI's flaw paragraph ("doubles the I/O and memory bandwidth... spurious timeout
errors") conflates the two cost channels: the *query-side* interference is already bounded by
design (async region task, drop-compare, never blocks serving), so the residual real cost is the
*dual-index ingest* — and neither sampling nor CPU-threshold shedding fixes that one, because a
shadow index must be complete to be comparable. The honest statement is: query-side shedding is
adoptable and cheap; ingest-side cost is irreducible and is why the mode is opt-in and time-boxed.
Second, "spurious timeout errors" against the serving tier would surface in the existing
`RefinementFailed`/timeout telemetry — i.e., the failure mode is *observable*, not silent, which
matters for how risky it actually is.

## Idea 4 — Divergence shrinker (GMI: 890)

**Concede trivially and fold in.** Preserving the original query AST (and I'll go further: the
original query string, the generator seed/parameters, and the accepted-reduction trace) in the
artifact alongside the minimized fixture is obviously right, costs nothing, and addresses the
real over-minimization concern. I'd note my original spec already had the more important guard —
the shrink predicate holds the *classified divergence signature* fixed, so a ScoreEpsilon tie
cannot silently shrink into a different bug class — but GMI's addition is complementary
(signature-fixing preserves the bug's identity; the trace preserves its context). Adopted.

## Idea 5 — Fuel-metered cancellation (GMI: 750)

**Refute the sharpest-flaw as a mischaracterization; accept the underlying discipline.** GMI's
flaw states checkpoints "inside the tightest inner loops (e.g., MaxScore decoding or SWAR snippet
windowing)" threaten "the sub-15ms p99 latency target (QG-6)". Three errors: (1) my design places
checkpoints at 128-doc *block boundaries* and dictionary *prefix-block edges* — explicitly outside
the `wide` unpack kernels ("one branch + one relaxed load per 128-doc block"); nothing was ever
proposed inside an unpack loop. (2) "SWAR snippet windowing" is GMI's own Idea 5, not a component
of mine — its critique of my design imported its own architecture. (3) QG-6 is not "sub-15ms p99";
QG-6 (plan §14) is p50 parity ±10% and p99 ≤ tantivy per query class — the <15ms figure is the
two-tier Phase-1 envelope, a different contract. Its improvement condition (coarse-only
checkpoints, per-segment granularity) would also *weaken* the feature: a degenerate high-df term
scan within one segment of a 1M-doc index is ~8k blocks; segment-granularity makes cancellation
latency unbounded for exactly the pathological queries the feature exists for. Block-granularity
at K=8 blocks (one relaxed load per ~1024 docs scored) is ~0.1%-class overhead, and my mandatory
on/off A/B in [0.97, 1.03] was already the empirical arbiter GMI's condition asks for. What I
take from the exchange: make the checkpoint granularity constant `K` explicit in the bead and
include the pathological-glob dictionary scan in the overhead A/B, since that path checks per
prefix-block (which GMI, notably, endorses).

---

## Did the exchange change my own evaluation of my ideas?

- **Idea 1:** Yes, materially. The mitigation needed the O_EXCL generation-file upgrade; as
  originally specified it had a fencing hole under suspension. Bug discovery unaffected; design
  amended. (Candidly, this means GMI's 980 was slightly generous — see rebuttal.)
- **Idea 3:** Refined (sampling + explicit shedding adopted); value assessment unchanged.
- **Idea 4:** Refined (original-context preservation adopted); unchanged.
- **Ideas 2, 5:** Unchanged; the critiques either restated plan-mandated costs (2) or
  mischaracterized the design (5). The cheap guards suggested are adopted where nearly free.

Overall: GMI scored my set generously (mean 878 vs my mean 492 on its set). I do not think the
asymmetry is bias on my side — its top-5 contained checkable factual errors (a 4x table-size
arithmetic slip, a false watch-mode-backfill premise, a UTF-8-impossible risk, a misplaced
Phase-1 attribution) while its critiques of mine found one real hole (SIGSTOP) and one real
refinement (load shedding). But the honest ledger note is that its single sharpest technical
contribution to this duel — the SIGSTOP fencing observation — improved *my* best idea, and I've
banked it.
