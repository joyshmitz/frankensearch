# WIZARD BLINDSPOTS — CC

After the full adversarial exchange (both top-5 lists, both appendices, both critique files), these
are ideas **neither wizard proposed and the plan does not contain**. Same rigor as the originals.

---

## Blind spot 1 — The cross-process visibility contract: Q3's "searchable immediately" is true in exactly one process

**What it is.** Bet Q3 (plan §6.3) decouples visibility from durability: ingested docs are
searchable the instant the shard's delta snapshot Arc-swaps. But the delta is **process-local
memory**. Every consumer in a *different* process — and the README's flagship workflow is exactly
that: a long-running `fsfs index ~/projects --watch` daemon plus agents invoking `fsfs search`
as separate CLI processes — opens the index from MANIFEST and can only ever see sealed, published
segments. For cross-process readers, freshness is governed by *commit cadence*, exactly as with
tantivy today. Neither wizard flagged this; the plan never states it; and two of the duel's own
ideas silently interact with it: my group-commit appendix idea (defer sealing for churn reduction)
would *widen* cross-process staleness, and the e7.2 debounce retune measures "update→searchable"
in-process, where QG-3's "≥4× headroom" claim is real — while being potentially *zero* improvement
for the CLI-per-query agent topology.

**Why it matters.** This is a semantics honesty gap with a user-visible failure mode: an agent
edits a file, the watch daemon ingests it into the delta, the agent immediately runs `fsfs search`
in a new process and does not find it — and nothing anywhere explains why. Worse, tuning driven by
in-process metrics (shrink debounce, defer seals) can actively degrade the cross-process
experience while all dashboards improve.

**Implementation.**
1. **Contract row (quill-e0.1 / language contract):** visibility classes pinned explicitly —
   *in-process* (delta, immediate) vs *cross-process* (sealed+published, commit-cadence). One
   paragraph, zero code, prevents the QG-3 claim from being marketed beyond its topology.
2. **Freshness surfacing:** `segment_stats()` exposes `published_generation`,
   `last_publish_unix`, and `live_writer: bool` (detectable from the writer LOCK of my Idea 1);
   `fsfs status` and `--format json` search responses carry `index_freshness` so agents can act
   on staleness instead of being confused by it.
3. **Staleness bound:** a `max_visibility_lag_ms` policy in Keeper — seal-and-publish is forced
   when the oldest unpublished delta entry exceeds the bound (a cheap timer piggybacked on the
   existing seal triggers). This makes cross-process staleness a *configured guarantee* instead of
   an emergent property of batch sizes. `fsfs flush` gives agents an explicit barrier.
4. **QG-3 amendment (quill-e0.6 manifests):** measure update→searchable in *both* topologies
   (in-process reader; fresh-process reader) so the leapfrog claim is honest per topology.

**Risk/cost.** Low. Items 1/2/4 are documentation, a struct field, and a bench lane. Item 3 is a
small Keeper policy with LabRuntime coverage; its only tension is with group-commit churn
reduction, which is exactly the tradeoff the knob exists to make explicit. ~2 agent-days.

**Confidence: 0.8** that the gap is real and cheap to close (the delta's process-locality is
structural; the fsfs CLI topology is the documented agent workflow).

---

## Blind spot 2 — Blue-green index directories with a CURRENT pointer: make the flip (and every rebuild) instantly reversible

**What it is.** Both the G3 default flip and the upgrade path rely on *rebuild* (plan §16.2:
"fsfs rebuilds lexical indexes from canonical storage"; bead quill-e7.5 "rebuild-on-detect").
As specified, the rebuild is in-place: detect a tantivy-format (or older-FSLX) directory, rebuild
into it. That has two operational defects: a window with no searchable lexical index during
rebuild, and **no rollback** — if post-flip Quill misbehaves on some corpus in the wild, the
tantivy index is already gone, and reverting the binary means another full rebuild. Neither
wizard proposed the standard remedy: versioned sibling index directories
(`lexical/quill-v1/`, `lexical/tantivy/`) plus a tiny `CURRENT` pointer file updated with the
same two-slot atomic protocol FSLX already uses for MANIFEST (§11.4 — the discipline is already
in-tree; this just applies it one level up).

**Why it matters.**
- **The flip becomes a pointer swap, not a leap.** Combined with shadow mode (both wizards
  independently proposed it), the shadow Quill index is *already built and validated* on every
  opted-in machine — flipping means repointing `CURRENT` to an index that has been serving shadow
  traffic for weeks, warm and divergence-clean. Rollback is the same operation in reverse,
  in milliseconds, with zero rebuild.
- **Old index retention is RULE-1-correct by construction.** AGENTS.md forbids deletion without
  explicit permission; blue-green *naturally* retains the previous directory until a human-approved
  retirement sweep (which quill-e9.3 already schedules with exactly that approval step). The
  plan's in-place rebuild is, strictly read, in tension with RULE 1; blue-green dissolves the
  tension instead of managing it.
- **Every future format bump inherits the machinery.** The FSLX format registry (§10.6)
  anticipates format evolution; each `format_version` bump becomes "build v(n+1) beside v(n),
  swap, retain" — no special-case migration code, ever.

**Implementation.** A `CURRENT` file (engine kind + directory name + format version, checksummed,
temp+rename publish) read by `QuillIndex::open`/fsfs backend selection; rebuild orchestration in
fsfs (e7.5) targets a fresh sibling dir and swaps on success; `fsfs doctor` reports both
directories, the pointer, and disk cost; retirement of the old directory is a listed,
permission-gated cleanup (e9.3). Disk cost during transition = two lexical indexes (bounded,
reported); the shadow-mode deployment already pays it, so for shadow-opted machines the marginal
cost is zero. ~3 agent-days plus one new bead under quill-e7 blocking e7.5/e7.6.

**Risk/cost.** Low mechanics risk (the atomic-pointer discipline is proven in-tree). The real
cost is transient double disk usage on non-shadow machines — bounded, visible, and temporary.

**Confidence: 0.75.** Standard practice (blue-green/atomic-symlink deploys) applied at the seam
where this project is most exposed; the RULE-1 and shadow-mode synergies are specific to this
repo and make it stronger here than in the generic case.

---

## Blind spot 3 — Crash-resumable bulk builds: intermediate publishes + per-doc content hash in IDMAP

**What it is.** A 1M-doc initial index (or a flip-era rebuild, or a post-corruption rebuild —
see the quarantine debate) is minutes of work. The plan's crash-only doctrine makes an
interrupted build *safe* (uncommitted work is invisible garbage, GC'd on open — §11.4) but
completely *unsaved*: nothing in either wizard's list or the plan lets a restarted build avoid
redoing everything. Two small mechanisms make bulk builds resumable:

1. **Intermediate manifest publishes during bulk mode.** `configure_bulk_load` (§12.2) suppresses
   *merging*, but there is no reason to suppress *publishing*: every N sealed mini-segments, run
   the normal commit choreography. Crash-recovery then lands on the last published generation
   with all prior work intact. This is nearly free — it is the ordinary commit path on a timer —
   and it also bounds bulk-mode memory (sealed-but-unpublished segments cannot pile up).
2. **A per-doc `content_hash: u64` (xxh3) in the IDMAP entry** (8 bytes/doc; format-registry row).
   On resume, the driver probes each candidate DocId via IDHASH: present + hash-equal → skip;
   present + hash-differs → upsert; absent → add. The resumed session takes fresh docid leases
   (Q1 is untouched — burned lease tails are explicitly legal, §7), and the skip probe is the
   same IDHASH lookup upserts already perform.

**Why it matters.** Rebuild is this architecture's answer to *everything* — upgrades (e7.5),
corruption recovery, the flip itself, cass-compat migration someday. Making the
answer-to-everything interruption-tolerant converts the worst realistic operational event
("agent OOM-killed / machine rebooted at minute 9 of 10") from full restart into ~seconds of
re-verification plus the lost tail. It also hands fsfs a second, index-side witness for change
detection: today's incremental contract trusts storage-side state (`incremental_change.rs`);
IDMAP content hashes let `fsfs doctor` *audit* index↔storage consistency cheaply (sample DocIds,
compare hashes) — which is precisely the verification the quarantine debate showed to be missing.

**Implementation.** Registry row + IDMAP writer/reader change (quill-e2.6 amendment) while FSLX
v1 is still on paper — this is the cheap moment; retrofitting costs a format bump. Bulk publish
cadence in Keeper's bulk mode (quill-e3.7 amendment). Resume loop in fsfs's build driver (new
small bead under quill-e7). Crash-matrix rows: kill mid-bulk at each publish arrow, resume,
assert final index ≡ uninterrupted build's result *sets* (docids differ by lease burn — scores
and results identical per Q1-OB3 logic).

**Risk/cost.** Low-moderate. 8 bytes/doc is honest but real (~8MB at 1M docs — inside QG-7's
bytes/doc budget; must be declared in the gate manifest). Resume-equivalence needs its own
metamorphic fixture (interrupted+resumed ≡ uninterrupted, result-level). ~4 agent-days.

**Confidence: 0.7.** The mechanisms are small and ride existing machinery; docked slightly
because the resume driver touches fsfs orchestration, which is the most concurrently-edited part
of the tree.

---

## Honorable mentions (real gaps, below the line)

- **Generation time-travel for forensics:** manifest generations are already immutable files;
  retaining the last K plus `fsfs search --as-of-generation N` would let agents bisect "results
  changed after batch X" and let shadow-mode divergences be replayed against the *exact* snapshot
  that produced them (stamp the generation into the divergence artifact). Cheap; niche.
- **Single-switch conformance mode:** one config flag forcing all reference paths at once
  (scalar decode, exhaustive top-k, no pruning, single-thread, seal-everything/no delta) so a
  field-reported divergence can be bisected across the optimization lattice in four runs. The
  internal differentials (§15.2) test these axes pairwise in CI, but no *product-level* switch
  exists for debugging a live index.
