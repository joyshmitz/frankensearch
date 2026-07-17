# Quill Performance Gates — Activation Rules & Standing Laws

**Status:** Normative companion to `quill-perf-gates.toml` (the machine-readable manifests).
**Owning bead:** `bd-quill-e0-contracts-j53p.6`. **Design of record:** plan §14. **Harness owner:** `quill-e8.1` (bench matrix), `quill-e8.2` (ratchet).

## Activation discipline

Every gate ships `activated = false` until ALL of its pins are real: fixture committed (or generator landed for xlarge lanes), oracle config verified byte-identical in the harness (same analyzer semantics, same heap budget, commits inside timed windows for both engines), build profile applied, and the statistical rule wired. Activating a gate = a PR flipping the flag with the evidence linked. **No number from a non-activated gate may be quoted anywhere** (README, docs, commit messages) except marked "provisional, gate inactive".

## The five standing laws (bind every published number)

1. **No benchmark-only semantics.** Durability settings, commits, and result consumption match shipped defaults; no marker-only "commit latency"; no positions-off numbers marketed against positions-on defaults without saying so.
2. **Distributions, not averages.** p50/p95/p99 + cv_pct always; extreme quantiles only with sufficient observations.
3. **Never hide maintenance.** Merge/compaction/GC time inside the bulk-index window; foreground latency during background work is part of QG-6.
4. **Memory is first-class.** Bytes/doc itemizes postings/positions/dict/blockmax/idmap(+content_hash)/tombstones; RSS probes are per-OS (see toml `defaults.rss_probe`).
5. **One lever per change.** Every optimization lands alone with ≥0.1% frame attribution (local flamegraph lanes — RCH cannot symbolize, bd-e41k), keep-gated by the ratchet, and ledgered (`docs/PERF_LEDGER.md` wins, `docs/NEGATIVE_EVIDENCE.md` rejects with the Ratio convention; pre-flight ledger grep mandatory).

## `.bench-history` layout (decided here)

```
.bench-history/
  QG-<n>.<machine-class>.latest.json     # committed; the ratchet baseline
  QG-<n>.<machine-class>.<date>.json     # rolling window (last 30 kept)
```
Schema per file: `{gate, machine_fingerprint, git_rev, corpus_manifest_hash, cells: [{fixture, metric, value, cv_pct, runs}], laws_attested: bool}`. The ratchet script (quill-e8.2) refuses cells with `cv_pct >= 5` and refuses comparisons across differing `corpus_manifest_hash`.

## Topology honesty (QG-3/QG-4)

Update→searchable and visibility claims carry topology labels per the cross-process visibility contract (`bd-quill-duel-visibility-contract`): **in-process** (delta-visible once e5.x lands) vs **fresh-process** (published-generation freshness). G1a (scalar checkpoint) has no delta: QG-4's visibility-lead clause is N/A until bet Q3 lands as a lever — the manifests encode this so nobody quotes a visibility number the architecture doesn't yet earn.

## Cross-references

Gate manifests: `quill-perf-gates.toml`. Oracle pinning: gauntlet version contract (e0.5). Fixture corpora: fsfs golden profiles + xlarge generator (e6.1). Scaling/attribution method: e8.3/e8.4 notes. Flip evidence: QG-10 delta in `quill-e7.6`'s bundle.
