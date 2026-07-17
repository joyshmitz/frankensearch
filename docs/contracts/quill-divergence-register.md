# Quill Divergence Register

**Status:** Living ledger (append-only entries; `decision` fields may be updated with a dated edit).
**Owning bead:** `bd-quill-e0-contracts-j53p.4`. **Design of record:** plan §15.6. **Oracle:** tantivy `0.26.1` + `frankensearch-lexical` (pinned by the gauntlet's version contract).

## Doctrine

Every **intentional or discovered-and-accepted** behavioral divergence between Quill and the oracle is recorded here. Two rules govern the ledger:

1. **An empty register is not the goal; an *unclassified* divergence is the only failure.** The conformance gate (G2) blocks on divergences that match no register class — never on the register being non-empty.
2. **`accept` decisions require a consumer-impact note and second-agent review** (fresh-eyes rule). `fix` decisions must name the spawned bead. Review sign-off is recorded in the entry.

The gauntlet's comparator auto-classifies against §2's classes; anything it cannot classify fails the run and lands in triage (bd-quill-duel-shrinker's factor-diff bucketing feeds this).

## 1. Entry schema

```
### DIV-<NNN>: <short title>
- Class: <one of §2, or a NEW class added in the same commit>
- First seen: <date> · <suite/fixture id or shadow-generation stamp>
- Root cause: <precise mechanism, file/section refs>
- Consumer impact: <what a frankensearch user/agent could observe; "none observable" needs justification>
- Fixture: <committed fixture id that reproduces/pins the divergence>
- Decision: accept | fix (bead: <id>) | pending
- Reviewer: <second agent name + date for accepts>
```

## 2. Divergence classes (taxonomy)

| Class | Meaning | Default posture |
|---|---|---|
| `ScoreEpsilon` | |Δscore|/score ≤ 1e-4 with identical result *sets*; rank flips only inside epsilon-tied groups. Expected from segment-geometry-dependent stats and libm `ln` platform variation (see quill_contract.rs conventions: never bit-pin `ln` outputs) | accept-by-class (bounded) |
| `TieOrder` | Equal-score results ordered differently; both engines valid under their own tie rules; comparator canonicalizes by (score desc, DocId asc) before diffing — entries here only for ties *visible through the public API* | accept-by-class |
| `SnippetWindow` | Same matched terms, different window choice on coverage ties; tags/lengths identical | accept-by-class (cosmetic) |
| `GlobExpansionLimit` | Wildcard expansion hits the bound with a different candidate subset/order than the oracle's expansion | accept per-entry (requires impact note) |
| `QueryCanonicalization` | Reserved for post-flip AST canonicalization if score-affecting dedup is ever enabled (bd-quill-duel-ast-dedup; the score-neutral cursor-sharing form needs NO entry) | fix/off by default |
| `OracleBug` | The oracle's behavior is wrong per its own documentation and Quill deliberately does not reproduce it | accept per-entry (needs upstream citation) |
| `StatsSemantics` | Deletes-vs-stats or delta-vs-sealed stats timing differences not covered by fixtures pinning oracle behavior (e0.1 row 3, e4.3/e5.2 notes) | pending → must converge to fix or a pinned accept |
| `UnicodeEdge` | Analyzer divergence on degenerate inputs (unpaired surrogates cannot occur in &str; this class covers e.g. exotic casing/width edge cases) proven byte-parity-impossible or oracle-inconsistent | accept per-entry |

Adding a class = a PR that adds the row here **and** teaches the comparator to classify it, in the same commit.

## 3. Seeded expectations (not yet observed — placeholders awaiting first evidence)

These are the classes the plan *predicts*; each becomes a numbered DIV entry when first observed with a real fixture:

- `ScoreEpsilon` from cross-platform `ln` (x86 vs Apple Silicon differential lanes).
- `TieOrder` on synthetic corpora with duplicated documents.
- `SnippetWindow` on documents where two windows have equal term coverage.
- `GlobExpansionLimit` on >limit-term dictionaries with `Complex` patterns.

## 4. Entries

*(none yet — first entries expected from the gauntlet's G1 fast-lane differentials)*

---

*Cross-references: comparator classes implemented in the gauntlet kernel (bead e0.5); auto-triage feeding this ledger (bd-quill-duel-shrinker); statistical gates consuming per-class pass rates (bead e6.6); G2 exit requires this register complete over two consecutive nightly runs (bead e6.8).*
