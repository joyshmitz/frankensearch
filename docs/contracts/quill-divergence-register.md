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
| `ScoreEpsilon` | `abs(a-b) / max(abs(a), abs(b), 1e-12) ≤ 1e-4` with identical result *sets*; rank flips only inside maximal connected components of epsilon-adjacent oracle scores, assigned in oracle total-order before inspecting Quill. Expected from segment-geometry-dependent stats and libm `ln` platform variation (see quill_contract.rs conventions: never bit-pin `ln` outputs) | accept-by-class (bounded) |
| `TieOrder` | Identical-score results differ only because Tantivy orders by ascending `DocAddress(segment_ord, segment-local doc_id)` while Quill orders by its global u32 docid. The comparator preserves native order, verifies the difference is confined to an expanded exact-score tie group (including top-k cutoff substitutions), and reports the public ordering impact; it never canonicalizes the difference away before classification. | accept-by-class |
| `SnippetWindow` | Same matched terms, different window choice on coverage ties; tags/lengths identical | accept-by-class (cosmetic) |
| `GlobExpansionLimit` | Wildcard expansion hits the bound with a different candidate subset/order than the oracle's expansion | accept per-entry (requires impact note) |
| `QueryCanonicalization` | Observable match- or score-affecting query-AST lowering differences, including parser repairs and any future score-affecting dedup (bd-quill-duel-ast-dedup). Score-neutral cursor sharing needs no entry. | fix/off by default |
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
- A dedicated class for score-neutral oversized query-token normalization is blocked on the G1 comparator and executable Boolean-shape proof (`bd-quill-e0-contracts-j53p.8`); it is not accepted under `QueryCanonicalization` by analogy.

## 4. Entries

### DIV-001: standalone CASS negation loses complement semantics

- Class: `QueryCanonicalization`
- First seen: 2026-07-17 · `query-boolean-negative-standalone-universe`
- Root cause: `cass_build_boolean_query_clauses` emits a lone `MustNot` clause for `-term`; Tantivy's raw negative-only `BooleanQuery` matches nothing, while complement semantics require an `All` clause alongside the exclusion. OR-operand lifting already creates that wrapper, so the shapes disagree inside the shipping adapter.
- Consumer impact: a standalone negative CASS query returns zero hits instead of every live document not matching the excluded term. Positive `AND NOT` shapes are unaffected and must not receive an `All` scorer.
- Fixture: `query-boolean-negative-standalone-universe`
- Decision: fix (bead: `bd-2b2u`)
- Reviewer: not required for a fix decision

### DIV-002: CASS anchored globs collapse to `AllQuery`

- Class: `OracleBug`
- First seen: 2026-07-17 · `query-glob-suffix` / `cass_parser_result_sets_match_the_shipping_tantivy_builder`
- Root cause: `CassWildcardPattern::to_regex` emits explicit `^`/`$` assertions, but pinned `tantivy-fst 0.5.0` rejects zero-width assertions and already matches regexes against the whole term. `cass_build_term_query_clauses` ignores both title/content construction failures; an empty top-level clause list then becomes `AllQuery`.
- Consumer impact: lone suffix globs such as `*bar` return every document. Affected complex wildcard operands can silently disappear from compound or filtered queries. Substring globs and complex globs bounded by `*` at both ends are unaffected by this anchor failure.
- Fixture: `query-glob-suffix` plus the result-level differential named above
- Decision: fix (bead: `bd-cass-wildcard-fst-anchors-t3f9`)
- Reviewer: not required for a fix decision

---

*Cross-references: comparator classes implemented in the gauntlet kernel (bead e0.5); auto-triage feeding this ledger (bd-quill-duel-shrinker); statistical gates consuming per-class pass rates (bead e6.6); G2 exit requires this register complete over two consecutive nightly runs (bead e6.8).*
