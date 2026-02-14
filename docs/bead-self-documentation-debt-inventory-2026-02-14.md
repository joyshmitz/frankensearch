# Bead Self-Documentation Debt Inventory (2026-02-14)

Generated: `2026-02-14T08:16:03Z`  
Insights data hash: `6e21828305e4d4cb`  
Insights generated_at: `2026-02-14T08:16:03Z`

## Summary
- Active beads with self-documentation debt: **144**
- Missing both rationale and evidence anchors: **144**
- Missing rationale only: **0**
- Missing evidence only: **0**
- Explicit exception markers: **0**

## Wave Assignment
- `wave_1`: **104**
- `wave_2`: **31**
- `wave_3`: **9**
- `exception_register`: **0**

## Deterministic Regeneration
```bash
# Rebuild inventory artifact from current .beads/issues.jsonl + bv insights
scripts/generate_bead_self_doc_inventory.sh --date 2026-02-14

# Sanity check item count
jq '.items | length' docs/bead-self-documentation-debt-inventory-2026-02-14.json
```

## True Debt vs Exception
- True debt: beads missing either `[bd-3qwe self-doc] RATIONALE` or `[bd-3qwe self-doc] EVIDENCE` (or both).
- Exception register: beads carrying explicit `[bd-3qwe self-doc] EXCEPTION` markers (currently 0).

## Owner Suggestions (Top 10)
| Suggested Owner | Items |
|---|---:|
| `component-owner` | 121 |
| `OliveBasin` | 2 |
| `SageHollow` | 2 |
| `SunnyCardinal` | 2 |
| `composition-lane` | 2 |
| `CopperCove` | 1 |
| `CrimsonBay` | 1 |
| `DarkFox` | 1 |
| `GentleOriole` | 1 |
| `IcyBeaver` | 1 |

## Top Wave-1 Candidates (first 20)
| Bead | Priority | Status | Class | Risk | Betweenness | Critical Path |
|---|---:|---|---|---:|---:|---:|
| `bd-ehuk` | 0 | open | gate | 10 | 1541 | 5 |
| `bd-3un.52` | 0 | open | implementation | 10 | 1097 | 13 |
| `bd-2yu.4.3` | 0 | open | implementation | 10 | 425 | 23 |
| `bd-2yu.4.2` | 0 | in_progress | implementation | 10 | 274 | 24 |
| `bd-2yu.4.1` | 0 | in_progress | implementation | 10 | 193 | 29 |
| `bd-2hz.3` | 0 | open | implementation | 10 | 165 | 7 |
| `bd-2yu.3.1` | 0 | in_progress | implementation | 9 | 82 | 25 |
| `bd-2hz.5` | 0 | open | implementation | 9 | 23 | 5 |
| `bd-2hz.4` | 0 | open | implementation | 9 | 19 | 6 |
| `bd-2hz.8` | 0 | open | implementation | 9 | 14 | 5 |
| `bd-ls2f` | 1 | open | implementation | 9 | 1460 | 14 |
| `bd-2hz.10.11.7` | 1 | open | implementation | 9 | 1213 | 15 |
| `bd-2hz.10.11.5` | 1 | open | implementation | 9 | 1038 | 16 |
| `bd-2yu.8.3` | 1 | open | implementation | 9 | 899 | 19 |
| `bd-2hz.3.4` | 1 | open | implementation | 9 | 831 | 27 |
| `bd-2hz.3.5` | 1 | open | implementation | 9 | 754 | 25 |
| `bd-2yu.9.4.1` | 1 | open | implementation | 9 | 701 | 12 |
| `bd-2hz.5.2` | 1 | open | implementation | 9 | 683 | 26 |
| `bd-2hz.10.5` | 1 | open | implementation | 9 | 679 | 20 |
| `bd-2yu.9.2` | 1 | open | implementation | 9 | 654 | 16 |

## Full Inventory Artifact
- `docs/bead-self-documentation-debt-inventory-2026-02-14.json` contains all 144 classified items with severity, centrality, risk, owner suggestions, and wave assignment.

## Downstream References
- Unblocks `bd-3qwe.3` and `bd-3qwe.5` with prioritized wave targets.
- Complements CI lint policy work under `bd-3qwe.7.*`.
