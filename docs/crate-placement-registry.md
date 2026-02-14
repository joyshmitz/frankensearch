# Crate-Placement Registry (bd-33iv)

## Purpose

This document defines the canonical placement contract for active implementation beads.  
The machine-readable source of truth is `docs/crate-placement-registry.json`, validated by:

- `schemas/crate-placement-registry-v1.schema.json`
- `scripts/check_crate_placement_registry.sh`

The goal is to stop crate-placement drift in bead comments and reviews by enforcing one shared mapping policy.

## Scope

The registry checker tracks implementation beads (`task`, `feature`, `bug`) in active states (`open`, `in_progress`) from `.beads/issues.jsonl`.

Coverage requirement:

1. Every active implementation bead must match exactly one placement rule.
2. Duplicate rule matches are treated as placement drift.
3. Unresolved placement lanes (`conflict` / `unknown`) must declare owner, deadline, and blocking impact.

## Canonical Files

- Registry: `docs/crate-placement-registry.json`
- Schema: `schemas/crate-placement-registry-v1.schema.json`
- Valid fixture: `schemas/fixtures/crate-placement-registry-v1.json`
- Duplicate-drift fixture: `schemas/fixtures-invalid/crate-placement-registry-invalid-duplicate-v1.json`
- Lint script: `scripts/check_crate_placement_registry.sh`

## Rule Summary

| Rule | Bead Pattern | Status | Primary Placement Surface |
|---|---|---|---|
| `R-2HZ` | `bd-2hz*` | resolved | `crates/frankensearch-fsfs/src/**` |
| `R-2YU` | `bd-2yu*` | resolved | `crates/frankensearch-ops/src/**` |
| `R-3W1` | `bd-3w1*` | resolved | `crates/frankensearch-storage/src/**`, `crates/frankensearch-durability/src/**` |
| `R-3UN` | `bd-3un*` | resolved | core/embed/index/lexical/fusion/rerank + facade crates |
| `R-3QWE` | `bd-3qwe*` | resolved | `.beads/issues.jsonl`, policy docs, lint scripts |
| `R-1GFX` | `bd-1gfx` | conflict | cross-crate hardening (`crates/**`, `tests/**`, `benches/**`) |
| `R-EHUK` | `bd-ehuk` | conflict | interaction matrix code + tests + policy docs |

Refer to the JSON registry for complete path targets, rationale, and ownership metadata.

## Validation Commands

```bash
# Schema + unit consistency + active coverage
scripts/check_crate_placement_registry.sh --mode all

# Unit only (schema, unique rules, conflict deadline sanity)
scripts/check_crate_placement_registry.sh --mode unit

# Integration against active implementation beads
scripts/check_crate_placement_registry.sh --mode integration --scope active

# Integration gate for newly edited bead plans (changed ids from git diff)
scripts/check_crate_placement_registry.sh --mode integration --scope changed
```

## Diagnostics Contract

The checker emits JSON findings (`rule_id`, `severity`, `bead_id`, `message`, `fix_hint`) and a final summary line.

Severity behavior:

- `error`: fails the check (missing mapping, duplicate mapping, invalid schema, overdue unresolved conflict, unresolved mapping in `--scope changed`)
- `warning`: non-blocking but tracked (active beads mapped to unresolved conflict lanes)

## Change-Management Workflow

1. Update bead metadata in `.beads/issues.jsonl` (new/edited implementation bead).
2. Update `docs/crate-placement-registry.json` with a resolved rule or explicit unresolved conflict record.
3. Validate schema:
   `jsonschema -i docs/crate-placement-registry.json schemas/crate-placement-registry-v1.schema.json`
4. Run lint checks:
   `scripts/check_crate_placement_registry.sh --mode all`
5. Post results in the bead thread (include unresolved conflicts and owner/deadline if any remain).

## Conflict Handling

For `placement_status = conflict` or `unknown`, the registry entry must include:

- `resolution_owner`
- `resolution_deadline` (ISO date)
- `blocking_impact` (`low|medium|high|critical`)

This keeps unresolved placement decisions visible and auditable while work continues.
