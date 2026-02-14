# fsfs Alien Recommendation Contracts v1

Issue: `bd-2hz.1.4`  
Parent: `bd-2hz.1`

## Goal

Define reusable recommendation-contract cards for top adaptive fsfs controllers:

- ingestion policy
- degradation scheduler
- ranking policy

Each card is machine-readable and directly consumable by implementation and test-planning workstreams.

## Required Card Fields

Every recommendation card MUST include:

- `ev_score`
- `priority_tier`
- `adoption_wedge`
- budgeted mode + fallback trigger
- baseline comparator
- isomorphism proof plan
- reproducibility artifact requirements
- rollback plan

## Card Catalog

## Ingestion Policy Card

Focus:
- high-cost include/skip/index-later tradeoffs
- utility-sensitive ingest under bounded compute

## Degradation Scheduler Card

Focus:
- pressure-state transitions
- safe fallback ladders and recovery gates

## Ranking Policy Card

Focus:
- quality/latency balancing under constrained resources
- stable tie-break and regression-safe rollout criteria

## Contract Semantics

- `ev_score` is numeric expected value (impact-confidence-reuse-effort normalization)
- `priority_tier` uses `A|B|C`
- `adoption_wedge` states where rollout starts first and why
- budgeted mode includes explicit defaults and exhaustion behavior
- fallback trigger includes `condition`, `fallback_action`, and `reason_code`
- baseline comparator names incumbent behavior being outperformed
- isomorphism proof plan defines invariants and replay checks
- reproducibility fields define required artifacts and replay command
- rollback plan defines deterministic rollback command and abort conditions

## Validation Artifacts

- `schemas/fsfs-alien-recommendations-v1.schema.json`
- `schemas/fixtures/fsfs-alien-recommendation-card-ingestion-v1.json`
- `schemas/fixtures/fsfs-alien-recommendation-bundle-v1.json`
- `schemas/fixtures-invalid/fsfs-alien-recommendation-invalid-*.json`
- `scripts/check_fsfs_alien_recommendations.sh`

## Validation Command

```bash
scripts/check_fsfs_alien_recommendations.sh --mode all
```
