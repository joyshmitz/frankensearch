# fsfs Expected-Loss Decision Contract v1

Issue: `bd-2hz.1.2`  
Parent: `bd-2hz.1`

## Goal

Make high-impact runtime decisions auditable by encoding explicit action/state/loss tradeoffs for:

- ingest decisions
- embedding decisions
- degradation/recovery decisions

This contract is normative for policy engines, tests, and decision telemetry.

## Normative Terms

- `MUST`: hard requirement
- `SHOULD`: expected default
- `MUST NOT`: forbidden behavior

## Decision Families

## Ingest Family

Action space:
- `index_now`
- `index_later`
- `skip`

Primary asymmetry:
- false include cost (wasted compute + noise) vs false exclude cost (recall loss)

## Embed Family

Action space:
- `embed_now`
- `embed_defer`
- `embed_disable`

Primary asymmetry:
- latency/compute spend vs semantic quality loss

## Degrade Family

Action space:
- `degrade_enter`
- `degrade_hold`
- `degrade_exit`

Primary asymmetry:
- availability/latency protection vs ranking quality and freshness

## Required Machine-Readable Fields

Every decision matrix artifact MUST encode:

- `family`
- `state_id`
- `action`
- `expected_loss`
- cost asymmetry fields:
  - `false_include_cost`
  - `false_exclude_cost`
  - `latency_cost`
  - `quality_cost`
  - `compute_cost`
- `risk_level`
- `reason_code`

## Fallback Trigger Contract

High-risk actions MUST define fallback triggers with:

- `condition` (machine-readable threshold expression)
- `fallback_action`
- `reason_code`
- `trip_threshold`
- `applies_to_actions[]`

If a decision event invokes fallback, the event MUST include fallback diagnostics.

## Auditing and Diagnostics Requirements

Decision events MUST include:

- `decision_id`
- `seed`
- `config_hash`
- `family`
- `state_id`
- `chosen_action`
- evaluated alternatives with expected losses
- `selected_reason_code`
- fallback invocation details (when applicable)

## Validation Artifacts

- `schemas/fsfs-expected-loss-v1.schema.json`
- `schemas/fixtures/fsfs-expected-loss-contract-v1.json`
- `schemas/fixtures/fsfs-expected-loss-matrix-v1.json`
- `schemas/fixtures/fsfs-expected-loss-decision-event-v1.json`
- `schemas/fixtures-invalid/fsfs-expected-loss-invalid-*.json`
- `scripts/check_fsfs_expected_loss_contract.sh`

## Validation Commands

```bash
scripts/check_fsfs_expected_loss_contract.sh --mode all
```
