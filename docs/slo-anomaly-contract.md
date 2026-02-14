# SLO + Error-Budget + Anomaly Contract v1

Issue: `bd-2yu.2.4`

## Scope

Defines the canonical, versioned contract for:

- SLO metrics
- error-budget computation across windows
- anomaly payload semantics
- alert reason codes + confidence semantics

Artifacts:

- Schema: `schemas/slo-anomaly-v1.schema.json`
- Fixtures: `schemas/fixtures/slo-anomaly-*.json`

## Canonical SLO Metrics

Required metric IDs:

1. `search_latency_p95`
2. `query_failure_rate`
3. `stale_index_lag`
4. `embedding_backlog_age`

Each metric defines:

- `objective_bad_ratio` (fraction in `[0,1]`)
- `objective_threshold` (numeric threshold in metric units)
- `unit`
- stable `reason_code_prefix`

## Required Windows

All calculations and payloads must support exactly:

- `1m`
- `15m`
- `1h`
- `6h`
- `24h`
- `3d`
- `1w`

## Error-Budget Formulas (Machine-Testable)

Inputs per metric/window:

- `bad_events_w`
- `total_events_w`
- `objective_bad_ratio`
- `budget_fraction_w`

Derived values:

```text
bad_ratio_w   = bad_events_w / max(total_events_w, 1)
consumed_w    = clamp01(bad_ratio_w / objective_bad_ratio)
remaining_w   = 1 - consumed_w
burn_rate_w   = consumed_w / budget_fraction_w
```

Contract requirements:

- formula version is explicit (`v1`)
- each window has `budget_fraction_w > 0`
- test vectors provide deterministic expected outputs per window

## Alert Signal Taxonomy

Severity levels:

- `info`
- `warn`
- `critical`

Reason code pattern:

- `slo.<metric>.<condition>`
- `anomaly.<metric>.<condition>`

Examples:

- `slo.search_latency_p95.budget_burn_high`
- `anomaly.query_failure_rate.spike`
- `anomaly.stale_index_lag.regression`

## Confidence Semantics

Every anomaly payload includes:

- `confidence.score` in `[0,1]`
- `confidence.band` in `{low, medium, high}`
- `confidence.evidence_points` (non-negative integer)

Interpretation:

- `low`: advisory only
- `medium`: operator attention recommended
- `high`: escalation eligible

## Anomaly Payload Contract

Required fields:

- metric identity (`metric_id`, `window`)
- reason metadata (`reason_code`, `severity`)
- baseline context:
  - `method`
  - `baseline_value`
  - `lookback_points`
- observed value + deviation:
  - `observed_value`
  - `deviation.absolute`
  - `deviation.relative_pct`
  - `deviation.z_score`
- suppression metadata:
  - `is_suppressed`
  - `policy_id` (nullable)
  - `until_ts` (nullable)
  - `suppress_reason_code` (nullable)
- confidence metadata

## Suppression Semantics

If `is_suppressed = true`:

- `policy_id` must be present.
- `suppress_reason_code` must be present.
- `until_ts` may be null only for indefinite suppression policies.

## Validation Strategy

## Schema validation

```bash
for f in schemas/fixtures/slo-anomaly-*.json; do
  jsonschema -i "$f" schemas/slo-anomaly-v1.schema.json
done
```

## Formula contract checks

Test vectors in the contract fixture are evaluated by downstream harnesses:

- recompute `bad_ratio/consumed/remaining/burn_rate`
- compare against expected values with tolerance
- fail on mismatch

## Consumer Guarantee

Dashboards, alerts, and harnesses can consume this contract without undocumented assumptions because:

- metric IDs are fixed
- windows are fixed
- formula version is explicit
- anomaly payload shape is fully declared
