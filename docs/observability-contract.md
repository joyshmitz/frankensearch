# Observability Contract: Evidence-Ledger Schema Checklist and Lint

Issue: `bd-tn1o`

## Purpose

Define the canonical evidence-ledger schema requirements and lint rules that every ranking/control/adaptive component must satisfy. This contract prevents semantic drift across beads and enables automated compliance checking at CI time.

## Scope

Components subject to this contract:

| Component | Bead(s) | Reason Codes |
|---|---|---|
| Decision Plane | bd-3un.24 | `decision.*` |
| Circuit Breaker | bd-1do | `circuit.*` |
| Calibration | bd-22k | `calibration.*` |
| Adaptive Fusion | bd-21g | `fusion.*` |
| Conformal Prediction | bd-2yj | `conformal.*` |
| Relevance Feedback | bd-2tv | `feedback.*` |
| Sequential Testing | bd-2ps | `testing.*` |
| PRF Expansion | bd-3st | `expansion.*` |
| E2E Testing | bd-2hz.10.11 | `e2e.*` |

## Evidence Record Requirements

Every adaptive component MUST emit `EvidenceRecord` instances for:

1. **State transitions**: entering/leaving any non-nominal state.
2. **Fallback triggers**: whenever a component falls back to a default/safe behavior.
3. **Parameter adjustments**: when Bayesian updates, Thompson sampling, or similar modify parameters.
4. **Degradation events**: when quality, latency, or resource constraints change behavior.

### Required Fields Per Record

| Field | Required | Constraint |
|---|---|---|
| `event_type` | Yes | One of: `decision`, `alert`, `degradation`, `transition`, `replay_marker` |
| `reason_code` | Yes | Must match `^[a-z0-9]+\.[a-z0-9_]+\.[a-z0-9_]+$` |
| `reason_human` | Yes | Non-empty, max 200 chars |
| `severity` | Yes | One of: `info`, `warn`, `error` |
| `pipeline_state` | Yes | One of: `nominal`, `degraded_quality`, `circuit_open`, `probing` |
| `source_component` | Yes | Component identifier matching bead scope |
| `action` | If decision | Pipeline action taken |
| `expected_loss` | If decision | Three-dimensional loss vector |
| `query_class` | If query-dependent | Query classification that influenced decision |

### Reason Code Registry

All reason codes MUST be declared as constants in `ReasonCode` (decision_plane.rs). Ad-hoc string codes are forbidden.

Naming convention:
- `{namespace}.{subject}.{detail}`
- Namespace = component family (e.g. `decision`, `circuit`, `calibration`)
- Subject = action category (e.g. `skip`, `open`, `fallback`)
- Detail = specific trigger (e.g. `fast_only`, `consecutive_failures`)

### Trace Linkage

Evidence records emitted in the context of a search query MUST include trace context:
- `trace_id` (ULID): root request correlation
- `event_id` (ULID): unique per event
- `parent_event_id` (nullable ULID): causal predecessor

### Replay Linkage

For deterministic replay support, evidence records MUST include:
- `seed`: randomness seed
- `config_snapshot` or `config_hash`: configuration state at emission time

## Lint Rules

The following lint rules are enforced by `ObservabilityLint`:

| Rule ID | Severity | Description |
|---|---|---|
| `OBS-001` | Error | Evidence record missing required field |
| `OBS-002` | Error | Reason code does not match `namespace.subject.detail` pattern |
| `OBS-003` | Error | Reason code not declared in ReasonCode registry |
| `OBS-004` | Warning | Component emits no evidence records (missing instrumentation) |
| `OBS-005` | Warning | Decision event without expected_loss field |
| `OBS-006` | Warning | State transition without preceding/following evidence |
| `OBS-007` | Info | Reason code human text exceeds 200 characters |
| `OBS-008` | Error | Evidence record severity contradicts event type |
| `OBS-009` | Error | Duplicate event_id in evidence stream |
| `OBS-010` | Warning | Evidence stream not ordered by timestamp |

### Severity Consistency Rules (OBS-008)

| Event Type | Minimum Severity |
|---|---|
| `degradation` | `warn` |
| `alert` | `warn` |
| `decision` | `info` |
| `transition` | `info` |
| `replay_marker` | `info` |

## Checklist for New Components

When implementing a new adaptive component:

- [ ] Declare all reason codes as `ReasonCode` constants
- [ ] Emit `EvidenceRecord` for every state transition
- [ ] Emit `EvidenceRecord` for every fallback trigger
- [ ] Include `source_component` matching bead ID
- [ ] Include `expected_loss` on all decision events
- [ ] Include `query_class` on query-dependent events
- [ ] Add reason codes to this contract table
- [ ] Add component to scope table above
- [ ] Add lint rule coverage in test suite

## Validation

Schema: `schemas/e2e-artifact-v1.schema.json` (events stream)
Lint: `frankensearch_core::observability_lint` module

## Integration with Release Gate

The release gate (bd-ehuk) requires:
- All lint rules pass at `Error` severity
- No `OBS-004` warnings (all components instrumented)
- Evidence stream validates against JSONL schema
