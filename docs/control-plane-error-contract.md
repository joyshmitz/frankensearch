# Control Plane Error Taxonomy + UI Mapping Contract v1

Issue: `bd-2yu.2.5`

## Scope

Defines the ops-control-plane error system (separate from search-engine `SearchError`):

- typed `ControlPlaneError` taxonomy
- severity classification (`Fatal`, `Degraded`, `Transient`)
- deterministic error-to-UI mapping
- recovery guidance contract
- structured logging fields (evidence-ledger compatible)
- error aggregation semantics

Artifacts:

- Schema: `schemas/control-plane-error-v1.schema.json`
- Fixtures: `schemas/fixtures/control-plane-error-*.json`

## Typed Error Taxonomy

Canonical `ControlPlaneError` variants:

1. `DiscoveryFailed`
2. `StorageError`
3. `StreamDisconnected`
4. `SchemaMismatch`
5. `IngestionOverflow`
6. `AttributionFailed`
7. `TelemetryGap`

No raw string-only error paths are allowed in control-plane surfaces.

## Severity Classes

- `Fatal`: operator intervention or restart required.
- `Degraded`: partial functionality available; user-visible caution state.
- `Transient`: auto-recoverable with retry/backoff.

## Error-to-UI Mapping (Normative)

| Error Type | Default Severity | UI Surface | Status Badge | Escalation | Operator Guidance |
|---|---|---|---|---|---|
| `DiscoveryFailed` | `Degraded` | `full_screen_panel` when no instances; else `toast` | `discovery` | after 3 consecutive failures | verify discovery roots/permissions, retry discovery scan |
| `StorageError` | `Fatal` | `full_screen_panel` | `storage` | immediate | check DB integrity/disk health, switch read-only fallback |
| `StreamDisconnected` | `Transient` | `toast` | `stream` | if >N in 1m, escalate to `warn` alert | auto-reconnect, inspect transport path |
| `SchemaMismatch` | `Fatal` | `full_screen_panel` | `schema` | immediate | pin compatible schema version or migrate adapters |
| `IngestionOverflow` | `Degraded` | `status_badge` + `toast` | `ingest` | if dropping persists, escalate | reduce ingestion rate or increase queue budget |
| `AttributionFailed` | `Degraded` | `status_badge` | `attribution` | after 5m unresolved | verify project attribution hints/handshake |
| `TelemetryGap` | `Transient` | `status_badge` | `telemetry` | if gap > threshold, escalate | inspect emitter health and lag metrics |

## Recovery Guidance Contract

Each error payload must include:

- `retry_policy` (`none`, `immediate`, `exponential_backoff`)
- `operator_steps[]` (ordered, human-readable)
- `suggested_commands[]` (actionable command verbs for palette/CLI)

## Structured Logging Fields

Every control-plane error log event must include:

- `event_id` (ULID)
- `error_type` (typed enum)
- `severity_class`
- `reason_code` (stable machine key)
- `message` (operator-facing short text)
- `project_key`
- `instance_id` (nullable where not yet known)
- `correlation.root_request_id` (nullable)
- `correlation.parent_event_id` (nullable)
- `retry_count`
- `recoverable` (boolean)
- `ui_surface`

These fields are required for evidence-ledger joins and replay triage.

## Error Aggregation Semantics

Aggregation key:

`(error_type, project_key, instance_id, reason_code, window)`

For each window (minimum: `1m`, `15m`, `1h`), compute:

- `occurrences`
- `first_seen_ts`
- `last_seen_ts`
- `escalated` (boolean)
- `aggregation_reason_code` (e.g., `control.stream_disconnected.burst`)

Burst example:

- if `StreamDisconnected` >= 50 in `1m`, emit one aggregated alert event and suppress duplicate toasts.

## Validation Requirements

## Unit

- enum variant coverage (all 7 required variants)
- severity/UI mapping completeness
- reason-code format checks

## Integration

- injected fault scenarios map to typed variants
- UI surface and badge outputs match contract
- structured log fields are always present

## E2E

- repeated disconnect burst aggregation behavior
- storage/schema fatal errors render full-screen panel + guidance
- replay artifacts include typed error stream and correlation fields
