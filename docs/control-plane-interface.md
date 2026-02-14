# Control Plane Snapshot + Stream Interface v1

Issue: `bd-2yu.2.2`

## Goal

Define a complete contract for control-plane clients to render dashboards and live feeds using only:

1. snapshot API responses
2. stream subscription frames
3. historical reads from frankensqlite

No hidden side channels are allowed.

## Interface Surfaces

## 1) Snapshot Query Interface

Semantic endpoint:

- `GET /v1/control/snapshot`

Request parameters:

- `project_filter[]` (optional)
- `include_windows[]` (`1m`, `15m`, `1h`, `6h`, `24h`, `3d`, `1w`)
- `include_unhealthy_only` (optional)

Response root fields:

- `snapshot_id` (ULID)
- `generated_ts` (RFC3339 UTC)
- `fleet_summary`:
  - `detected_instances`
  - `healthy_instances`
  - `degraded_instances`
  - `stale_instances`
- `instances[]` where each instance has:
  - attribution: `instance_id`, `project_key`, `host_name`, `attribution_confidence`
  - health: `lifecycle_state`, `slo_status`, `error_budget`
  - latest_metrics: search/embed/index/resource summary
  - anomaly_summary: active anomalies + severities
  - lag: latest ingest lag and stream lag counters

## 2) Streaming Interface

Semantic endpoint:

- `SUBSCRIBE /v1/control/stream`

Subscribe request:

- `client_id` (ULID)
- `topics[]` (`search`, `embedding`, `index`, `resource`, `anomaly`, `lifecycle`)
- `project_filter[]` (optional)
- `resume_cursor` (optional opaque string)
- `max_inflight` (required, flow-control budget)
- `heartbeat_ms` (required)

Frame families:

- `event`: telemetry payload with `cursor`, `topic`, `lag_ms`
- `control`: control-plane state change (backpressure, reconnect advisory, sampling mode)
- `heartbeat`: keepalive with current lag + queue depth
- `error`: terminal/non-terminal stream errors with retry guidance

## Dashboard Data Path Coverage

This mapping is required so clients can render every screen without undocumented assumptions:

| Screen | Snapshot fields required | Stream fields required |
|---|---|---|
| Fleet overview | `fleet_summary`, `instances[].health`, `instances[].attribution` | `control` (cluster state), `lifecycle` |
| Project dashboard | `instances[].latest_metrics`, `instances[].slo_status` | `search`, `embedding`, `index`, `resource`, `anomaly` |
| Live search stream | optional initial `instances[].latest_metrics.search` | `search` |
| Index + embedding progress | `instances[].latest_metrics.embed/index` | `embedding`, `index` |
| Resource trends | `instances[].latest_metrics.resource` | `resource` |
| Historical analytics | snapshot IDs + instance attribution for joins | stream cursors for boundary alignment |
| Alerts/timeline | `instances[].anomaly_summary` | `anomaly`, `lifecycle`, `control` |
| Explainability cockpit | correlation context from latest search metrics | `search` (correlation IDs), optional linked `embedding/index` |

## Backpressure, Lag, and Delivery Semantics

## Flow Control

- Client advertises `max_inflight`.
- Server must never exceed `max_inflight` unacked `event` frames per client.
- Acks are cursor-based (`ack_cursor`).

## Backpressure States

`control.backpressure_state` enum:

- `normal`
- `constrained` (queue growth detected)
- `dropping` (sampling/drop policy active)

When transitioning to `dropping`, server emits:

- `dropped_count_window`
- `sampling_ratio`
- `reason_code`

## Lag Reporting

Every stream frame carries:

- `producer_ts` (event production time)
- `dispatch_ts` (send time)
- `lag_ms = dispatch_ts - producer_ts`

Snapshot responses carry:

- `ingest_lag_ms_p50`
- `ingest_lag_ms_p95`
- `stream_queue_depth`

## Reconnect Semantics

- Cursor-based resume is first-class (`resume_cursor`).
- Server may emit `control.reconnect_advisory` with:
  - `retry_after_ms`
  - `resume_cursor_hint`
  - `reason_code`
- On reconnect, client resubscribes with previous `resume_cursor`.
- If cursor expired, server returns `error.code=resume_not_available` and provides a snapshot refresh requirement.

## Client Consumption Contract (No Hidden Assumptions)

Clients may assume only:

1. Schema version `v` guards compatibility.
2. Unknown optional fields can be ignored.
3. Unknown required fields for current `v` are contract violations.
4. Stream delivery is at-least-once; dedupe by `(cursor,event_id)`.
5. Ordering is guaranteed per topic per instance, not globally across all topics.

Clients must not assume:

- fixed event rates
- zero lag
- infinite retention of resume cursors
- anomaly events always preceding lifecycle transitions

## Validation Requirements

Required tests for downstream implementations:

- unit: schema decode + required-field enforcement
- integration: cursor resume, reconnect advisory handling, lag math verification
- integration: backpressure transitions (`normal` -> `constrained` -> `dropping`)
- e2e: render all required screens from snapshot + stream only
- e2e: degrade/recover host under bursty load and verify UI reflects lag/backpressure states

Artifacts:

- structured JSONL frames (snapshot fetch, stream frames, control transitions)
- replay bundle with seed/timing metadata
- scenario report containing dropped counts, lag percentiles, reconnect attempts
