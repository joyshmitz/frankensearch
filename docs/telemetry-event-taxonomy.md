# Telemetry Event Taxonomy + Payload Schema v1

Issue: `bd-2yu.2.1`

## Scope

This contract defines the canonical telemetry event envelope and event-family taxonomy for the frankensearch control plane.

It covers:

- instance identity and host project attribution
- search request/result/latency/memory telemetry
- embedding queue and job progress telemetry
- index inventory snapshot telemetry
- resource footprint telemetry (CPU/RSS/IO)
- lifecycle/health telemetry
- versioning, correlation IDs, nullability, and compatibility policy

Schema artifact: `schemas/telemetry-event-v1.schema.json`  
Representative fixtures: `schemas/fixtures/telemetry-*.json`

Cross-epic compatibility and rollout policy:
`docs/cross-epic-telemetry-adapter-lockstep-contract.md`

## Canonical Envelope

Every event uses the same envelope:

```json
{
  "v": 1,
  "ts": "2026-02-14T00:00:00Z",
  "event": {
    "type": "search",
    "...": "variant payload"
  }
}
```

Rules:

- `v` is required and monotonic.
- `ts` is UTC RFC3339 timestamp.
- `event.type` is the discriminator for payload family.

## Event Families

## 1) `search`

Purpose: query execution and ranking phase telemetry.

Required fields:

- identity: `instance.instance_id`, `instance.project_key`, `instance.host_name`
- correlation: `correlation.event_id`, `correlation.root_request_id`
- query: canonicalized/truncated text, query class, phase
- results: result count + source counts
- metrics: latency + memory

## 2) `embedding`

Purpose: embedding job lifecycle and queue pressure.

Required fields:

- job identity (`job.job_id`)
- queue depth and document count
- embedder identity (`id`, `tier`, `dimension`)
- stage and status
- duration

## 3) `index`

Purpose: index operation progress and corpus inventory snapshots.

Required fields:

- operation type (`build`, `rebuild`, `append`, `compact`, `repair`, `snapshot`)
- inventory snapshot (`words`, `tokens`, `lines`, `bytes`, `docs`)
- dimension + quantization
- duration + status

## 4) `resource`

Purpose: host pressure and runtime footprint.

Required fields:

- cpu percent
- rss bytes
- IO read/write bytes
- sample interval

## 5) `lifecycle`

Purpose: instance health and state transitions.

Required fields:

- state (`started`, `stopped`, `healthy`, `degraded`, `stale`, `recovering`)
- severity

## Correlation ID Contract

ULID format is required for IDs:

- `correlation.event_id`: unique ID of this telemetry event
- `correlation.root_request_id`: root search/request correlation chain
- `correlation.parent_event_id`: optional direct parent for sub-events

Topology:

`root_request_id` -> search event -> embedding/index/resource/lifecycle events linked through `parent_event_id`.

## Versioning + Compatibility Policy

- Breaking schema changes: increment `v` (e.g., `1` -> `2`).
- Additive optional fields: stay on same version.
- Field removals or required-field changes: breaking, must bump version.
- Unknown `v`: consumers store raw payload and do not reject ingestion.
- Unknown additional fields at same `v`: reject in strict contract validation (this schema), but archival ingest may keep raw copy for forward migration workflows.

## Nullability Policy

Required non-null fields:

- envelope: `v`, `ts`, `event`
- correlation: `event_id`, `root_request_id`
- identity: `instance_id`, `project_key`, `host_name`

Nullable fields are explicitly typed as `["<type>", "null"]` where platform or context availability varies:

- `search.metrics.memory_bytes`
- `resource.sample.load_avg_1m`
- `resource.sample.pressure_profile`
- `lifecycle.uptime_ms`
- `correlation.parent_event_id`

Omitting optional nullable fields is allowed; if emitted, they must conform to the declared nullable type.

## Validation Fixtures

Representative v1 fixtures exist for each family:

- `schemas/fixtures/telemetry-search-v1.json`
- `schemas/fixtures/telemetry-embedding-v1.json`
- `schemas/fixtures/telemetry-index-v1.json`
- `schemas/fixtures/telemetry-resource-v1.json`
- `schemas/fixtures/telemetry-lifecycle-v1.json`

These fixtures are the baseline contract tests for downstream schema validation in collectors/storage/adapters.

## Adapter Instrumentation + Conformance

Host integrations should emit this taxonomy through the adapter SDK in
`crates/frankensearch-core/src/host_adapter.rs`:

- `HostAdapter` defines the required identity handshake, telemetry emission, and lifecycle hooks.
- `ForwardingHostAdapter` provides canonical identity defaults for
  `coding_agent_session_search`, `xf`, `mcp_agent_mail_rust`, and `frankenterm`.
- `AdapterIdentity` must report:
  `adapter_id`, `adapter_version`, `host_project`,
  `telemetry_schema_version`, and `redaction_policy_version`.

Conformance checks are executed with `ConformanceHarness`:

- identity checks: required fields + schema version + redaction policy alignment + canonical `host_project`/`adapter_id` pairing for known hosts
- envelope checks: schema version, timestamp, per-event required fields
- correlation checks: ULID validity for `event_id`/`root_request_id`/optional `parent_event_id`
- redaction checks: forbidden secret patterns must not appear in serialized payloads
- lifecycle checks: adapters must handle start/tick/stop hooks deterministically

Recommended validation commands:

```bash
# Adapter harness and fixture conformance
cargo test -p frankensearch-core host_adapter::tests -- --nocapture

# Cross-epic lockstep contract (schema/version drift diagnostics)
cargo test -p frankensearch-core contract_sanity::tests -- --nocapture
```

For rollout policy and compatibility window behavior, see:
`docs/cross-epic-telemetry-adapter-lockstep-contract.md`.
