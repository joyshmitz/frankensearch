# Evidence JSONL Contract + Redaction Policy v1

Issue: `bd-2yu.2.3`

## Purpose

Define a safe-by-default evidence logging contract that preserves postmortem and explainability value while enforcing deterministic replay metadata and privacy boundaries.

Artifacts:

- Schema: `schemas/evidence-jsonl-v1.schema.json`
- Valid fixtures: `schemas/fixtures/evidence-*.json`
- Negative fixtures: `schemas/fixtures-invalid/evidence-*.json`

## JSONL Line Shape

Each JSONL line is a standalone envelope:

```json
{
  "v": 1,
  "ts": "2026-02-14T00:00:00Z",
  "event": {
    "...": "evidence payload"
  }
}
```

## Required Evidence Payload Fields

- identity:
  - `event_id` (ULID)
  - `project_key`
  - `instance_id` (ULID)
- trace:
  - `root_request_id` (ULID)
  - `parent_event_id` (nullable ULID)
- event classification:
  - `type` (`decision`, `alert`, `degradation`, `transition`, `replay_marker`)
  - `reason.code` (machine-stable code)
  - `reason.human` (operator-facing explanation)
  - `reason.severity` (`info`, `warn`, `error`)
- replay metadata:
  - `replay.mode` (`live`, `deterministic`)
  - if `deterministic`, all required:
    - `seed`
    - `tick_ms`
    - `frame_seq`
- redaction metadata:
  - `redaction.policy_version`
  - `redaction.transforms_applied[]`
  - `redaction.contains_sensitive_source` (boolean)
- payload summary:
  - optional sanitized fields only (strictly enumerated by schema)

## Deterministic Replay Requirements

For replay-capable incidents, logs must be sufficient to reconstruct ordering and timing:

- stable `frame_seq` ordering
- fixed `tick_ms`
- explicit `seed`

Deterministic logs missing these fields are invalid by schema.

## Reason-Code Semantics

Reason code format:

- pattern: `namespace.subject.detail`
- examples:
  - `search.phase.refinement_failed`
  - `control.backpressure.dropping`
  - `slo.latency.p95_exceeded`

Rules:

- code must be machine-stable and used for aggregation.
- `human` text can evolve but should remain concise.

## Sensitive Data Classification + Redaction Rules

| Class | Examples | Default Transform | Allowed Output |
|---|---|---|---|
| `query_content` | user query text | `hash_sha256`, `truncate_preview` | `query_hash`, `query_preview` (max 120 chars) |
| `filesystem_path` | absolute paths | `path_tokenize` | project-relative tokenized path only |
| `identifier` | user IDs, email-like IDs | `hash_sha256` | one-way hash |
| `document_text` | full content snippets | `drop` or `truncate_preview` | no raw full text |

Policy constraints:

- Redaction is mandatory and on by default.
- Raw sensitive fields are forbidden in evidence payload.
- Evidence records must declare transforms actually applied.

## Compatibility Policy

- Breaking changes bump envelope `v`.
- Additive optional fields at same version are allowed only if explicitly added to schema.
- Unknown extra fields are rejected in strict validation mode.

## Validation Strategy (E2E/CI)

## Positive validation

```bash
for f in schemas/fixtures/evidence-*.json; do
  jsonschema -i "$f" schemas/evidence-jsonl-v1.schema.json
done
```

## Negative validation (must fail)

```bash
for f in schemas/fixtures-invalid/evidence-*.json; do
  if jsonschema -i "$f" schemas/evidence-jsonl-v1.schema.json; then
    echo "unexpected pass: $f" && exit 1
  fi
done
```

Contract gate:

- CI fails if any positive fixture fails or any negative fixture passes.
