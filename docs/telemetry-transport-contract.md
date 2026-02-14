# Telemetry Transport Mechanism Contract v1

Issue: `bd-2yu.3.4`

## Goal

Define how telemetry moves from running frankensearch instances to the ops control plane.

This contract is transport-level and complements:

- `docs/control-plane-interface.md` (what data consumers expect)
- `docs/telemetry-event-taxonomy.md` (event payload families)

## Transport Topology (Normative)

## 1) Primary Transport: Unix Domain Socket

- Transport type: `unix_domain_socket`
- Socket path template: `${XDG_RUNTIME_DIR}/frankensearch/{instance_id}.sock`
- Framing: `length_prefixed`
- Payload codecs: `msgpack` and `cbor`
- Authentication: local UID match (`uid_match`)

Rationale:

- machine-local by design
- no network exposure
- low framing/parse overhead for high event rates

## 2) Fallback Transport: JSONL File

- Transport type: `jsonl_file`
- Path template: `{data_dir}/telemetry/{instance_id}.jsonl`
- Record format: one JSON object per line (schema-compatible envelope)

Fallback is mandatory for environments where socket lifecycle is constrained or unavailable.

## Lifecycle + Subscription Contract

Connection sequence:

1. connect to UDS endpoint
2. authenticate (`uid_match`)
3. send subscribe frame (`topic_filter`, `max_inflight`, `heartbeat_ms`, optional `resume_cursor`)
4. receive stream frames (`event|control|heartbeat|error`)
5. acknowledge progress via cursor semantics
6. disconnect gracefully or retry with backoff + resume cursor

Required lifecycle invariants:

- handshake is required before event delivery
- heartbeat is required to detect dead peers
- resume cursor is first-class for reconnect
- disconnect behavior must support `graceful_or_retry`

## Backpressure + Drop Semantics

Required policy:

- `drop_not_block`
- producer/search pipeline must not block on control-plane lag
- dropped events must be counted and surfaced in stream metadata/control frames

Required fields:

- `max_inflight`
- `dropped_since_last`
- control frame with reason code when entering constrained/drop states

## Multi-Consumer Contract

Required mode:

- `fan_out`

Implications:

- multiple control-plane consumers may subscribe to one producer instance
- one slow consumer must not stall producer emission for others

## Security Scope

Required constraints:

- `local_only = true`
- `network_transport_allowed = false` (v1)
- identity/auth bound to local OS user context

## Performance/SLO Targets

Contract targets:

- telemetry delivery lag p95 target: `<= 100ms`
- throughput target: `>= 10_000 events/sec`
- fallback JSONL must preserve delivery without search-path blocking

## Integration Mapping

- consumed by `bd-2yu.3.1` discovery (socket presence/lifecycle signals)
- consumed by `bd-2yu.5.1` collector implementation
- consumed by `bd-2yu.4.2` historical ingestion pipeline

## Validation Matrix

Required tests:

1. unit:
   - subscribe frame decode/validation
   - stream frame discriminator + required payload fields
   - drop-policy invariants (`drop_not_block`)
2. integration:
   - UDS connect/handshake/stream
   - reconnect with resume cursor
   - fallback JSONL ingestion path
   - fault injection (disconnect, partial write, delayed consumer)
3. e2e:
   - multi-instance fan-out scenario with <100ms p95 lag
   - 10k events/sec burst with no producer blocking

Artifacts:

- `schemas/telemetry-transport-v1.schema.json`
- `schemas/fixtures/telemetry-transport-*.json`
- `schemas/fixtures-invalid/telemetry-transport-*.json`

## Validation Commands

```bash
for f in schemas/fixtures/telemetry-transport-*.json; do
  jsonschema -i "$f" schemas/telemetry-transport-v1.schema.json
done

for f in schemas/fixtures-invalid/telemetry-transport-*.json; do
  if jsonschema -i "$f" schemas/telemetry-transport-v1.schema.json; then
    echo "unexpected pass: $f" && exit 1
  fi
done
```
