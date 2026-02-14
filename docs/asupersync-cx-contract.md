# asupersync + Cx Propagation Contract v1

Issue: `bd-3un.50`

## Goal

Define one authoritative contract for async/concurrency behavior in frankensearch:

- Cx propagation through async APIs
- sync-vs-async boundaries
- asupersync vs rayon usage boundaries
- timeout/cancellation/region/channel patterns
- LabRuntime testing and diagnostics requirements

Artifacts:

- Schema: `schemas/asupersync-cx-contract-v1.schema.json`
- Fixtures: `schemas/fixtures/asupersync-cx-*.json`
- Invalid fixtures: `schemas/fixtures-invalid/asupersync-cx-*.json`

## Rule Set (Normative)

## Rule 1: async APIs must take `&Cx` first

Allowed:

```rust
pub async fn search(&self, cx: &Cx, query: &str, limit: usize) -> Outcome<_, _>
```

Disallowed:

- async function missing `&Cx`
- async function with `&Cx` but not first parameter

## Rule 2: sync APIs must not take `&Cx`

Allowed:

```rust
pub fn canonicalize(&self, text: &str) -> String
```

Disallowed:

- synchronous APIs with `cx: &Cx` parameter

## Rule 3: execution-domain boundaries

- asupersync:
  - I/O-bound operations
  - structured concurrency (`region`, `scope.spawn`)
  - timeout/race/join orchestration
- rayon:
  - CPU-bound data parallelism only (SIMD dot products, batch scoring)
- synchronous:
  - pure deterministic transformations (parsing/canonicalization/math helpers)

## Standard Patterns

## Timeout-bounded operation

- use `asupersync::combinator::timeout`
- map cancellation to typed timeout error with reason code

## Structured worker pool

- spawn worker tasks in one `region`
- region exit guarantees worker cleanup (no orphan tasks)

## Two-phase channel send

- `reserve()` then `send()` for cancel-correct delivery

## Diagnostics Contract

Every async pattern emits structured fields:

- `pattern` (`timeout`, `worker_pool`, `two_phase_channel`)
- `operation`
- `reason_code`
- `elapsed_ms`
- `outcome` (`ok`, `err`, `cancelled`, `panicked`)

## Test Contract

Required lanes:

1. Unit:
   - async signatures include `&Cx` first
   - sync signatures exclude `&Cx`
2. Integration:
   - LabRuntime deterministic replay for same seed
   - timeout path emits `Outcome::Cancelled` handling
   - region cleanup leaves no leaked tasks
3. E2E:
   - representative concurrent flow with diagnostic artifact bundle

LabRuntime requirements:

- fixed seed capture
- quiescence oracle check
- obligation leak oracle check
- task leak oracle check (if available in harness)

## Validation Strategy

## Positive fixtures

```bash
for f in schemas/fixtures/asupersync-cx-*.json; do
  jsonschema -i "$f" schemas/asupersync-cx-contract-v1.schema.json
done
```

## Negative fixtures (must fail)

```bash
for f in schemas/fixtures-invalid/asupersync-cx-*.json; do
  if jsonschema -i "$f" schemas/asupersync-cx-contract-v1.schema.json; then
    echo "unexpected pass: $f" && exit 1
  fi
done
```

## Primary Consumers

- `bd-3un.24` (`TwoTierSearcher`)
- `bd-3un.27` (embedding job queue/backpressure workers)
