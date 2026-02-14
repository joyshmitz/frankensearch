# fsfs Pressure Profile and Override Contract v1

Issue: `bd-2hz.4.5`  
Parent: `bd-2hz.4`

## Goal

Define strict/performance/degraded operating profiles with deterministic override precedence and migration-safe evolution rules.

This contract is normative for:

- profile defaults and capability boundaries
- profile-vs-override resolution order
- conflict handling and reason-coded diagnostics
- profile version evolution without silent drift

## Normative Terms

- `MUST`: hard requirement
- `SHOULD`: expected default unless explicitly justified
- `MUST NOT`: forbidden behavior

## Profile Definitions (Required)

`fsfs` MUST provide exactly these profile IDs:

- `strict`
- `performance`
- `degraded`

Each profile MUST define:

- scheduler mode
- embed/index concurrency limits
- quality-tier enablement boundary
- background indexing allowance
- pressure enter/exit thresholds
- explicit override policy (overridable vs locked fields)

## Capability Boundaries

## strict

- safety-first
- conservative concurrency limits
- aggressive fallback entry under pressure
- locked fields for unsafe cost expansion

## performance

- balanced throughput/latency
- moderate concurrency and quality access
- controlled fallback behavior

## degraded

- correctness-preserving minimal mode
- reduced features and bounded resource spend
- explicit recovery requirements before escalation back to richer profiles

## Deterministic Override Precedence

Resolution MUST apply in this exact order:

1. hard safety guards
2. CLI overrides
3. environment overrides
4. config-file overrides
5. selected profile defaults

If an override targets a locked field, the override MUST be rejected with deterministic reason code and no partial application.

## Conflict and Safety Semantics

- conflict detection MUST be explicit and machine-readable
- all conflicts MUST emit stable reason codes
- safety clamps MUST be visible in diagnostics
- effective profile state MUST be reconstructable from emitted fields

## Required Diagnostics Fields

Profile-resolution diagnostics MUST include:

- `event`
- `trace_id`
- `selected_profile`
- `precedence_chain`
- `overrides[]` (source, field, applied/rejected, reason)
- `safety_clamps[]`
- `conflict_detected`
- `reason_code`
- `effective_profile_version`

## Migration-Safe Evolution Strategy

1. Profile schema version is explicit and required.
2. Semantic profile changes MUST increment profile revision.
3. New profile fields MUST provide deterministic defaults for all profiles.
4. Removing or repurposing profile fields MUST NOT happen without explicit migration metadata.
5. Existing profile IDs (`strict|performance|degraded`) MUST remain stable across minor revisions.

## Validation Artifacts

- `schemas/fsfs-pressure-profiles-v1.schema.json`
- `schemas/fixtures/fsfs-pressure-profiles-contract-v1.json`
- `schemas/fixtures/fsfs-pressure-profiles-decision-v1.json`
- `schemas/fixtures-invalid/fsfs-pressure-profiles-invalid-*.json`
- `scripts/check_fsfs_pressure_profiles_contract.sh`

## Validation Command

```bash
scripts/check_fsfs_pressure_profiles_contract.sh --mode all
```
