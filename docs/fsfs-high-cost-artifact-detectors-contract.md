# fsfs High-Cost Artifact Detectors Contract v1

Issue: `bd-2hz.2.3`  
Parent: `bd-2hz.2`

## Goal

Define deterministic heuristics for detecting low-value, high-cost artifacts so `fsfs` can skip or downgrade expensive indexing work by default while allowing explicit user override.

Required detector families:

- giant logs (size/churn/redundancy)
- vendored/generated/library trees
- compressed/archive artifacts
- transient build/runtime artifacts

## Normative Terms

- `MUST`: hard requirement
- `SHOULD`: expected default
- `MUST NOT`: forbidden behavior

## Detector Families

### Giant Log Detector

`fsfs` MUST classify giant logs using combined evidence:

- size threshold (`max_size_mb`)
- churn rate window (`churn_window_minutes`)
- redundancy ratio (`redundancy_ratio_threshold`)

Default action SHOULD be `index_metadata_only` or `skip`.

### Vendor / Generated / Library Detector

`fsfs` MUST identify expensive low-signal trees using path and content markers:

- vendored path patterns (`vendor/`, `node_modules/`, third-party mirrors)
- generated file markers (`@generated`, `DO NOT EDIT`, codegen headers)
- library-depth/path heuristics for dependency trees

Default action SHOULD avoid full embedding.

### Archive / Transient Artifact Detector

`fsfs` MUST detect compressed containers and transient build products:

- archive extensions (`.zip`, `.tar`, `.gz`, `.xz`, `.7z`)
- transient directories (`target/`, `dist/`, `.cache/`, `.tmp/`)
- build artifact patterns (`*.min.js`, `*.pyc`, `*.o`, `*.class`)

Default action MUST be non-full-index unless explicitly overridden.

## Override Hooks

The policy MUST support user-forced inclusion with guardrails:

- explicit override reason string
- optional expiry time
- audit event with reason code and final action

`index_full` for a high-cost artifact MUST require an explicit override.

## Required Decision Fields

Every detector decision MUST include:

- `detectors_fired[]`
- `final_action`
- `cost_score` in `[0,1]`
- `reason_code` (`FSFS_*`)
- override metadata (`override_applied`, optional `user_override`)

## Validation Artifacts

- `schemas/fsfs-high-cost-artifact-detectors-v1.schema.json`
- `schemas/fixtures/fsfs-high-cost-artifact-detectors-contract-v1.json`
- `schemas/fixtures/fsfs-high-cost-artifact-detectors-decision-v1.json`
- `schemas/fixtures/fsfs-high-cost-artifact-detectors-override-v1.json`
- `schemas/fixtures-invalid/fsfs-high-cost-artifact-detectors-invalid-*.json`
- `scripts/check_fsfs_high_cost_artifact_detectors_contract.sh`

## Validation Command

```bash
scripts/check_fsfs_high_cost_artifact_detectors_contract.sh --mode all
```
