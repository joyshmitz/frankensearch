# fsfs Provenance Attestation and Startup Verification Contract v1

Issue: `bd-2hz.8.3`  
Parent: `bd-2hz.8`

## Goal

Define how fsfs records build/runtime provenance and how startup verification
converts mismatches into deterministic fallback behavior with auditable alerts.

## Normative Terms

- `MUST`: hard requirement
- `SHOULD`: default expectation unless justified otherwise
- `MUST NOT`: forbidden behavior

## Attestation Envelope

Every `provenance-attestation.json` payload MUST include:

- identity:
  - `schema_version` (`1`)
  - `attestation_id`
  - `generated_at` (RFC3339 UTC)
- build provenance:
  - `build.source_commit`
  - `build.build_profile`
  - `build.rustc_version`
  - `build.target_triple`
- runtime provenance:
  - `runtime.binary_hash_sha256`
  - `runtime.config_hash_sha256`
  - `runtime.index_manifest_hash_sha256`
- artifact inventory:
  - `artifact_hashes[]` entries with `path` and `sha256`
- optional signature:
  - `signature.algorithm`
  - `signature.key_id`
  - `signature.signature_b64`

Hash fields MUST use `sha256:<64 lowercase hex>`.

## Startup Verification Flow

Startup verification MUST follow this order:

1. Load and parse attestation payload.
2. Enforce attestation/signature requirements from policy.
3. Compare runtime hashes against live binary/config/index inputs.
4. Derive one fallback action:
   - `continue`
   - `continue_with_alert`
   - `enter_read_only`
   - `enter_safe_mode`
   - `abort_startup`
5. Emit structured alert(s) with stable reason code(s).

## Required Reason Codes

- `provenance.startup.attestation_missing`
- `provenance.startup.signature_missing`
- `provenance.startup.signature_invalid`
- `provenance.startup.hash_mismatch`

## Mismatch Matrix

| Condition | Default Action | Required Alert |
|---|---|---|
| Attestation required but absent | `enter_safe_mode` | `provenance.startup.attestation_missing` |
| Signature required but absent | `enter_safe_mode` | `provenance.startup.signature_missing` |
| Signature present but invalid | `abort_startup` | `provenance.startup.signature_invalid` |
| Binary/config/index hash mismatch | `abort_startup` | `provenance.startup.hash_mismatch` |

If multiple mismatches occur, startup MUST select the strictest action.

## Fallback Semantics

- `continue`: normal startup, no mismatch alert.
- `continue_with_alert`: startup proceeds; operator alert required.
- `enter_read_only`: startup proceeds in non-mutating mode.
- `enter_safe_mode`: startup proceeds with constrained ingest/search behavior.
- `abort_startup`: process MUST refuse normal startup path.

## Debug and Replay Compatibility

Startup verification outputs MUST be consumable by both CLI and TUI debug flows
without adaptation. Minimum fields:

- `trace_id`
- `attestation_id`
- `status`
- `action`
- `checks.*`
- `alerts[]`

## Validation Artifacts

- `schemas/fsfs-provenance-attestation-v1.schema.json`
- `schemas/fixtures/fsfs-provenance-attestation-contract-v1.json`
- `schemas/fixtures/fsfs-provenance-attestation-manifest-v1.json`
- `schemas/fixtures/fsfs-provenance-attestation-startup-check-v1.json`
- `schemas/fixtures-invalid/fsfs-provenance-attestation-invalid-*.json`
- `scripts/check_fsfs_provenance_attestation_contract.sh`

## Validation Command

```bash
scripts/check_fsfs_provenance_attestation_contract.sh --mode all
```

