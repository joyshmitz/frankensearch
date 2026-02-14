# Unified E2E Artifact Schema and File Naming Contract v1

Issue: `bd-2hz.10.11.1`
Parent: `bd-2hz.10.11`

## Purpose

Define one canonical artifact schema shared by all end-to-end test suites across core, fsfs, and ops. Each e2e run produces a self-contained artifact pack that enables postmortem analysis, deterministic replay, regression comparison, and CI gating.

## Artifact Types

Every e2e run produces a subset of these artifact types:

| Type | Description | Required |
|---|---|---|
| `manifest` | Run metadata, config, seeds, model versions, checksums | Yes |
| `events` | Ordered JSONL stream of evidence/telemetry events | Yes |
| `oracle_report` | Aggregated oracle verdicts per lane | If interaction tests |
| `replay` | Input recording for deterministic re-execution | If deterministic mode |
| `snapshot_diff` | Before/after index or state comparison | If regression check |
| `transcript` | Human-readable execution summary | Optional |

## File Naming Convention

All artifact files live in a single directory per run:

```
<output_dir>/<run_id>/
  manifest.json
  events.jsonl
  oracle-report.json
  replay.jsonl
  snapshot-diff.json
  transcript.txt
```

Rules:

- `run_id` is a ULID (26-char Crockford base32).
- Filenames are fixed strings (no timestamps, no sequence numbers in names).
- File extensions: `.json` for single objects, `.jsonl` for line-delimited, `.txt` for plain text.
- No nested subdirectories within a run directory.
- Absent optional artifacts simply omit the file.

## Envelope Structure

All JSON/JSONL artifacts share a common envelope:

```json
{
  "v": 1,
  "schema": "e2e-manifest-v1",
  "run_id": "<ULID>",
  "ts": "2026-02-14T00:00:00Z",
  "body": { ... }
}
```

Required envelope fields:

| Field | Type | Description |
|---|---|---|
| `v` | integer | Schema version (currently `1`) |
| `schema` | string | Artifact type identifier |
| `run_id` | ULID | Unique run identifier, shared across all artifacts in the pack |
| `ts` | RFC 3339 | Timestamp of artifact creation |
| `body` | object | Type-specific payload |

JSONL files (`events.jsonl`, `replay.jsonl`) use one envelope per line.

## Manifest Schema (`manifest.json`)

The manifest is the entry point for any artifact pack. Required body fields:

```json
{
  "body": {
    "suite": "core|fsfs|ops|interaction",
    "determinism_tier": "bit_exact|semantic|statistical",
    "seed": 42,
    "config_hash": "sha256:abcdef...",
    "index_version": "fsvi-v3",
    "model_versions": [
      { "name": "potion-128M", "revision": "abc123", "digest": "sha256:..." }
    ],
    "platform": {
      "os": "linux",
      "arch": "x86_64",
      "rustc": "nightly-2026-02-01"
    },
    "clock_mode": "simulated|frozen|realtime",
    "tie_break_policy": "doc_id_lexical",
    "artifacts": [
      { "file": "events.jsonl", "checksum": "sha256:...", "line_count": 147 },
      { "file": "oracle-report.json", "checksum": "sha256:..." }
    ],
    "duration_ms": 1234,
    "exit_status": "pass|fail|error"
  }
}
```

Required manifest body fields:

| Field | Required | Description |
|---|---|---|
| `suite` | Yes | Which test suite produced this pack |
| `determinism_tier` | Yes | Reproducibility guarantee level |
| `seed` | Yes | Master randomness seed |
| `config_hash` | Yes | SHA-256 of the effective test configuration |
| `model_versions` | Yes | List of model name/revision/digest tuples |
| `platform` | Yes | OS, architecture, and compiler version |
| `clock_mode` | Yes | Clock strategy used |
| `tie_break_policy` | Yes | Deterministic tie-break strategy |
| `artifacts` | Yes | List of files in this pack with checksums |
| `duration_ms` | Yes | Total wall-clock time |
| `exit_status` | Yes | Overall pass/fail/error |
| `index_version` | No | Index format version string |

## Events Schema (`events.jsonl`)

Each line in `events.jsonl` is an envelope wrapping either:
- An evidence event (from `evidence-jsonl-v1` schema)
- A telemetry event (from `telemetry-event-v1` schema)
- An e2e-specific lifecycle event

E2e-specific event body fields:

```json
{
  "body": {
    "type": "e2e_start|e2e_end|lane_start|lane_end|oracle_check|phase_transition|assertion",
    "correlation": {
      "event_id": "<ULID>",
      "root_request_id": "<ULID>",
      "parent_event_id": "<ULID or null>"
    },
    "lane_id": "baseline|explain_mmr|...",
    "oracle_id": "ORACLE_DETERMINISTIC_ORDERING|...",
    "outcome": "pass|fail|skip",
    "reason_code": "e2e.oracle.ordering_violated",
    "severity": "info|warn|error",
    "context": "free-form diagnostic text",
    "metrics": {}
  }
}
```

Required event body fields:

| Field | Required | Description |
|---|---|---|
| `type` | Yes | E2e event type discriminant |
| `correlation` | Yes | Trace linking (ULID event chain) |
| `severity` | Yes | Event severity level |
| `lane_id` | If lane test | Interaction lane identifier |
| `oracle_id` | If oracle check | Oracle assertion identifier |
| `outcome` | If oracle check | Pass/fail/skip |
| `reason_code` | If fail/skip | Machine-stable reason code |
| `context` | No | Human-readable diagnostic text |
| `metrics` | No | Arbitrary numeric metrics |

## Oracle Report Schema (`oracle-report.json`)

Aggregated interaction test results:

```json
{
  "body": {
    "lanes": [
      {
        "lane_id": "baseline",
        "seed": 3405643776,
        "query_count": 5,
        "verdicts": [
          {
            "oracle_id": "ORACLE_NO_DUPLICATES",
            "outcome": "pass",
            "context": ""
          }
        ],
        "pass_count": 8,
        "fail_count": 0,
        "skip_count": 2,
        "all_passed": true
      }
    ],
    "totals": {
      "lanes_run": 12,
      "lanes_passed": 12,
      "oracles_pass": 96,
      "oracles_fail": 0,
      "oracles_skip": 24,
      "all_passed": true
    }
  }
}
```

## Replay Schema (`replay.jsonl`)

Input recording for deterministic re-execution. Each line:

```json
{
  "body": {
    "type": "query|config_change|clock_advance|signal",
    "offset_ms": 0,
    "seq": 0,
    "payload": {
      "query_text": "search term",
      "k": 10
    }
  }
}
```

Required replay body fields:

| Field | Required | Description |
|---|---|---|
| `type` | Yes | Replay event discriminant |
| `offset_ms` | Yes | Milliseconds since run start |
| `seq` | Yes | Monotonic sequence number |
| `payload` | Yes | Type-specific replay data |

## Snapshot Diff Schema (`snapshot-diff.json`)

Before/after comparison for regression detection:

```json
{
  "body": {
    "comparison_mode": "bit_exact|semantic|statistical",
    "baseline_run_id": "<ULID>",
    "diffs": [
      {
        "field_path": "results[0].score",
        "baseline": "0.8765",
        "current": "0.8764",
        "delta": "0.0001",
        "within_tolerance": true,
        "tolerance": "0.001"
      }
    ],
    "pass": true,
    "mismatch_count": 0
  }
}
```

## Transcript Format (`transcript.txt`)

Plain text execution log with fixed section markers:

```
=== E2E RUN <run_id> ===
Suite: core
Started: 2026-02-14T00:00:00Z
Seed: 42

--- PHASE: setup ---
[00:00.000] Index built: 100 docs, 256d fast, 384d quality

--- PHASE: execution ---
[00:00.012] Lane baseline: 8 pass, 0 fail, 2 skip
[00:00.024] Lane explain_mmr: 10 pass, 0 fail, 0 skip

--- SUMMARY ---
12 lanes, 96 pass, 0 fail, 24 skip
Duration: 1234ms
Exit: PASS
```

## Checksum Policy

- All checksums use SHA-256, formatted as `sha256:<64 hex chars>`.
- The manifest `artifacts` array lists every file in the pack with its checksum.
- The manifest itself is not self-checksummed (it is the root of trust).
- CI gates verify checksums before consuming any artifact file.

## Versioning Strategy

- The envelope `v` field is the schema version.
- Breaking changes (field removal, type change, semantic change) bump `v`.
- Additive optional fields are allowed at the same version if added to the JSON schema.
- The `schema` field disambiguates artifact types at the same version.
- Old consumers that encounter `v > 1` must reject the artifact with a clear error.

## Reason Code Namespace

E2e-specific reason codes use the `e2e.*` namespace:

| Code | Meaning |
|---|---|
| `e2e.oracle.pass` | Oracle assertion passed |
| `e2e.oracle.ordering_violated` | Result ordering invariant broken |
| `e2e.oracle.duplicates_found` | Duplicate results detected |
| `e2e.oracle.phase_mismatch` | Search phase did not match expected |
| `e2e.oracle.score_non_monotonic` | Scores not monotonically decreasing |
| `e2e.oracle.skip_feature_disabled` | Oracle skipped (feature not enabled) |
| `e2e.oracle.skip_stub_backend` | Oracle skipped (stub backend) |
| `e2e.run.setup_failed` | Test setup failed before execution |
| `e2e.run.timeout` | Run exceeded time budget |
| `e2e.replay.seed_mismatch` | Replay seed did not match manifest |
| `e2e.diff.tolerance_exceeded` | Snapshot diff exceeded tolerance |
| `e2e.diff.field_missing` | Expected field absent in current run |

## Validation Artifacts

- Schema: `schemas/e2e-artifact-v1.schema.json`
- Valid fixtures: `schemas/fixtures/e2e-*.json`
- Negative fixtures: `schemas/fixtures-invalid/e2e-*.json`

## Validation Commands

```bash
for f in schemas/fixtures/e2e-*.json; do
  jsonschema -i "$f" schemas/e2e-artifact-v1.schema.json
done

for f in schemas/fixtures-invalid/e2e-*.json; do
  if jsonschema -i "$f" schemas/e2e-artifact-v1.schema.json; then
    echo "unexpected pass: $f" && exit 1
  fi
done
```
