# fsfs Replay Bundle Contract

The canonical fsfs replay bundle is the shared replay contract for search,
index, doctor, audit, and degraded-mode scenarios. It gives every captured
scenario one machine-readable manifest that can be validated before replay
and round-tripped by Rust without relying on a scenario-specific artifact
shape.

## Required Bundle Shape

Every replay bundle manifest uses `kind = "fsfs_replay_bundle_manifest"` and
`v = 1`. The top-level manifest must include:

- `bundle_id`
- `scenario_id`
- `scenario_kind`
- `created_at`
- `command`
- `environment`
- `fixture_refs`
- `expected_phase_outcomes`
- `artifact_manifest`

`scenario_kind` is one of `search`, `index`, `doctor`, `audit`, or
`degraded_mode`.

## Command Contract

`command` records the exact invocation needed to replay the scenario:

- `client_surface`: `cli` or `tui`
- `argv`: full argument vector, including executable name
- `working_dir`: replay working directory

The argument vector is intentionally structured as an array. Callers must not
parse or reconstruct it from a shell string.

## Environment Contract

`environment` records deterministic identity:

- `seed`: required replay seed
- `config_hash`: `sha256:<64 lowercase hex chars>`
- `snapshot`: redacted environment variables using the fsfs env snapshot format

The manifest is invalid without a seed or config hash. Sensitive values in the
environment snapshot must be redacted before writing the fixture.

## Fixtures and Expected Outcomes

`fixture_refs` names every input fixture consumed by the replay. Each fixture
has a stable `fixture_id`, relative `path`, and `checksum_sha256`.

`expected_phase_outcomes` describes the replay phases that must be observed.
Supported phases are `initial`, `refined`, `index_build`, `doctor`, `audit`,
and `degraded_mode`. A `degraded`, `skipped`, or `failed` phase must include a
non-empty `reason_code`.

Each phase lists `artifact_refs`, which must point to entries in
`artifact_manifest.artifacts`.

## Artifact Manifest

`artifact_manifest.artifacts` lists emitted or required replay artifacts:

- `artifact_id`
- `path`
- `content_type`
- `checksum_sha256`
- `required`

Rust validation rejects duplicate fixture IDs, duplicate artifact IDs, unknown
phase artifact refs, missing command data, missing seed/config hash, missing
artifact manifests, and malformed SHA-256 digests.

## Operator Workflow Playback Fixtures

`operator_workflow_playback_suite()` provides deterministic playback fixtures
for common operator workflows that do not require live interaction:

- fsfs search
- fsfs explain
- fsfs doctor
- fsfs audit
- fsfs degraded mode
- fsfs profile review
- ops TUI triage

Each workflow fixture records a stable command vector, replay input sequence,
expected screen/output snapshots, 16-hex state checksums, reason codes, and
artifact references. TUI fixtures must pin viewport size with a `resize:*`
input before assertions.

The replay command embedded in the suite is:

```bash
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR rch exec -- cargo test -p frankensearch-fsfs --lib operator_workflow_playback -- --nocapture
```

## Validation

The contract artifacts are:

- `schemas/fsfs-replay-bundle-v1.schema.json`
- `schemas/fixtures/fsfs-replay-bundle-contract-v1.json`
- `schemas/fixtures/fsfs-replay-bundle-manifest-v1.json`
- `schemas/fixtures-invalid/fsfs-replay-bundle-invalid-*.json`

Run:

```bash
scripts/check_fsfs_replay_bundle_contract.sh --mode all
```

Rust schema conformance tests parse the valid fixtures, reject invalid fixtures
during deserialization, and compare golden pretty-JSON round-trips.
