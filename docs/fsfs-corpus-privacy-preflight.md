# fsfs Corpus Privacy Preflight v1

Issue: `bd-pkl0.7`

## Goal

Prevent sensitive or low-value corpus content from entering semantic/vector stores by
running a deterministic dry-run preflight before indexing. The preflight reports
what would be indexed, skipped, or deferred, and emits only redacted evidence.

## Required Signals

The v1 rule matrix covers:

- credential/token-like content
- private keys
- generated artifacts
- oversized binaries
- sensitive paths
- personal data

Each rule has a stable `privacy.*` reason code, a default include/skip/defer
decision, and a redaction action. User overrides are allowed only on rules that
explicitly opt in to overrides.

## Report Contract

Every report is dry-run only:

- `dry_run = true`
- `destructive_cleanup_allowed = false`
- `raw_content_present = false` at both evidence and summary level
- include/skip/defer counts must match the emitted decisions
- approved overrides require a non-empty reason and expiry
- false-positive suppressions require a named suppressor

The report can be replayed with:

```bash
scripts/check_fsfs_corpus_privacy_preflight.sh --mode all
```

## Validation Artifacts

- `schemas/fsfs-corpus-privacy-preflight-v1.schema.json`
- `schemas/fixtures/fsfs-corpus-privacy-preflight-contract-v1.json`
- `schemas/fixtures/fsfs-corpus-privacy-preflight-report-v1.json`
- `schemas/fixtures/fsfs-corpus-privacy-preflight-override-v1.json`
- `schemas/fixtures-invalid/fsfs-corpus-privacy-preflight-invalid-*.json`
- `crates/frankensearch-fsfs/tests/golden/fsfs_corpus_privacy_preflight_*_roundtrip_v1.golden.json`
