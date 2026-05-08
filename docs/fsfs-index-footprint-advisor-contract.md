# fsfs Index Footprint Advisor Contract

`bd-pkl0.8` adds a deterministic dry-run advisor for fsfs index footprint decisions. The advisor measures vector index, lexical index, metadata, model-cache, and artifact footprints, then emits recommendations for compaction, rebuild, retention, or feature adjustment without deleting or rewriting anything.

The contract guarantees:

- dry-run-only reports with `automatic_deletion_allowed: false`
- required coverage for all five footprint domains
- deterministic threshold policy for small, fragmented, and oversized index states
- projected byte savings on every recommendation
- explicit risk labels for low, medium, and high impact operator choices
- exact replay commands for each domain recommendation

Artifacts:

- `schemas/fsfs-index-footprint-advisor-v1.schema.json`
- `schemas/fixtures/fsfs-index-footprint-advisor-contract-v1.json`
- `schemas/fixtures/fsfs-index-footprint-advisor-small-v1.json`
- `schemas/fixtures/fsfs-index-footprint-advisor-fragmented-v1.json`
- `schemas/fixtures/fsfs-index-footprint-advisor-oversized-v1.json`
- `schemas/fixtures-invalid/fsfs-index-footprint-advisor-invalid-auto-delete-v1.json`
- `schemas/fixtures-invalid/fsfs-index-footprint-advisor-invalid-missing-replay-v1.json`

Replay one recommendation:

```bash
FSFS_INDEX_FOOTPRINT_FIXTURE=fragmented FSFS_INDEX_FOOTPRINT_DOMAIN=vector_index cargo test -p frankensearch-fsfs index_footprint_advisor_policy_suite -- --nocapture
```

Validate schema fixtures and the focused Rust suite:

```bash
scripts/check_fsfs_index_footprint_advisor.sh --mode all
```
