# fsfs Degraded Incident Suite

The degraded incident suite is a deterministic contract for replaying failure modes without real network access or destructive filesystem actions.

It covers:

- quality embedder timeout
- model unavailable
- corrupt vector artifact
- lexical backend failure
- storage lock pressure
- watcher backlog

Artifacts live in:

- `schemas/fsfs-degraded-incident-suite-v1.schema.json`
- `schemas/fixtures/fsfs-degraded-incident-suite-*.json`
- `crates/frankensearch-fsfs/tests/golden/fsfs_degraded_incident_suite_*`

Validation:

```bash
scripts/check_fsfs_degraded_incident_suite.sh --mode all
cargo test -p frankensearch-fsfs degraded_incident -- --nocapture
cargo test -p frankensearch-fsfs --test schema_conformance degraded_incident -- --nocapture
```

Every suite fixture carries a seed, config hash, offline network policy, non-destructive flag, structured log field requirements, expected output surface, reason code, and artifact references.
