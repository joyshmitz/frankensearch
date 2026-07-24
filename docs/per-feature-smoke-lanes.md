# Per-Feature Smoke Lanes

The minimal feature smoke lanes are owned by `scripts/check_feature_matrix.sh`.
The script validates that every required lane has a compile target, a behavior
test, and a deterministic artifact file before it runs any cargo command. A
lane is rejected unless its exact behavior filter executes one test; Cargo's
otherwise-successful zero-test result is not accepted.

Run every lane locally:

```bash
scripts/check_feature_matrix.sh
```

Run one lane through `rch` with an isolated target directory:

```bash
FRANKENSEARCH_FEATURE_MATRIX_USE_RCH=1 \
CARGO_TARGET_DIR=/tmp/rch_target_frankensearch_${AGENT_NAME:-agent}_feature_matrix \
scripts/check_feature_matrix.sh --lane hybrid
```

Validate the lane contract without compiling:

```bash
scripts/check_feature_matrix.sh --mode validate
```

| Lane | Compile command | Behavior test command | Artifact |
|---|---|---|---|
| `default` | `cargo check -p frankensearch --all-targets` | `cargo test -p frankensearch --lib feature_matrix_smoke::default_lane_behavior -- --exact --nocapture` | `feature-smoke-default.json` |
| `quill` | `cargo check -p frankensearch --lib --no-default-features --features quill` | `cargo test -p frankensearch --lib --no-default-features --features quill feature_matrix_smoke::quill_lane_behavior -- --exact --nocapture` | `feature-smoke-quill.json` |
| `lexical-tantivy` | `cargo check -p frankensearch --lib --no-default-features --features lexical-tantivy` | `cargo test -p frankensearch --lib --no-default-features --features lexical-tantivy feature_matrix_smoke::lexical_tantivy_lane_behavior -- --exact --nocapture` | `feature-smoke-lexical-tantivy.json` |
| `cass-compat` | `cargo check -p frankensearch --lib --no-default-features --features cass-compat` | `cargo test -p frankensearch --lib --no-default-features --features cass-compat feature_matrix_smoke::cass_compat_lane_behavior -- --exact --nocapture` | `feature-smoke-cass-compat.json` |
| `semantic` | `cargo check -p frankensearch --lib --no-default-features --features semantic` | `cargo test -p frankensearch --lib --no-default-features --features semantic feature_matrix_smoke::semantic_lane_behavior -- --exact --nocapture` | `feature-smoke-semantic.json` |
| `hybrid` | `cargo check -p frankensearch --lib --no-default-features --features hybrid` | `cargo test -p frankensearch --lib --no-default-features --features hybrid feature_matrix_smoke::hybrid_lane_behavior -- --exact --nocapture` | `feature-smoke-hybrid.json` |
| `persistent` | `cargo check -p frankensearch --lib --no-default-features --features persistent` | `cargo test -p frankensearch --lib --no-default-features --features persistent feature_matrix_smoke::persistent_lane_behavior -- --exact --nocapture` | `feature-smoke-persistent.json` |
| `durable` | `cargo check -p frankensearch --lib --no-default-features --features durable` | `cargo test -p frankensearch --lib --no-default-features --features durable feature_matrix_smoke::durable_lane_behavior -- --exact --nocapture` | `feature-smoke-durable.json` |
| `full` | `cargo check -p frankensearch --lib --no-default-features --features full` | `cargo test -p frankensearch --lib --no-default-features --features full feature_matrix_smoke::full_lane_behavior -- --exact --nocapture` | `feature-smoke-full.json` |
| `full-fts5` | `cargo check -p frankensearch --lib --no-default-features --features full-fts5` | `cargo test -p frankensearch --lib --no-default-features --features full-fts5 feature_matrix_smoke::full_fts5_lane_behavior -- --exact --nocapture` | `feature-smoke-full-fts5.json` |

CI runs the same script once per lane and uploads the generated artifact files
with deterministic names. Each per-lane artifact includes the lane name,
feature set, compile command, behavior test command, and status. The companion
`feature-smoke-matrix.json` records the complete required lane set for audit
and replay.
