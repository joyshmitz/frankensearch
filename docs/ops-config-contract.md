# Unified OpsConfig Contract v1

Issue: `bd-2yu.2.6`

## Goal

Provide one canonical config surface for the ops control plane, independent from search-engine `TwoTierConfig`.

This contract defines:

- complete `OpsConfig` parameter set with defaults
- source precedence (`env > file > defaults`)
- config file location policy
- validation rules and error expectations
- runtime-reload constraints

Artifacts:

- Schema: `schemas/ops-config-v1.schema.json`
- Valid fixtures: `schemas/fixtures/ops-config-*.json`
- Invalid fixtures: `schemas/fixtures-invalid/ops-config-*.json`

## Source Precedence (Normative)

Exact precedence order:

1. environment variables
2. config file
3. compiled defaults

No component may override this order.

## Config File Location

Preferred path (XDG-compliant):

- `${XDG_CONFIG_HOME}/frankensearch/ops.toml`

Fallback:

- `~/.config/frankensearch/ops.toml`

## OpsConfig Parameters

| Field | Type | Default | Env Var | Validation | Reloadable |
|---|---|---|---|---|---|
| `telemetry_collection_interval_ms` | int | `1000` | `FRANKENSEARCH_OPS_TELEMETRY_COLLECTION_INTERVAL_MS` | `>=100` | no |
| `ingestion_batch_size` | int | `100` | `FRANKENSEARCH_OPS_INGESTION_BATCH_SIZE` | `1..100000` | no |
| `retention_raw_days` | int | `7` | `FRANKENSEARCH_OPS_RETENTION_RAW_DAYS` | `1..3650` | no |
| `retention_summary_days` | int | `90` | `FRANKENSEARCH_OPS_RETENTION_SUMMARY_DAYS` | `1..3650` | no |
| `slo_search_p99_ms` | int | `500` | `FRANKENSEARCH_OPS_SLO_SEARCH_P99_MS` | `>=1` | yes |
| `slo_embedding_throughput_min_docs_per_s` | number | `10.0` | `FRANKENSEARCH_OPS_SLO_EMBED_MIN_DOCS_PER_S` | `>0` | yes |
| `discovery_scan_interval_ms` | int | `30000` | `FRANKENSEARCH_OPS_DISCOVERY_SCAN_INTERVAL_MS` | `>=1000` | yes |
| `ui_refresh_interval_ms` | int | `250` | `FRANKENSEARCH_OPS_UI_REFRESH_INTERVAL_MS` | `>=50` | yes |
| `backpressure_queue_depth_limit` | int | `10000` | `FRANKENSEARCH_OPS_BACKPRESSURE_QUEUE_DEPTH_LIMIT` | `>=100` | yes |

Additional invariants:

- `retention_summary_days >= retention_raw_days`
- intervals must be positive and nonzero

## Runtime Reload Rules

Reloadable at runtime:

- `slo_search_p99_ms`
- `slo_embedding_throughput_min_docs_per_s`
- `discovery_scan_interval_ms`
- `ui_refresh_interval_ms`
- `backpressure_queue_depth_limit`

Restart-required:

- `telemetry_collection_interval_ms`
- `ingestion_batch_size`
- `retention_raw_days`
- `retention_summary_days`

## Validation and Error Messaging Contract

Config validation failures must include:

- `field`
- `provided_value`
- `constraint`
- `source` (`env`, `file`, `default`)
- stable `reason_code` (`config.<field>.<error>`)

Example reason codes:

- `config.ui_refresh_interval_ms.out_of_range`
- `config.retention_summary_days.lt_raw_days`
- `config.unknown_key.present`

## Documentation Generation Contract

Config docs should be generated from field metadata:

- field name
- default
- env var
- validation
- reloadable flag
- description

The schema fixtures in this bead are the source-of-truth contract for that metadata model.

## Validation Strategy

## Positive fixtures

```bash
for f in schemas/fixtures/ops-config-*.json; do
  jsonschema -i "$f" schemas/ops-config-v1.schema.json
done
```

## Negative fixtures (must fail)

```bash
for f in schemas/fixtures-invalid/ops-config-*.json; do
  if jsonschema -i "$f" schemas/ops-config-v1.schema.json; then
    echo "unexpected pass: $f" && exit 1
  fi
done
```

These checks enforce precedence shape, unknown-key rejection, and numeric validation bounds.
