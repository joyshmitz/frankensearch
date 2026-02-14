# fsfs Configuration Contract v1

Issue: `bd-2hz.13`  
Parent: `bd-2hz`

## Goal

Define the canonical fsfs configuration model, precedence rules, validation semantics, and diagnostics contract so downstream implementation beads can consume one unambiguous source of truth.

This contract is normative for both fsfs UX modes:

- agent-first CLI
- deluxe TUI

Artifacts:

- Schema: `schemas/fsfs-config-v1.schema.json`
- Valid fixtures: `schemas/fixtures/fsfs-config-*.json`
- Invalid fixtures: `schemas/fixtures-invalid/fsfs-config-invalid-*.json`
- Checker: `scripts/check_fsfs_config_contract.sh`

## Source Precedence (Normative)

Exact precedence order (highest to lowest):

1. CLI flags
2. environment variables
3. config file
4. compiled defaults

No component may override this order.

## Config File Location Policy

Primary path:

- `${XDG_CONFIG_HOME}/fsfs/config.toml`

Fallback path:

- `~/.config/fsfs/config.toml`

Path expansion rule:

- Any path value starting with `~` MUST expand to the current user home directory before validation.

## Configuration Sections

## `[discovery]`

- `roots: string[]` (default: `[$HOME]`)
- `exclude_patterns: string[]`
- `text_selection_mode: "blocklist" | "allowlist"` (default: `blocklist`)
- `binary_blocklist_extensions: string[]`
- `max_file_size_mb: int` (`1..1024`)
- `follow_symlinks: bool`

## `[indexing]`

- `fast_model: string`
- `quality_model: string`
- `model_dir: string`
- `embedding_batch_size: int` (`1..4096`)
- `reindex_on_change: bool`
- `watch_mode: bool`

## `[search]`

- `default_limit: int` (`1..200`)
- `quality_weight: number` (`0.0..1.0`)
- `rrf_k: number` (`>=1.0`)
- `quality_timeout_ms: int` (`>=50`)
- `fast_only: bool`
- `explain: bool`

## `[pressure]`

- `profile: "strict" | "performance" | "degraded"`
- `cpu_ceiling_pct: int` (`1..100`)
- `memory_ceiling_mb: int` (`>=128`)

## `[tui]`

- `theme: "auto" | "light" | "dark"`
- `frame_budget_ms: int` (`8..200`)
- `show_explanations: bool`
- `density: "compact" | "normal" | "expanded"`

## `[storage]`

- `db_path: string`
- `evidence_retention_days: int` (`1..3650`)
- `summary_retention_days: int` (`1..3650`)

## `[privacy]`

- `redact_file_contents_in_logs: bool` (default MUST be `true`)
- `redact_paths_in_telemetry: bool` (default MUST be `true`)

## Validation Rules (Normative)

- Unknown keys in config files MUST generate warnings, not hard errors.
- Validation failures MUST include stable reason codes and field paths.
- `summary_retention_days` MUST be greater than or equal to `evidence_retention_days`.
- If `search.fast_only = true` while `indexing.quality_model` is configured, a warning MUST be emitted:
  `config.search.fast_only_with_quality_model`.

## Environment Variable Mapping

Every key has an env var mapped with `FSFS_{SECTION}_{KEY}` in `SCREAMING_SNAKE_CASE`.

Examples:

- `search.quality_weight` -> `FSFS_SEARCH_QUALITY_WEIGHT`
- `pressure.profile` -> `FSFS_PRESSURE_PROFILE`
- `privacy.redact_paths_in_telemetry` -> `FSFS_PRIVACY_REDACT_PATHS_IN_TELEMETRY`

## CLI Flag Mapping (Required Surface)

- `--roots`
- `--exclude`
- `--limit`
- `--fast-only`
- `--explain`
- `--profile`
- `--theme`

## Diagnostics + Logging Contract

`config_loaded` event MUST include:

- `event`
- `source_precedence_applied`
- `cli_flags_used`
- `env_keys_used`
- `config_file_used`
- `resolved_values`
- `warnings`
- `reason_codes`

All diagnostics MUST be machine-readable and replay-safe.

## Validation Commands

```bash
scripts/check_fsfs_config_contract.sh --mode unit
scripts/check_fsfs_config_contract.sh --mode integration
scripts/check_fsfs_config_contract.sh --mode e2e
scripts/check_fsfs_config_contract.sh --mode all
```

## Integration Mapping

- `bd-2hz.3.1`: consumes section/type model for config loader and CLI surface.
- `bd-2hz.4.5`: consumes `[pressure]` profile contract.
- `bd-2hz.1.3`: consumes `[privacy]` defaults and telemetry redaction semantics.
- `bd-2hz.3.8`: consumes file policy and restart/reload expectations.
