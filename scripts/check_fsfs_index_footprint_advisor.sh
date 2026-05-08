#!/usr/bin/env bash
set -euo pipefail

MODE="all"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCHEMA="$ROOT_DIR/schemas/fsfs-index-footprint-advisor-v1.schema.json"
FAILURES=0

usage() {
  cat <<USAGE
Usage: scripts/check_fsfs_index_footprint_advisor.sh [--mode schema|rust|all]

Validates deterministic fsfs index-footprint advisor fixtures for bd-pkl0.8.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$MODE" in
  schema|rust|all) ;;
  *)
    echo "ERROR: invalid mode '$MODE' (expected schema|rust|all)" >&2
    exit 2
    ;;
esac

if [[ ! -f "$SCHEMA" ]]; then
  echo "ERROR: schema not found: $SCHEMA" >&2
  exit 2
fi

if ! command -v jsonschema >/dev/null 2>&1; then
  echo "ERROR: jsonschema CLI not found in PATH" >&2
  exit 2
fi

check_valid() {
  local file="$1"
  if jsonschema -i "$file" "$SCHEMA" >/dev/null 2>&1; then
    echo "[schema][OK]   valid fixture accepted: $file"
  else
    echo "[schema][FAIL] valid fixture rejected: $file"
    FAILURES=$((FAILURES + 1))
  fi
}

check_invalid() {
  local file="$1"
  if jsonschema -i "$file" "$SCHEMA" >/dev/null 2>&1; then
    echo "[schema][FAIL] invalid fixture unexpectedly accepted: $file"
    FAILURES=$((FAILURES + 1))
  else
    echo "[schema][OK]   invalid fixture rejected: $file"
  fi
}

check_schema() {
  echo "[schema] validating index-footprint advisor fixtures"
  check_valid "$ROOT_DIR/schemas/fixtures/fsfs-index-footprint-advisor-contract-v1.json"
  check_valid "$ROOT_DIR/schemas/fixtures/fsfs-index-footprint-advisor-small-v1.json"
  check_valid "$ROOT_DIR/schemas/fixtures/fsfs-index-footprint-advisor-fragmented-v1.json"
  check_valid "$ROOT_DIR/schemas/fixtures/fsfs-index-footprint-advisor-oversized-v1.json"
  check_invalid "$ROOT_DIR/schemas/fixtures-invalid/fsfs-index-footprint-advisor-invalid-auto-delete-v1.json"
  check_invalid "$ROOT_DIR/schemas/fixtures-invalid/fsfs-index-footprint-advisor-invalid-missing-replay-v1.json"
}

check_rust() {
  echo "[rust] running focused index-footprint advisor tests"
  cargo test -p frankensearch-fsfs index_footprint_advisor -- --nocapture
}

if [[ "$MODE" == "schema" || "$MODE" == "all" ]]; then
  check_schema
fi
if [[ "$MODE" == "rust" || "$MODE" == "all" ]]; then
  check_rust
fi

if ((FAILURES > 0)); then
  echo "Result: FAIL ($FAILURES violation(s))"
  exit 1
fi

echo "Result: PASS"
