#!/usr/bin/env bash
set -euo pipefail

MODE="all"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCHEMA="$ROOT_DIR/schemas/fsfs-degraded-incident-suite-v1.schema.json"
FAILURES=0

usage() {
  cat <<USAGE
Usage: scripts/check_fsfs_degraded_incident_suite.sh [--mode unit|smoke|full|all]

Validates deterministic fsfs degraded-mode incident suite fixtures for bd-pkl0.12.
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
  unit|smoke|full|all) ;;
  *)
    echo "ERROR: invalid mode '$MODE' (expected unit|smoke|full|all)" >&2
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
  local scope="$1"
  local file="$2"
  if jsonschema -i "$file" "$SCHEMA" >/dev/null 2>&1; then
    echo "[$scope][OK]   valid fixture accepted: $file"
  else
    echo "[$scope][FAIL] valid fixture rejected: $file"
    FAILURES=$((FAILURES + 1))
  fi
}

check_invalid() {
  local scope="$1"
  local file="$2"
  if jsonschema -i "$file" "$SCHEMA" >/dev/null 2>&1; then
    echo "[$scope][FAIL] invalid fixture unexpectedly accepted: $file"
    FAILURES=$((FAILURES + 1))
  else
    echo "[$scope][OK]   invalid fixture rejected: $file"
  fi
}

check_unit() {
  echo "[unit] validating degraded incident suite contract"
  check_valid "unit" "$ROOT_DIR/schemas/fixtures/fsfs-degraded-incident-suite-contract-v1.json"
}

check_smoke() {
  echo "[smoke] validating degraded incident smoke lane"
  check_valid "smoke" "$ROOT_DIR/schemas/fixtures/fsfs-degraded-incident-suite-smoke-v1.json"
  check_invalid "smoke" "$ROOT_DIR/schemas/fixtures-invalid/fsfs-degraded-incident-suite-invalid-missing-reason-v1.json"
}

check_full() {
  echo "[full] validating full degraded incident suite and safety rejects"
  check_valid "full" "$ROOT_DIR/schemas/fixtures/fsfs-degraded-incident-suite-full-v1.json"
  check_invalid "full" "$ROOT_DIR/schemas/fixtures-invalid/fsfs-degraded-incident-suite-invalid-network-required-v1.json"
  check_invalid "full" "$ROOT_DIR/schemas/fixtures-invalid/fsfs-degraded-incident-suite-invalid-destructive-command-v1.json"
}

if [[ "$MODE" == "unit" || "$MODE" == "all" ]]; then
  check_unit
fi
if [[ "$MODE" == "smoke" || "$MODE" == "all" ]]; then
  check_smoke
fi
if [[ "$MODE" == "full" || "$MODE" == "all" ]]; then
  check_full
fi

if ((FAILURES > 0)); then
  echo "Result: FAIL ($FAILURES violation(s))"
  exit 1
fi

echo "Result: PASS"
