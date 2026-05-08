#!/usr/bin/env bash
set -euo pipefail

MODE="all"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCHEMA="$ROOT_DIR/schemas/fsfs-corpus-privacy-preflight-v1.schema.json"
FAILURES=0

usage() {
  cat <<USAGE
Usage: scripts/check_fsfs_corpus_privacy_preflight.sh [--mode unit|smoke|e2e|all]

Validates deterministic fsfs corpus privacy preflight fixtures for bd-pkl0.7.
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
  unit|smoke|e2e|all) ;;
  *)
    echo "ERROR: invalid mode '$MODE' (expected unit|smoke|e2e|all)" >&2
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
  echo "[unit] validating corpus privacy preflight contract"
  check_valid "unit" "$ROOT_DIR/schemas/fixtures/fsfs-corpus-privacy-preflight-contract-v1.json"
}

check_smoke() {
  echo "[smoke] validating dry-run include/skip/defer decisions and redaction proof"
  check_valid "smoke" "$ROOT_DIR/schemas/fixtures/fsfs-corpus-privacy-preflight-report-v1.json"
  check_invalid "smoke" "$ROOT_DIR/schemas/fixtures-invalid/fsfs-corpus-privacy-preflight-invalid-raw-content-v1.json"
}

check_e2e() {
  echo "[e2e] validating override guardrails and no-cleanup safety"
  check_valid "e2e" "$ROOT_DIR/schemas/fixtures/fsfs-corpus-privacy-preflight-override-v1.json"
  check_invalid "e2e" "$ROOT_DIR/schemas/fixtures-invalid/fsfs-corpus-privacy-preflight-invalid-destructive-cleanup-v1.json"
  check_invalid "e2e" "$ROOT_DIR/schemas/fixtures-invalid/fsfs-corpus-privacy-preflight-invalid-override-missing-reason-v1.json"
}

if [[ "$MODE" == "unit" || "$MODE" == "all" ]]; then
  check_unit
fi
if [[ "$MODE" == "smoke" || "$MODE" == "all" ]]; then
  check_smoke
fi
if [[ "$MODE" == "e2e" || "$MODE" == "all" ]]; then
  check_e2e
fi

if ((FAILURES > 0)); then
  echo "Result: FAIL ($FAILURES violation(s))"
  exit 1
fi

echo "Result: PASS"
