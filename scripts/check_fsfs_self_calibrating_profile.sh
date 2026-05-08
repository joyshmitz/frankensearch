#!/usr/bin/env bash
set -euo pipefail

MODE="all"
RUN_ID="bd-pkl0.1-self-calibrating"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCHEMA_VERSION="fsfs-self-calibrating-profile-v1"

usage() {
  cat <<USAGE
Usage: scripts/check_fsfs_self_calibrating_profile.sh [--mode unit|integration|e2e|all] [--run-id <id>]

Validates the fsfs self-calibrating host/corpus profile contract for bd-pkl0.1.
Emits structured JSONL progress events to stdout. Set FSFS_SELF_CAL_USE_RCH=1
to run cargo checks through rch.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --run-id)
      RUN_ID="${2:-}"
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
  unit|integration|e2e|all) ;;
  *)
    echo "ERROR: invalid mode '$MODE' (expected unit|integration|e2e|all)" >&2
    exit 2
    ;;
esac

case "$RUN_ID" in
  *[!A-Za-z0-9._-]*|"")
    echo "ERROR: --run-id must contain only letters, digits, '.', '_', or '-'" >&2
    exit 2
    ;;
esac

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/tmp/frankensearch-fsfs-self-calibrating-profile-target}"
REPLAY_COMMAND="scripts/check_fsfs_self_calibrating_profile.sh --mode e2e --run-id ${RUN_ID}"

emit_event() {
  local phase="$1"
  local status="$2"
  local artifact="$3"
  printf '{"schema_version":"%s","run_id":"%s","phase":"%s","status":"%s","artifact":"%s","replay_command":"%s"}\n' \
    "$SCHEMA_VERSION" "$RUN_ID" "$phase" "$status" "$artifact" "$REPLAY_COMMAND"
}

run_cargo() {
  if [[ "${FSFS_SELF_CAL_USE_RCH:-0}" == "1" ]]; then
    RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR rch exec -- cargo "$@"
  else
    cargo "$@"
  fi
}

check_unit() {
  emit_event "unit" "start" "runs/${RUN_ID}/self_calibrating/unit/profile-events.jsonl"
  (
    cd "$ROOT_DIR"
    run_cargo test -p frankensearch-fsfs --lib self_calibrating_profile -- --nocapture
  )
  emit_event "unit" "pass" "runs/${RUN_ID}/self_calibrating/unit/recommendation.json"
}

check_integration() {
  emit_event "integration" "start" "runs/${RUN_ID}/self_calibrating/integration/profile-events.jsonl"
  (
    cd "$ROOT_DIR"
    run_cargo test -p frankensearch-fsfs --test profiling_harness_workflow self_calibrating_profile_report -- --nocapture
  )
  emit_event "integration" "pass" "runs/${RUN_ID}/self_calibrating/integration/recommendation.json"
}

check_e2e() {
  emit_event "e2e" "start" "runs/${RUN_ID}/self_calibrating/e2e/profile-events.jsonl"
  (
    cd "$ROOT_DIR"
    run_cargo test -p frankensearch-fsfs --test profiling_harness_workflow self_calibrating_profile_report_exposes_recommendation_artifacts -- --exact --nocapture
  )
  emit_event "e2e" "pass" "runs/${RUN_ID}/self_calibrating/e2e/replay-manifest.json"
}

if [[ "$MODE" == "unit" || "$MODE" == "all" ]]; then
  check_unit
fi
if [[ "$MODE" == "integration" || "$MODE" == "all" ]]; then
  check_integration
fi
if [[ "$MODE" == "e2e" || "$MODE" == "all" ]]; then
  check_e2e
fi

emit_event "summary" "pass" "runs/${RUN_ID}/self_calibrating/profile-events.jsonl"
