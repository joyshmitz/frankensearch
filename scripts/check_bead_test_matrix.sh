#!/usr/bin/env bash
set -euo pipefail

MODE="all"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISSUES_FILE="$ROOT_DIR/.beads/issues.jsonl"

usage() {
  cat <<USAGE
Usage: scripts/check_bead_test_matrix.sh [--mode unit|integration|all] [--issues <path>]

Validates per-bead test matrix policy anchors for bd-264r.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --issues)
      ISSUES_FILE="${2:-}"
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
  unit|integration|all) ;;
  *)
    echo "ERROR: invalid mode '$MODE' (expected unit|integration|all)" >&2
    exit 2
    ;;
esac

if [[ ! -f "$ISSUES_FILE" ]]; then
  echo "ERROR: issues file not found: $ISSUES_FILE" >&2
  exit 2
fi

DATA="$(jq -cs '.' "$ISSUES_FILE")"
FAILURES=0

WAVE1=(bd-3un.31 bd-3un.32 bd-3un.40 bd-3un.52)
WAVE2_EXCEPTIONS=(bd-2hz.10 bd-2yu.8)

check_wave1_matrix() {
  local issue_id="$1"
  local comments
  comments="$(jq -r --arg id "$issue_id" '
    ([.[] | select(.id == $id)] | .[0]) as $issue
    | ($issue.comments // []) | map(.text) | join("\n")
  ' <<<"$DATA")"

  if ! grep -Fq "[bd-264r test-matrix] TEST_MATRIX" <<<"$comments"; then
    echo "[unit][FAIL] $issue_id missing TEST_MATRIX annotation"
    FAILURES=$((FAILURES + 1))
    return
  fi

  local required_sections=("Unit tests:" "Integration tests:" "E2E tests:" "Performance" "Logs/artifacts")
  for section in "${required_sections[@]}"; do
    if ! grep -Fqi "$section" <<<"$comments"; then
      echo "[unit][FAIL] $issue_id missing section hint '$section'"
      FAILURES=$((FAILURES + 1))
    fi
  done

  echo "[unit][OK]   $issue_id has explicit TEST_MATRIX annotation"
}

check_wave2_exception() {
  local issue_id="$1"
  local comments
  comments="$(jq -r --arg id "$issue_id" '
    ([.[] | select(.id == $id)] | .[0]) as $issue
    | ($issue.comments // []) | map(.text) | join("\n")
  ' <<<"$DATA")"

  if ! grep -Fq "[bd-264r test-matrix] EXCEPTION" <<<"$comments"; then
    echo "[unit][FAIL] $issue_id missing EXCEPTION annotation"
    FAILURES=$((FAILURES + 1))
  else
    echo "[unit][OK]   $issue_id has explicit EXCEPTION rationale"
  fi
}

check_unit() {
  echo "[unit] validating wave-1 and wave-2 policy anchors"

  for issue_id in "${WAVE1[@]}"; do
    check_wave1_matrix "$issue_id"
  done

  for issue_id in "${WAVE2_EXCEPTIONS[@]}"; do
    check_wave2_exception "$issue_id"
  done
}

check_integration() {
  echo "[integration] surfacing candidate open implementation beads missing explicit matrix markers"

  local candidates
  candidates="$(jq -r '
    .[]
    | select((.issue_type == "task" or .issue_type == "feature") and (.status == "open" or .status == "in_progress"))
    | . as $issue
    | ($issue.description // "") as $desc
    | (($issue.comments // []) | map(.text) | join("\n")) as $comments
    | (
        ($desc | test("(?i)unit")) and
        ($desc | test("(?i)integration")) and
        ($desc | test("(?i)e2e")) and
        ($desc | test("(?i)bench|perf|performance")) and
        ($desc | test("(?i)log|metric|artifact"))
      ) as $desc_has_matrix
    | (
        ($comments | test("\\[bd-264r test-matrix\\] TEST_MATRIX")) or
        ($comments | test("\\[bd-264r test-matrix\\] EXCEPTION"))
      ) as $comment_has_marker
    | select(($desc_has_matrix | not) and ($comment_has_marker | not))
    | $issue.id
  ' <<<"$DATA" | sort -u)"

  if [[ -n "$candidates" ]]; then
    echo "[integration][INFO] potential missing-matrix candidates (outside scoped retrofit):"
    echo "$candidates" | sed 's/^/  - /'
  else
    echo "[integration][OK]   no additional candidates detected"
  fi

  echo "[integration][OK]   scoped retrofit anchors remain enforceable via unit checks"
}

if [[ "$MODE" == "unit" || "$MODE" == "all" ]]; then
  check_unit
fi
if [[ "$MODE" == "integration" || "$MODE" == "all" ]]; then
  check_integration
fi

if [[ "$FAILURES" -gt 0 ]]; then
  echo "Result: FAIL ($FAILURES violation(s))"
  exit 1
fi

echo "Result: PASS"
