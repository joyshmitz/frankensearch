#!/usr/bin/env bash
set -euo pipefail

MODE="all"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISSUES_FILE="$ROOT_DIR/.beads/issues.jsonl"
POLICY_TAG="[bd-2l7y baseline-budget]"

TARGETS=(
  bd-21g
  bd-22k
  bd-2ps
  bd-2yj
  bd-1do
  bd-2tv
  bd-i37
  bd-l7v
  bd-1co
  bd-2rq
  bd-2u4
  bd-6sj
)

REQUIRED_FIELDS=(
  BASELINE_COMPARATOR
  BUDGETED_MODE_DEFAULTS
  ON_EXHAUSTION
  SUCCESS_THRESHOLDS_AND_STOP_CONDITIONS
)

FAILURES=0

usage() {
  cat <<USAGE
Usage: scripts/check_bead_baseline_budget.sh [--mode unit|integration|e2e|all] [--issues <path>]

Checks baseline/budget planning policy anchors for bd-2l7y.
USAGE
}

contains_id() {
  local needle="$1"
  shift
  local current
  for current in "$@"; do
    if [[ "$current" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

missing_fields_for_text() {
  local text="$1"
  local field
  for field in "${REQUIRED_FIELDS[@]}"; do
    if ! grep -Fq "${field}:" <<<"$text"; then
      echo "$field"
    fi
  done
}

emit_actionable_template() {
  cat <<'TEMPLATE'
      Add/repair template:
        [bd-2l7y baseline-budget]
        BASELINE_COMPARATOR: ...
        BUDGETED_MODE_DEFAULTS: ...
        ON_EXHAUSTION: ...
        SUCCESS_THRESHOLDS_AND_STOP_CONDITIONS: ...
TEMPLATE
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
  unit|integration|e2e|all) ;;
  *)
    echo "ERROR: invalid mode '$MODE' (expected unit|integration|e2e|all)" >&2
    exit 2
    ;;
esac

if [[ ! -f "$ISSUES_FILE" ]]; then
  echo "ERROR: issues file not found: $ISSUES_FILE" >&2
  exit 2
fi

DATA="$(jq -cs '.' "$ISSUES_FILE")"

issue_exists() {
  local issue_id="$1"
  jq -e --arg id "$issue_id" 'any(.[]; .id == $id)' <<<"$DATA" >/dev/null
}

issue_comments() {
  local issue_id="$1"
  jq -r --arg id "$issue_id" '
    ([.[] | select(.id == $id)] | .[0]) as $issue
    | (($issue.comments // []) | map(.text) | join("\n"))
  ' <<<"$DATA"
}

validate_text_payload() {
  local scope="$1"
  local subject="$2"
  local text="$3"
  local allow_missing="$4"
  local -a missing=()
  local has_tag=true

  if ! grep -Fq "$POLICY_TAG" <<<"$text"; then
    has_tag=false
    missing+=("$POLICY_TAG")
  fi

  local field
  for field in $(missing_fields_for_text "$text"); do
    missing+=("$field")
  done

  if [[ "$allow_missing" == "true" ]]; then
    if ((${#missing[@]} == 0)); then
      echo "[$scope][FAIL] $subject expected a policy violation but none was detected"
      FAILURES=$((FAILURES + 1))
    else
      echo "[$scope][OK]   $subject policy violation detected as expected"
      echo "[$scope][INFO] missing: ${missing[*]}"
      emit_actionable_template
    fi
    return
  fi

  if ((${#missing[@]} > 0)); then
    echo "[$scope][FAIL] $subject missing required policy anchors: ${missing[*]}"
    if [[ "$has_tag" == "false" ]]; then
      echo "[$scope][INFO] marker must include literal tag: $POLICY_TAG"
    fi
    emit_actionable_template
    FAILURES=$((FAILURES + 1))
    return
  fi

  echo "[$scope][OK]   $subject has full baseline/budget policy fields"
}

check_unit() {
  echo "[unit] validating bd-2l7y retrofit targets"

  local issue_id
  for issue_id in "${TARGETS[@]}"; do
    if ! issue_exists "$issue_id"; then
      echo "[unit][FAIL] $issue_id does not exist in issues file"
      FAILURES=$((FAILURES + 1))
      continue
    fi

    local comments
    comments="$(issue_comments "$issue_id")"
    validate_text_payload "unit" "$issue_id" "$comments" "false"
  done
}

check_integration() {
  echo "[integration] validating global marker integrity and surfacing adoption candidates"

  local -a annotated_ids=()
  mapfile -t annotated_ids < <(jq -r --arg tag "$POLICY_TAG" '
    .[]
    | . as $issue
    | (($issue.comments // []) | map(.text) | join("\n")) as $comments
    | select($comments | contains($tag))
    | .id
  ' <<<"$DATA" | sort -u)

  if ((${#annotated_ids[@]} == 0)); then
    echo "[integration][FAIL] no beads currently contain the $POLICY_TAG marker"
    FAILURES=$((FAILURES + 1))
  else
    local issue_id
    for issue_id in "${annotated_ids[@]}"; do
      local comments
      comments="$(issue_comments "$issue_id")"
      validate_text_payload "integration" "$issue_id" "$comments" "false"
    done
  fi

  local -a candidate_ids=()
  mapfile -t candidate_ids < <(jq -r '
    .[]
    | select((.issue_type == "task" or .issue_type == "feature") and (.status == "open" or .status == "in_progress"))
    | . as $issue
    | (($issue.labels // []) | map(ascii_downcase)) as $labels
    | select(
        ($labels | index("controls")) or
        ($labels | index("performance")) or
        ($labels | index("fusion")) or
        ($labels | index("ranking"))
      )
    | .id
  ' <<<"$DATA" | sort -u)

  local -a missing_marker=()
  local candidate_id
  for candidate_id in "${candidate_ids[@]}"; do
    if contains_id "$candidate_id" "${annotated_ids[@]}"; then
      continue
    fi
    if contains_id "$candidate_id" "${TARGETS[@]}"; then
      continue
    fi
    missing_marker+=("$candidate_id")
  done

  if ((${#missing_marker[@]} > 0)); then
    echo "[integration][INFO] additional control/performance candidates without marker (outside retrofit scope):"
    printf '  - %s\n' "${missing_marker[@]}"
  else
    echo "[integration][OK]   no additional labeled candidates missing policy marker"
  fi
}

check_e2e() {
  echo "[e2e] governance behavior self-check with template fixtures"

  local valid_payload
  valid_payload="$(cat <<'PAYLOAD'
[bd-2l7y baseline-budget] BASELINE_COMPARATOR: static blending.
BUDGETED_MODE_DEFAULTS: timeout_ms=150, max_memory_mb=64, retry_budget=1.
ON_EXHAUSTION: fallback to fast_only and emit reason_code=budget_exhausted.
SUCCESS_THRESHOLDS_AND_STOP_CONDITIONS: p95 improves >= 10%; stop if timeout breaches > 5%.
PAYLOAD
)"

  local invalid_payload
  invalid_payload="$(cat <<'PAYLOAD'
[bd-2l7y baseline-budget] BASELINE_COMPARATOR: static blending.
BUDGETED_MODE_DEFAULTS: timeout_ms=150, max_memory_mb=64.
PAYLOAD
)"

  validate_text_payload "e2e" "fixture-valid" "$valid_payload" "false"
  validate_text_payload "e2e" "fixture-invalid" "$invalid_payload" "true"
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

if ((FAILURES > 0)); then
  echo "Result: FAIL ($FAILURES violation(s))"
  exit 1
fi

echo "Result: PASS"
