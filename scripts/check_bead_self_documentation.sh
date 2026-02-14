#!/usr/bin/env bash
set -euo pipefail

MODE="all"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISSUES_FILE="$ROOT_DIR/.beads/issues.jsonl"
FAILURES=0

RATIONALE_TAG="[bd-3qwe self-doc] RATIONALE"
EVIDENCE_TAG="[bd-3qwe self-doc] EVIDENCE"
EXCEPTION_TAG="[bd-3qwe self-doc] EXCEPTION"

RULE_POLICY_SELFTEST="SDOC-RUBRIC-000"
RULE_MISSING_RATIONALE_TAG="SDOC-RUBRIC-001"
RULE_MISSING_RATIONALE_FIELDS="SDOC-RUBRIC-002"
RULE_MISSING_EVIDENCE_TAG="SDOC-RUBRIC-003"
RULE_MISSING_EVIDENCE_FIELDS="SDOC-RUBRIC-004"
RULE_BAD_EXCEPTION_PAYLOAD="SDOC-RUBRIC-005"

RATIONALE_TARGETS=(
  bd-3qwe.1
)

EVIDENCE_TARGETS=(
  bd-3qwe.1
)

RATIONALE_FIELDS=(
  PROBLEM_CONTEXT
  WHY_NOW
  SCOPE_BOUNDARY
  PRIMARY_SURFACES
)

EVIDENCE_FIELDS=(
  UNIT_TESTS
  INTEGRATION_TESTS
  E2E_TESTS
  PERFORMANCE_VALIDATION
  LOGGING_ARTIFACTS
)

EXCEPTION_FIELDS=(
  RULE_ID
  OWNER
  JUSTIFICATION
  EXPIRES_ON
  FOLLOW_UP_BEAD
  APPROVED_BY
)

usage() {
  cat <<USAGE
Usage: scripts/check_bead_self_documentation.sh [--mode unit|integration|e2e|all] [--issues <path>]

Checks bead self-documentation rationale/evidence/exception policy anchors for bd-3qwe.1.
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

classify_bead_class() {
  local issue_type="$1"
  local labels_csv="${2:-}"

  case ",$labels_csv," in
    *,release-gate,*|*,gate,*)
      echo "gate"
      return
      ;;
  esac

  case "$issue_type" in
    task|feature|bug) echo "implementation" ;;
    epic) echo "program" ;;
    question|docs) echo "exploratory" ;;
    *) echo "implementation" ;;
  esac
}

severity_for_missing_anchor() {
  local bead_class="$1"
  case "$bead_class" in
    implementation|gate) echo "error" ;;
    program) echo "warning" ;;
    exploratory) echo "info" ;;
    *) echo "error" ;;
  esac
}

assert_eq() {
  local actual="$1"
  local expected="$2"
  local context="$3"
  if [[ "$actual" != "$expected" ]]; then
    echo "[$RULE_POLICY_SELFTEST][FAIL] $context expected '$expected' but found '$actual'"
    FAILURES=$((FAILURES + 1))
  else
    echo "[unit][OK]   $context"
  fi
}

issue_exists() {
  local issue_id="$1"
  jq -e --arg id "$issue_id" 'any(.[]; .id == $id)' <<<"$DATA" >/dev/null
}

issue_comments() {
  local issue_id="$1"
  jq -r --arg id "$issue_id" '
    ([.[] | select(.id == $id)] | .[0]) as $issue
    | (($issue.comments // []) | map(.text // "") | join("\n"))
  ' <<<"$DATA"
}

missing_fields_for_text() {
  local text="$1"
  shift
  local field
  for field in "$@"; do
    if ! grep -Fq "${field}:" <<<"$text"; then
      echo "$field"
    fi
  done
}

validate_anchor_payload() {
  local scope="$1"
  local subject="$2"
  local text="$3"
  local tag="$4"
  local missing_rule="$5"
  local fields_rule="$6"
  local allow_missing="$7"
  shift 7
  local -a required_fields=("$@")
  local -a missing=()

  if ! grep -Fq "$tag" <<<"$text"; then
    missing+=("$tag")
  fi

  local field
  for field in $(missing_fields_for_text "$text" "${required_fields[@]}"); do
    missing+=("$field")
  done

  if [[ "$allow_missing" == "true" ]]; then
    if ((${#missing[@]} == 0)); then
      echo "[$scope][FAIL] $subject expected policy violation but none was detected"
      FAILURES=$((FAILURES + 1))
    else
      echo "[$scope][OK]   $subject violation detected as expected (${missing[*]})"
    fi
    return
  fi

  if ! grep -Fq "$tag" <<<"$text"; then
    echo "[$missing_rule][FAIL] $subject missing anchor: $tag"
    FAILURES=$((FAILURES + 1))
  fi

  local missing_field
  for missing_field in "${required_fields[@]}"; do
    if ! grep -Fq "${missing_field}:" <<<"$text"; then
      echo "[$fields_rule][FAIL] $subject missing field: ${missing_field}:"
      FAILURES=$((FAILURES + 1))
    fi
  done

  if grep -Fq "$tag" <<<"$text"; then
    if [[ "$tag" == "$RATIONALE_TAG" || "$tag" == "$EVIDENCE_TAG" ]]; then
      if ! grep -Eq 'N/A[[:space:]]*-[[:space:]]*' <<<"$text"; then
        :
      fi
    fi
  fi

  if [[ "$allow_missing" == "false" ]] && ((${#missing[@]} == 0)); then
    echo "[$scope][OK]   $subject has complete anchor payload"
  fi
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

check_policy_selftests() {
  echo "[unit] validating class and severity policy mappings"
  assert_eq "$(classify_bead_class "task" "")" "implementation" "task bead class"
  assert_eq "$(classify_bead_class "epic" "")" "program" "epic bead class"
  assert_eq "$(classify_bead_class "question" "")" "exploratory" "question bead class"
  assert_eq "$(classify_bead_class "task" "ci,release-gate")" "gate" "release-gate label class"
  assert_eq "$(severity_for_missing_anchor "implementation")" "error" "implementation severity"
  assert_eq "$(severity_for_missing_anchor "program")" "warning" "program severity"
  assert_eq "$(severity_for_missing_anchor "exploratory")" "info" "exploratory severity"
}

check_unit() {
  check_policy_selftests
  echo "[unit] validating scoped anchor targets"

  local issue_id comments
  for issue_id in "${RATIONALE_TARGETS[@]}"; do
    if ! issue_exists "$issue_id"; then
      echo "[unit][FAIL] $issue_id does not exist in issues file"
      FAILURES=$((FAILURES + 1))
      continue
    fi
    comments="$(issue_comments "$issue_id")"
    validate_anchor_payload \
      "unit" \
      "$issue_id rationale" \
      "$comments" \
      "$RATIONALE_TAG" \
      "$RULE_MISSING_RATIONALE_TAG" \
      "$RULE_MISSING_RATIONALE_FIELDS" \
      "false" \
      "${RATIONALE_FIELDS[@]}"
  done

  for issue_id in "${EVIDENCE_TARGETS[@]}"; do
    if ! issue_exists "$issue_id"; then
      echo "[unit][FAIL] $issue_id does not exist in issues file"
      FAILURES=$((FAILURES + 1))
      continue
    fi
    comments="$(issue_comments "$issue_id")"
    validate_anchor_payload \
      "unit" \
      "$issue_id evidence" \
      "$comments" \
      "$EVIDENCE_TAG" \
      "$RULE_MISSING_EVIDENCE_TAG" \
      "$RULE_MISSING_EVIDENCE_FIELDS" \
      "false" \
      "${EVIDENCE_FIELDS[@]}"
  done
}

check_integration() {
  echo "[integration] surfacing open beads missing self-doc anchors"

  local candidates
  candidates="$(jq -r --arg rationale_tag "$RATIONALE_TAG" --arg evidence_tag "$EVIDENCE_TAG" '
    .[]
    | select((.status == "open" or .status == "in_progress"))
    | . as $issue
    | (($issue.comments // []) | map(.text // "") | join("\n")) as $comments
    | ($comments | contains($rationale_tag)) as $has_rationale
    | ($comments | contains($evidence_tag)) as $has_evidence
    | select(($has_rationale | not) or ($has_evidence | not))
    | [
        ($issue.id // ""),
        ($issue.issue_type // ""),
        (($issue.labels // []) | join(",")),
        (if $has_rationale then "yes" else "no" end),
        (if $has_evidence then "yes" else "no" end)
      ]
    | @tsv
  ' <<<"$DATA" | sort -u)"

  if [[ -n "$candidates" ]]; then
    echo "[integration][INFO] candidate backlog items missing anchors:"
    while IFS=$'\t' read -r issue_id issue_type labels_csv has_rationale has_evidence; do
      local bead_class severity
      bead_class="$(classify_bead_class "$issue_type" "$labels_csv")"
      severity="$(severity_for_missing_anchor "$bead_class")"
      echo "  - $issue_id (class=$bead_class default_severity=$severity rationale=$has_rationale evidence=$has_evidence)"
    done <<<"$candidates"
  else
    echo "[integration][OK]   no missing-anchor candidates detected"
  fi

  local -a exception_ids=()
  mapfile -t exception_ids < <(jq -r --arg tag "$EXCEPTION_TAG" '
    .[]
    | . as $issue
    | (($issue.comments // []) | map(.text // "") | join("\n")) as $comments
    | select($comments | contains($tag))
    | .id
  ' <<<"$DATA" | sort -u)

  local issue_id comments
  for issue_id in "${exception_ids[@]}"; do
    comments="$(issue_comments "$issue_id")"
    validate_anchor_payload \
      "integration" \
      "$issue_id exception" \
      "$comments" \
      "$EXCEPTION_TAG" \
      "$RULE_BAD_EXCEPTION_PAYLOAD" \
      "$RULE_BAD_EXCEPTION_PAYLOAD" \
      "false" \
      "${EXCEPTION_FIELDS[@]}"
  done
}

check_e2e() {
  echo "[e2e] validating fixture payload behavior"

  local valid_rationale
  valid_rationale="$(cat <<'PAYLOAD'
[bd-3qwe self-doc] RATIONALE
PROBLEM_CONTEXT: backlog review lacks explicit why/where details.
WHY_NOW: policy lane must be stable before CI hardening.
SCOPE_BOUNDARY: rubric/policy/lint only.
PRIMARY_SURFACES: docs/*.md, scripts/check_bead_*.sh, .beads/issues.jsonl comments.
PAYLOAD
)"

  local invalid_rationale
  invalid_rationale="$(cat <<'PAYLOAD'
[bd-3qwe self-doc] RATIONALE
PROBLEM_CONTEXT: missing required fields.
PAYLOAD
)"

  local valid_evidence
  valid_evidence="$(cat <<'PAYLOAD'
[bd-3qwe self-doc] EVIDENCE
UNIT_TESTS: scripts/check_bead_self_documentation.sh --mode unit
INTEGRATION_TESTS: scripts/check_bead_self_documentation.sh --mode integration
E2E_TESTS: scripts/check_bead_self_documentation.sh --mode e2e
PERFORMANCE_VALIDATION: N/A - policy checker (no runtime hot path)
LOGGING_ARTIFACTS: JSON findings from checker output
PAYLOAD
)"

  local invalid_exception
  invalid_exception="$(cat <<'PAYLOAD'
[bd-3qwe self-doc] EXCEPTION
RULE_ID: SDOC-RUBRIC-003
OWNER: CrimsonBay
JUSTIFICATION: temporary waiver
PAYLOAD
)"

  validate_anchor_payload \
    "e2e" \
    "fixture-rationale-valid" \
    "$valid_rationale" \
    "$RATIONALE_TAG" \
    "$RULE_MISSING_RATIONALE_TAG" \
    "$RULE_MISSING_RATIONALE_FIELDS" \
    "false" \
    "${RATIONALE_FIELDS[@]}"

  validate_anchor_payload \
    "e2e" \
    "fixture-rationale-invalid" \
    "$invalid_rationale" \
    "$RATIONALE_TAG" \
    "$RULE_MISSING_RATIONALE_TAG" \
    "$RULE_MISSING_RATIONALE_FIELDS" \
    "true" \
    "${RATIONALE_FIELDS[@]}"

  validate_anchor_payload \
    "e2e" \
    "fixture-evidence-valid" \
    "$valid_evidence" \
    "$EVIDENCE_TAG" \
    "$RULE_MISSING_EVIDENCE_TAG" \
    "$RULE_MISSING_EVIDENCE_FIELDS" \
    "false" \
    "${EVIDENCE_FIELDS[@]}"

  validate_anchor_payload \
    "e2e" \
    "fixture-exception-invalid" \
    "$invalid_exception" \
    "$EXCEPTION_TAG" \
    "$RULE_BAD_EXCEPTION_PAYLOAD" \
    "$RULE_BAD_EXCEPTION_PAYLOAD" \
    "true" \
    "${EXCEPTION_FIELDS[@]}"
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
