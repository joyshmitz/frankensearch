#!/usr/bin/env bash
set -euo pipefail

MODE="all"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISSUES_FILE="$ROOT_DIR/.beads/issues.jsonl"
FAILURES=0
FORMAT="${SELFDOC_LINT_FORMAT:-human}"
PHASE="${SELFDOC_LINT_PHASE:-default}"
MIN_FAIL_SEVERITY="${SELFDOC_LINT_MIN_FAIL_SEVERITY:-}"
REPORT_PATH=""
declare -a FINDINGS=()

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
Usage: scripts/check_bead_self_documentation.sh [--mode unit|integration|e2e|all] [--issues <path>] [--format human|json|ci] [--phase audit|default|strict] [--min-fail-severity error|warning|info] [--report-path <path>]

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

severity_rank() {
  local severity="${1:-error}"
  case "$severity" in
    error) echo 3 ;;
    warning) echo 2 ;;
    info) echo 1 ;;
    *) echo 3 ;;
  esac
}

phase_default_min_fail_severity() {
  case "$PHASE" in
    audit) echo "" ;;
    default) echo "error" ;;
    strict) echo "warning" ;;
    *)
      echo "ERROR: invalid phase '$PHASE' (expected audit|default|strict)" >&2
      exit 2
      ;;
  esac
}

record_finding() {
  local rule_id="$1"
  local severity="$2"
  local bead_id="$3"
  local message="$4"
  local fix_hint="$5"
  FINDINGS+=("${rule_id}"$'\t'"${severity}"$'\t'"${bead_id}"$'\t'"${message}"$'\t'"${fix_hint}")
}

finding_triggers_failure() {
  local finding_severity="$1"
  local threshold="$2"
  if [[ -z "$threshold" ]]; then
    return 1
  fi
  local finding_rank threshold_rank
  finding_rank="$(severity_rank "$finding_severity")"
  threshold_rank="$(severity_rank "$threshold")"
  if (( finding_rank >= threshold_rank )); then
    return 0
  fi
  return 1
}

emit_findings() {
  local finding
  if [[ "$FORMAT" == "human" ]]; then
    if ((${#FINDINGS[@]} == 0)); then
      echo "[report][OK] no lint findings"
      return
    fi
    echo "[report] lint findings:"
    for finding in "${FINDINGS[@]}"; do
      IFS=$'\t' read -r rule_id severity bead_id message fix_hint <<<"$finding"
      echo "  - [$rule_id][$severity] $bead_id: $message"
      echo "    fix_hint: $fix_hint"
    done
    return
  fi

  if [[ "$FORMAT" == "ci" ]]; then
    local level
    for finding in "${FINDINGS[@]}"; do
      IFS=$'\t' read -r rule_id severity bead_id message fix_hint <<<"$finding"
      level="$severity"
      if [[ "$level" == "info" ]]; then
        level="notice"
      fi
      echo "::${level} title=${rule_id},bead=${bead_id}::${message} | fix_hint=${fix_hint}"
    done
    return
  fi

  # json (default fallback for non-human/non-ci after validation)
  for finding in "${FINDINGS[@]}"; do
    IFS=$'\t' read -r rule_id severity bead_id message fix_hint <<<"$finding"
    jq -cn \
      --arg rule_id "$rule_id" \
      --arg severity "$severity" \
      --arg bead_id "$bead_id" \
      --arg message "$message" \
      --arg fix_hint "$fix_hint" \
      '{rule_id:$rule_id,severity:$severity,bead_id:$bead_id,message:$message,fix_hint:$fix_hint}'
  done
}

write_report_if_requested() {
  if [[ -z "$REPORT_PATH" ]]; then
    return
  fi
  local tmp
  tmp="$(mktemp)"
  if ((${#FINDINGS[@]} == 0)); then
    echo "[]" > "$tmp"
  else
    for finding in "${FINDINGS[@]}"; do
      IFS=$'\t' read -r rule_id severity bead_id message fix_hint <<<"$finding"
      jq -cn \
        --arg rule_id "$rule_id" \
        --arg severity "$severity" \
        --arg bead_id "$bead_id" \
        --arg message "$message" \
        --arg fix_hint "$fix_hint" \
        '{rule_id:$rule_id,severity:$severity,bead_id:$bead_id,message:$message,fix_hint:$fix_hint}' >> "$tmp"
    done
    jq -s '.' "$tmp" > "${tmp}.json"
    mv "${tmp}.json" "$tmp"
  fi
  mkdir -p "$(dirname "$REPORT_PATH")"
  mv "$tmp" "$REPORT_PATH"
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
    --format)
      FORMAT="${2:-}"
      shift 2
      ;;
    --phase)
      PHASE="${2:-}"
      shift 2
      ;;
    --min-fail-severity)
      MIN_FAIL_SEVERITY="${2:-}"
      shift 2
      ;;
    --report-path)
      REPORT_PATH="${2:-}"
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

case "$FORMAT" in
  human|json|ci) ;;
  *)
    echo "ERROR: invalid format '$FORMAT' (expected human|json|ci)" >&2
    exit 2
    ;;
esac

if [[ -z "$MIN_FAIL_SEVERITY" ]]; then
  MIN_FAIL_SEVERITY="$(phase_default_min_fail_severity)"
fi

if [[ -n "$MIN_FAIL_SEVERITY" ]]; then
  case "$MIN_FAIL_SEVERITY" in
    error|warning|info) ;;
    *)
      echo "ERROR: invalid min fail severity '$MIN_FAIL_SEVERITY' (expected error|warning|info)" >&2
      exit 2
      ;;
  esac
fi

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
  echo "[integration] evaluating open/in-progress self-doc lint findings"

  local candidates
  candidates="$(jq -r --arg rationale_tag "$RATIONALE_TAG" --arg evidence_tag "$EVIDENCE_TAG" '
    .[]
    | select((.status == "open" or .status == "in_progress"))
    | . as $issue
    | (($issue.comments // []) | map(.text // "") | join("\n")) as $comments
    | ($comments | test("(^|\\n)\\Q" + $rationale_tag + "\\E(\\n|$)")) as $has_rationale
    | ($comments | test("(^|\\n)\\Q" + $evidence_tag + "\\E(\\n|$)")) as $has_evidence
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

  local finding_count=0
  local issue_id issue_type labels_csv has_rationale has_evidence bead_class severity comments
  local missing_field fields_csv message fix_hint
  local -a missing_fields=()

  if [[ -n "$candidates" ]]; then
    while IFS=$'\t' read -r issue_id issue_type labels_csv has_rationale has_evidence; do
      bead_class="$(classify_bead_class "$issue_type" "$labels_csv")"
      severity="$(severity_for_missing_anchor "$bead_class")"
      comments="$(issue_comments "$issue_id")"

      if [[ "$has_rationale" != "yes" ]]; then
        message="missing rationale anchor $RATIONALE_TAG"
        fix_hint="add rationale anchor with PROBLEM_CONTEXT/WHY_NOW/SCOPE_BOUNDARY/PRIMARY_SURFACES"
        record_finding "$RULE_MISSING_RATIONALE_TAG" "$severity" "$issue_id" "$message" "$fix_hint"
        finding_count=$((finding_count + 1))
        if finding_triggers_failure "$severity" "$MIN_FAIL_SEVERITY"; then
          FAILURES=$((FAILURES + 1))
        fi
      else
        mapfile -t missing_fields < <(missing_fields_for_text "$comments" "${RATIONALE_FIELDS[@]}")
        if ((${#missing_fields[@]} > 0)); then
          fields_csv="$(IFS=,; echo "${missing_fields[*]}")"
          message="rationale anchor missing field(s): ${fields_csv}"
          fix_hint="fill all rationale fields: PROBLEM_CONTEXT, WHY_NOW, SCOPE_BOUNDARY, PRIMARY_SURFACES"
          record_finding "$RULE_MISSING_RATIONALE_FIELDS" "$severity" "$issue_id" "$message" "$fix_hint"
          finding_count=$((finding_count + 1))
          if finding_triggers_failure "$severity" "$MIN_FAIL_SEVERITY"; then
            FAILURES=$((FAILURES + 1))
          fi
        fi
      fi

      if [[ "$has_evidence" != "yes" ]]; then
        message="missing evidence anchor $EVIDENCE_TAG"
        fix_hint="add evidence anchor with UNIT_TESTS/INTEGRATION_TESTS/E2E_TESTS/PERFORMANCE_VALIDATION/LOGGING_ARTIFACTS"
        record_finding "$RULE_MISSING_EVIDENCE_TAG" "$severity" "$issue_id" "$message" "$fix_hint"
        finding_count=$((finding_count + 1))
        if finding_triggers_failure "$severity" "$MIN_FAIL_SEVERITY"; then
          FAILURES=$((FAILURES + 1))
        fi
      else
        mapfile -t missing_fields < <(missing_fields_for_text "$comments" "${EVIDENCE_FIELDS[@]}")
        if ((${#missing_fields[@]} > 0)); then
          fields_csv="$(IFS=,; echo "${missing_fields[*]}")"
          message="evidence anchor missing field(s): ${fields_csv}"
          fix_hint="fill all evidence fields: UNIT_TESTS, INTEGRATION_TESTS, E2E_TESTS, PERFORMANCE_VALIDATION, LOGGING_ARTIFACTS"
          record_finding "$RULE_MISSING_EVIDENCE_FIELDS" "$severity" "$issue_id" "$message" "$fix_hint"
          finding_count=$((finding_count + 1))
          if finding_triggers_failure "$severity" "$MIN_FAIL_SEVERITY"; then
            FAILURES=$((FAILURES + 1))
          fi
        fi
      fi
    done <<<"$candidates"
  fi

  if ((finding_count == 0)); then
    echo "[integration][OK]   no self-doc lint findings"
  else
    echo "[integration][INFO] recorded $finding_count finding(s) (phase=$PHASE min_fail_severity=${MIN_FAIL_SEVERITY:-none})"
  fi

  local -a exception_ids=()
  mapfile -t exception_ids < <(jq -r --arg tag "$EXCEPTION_TAG" '
    .[]
    | . as $issue
    | (($issue.comments // []) | map(.text // "") | join("\n")) as $comments
    | select($comments | test("(^|\\n)\\Q" + $tag + "\\E(\\n|$)"))
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

write_report_if_requested
emit_findings

if ((FAILURES > 0)); then
  echo "Result: FAIL ($FAILURES violation(s))"
  exit 1
fi

echo "Result: PASS"
