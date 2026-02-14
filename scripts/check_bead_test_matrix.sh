#!/usr/bin/env bash
set -euo pipefail

MODE="all"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISSUES_FILE="$ROOT_DIR/.beads/issues.jsonl"
FAILURES=0

RULE_POLICY_UNIT="SDOC-POLICY-000"
RULE_MISSING_MATRIX_MARKER="SDOC-MATRIX-001"
RULE_MISSING_MATRIX_SECTION="SDOC-MATRIX-002"
RULE_MISSING_EXCEPTION_MARKER="SDOC-MATRIX-003"
RULE_SCANNER_PARSE_ERROR="SDOC-SCAN-001"
RULE_SCANNER_NORMALIZATION_ERROR="SDOC-SCAN-002"

DATA="[]"
NORMALIZED="[]"

report_finding() {
  local rule_id="$1"
  local severity="$2"
  local bead_id="$3"
  local message="$4"
  local fix_hint="$5"

  jq -cn \
    --arg rule_id "$rule_id" \
    --arg severity "$severity" \
    --arg bead_id "$bead_id" \
    --arg message "$message" \
    --arg fix_hint "$fix_hint" \
    '{
      rule_id: $rule_id,
      severity: $severity,
      bead_id: $bead_id,
      message: $message,
      fix_hint: $fix_hint
    }'
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

severity_for_missing_matrix() {
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
    report_finding \
      "$RULE_POLICY_UNIT" \
      "error" \
      "policy-selftest" \
      "$context expected '$expected' but found '$actual'" \
      "update classify_bead_class()/severity_for_missing_matrix() to match policy contract"
    FAILURES=$((FAILURES + 1))
  else
    echo "[unit][OK]   $context"
  fi
}

run_policy_unit_tests() {
  echo "[unit] validating lint policy classification and severity mappings"

  assert_eq "$(classify_bead_class "task" "")" "implementation" "task bead class"
  assert_eq "$(classify_bead_class "epic" "")" "program" "epic bead class"
  assert_eq "$(classify_bead_class "question" "")" "exploratory" "question bead class"
  assert_eq \
    "$(classify_bead_class "task" "ci,release-gate,lint")" \
    "gate" \
    "release-gate label bead class"

  assert_eq \
    "$(severity_for_missing_matrix "implementation")" \
    "error" \
    "implementation missing-matrix severity"
  assert_eq \
    "$(severity_for_missing_matrix "program")" \
    "warning" \
    "program missing-matrix severity"
  assert_eq \
    "$(severity_for_missing_matrix "exploratory")" \
    "info" \
    "exploratory missing-matrix severity"
}

load_issue_data_from_file() {
  local issues_file="$1"
  local line_no=0
  local line parsed
  local -a parsed_rows=()

  while IFS= read -r line || [[ -n "$line" ]]; do
    line_no=$((line_no + 1))

    if [[ -z "$line" ]]; then
      report_finding \
        "$RULE_SCANNER_PARSE_ERROR" \
        "error" \
        "line:$line_no" \
        "empty JSONL record" \
        "remove blank lines from issues.jsonl"
      FAILURES=$((FAILURES + 1))
      continue
    fi

    if ! parsed="$(jq -c '.' <<<"$line" 2>/dev/null)"; then
      report_finding \
        "$RULE_SCANNER_PARSE_ERROR" \
        "error" \
        "line:$line_no" \
        "malformed JSON record" \
        "ensure each line is a valid JSON object"
      FAILURES=$((FAILURES + 1))
      continue
    fi

    parsed="$(jq -c --argjson source_line "$line_no" '. + {__source_line: $source_line}' <<<"$parsed")"
    parsed_rows+=("$parsed")
  done < "$issues_file"

  if [[ "${#parsed_rows[@]}" -eq 0 ]]; then
    DATA="[]"
    return
  fi

  DATA="$(printf '%s\n' "${parsed_rows[@]}" | jq -cs '.')"
}

normalize_issue_model() {
  NORMALIZED="$(jq -c '
    def scalar_text:
      if . == null then ""
      elif type == "string" then .
      elif type == "number" or type == "boolean" then tostring
      elif type == "array" then map(
        if type == "string" then . else tostring end
      ) | join("\n")
      elif type == "object" then
        if has("text") then (.text | scalar_text)
        elif has("message") then (.message | scalar_text)
        else tojson
        end
      else tostring
      end;

    def normalize_labels:
      (.labels // []) as $labels
      | if ($labels | type) == "array" then $labels else [$labels] end
      | map(tostring)
      | sort
      | unique;

    def normalize_comments:
      (.comments // []) as $comments
      | if ($comments | type) == "array" then $comments else [$comments] end
      | map(
          if type == "object" then {
            author: ((.author // .created_by // "") | tostring),
            text: ((.text // .message // .body // "") | scalar_text),
            created_at: ((.created_at // "") | tostring)
          }
          elif type == "string" then {
            author: "",
            text: .,
            created_at: ""
          }
          else {
            author: "",
            text: tostring,
            created_at: ""
          }
          end
        );

    def normalize_dependencies:
      (.dependencies // []) as $deps
      | if ($deps | type) == "array" then $deps else [$deps] end
      | map(
          if type == "object" then {
            depends_on_id: ((.depends_on_id // .id // .dependency_id // "") | tostring),
            dep_type: ((.type // .dependency_type // "") | tostring)
          }
          elif type == "string" then {
            depends_on_id: .,
            dep_type: ""
          }
          else {
            depends_on_id: "",
            dep_type: ""
          }
          end
        );

    map(
      . as $issue
      | (normalize_comments) as $comments
      | (normalize_dependencies) as $deps
      | {
          id: ((.id // "") | tostring),
          issue_type: ((.issue_type // "task") | tostring),
          status: ((.status // "open") | tostring),
          priority: ((.priority // 4) | tonumber? // 4),
          title: ((.title // "") | scalar_text),
          description: ((.description // "") | scalar_text),
          acceptance_criteria: ((.acceptance_criteria // "") | scalar_text),
          notes: ((.notes // "") | scalar_text),
          labels: normalize_labels,
          comments: $comments,
          comment_text: ($comments | map(.text) | join("\n")),
          dependencies: $deps,
          source_line: (.__source_line // -1)
        }
    )
    | sort_by(.id, .source_line)
  ' <<<"$DATA")"
}

validate_normalized_model() {
  local errors
  errors="$(jq -r '
    .[] as $issue
    | (
        if ($issue.id | length) == 0 then
          "missing_id\tline:\($issue.source_line)\tid"
        else
          empty
        end
      ),
      (
        $issue.dependencies[]
        | select((.dep_type | length) > 0 and (.depends_on_id | length) == 0)
        | "missing_dep_target\t\($issue.id)\tdependencies.depends_on_id"
      )
  ' <<<"$NORMALIZED")"

  if [[ -z "$errors" ]]; then
    return
  fi

  while IFS=$'\t' read -r err_code bead_id field_path; do
    [[ -z "$err_code" ]] && continue
    report_finding \
      "$RULE_SCANNER_NORMALIZATION_ERROR" \
      "error" \
      "$bead_id" \
      "normalization error ($err_code) at $field_path" \
      "fix malformed record fields in .beads/issues.jsonl"
    FAILURES=$((FAILURES + 1))
  done <<<"$errors"
}

run_scanner_unit_tests() {
  echo "[unit] validating deterministic scanner normalization"

  local test_data test_normalized
  test_data="$(jq -cn '
    [
      {
        id: "bd-z",
        issue_type: "task",
        status: "open",
        labels: "lint",
        comments: "single comment string",
        dependencies: {depends_on_id: "bd-a", type: "blocks"},
        acceptance_criteria: ["a", "b"]
      },
      {
        issue_type: "feature",
        status: "open",
        comments: [{text: "missing id comment"}],
        dependencies: [{type: "blocks"}]
      },
      {
        id: "bd-a",
        issue_type: "epic",
        status: "in_progress",
        labels: ["release-gate", "ci"],
        comments: [{author: "ops", text: "obj comment"}],
        dependencies: ["bd-root"]
      }
    ]
  ')"

  test_normalized="$(jq -c '
    def scalar_text:
      if . == null then ""
      elif type == "string" then .
      elif type == "number" or type == "boolean" then tostring
      elif type == "array" then map(
        if type == "string" then . else tostring end
      ) | join("\n")
      elif type == "object" then
        if has("text") then (.text | scalar_text)
        elif has("message") then (.message | scalar_text)
        else tojson
        end
      else tostring
      end;
    def normalize_labels:
      (.labels // []) as $labels
      | if ($labels | type) == "array" then $labels else [$labels] end
      | map(tostring)
      | sort
      | unique;
    def normalize_comments:
      (.comments // []) as $comments
      | if ($comments | type) == "array" then $comments else [$comments] end
      | map(
          if type == "object" then {
            author: ((.author // .created_by // "") | tostring),
            text: ((.text // .message // .body // "") | scalar_text),
            created_at: ((.created_at // "") | tostring)
          }
          elif type == "string" then {author: "", text: ., created_at: ""}
          else {author: "", text: tostring, created_at: ""}
          end
        );
    def normalize_dependencies:
      (.dependencies // []) as $deps
      | if ($deps | type) == "array" then $deps else [$deps] end
      | map(
          if type == "object" then {
            depends_on_id: ((.depends_on_id // .id // .dependency_id // "") | tostring),
            dep_type: ((.type // .dependency_type // "") | tostring)
          }
          elif type == "string" then {depends_on_id: ., dep_type: ""}
          else {depends_on_id: "", dep_type: ""}
          end
        );
    map(
      . as $issue
      | (normalize_comments) as $comments
      | (normalize_dependencies) as $deps
      | {
          id: ((.id // "") | tostring),
          issue_type: ((.issue_type // "task") | tostring),
          status: ((.status // "open") | tostring),
          priority: ((.priority // 4) | tonumber? // 4),
          title: ((.title // "") | scalar_text),
          description: ((.description // "") | scalar_text),
          acceptance_criteria: ((.acceptance_criteria // "") | scalar_text),
          notes: ((.notes // "") | scalar_text),
          labels: normalize_labels,
          comments: $comments,
          comment_text: ($comments | map(.text) | join("\n")),
          dependencies: $deps,
          source_line: (.__source_line // -1)
        }
    )
    | sort_by(.id, .source_line)
  ' <<<"$test_data")"

  assert_eq "$(jq -r 'length' <<<"$test_normalized")" "3" "normalized fixture count"
  assert_eq "$(jq -r '.[0].id' <<<"$test_normalized")" "" "deterministic sort with malformed id first"
  assert_eq \
    "$(jq -r '.[] | select(.id == "bd-z") | .acceptance_criteria' <<<"$test_normalized")" \
    "a"$'\n'"b" \
    "array acceptance criteria stringification"
  assert_eq \
    "$(jq -r '.[] | select(.id == "bd-z") | .comments[0].text' <<<"$test_normalized")" \
    "single comment string" \
    "string comment normalization"
  assert_eq \
    "$(jq -r '.[] | select(.id == "bd-a") | .dependencies[0].depends_on_id' <<<"$test_normalized")" \
    "bd-root" \
    "string dependency normalization"
  assert_eq \
    "$(jq -r '.[] | select(.id == "") | .dependencies[0].depends_on_id' <<<"$test_normalized")" \
    "" \
    "malformed dependency target normalization"
}

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

load_issue_data_from_file "$ISSUES_FILE"
normalize_issue_model
validate_normalized_model

WAVE1=(bd-3un.31 bd-3un.32 bd-3un.40 bd-3un.52)
WAVE2_EXCEPTIONS=(bd-2hz.10 bd-2yu.8)

check_wave1_matrix() {
  local issue_id="$1"
  local comments
  comments="$(jq -r --arg id "$issue_id" '
    ([.[] | select(.id == $id)] | .[0].comment_text) // ""
  ' <<<"$NORMALIZED")"

  if ! grep -Fq "[bd-264r test-matrix] TEST_MATRIX" <<<"$comments"; then
    report_finding \
      "$RULE_MISSING_MATRIX_MARKER" \
      "error" \
      "$issue_id" \
      "missing TEST_MATRIX annotation marker" \
      "add '[bd-264r test-matrix] TEST_MATRIX' comment with full matrix template"
    FAILURES=$((FAILURES + 1))
    return
  fi

  local required_sections=("Unit tests:" "Integration tests:" "E2E tests:" "Performance" "Logs/artifacts")
  for section in "${required_sections[@]}"; do
    if ! grep -Fqi "$section" <<<"$comments"; then
      report_finding \
        "$RULE_MISSING_MATRIX_SECTION" \
        "error" \
        "$issue_id" \
        "missing required section '$section'" \
        "add '$section' entry under the TEST_MATRIX annotation"
      FAILURES=$((FAILURES + 1))
    fi
  done

  echo "[unit][OK]   $issue_id has explicit TEST_MATRIX annotation"
}

check_wave2_exception() {
  local issue_id="$1"
  local comments
  comments="$(jq -r --arg id "$issue_id" '
    ([.[] | select(.id == $id)] | .[0].comment_text) // ""
  ' <<<"$NORMALIZED")"

  if ! grep -Fq "[bd-264r test-matrix] EXCEPTION" <<<"$comments"; then
    report_finding \
      "$RULE_MISSING_EXCEPTION_MARKER" \
      "error" \
      "$issue_id" \
      "missing EXCEPTION annotation marker" \
      "add '[bd-264r test-matrix] EXCEPTION' with explicit rationale"
    FAILURES=$((FAILURES + 1))
  else
    echo "[unit][OK]   $issue_id has explicit EXCEPTION rationale"
  fi
}

check_unit() {
  run_policy_unit_tests
  run_scanner_unit_tests
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
    | ($issue.comment_text // "") as $comments
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
    | [
        $issue.id,
        $issue.issue_type,
        (($issue.labels // []) | join(","))
      ]
    | @tsv
  ' <<<"$NORMALIZED" | sort -u)"

  if [[ -n "$candidates" ]]; then
    echo "[integration][INFO] potential missing-matrix candidates (outside scoped retrofit):"
    while IFS=$'\t' read -r issue_id issue_type labels_csv; do
      local bead_class severity
      bead_class="$(classify_bead_class "$issue_type" "$labels_csv")"
      severity="$(severity_for_missing_matrix "$bead_class")"
      echo "  - $issue_id (class=$bead_class default_severity=$severity)"
    done <<<"$candidates"
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
