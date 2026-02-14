#!/usr/bin/env bash
set -euo pipefail

MODE="all"
SCOPE="active"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISSUES_FILE="$ROOT_DIR/.beads/issues.jsonl"
REGISTRY_FILE="$ROOT_DIR/docs/crate-placement-registry.json"
SCHEMA_FILE="$ROOT_DIR/schemas/crate-placement-registry-v1.schema.json"
FAILURES=0
WARNINGS=0

usage() {
  cat <<USAGE
Usage: scripts/check_crate_placement_registry.sh [OPTIONS]

Validates the crate-placement registry for bead bd-33iv.

Options:
  --mode <unit|integration|all>   Check mode (default: all)
  --scope <active|changed|all>    Integration scope (default: active)
  --issues <path>                 Path to issues.jsonl (default: .beads/issues.jsonl)
  --registry <path>               Path to registry JSON (default: docs/crate-placement-registry.json)
  --schema <path>                 Path to registry schema (default: schemas/crate-placement-registry-v1.schema.json)
  -h, --help                      Show this help
USAGE
}

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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --scope)
      SCOPE="${2:-}"
      shift 2
      ;;
    --issues)
      ISSUES_FILE="${2:-}"
      shift 2
      ;;
    --registry)
      REGISTRY_FILE="${2:-}"
      shift 2
      ;;
    --schema)
      SCHEMA_FILE="${2:-}"
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

case "$SCOPE" in
  active|changed|all) ;;
  *)
    echo "ERROR: invalid scope '$SCOPE' (expected active|changed|all)" >&2
    exit 2
    ;;
esac

for path in "$ISSUES_FILE" "$REGISTRY_FILE" "$SCHEMA_FILE"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: file not found: $path" >&2
    exit 2
  fi
done

REGISTRY_JSON="$(jq -c '.' "$REGISTRY_FILE")"
ISSUES_JSON="$(jq -cs '.' "$ISSUES_FILE")"
TODAY="$(date +%F)"

schema_validate() {
  if ! command -v jsonschema >/dev/null 2>&1; then
    echo "[unit][WARN] jsonschema command not found; skipping schema validation"
    WARNINGS=$((WARNINGS + 1))
    return
  fi

  if jsonschema -i "$REGISTRY_FILE" "$SCHEMA_FILE" >/dev/null 2>&1; then
    echo "[unit][OK]   schema validation passed"
  else
    report_finding \
      "PLACEMENT-SCHEMA-001" \
      "error" \
      "registry" \
      "registry does not match schema" \
      "run: jsonschema -i $REGISTRY_FILE $SCHEMA_FILE"
    FAILURES=$((FAILURES + 1))
  fi
}

check_rule_uniqueness() {
  local duplicate_ids duplicate_patterns
  duplicate_ids="$(jq -r '.rules | group_by(.rule_id)[] | select(length > 1) | .[0].rule_id' <<<"$REGISTRY_JSON" || true)"
  duplicate_patterns="$(jq -r '.rules | group_by(.id_pattern)[] | select(length > 1) | .[0].id_pattern' <<<"$REGISTRY_JSON" || true)"

  if [[ -n "$duplicate_ids" ]]; then
    while IFS= read -r dup; do
      [[ -z "$dup" ]] && continue
      report_finding \
        "PLACEMENT-UNIT-002" \
        "error" \
        "$dup" \
        "duplicate rule_id detected in registry" \
        "ensure every rule_id is globally unique"
      FAILURES=$((FAILURES + 1))
    done <<<"$duplicate_ids"
  else
    echo "[unit][OK]   rule_id values are unique"
  fi

  if [[ -n "$duplicate_patterns" ]]; then
    while IFS= read -r dup; do
      [[ -z "$dup" ]] && continue
      report_finding \
        "PLACEMENT-UNIT-003" \
        "error" \
        "$dup" \
        "duplicate id_pattern detected in registry" \
        "deduplicate or merge overlapping rules"
      FAILURES=$((FAILURES + 1))
    done <<<"$duplicate_patterns"
  else
    echo "[unit][OK]   id_pattern values are unique"
  fi
}

check_rule_targets() {
  local invalid
  invalid="$(jq -r '
    .rules[]
    | select((.target_paths | length) == 0)
    | .rule_id
  ' <<<"$REGISTRY_JSON" || true)"

  if [[ -n "$invalid" ]]; then
    while IFS= read -r rule_id; do
      [[ -z "$rule_id" ]] && continue
      report_finding \
        "PLACEMENT-UNIT-004" \
        "error" \
        "$rule_id" \
        "rule has no target_paths" \
        "set at least one target_paths entry"
      FAILURES=$((FAILURES + 1))
    done <<<"$invalid"
  else
    echo "[unit][OK]   all rules include non-empty target_paths"
  fi
}

check_conflict_deadlines() {
  local conflicts
  conflicts="$(jq -r '
    .rules[]
    | select(.placement_status == "conflict" or .placement_status == "unknown")
    | [.rule_id, .resolution_deadline, .resolution_owner, .blocking_impact]
    | @tsv
  ' <<<"$REGISTRY_JSON" || true)"

  if [[ -z "$conflicts" ]]; then
    echo "[unit][OK]   no unresolved placement conflicts registered"
    return
  fi

  while IFS=$'\t' read -r rule_id deadline owner impact; do
    [[ -z "$rule_id" ]] && continue
    if [[ "$deadline" < "$TODAY" ]]; then
      report_finding \
        "PLACEMENT-UNIT-005" \
        "error" \
        "$rule_id" \
        "unresolved placement conflict deadline has passed ($deadline)" \
        "resolve conflict or set a new realistic deadline with updated owner"
      FAILURES=$((FAILURES + 1))
    else
      report_finding \
        "PLACEMENT-UNIT-006" \
        "warning" \
        "$rule_id" \
        "unresolved placement conflict tracked (owner=$owner, impact=$impact, due=$deadline)" \
        "close conflict by converting placement_status to resolved once placement is finalized"
      WARNINGS=$((WARNINGS + 1))
    fi
  done <<<"$conflicts"
}

active_implementation_ids() {
  jq -r '
    .[]
    | select((.status == "open" or .status == "in_progress")
      and (.issue_type == "task" or .issue_type == "feature" or .issue_type == "bug"))
    | .id
  ' <<<"$ISSUES_JSON" | sort -u
}

all_implementation_ids() {
  jq -r '
    .[]
    | select(.issue_type == "task" or .issue_type == "feature" or .issue_type == "bug")
    | .id
  ' <<<"$ISSUES_JSON" | sort -u
}

changed_implementation_ids() {
  local diff_ids
  diff_ids="$(git -C "$ROOT_DIR" diff --unified=0 -- "$ISSUES_FILE" \
    | sed -n 's/^[+-]{"id":"\(bd-[^"]*\)".*/\1/p' \
    | sort -u || true)"

  if [[ -z "$diff_ids" ]]; then
    active_implementation_ids
    return
  fi

  jq -r --argjson changed "$(printf '%s\n' "$diff_ids" | jq -R . | jq -s .)" '
    .[]
    | select((.issue_type == "task" or .issue_type == "feature" or .issue_type == "bug")
      and (.id as $id | $changed | index($id)))
    | .id
  ' <<<"$ISSUES_JSON" | sort -u
}

scope_ids() {
  case "$SCOPE" in
    active) active_implementation_ids ;;
    changed) changed_implementation_ids ;;
    all) all_implementation_ids ;;
  esac
}

matching_rule_lines() {
  local bead_id="$1"
  jq -r --arg bead_id "$bead_id" '
    .rules[]
    | . as $rule
    | select($bead_id | test($rule.id_pattern))
    | [.rule_id, .placement_status]
    | @tsv
  ' <<<"$REGISTRY_JSON"
}

check_registry_coverage() {
  local ids checked=0 mapped=0 unresolved=0 missing=0 duplicate=0
  ids="$(scope_ids)"

  if [[ -z "$ids" ]]; then
    echo "[integration][WARN] no implementation bead IDs found for scope '$SCOPE'"
    WARNINGS=$((WARNINGS + 1))
    return
  fi

  while IFS= read -r bead_id; do
    [[ -z "$bead_id" ]] && continue
    checked=$((checked + 1))
    local matches match_count
    matches="$(matching_rule_lines "$bead_id")"
    match_count="$(grep -c '.' <<<"$matches" || true)"

    if [[ "$match_count" -eq 0 ]]; then
      report_finding \
        "PLACEMENT-INTEGRATION-101" \
        "error" \
        "$bead_id" \
        "no crate-placement rule matches this bead" \
        "add a rule or override in docs/crate-placement-registry.json"
      FAILURES=$((FAILURES + 1))
      missing=$((missing + 1))
      continue
    fi

    if [[ "$match_count" -gt 1 ]]; then
      report_finding \
        "PLACEMENT-INTEGRATION-102" \
        "error" \
        "$bead_id" \
        "bead matches multiple placement rules (duplicate placement drift)" \
        "narrow id_pattern expressions or add an explicit override"
      FAILURES=$((FAILURES + 1))
      duplicate=$((duplicate + 1))
      continue
    fi

    local rule_id status
    rule_id="$(cut -f1 <<<"$matches")"
    status="$(cut -f2 <<<"$matches")"
    mapped=$((mapped + 1))

    if [[ "$status" != "resolved" ]]; then
      unresolved=$((unresolved + 1))
      if [[ "$SCOPE" == "changed" ]]; then
        report_finding \
          "PLACEMENT-INTEGRATION-103" \
          "warning" \
          "$bead_id" \
          "changed bead maps to unresolved placement rule ($rule_id)" \
          "resolve placement_status before merge when practical; unresolved conflicts remain explicitly tracked"
        WARNINGS=$((WARNINGS + 1))
      else
        report_finding \
          "PLACEMENT-INTEGRATION-104" \
          "warning" \
          "$bead_id" \
          "bead maps to unresolved placement rule ($rule_id)" \
          "track conflict to closure in registry resolution fields"
        WARNINGS=$((WARNINGS + 1))
      fi
    fi
  done <<<"$ids"

  echo "[integration] scope=$SCOPE checked=$checked mapped=$mapped missing=$missing duplicate=$duplicate unresolved=$unresolved"
}

if [[ "$MODE" == "unit" || "$MODE" == "all" ]]; then
  echo "[unit] running schema and rule consistency checks"
  schema_validate
  check_rule_uniqueness
  check_rule_targets
  check_conflict_deadlines
fi

if [[ "$MODE" == "integration" || "$MODE" == "all" ]]; then
  echo "[integration] validating bead-to-placement coverage"
  check_registry_coverage
fi

if [[ "$FAILURES" -gt 0 ]]; then
  echo "Result: FAIL (failures=$FAILURES warnings=$WARNINGS)"
  exit 1
fi

echo "Result: PASS (failures=0 warnings=$WARNINGS)"
