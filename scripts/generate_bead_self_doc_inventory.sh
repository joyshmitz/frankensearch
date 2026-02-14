#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATE_UTC="$(date -u +%F)"
DATE="$DATE_UTC"
INSIGHTS_PATH=""

usage() {
  cat <<'USAGE'
Usage: scripts/generate_bead_self_doc_inventory.sh [--date YYYY-MM-DD] [--insights /path/to/bv_insights.json]

Generates:
  docs/bead-self-documentation-debt-inventory-<date>.json

Inputs:
  - .beads/issues.jsonl
  - bv --robot-insights output (generated automatically unless --insights is provided)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --date)
      DATE="${2:-}"
      shift 2
      ;;
    --insights)
      INSIGHTS_PATH="${2:-}"
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

if [[ -z "$DATE" ]]; then
  echo "ERROR: --date cannot be empty" >&2
  exit 2
fi

OUT_JSON="$ROOT_DIR/docs/bead-self-documentation-debt-inventory-${DATE}.json"
ISSUES_FILE="$ROOT_DIR/.beads/issues.jsonl"

if [[ ! -f "$ISSUES_FILE" ]]; then
  echo "ERROR: missing $ISSUES_FILE" >&2
  exit 2
fi

TMP_INSIGHTS=""
cleanup() {
  if [[ -n "$TMP_INSIGHTS" && -f "$TMP_INSIGHTS" ]]; then
    rm -f "$TMP_INSIGHTS"
  fi
}
trap cleanup EXIT

if [[ -n "$INSIGHTS_PATH" ]]; then
  if [[ ! -f "$INSIGHTS_PATH" ]]; then
    echo "ERROR: --insights file not found: $INSIGHTS_PATH" >&2
    exit 2
  fi
  USE_INSIGHTS="$INSIGHTS_PATH"
else
  TMP_INSIGHTS="$(mktemp)"
  (cd "$ROOT_DIR" && bv --robot-insights > "$TMP_INSIGHTS")
  USE_INSIGHTS="$TMP_INSIGHTS"
fi

GENERATED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

jq -n \
  --arg generated_at "$GENERATED_AT" \
  --arg inventory_date "$DATE" \
  --slurpfile issues "$ISSUES_FILE" \
  --slurpfile insights "$USE_INSIGHTS" \
  '
  def class_of($issue):
    if (($issue.labels // []) | map(ascii_downcase) | index("release-gate")) != null
       or (($issue.labels // []) | map(ascii_downcase) | index("gate")) != null
      then "gate"
    elif ($issue.issue_type == "epic") then "program"
    elif ($issue.issue_type == "question" or $issue.issue_type == "docs") then "exploratory"
    else "implementation"
    end;
  def severity_of($class):
    if ($class == "implementation" or $class == "gate") then "error"
    elif ($class == "program") then "warning"
    else "info"
    end;
  def severity_weight($sev):
    if $sev == "error" then 2 elif $sev == "warning" then 1 else 0 end;
  def priority_weight($p):
    if $p == 0 then 3 elif $p == 1 then 2 elif $p == 2 then 1 else 0 end;
  def bucket_bet($v):
    if $v >= 100 then 3 elif $v >= 10 then 2 elif $v >= 1 then 1 else 0 end;
  def bucket_cp($v):
    if $v >= 5 then 2 elif $v >= 1 then 1 else 0 end;
  def owner_suggestion($issue; $class):
    if (($issue.assignee // "") | type) == "string" and (($issue.assignee // "") | length) > 0
      then ($issue.assignee // "")
    elif $class == "gate" then "release-owner"
    elif $class == "program" then "program-owner"
    elif $class == "exploratory" then "docs-owner"
    else "component-owner"
    end;
  def owner_basis($issue):
    if (($issue.assignee // "") | type) == "string" and (($issue.assignee // "") | length) > 0
      then "issue.assignee"
    else "class-fallback"
    end;

  ($insights[0].full_stats.pagerank // {}) as $pr
  | ($insights[0].full_stats.betweenness // {}) as $bw
  | ($insights[0].full_stats.critical_path_score // {}) as $cp
  | {
      generated_at: $generated_at,
      inventory_date: $inventory_date,
      source: {
        issues_jsonl: ".beads/issues.jsonl",
        insights_data_hash: ($insights[0].data_hash // ""),
        insights_generated_at: ($insights[0].generated_at // "")
      },
      scoring: {
        risk_score: "priority_weight + betweenness_bucket + critical_path_bucket + severity_weight",
        owner_suggestion: "use issue assignee when set; otherwise class-based fallback",
        wave_assignment: {
          wave_1: "risk_score >= 6 and no explicit EXCEPTION",
          wave_2: "risk_score in [4,5] and no explicit EXCEPTION",
          wave_3: "risk_score <= 3 and no explicit EXCEPTION",
          exception_register: "explicit [bd-3qwe self-doc] EXCEPTION marker"
        }
      },
      items: [
        $issues[]
        | select((.status == "open" or .status == "in_progress") and (.issue_type != "question"))
        | . as $issue
        | ((.comments // []) | map(.text // "") | join("\n")) as $comments
        | ($comments | contains("[bd-3qwe self-doc] RATIONALE")) as $has_rationale
        | ($comments | contains("[bd-3qwe self-doc] EVIDENCE")) as $has_evidence
        | ($comments | contains("[bd-3qwe self-doc] EXCEPTION")) as $has_exception
        | (($has_rationale and $has_evidence) | not) as $has_debt
        | select($has_debt)
        | (class_of($issue)) as $class
        | (severity_of($class)) as $severity
        | (owner_suggestion($issue; $class)) as $suggested_owner
        | (owner_basis($issue)) as $owner_suggestion_basis
        | (($pr[$issue.id] // 0) | tonumber) as $pagerank
        | (($bw[$issue.id] // 0) | tonumber) as $betweenness
        | (($cp[$issue.id] // 0) | tonumber) as $critical_path
        | (priority_weight(($issue.priority // 4)) + bucket_bet($betweenness) + bucket_cp($critical_path) + severity_weight($severity)) as $risk_score
        | {
            id: $issue.id,
            title: $issue.title,
            issue_type: $issue.issue_type,
            status: $issue.status,
            priority: ($issue.priority // 4),
            class: $class,
            severity: $severity,
            has_rationale: $has_rationale,
            has_evidence: $has_evidence,
            has_exception: $has_exception,
            suggested_owner: $suggested_owner,
            owner_suggestion_basis: $owner_suggestion_basis,
            pagerank: $pagerank,
            betweenness: $betweenness,
            critical_path: $critical_path,
            risk_score: $risk_score,
            recommended_wave: (
              if $has_exception then "exception_register"
              elif $risk_score >= 6 then "wave_1"
              elif $risk_score >= 4 then "wave_2"
              else "wave_3"
              end
            )
          }
      ]
      | sort_by(-.risk_score, .priority, -.betweenness, -.pagerank, .id)
    }
  ' > "$OUT_JSON"

TOTAL="$(jq '.items | length' "$OUT_JSON")"
echo "Wrote $OUT_JSON ($TOTAL items)"
