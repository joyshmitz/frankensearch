#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${CARGO_TARGET_DIR:-${ROOT_DIR}/target_codex}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/data/e2e_optimize_runs}"
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
RUN_ONE="${RUN_ROOT}/run1-${STAMP}"
RUN_TWO="${RUN_ROOT}/run2-${STAMP}"

mkdir -p "${RUN_ONE}" "${RUN_TWO}"

run_optimizer() {
  local output_dir="$1"
  local run_log="$2"
  local started_at finished_at elapsed
  started_at="$(date +%s)"
  "${OPTIMIZER_BIN}" \
    --max-generations 5 \
    --max-evaluations 500 \
    --folds 1 \
    --seed 42 \
    --output-dir "${output_dir}" 2>&1 | tee "${run_log}" >&2
  finished_at="$(date +%s)"
  elapsed=$((finished_at - started_at))
  echo "${elapsed}"
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing expected artifact: ${path}" >&2
    exit 1
  fi
}

validate_log_jsonl() {
  local log_path="$1"
  local generation_count
  generation_count="$(wc -l < "${log_path}" | tr -d ' ')"
  if [[ "${generation_count}" -ne 5 ]]; then
    echo "Unexpected generation count in ${log_path}: ${generation_count} (expected 5)" >&2
    exit 1
  fi

  local prev_best=""
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    jq -e '
      has("generation")
      and has("evaluations")
      and has("best_fitness")
      and has("best_ndcg")
      and has("params")
    ' <<<"${line}" >/dev/null

    local best
    best="$(jq -r '.best_fitness' <<<"${line}")"
    if [[ -n "${prev_best}" ]]; then
      awk -v current="${best}" -v prev="${prev_best}" 'BEGIN { exit (current <= prev + 1e-12 ? 0 : 1) }' || {
        echo "best_fitness regressed in ${log_path}: ${best} > ${prev_best}" >&2
        exit 1
      }
    fi
    prev_best="${best}"
  done < "${log_path}"
}

require_toml_keys() {
  local toml_path="$1"
  local keys=(
    quality_weight
    rrf_k
    candidate_multiplier
    quality_timeout_ms
    hnsw_ef_search
    mrl_rescore_top_k
  )
  for key in "${keys[@]}"; do
    if ! rg -q "^${key}\\s*=" "${toml_path}"; then
      echo "Missing key '${key}' in ${toml_path}" >&2
      exit 1
    fi
  done
}

extract_optimized_ndcg() {
  local run_log="$1"
  local ndcg
  ndcg="$(rg -o 'Optimized nDCG@10:\s+[0-9]+(\.[0-9]+)?' "${run_log}" | tail -n 1 | awk '{print $3}')"
  if [[ -z "${ndcg}" ]]; then
    echo "Unable to find optimized nDCG@10 in ${run_log}" >&2
    exit 1
  fi
  awk -v value="${ndcg}" 'BEGIN { exit (value + 0.0 == value ? 0 : 1) }' || {
    echo "optimized nDCG@10 is not numeric: ${ndcg}" >&2
    exit 1
  }
  echo "${ndcg}"
}

extract_cv_variance() {
  local run_log="$1"
  local variance
  variance="$(rg -o 'CV variance:\s+[0-9]+(\.[0-9]+)?' "${run_log}" | tail -n 1 | awk '{print $3}')"
  if [[ -z "${variance}" ]]; then
    echo "Unable to find CV variance in ${run_log}" >&2
    exit 1
  fi
  awk -v value="${variance}" 'BEGIN { exit (value + 0.0 == value ? 0 : 1) }' || {
    echo "CV variance is not numeric: ${variance}" >&2
    exit 1
  }
  echo "${variance}"
}

echo "[1/6] Running optimizer smoke pass (seed=42, generations=5, folds=1)"
echo "Building optimize-params binary once (avoids repeated cargo lock contention)"
CARGO_TARGET_DIR="${TARGET_DIR}" cargo build -p optimize-params >/dev/null
OPTIMIZER_BIN="${TARGET_DIR}/debug/optimize-params"
if [[ ! -x "${OPTIMIZER_BIN}" ]]; then
  echo "Optimizer binary not found at ${OPTIMIZER_BIN}" >&2
  exit 1
fi

LOG_ONE_RUN="${RUN_ONE}/optimizer.log"
RUNTIME_ONE_SECONDS="$(run_optimizer "${RUN_ONE}" "${LOG_ONE_RUN}")"
if [[ "${RUNTIME_ONE_SECONDS}" -ge 60 ]]; then
  echo "Optimizer runtime exceeded 60s: ${RUNTIME_ONE_SECONDS}s" >&2
  exit 1
fi

PARAMS_ONE="${RUN_ONE}/optimized_params.toml"
LOG_ONE="${RUN_ONE}/optimization_log.jsonl"
require_file "${PARAMS_ONE}"
require_file "${LOG_ONE}"

echo "[2/6] Verifying optimized_params.toml structure"
require_toml_keys "${PARAMS_ONE}"

echo "[3/6] Verifying optimization_log.jsonl structure and monotonic best_fitness"
validate_log_jsonl "${LOG_ONE}"

echo "[4/6] Extracting optimized nDCG@10 from run output"
NDCG_ONE="$(extract_optimized_ndcg "${LOG_ONE_RUN}")"
echo "optimized nDCG@10: ${NDCG_ONE}"
VARIANCE_ONE="$(extract_cv_variance "${LOG_ONE_RUN}")"
awk -v variance="${VARIANCE_ONE}" 'BEGIN { exit (variance < 0.05 ? 0 : 1) }' || {
  echo "CV variance is too high: ${VARIANCE_ONE} (expected < 0.05)" >&2
  exit 1
}
echo "cv variance: ${VARIANCE_ONE}"

echo "[5/6] Re-running with same seed and asserting bit-identical artifacts"
LOG_TWO_RUN="${RUN_TWO}/optimizer.log"
run_optimizer "${RUN_TWO}" "${LOG_TWO_RUN}" >/dev/null

PARAMS_TWO="${RUN_TWO}/optimized_params.toml"
LOG_TWO="${RUN_TWO}/optimization_log.jsonl"
require_file "${PARAMS_TWO}"
require_file "${LOG_TWO}"

cmp -s "${PARAMS_ONE}" "${PARAMS_TWO}" || {
  echo "Reproducibility check failed: optimized_params.toml differs across identical seeds" >&2
  exit 1
}
cmp -s "${LOG_ONE}" "${LOG_TWO}" || {
  echo "Reproducibility check failed: optimization_log.jsonl differs across identical seeds" >&2
  exit 1
}

echo "[6/6] Loading optimized config into TwoTierSearcher for three-query smoke"
CARGO_TARGET_DIR="${TARGET_DIR}" cargo test -p frankensearch --test integration optimized_config_can_drive_searcher_for_multiple_queries -- --nocapture

echo "optimize-params e2e smoke passed"
echo "Artifacts:"
echo "  ${RUN_ONE}"
echo "  ${RUN_TWO}"
