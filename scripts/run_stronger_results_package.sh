#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_ENABLE_PARALLEL_LOADING="${HF_ENABLE_PARALLEL_LOADING:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

CATEGORIES="${CATEGORIES:-knowledge-update single-session-preference multi-session single-session-user}"
POLICIES="${POLICIES:-mem0 odv2_support_compact odv2_stale_guard odv2_mem0_selective odv2_mem0_aggressive}"
LONGMEMEVAL_INPUT="${LONGMEMEVAL_INPUT:-data/longmemeval_s_cleaned.json}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

mkdir -p results

CASE_PATHS=()
for CATEGORY in ${CATEGORIES}; do
  SAFE_CATEGORY="${CATEGORY//[^a-zA-Z0-9_-]/_}"
  POLICIES="${POLICIES}" LONGMEMEVAL_INPUT="${LONGMEMEVAL_INPUT}" RESULT_PREFIX=longmemeval \
    bash scripts/run_longmemeval_slice.sh "${CATEGORY}"
  CASE_PATHS+=("results/longmemeval_${SAFE_CATEGORY}_cases.jsonl")
done

"${PYTHON_BIN}" scripts/compile_stronger_results.py \
  --audit-output results/stronger_results_audit.jsonl \
  "${CASE_PATHS[@]}" \
  > results/stronger_results_summary.csv

if [[ -f "${LONGMEMEVAL_INPUT}" ]]; then
  shasum -a 256 "${LONGMEMEVAL_INPUT}" > results/longmemeval_input.sha256
fi

echo "Wrote results/stronger_results_summary.csv"
echo "Wrote results/stronger_results_audit.jsonl"
if [[ -f results/longmemeval_input.sha256 ]]; then
  echo "Wrote results/longmemeval_input.sha256"
fi
