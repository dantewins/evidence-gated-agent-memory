#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_ENABLE_PARALLEL_LOADING="${HF_ENABLE_PARALLEL_LOADING:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

SLICE_NAME="${1:-knowledge-update}"
SAFE_SLICE_NAME="${SLICE_NAME//[^a-zA-Z0-9_-]/_}"
RESULT_PREFIX="${RESULT_PREFIX:-longmemeval}"

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

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-12}"
READER_FLUSH_SIZE="${READER_FLUSH_SIZE:-${INFERENCE_BATCH_SIZE}}"
CONTEXT_WINDOW="${CONTEXT_WINDOW:-8192}"
LONGMEMEVAL_INPUT="${LONGMEMEVAL_INPUT:-data/longmemeval_s_cleaned.json}"
POLICIES="${POLICIES:-mem0 odv2_mem0_selective}"
COMPILE_SLICE_REPORT="${COMPILE_SLICE_REPORT:-1}"
PROGRESS="${PROGRESS:-0}"
RESULT_DIR="${RESULT_DIR:-results}"
OVERWRITE_RESULTS="${OVERWRITE_RESULTS:-0}"

mkdir -p "${RESULT_DIR}"

POLICY_ARGS=()
for POLICY in ${POLICIES}; do
  POLICY_ARGS+=(--policy "${POLICY}")
done

PROGRESS_ARGS=()
if [[ "${PROGRESS}" == "1" || "${PROGRESS}" == "true" ]]; then
  PROGRESS_ARGS+=(--progress)
fi

OVERWRITE_ARGS=()
if [[ "${OVERWRITE_RESULTS}" == "1" || "${OVERWRITE_RESULTS}" == "true" ]]; then
  OVERWRITE_ARGS+=(--overwrite-output)
fi

OUTPUT_PATH="${RESULT_DIR}/${RESULT_PREFIX}_${SAFE_SLICE_NAME}.json"
CASES_OUTPUT_PATH="${RESULT_DIR}/${RESULT_PREFIX}_${SAFE_SLICE_NAME}_cases.jsonl"

"${PYTHON_BIN}" -m memory_inference.cli longmemeval \
  --input "${LONGMEMEVAL_INPUT}" \
  --input-format raw \
  --reasoner local-hf \
  --model-id "${MODEL_ID}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --inference-batch-size "${INFERENCE_BATCH_SIZE}" \
  --context-window "${CONTEXT_WINDOW}" \
  --reader-flush-size "${READER_FLUSH_SIZE}" \
  "${POLICY_ARGS[@]}" \
  --category "${SLICE_NAME}" \
  --cache-dir ".cache/memory_inference_${RESULT_PREFIX}_${SAFE_SLICE_NAME}" \
  --output "${OUTPUT_PATH}" \
  --cases-output "${CASES_OUTPUT_PATH}" \
  "${PROGRESS_ARGS[@]}" \
  "${OVERWRITE_ARGS[@]}"

if [[ "${COMPILE_SLICE_REPORT}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/compile_boss_results.py "${CASES_OUTPUT_PATH}"
fi
