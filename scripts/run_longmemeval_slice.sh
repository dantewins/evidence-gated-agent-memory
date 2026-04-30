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
LONGMEMEVAL_INPUT="${LONGMEMEVAL_INPUT:-data/longmemeval_s_cleaned.json}"
POLICIES="${POLICIES:-mem0 odv2_mem0_selective}"
COMPILE_SLICE_REPORT="${COMPILE_SLICE_REPORT:-1}"

POLICY_ARGS=()
for POLICY in ${POLICIES}; do
  POLICY_ARGS+=(--policy "${POLICY}")
done

"${PYTHON_BIN}" -m memory_inference.cli longmemeval \
  --input "${LONGMEMEVAL_INPUT}" \
  --input-format raw \
  --reasoner local-hf \
  --model-id "${MODEL_ID}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --inference-batch-size "${INFERENCE_BATCH_SIZE}" \
  "${POLICY_ARGS[@]}" \
  --category "${SLICE_NAME}" \
  --cache-dir ".cache/memory_inference_${RESULT_PREFIX}_${SAFE_SLICE_NAME}" \
  --output "results/${RESULT_PREFIX}_${SAFE_SLICE_NAME}.json" \
  --cases-output "results/${RESULT_PREFIX}_${SAFE_SLICE_NAME}_cases.jsonl"

if [[ "${COMPILE_SLICE_REPORT}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/compile_boss_results.py "results/${RESULT_PREFIX}_${SAFE_SLICE_NAME}_cases.jsonl"
fi
