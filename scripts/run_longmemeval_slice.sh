#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_ENABLE_PARALLEL_LOADING="${HF_ENABLE_PARALLEL_LOADING:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

SLICE_NAME="${1:-knowledge-update}"
SAFE_SLICE_NAME="${SLICE_NAME//[^a-zA-Z0-9_-]/_}"

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

"${PYTHON_BIN}" -m memory_inference.cli longmemeval \
  --input "${LONGMEMEVAL_INPUT}" \
  --input-format raw \
  --reasoner local-hf \
  --model-id "${MODEL_ID}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --inference-batch-size "${INFERENCE_BATCH_SIZE}" \
  --policy mem0 \
  --policy odv2_mem0_selective \
  --policy odv2_recovery \
  --policy odv2_dense_compact \
  --category "${SLICE_NAME}" \
  --cache-dir ".cache/memory_inference_longmemeval_${SAFE_SLICE_NAME}" \
  --output "results/longmemeval_${SAFE_SLICE_NAME}.json" \
  --cases-output "results/longmemeval_${SAFE_SLICE_NAME}_cases.jsonl"

python scripts/summarize_diagnostics.py "results/longmemeval_${SAFE_SLICE_NAME}_cases.jsonl" mem0
