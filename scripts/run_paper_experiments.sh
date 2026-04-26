#!/usr/bin/env bash
set -euo pipefail

# Reproducible commands for the final paper runs.
# Run from the repository root on a machine with the project venv activated.
# These commands assume the cleaned LongMemEval and official LoCoMo JSON files
# are available under data/.

export PYTHONPATH=src
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_ENABLE_PARALLEL_LOADING="${HF_ENABLE_PARALLEL_LOADING:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-12}"

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

# Download benchmark inputs if needed:
#   curl -L https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json -o data/longmemeval_s_cleaned.json
#   curl -L https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json -o data/locomo10.json

"${PYTHON_BIN}" -m memory_inference.cli longmemeval \
  --input data/longmemeval_s_cleaned.json \
  --input-format raw \
  --reasoner local-hf \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --device cuda \
  --dtype bfloat16 \
  --inference-batch-size "${INFERENCE_BATCH_SIZE}" \
  --policy mem0 \
  --policy odv2_mem0_selective \
  --cache-dir .cache/memory_inference_longmemeval_final \
  --output results/longmemeval_qwen25_7b_final.json \
  --cases-output results/longmemeval_qwen25_7b_final_cases.jsonl

"${PYTHON_BIN}" scripts/compile_boss_results.py results/longmemeval_qwen25_7b_final_cases.jsonl

"${PYTHON_BIN}" -m memory_inference.cli locomo \
  --input data/locomo10.json \
  --input-format raw \
  --reasoner local-hf \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --device cuda \
  --dtype bfloat16 \
  --inference-batch-size "${INFERENCE_BATCH_SIZE}" \
  --policy mem0 \
  --policy odv2_mem0_selective \
  --cache-dir .cache/memory_inference_locomo_final \
  --output results/locomo_qwen25_7b_final.json \
  --cases-output results/locomo_qwen25_7b_final_cases.jsonl

"${PYTHON_BIN}" scripts/compile_boss_results.py results/locomo_qwen25_7b_final_cases.jsonl
