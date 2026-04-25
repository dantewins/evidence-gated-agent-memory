#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_ENABLE_PARALLEL_LOADING="${HF_ENABLE_PARALLEL_LOADING:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-12}"
LOCOMO_INPUT="${LOCOMO_INPUT:-data/locomo10.json}"

COMMON_POLICIES=(
  --policy strong_retrieval
  --policy dense_retrieval
  --policy mem0
  --policy mem0_validity_guard
  --policy odv2_mem0_hybrid
  --policy odv2_dense
)

python -m memory_inference.cli locomo \
  --input "${LOCOMO_INPUT}" \
  --input-format raw \
  --reasoner local-hf \
  --model-id "${MODEL_ID}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --inference-batch-size "${INFERENCE_BATCH_SIZE}" \
  "${COMMON_POLICIES[@]}" \
  --cache-dir .cache/memory_inference_locomo_recovery \
  --output results/locomo_recovery.json
