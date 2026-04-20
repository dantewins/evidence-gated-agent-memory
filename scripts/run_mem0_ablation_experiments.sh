#!/usr/bin/env bash
set -euo pipefail

# Reproducible Mem0 feature-ablation runs across LongMemEval and LoCoMo.
# Run from the repository root on the Brev machine with the project env active.
#
# Current note:
# - `mem0` already includes support-entry expansion for structured facts.
# - The informative new ablations are:
#   * `mem0_archive_conflict`
#   * `mem0_history_aware`
#   * `mem0_all_features`

export PYTHONPATH=src
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
LONGMEMEVAL_INPUT="${LONGMEMEVAL_INPUT:-data/longmemeval_s_cleaned.json}"
LOCOMO_INPUT="${LOCOMO_INPUT:-data/locomo10.json}"

COMMON_POLICIES=(
  --policy strong_retrieval
  --policy dense_retrieval
  --policy mem0
  --policy mem0_archive_conflict
  --policy mem0_history_aware
  --policy mem0_all_features
  --policy offline_delta_v2
  --policy odv2_strong
  --policy odv2_dense
)

python -m memory_inference.cli longmemeval \
  --input "${LONGMEMEVAL_INPUT}" \
  --input-format raw \
  --reasoner local-hf \
  --model-id "${MODEL_ID}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  "${COMMON_POLICIES[@]}" \
  --cache-dir .cache/memory_inference_longmemeval_mem0_ablation \
  --output results/longmemeval_mem0_ablation.json

python -m memory_inference.cli locomo \
  --input "${LOCOMO_INPUT}" \
  --input-format raw \
  --reasoner local-hf \
  --model-id "${MODEL_ID}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  "${COMMON_POLICIES[@]}" \
  --cache-dir .cache/memory_inference_locomo_mem0_ablation \
  --output results/locomo_mem0_ablation.json
