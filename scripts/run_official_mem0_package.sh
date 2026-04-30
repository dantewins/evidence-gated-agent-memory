#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_ENABLE_PARALLEL_LOADING="${HF_ENABLE_PARALLEL_LOADING:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

CATEGORIES="${CATEGORIES:-knowledge-update single-session-preference multi-session single-session-user}"
POLICIES="${POLICIES:-official_mem0 official_mem0_odv2_selective}"
LONGMEMEVAL_INPUT="${LONGMEMEVAL_INPUT:-data/longmemeval_s_cleaned.json}"

MEM0_LLM_PROVIDER="${MEM0_LLM_PROVIDER:-ollama}"
MEM0_LLM_MODEL="${MEM0_LLM_MODEL:-llama3.1:8b}"
MEM0_EMBEDDER_PROVIDER="${MEM0_EMBEDDER_PROVIDER:-ollama}"
MEM0_EMBEDDER_MODEL="${MEM0_EMBEDDER_MODEL:-nomic-embed-text:latest}"
MEM0_VECTOR_STORE_PROVIDER="${MEM0_VECTOR_STORE_PROVIDER:-qdrant}"
MEM0_EMBEDDING_DIMS="${MEM0_EMBEDDING_DIMS:-768}"

export MEM0_LLM_PROVIDER
export MEM0_LLM_MODEL
export MEM0_EMBEDDER_PROVIDER
export MEM0_EMBEDDER_MODEL
export MEM0_VECTOR_STORE_PROVIDER
export MEM0_EMBEDDING_DIMS

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
  POLICIES="${POLICIES}" LONGMEMEVAL_INPUT="${LONGMEMEVAL_INPUT}" \
    RESULT_PREFIX=official_mem0_longmemeval COMPILE_SLICE_REPORT=0 \
    bash scripts/run_longmemeval_slice.sh "${CATEGORY}"
  CASE_PATHS+=("results/official_mem0_longmemeval_${SAFE_CATEGORY}_cases.jsonl")
done

"${PYTHON_BIN}" scripts/compile_stronger_results.py \
  --baseline-policy official_mem0 \
  --target-policy official_mem0_odv2_selective \
  --audit-output results/official_mem0_audit.jsonl \
  "${CASE_PATHS[@]}" \
  > results/official_mem0_summary.csv

if [[ -f "${LONGMEMEVAL_INPUT}" ]]; then
  shasum -a 256 "${LONGMEMEVAL_INPUT}" > results/longmemeval_input.sha256
fi

echo "Wrote results/official_mem0_summary.csv"
echo "Wrote results/official_mem0_audit.jsonl"
if [[ -f results/longmemeval_input.sha256 ]]; then
  echo "Wrote results/longmemeval_input.sha256"
fi
