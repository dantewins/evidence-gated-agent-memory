#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_ENABLE_PARALLEL_LOADING="${HF_ENABLE_PARALLEL_LOADING:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning:mem0.embeddings.huggingface}"

CATEGORIES="${CATEGORIES:-knowledge-update single-session-preference multi-session single-session-user}"
POLICIES="${POLICIES:-official_mem0 official_mem0_odv2_selective}"
LONGMEMEVAL_INPUT="${LONGMEMEVAL_INPUT:-data/longmemeval_s_cleaned.json}"

MEM0_LLM_PROVIDER="${MEM0_LLM_PROVIDER:-ollama}"
MEM0_LLM_MODEL="${MEM0_LLM_MODEL:-llama3.1:8b}"
MEM0_EMBEDDER_PROVIDER="${MEM0_EMBEDDER_PROVIDER:-huggingface}"
MEM0_EMBEDDER_MODEL="${MEM0_EMBEDDER_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
MEM0_VECTOR_STORE_PROVIDER="${MEM0_VECTOR_STORE_PROVIDER:-qdrant}"
MEM0_EMBEDDING_DIMS="${MEM0_EMBEDDING_DIMS:-384}"
MEM0_ADD_INFER="${MEM0_ADD_INFER:-true}"
MEM0_ADD_BATCH_SIZE="${MEM0_ADD_BATCH_SIZE:-8}"
MEM0_ADD_MAX_MESSAGE_CHARS="${MEM0_ADD_MAX_MESSAGE_CHARS:-2000}"
MEM0_RAW_FALLBACK_ON_EMPTY="${MEM0_RAW_FALLBACK_ON_EMPTY:-true}"
MEM0_REQUIRE_NONEMPTY="${MEM0_REQUIRE_NONEMPTY:-true}"
SKIP_MEM0_SMOKE="${SKIP_MEM0_SMOKE:-0}"

export MEM0_LLM_PROVIDER
export MEM0_LLM_MODEL
export MEM0_EMBEDDER_PROVIDER
export MEM0_EMBEDDER_MODEL
export MEM0_VECTOR_STORE_PROVIDER
export MEM0_EMBEDDING_DIMS
export MEM0_ADD_INFER
export MEM0_ADD_BATCH_SIZE
export MEM0_ADD_MAX_MESSAGE_CHARS
export MEM0_RAW_FALLBACK_ON_EMPTY
export MEM0_REQUIRE_NONEMPTY

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

if [[ "${SKIP_MEM0_SMOKE}" != "1" ]]; then
  "${PYTHON_BIN}" scripts/check_official_mem0_smoke.py
fi

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
