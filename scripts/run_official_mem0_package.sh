#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_ENABLE_PARALLEL_LOADING="${HF_ENABLE_PARALLEL_LOADING:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-critical}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export PYTHONWARNINGS="${PYTHONWARNINGS:+${PYTHONWARNINGS},}ignore::FutureWarning"

CATEGORIES="${CATEGORIES:-knowledge-update single-session-preference multi-session single-session-user}"
POLICIES="${POLICIES:-official_mem0 official_mem0_odv2_selective}"
LONGMEMEVAL_INPUT="${LONGMEMEVAL_INPUT:-data/longmemeval_s_cleaned.json}"
RUN_ID="${RUN_ID:-official_mem0_$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_DIR="${RESULT_DIR:-results/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${RESULT_DIR}/logs}"
DIAGNOSTIC_DIR="${DIAGNOSTIC_DIR:-${RESULT_DIR}/diagnostics}"

MEM0_LLM_PROVIDER="${MEM0_LLM_PROVIDER:-ollama}"
MEM0_LLM_MODEL="${MEM0_LLM_MODEL:-llama3.1:8b}"
if [[ "${MEM0_LLM_PROVIDER}" == "vllm" ]]; then
  MEM0_LLM_MAX_TOKENS="${MEM0_LLM_MAX_TOKENS:-512}"
else
  MEM0_LLM_MAX_TOKENS="${MEM0_LLM_MAX_TOKENS:-2000}"
fi
MEM0_EMBEDDER_PROVIDER="${MEM0_EMBEDDER_PROVIDER:-huggingface}"
MEM0_EMBEDDER_MODEL="${MEM0_EMBEDDER_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
MEM0_VECTOR_STORE_PROVIDER="${MEM0_VECTOR_STORE_PROVIDER:-qdrant}"
MEM0_EMBEDDING_DIMS="${MEM0_EMBEDDING_DIMS:-384}"
MEM0_ADD_INFER="${MEM0_ADD_INFER:-true}"
MEM0_ADD_BATCH_SIZE="${MEM0_ADD_BATCH_SIZE:-8}"
MEM0_ADD_MAX_MESSAGE_CHARS="${MEM0_ADD_MAX_MESSAGE_CHARS:-2000}"
MEM0_RAW_FALLBACK_ON_EMPTY="${MEM0_RAW_FALLBACK_ON_EMPTY:-true}"
MEM0_FAIL_ON_RAW_FALLBACK="${MEM0_FAIL_ON_RAW_FALLBACK:-true}"
MEM0_ALLOW_RAW_FALLBACK_SMOKE="${MEM0_ALLOW_RAW_FALLBACK_SMOKE:-0}"
MEM0_REQUIRE_NONEMPTY="${MEM0_REQUIRE_NONEMPTY:-true}"
MEM0_QUIET="${MEM0_QUIET:-true}"
MEM0_REUSE_CLIENT="${MEM0_REUSE_CLIENT:-true}"
SKIP_MEM0_SMOKE="${SKIP_MEM0_SMOKE:-0}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-64}"
READER_FLUSH_SIZE="${READER_FLUSH_SIZE:-${INFERENCE_BATCH_SIZE}}"
PROGRESS="${PROGRESS:-1}"
OVERWRITE_RESULTS="${OVERWRITE_RESULTS:-0}"

export MEM0_LLM_PROVIDER
export MEM0_LLM_MODEL
export MEM0_LLM_MAX_TOKENS
export MEM0_EMBEDDER_PROVIDER
export MEM0_EMBEDDER_MODEL
export MEM0_VECTOR_STORE_PROVIDER
export MEM0_EMBEDDING_DIMS
export MEM0_ADD_INFER
export MEM0_ADD_BATCH_SIZE
export MEM0_ADD_MAX_MESSAGE_CHARS
export MEM0_RAW_FALLBACK_ON_EMPTY
export MEM0_FAIL_ON_RAW_FALLBACK
export MEM0_ALLOW_RAW_FALLBACK_SMOKE
export MEM0_REQUIRE_NONEMPTY
export MEM0_QUIET
export MEM0_REUSE_CLIENT
export INFERENCE_BATCH_SIZE
export READER_FLUSH_SIZE
export PROGRESS
export RESULT_DIR
export LOG_DIR
export DIAGNOSTIC_DIR
export OVERWRITE_RESULTS

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

mkdir -p "${RESULT_DIR}" "${LOG_DIR}" "${DIAGNOSTIC_DIR}"

RUN_LOG="${LOG_DIR}/run.log"
ERROR_LOG="${LOG_DIR}/run.err"
touch "${RUN_LOG}" "${ERROR_LOG}"
exec > >(tee -a "${RUN_LOG}") 2> >(tee -a "${ERROR_LOG}" >&2)

write_failure_report() {
  local status="$1"
  if [[ "${status}" == "0" ]]; then
    return
  fi
  {
    echo "status=${status}"
    date -u
    echo
    echo "Last stderr lines:"
    tail -200 "${ERROR_LOG}" || true
    echo
    echo "Last stdout lines:"
    tail -200 "${RUN_LOG}" || true
  } > "${DIAGNOSTIC_DIR}/failure_report.txt"
  echo "Runner failed with status ${status}; wrote ${DIAGNOSTIC_DIR}/failure_report.txt"
}
trap 'status=$?; write_failure_report "${status}"' EXIT

echo "runner starting"
echo "Run ID: ${RUN_ID}"
echo "Writing official Mem0 outputs to ${RESULT_DIR}"
echo "Writing logs to ${LOG_DIR}"
echo "Writing diagnostics to ${DIAGNOSTIC_DIR}"

{
  date -u
  echo "PWD=$(pwd)"
  echo "PYTHON_BIN=${PYTHON_BIN}"
  echo "CATEGORIES=${CATEGORIES}"
  echo "POLICIES=${POLICIES}"
  echo "LONGMEMEVAL_INPUT=${LONGMEMEVAL_INPUT}"
  echo "MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
  echo "INFERENCE_BATCH_SIZE=${INFERENCE_BATCH_SIZE}"
  echo "READER_FLUSH_SIZE=${READER_FLUSH_SIZE}"
  echo "OVERWRITE_RESULTS=${OVERWRITE_RESULTS}"
  echo "MEM0_LLM_PROVIDER=${MEM0_LLM_PROVIDER}"
  echo "MEM0_LLM_MODEL=${MEM0_LLM_MODEL}"
  echo "MEM0_LLM_MAX_TOKENS=${MEM0_LLM_MAX_TOKENS}"
  echo "VLLM_BASE_URL=${VLLM_BASE_URL:-}"
  echo "VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-}"
  echo "MEM0_EMBEDDER_PROVIDER=${MEM0_EMBEDDER_PROVIDER}"
  echo "MEM0_EMBEDDER_MODEL=${MEM0_EMBEDDER_MODEL}"
  echo "MEM0_VECTOR_STORE_PROVIDER=${MEM0_VECTOR_STORE_PROVIDER}"
  echo "MEM0_ADD_INFER=${MEM0_ADD_INFER}"
  echo "MEM0_ADD_BATCH_SIZE=${MEM0_ADD_BATCH_SIZE}"
  echo "MEM0_ADD_MAX_MESSAGE_CHARS=${MEM0_ADD_MAX_MESSAGE_CHARS}"
  echo "MEM0_RAW_FALLBACK_ON_EMPTY=${MEM0_RAW_FALLBACK_ON_EMPTY}"
  echo "MEM0_FAIL_ON_RAW_FALLBACK=${MEM0_FAIL_ON_RAW_FALLBACK}"
  echo "MEM0_ALLOW_RAW_FALLBACK_SMOKE=${MEM0_ALLOW_RAW_FALLBACK_SMOKE}"
  echo "MEM0_REQUIRE_NONEMPTY=${MEM0_REQUIRE_NONEMPTY}"
  echo "MEM0_REUSE_CLIENT=${MEM0_REUSE_CLIENT}"
} > "${DIAGNOSTIC_DIR}/runner_environment.txt"

git status --short > "${DIAGNOSTIC_DIR}/git_status_start.txt" 2>&1 || true
git diff --stat > "${DIAGNOSTIC_DIR}/git_diff_stat_start.txt" 2>&1 || true
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "${DIAGNOSTIC_DIR}/nvidia_smi_start.txt" 2>&1 || true
  nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv \
    > "${DIAGNOSTIC_DIR}/gpu_snapshot_start.csv" 2>&1 || true
fi

if [[ "${MEM0_LLM_PROVIDER}" == "vllm" ]]; then
  "${PYTHON_BIN}" scripts/check_mem0_vllm_budget.py
fi

if [[ "${SKIP_MEM0_SMOKE}" != "1" ]]; then
  "${PYTHON_BIN}" scripts/check_official_mem0_smoke.py
fi

CASE_PATHS=()
for CATEGORY in ${CATEGORIES}; do
  SAFE_CATEGORY="${CATEGORY//[^a-zA-Z0-9_-]/_}"
  echo "slice starting: ${CATEGORY}"
  POLICIES="${POLICIES}" LONGMEMEVAL_INPUT="${LONGMEMEVAL_INPUT}" \
    RESULT_PREFIX=official_mem0_longmemeval COMPILE_SLICE_REPORT=0 \
    bash scripts/run_longmemeval_slice.sh "${CATEGORY}"
  CASE_PATHS+=("${RESULT_DIR}/official_mem0_longmemeval_${SAFE_CATEGORY}_cases.jsonl")
  echo "slice finished: ${CATEGORY}"
done

"${PYTHON_BIN}" scripts/compile_stronger_results.py \
  --baseline-policy official_mem0 \
  --audit-output "${RESULT_DIR}/official_mem0_audit.jsonl" \
  "${CASE_PATHS[@]}" \
  > "${RESULT_DIR}/official_mem0_summary.csv"

if [[ -f "${LONGMEMEVAL_INPUT}" ]]; then
  shasum -a 256 "${LONGMEMEVAL_INPUT}" > "${RESULT_DIR}/longmemeval_input.sha256"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "${DIAGNOSTIC_DIR}/nvidia_smi_end.txt" 2>&1 || true
  nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv \
    > "${DIAGNOSTIC_DIR}/gpu_snapshot_end.csv" 2>&1 || true
fi
git status --short > "${DIAGNOSTIC_DIR}/git_status_end.txt" 2>&1 || true

echo "Wrote ${RESULT_DIR}/official_mem0_summary.csv"
echo "Wrote ${RESULT_DIR}/official_mem0_audit.jsonl"
if [[ -f "${RESULT_DIR}/longmemeval_input.sha256" ]]; then
  echo "Wrote ${RESULT_DIR}/longmemeval_input.sha256"
fi
echo "runner finished"
