# Evidence-Gated Agent Memory

This repository contains the experimental code for a research project on validity-aware memory for long-horizon LLM agents. The core question is whether a lightweight memory gate can sit on top of Mem0-style retrieval and remove stale or redundant context without changing the frozen reader model.

The current paper claim is intentionally narrow: ODV2 is not a replacement for Mem0 and is not presented as a broad benchmark-beating retriever. It is evaluated as a conservative post-retrieval filter that should preserve answer accuracy while reducing prompt context on validity-sensitive cases.

## Method

ODV2 maintains a small validity ledger over extracted memory state:

- current state
- archived or superseded state
- unresolved same-key conflicts
- provenance links back to support text

At query time, the conservative policy starts from the baseline retriever output and only filters context when the retrieved evidence makes the edit low-risk. The main policy does not inject ODV2-only evidence into the prompt.

## Main Comparisons

The reviewer-facing comparison is:

```text
official_mem0
official_mem0_odv2_selective
```

The local ablation comparison is:

```text
mem0
odv2_support_compact
odv2_stale_guard
odv2_mem0_selective
odv2_mem0_aggressive
```

`odv2_mem0_aggressive` is a negative/control variant for measuring the cost of less conservative filtering. It should not be used as the headline method unless the results support it.

## Benchmarks

The primary benchmark path is LongMemEval. The default official-Mem0 run covers these categories:

- `knowledge-update`
- `single-session-preference`
- `multi-session`
- `single-session-user`

The compiler reports all paired cases plus predeclared diagnostic slices:

- `Predeclared validity-sensitive union`
- `Current-state same-key evidence retrieved`
- `Gold-mismatched same-key state exposed`
- `Same-key state conflict exposed`
- `ODV2 intervened`

The gold-mismatch fields are offline diagnostics for analysis. They are not runtime signals available to the memory policy.

## Setup

Create or activate a virtual environment, then install the package:

```bash
python -m pip install -e ".[official-mem0]"
```

For local Hugging Face reader runs, install:

```bash
python -m pip install -e ".[local-hf]"
```

## Run Official Mem0 Comparison

If using Ollama for Mem0 extraction:

```bash
bash scripts/run_official_mem0_package.sh
```

Defaults:

```text
MEM0_LLM_PROVIDER=ollama
MEM0_LLM_MODEL=llama3.1:8b
MEM0_LLM_MAX_TOKENS=2000
MEM0_EMBEDDER_PROVIDER=huggingface
MEM0_EMBEDDER_MODEL=sentence-transformers/all-MiniLM-L6-v2
MEM0_EMBEDDING_DIMS=384
MEM0_VECTOR_STORE_PROVIDER=qdrant
MEM0_ADD_INFER=true
MEM0_ADD_BATCH_SIZE=8
MEM0_ADD_MAX_MESSAGE_CHARS=2000
MEM0_RAW_FALLBACK_ON_EMPTY=true
MEM0_FAIL_ON_RAW_FALLBACK=true
MEM0_ALLOW_RAW_FALLBACK_SMOKE=0
MEM0_REQUIRE_NONEMPTY=true
MEM0_QUIET=true
MEM0_REUSE_CLIENT=true
INFERENCE_BATCH_SIZE=64
READER_FLUSH_SIZE=64
PROGRESS=1
RESULT_DIR=results/official_mem0_<utc-run-id>
LOG_DIR=results/official_mem0_<utc-run-id>/logs
DIAGNOSTIC_DIR=results/official_mem0_<utc-run-id>/diagnostics
OVERWRITE_RESULTS=0
```

The official Mem0 runner starts with a smoke test. If Mem0 stores no searchable memory, or if it only succeeds by falling back to raw `infer=False` storage, the run fails instead of producing an invalid report. The raw fallback path is still available for explicit debugging, but official package runs set `MEM0_FAIL_ON_RAW_FALLBACK=true` by default. Loader warnings from Mem0 and SentenceTransformers are suppressed by default; set `MEM0_QUIET=false` if you need to debug provider initialization.

The runner prints `runner starting`, `context started`, `context finished`, `case prepared`, `case finished`, `policy finished`, and `runner finished` progress lines. Per-case diagnostics are streamed to JSONL as cases finish, so interrupted runs leave partial case files. The run directory contains `logs/run.log`, `logs/run.err`, and diagnostic snapshots for the environment, git state, and GPU state. If the script exits nonzero, it writes `diagnostics/failure_report.txt`. Existing output files are not overwritten unless `OVERWRITE_RESULTS=1`. Reader calls are accumulated across contexts up to `READER_FLUSH_SIZE`, which keeps local Hugging Face inference better batched on large GPUs such as an H100. This does not parallelize Mem0 extraction itself; if `MEM0_ADD_INFER=true` with Ollama, extraction can still be the slow stage.

If using vLLM instead of Ollama:

```bash
MEM0_LLM_PROVIDER=vllm \
MEM0_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
MEM0_LLM_MAX_TOKENS=512 \
VLLM_BASE_URL=http://localhost:8000/v1 \
bash scripts/run_official_mem0_package.sh
```

For vLLM, the runner queries `/v1/models` before the smoke test and rejects Mem0 add batches that are estimated to exceed the server context window. If you increase `MEM0_ADD_BATCH_SIZE`, either also lower `MEM0_ADD_MAX_MESSAGE_CHARS`/`MEM0_LLM_MAX_TOKENS` or start vLLM with a larger `--max-model-len`. Set `VLLM_MAX_MODEL_LEN` only if the server does not report `max_model_len`.

Outputs:

```text
results/official_mem0_<utc-run-id>/official_mem0_summary.csv
results/official_mem0_<utc-run-id>/official_mem0_audit.jsonl
results/official_mem0_<utc-run-id>/official_mem0_longmemeval_*_cases.jsonl
results/official_mem0_<utc-run-id>/longmemeval_input.sha256
results/official_mem0_<utc-run-id>/logs/run.log
results/official_mem0_<utc-run-id>/logs/run.err
results/official_mem0_<utc-run-id>/diagnostics/*
```

## Run Local Ablations

```bash
bash scripts/run_stronger_results_package.sh
```

Outputs:

```text
results/stronger_results_summary.csv
results/stronger_results_audit.jsonl
results/longmemeval_*_cases.jsonl
results/longmemeval_input.sha256
```

## Result Interpretation

Use ODV2 as a positive result only when the paired comparison shows:

- no or very few paired losses
- non-negative accuracy difference
- lower prompt-token usage, especially on intervention cases
- auditable per-case JSONL evidence

Do not claim broad Mem0 superiority unless the official-Mem0 comparison supports it. The intended contribution is a precision-first validity gate for memory retrieval, not a universal long-memory retriever.

## Tests

```bash
python -m pytest
```
