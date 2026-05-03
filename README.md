# Same-Evidence Reader-Budget Gating for Agent Memory

This repository contains the code and diagnostics for a token-efficiency study of
reader-budget gating for frozen LLM agents. The current paper asks a narrow
systems question: can we reduce the reader context passed by an open-source
Mem0 pipeline while using the exact same retrieved evidence and treating
accuracy as a guardrail?

The headline result is not an accuracy improvement. It is a cost-quality
operating point. On a 311-question LongMemEval subset, the same-evidence
adaptive gate routes official Mem0 top-5 evidence to either top-1 or top-3
reader prompts. It uses only Mem0 record text, answers 61/311 cases versus
62/311 for official Mem0 top-5, and reduces reader tokens by 37.0%.

## Current Result

Completed run:

```text
results/official_mem0_basecompact_full_20260502T072459Z
```

Main paired comparison:

| Policy | Correct | Reader tokens | Delta tokens | Tokens/correct |
| --- | ---: | ---: | ---: | ---: |
| `official_mem0` | 62/311 | 151,558 | baseline | 2,444 |
| `official_mem0_same_evidence_adaptive` | 61/311 | 95,466 | -37.0% | 1,565 |
| `official_mem0_top2` | 54/311 | 86,002 | -43.3% | 1,593 |
| `official_mem0_odv2_selective` | 58/311 | 86,838 | -42.7% | 1,497 |
| `official_mem0_staleaware_gate` | 62/311 | 95,685 | -36.9% | 1,543 |
| `official_mem0_top3` | 60/311 | 110,058 | -27.4% | 1,834 |

Additional checks produced for the paper:

- Same-evidence adaptive route: 105 cases use top-1 evidence and 206 use top-3 evidence; wins/losses versus top-5 are 5/6.
- Bootstrap CI for same-evidence adaptive reader-token reduction: 35.5% to 38.5%.
- Bootstrap CI for same-evidence adaptive tokens-per-correct reduction: 28.5% to 42.7%.
- Reviewed 50-case manual audit: automatic/manual agreement is 91/100 policy-case decisions; reviewed counts are 14/50 for official Mem0 and 11/50 for ODV2 compact.
- Cache-free 64-case reader replay: same-evidence adaptive takes 37.2 ms/case and answers 10/64 versus 62.6 ms/case and 9/64 for Mem0 top-5; fixed top-1/top-2/top-3 take 26.2/33.4/39.6 ms/case.
- Oracle answer-session sanity check: 27/64 correct when the reader receives only LongMemEval sessions marked as containing the answer.
- Evidence-overlap audit: ODV2 compact is secondary because 63/311 ODV2 rows include at least one record outside the corresponding official Mem0 top-5.

Interpretation:

- Supported claim: same-evidence adaptive reader-budget gating reduces reader-token spend under a fixed OSS Mem0 setup.
- Supported claim: the adaptive gate is a lower-cost operating point than top-3 while preserving more accuracy than top-1/top-2.
- Secondary diagnostic: ODV2 compact and ODV2/top-3 composition are useful operating points, but they are not the main causal evidence because their retrieved records can differ from the baseline top-5.
- Unsupported claim: the adaptive gate improves accuracy over full Mem0 top-5 retrieval.
- Unsupported claim: this reproduces Mem0 platform benchmark accuracy.

## Method

`official_mem0_same_evidence_adaptive` is composed from official Mem0 top-k
reader replays. It does not rerun retrieval and it does not use ODV2 records,
gold labels, predictions, or category labels.

The current adaptive gate:

1. Starts from official Mem0's saved top-5 records.
2. Computes risk features from the first four records.
3. Uses top-3 if there is at least one revision marker or at least two distinct answer-like numeric/time/money/duration values.
4. Uses top-1 otherwise.

`official_mem0_odv2_selective` and `official_mem0_staleaware_gate` remain in the
repository as secondary diagnostics, not as the main same-evidence result.

## Benchmarks

The primary benchmark path is LongMemEval. The completed official-Mem0 run
covers:

- `knowledge-update`
- `single-session-preference`
- `multi-session`
- `single-session-user`

The experiment is a paired efficiency ablation under one constrained OSS Mem0
setup and one local frozen reader. The reader is `Qwen/Qwen2.5-7B-Instruct`.
The Mem0 extraction LLM is also `Qwen/Qwen2.5-7B-Instruct` served through vLLM
for the final run.

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

The main runner is:

```bash
bash scripts/run_official_mem0_package.sh
```

For the final vLLM-style run, use:

```bash
RUN_ID=official_mem0_basecompact_full_$(date -u +%Y%m%dT%H%M%SZ) \
MEM0_LLM_PROVIDER=vllm \
MEM0_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
MEM0_LLM_MAX_TOKENS=512 \
MEM0_VLLM_DISABLE_RESPONSE_FORMAT=true \
VLLM_BASE_URL=http://localhost:8000/v1 \
VLLM_MAX_MODEL_LEN=16384 \
MEM0_ADD_BATCH_SIZE=16 \
MEM0_ADD_MAX_MESSAGE_CHARS=2000 \
OFFICIAL_MEM0_ODV2_GATE_MODE=compact \
OFFICIAL_MEM0_ODV2_COMPACT_TOP_K=2 \
READER_FLUSH_SIZE=8 \
INFERENCE_BATCH_SIZE=64 \
CONTEXT_WINDOW=8192 \
bash scripts/run_official_mem0_package.sh
```

The runner prints progress lines for runner start, context start/finish, case
finish, policy finish, and runner finish. Per-case diagnostics are streamed to
JSONL so interrupted runs leave usable partial files. The run directory contains
logs, environment snapshots, git state, GPU diagnostics, and a failure report if
the script exits nonzero.

Raw Mem0 fallback is disabled for official package runs. If Mem0 stores no
searchable memories or only succeeds by falling back to raw `infer=False`
storage, the smoke test fails rather than producing an invalid result.

## Analyze Results

Aggregate token savings and top-k comparisons:

```bash
python scripts/analyze_official_mem0_token_savings.py \
  results/official_mem0_basecompact_full_20260502T072459Z \
  --extra-policy official_mem0_same_evidence_adaptive \
  --extra-policy official_mem0_staleaware_gate \
  --extra-policy official_mem0_top1 \
  --extra-policy official_mem0_top3 \
  --extra-policy official_mem0_top4
```

Compose the same-evidence adaptive gate from existing official Mem0 top-k reader outputs:

```bash
python scripts/compose_official_mem0_same_evidence_adaptive.py \
  results/official_mem0_basecompact_full_20260502T072459Z \
  --output results/official_mem0_basecompact_full_20260502T072459Z/official_mem0_same_evidence_adaptive_cases.jsonl \
  --risk-candidate-k 4 \
  --revision-signal-threshold 1 \
  --distinct-value-threshold 2 \
  --overwrite
```

Compose the stale-aware gate from existing ODV2 and top-3 reader outputs:

```bash
python scripts/compose_official_mem0_staleaware_gate.py \
  results/official_mem0_basecompact_full_20260502T072459Z \
  --output results/official_mem0_basecompact_full_20260502T072459Z/official_mem0_staleaware_gate_cases.jsonl \
  --overwrite
```

Replay naive top-k reader baselines from existing Mem0 records:

```bash
for K in 1 2 3 4; do
  python scripts/replay_official_mem0_topk.py \
    results/official_mem0_basecompact_full_20260502T072459Z \
    --output results/official_mem0_basecompact_full_20260502T072459Z/official_mem0_top${K}_cases.jsonl \
    --top-k "$K" \
    --policy-name "official_mem0_top${K}" \
    --model-id Qwen/Qwen2.5-7B-Instruct \
    --device cuda \
    --dtype bfloat16 \
    --inference-batch-size 64 \
    --context-window 8192 \
    --overwrite
done
```

Generate submission checks:

```bash
python scripts/prepare_official_mem0_submission_checks.py \
  results/official_mem0_basecompact_full_20260502T072459Z \
  --output-dir results/official_mem0_basecompact_full_20260502T072459Z/submission_checks
```

Run cache-free reader systems replay:

```bash
PYTHONPATH=src python scripts/benchmark_official_mem0_reader_cache_free.py \
  results/official_mem0_basecompact_full_20260502T072459Z \
  --output-dir results/official_mem0_basecompact_full_20260502T072459Z/submission_checks \
  --limit 64 \
  --inference-batch-size 16
```

By default, this replays official Mem0 top-5, the same-evidence adaptive row,
and fixed Mem0 top-1/top-2/top-3 rows.

Run oracle answer-session sanity replay:

```bash
PYTHONPATH=src python scripts/replay_longmemeval_answer_context.py \
  results/official_mem0_basecompact_full_20260502T072459Z \
  --longmemeval data/longmemeval_s_cleaned.json \
  --output-dir results/official_mem0_basecompact_full_20260502T072459Z/submission_checks \
  --limit 64
```

## Output Layout

Official run outputs:

```text
results/<run-id>/official_mem0_summary.csv
results/<run-id>/official_mem0_audit.jsonl
results/<run-id>/official_mem0_longmemeval_*_cases.jsonl
results/<run-id>/official_mem0_top*_cases.jsonl
results/<run-id>/official_mem0_same_evidence_adaptive_cases.jsonl
results/<run-id>/official_mem0_staleaware_gate_cases.jsonl
results/<run-id>/longmemeval_input.sha256
results/<run-id>/logs/run.log
results/<run-id>/logs/run.err
results/<run-id>/diagnostics/*
```

Submission-check outputs:

```text
results/<run-id>/submission_checks/policy_efficiency.csv
results/<run-id>/submission_checks/paired_bootstrap_summary.csv
results/<run-id>/submission_checks/manual_audit_sample.csv
results/<run-id>/submission_checks/manual_audit_sample_reviewed.csv
results/<run-id>/submission_checks/manual_audit_summary.csv
results/<run-id>/submission_checks/state_guard_isolation.csv
results/<run-id>/submission_checks/cache_free_reader_systems.csv
results/<run-id>/submission_checks/official_mem0_same_evidence_adaptive_cache_free_traces.jsonl
results/<run-id>/submission_checks/oracle_answer_context_summary.csv
```

## What To Report

Use the current result as a token-saving systems/work-in-progress paper:

- Report reader-token reduction, retrieved-context reduction, and tokens per correct answer.
- Treat accuracy as a guardrail.
- Compare against naive top-k replay baselines.
- Say explicitly that the headline result is exact same-evidence reader-budgeting.
- Keep ODV2/stale-aware rows as secondary diagnostics.
- Do not compare the low absolute accuracy to Mem0 platform results.

The reviewed manual audit is saved at:

```text
results/<run-id>/submission_checks/manual_audit_sample_reviewed.csv
```

That audit checks whether the local span-match scorer undercounts correctness.
On the reviewed 50-case sample, manual review marks official Mem0 correct on
14/50 cases and ODV2 compact correct on 11/50 cases, while the automatic scorer
marks 13/50 and 9/50 respectively.

## Tests

```bash
python -m pytest
```
