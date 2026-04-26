# Diagnostic Evaluation Framing

This repository should not frame ODV2 as a state-of-the-art replacement for Mem0
on real long-horizon QA benchmarks unless the metrics support that claim. The
defensible research question is:

> When does explicit validity maintenance help or hurt long-horizon agent memory,
> and how should that tradeoff be measured beyond raw QA accuracy?

The boss-facing output is now a single Markdown report:

```bash
python scripts/compile_boss_results.py results/longmemeval_knowledge-update_cases.jsonl
```

It reports only the final comparison: `mem0` vs `odv2_mem0_selective`.

## Run

```bash
bash scripts/run_longmemeval_slice.sh knowledge-update
bash scripts/run_longmemeval_slice.sh temporal-reasoning
```

For LongMemEval iteration, use a small limit before running the full set:

```bash
PYTHONPATH=src python -m memory_inference.cli longmemeval \
  --input data/longmemeval_s_cleaned.json \
  --input-format raw \
  --limit 25 \
  --reasoner local-hf \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --device cuda \
  --dtype bfloat16 \
  --policy mem0 \
  --policy odv2_mem0_selective \
  --cache-dir .cache/memory_inference_longmemeval_debug \
  --output results/longmemeval_debug_25.json \
  --cases-output results/longmemeval_debug_25_cases.jsonl
```

## How To Read The Tables

- `delta_acc < 0` versus Mem0 means the validity variant hurts end-task QA.
- `delta_stale < 0` means the validity variant suppresses more stale state than Mem0.
- `delta_ctx < 0` means the validity variant uses fewer prompt tokens than Mem0.
- `wins` are cases where the validity variant is correct and Mem0 is wrong.
- `losses` are cases where Mem0 is correct and the validity variant is wrong.

The paper claim should be built around these tradeoffs, not around benchmark
dominance.

## Validity-Sensitive Slices

Run the LongMemEval categories most aligned with ODV2:

```bash
bash scripts/run_longmemeval_slice.sh knowledge-update
bash scripts/run_longmemeval_slice.sh temporal-reasoning
```

The highest-priority comparison is `odv2_mem0_selective` against `mem0`.
`odv2_mem0_selective` preserves Mem0 retrieval by default and only applies the
ODV2 ledger when it can confidently suppress stale same-key state or compact
redundant support evidence.
