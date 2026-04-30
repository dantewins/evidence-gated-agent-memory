# Technical Brief: Validity-Aware Memory for Frozen Long-Horizon LLM Agents

## Overview

This project studies a narrow but important memory problem in long-horizon LLM agents:

**when the base model is frozen, can an external memory layer improve inference by explicitly maintaining which information is still valid over time?**

The project is not framed around the general claim that LLM agents need memory. That point is already well established. There is substantial prior work on retrieval-augmented generation, memory streams, summarization, reflection, hierarchical memory, and long-term memory systems for agents. The focus here is more specific: **semantic memory updating under frozen-model interference**.

The motivating observation is that long-running agents often accumulate multiple versions of the same fact, preference, or task state. In these settings, the failure mode is not simply missing retrieval. A retrieval system may surface relevant evidence while still leaving the frozen model to infer which memory version supersedes which, whether an earlier state has been restored, whether two memories conflict, or whether two values can coexist under different scopes. This project asks whether making that validity structure explicit in the external memory layer improves downstream inference.

## Research goal

The central research question is:

**With model weights frozen, how does explicit memory-state maintenance affect inference quality, robustness, and efficiency in long-horizon LLM agents?**

The scope is intentionally restricted to inference-time memory methods:

- no fine-tuning
- no model weight updates
- no retriever training during evaluation
- only the external memory layer changes

The target setting includes memory streams containing:

- revisions such as `v1 -> v2`
- reversions such as `v1 -> v2 -> v1`
- unresolved contradictions
- scope-dependent coexistence
- low-confidence or noisy updates

## Relationship to current methods

The project does not assume that existing work ignores memory updating. Several relevant lines of work already address storage, retrieval, compression, and explicit memory management.

Representative prior work includes:

- [RAG](https://arxiv.org/abs/2005.11401), which established external non-parametric memory as a mechanism for improving access to knowledge without changing model weights
- [Generative Agents](https://arxiv.org/abs/2304.03442), which emphasized memory streams, reflection, and synthesis
- [MemGPT](https://arxiv.org/abs/2310.08560), which framed long-horizon agent memory as hierarchical context management and paging
- [LoCoMo](https://arxiv.org/abs/2402.17753) and [LongMemEval](https://arxiv.org/abs/2410.10813), which evaluate long-term conversational memory and update-sensitive reasoning
- [Mem0](https://arxiv.org/abs/2504.19413), which moves toward production long-term memory with explicit extraction and memory management

Because of this prior literature, the gap addressed here is not “memory updating does not exist.” A more precise characterization is:

**Existing systems can store, retrieve, compress, and in some cases update memory, but fewer methods explicitly model semantic memory updating as a validity-maintenance problem under frozen-model interference.**

This is the point of differentiation. The project is specifically concerned with whether the external memory system can maintain a compact, queryable representation of **current semantic state** rather than requiring the frozen model to reconstruct that state from raw or weakly consolidated history at inference time.

## Gap statement

The gap targeted by this project is:

**a frozen model may still have to infer validity from a semantically overlapping history of memories, even when the memory system has already stored all relevant evidence.**

That gap appears in several concrete forms:

- the system may retrieve both old and new versions of the same fact
- the model may need to infer whether a later update supersedes, conflicts with, or coexists with an earlier one
- the memory layer may fail to distinguish reversion from ordinary overwrite
- low-confidence or aliased memories may remain visible to inference even when they should not affect active state

The research hypothesis is that explicit validity-aware memory maintenance can reduce stale-memory interference and lower the amount of history that must be surfaced during inference.

## Proposed solution direction

The method explored in this repository is a **validity-aware external memory architecture** with an **incremental consolidation layer**.

At a high level, the system maintains:

- a full episodic log
- a compact current-state store
- an archive for superseded or low-confidence entries
- an explicit conflict table for unresolved contradictions

New memory updates are classified into validity-relevant operations such as:

- new
- reinforcing
- superseding
- reverting
- scope-splitting
- unresolved conflict
- low confidence

Rather than rewriting the full memory store, the method updates only the affected slice of state. At inference time, the system retrieves current-state memory first and falls back to provenance or history only when needed.

The intended benefit is not only lower context cost. More importantly, the method is designed to give the agent an explicit mechanism for **semantic memory updating**, so that the frozen model is not forced to infer the full revision structure from a noisy history on every query.

## Evaluation design

The evaluation path in this repository is focused on two real conversational memory benchmarks. The current codebase runs:

- LongMemEval-style question answering over long conversational histories
- LoCoMo-style question answering over multi-session conversational histories

Those runs are evaluated with downstream task metrics such as QA accuracy, context size, and latency under a frozen reader. The main comparison is therefore between memory policies, not between custom benchmark generators.

## Final reporting path

The current reviewer-facing comparison is intentionally narrow:

- baseline: `mem0`
- target: `odv2_mem0_selective`
- primary benchmark: selected LongMemEval current-state categories
- diagnostic slices: ODV2 intervention cases, gold-mismatched same-key exposure, and same-key conflicts

Run the stronger results package with Mem0, ODV2 ablations, full ODV2-selective,
and an aggressive negative-control variant:

```bash
bash scripts/run_stronger_results_package.sh
```

This writes:

- `results/stronger_results_summary.csv`
- `results/stronger_results_audit.jsonl`
- `results/longmemeval_input.sha256` when the input file is available

The summary CSV reports all paired cases plus predeclared diagnostic slices:

- `Predeclared validity-sensitive union`
- `Current-state same-key evidence retrieved`
- `Gold-mismatched same-key state exposed`
- `Same-key state conflict exposed`
- `ODV2 intervened`

To run only one slice with the default Mem0 vs ODV2-selective comparison:


```bash
bash scripts/run_longmemeval_slice.sh knowledge-update
```

To run one slice with ablations:

```bash
POLICIES="mem0 odv2_support_compact odv2_stale_guard odv2_mem0_selective odv2_mem0_aggressive" \
  bash scripts/run_longmemeval_slice.sh multi-session
```

To run the official Mem0 comparison with a local Llama-backed Mem0 OSS stack:

```bash
pip install ".[official-mem0]"
ollama pull llama3.1:8b
ollama pull nomic-embed-text:latest
bash scripts/run_official_mem0_package.sh
```

The official Mem0 path defaults to local providers:

- `MEM0_LLM_PROVIDER=ollama`
- `MEM0_LLM_MODEL=llama3.1:8b`
- `MEM0_EMBEDDER_PROVIDER=ollama`
- `MEM0_EMBEDDER_MODEL=nomic-embed-text:latest`
- `MEM0_VECTOR_STORE_PROVIDER=qdrant`

The Llama setting controls Mem0's memory-extraction LLM. The embedder remains a
separate embedding model because Mem0 still needs vector representations for
search; this is an implementation detail for a valid official-Mem0 comparison,
not a separate research claim.

For a vLLM-served local Llama model, start vLLM separately and run:

```bash
MEM0_LLM_PROVIDER=vllm \
MEM0_LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct \
VLLM_BASE_URL=http://localhost:8000/v1 \
bash scripts/run_official_mem0_package.sh
```

Compile an existing result without rerunning the benchmark:

```bash
python scripts/compile_stronger_results.py \
  --audit-output results/stronger_results_audit.jsonl \
  results/longmemeval_knowledge-update_cases.jsonl \
  results/longmemeval_multi-session_cases.jsonl \
  > results/stronger_results_summary.csv
```

The report is designed to answer one question clearly: does ODV2 preserve Mem0
accuracy while improving prompt context, and which ODV2 component causes the
effect? The gold-mismatch columns are offline diagnostics, not a claim that the
runtime system has access to gold labels.

## Repository implementation

This repository serves as the experimental scaffold for that research direction. It currently includes:

- a typed memory schema with explicit validity-state fields
- multiple baseline memory policies
- a validity-aware incremental policy
- real-benchmark adapters for LongMemEval-style and LoCoMo-style inputs
- evaluation metrics for QA, context footprint, and latency
- deterministic readers and a local Hugging Face reasoner path
- dense retrieval and Mem0-style baselines built on the same encoder stack

## Intended contribution

The intended contribution is not a generic claim about memory for agents. A more defensible formulation is:

**This project studies whether external memory can explicitly maintain semantic validity under revision and conflict, and whether that reduces stale-memory interference for frozen LLM agents.**

That framing aligns the method, benchmark, and evaluation target. It also positions the work more carefully relative to current literature by focusing on validity maintenance as a distinct technical problem within the broader agent-memory space.

## References

1. Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (NeurIPS 2020).  
   https://arxiv.org/abs/2005.11401

2. Park et al., *Generative Agents: Interactive Simulacra of Human Behavior* (UIST 2023).  
   https://arxiv.org/abs/2304.03442

3. Packer et al., *MemGPT: Towards LLMs as Operating Systems* (2024).  
   https://arxiv.org/abs/2310.08560

4. Maharana et al., *Evaluating Very Long-Term Conversational Memory of LLM Agents* (LoCoMo, 2024).  
   https://arxiv.org/abs/2402.17753

5. Wu et al., *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory* (ICLR 2025).  
   https://arxiv.org/abs/2410.10813

6. Chhikara et al., *Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory* (2025).  
   https://arxiv.org/abs/2504.19413
