from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Callable, Sequence

from memory_inference.agent import AgentRunner
from memory_inference.benchmarks.locomo_adapter import LoCoMoAdapter
from memory_inference.benchmarks.locomo_preprocess import load_preprocessed_locomo, preprocess_locomo
from memory_inference.benchmarks.longmemeval_preprocess import load_preprocessed_longmemeval, preprocess_longmemeval
from memory_inference.experiment_registry import (
    ablation_policy_factories,
    build_synthetic_batches,
    default_policy_factories,
    load_longmemeval_batches,
    policy_factory_by_name,
)
from memory_inference.llm.deterministic_reader import DeterministicValidityReader
from memory_inference.llm.fixed_prompt_reader import FixedPromptReader
from memory_inference.llm.local_config import LocalModelConfig
from memory_inference.llm.local_hf_reasoner import LocalHFReasoner
from memory_inference.llm.mock_consolidator import MockConsolidator
from memory_inference.metrics import compute_metrics
from memory_inference.results import build_manifest, write_manifest
from memory_inference.run_experiment import evaluate_policy


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run validity-aware memory experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser("synthetic", help="Run the synthetic revision benchmark.")
    synthetic.add_argument("--reasoner", choices=["deterministic", "fixed", "local-hf"], default="deterministic")
    synthetic.add_argument("--model-id", default="")
    synthetic.add_argument("--cache-dir", default=".cache/memory_inference")
    synthetic.add_argument("--output", default="")
    synthetic.add_argument("--policy", action="append", default=[])
    synthetic.add_argument("--ablation", action="store_true", help="Run ablation variants of ODV2.")
    synthetic.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    _add_local_model_args(synthetic)

    preprocess = subparsers.add_parser("preprocess-longmemeval", help="Preprocess LongMemEval-style JSON.")
    preprocess.add_argument("--input", required=True)
    preprocess.add_argument("--output", required=True)

    longmemeval = subparsers.add_parser("longmemeval", help="Run evaluation on LongMemEval-style JSON.")
    longmemeval.add_argument("--input", required=True)
    longmemeval.add_argument("--preprocessed", action="store_true")
    longmemeval.add_argument("--input-format", choices=["raw", "normalized"], default="normalized")
    longmemeval.add_argument("--reasoner", choices=["deterministic", "fixed", "local-hf"], default="deterministic")
    longmemeval.add_argument("--model-id", default="")
    longmemeval.add_argument("--cache-dir", default=".cache/memory_inference")
    longmemeval.add_argument("--output", default="")
    longmemeval.add_argument("--policy", action="append", default=[])
    longmemeval.add_argument("--limit", type=int, default=None, help="Max records to process.")
    longmemeval.add_argument("--seed", type=int, default=42)
    _add_local_model_args(longmemeval)

    preprocess_locomo_parser = subparsers.add_parser("preprocess-locomo", help="Preprocess LoCoMo-style JSON.")
    preprocess_locomo_parser.add_argument("--input", required=True)
    preprocess_locomo_parser.add_argument("--output", required=True)

    locomo = subparsers.add_parser("locomo", help="Run evaluation on LoCoMo-style JSON.")
    locomo.add_argument("--input", required=True)
    locomo.add_argument("--preprocessed", action="store_true")
    locomo.add_argument("--input-format", choices=["raw", "normalized"], default="normalized")
    locomo.add_argument("--reasoner", choices=["deterministic", "fixed", "local-hf"], default="deterministic")
    locomo.add_argument("--model-id", default="")
    locomo.add_argument("--cache-dir", default=".cache/memory_inference")
    locomo.add_argument("--output", default="")
    locomo.add_argument("--policy", action="append", default=[])
    locomo.add_argument("--limit", type=int, default=None, help="Max records to process.")
    locomo.add_argument("--seed", type=int, default=42)
    _add_local_model_args(locomo)

    args = parser.parse_args(argv)

    if args.command == "synthetic":
        _run_synthetic(args)
        return
    if args.command == "preprocess-longmemeval":
        preprocess_longmemeval(args.input, args.output)
        return
    if args.command == "longmemeval":
        _run_longmemeval(args)
        return
    if args.command == "preprocess-locomo":
        preprocess_locomo(
            LoCoMoAdapter(consolidator=MockConsolidator(), cache_path=None),
            args.input,
            args.output,
        )
        return
    if args.command == "locomo":
        _run_locomo(args)
        return


def _run_synthetic(args: argparse.Namespace) -> None:
    scenarios = build_synthetic_batches()
    reasoner = _build_reasoner(args)
    if getattr(args, "ablation", False):
        policies = ablation_policy_factories()
    else:
        policies = _selected_policy_factories(args.policy)
    metrics = [evaluate_policy(factory, reasoner, scenarios) for factory in policies]
    for row in metrics:
        print(
            f"{row.policy_name}: accuracy={row.accuracy:.3f} "
            f"state_em={row.current_state_exact_match:.3f} "
            f"amortized_tokens={row.amortized_end_to_end_tokens:.2f}"
        )
    if args.output:
        write_manifest(
            args.output,
            build_manifest(
                benchmark="synthetic_revision" + ("_ablation" if getattr(args, "ablation", False) else ""),
                reasoner=reasoner.__class__.__name__,
                policy_names=[row.policy_name for row in metrics],
                metrics=[asdict(row) for row in metrics],
                config=_manifest_config(args),
            ),
        )


def _build_reasoner(args: argparse.Namespace):
    if args.reasoner == "fixed":
        return FixedPromptReader()
    if args.reasoner == "local-hf":
        if not args.model_id:
            raise ValueError("--model-id is required for local-hf runs")
        return LocalHFReasoner(
            LocalModelConfig(
                model_id=args.model_id,
                cache_dir=Path(args.cache_dir),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample,
                repetition_penalty=args.repetition_penalty,
                device=args.device,
                dtype=args.dtype,
                prompt_template_id=args.prompt_template_id,
                trust_remote_code=args.trust_remote_code,
                use_chat_template=not args.no_chat_template,
            )
        )
    return DeterministicValidityReader()


def _run_longmemeval(args: argparse.Namespace) -> None:
    input_format = getattr(args, "input_format", "normalized")
    limit = getattr(args, "limit", None)
    if args.preprocessed:
        batches = load_preprocessed_longmemeval(args.input)
    elif input_format == "raw":
        from memory_inference.benchmarks.longmemeval_raw import load_raw_longmemeval
        batches = load_raw_longmemeval(args.input, limit=limit)
    else:
        batches = load_longmemeval_batches(args.input)
    reasoner = _build_reasoner(args)
    _run_structured_batches(
        benchmark_name="longmemeval",
        batches=batches,
        reasoner=reasoner,
        policy_names=args.policy,
        output=args.output,
        manifest_config={**_manifest_config(args), "input": args.input, "preprocessed": args.preprocessed},
    )


def _run_locomo(args: argparse.Namespace) -> None:
    input_format = getattr(args, "input_format", "normalized")
    limit = getattr(args, "limit", None)
    if args.preprocessed:
        batches = load_preprocessed_locomo(args.input)
    elif input_format == "raw":
        from memory_inference.benchmarks.locomo_raw import load_raw_locomo
        batches = load_raw_locomo(args.input, limit=limit)
    else:
        batches = LoCoMoAdapter(consolidator=MockConsolidator(), cache_path=None).from_records(
            json.loads(Path(args.input).read_text())
        )
    reasoner = _build_reasoner(args)
    _run_structured_batches(
        benchmark_name="locomo",
        batches=batches,
        reasoner=reasoner,
        policy_names=args.policy,
        output=args.output,
        manifest_config={**_manifest_config(args), "input": args.input, "preprocessed": args.preprocessed},
    )


def _run_structured_batches(
    *,
    benchmark_name: str,
    batches,
    reasoner,
    policy_names: Sequence[str],
    output: str,
    manifest_config: dict[str, object],
) -> None:
    batch_list = list(batches)
    policies = _selected_policy_factories(policy_names)
    metrics = []
    for factory in policies:
        policy = factory()
        runner = AgentRunner(policy=policy, reasoner=reasoner)
        examples = runner.run_batches(batch_list)
        row = compute_metrics(
            policy.name,
            examples,
            snapshot_sizes=[policy.snapshot_size()],
            maintenance_tokens=policy.maintenance_tokens,
            maintenance_latency_ms=policy.maintenance_latency_ms,
        )
        metrics.append(row)
        print(
            f"{row.policy_name}: accuracy={row.accuracy:.3f} "
            f"context_tokens={row.avg_context_tokens:.2f} "
            f"latency_ms={row.avg_query_latency_ms:.2f}"
        )
    if output:
        write_manifest(
            output,
            build_manifest(
                benchmark=benchmark_name,
                reasoner=reasoner.__class__.__name__,
                policy_names=[row.policy_name for row in metrics],
                metrics=[asdict(row) for row in metrics],
                config=manifest_config,
            ),
        )


def _selected_policy_factories(names: Sequence[str]) -> list[Callable[[], object]]:
    if not names:
        return list(default_policy_factories())
    return [policy_factory_by_name(name) for name in names]


def _add_local_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--prompt-template-id", default="validity-v1")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")


def _manifest_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "reasoner": args.reasoner,
        "model_id": args.model_id,
        "policy": list(args.policy),
        "cache_dir": getattr(args, "cache_dir", ""),
        "max_new_tokens": getattr(args, "max_new_tokens", None),
        "temperature": getattr(args, "temperature", None),
        "top_p": getattr(args, "top_p", None),
        "do_sample": getattr(args, "do_sample", None),
        "repetition_penalty": getattr(args, "repetition_penalty", None),
        "device": getattr(args, "device", None),
        "dtype": getattr(args, "dtype", None),
        "prompt_template_id": getattr(args, "prompt_template_id", None),
        "trust_remote_code": getattr(args, "trust_remote_code", None),
        "use_chat_template": not getattr(args, "no_chat_template", False),
    }


if __name__ == "__main__":
    main()
