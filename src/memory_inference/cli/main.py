from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from memory_inference.datasets.normalized_io import NormalizedDataset
from memory_inference.datasets.preprocessing import (
    load_preprocessed_locomo,
    load_preprocessed_longmemeval,
    load_raw_locomo_dataset,
    load_raw_longmemeval_dataset,
    preprocess_locomo,
    preprocess_longmemeval,
)
from memory_inference.llm.deterministic_reader import DeterministicValidityReader
from memory_inference.llm.fixed_prompt_reader import FixedPromptReader
from memory_inference.llm.local_config import LocalModelConfig
from memory_inference.llm.local_hf_reasoner import LocalHFReasoner
from memory_inference.orchestration.experiment import run_dataset_experiment
from memory_inference.orchestration.presets import all_policy_factories, policy_factories_for_names


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "preprocess-longmemeval":
        preprocess_longmemeval(args.input, args.output)
        return
    if args.command == "preprocess-locomo":
        preprocess_locomo(args.input, args.output)
        return

    reasoner = build_reasoner(args)
    dataset = load_dataset(args)
    policy_factories = select_policy_factories(args.policy)
    result = run_dataset_experiment(
        benchmark_name=args.command,
        dataset=dataset,
        reasoner=reasoner,
        policy_factories=policy_factories,
        manifest_config=manifest_config(args),
        manifest_output=args.output,
    )
    for row in result.metrics:
        print(
            f"{row.policy_name}: accuracy={row.accuracy:.3f} "
            f"context_tokens={row.avg_context_tokens:.2f} "
            f"latency_ms={row.avg_query_latency_ms:.2f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run validity-aware memory experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess-longmemeval", help="Preprocess LongMemEval-style JSON.")
    preprocess.add_argument("--input", required=True)
    preprocess.add_argument("--output", required=True)

    longmemeval = subparsers.add_parser("longmemeval", help="Run evaluation on LongMemEval-style JSON.")
    _add_benchmark_args(longmemeval)

    preprocess_locomo_parser = subparsers.add_parser("preprocess-locomo", help="Preprocess LoCoMo-style JSON.")
    preprocess_locomo_parser.add_argument("--input", required=True)
    preprocess_locomo_parser.add_argument("--output", required=True)

    locomo = subparsers.add_parser("locomo", help="Run evaluation on LoCoMo-style JSON.")
    _add_benchmark_args(locomo)

    return parser


def build_reasoner(args: argparse.Namespace):
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


def load_dataset(args: argparse.Namespace) -> NormalizedDataset:
    input_format = getattr(args, "input_format", "raw")
    limit = getattr(args, "limit", None)
    if args.command == "longmemeval":
        if input_format == "raw":
            return load_raw_longmemeval_dataset(args.input, limit=limit)
        return load_preprocessed_longmemeval(args.input)
    if args.command == "locomo":
        if input_format == "raw":
            return load_raw_locomo_dataset(args.input, limit=limit)
        return load_preprocessed_locomo(args.input)
    raise ValueError(f"Unsupported benchmark command: {args.command}")


def select_policy_factories(policy_names: Sequence[str]):
    if not policy_names:
        return all_policy_factories()
    return policy_factories_for_names(list(policy_names))


def manifest_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "reasoner": args.reasoner,
        "model_id": args.model_id,
        "policy": list(args.policy),
        "input": args.input,
        "input_format": getattr(args, "input_format", "raw"),
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


def _add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True)
    parser.add_argument("--input-format", choices=["raw", "normalized"], default="raw")
    parser.add_argument("--reasoner", choices=["deterministic", "fixed", "local-hf"], default="deterministic")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--cache-dir", default=".cache/memory_inference")
    parser.add_argument("--output", default="")
    parser.add_argument("--policy", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None, help="Max records to process.")
    _add_local_model_args(parser)


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
