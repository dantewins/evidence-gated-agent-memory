from __future__ import annotations

from pathlib import Path

from memory_inference.datasets.compiler import compile_locomo_samples, compile_longmemeval_records
from memory_inference.datasets.normalized_io import NormalizedDataset, load_normalized, serialize_normalized
from memory_inference.ingestion.locomo_loader import load_locomo_samples
from memory_inference.ingestion.longmemeval_loader import load_longmemeval_records


def load_raw_longmemeval_dataset(
    path: str | Path,
    *,
    split: str = "default",
    limit: int | None = None,
) -> NormalizedDataset:
    records = load_longmemeval_records(path, limit=limit)
    return compile_longmemeval_records(records, split=split)


def load_raw_locomo_dataset(
    path: str | Path,
    *,
    split: str = "default",
    limit: int | None = None,
) -> NormalizedDataset:
    samples = load_locomo_samples(path, limit=limit)
    return compile_locomo_samples(samples, split=split)


def preprocess_longmemeval(source_path: str | Path, output_path: str | Path) -> NormalizedDataset:
    dataset = load_raw_longmemeval_dataset(source_path)
    serialize_normalized(dataset, output_path)
    return dataset


def preprocess_locomo(source_path: str | Path, output_path: str | Path) -> NormalizedDataset:
    dataset = load_raw_locomo_dataset(source_path)
    serialize_normalized(dataset, output_path)
    return dataset


def load_preprocessed_longmemeval(path: str | Path) -> NormalizedDataset:
    return load_normalized(path)


def load_preprocessed_locomo(path: str | Path) -> NormalizedDataset:
    return load_normalized(path)
