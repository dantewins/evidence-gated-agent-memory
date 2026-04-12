from __future__ import annotations

from pathlib import Path

from memory_inference.benchmarks.locomo_raw import preprocess_raw_locomo
from memory_inference.benchmarks.normalized_schema import load_normalized, serialize_normalized
from memory_inference.types import BenchmarkBatch


def preprocess_locomo(
    records_path: str | Path,
    output_path: str | Path,
) -> list[BenchmarkBatch]:
    dataset = preprocess_raw_locomo(records_path)
    serialize_normalized(dataset, output_path)
    return [record.batch for record in dataset.records]


def load_preprocessed_locomo(path: str | Path) -> list[BenchmarkBatch]:
    dataset = load_normalized(path)
    return [record.batch for record in dataset.records]
