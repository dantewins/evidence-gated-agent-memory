from __future__ import annotations

import re
import warnings
from typing import Any, Protocol, Sequence

from memory_inference.domain.memory import MemoryRecord
from memory_inference.domain.query import RuntimeQuery

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_DEFAULT_DENSE_MODEL_ID = "intfloat/e5-base-v2"


def normalize_text(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower()))


def query_search_text(query: RuntimeQuery) -> str:
    extra_attributes = " ".join(query.multi_attributes)
    return " ".join(
        part
        for part in (
            query.entity,
            query.attribute,
            extra_attributes,
            query.question,
        )
        if part
    )


def entry_search_text(entry: MemoryRecord) -> str:
    metadata_parts = [
        f"{key}={value}"
        for key, value in sorted(entry.metadata.items())
        if value and key != "support_text"
    ]
    return " ".join(
        part
        for part in (
            entry.entity,
            entry.attribute,
            entry.value,
            entry.support_text,
            entry.provenance,
            " ".join(metadata_parts),
        )
        if part
    )


class DenseEncoder(Protocol):
    def encode_query(self, text: str) -> tuple[float, ...]:
        ...

    def encode_passage(self, text: str) -> tuple[float, ...]:
        ...

    def encode_passages(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        ...

    def similarity(self, left: Sequence[float], right: Sequence[float]) -> float:
        ...


class TransformerDenseEncoder:
    """Standard dense encoder backed by a transformer text embedding model."""

    _BACKEND_CACHE: dict[tuple[str, str], tuple[Any, Any, Any]] = {}

    def __init__(
        self,
        model_id: str = _DEFAULT_DENSE_MODEL_ID,
        *,
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self._cache: dict[tuple[str, str], tuple[float, ...]] = {}
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._device: str | None = None

    def encode_query(self, text: str) -> tuple[float, ...]:
        return self._encode_texts([self._format_query(text)], mode="query")[0]

    def encode_passage(self, text: str) -> tuple[float, ...]:
        return self._encode_texts([self._format_passage(text)], mode="passage")[0]

    def encode_passages(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return self._encode_texts([self._format_passage(text) for text in texts], mode="passage")

    def similarity(self, left: Sequence[float], right: Sequence[float]) -> float:
        return sum(left_value * right_value for left_value, right_value in zip(left, right))

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None and self._torch is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            from transformers.utils import logging as transformers_logging
        except ImportError as exc:
            raise ImportError(
                "TransformerDenseEncoder requires `transformers` and `torch` to be installed."
            ) from exc

        resolved_device = self._resolve_device(torch)
        cache_key = (self.model_id, resolved_device)
        cached_backend = self._BACKEND_CACHE.get(cache_key)
        if cached_backend is not None:
            self._model, self._tokenizer, self._torch = cached_backend
            self._device = resolved_device
            return

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        previous_verbosity = transformers_logging.get_verbosity()
        try:
            transformers_logging.set_verbosity_error()
            loaded = AutoModel.from_pretrained(
                self.model_id,
                output_loading_info=True,
            )
        except TypeError:
            loaded = AutoModel.from_pretrained(self.model_id)
            loading_info: dict[str, Any] = {}
        finally:
            transformers_logging.set_verbosity(previous_verbosity)
        if isinstance(loaded, tuple):
            self._model, loading_info = loaded
        else:
            self._model = loaded
            loading_info = {}
        self._handle_loading_info(loading_info)
        self._device = resolved_device
        self._model = self._model.to(self._device)
        self._model.eval()
        self._BACKEND_CACHE[cache_key] = (self._model, self._tokenizer, self._torch)

    def _encode_texts(self, texts: Sequence[str], *, mode: str) -> list[tuple[float, ...]]:
        self._ensure_loaded()
        cached: list[tuple[float, ...] | None] = []
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for idx, text in enumerate(texts):
            cache_key = (mode, text)
            vector = self._cache.get(cache_key)
            cached.append(vector)
            if vector is None:
                missing_indices.append(idx)
                missing_texts.append(text)

        if missing_texts:
            computed = self._encode_missing(missing_texts)
            for idx, text, vector in zip(missing_indices, missing_texts, computed):
                self._cache[(mode, text)] = vector
                cached[idx] = vector

        return [vector for vector in cached if vector is not None]

    def _encode_missing(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None
        assert self._device is not None

        outputs: list[tuple[float, ...]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            encoded = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            encoded = {
                key: value.to(self._device)
                for key, value in encoded.items()
            }
            with self._torch.no_grad():
                model_output = self._model(**encoded)
            hidden_states = getattr(model_output, "last_hidden_state", model_output[0])
            pooled = self._mean_pool(hidden_states, encoded["attention_mask"])
            normalized = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
            outputs.extend(tuple(float(value) for value in row) for row in normalized.cpu().tolist())
        return outputs

    def _mean_pool(self, hidden_states: Any, attention_mask: Any) -> Any:
        expanded_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        weighted_sum = (hidden_states * expanded_mask).sum(dim=1)
        counts = expanded_mask.sum(dim=1).clamp(min=1e-9)
        return weighted_sum / counts

    def _resolve_device(self, torch: Any) -> str:
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def _format_query(self, text: str) -> str:
        return f"query: {text}"

    def _format_passage(self, text: str) -> str:
        return f"passage: {text}"

    def _handle_loading_info(self, loading_info: dict[str, Any]) -> None:
        benign_unexpected = {"embeddings.position_ids"}
        unexpected = [
            key
            for key in loading_info.get("unexpected_keys", [])
            if key not in benign_unexpected
        ]
        missing = list(loading_info.get("missing_keys", []))
        mismatched = list(loading_info.get("mismatched_keys", []))
        errors = [message for message in loading_info.get("error_msgs", []) if message]

        if mismatched or errors:
            raise RuntimeError(
                f"Unexpected dense encoder load issues for {self.model_id}: "
                f"mismatched={mismatched!r} errors={errors!r}"
            )
        if missing or unexpected:
            warnings.warn(
                f"Dense encoder load for {self.model_id} reported "
                f"missing={missing!r} unexpected={unexpected!r}.",
                stacklevel=2,
            )
