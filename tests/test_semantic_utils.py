import sys
import types
import warnings

import pytest

from memory_inference.consolidation.semantic_utils import TransformerDenseEncoder


class _DummyTokenizer:
    pad_token_id = None
    eos_token_id = 7

    @classmethod
    def from_pretrained(cls, model_id):
        return cls()


class _DummyModel:
    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.is_eval = True


def _install_fake_transformers(monkeypatch, *, loading_info):
    auto_model_calls = []
    auto_tokenizer_calls = []
    logging_state = {"verbosity": 5}

    class _DummyAutoModel:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            auto_model_calls.append(kwargs)
            return _DummyModel(), loading_info

    class _TrackingTokenizer(_DummyTokenizer):
        @classmethod
        def from_pretrained(cls, model_id):
            auto_tokenizer_calls.append(model_id)
            return cls()

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=None),
    )
    fake_logging = types.SimpleNamespace(
        get_verbosity=lambda: logging_state["verbosity"],
        set_verbosity=lambda value: logging_state.__setitem__("verbosity", value),
        set_verbosity_error=lambda: logging_state.__setitem__("verbosity", -1),
    )
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModel = _DummyAutoModel
    fake_transformers.AutoTokenizer = _TrackingTokenizer
    fake_transformers_utils = types.ModuleType("transformers.utils")
    fake_transformers_utils.logging = fake_logging

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.utils", fake_transformers_utils)
    TransformerDenseEncoder._BACKEND_CACHE.clear()
    return auto_model_calls, auto_tokenizer_calls, logging_state


def test_transformer_dense_encoder_suppresses_position_id_warning(monkeypatch) -> None:
    auto_model_calls, _, logging_state = _install_fake_transformers(
        monkeypatch,
        loading_info={
            "unexpected_keys": ["embeddings.position_ids"],
            "missing_keys": [],
            "mismatched_keys": [],
            "error_msgs": [],
        },
    )
    encoder = TransformerDenseEncoder()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        encoder._ensure_loaded()

    assert caught == []
    assert auto_model_calls[0]["output_loading_info"] is True
    assert logging_state["verbosity"] == 5


def test_transformer_dense_encoder_warns_on_other_unexpected_keys(monkeypatch) -> None:
    _install_fake_transformers(
        monkeypatch,
        loading_info={
            "unexpected_keys": ["encoder.bad_key"],
            "missing_keys": [],
            "mismatched_keys": [],
            "error_msgs": [],
        },
    )
    encoder = TransformerDenseEncoder()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        encoder._ensure_loaded()

    assert len(caught) == 1
    assert "encoder.bad_key" in str(caught[0].message)


def test_transformer_dense_encoder_raises_on_mismatched_weights() -> None:
    encoder = TransformerDenseEncoder()

    with pytest.raises(RuntimeError):
        encoder._handle_loading_info(
            {
                "unexpected_keys": [],
                "missing_keys": [],
                "mismatched_keys": ["embeddings.word_embeddings.weight"],
                "error_msgs": [],
            }
        )


def test_transformer_dense_encoder_reuses_loaded_backend_across_instances(monkeypatch) -> None:
    auto_model_calls, auto_tokenizer_calls, _ = _install_fake_transformers(
        monkeypatch,
        loading_info={
            "unexpected_keys": ["embeddings.position_ids"],
            "missing_keys": [],
            "mismatched_keys": [],
            "error_msgs": [],
        },
    )

    first = TransformerDenseEncoder()
    second = TransformerDenseEncoder()
    first._ensure_loaded()
    second._ensure_loaded()

    assert len(auto_model_calls) == 1
    assert len(auto_tokenizer_calls) == 1
    assert first._model is second._model
    assert first._tokenizer is second._tokenizer
