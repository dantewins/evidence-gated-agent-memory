from __future__ import annotations

import json
import math
import os
import sys
import urllib.error
import urllib.request
from typing import Any


def main() -> int:
    if os.getenv("MEM0_LLM_PROVIDER", "").strip().lower() != "vllm":
        return 0
    if _env_bool("MEM0_SKIP_VLLM_BUDGET_CHECK", False):
        print("official_mem0 vLLM budget check skipped")
        return 0

    budget = VLLMBudget.from_env()
    validate_budget(budget)
    print(
        "official_mem0 vLLM budget ok: "
        f"estimated_request_tokens={budget.estimated_request_tokens} "
        f"max_model_len={budget.max_model_len} "
        f"batch_size={budget.batch_size} "
        f"max_message_chars={budget.max_message_chars} "
        f"max_tokens={budget.max_tokens}"
    )
    return 0


class VLLMBudget:
    def __init__(
        self,
        *,
        max_model_len: int,
        batch_size: int,
        max_message_chars: int,
        max_tokens: int,
        chars_per_token: float,
        prompt_overhead_tokens: int,
        margin_tokens: int,
    ) -> None:
        self.max_model_len = max_model_len
        self.batch_size = batch_size
        self.max_message_chars = max_message_chars
        self.max_tokens = max_tokens
        self.chars_per_token = chars_per_token
        self.prompt_overhead_tokens = prompt_overhead_tokens
        self.margin_tokens = margin_tokens

    @classmethod
    def from_env(cls) -> "VLLMBudget":
        return cls(
            max_model_len=_vllm_max_model_len_from_env_or_server(),
            batch_size=max(1, _env_int("MEM0_ADD_BATCH_SIZE", 8)),
            max_message_chars=max(0, _env_int("MEM0_ADD_MAX_MESSAGE_CHARS", 2000)),
            max_tokens=max(1, _env_int("MEM0_LLM_MAX_TOKENS", 512)),
            chars_per_token=max(1.0, _env_float("MEM0_VLLM_CHARS_PER_TOKEN", 4.0)),
            prompt_overhead_tokens=max(0, _env_int("MEM0_VLLM_PROMPT_OVERHEAD_TOKENS", 2000)),
            margin_tokens=max(0, _env_int("MEM0_VLLM_CONTEXT_MARGIN_TOKENS", 512)),
        )

    @property
    def estimated_prompt_tokens(self) -> int:
        if self.max_message_chars <= 0:
            message_tokens = 0
        else:
            message_tokens = math.ceil(
                (self.batch_size * self.max_message_chars) / self.chars_per_token
            )
        return message_tokens + self.prompt_overhead_tokens

    @property
    def estimated_request_tokens(self) -> int:
        return self.estimated_prompt_tokens + self.max_tokens

    @property
    def safe_batch_size(self) -> int:
        if self.max_message_chars <= 0:
            return self.batch_size
        available = (
            self.max_model_len
            - self.max_tokens
            - self.prompt_overhead_tokens
            - self.margin_tokens
        )
        return max(1, math.floor((available * self.chars_per_token) / self.max_message_chars))


def validate_budget(budget: VLLMBudget) -> None:
    if budget.estimated_request_tokens + budget.margin_tokens <= budget.max_model_len:
        return
    raise RuntimeError(
        "Official Mem0 vLLM preflight failed: estimated Mem0 add request is too large "
        "for the vLLM context window. "
        f"estimated_request_tokens={budget.estimated_request_tokens}, "
        f"margin_tokens={budget.margin_tokens}, max_model_len={budget.max_model_len}, "
        f"MEM0_ADD_BATCH_SIZE={budget.batch_size}, "
        f"MEM0_ADD_MAX_MESSAGE_CHARS={budget.max_message_chars}, "
        f"MEM0_LLM_MAX_TOKENS={budget.max_tokens}. "
        f"For this server, use MEM0_ADD_BATCH_SIZE<={budget.safe_batch_size}, lower "
        "MEM0_ADD_MAX_MESSAGE_CHARS/MEM0_LLM_MAX_TOKENS, or restart vLLM with a larger "
        "--max-model-len."
    )


def _vllm_max_model_len_from_env_or_server() -> int:
    env_value = os.getenv("VLLM_MAX_MODEL_LEN")
    if env_value:
        return max(1, int(env_value))

    base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
    url = f"{base_url}/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Official Mem0 vLLM preflight could not query the vLLM model endpoint. "
            f"Set VLLM_BASE_URL correctly or provide VLLM_MAX_MODEL_LEN. url={url!r}"
        ) from exc

    max_model_len = _extract_max_model_len(payload)
    if max_model_len is None:
        raise RuntimeError(
            "Official Mem0 vLLM preflight could not determine max_model_len from "
            f"{url!r}. Set VLLM_MAX_MODEL_LEN explicitly."
        )
    return max_model_len


def _extract_max_model_len(payload: Any) -> int | None:
    if isinstance(payload, dict):
        raw_value = payload.get("max_model_len")
        if raw_value is not None:
            return int(raw_value)
        for key in ("data", "models"):
            nested = payload.get(key)
            found = _extract_max_model_len(nested)
            if found is not None:
                return found
    if isinstance(payload, list):
        for item in payload:
            found = _extract_max_model_len(item)
            if found is not None:
                return found
    return None


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
