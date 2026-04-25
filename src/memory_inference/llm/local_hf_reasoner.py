from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import re
import time
from typing import Any, Optional, Sequence

from memory_inference.domain.memory import MemoryRecord
from memory_inference.domain.query import RuntimeQuery
from memory_inference.llm.base import BaseReasoner, ReasonerTrace
from memory_inference.llm.cache import ResponseCache, cache_key
from memory_inference.llm.local_config import LocalModelConfig
from memory_inference.llm.prompting import build_reasoning_prompt, render_prompt


@dataclass(slots=True)
class _PreparedPrompt:
    index: int
    rendered_prompt: str
    cache_key_value: str
    template_id: str


class LocalHFReasoner(BaseReasoner):
    """Local Hugging Face reasoner for real frozen-model experiments."""

    def __init__(self, config: LocalModelConfig) -> None:
        self.config = config
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._cache = ResponseCache(config.cache_dir) if config.cache_dir is not None else None

    def answer(self, query: RuntimeQuery, context: Sequence[MemoryRecord]) -> str:
        return self.answer_with_trace(query, context).answer

    def answer_with_trace(
        self,
        query: RuntimeQuery,
        context: Sequence[MemoryRecord],
    ) -> ReasonerTrace:
        return self.answer_many_with_traces([query], [context])[0]

    def answer_many_with_traces(
        self,
        queries: Sequence[RuntimeQuery],
        contexts: Sequence[Sequence[MemoryRecord]],
    ) -> list[ReasonerTrace]:
        if len(queries) != len(contexts):
            raise ValueError(
                f"Expected the same number of queries and contexts, got {len(queries)} and {len(contexts)}."
            )
        self._ensure_loaded()
        prepared = [
            self._prepare_prompt(index, query, context)
            for index, (query, context) in enumerate(zip(queries, contexts))
        ]
        traces: list[ReasonerTrace | None] = [None] * len(prepared)
        uncached: list[_PreparedPrompt] = []

        for prompt in prepared:
            cached = self._load_cached_trace(prompt.cache_key_value)
            if cached is not None:
                traces[prompt.index] = cached
                continue
            uncached.append(prompt)

        if uncached:
            for prompt, trace in zip(uncached, self._generate_missing_traces(uncached)):
                traces[prompt.index] = trace
                if self._cache is not None:
                    self._cache.save(prompt.cache_key_value, trace)

        return [trace for trace in traces if trace is not None]

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "LocalHFReasoner requires `transformers` and `torch` to be installed."
            ) from exc

        self._torch = torch
        self._configure_torch_runtime(torch)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        if hasattr(self._tokenizer, "padding_side"):
            self._tokenizer.padding_side = "left"
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
            **self._model_load_kwargs(torch),
        )
        if self.config.device != "auto":
            self._model = self._model.to(self.config.device)
        self._configure_generation_defaults()
        self._model.eval()

    def _decode_completion(self, completion_ids: Any) -> str:
        return self._tokenizer.decode(completion_ids, skip_special_tokens=True)

    def _token_length(self, token_ids: Any) -> int:
        shape = getattr(token_ids, "shape", None)
        if shape is not None:
            return int(shape[-1])
        return len(token_ids)

    def _model_load_kwargs(self, torch: Any) -> dict[str, Any]:
        model_kwargs: dict[str, Any] = {}
        if self.config.device == "auto":
            model_kwargs["device_map"] = "auto"
        if self.config.dtype != "auto":
            model_kwargs["dtype"] = getattr(torch, self.config.dtype)
        attention_impl = self._attention_implementation(torch)
        if attention_impl is not None:
            model_kwargs["attn_implementation"] = attention_impl
        return model_kwargs

    def _configure_torch_runtime(self, torch: Any) -> None:
        if not self._using_cuda(torch):
            return
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        cuda_backend = getattr(getattr(torch, "backends", None), "cuda", None)
        if cuda_backend is not None and hasattr(cuda_backend, "matmul"):
            cuda_backend.matmul.allow_tf32 = True
        cudnn_backend = getattr(getattr(torch, "backends", None), "cudnn", None)
        if cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32"):
            cudnn_backend.allow_tf32 = True

    def _attention_implementation(self, torch: Any) -> str | None:
        if not self._using_cuda(torch):
            return None
        if importlib.util.find_spec("flash_attn") is not None:
            return "flash_attention_2"
        return "sdpa"

    def _using_cuda(self, torch: Any) -> bool:
        if isinstance(self.config.device, str) and self.config.device.startswith("cuda"):
            return True
        cuda = getattr(torch, "cuda", None)
        return bool(cuda is not None and cuda.is_available())

    def _generate_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "repetition_penalty": self.config.repetition_penalty,
            "pad_token_id": getattr(self._tokenizer, "eos_token_id", None),
        }
        if self.config.do_sample:
            kwargs["temperature"] = self.config.temperature
            kwargs["top_p"] = self.config.top_p
        return kwargs

    def _configure_generation_defaults(self) -> None:
        generation_config = getattr(self._model, "generation_config", None)
        if generation_config is None:
            return
        generation_config.do_sample = self.config.do_sample
        generation_config.max_new_tokens = self.config.max_new_tokens
        generation_config.repetition_penalty = self.config.repetition_penalty
        if self.config.do_sample:
            generation_config.temperature = self.config.temperature
            generation_config.top_p = self.config.top_p
            return
        if hasattr(generation_config, "temperature"):
            generation_config.temperature = 1.0
        if hasattr(generation_config, "top_p"):
            generation_config.top_p = 1.0
        if hasattr(generation_config, "top_k"):
            generation_config.top_k = 50

    def _extract_answer(self, generated_text: str, prompt: str) -> str:
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]
        cleaned = generated_text.strip()
        if not cleaned:
            return "UNKNOWN"
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*", "", cleaned).strip()
        cleaned = cleaned.removesuffix("```").strip()
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if not lines:
            return "UNKNOWN"
        first_line = lines[0]
        if first_line.lower() == "assistant":
            first_line = lines[1] if len(lines) > 1 else "UNKNOWN"
        first_line = re.sub(r"^(assistant)\s*:?\s*", "", first_line, flags=re.IGNORECASE).strip()
        if "ABSTAIN" in first_line:
            return "ABSTAIN"
        first_line = re.sub(r"^(Answer:|Response:|Final answer:)\s*", "", first_line, flags=re.IGNORECASE)
        return first_line.strip(" \"'`") or "UNKNOWN"

    def _prepare_prompt(
        self,
        index: int,
        query: RuntimeQuery,
        context: Sequence[MemoryRecord],
    ) -> _PreparedPrompt:
        package = build_reasoning_prompt(
            query,
            context,
            template_id=self.config.prompt_template_id,
            system_prompt=self.config.system_prompt,
        )
        rendered_prompt = render_prompt(
            package,
            tokenizer=self._tokenizer,
            use_chat_template=self.config.use_chat_template,
        )
        return _PreparedPrompt(
            index=index,
            rendered_prompt=rendered_prompt,
            cache_key_value=cache_key(
                self.config.model_id,
                package.template_id,
                rendered_prompt,
                str(self.config.max_new_tokens),
                str(self.config.temperature),
                str(self.config.top_p),
                str(self.config.do_sample),
                str(self.config.repetition_penalty),
            ),
            template_id=package.template_id,
        )

    def _load_cached_trace(self, cache_key_value: str) -> ReasonerTrace | None:
        if self._cache is None:
            return None
        cached = self._cache.load(cache_key_value)
        if cached is None:
            return None
        cached.cache_hit = True
        return cached

    def _generate_missing_traces(self, prompts: Sequence[_PreparedPrompt]) -> list[ReasonerTrace]:
        if not prompts:
            return []
        batch_size = max(1, self.config.inference_batch_size)
        ordered = sorted(prompts, key=lambda prompt: len(prompt.rendered_prompt), reverse=True)
        generated: dict[int, ReasonerTrace] = {}
        for start in range(0, len(ordered), batch_size):
            batch = ordered[start:start + batch_size]
            for prompt, trace in zip(batch, self._generate_batch(batch)):
                generated[prompt.index] = trace
        return [generated[prompt.index] for prompt in prompts]

    def _generate_batch(self, prompts: Sequence[_PreparedPrompt]) -> list[ReasonerTrace]:
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None

        rendered_prompts = [prompt.rendered_prompt for prompt in prompts]
        encoded = self._tokenizer(
            rendered_prompts,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=8 if self._using_cuda(self._torch) else None,
            truncation=True,
            max_length=self.config.context_window,
        )
        prompt_lengths = self._prompt_lengths(encoded["attention_mask"])
        input_width = int(encoded["input_ids"].shape[-1])
        if hasattr(encoded, "to"):
            encoded = encoded.to(self._model.device)
        started = time.perf_counter()
        with self._torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                **self._generate_kwargs(),
            )
        per_query_latency_ms = ((time.perf_counter() - started) * 1000.0) / len(prompts)

        traces: list[ReasonerTrace] = []
        for prompt, prompt_tokens, generated_ids in zip(prompts, prompt_lengths, generated):
            completion_ids = generated_ids[input_width:]
            completion_text = self._decode_completion(completion_ids)
            answer_text = self._extract_answer(completion_text, prompt.rendered_prompt)
            completion_tokens = self._token_length(completion_ids)
            traces.append(
                ReasonerTrace(
                    answer=answer_text,
                    model_id=self.config.model_id,
                    prompt=prompt.rendered_prompt,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    latency_ms=per_query_latency_ms,
                    cache_hit=False,
                    raw_output=completion_text,
                    metadata={
                        "backend": self.config.backend,
                        "template_id": prompt.template_id,
                        "use_chat_template": str(self.config.use_chat_template),
                        "batch_size": str(len(prompts)),
                    },
                )
            )
        return traces

    def _prompt_lengths(self, attention_mask: Any) -> list[int]:
        summed = attention_mask.sum(dim=1)
        if hasattr(summed, "tolist"):
            return [int(value) for value in summed.tolist()]
        return [int(value) for value in summed]
