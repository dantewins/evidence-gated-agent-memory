from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class LocalModelConfig:
    model_id: str
    backend: str = "hf"
    device: str = "auto"
    dtype: str = "auto"
    inference_batch_size: int = 8
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    repetition_penalty: float = 1.0
    context_window: int = 4096
    cache_dir: Optional[Path] = None
    prompt_template_id: str = "validity-v1"
    trust_remote_code: bool = False
    use_chat_template: bool = True
    system_prompt: str = (
        "You are evaluating a frozen-weight memory system. "
        "Answer strictly from the supplied external memory."
    )
