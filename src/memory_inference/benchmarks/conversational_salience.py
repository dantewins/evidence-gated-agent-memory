from __future__ import annotations

import re

_FIRST_PERSON_RE = re.compile(r"\b(i|i'm|i've|i'd|me|my|mine|we|our|ours)\b", re.IGNORECASE)
_FACTUAL_RE = re.compile(
    r"\b(am|was|were|work|worked|live|lived|moved|graduate|graduated|study|studied|bought|"
    r"have|had|plan|planned|going|went|met|read|signed|joined|collect|collects|prefer|like|love)\b",
    re.IGNORECASE,
)
_TEMPORAL_RE = re.compile(
    r"\b("
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"today|tomorrow|yesterday|week|month|year|before|after|ago|next|last"
    r")\b|\b\d{4}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    re.IGNORECASE,
)
_HELP_SEEKING_RE = re.compile(
    r"\b(can you|could you|would you|please|help me|recommend|suggest|good luck|if you're stuck|"
    r"strategic thinking|happy to help|congratulations|kudos)\b",
    re.IGNORECASE,
)
_QUESTION_ONLY_RE = re.compile(r"^\s*(can|could|would|should|what|when|where|who|why|how)\b", re.IGNORECASE)
_QUOTE_OR_TITLE_RE = re.compile(r"\"[^\"]+\"|'[^']+'|\b[A-Z][a-z]+(?:\s+[A-Z][a-z0-9+.-]+)+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def estimate_importance(text: str, *, speaker: str = "", attribute: str = "") -> float:
    content = text.strip()
    if not content:
        return 0.1

    importance = 0.35
    token_count = len(_TOKEN_RE.findall(content))

    if _FIRST_PERSON_RE.search(content):
        importance += 0.18
    if _FACTUAL_RE.search(content):
        importance += 0.2
    if _TEMPORAL_RE.search(content):
        importance += 0.14
    if _QUOTE_OR_TITLE_RE.search(content):
        importance += 0.08
    if speaker.lower() not in {"assistant", "system"}:
        importance += 0.04
    if attribute == "event":
        importance += 0.18
    if token_count <= 28:
        importance += 0.08
    elif token_count >= 90:
        importance -= 0.08

    question_marks = content.count("?")
    if question_marks:
        importance -= min(0.18, 0.06 * question_marks)
    if _HELP_SEEKING_RE.search(content) or _QUESTION_ONLY_RE.search(content):
        importance -= 0.22

    return _clamp(importance, low=0.1, high=1.8)


def estimate_confidence(text: str, *, speaker: str = "", attribute: str = "") -> float:
    content = text.strip()
    if not content:
        return 0.1

    confidence = 0.78
    if _FACTUAL_RE.search(content):
        confidence += 0.1
    if _TEMPORAL_RE.search(content):
        confidence += 0.06
    if _HELP_SEEKING_RE.search(content) or _QUESTION_ONLY_RE.search(content):
        confidence -= 0.18
    if speaker.lower() == "assistant":
        confidence -= 0.04
    if attribute == "event":
        confidence += 0.08
    return _clamp(confidence, low=0.2, high=1.0)


def _clamp(value: float, *, low: float, high: float) -> float:
    return max(low, min(high, value))
