"""Small text helpers shared by the LLM and patch modules."""

from __future__ import annotations

import re

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CLOSE_TAG = "</think>"
_OPEN_TAG = "<think>"


def strip_reasoning(text: str | None) -> str:
    """Remove chain-of-thought markup that reasoning models may put in ``content``.

    Handles three shapes: a complete ``<think>...</think>`` block, a response
    where only the closing tag is present (the opening tag was consumed by the
    chat template), and a truncated response with an opening tag but no close.
    """
    if not text:
        return ""
    text = _THINK_BLOCK.sub("", text)

    lower = text.lower()
    if _CLOSE_TAG in lower:
        text = text[lower.rindex(_CLOSE_TAG) + len(_CLOSE_TAG):]

    lower = text.lower()
    if _OPEN_TAG in lower:
        text = text[: lower.index(_OPEN_TAG)]

    return text.strip()
