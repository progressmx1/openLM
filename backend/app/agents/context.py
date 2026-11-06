"""Conversation context management: token estimation and history trimming."""
from __future__ import annotations

from typing import Any

# Heuristic: roughly 4 characters per token for English text. This is an estimate
# only (the UI labels it as such); exact tokenization requires the model's tokenizer.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        content = message.get("content") or ""
        total += estimate_tokens(str(content)) + 4  # per-message overhead
    return total


def trim_history(
    messages: list[dict[str, Any]],
    max_tokens: int,
    keep_system: int = 1,
) -> list[dict[str, Any]]:
    """Keep leading system messages and the most recent messages that fit.

    Older non-system messages are dropped first to stay within ``max_tokens``.
    """
    system = [m for m in messages if m.get("role") == "system"][:keep_system]
    rest = [m for m in messages if m.get("role") != "system"]

    budget = max_tokens - estimate_messages_tokens(system)
    kept: list[dict[str, Any]] = []
    # Walk backwards, keeping the newest messages that fit.
    for message in reversed(rest):
        cost = estimate_tokens(str(message.get("content") or "")) + 4
        if estimate_messages_tokens(kept) + cost > budget:
            break
        kept.append(message)
    kept.reverse()

    return system + kept


def build_prompt_with_budget(
    messages: list[dict[str, Any]], context_length: int, reserve_ratio: float = 0.7
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Trim ``messages`` to a safe fraction of ``context_length``.

    Returns (trimmed_messages, stats) where stats reports estimated token usage.
    """
    budget = int(context_length * reserve_ratio)
    trimmed = trim_history(messages, budget)
    stats = {
        "estimated_tokens": estimate_messages_tokens(trimmed),
        "budget_tokens": budget,
        "context_length": context_length,
        "truncated": len(trimmed) < len(messages),
    }
    return trimmed, stats
