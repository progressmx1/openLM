"""Tests for conversation context management."""
from __future__ import annotations

from app.agents.context import build_prompt_with_budget, estimate_tokens, trim_history


def test_estimate_tokens() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 40) == 10


def test_trim_history_keeps_system_and_recent() -> None:
    messages = [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "old message " * 200},
        {"role": "assistant", "content": "old reply " * 200},
        {"role": "user", "content": "the recent question"},
    ]
    trimmed = trim_history(messages, max_tokens=60)
    assert trimmed[0]["role"] == "system"
    assert trimmed[-1]["content"] == "the recent question"
    assert len(trimmed) < len(messages)


def test_build_prompt_with_budget_truncates() -> None:
    messages = [{"role": "user", "content": "x" * 1000}]
    trimmed, stats = build_prompt_with_budget(messages, context_length=256)
    assert stats["truncated"] is True
    assert stats["estimated_tokens"] <= stats["budget_tokens"]


def test_build_prompt_with_budget_no_truncation() -> None:
    messages = [{"role": "user", "content": "hi"}]
    trimmed, stats = build_prompt_with_budget(messages, context_length=8192)
    assert stats["truncated"] is False
    assert len(trimmed) == 1
