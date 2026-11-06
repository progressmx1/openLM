"""Parse an OpenAI-compatible ``/chat/completions`` SSE stream into adapter events.

Shared by llama-server (local) and OpenAI / DeepSeek / other OpenAI-compatible cloud APIs.
Events: ``{"type": "reasoning"|"content", "text"}`` and ``{"type": "tool_call", "index", "id",
"name", "arguments"}`` (argument fragments). Finish reason, usage and llama.cpp timings are
collected on the :class:`StreamState`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Optional


@dataclass
class StreamState:
    finish_reason: Optional[str] = None
    usage: dict[str, Any] = field(default_factory=dict)
    timings: Optional[dict[str, Any]] = None


def iter_openai_events(lines: Iterable[str], state: StreamState) -> Iterator[dict[str, Any]]:
    for line in lines:
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        if chunk.get("error"):
            err = chunk["error"]
            raise RuntimeError(err.get("message") if isinstance(err, dict) else str(err))
        if chunk.get("timings"):
            state.timings = chunk["timings"]
        if chunk.get("usage"):
            state.usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("reasoning_content"):
                yield {"type": "reasoning", "text": delta["reasoning_content"]}
            if delta.get("content"):
                yield {"type": "content", "text": delta["content"]}
            for call in delta.get("tool_calls") or []:
                fn = call.get("function") or {}
                yield {
                    "type": "tool_call",
                    "index": call.get("index", 0),
                    "id": call.get("id"),
                    "name": fn.get("name"),
                    "arguments": fn.get("arguments") or "",
                }
            if choice.get("finish_reason"):
                state.finish_reason = choice["finish_reason"]
