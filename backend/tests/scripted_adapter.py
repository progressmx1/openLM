"""A model adapter that replays scripted responses, for deterministic agent-loop tests."""
from __future__ import annotations

import json
import threading
from typing import Any, Callable, Iterator, Union

from .fake_adapter import FakeAdapter

Script = Union[list[dict[str, Any]], Callable[[list[dict[str, Any]], dict[str, Any]], list[dict[str, Any]]]]


def text(content: str, reasoning: str = "") -> list[dict[str, Any]]:
    events = []
    if reasoning:
        events.append({"type": "reasoning", "text": reasoning})
    # Split into small chunks to exercise streaming.
    for i in range(0, len(content), 7):
        events.append({"type": "content", "text": content[i:i + 7]})
    return events


def call(name: str, args: dict[str, Any], call_id: str = "c1", reasoning: str = "") -> list[dict[str, Any]]:
    raw = json.dumps(args)
    events = [{"type": "reasoning", "text": reasoning}] if reasoning else []
    events.append({"type": "tool_call", "index": 0, "id": call_id, "name": name, "arguments": raw[: len(raw) // 2]})
    events.append({"type": "tool_call", "index": 0, "arguments": raw[len(raw) // 2:]})
    return events


def text_call(name: str, args: dict[str, Any], before: str = "") -> list[dict[str, Any]]:
    """A text-protocol tool call written into the content stream."""
    body = before + "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"
    return text(body)


class ScriptedAdapter(FakeAdapter):
    runtime_name = "scripted"

    def __init__(self, native: bool = True, n_ctx: int = 32768) -> None:
        super().__init__()
        self.native = native
        self.n_ctx = n_ctx
        self.script: list[Script] = []
        self.requests: list[dict[str, Any]] = []
        self.chat_calls: list[list[dict[str, Any]]] = []
        self.gate = threading.Event()
        self.gate.set()

    def push(self, *responses: Script) -> None:
        self.script.extend(responses)

    def load_model(self, model_id: str, settings: dict[str, Any]) -> dict[str, Any]:
        self.loaded[model_id] = {"n_ctx_effective": self.n_ctx}
        return {"loaded": True}

    def get_model_status(self, model_id: str) -> dict[str, Any]:
        if model_id not in self.loaded:
            return {"loaded": False}
        return {"loaded": True, "state": "ready", "n_ctx_effective": self.n_ctx, "last_timings": {"predicted_per_second": 42.0}}

    def supports_native_tools(self, model_id: str) -> bool:
        return self.native

    def stream_events(self, model_id: str, messages, options) -> Iterator[dict[str, Any]]:
        self.requests.append({"messages": [dict(m) for m in messages], "options": dict(options)})
        self.gate.wait(10)
        if not self.script:
            events = text("(script exhausted)")
        else:
            item = self.script.pop(0)
            events = item(messages, options) if callable(item) else item
        finish = "stop"
        for event in events:
            if event["type"] == "tool_call":
                finish = "tool_calls"
            yield event
        yield {"type": "done", "finish_reason": finish, "usage": {"prompt_tokens": 100, "completion_tokens": 20}, "timings": None}

    def chat(self, model_id: str, messages, options) -> dict[str, Any]:
        self.chat_calls.append(messages)
        return {"content": "<think>summarising</think>SUMMARY: user wants a CLI; files a.py created.", "usage": {}}
