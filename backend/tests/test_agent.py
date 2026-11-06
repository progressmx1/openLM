"""Tests for the agent orchestration loop (tool-call extraction + execution)."""
from __future__ import annotations

import json

from app.agents.orchestrator import AgentLoop, extract_tool_calls
from app.agents.tool_executor import ToolExecutor
from app.models.registry import ModelRegistry
from app.security.audit import AuditLog
from app.tools.build import build_default_registry

from .fake_adapter import FakeAdapter


class ScriptedAdapter(FakeAdapter):
    def __init__(self, responses: list[str]) -> None:
        super().__init__()
        self.responses = list(responses)
        self.sent_messages: list[list[dict]] = []

    def chat(self, model_id, messages, options):
        self.sent_messages.append(messages)
        content = self.responses.pop(0) if self.responses else "done"
        return {"content": content, "usage": {}}


def _tool_call(name: str, args: dict) -> str:
    return "```tool\n" + json.dumps({"name": name, "args": args}) + "\n```"


def _setup(repository, settings_store, adapter):
    model_registry = ModelRegistry(repository, settings_store, adapter=adapter)
    tool_registry = build_default_registry()
    audit = AuditLog(repository)
    tool_executor = ToolExecutor(tool_registry, repository, audit, settings_store)
    agent_loop = AgentLoop(model_registry, tool_registry, tool_executor, repository, settings_store)
    model_registry.discover(["x"])
    model_id = model_registry.list_models()[0]["id"]
    model_registry.load(model_id)
    conversation_id = repository.create_conversation()["id"]
    return agent_loop, model_registry, model_id, conversation_id


def _set_workspace(settings_store, path) -> None:
    settings = settings_store.load()
    settings.approved_workspaces = [str(path)]
    settings_store.save(settings)


def test_extract_tool_calls() -> None:
    text = 'Let me check.\n```tool\n{"name": "read_file", "args": {"path": "a.txt"}}\n```'
    calls = extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "read_file"
    assert calls[0]["args"] == {"path": "a.txt"}


def test_extract_tool_calls_ignores_malformed() -> None:
    text = '```tool\nnot json\n```\n```tool\n{"name":"x","args":{}}\n```'
    calls = extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "x"


def test_agent_executes_tool_then_answers(repository, settings_store, tmp_path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.txt").write_text("hi there", encoding="utf-8")
    _set_workspace(settings_store, ws)

    adapter = ScriptedAdapter([
        _tool_call("read_file", {"path": str(ws / "a.txt")}),
        "The file says: hi there",
    ])
    agent_loop, _registry, model_id, conv_id = _setup(repository, settings_store, adapter)

    result = agent_loop.run(conv_id, model_id, "what is in a.txt?")
    assert result["status"] == "done"
    assert result["answer"] == "The file says: hi there"
    assert result["trace"][0]["type"] == "tool"
    assert result["trace"][0]["result"]["status"] == "success"
    # assistant message persisted
    messages = repository.list_messages(conv_id)
    assert messages[-1]["role"] == "assistant"


def test_agent_stops_at_approval(repository, settings_store, tmp_path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    _set_workspace(settings_store, ws)

    adapter = ScriptedAdapter([
        _tool_call("write_file", {"path": str(other / "x.txt"), "content": "hi"}),
        "done",
    ])
    agent_loop, _registry, model_id, conv_id = _setup(repository, settings_store, adapter)

    result = agent_loop.run(conv_id, model_id, "write outside")
    assert result["status"] == "needs_approval"
    assert not (other / "x.txt").exists()  # not executed


def test_agent_handles_unknown_tool(repository, settings_store) -> None:
    adapter = ScriptedAdapter([
        _tool_call("no_such_tool", {}),
        "I cannot do that.",
    ])
    agent_loop, _registry, model_id, conv_id = _setup(repository, settings_store, adapter)

    result = agent_loop.run(conv_id, model_id, "do a thing")
    assert result["status"] == "done"
    assert result["trace"][0]["result"]["status"] == "error"


def test_agent_max_steps(repository, settings_store) -> None:
    adapter = ScriptedAdapter([_tool_call("web_search", {"query": "x"})] * 10)
    agent_loop, _registry, model_id, conv_id = _setup(repository, settings_store, adapter)

    result = agent_loop.run(conv_id, model_id, "loop", max_steps=2)
    assert result["status"] == "max_steps"


def test_agent_unloaded_model(repository, settings_store) -> None:
    adapter = ScriptedAdapter(["hello"])
    agent_loop, model_registry, model_id, conv_id = _setup(repository, settings_store, adapter)
    model_registry.unload(model_id)

    result = agent_loop.run(conv_id, model_id, "hi")
    assert result["status"] == "error"
