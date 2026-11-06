"""The coding-agent loop: approvals, modes, text protocol, interrupts, undo, slash commands,
context compaction, and the HTTP API (scripted model, real files and commands)."""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.agents.coding.prompts import PromptLibrary
from app.agents.coding.service import AgentError, AgentService
from app.agents.coding.store import AgentStore
from app.agents.coding.tools import Workspace
from app.api.deps import AppState
from app.main import create_app_with_state
from app.models.registry import ModelRegistry
from app.security.audit import AuditLog

from .scripted_adapter import ScriptedAdapter, call, text, text_call


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    return root


@pytest.fixture
def scripted() -> ScriptedAdapter:
    return ScriptedAdapter(native=True)


@pytest.fixture
def agent_env(repository, settings_store, scripted, tmp_path):
    registry = ModelRegistry(repository, settings_store, adapter=scripted)
    registry.discover(["/models"])
    model_id = registry.list_models()[0]["id"]
    registry.load(model_id)

    def factory(root: str, internet: bool) -> Workspace:
        ws = Workspace(root, internet_enabled=internet)
        ws.command_runner = lambda argv, cwd, timeout, on_output, cancel: {
            "exit_code": 0, "output": "3 passed in 0.01s", "elapsed": 0.1
        }
        return ws

    service = AgentService(
        AgentStore(repository.db), registry, settings_store, AuditLog(repository),
        PromptLibrary(tmp_path / "prompt-overrides"), workspace_factory=factory,
    )
    return service, model_id


def _wait_for(service: AgentService, sid: str, predicate, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        events = service.events(sid)
        hit = predicate(events)
        if hit:
            return hit
        time.sleep(0.02)
    raise AssertionError(f"timed out; events: {[e['type'] for e in service.events(sid)]}")


def _pending_tool(events):
    for e in events:
        if e["type"] == "tool" and e["status"] == "awaiting_approval":
            return e
    return None


def _ended(events):
    ends = [e for e in events if e["type"] == "turn_end"]
    return ends[-1] if ends else None


def test_agent_mode_edit_waits_for_approval_then_applies(agent_env, scripted, workspace) -> None:
    service, model_id = agent_env
    scripted.push(
        call("read_file", {"path": "calc.py"}, "c1", reasoning="Look first."),
        call("edit_file", {"path": "calc.py", "old_string": "return a - b", "new_string": "return a + b"}, "c2"),
        text("Fixed `add` in calc.py:2."),
    )
    session = service.create_session(str(workspace), model_id, "agent")
    service.send(session["id"], "fix add")
    pending = _wait_for(service, session["id"], _pending_tool)
    assert pending["name"] == "edit_file" and pending["risk"] == "edit"
    assert "-    return a - b" in pending["diff"] and "+    return a + b" in pending["diff"]
    # Not applied before approval, and the model saying so would not change that.
    assert "a - b" in (workspace / "calc.py").read_text(encoding="utf-8")
    assert service.describe(session["id"])["pending_approvals"] == ["c2"]
    assert service.decide(session["id"], "c2", approve=True)
    end = _wait_for(service, session["id"], _ended)
    assert end["reason"] == "completed"
    assert (workspace / "calc.py").read_text(encoding="utf-8") == "def add(a, b):\n    return a + b\n"
    types = [e["type"] for e in service.events(session["id"])]
    assert types[:2] == ["turn_start", "user"] and "usage" in types
    final = [e for e in service.events(session["id"]) if e["type"] == "assistant"][-1]
    assert final["content"] == "Fixed `add` in calc.py:2."
    # The model got the tool results back as tool messages with matching ids.
    last_request = scripted.requests[-1]["messages"]
    tool_msgs = [m for m in last_request if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2"]
    assert "Updated calc.py" in tool_msgs[1]["content"]
    assert scripted.requests[0]["options"]["tools"][0]["type"] == "function"


def test_denied_action_is_reported_to_the_model(agent_env, scripted, workspace) -> None:
    service, model_id = agent_env
    scripted.push(call("run_command", {"command": "python -m pytest -q"}, "c1"), text("OK, I won't run it."))
    session = service.create_session(str(workspace), model_id, "agent")
    service.send(session["id"], "run tests")
    _wait_for(service, session["id"], _pending_tool)
    service.decide(session["id"], "c1", approve=False)
    _wait_for(service, session["id"], _ended)
    updates = [e for e in service.events(session["id"]) if e["type"] == "tool_update"]
    assert updates[-1]["status"] == "denied"
    tool_msg = [m for m in scripted.requests[-1]["messages"] if m["role"] == "tool"][-1]
    assert "denied" in tool_msg["content"]


def test_auto_mode_runs_edits_and_commands_but_asks_for_destructive(agent_env, scripted, workspace) -> None:
    service, model_id = agent_env
    scripted.push(
        call("write_file", {"path": "tests/test_calc.py", "content": "from calc import add\n"}, "c1"),
        call("run_command", {"command": "python -m pytest -q"}, "c2"),
        call("run_command", {"command": "npm install"}, "c3"),
        call("run_command", {"command": "Remove-Item -Recurse -Force tests"}, "c4"),
        text("Done."),
    )
    session = service.create_session(str(workspace), model_id, "auto")
    service.send(session["id"], "add tests")
    pending = _wait_for(service, session["id"], _pending_tool)
    # Edits and ordinary commands ran without asking; only the destructive one waits.
    assert pending["call_id"] == "c4" and pending["risk"] == "destructive"
    assert (workspace / "tests" / "test_calc.py").exists()
    statuses = {e["call_id"]: e["status"] for e in service.events(session["id"]) if e["type"] == "tool"}
    assert statuses["c2"] == "running" and statuses["c3"] == "running"
    service.decide(session["id"], "c4", approve=False)
    _wait_for(service, session["id"], _ended)
    assert (workspace / "tests").exists()


def test_agent_mode_remembers_always_allow(agent_env, scripted, workspace) -> None:
    service, model_id = agent_env
    scripted.push(
        call("run_command", {"command": "python -m pytest -q"}, "c1"),
        call("run_command", {"command": "python -m pytest -q tests"}, "c2"),
        text("Done."),
    )
    session = service.create_session(str(workspace), model_id, "agent")
    service.send(session["id"], "test")
    pending = _wait_for(service, session["id"], _pending_tool)
    assert pending["prefix"] == "python -m pytest"
    service.decide(session["id"], "c1", approve=True, remember=True)
    _wait_for(service, session["id"], _ended)
    c2 = next(e for e in service.events(session["id"]) if e["type"] == "tool" and e["call_id"] == "c2")
    assert c2["status"] == "running"
    assert service.describe(session["id"])["options"]["allowed_commands"] == ["python -m pytest"]


def test_auto_continues_when_model_stops_with_work_left(agent_env, scripted, workspace) -> None:
    service, model_id = agent_env
    scripted.push(
        call("write_file", {"path": "a.py", "content": "x = 1\n"}, "c1"),
        text("The initial structure has been created. The next steps are to implement the features."),
        call("write_file", {"path": "b.py", "content": "y = 2\n"}, "c2"),
        text("Created a.py and b.py; everything is done."),
    )
    session = service.create_session(str(workspace), model_id, "auto")
    service.send(session["id"], "build it")
    end = _wait_for(service, session["id"], _ended)
    assert end["reason"] == "completed"
    assert (workspace / "b.py").exists()  # it kept going without the user typing "proceed"
    push = scripted.requests[2]["messages"][-1]
    assert push["role"] == "user" and "still work to do" in push["content"]
    notices = [e["text"] for e in service.events(session["id"]) if e["type"] == "notice"]
    assert any("continuing automatically" in n for n in notices)


def test_auto_does_not_ask_user_to_choose(agent_env, scripted, workspace) -> None:
    service, model_id = agent_env
    scripted.push(
        text("There is a conflict. Please choose an option and proceed."),
        call("write_file", {"path": "pkg.json", "content": "{}"}, "c1"),
        text("Chose option A and fixed it."),
    )
    session = service.create_session(str(workspace), model_id, "auto")
    service.send(session["id"], "fix deps")
    _wait_for(service, session["id"], _ended)
    push = scripted.requests[1]["messages"][-1]
    assert "do not ask the user to choose" in push["content"]
    assert (workspace / "pkg.json").exists()


def test_repeated_status_message_gets_pushed_and_continuations_are_bounded(agent_env, scripted, workspace) -> None:
    service, model_id = agent_env
    stuck_text = "The next steps are to implement the remaining features."
    scripted.push(call("list_dir", {"path": "."}, "c1"), *[text(stuck_text) for _ in range(20)])
    session = service.create_session(str(workspace), model_id, "auto")
    service.send(session["id"], "go")
    end = _wait_for(service, session["id"], _ended)
    assert end["reason"] == "completed"
    pushes = [r["content"] for r in scripted.requests[-1]["messages"] if r["role"] == "user"] + [
        m["messages"][-1]["content"] for m in scripted.requests[1:] if m["messages"][-1]["role"] == "user"
    ]
    assert any("repeated your previous message" in p for p in pushes)
    notices = [e["text"] for e in service.events(session["id"]) if e["type"] == "notice" and "continuing automatically" in e["text"]]
    assert len(notices) == 12  # CONTINUATIONS["auto"]: bounded, never an endless loop
    # Only the latest push stays in the model's context, and the repeated status replies are dropped.
    active_users = [m for m in service.store.messages(session["id"]) if m["role"] == "user"]
    assert len(active_users) == 2  # the task + the latest push


def test_agent_mode_continuation_is_smaller_and_ask_mode_never(agent_env, scripted, workspace) -> None:
    service, model_id = agent_env
    scripted.push(call("list_dir", {"path": "."}, "c1"), *[text(f"Next, I will do step {i}.") for i in range(10)])
    session = service.create_session(str(workspace), model_id, "agent")
    service.send(session["id"], "go")
    _wait_for(service, session["id"], _ended)
    notices = [e for e in service.events(session["id"]) if e["type"] == "notice" and "continuing automatically" in e["text"]]
    assert len(notices) == 3
    scripted.script.clear()
    scripted.push(text("The next steps would be X and Y."))
    ask = service.create_session(str(workspace), model_id, "ask")
    before = len(scripted.requests)
    service.send(ask["id"], "what next?")
    _wait_for(service, ask["id"], _ended)
    assert len(scripted.requests) == before + 1


def test_reasoning_only_reply_continues_automatically(agent_env, scripted, workspace) -> None:
    service, model_id = agent_env
    scripted.push(
        [{"type": "reasoning", "text": "I should create the file."}],
        call("write_file", {"path": "r.py", "content": "z = 3\n"}, "c1"),
        text("Created r.py."),
    )
    session = service.create_session(str(workspace), model_id, "auto")
    service.send(session["id"], "make r.py")
    end = _wait_for(service, session["id"], _ended)
    assert end["reason"] == "completed" and (workspace / "r.py").exists()


def test_auto_has_a_larger_step_budget(agent_env, scripted, workspace) -> None:
    from app.agents.coding.service import AUTO_MAX_STEPS, DEFAULT_MAX_STEPS

    assert AUTO_MAX_STEPS > DEFAULT_MAX_STEPS


def test_ask_mode_cannot_edit(agent_env, scripted, workspace) -> None:
    service, model_id = agent_env
    scripted.push(call("write_file", {"path": "x.py", "content": "x"}, "c1"), text("I can't edit in Ask mode."))
    session = service.create_session(str(workspace), model_id, "ask")
    service.send(session["id"], "write x")
    _wait_for(service, session["id"], _ended)
    assert not (workspace / "x.py").exists()
    offered = {t["function"]["name"] for t in scripted.requests[0]["options"]["tools"]}
    assert "write_file" not in offered and "read_file" in offered
    update = next(e for e in service.events(session["id"]) if e["type"] == "tool_update")
    assert update["status"] == "error" and "not available in ask mode" in update["output"]


def test_text_protocol_for_models_without_native_tools(agent_env, scripted, workspace) -> None:
    service, model_id = agent_env
    scripted.native = False
    scripted.push(
        text_call("read_file", {"path": "calc.py"}, before="<think>I should read it.</think>Reading the file.\n"),
        text("The bug is on calc.py:2 (it subtracts)."),
    )
    session = service.create_session(str(workspace), model_id, "ask")
    service.send(session["id"], "what is wrong?")
    _wait_for(service, session["id"], _ended)
    first = scripted.requests[0]
    assert "tools" not in first["options"]
    system = first["messages"][0]["content"]
    assert "<tool_call>" in system and "read_file" in system
    events = service.events(session["id"])
    assistant = [e for e in events if e["type"] == "assistant"]
    assert assistant[0]["content"] == "Reading the file." and assistant[0]["reasoning"] == "I should read it."
    tool_update = next(e for e in events if e["type"] == "tool_update")
    assert tool_update["status"] == "success" and "return a - b" in tool_update["output"]
    second = scripted.requests[1]["messages"]
    assert second[-1]["role"] == "user" and second[-1]["content"].startswith("Tool result for read_file")
    assert all("tool_calls" not in m for m in second)


def test_tool_errors_are_recoverable(agent_env, scripted, workspace) -> None:
    service, model_id = agent_env
    scripted.push(
        call("edit_file", {"path": "calc.py", "old_string": "nope", "new_string": "x"}, "c1"),
        call("read_file", {"path": "missing.py"}, "c2"),
        call("no_such_tool", {}, "c3"),
        text("Recovered."),
    )
    session = service.create_session(str(workspace), model_id, "auto")
    service.send(session["id"], "go")
    end = _wait_for(service, session["id"], _ended)
    assert end["reason"] == "completed"
    tool_msgs = [m["content"] for m in scripted.requests[-1]["messages"] if m["role"] == "tool"]
    assert "not found in calc.py" in tool_msgs[0]
    assert "File not found" in tool_msgs[1]
    assert "Unknown tool 'no_such_tool'" in tool_msgs[2]


def test_interrupt_while_waiting_for_approval(agent_env, scripted, workspace) -> None:
    service, model_id = agent_env
    scripted.push(call("run_command", {"command": "python app.py"}, "c1"))
    session = service.create_session(str(workspace), model_id, "agent")
    service.send(session["id"], "run it")
    _wait_for(service, session["id"], _pending_tool)
    assert service.interrupt(session["id"])
    end = _wait_for(service, session["id"], _ended)
    assert end["reason"] == "interrupted"
    assert service.describe(session["id"])["state"] == "idle"
    assert not service.decide(session["id"], "c1", True)  # nothing left to approve


def test_step_budget(agent_env, scripted, workspace) -> None:
    service, model_id = agent_env
    scripted.push(*[call("list_dir", {"path": "."}, f"c{i}") for i in range(5)])
    session = service.create_session(str(workspace), model_id, "agent")
    service.update_session(session["id"], options={"max_steps": 3})
    service.send(session["id"], "loop")
    end = _wait_for(service, session["id"], _ended)
    assert end["reason"] == "step_budget"
    assert len(scripted.requests) == 3


def test_repeated_identical_calls_are_stopped(agent_env, scripted, workspace) -> None:
    service, model_id = agent_env
    scripted.push(*[call("list_dir", {"path": "."}, f"c{i}") for i in range(6)], text("ok"))
    session = service.create_session(str(workspace), model_id, "agent")
    service.send(session["id"], "loop")
    end = _wait_for(service, session["id"], _ended)
    assert end["reason"] == "stuck"
    tool_msgs = [m["content"] for m in service.store.messages(session["id"]) if m["role"] == "tool"]
    assert tool_msgs[-1].startswith("Refused: you already made this exact call")
    assert len(scripted.requests) == 5  # 2 executed + 3 refused, then the turn stops
    # The temperature rises while the model loops.
    temps = [r["options"]["temperature"] for r in scripted.requests]
    assert temps[3] > temps[0]


def test_identical_edit_is_applied_once_and_result_shows_region(agent_env, scripted, workspace) -> None:
    service, model_id = agent_env
    edit = {"path": "calc.py", "old_string": "return a - b", "new_string": "return a - b  # fixme"}
    scripted.push(call("edit_file", edit, "c1"), call("edit_file", edit, "c2"), text("done"))
    session = service.create_session(str(workspace), model_id, "auto")
    service.send(session["id"], "edit")
    _wait_for(service, session["id"], _ended)
    assert (workspace / "calc.py").read_text(encoding="utf-8").count("# fixme") == 1
    tool_msgs = [m["content"] for m in service.store.messages(session["id"]) if m["role"] == "tool"]
    assert "Lines 1-2 now read:" in tool_msgs[0] and "return a - b  # fixme" in tool_msgs[0]
    assert "already changed" in tool_msgs[1]


def test_undo_and_per_file_reject(agent_env, scripted, workspace) -> None:
    service, model_id = agent_env
    scripted.push(
        call("edit_file", {"path": "calc.py", "old_string": "a - b", "new_string": "a + b"}, "c1"),
        call("write_file", {"path": "notes.md", "content": "hello\n"}, "c2"),
        text("done"),
    )
    session = service.create_session(str(workspace), model_id, "auto")
    service.send(session["id"], "edit")
    _wait_for(service, session["id"], _ended)
    changes = service.changes(session["id"])
    assert {c["path"] for c in changes} == {"calc.py", "notes.md"}
    assert next(c for c in changes if c["path"] == "notes.md")["created"] is True
    service.reject_file(session["id"], 1, "calc.py")
    assert "a - b" in (workspace / "calc.py").read_text(encoding="utf-8")
    assert (workspace / "notes.md").exists()
    assert service.undo_turn(session["id"]) == ["notes.md"]
    assert not (workspace / "notes.md").exists()
    statuses = {c["path"]: c["status"] for c in service.changes(session["id"])}
    assert statuses == {"calc.py": "rejected", "notes.md": "rejected"}


def test_reject_refuses_when_user_changed_the_file(agent_env, scripted, workspace) -> None:
    service, model_id = agent_env
    scripted.push(call("edit_file", {"path": "calc.py", "old_string": "a - b", "new_string": "a + b"}, "c1"), text("done"))
    session = service.create_session(str(workspace), model_id, "auto")
    service.send(session["id"], "edit")
    _wait_for(service, session["id"], _ended)
    (workspace / "calc.py").write_text("user edit\n", encoding="utf-8")
    with pytest.raises(AgentError, match="changed after the agent edited it"):
        service.reject_file(session["id"], 1, "calc.py")


def test_system_prompt_has_environment_and_project_instructions(agent_env, scripted, workspace) -> None:
    service, model_id = agent_env
    (workspace / "AGENTS.md").write_text("Always use tabs. Run tests with `python -m pytest`.", encoding="utf-8")
    session = service.create_session(str(workspace), model_id, "plan")
    prompt = service.preview_system_prompt(session["id"])
    assert str(workspace.resolve()) in prompt
    assert "PowerShell" in prompt or "bash" in prompt
    assert "Project instructions (from AGENTS.md)" in prompt and "Always use tabs" in prompt
    assert "Mode: Plan" in prompt
    assert "calc.py" in prompt  # file tree
    assert "{{" not in prompt


def test_prompt_overrides_and_reset(tmp_path) -> None:
    library = PromptLibrary(tmp_path / "overrides")
    library.set("modes.md", "## ask\nCUSTOM ASK\n")
    assert library.mode_instructions("ask") == "CUSTOM ASK"
    assert next(p for p in library.describe() if p["name"] == "modes.md")["overridden"] is True
    with pytest.raises(ValueError):
        library.set("variants.json", "{not json")
    library.reset("modes.md")
    assert "Mode: Ask" in library.mode_instructions("ask")
    assert library.variant_for("DeepSeek-R1-Distill-Qwen-1.5B-f16.gguf")["tool_format"] == "text"
    assert library.variant_for("Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf")["tool_format"] == "json"


def test_slash_commands(agent_env, scripted, workspace) -> None:
    service, model_id = agent_env
    session = service.create_session(str(workspace), model_id, "agent")
    sid = session["id"]
    service.send(sid, "/help")
    assert "/compact" in service.events(sid)[-1]["text"]
    service.send(sid, "/mode plan")
    assert service.describe(sid)["mode"] == "plan"
    with pytest.raises(AgentError, match="needs Agent or Auto"):
        service.send(sid, "/init")
    service.send(sid, "/mode agent")
    service.send(sid, "/model fake")
    assert service.describe(sid)["model_id"] == model_id
    with pytest.raises(AgentError, match="Unknown command"):
        service.send(sid, "/bogus")
    scripted.push(call("write_file", {"path": "AGENTS.md", "content": "# Project\n"}, "c1"), text("Wrote AGENTS.md"))
    service.send(sid, "/init")
    _wait_for(service, sid, _pending_tool)
    assert "AGENTS.md" in scripted.requests[0]["messages"][-1]["content"]
    service.decide(sid, "c1", True)
    _wait_for(service, sid, _ended)
    assert (workspace / "AGENTS.md").exists()
    service.send(sid, "/clear")
    assert service.store.messages(sid) == []
    service.send(sid, "/undo")
    assert not (workspace / "AGENTS.md").exists()


def test_compaction_when_context_is_small(agent_env, scripted, workspace) -> None:
    service, model_id = agent_env
    scripted.n_ctx = 2048
    big = "x" * 9000
    (workspace / "big.txt").write_text(big, encoding="utf-8")
    scripted.push(call("read_file", {"path": "big.txt"}, "c1"), text("read it"))
    session = service.create_session(str(workspace), model_id, "agent")
    service.send(session["id"], "read big.txt")
    _wait_for(service, session["id"], _ended)
    scripted.push(text("continuing"))
    service.send(session["id"], "next step")
    _wait_for(service, session["id"], lambda ev: len([e for e in ev if e["type"] == "turn_end"]) == 2 and ev)
    assert scripted.chat_calls, "compaction should have asked the model for a summary"
    messages = service.store.messages(session["id"])
    assert messages[0]["content"].startswith("Summary of the earlier conversation")
    assert "SUMMARY: user wants a CLI" in messages[0]["content"] and "<think>" not in messages[0]["content"]
    notices = [e["text"] for e in service.events(session["id"]) if e["type"] == "notice"]
    assert any(n.startswith("Context was getting full; summarised") for n in notices)


def test_unloaded_model_gives_clear_error(agent_env, scripted, workspace) -> None:
    service, model_id = agent_env
    session = service.create_session(str(workspace), model_id, "agent")
    scripted.unload_model(model_id)
    with pytest.raises(AgentError, match="not loaded"):
        service.send(session["id"], "hi")


def test_creating_a_session_approves_the_workspace(agent_env, settings_store, workspace) -> None:
    service, model_id = agent_env
    service.create_session(str(workspace), model_id)
    assert str(workspace.resolve()) in settings_store.load().approved_workspaces
    with pytest.raises(AgentError, match="Folder not found"):
        service.create_session(str(workspace / "nope"), model_id)


def test_audit_log_records_agent_tools(agent_env, scripted, workspace, repository) -> None:
    service, model_id = agent_env
    scripted.push(call("list_dir", {"path": "."}, "c1"), text("ok"))
    session = service.create_session(str(workspace), model_id, "agent")
    service.send(session["id"], "ls")
    _wait_for(service, session["id"], _ended)
    logs = repository.list_tool_logs()
    assert any(log["tool_name"] == "agent.list_dir" and log["task_id"] == session["id"] for log in logs)


# ------------------------------------------------------------------ HTTP API
def test_http_api_end_to_end(repository, settings_store, chat_service, tool_executor, agent_loop, retrieval, multi_agent, agent_env, scripted, workspace) -> None:
    service, model_id = agent_env
    state = AppState(repository, settings_store, service.registry, chat_service, tool_executor, agent_loop, retrieval, multi_agent, service)
    with TestClient(create_app_with_state(state)) as client:
        created = client.post("/api/agent/sessions", json={"workspace": str(workspace), "model_id": model_id, "mode": "agent"})
        assert created.status_code == 200, created.text
        sid = created.json()["id"]
        scripted.push(call("write_file", {"path": "hello.py", "content": "print('hi')\n"}, "c1"), text("Created hello.py"))
        assert client.post(f"/api/agent/sessions/{sid}/messages", json={"text": "make hello"}).json()["started"]
        _wait_for(service, sid, _pending_tool)
        assert client.post(f"/api/agent/sessions/{sid}/decision", json={"call_id": "c1", "approve": True}).json() == {"ok": True}
        _wait_for(service, sid, _ended)
        events = client.get(f"/api/agent/sessions/{sid}/events").json()
        assert events[-1]["type"] == "turn_end"
        changes = client.get(f"/api/agent/sessions/{sid}/changes").json()
        assert changes[0]["path"] == "hello.py" and changes[0]["created"]
        tree = client.get("/api/workspace/tree", params={"path": str(workspace)}).json()
        assert {e["name"] for e in tree["entries"]} >= {"calc.py", "hello.py"}
        assert client.get("/api/workspace/file", params={"path": str(workspace / "hello.py")}).json()["content"] == "print('hi')\n"
        assert client.get("/api/workspace/tree", params={"path": str(workspace.parent)}).status_code == 403
        assert client.post(f"/api/agent/sessions/{sid}/messages", json={"text": "/bogus"}).status_code == 400
        assert client.post(f"/api/agent/sessions/{sid}/decision", json={"call_id": "zzz", "approve": True}).status_code == 409
        assert len(client.get("/api/agent/prompts").json()) == 9
        assert "Mode: Agent" in client.get(f"/api/agent/sessions/{sid}/system-prompt").json()["prompt"]
        assert client.get("/api/workspace/recent").json()[0]["path"] == str(workspace.resolve())


def test_described_but_not_done_escalates_to_constrained_tool_calls(agent_env, scripted, workspace) -> None:
    service, model_id = agent_env
    scripted.push(
        text("### Plan\n1. Create wordstats.py\n```python\nprint(1)\n```"),
        text('{"name": "write_file", "arguments": {"path": "wordstats.py", "content": "print(1)\\n"}}'),
        text('{"name": "respond", "arguments": {"message": "Created `wordstats.py`."}}'),
    )
    session = service.create_session(str(workspace), model_id, "auto")
    service.send(session["id"], "build it")
    end = _wait_for(service, session["id"], _ended)
    assert end["reason"] == "completed"
    assert (workspace / "wordstats.py").read_text(encoding="utf-8") == "print(1)\n"
    # Steps 2 and 3 were grammar-constrained, with the tools + "respond" in the schema.
    fmt = scripted.requests[1]["options"]["response_format"]
    names = [o["properties"]["name"]["const"] for o in fmt["json_schema"]["schema"]["oneOf"]]
    assert "write_file" in names and names[-1] == "respond"
    assert "tools" not in scripted.requests[1]["options"]
    assert "response_format" in scripted.requests[2]["options"]
    # The dodged prose is not kept in the model's context.
    assert all("### Plan" not in str(m.get("content")) for m in scripted.requests[1]["messages"])
    events = service.events(session["id"])
    assert any("constrained tool calls" in e.get("text", "") for e in events if e["type"] == "notice")
    assert [e for e in events if e["type"] == "assistant"][-1]["content"] == "Created `wordstats.py`."


def test_constrained_schema_lifts_defs(agent_env) -> None:
    from app.agents.coding.tools import tools_for_mode

    schema = AgentService.constrained_schema(tools_for_mode("agent", False))
    assert "TodoItem" in schema["$defs"]
    assert all("$defs" not in o["properties"]["arguments"] for o in schema["oneOf"])


def test_no_nudge_for_plain_answers_or_ask_mode(agent_env, scripted, workspace) -> None:
    service, model_id = agent_env
    scripted.push(text("1. It adds numbers\n2. It is buggy"))
    session = service.create_session(str(workspace), model_id, "ask")
    service.send(session["id"], "what does calc do?")
    _wait_for(service, session["id"], _ended)
    assert len(scripted.requests) == 1


def test_completion_check_when_tests_still_fail(agent_env, scripted, workspace) -> None:
    service, model_id = agent_env
    results = iter([
        {"exit_code": 1, "output": "1 failed", "elapsed": 0.1},
        {"exit_code": 0, "output": "1 passed", "elapsed": 0.1},
    ])
    service._workspace_factory = lambda root, internet: Workspace(
        root, command_runner=lambda argv, cwd, timeout, on_output, cancel: next(results)
    )
    scripted.push(
        call("run_command", {"command": "python -m pytest -q"}, "c1"),
        text("All done!"),
        call("run_command", {"command": "python -m pytest -q tests"}, "c2"),
        text("Fixed; tests pass."),
    )
    session = service.create_session(str(workspace), model_id, "auto")
    service.update_session(session["id"], options={"allowed_commands": ["python -m pytest"]})
    service.send(session["id"], "make tests pass")
    end = _wait_for(service, session["id"], _ended)
    assert end["reason"] == "completed"
    reminder = scripted.requests[2]["messages"][-1]
    assert reminder["role"] == "user" and "failed with exit code 1" in reminder["content"]
    assert len(scripted.requests) == 4  # checked once, then finished normally


def test_no_completion_check_after_denied_command(agent_env, scripted, workspace) -> None:
    service, model_id = agent_env
    scripted.push(call("run_command", {"command": "python app.py"}, "c1"), text("OK, not running it."))
    session = service.create_session(str(workspace), model_id, "agent")
    service.send(session["id"], "run")
    _wait_for(service, session["id"], _pending_tool)
    service.decide(session["id"], "c1", False)
    _wait_for(service, session["id"], _ended)
    assert len(scripted.requests) == 2


def test_open_workspace_accepts_quoted_paths_and_rejects_relative(client, tmp_path) -> None:
    folder = tmp_path / "my project"
    folder.mkdir()
    # Explorer's "Copy as path" wraps the path in quotes; pasted text may have spaces.
    response = client.post("/api/workspace/open", json={"workspace": f'  "{folder}"  '})
    assert response.status_code == 200, response.text
    assert response.json()["path"] == str(folder.resolve())
    assert str(folder.resolve()) in client.get("/api/settings").json()["approved_workspaces"]
    relative = client.post("/api/workspace/open", json={"workspace": "my project"})
    assert relative.status_code == 400 and "full path" in relative.json()["detail"]
    missing = client.post("/api/workspace/open", json={"workspace": str(folder / "nope")})
    assert missing.status_code == 400 and "Folder not found" in missing.json()["detail"]
    file = folder / "a.txt"
    file.write_text("x", encoding="utf-8")
    assert "a file, not a folder" in client.post("/api/workspace/open", json={"workspace": str(file)}).json()["detail"]


def test_browse_lists_only_folders(client, tmp_path) -> None:
    (tmp_path / "alpha").mkdir()
    (tmp_path / "beta").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "file.txt").write_text("x", encoding="utf-8")
    listing = client.get("/api/workspace/browse", params={"path": f'"{tmp_path}"'}).json()
    names = [d["name"] for d in listing["dirs"]]
    assert "alpha" in names and "beta" in names
    assert ".hidden" not in names and "file.txt" not in names
    assert listing["parent"] == str(tmp_path.resolve().parent)
    roots = client.get("/api/workspace/browse").json()["dirs"]
    assert roots[0]["name"].startswith("Home")


def test_browse_can_include_model_files(client, tmp_path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "model-q4.gguf").write_bytes(b"GGUF" + b"0" * 10)
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    plain = client.get("/api/workspace/browse", params={"path": str(tmp_path)}).json()
    assert plain["files"] == []
    listing = client.get("/api/workspace/browse", params={"path": str(tmp_path), "ext": "gguf"}).json()
    assert [f["name"] for f in listing["files"]] == ["model-q4.gguf"]
    assert listing["files"][0]["size"] == 14
    assert [d["name"] for d in listing["dirs"]] == ["sub"]


def test_model_register_and_scan_accept_quoted_paths(client, tmp_path) -> None:
    missing = client.post("/api/models/register", json={"path": f'"{tmp_path / "nope.gguf"}"'})
    assert missing.status_code == 400 and "File not found" in missing.json()["detail"]
    from app.models.registry import clean_path

    assert clean_path(r'  "C:\models\a.gguf" ') == r"C:\models\a.gguf"


def test_auto_redirects_a_stuck_model_before_giving_up(agent_env, scripted, workspace) -> None:
    service, model_id = agent_env
    scripted.push(*[call("list_dir", {"path": "."}, f"c{i}") for i in range(5)],
                  call("write_file", {"path": "done.py", "content": "ok = True\n"}, "w1"), text("Finished."))
    session = service.create_session(str(workspace), model_id, "auto")
    service.send(session["id"], "go")
    end = _wait_for(service, session["id"], _ended)
    assert end["reason"] == "completed" and (workspace / "done.py").exists()
    notices = [e["text"] for e in service.events(session["id"]) if e["type"] == "notice"]
    assert any("redirected it" in n for n in notices)



def test_after_a_push_the_model_must_act_before_answering(agent_env, scripted, workspace) -> None:
    service, model_id = agent_env
    scripted.native = False
    service.prompts.set("variants.json", '{"version": 1, "variants": [{"match": "", "tool_format": "json"}]}')
    scripted.push(
        text('{"thought": "set up", "name": "write_file", "arguments": {"path": "app.py", "content": "x = 1\\n"}}'),
        text('{"thought": "status", "name": "respond", "arguments": {"message": "Structure created. The next step is to implement the services."}}'),
        text('{"thought": "act", "name": "write_file", "arguments": {"path": "services.py", "content": "y = 2\\n"}}'),
        text('{"thought": "done", "name": "respond", "arguments": {"message": "Implemented app.py and services.py."}}'),
    )
    session = service.create_session(str(workspace), model_id, "auto")
    service.send(session["id"], "build the app")
    end = _wait_for(service, session["id"], _ended)
    assert end["reason"] == "completed" and (workspace / "services.py").exists()

    def names(request):
        return [o["properties"]["name"]["const"] for o in request["options"]["response_format"]["json_schema"]["schema"]["oneOf"]]

    assert "respond" in names(scripted.requests[1])
    assert "respond" not in names(scripted.requests[2])  # forced to act after the push
    assert "respond" in names(scripted.requests[3])  # allowed again once it edited a file
    # The status reply is not in the context the model sees next; the push asks for a full todo list.
    third = scripted.requests[2]["messages"]
    assert all("Structure created" not in str(m.get("content")) for m in third)
    assert "covers the user's whole request" in third[-1]["content"]


def test_history_drops_older_duplicate_status_replies(agent_env, workspace) -> None:
    service, model_id = agent_env
    session = service.create_session(str(workspace), model_id, "auto")
    sid = session["id"]
    status = "The project structure is set up. The next step is to implement the services."
    for text_ in ("build it", None, "proceed", None, "implement everything", None):
        message = {"role": "user", "content": text_} if text_ else {"role": "assistant", "content": status}
        service.store.add_message(sid, 1, message)
    history = service._history(sid, native=True)
    assert [m["content"] for m in history].count(status) == 1
    assert history[-1]["content"] == status and history[-2]["content"] == "implement everything"


def test_compaction_inside_one_long_turn_keeps_task_and_runs_rarely(agent_env, scripted, workspace) -> None:
    service, model_id = agent_env
    scripted.n_ctx = 8192
    big = "y" * 2500
    for i in range(8):
        (workspace / f"f{i}.txt").write_text(big + str(i), encoding="utf-8")
    scripted.push(*[call("read_file", {"path": f"f{i}.txt"}, f"c{i}") for i in range(8)], text("Read all files."))
    session = service.create_session(str(workspace), model_id, "auto")
    service.send(session["id"], "THE TASK: read every file")
    end = _wait_for(service, session["id"], _ended)
    assert end["reason"] == "completed"
    notices = [e["text"] for e in service.events(session["id"]) if e["type"] == "notice" and "summarised" in e["text"]]
    assert 1 <= len(notices) <= 3  # not on every step
    # Every request after compaction still contains the original task verbatim.
    for request in scripted.requests:
        assert any("THE TASK: read every file" in str(m.get("content")) for m in request["messages"])
