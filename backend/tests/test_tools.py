"""Tests for the tool executor, filesystem tools, and terminal classification."""
from __future__ import annotations

import pytest

from app.agents.tool_executor import ToolExecutor
from app.security.audit import AuditLog
from app.tools.build import build_default_registry
from app.tools.registry import InvalidToolArgumentsError, UnknownToolError
from app.tools.terminal.terminal import classify_command


def _set_workspace(settings_store, path) -> None:
    settings = settings_store.load()
    settings.approved_workspaces = [str(path)]
    settings_store.save(settings)


# ------------------------------------------------------------- terminal rules


def test_classify_normal_command() -> None:
    assert classify_command("pytest -q").value == "system_action"


@pytest.mark.parametrize(
    "command",
    ["rm -rf /tmp/x", "rd /s /q C:\\data", "format c:", "reg delete HKLM\\x", "shutdown /s"],
)
def test_classify_destructive_commands(command) -> None:
    assert classify_command(command).value == "high_impact"


# ------------------------------------------------------------- registry


def test_list_tools(tool_executor) -> None:
    names = {t["name"] for t in tool_executor.list_tools()}
    assert {"list_directory", "read_file", "write_file", "create_folder",
            "search_files", "copy_file", "move_file", "delete",
            "run_command", "open_file", "reveal_in_folder"} <= names


def test_unknown_tool(tool_executor) -> None:
    with pytest.raises(UnknownToolError):
        tool_executor.submit("does_not_exist", {})


def test_malformed_args(tool_executor) -> None:
    with pytest.raises(InvalidToolArgumentsError):
        tool_executor.submit("read_file", {"path": 123})
    with pytest.raises(InvalidToolArgumentsError):
        tool_executor.submit("read_file", {})


# ------------------------------------------------------------- filesystem


def test_filesystem_within_workspace_auto(tool_executor, settings_store, tmp_path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    _set_workspace(settings_store, ws)

    written = tool_executor.submit("write_file", {"path": str(ws / "a.txt"), "content": "hello"})
    assert written["status"] == "success"
    assert written["auto"] is True

    read = tool_executor.submit("read_file", {"path": str(ws / "a.txt")})
    assert read["status"] == "success"
    assert read["result"]["output"] == "hello"

    listed = tool_executor.submit("list_directory", {"path": str(ws)})
    assert listed["status"] == "success"
    assert any(e["name"] == "a.txt" for e in listed["result"]["data"]["entries"])


def test_write_outside_workspace_requires_approval(tool_executor, settings_store, tmp_path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    _set_workspace(settings_store, ws)

    result = tool_executor.submit("write_file", {"path": str(other / "x.txt"), "content": "hi"})
    assert result["status"] == "awaiting_approval"
    assert result["level"] == "workspace_write"
    assert not (other / "x.txt").exists()


def test_write_existing_without_overwrite_errors(tool_executor, settings_store, tmp_path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    _set_workspace(settings_store, ws)
    tool_executor.submit("write_file", {"path": str(ws / "a.txt"), "content": "one"})
    result = tool_executor.submit("write_file", {"path": str(ws / "a.txt"), "content": "two"})
    assert result["status"] == "error"
    assert "already exists" in result["result"]["error"]


def test_delete_requires_approval(tool_executor, settings_store, tmp_path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "f.txt").write_text("x")
    _set_workspace(settings_store, ws)

    result = tool_executor.submit("delete", {"path": str(ws / "f.txt")})
    assert result["status"] == "awaiting_approval"
    assert result["level"] == "high_impact"
    assert (ws / "f.txt").exists()  # not deleted before approval


def test_run_command_requires_approval(tool_executor) -> None:
    result = tool_executor.submit("run_command", {"command": "echo hi"})
    assert result["status"] == "awaiting_approval"
    assert result["level"] == "system_action"


def test_destructive_command_escalates(tool_executor) -> None:
    result = tool_executor.submit("run_command", {"command": "rm -rf /tmp/x"})
    assert result["status"] == "awaiting_approval"
    assert result["level"] == "high_impact"


# ------------------------------------------------------------- approval flow


def test_approval_approve_runs_tool(tool_executor, settings_store, tmp_path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    _set_workspace(settings_store, ws)

    pending = tool_executor.submit("write_file", {"path": str(other / "x.txt"), "content": "hi"})
    approval_id = pending["approval_id"]
    assert any(a["id"] == approval_id for a in tool_executor.pending_approvals())

    decided = tool_executor.decide(approval_id, True)
    assert decided["status"] == "success"
    assert (other / "x.txt").read_text() == "hi"
    assert tool_executor.pending_approvals() == []


def test_approval_deny_does_not_run(tool_executor, settings_store, tmp_path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    _set_workspace(settings_store, ws)
    (ws / "f.txt").write_text("x")

    pending = tool_executor.submit("delete", {"path": str(ws / "f.txt")})
    decided = tool_executor.decide(pending["approval_id"], False)
    assert decided["status"] == "denied"
    assert (ws / "f.txt").exists()


def test_decide_unknown_approval(tool_executor) -> None:
    result = tool_executor.decide("nope", True)
    assert result["status"] == "error"


# ------------------------------------------------------------- terminal runner


def test_run_command_success_with_fake_runner(repository, audit, settings_store) -> None:
    def runner(command, cwd, timeout):
        return {"exit_code": 0, "stdout": "out", "stderr": "", "elapsed": 0.1}

    executor = ToolExecutor(build_default_registry(runner=runner), repository, audit, settings_store)
    pending = executor.submit("run_command", {"command": "echo hi"})
    decided = executor.decide(pending["approval_id"], True)
    assert decided["status"] == "success"
    assert decided["result"]["data"]["exit_code"] == 0
    assert decided["result"]["output"] == "out"


def test_run_command_nonzero_exit(repository, audit, settings_store) -> None:
    def runner(command, cwd, timeout):
        return {"exit_code": 7, "stdout": "", "stderr": "boom", "elapsed": 0.1}

    executor = ToolExecutor(build_default_registry(runner=runner), repository, audit, settings_store)
    pending = executor.submit("run_command", {"command": "false"})
    decided = executor.decide(pending["approval_id"], True)
    assert decided["status"] == "error"
    assert decided["result"]["data"]["exit_code"] == 7


def test_run_command_timeout(repository, audit, settings_store) -> None:
    def runner(command, cwd, timeout):
        raise TimeoutError("too slow")

    executor = ToolExecutor(build_default_registry(runner=runner), repository, audit, settings_store)
    pending = executor.submit("run_command", {"command": "sleep 100"})
    decided = executor.decide(pending["approval_id"], True)
    assert decided["status"] == "timeout"


# ------------------------------------------------------------- applications


def test_open_executable_escalates(repository, audit, settings_store, tmp_path) -> None:
    opened = []

    def opener(path):
        opened.append(path)

    ws = tmp_path / "ws"
    ws.mkdir()
    _set_workspace(settings_store, ws)
    executor = ToolExecutor(build_default_registry(opener=opener, revealer=lambda p: None), repository, audit, settings_store)

    exe = ws / "run.exe"
    exe.write_text("fake")
    result = executor.submit("open_file", {"path": str(exe)})
    assert result["status"] == "awaiting_approval"
    assert result["level"] == "system_action"
    assert opened == []


def test_open_text_file_auto_allowed(repository, audit, settings_store, tmp_path) -> None:
    opened = []

    def opener(path):
        opened.append(path)

    ws = tmp_path / "ws"
    ws.mkdir()
    _set_workspace(settings_store, ws)
    executor = ToolExecutor(build_default_registry(opener=opener, revealer=lambda p: None), repository, audit, settings_store)

    txt = ws / "notes.txt"
    txt.write_text("hi")
    result = executor.submit("open_file", {"path": str(txt)})
    assert result["status"] == "success"
    assert opened == [str(txt)]
