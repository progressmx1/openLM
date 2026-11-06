"""Tests for the coding tools (content search and read-only Git inspection)."""
from __future__ import annotations

from app.agents.tool_executor import ToolExecutor
from app.tools.build import build_default_registry
from app.tools.coding.coding import search_content


def _set_workspace(settings_store, path) -> None:
    settings = settings_store.load()
    settings.approved_workspaces = [str(path)]
    settings_store.save(settings)


def test_search_content_function(tmp_path) -> None:
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    matches = search_content(str(tmp_path), "foo")
    assert len(matches) == 1
    assert matches[0]["line"] == 1


def test_search_content_tool(tool_executor, settings_store, tmp_path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    _set_workspace(settings_store, ws)
    (ws / "a.py").write_text("print('hi')\n", encoding="utf-8")

    result = tool_executor.submit("search_content", {"directory": str(ws), "pattern": "print"})
    assert result["status"] == "success"
    assert len(result["result"]["data"]["matches"]) == 1


def test_git_status_with_fake_runner(repository, audit, settings_store, tmp_path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    _set_workspace(settings_store, ws)

    def runner(args, timeout):
        return {"exit_code": 0, "stdout": " M file.txt\n", "stderr": ""}

    executor = ToolExecutor(build_default_registry(git_runner=runner), repository, audit, settings_store)
    result = executor.submit("git_status", {"path": str(ws)})
    assert result["status"] == "success"
    assert "M file.txt" in result["result"]["output"]


def test_git_diff_with_fake_runner(repository, audit, settings_store, tmp_path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    _set_workspace(settings_store, ws)

    def runner(args, timeout):
        return {"exit_code": 0, "stdout": "diff --git a/x b/x\n", "stderr": ""}

    executor = ToolExecutor(build_default_registry(git_runner=runner), repository, audit, settings_store)
    result = executor.submit("git_diff", {"path": str(ws)})
    assert result["status"] == "success"
    assert "diff --git" in result["result"]["output"]


def test_git_nonzero_exit_is_error(repository, audit, settings_store, tmp_path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    _set_workspace(settings_store, ws)

    def runner(args, timeout):
        return {"exit_code": 128, "stdout": "", "stderr": "not a git repository"}

    executor = ToolExecutor(build_default_registry(git_runner=runner), repository, audit, settings_store)
    result = executor.submit("git_status", {"path": str(ws)})
    assert result["status"] == "error"
    assert "not a git repository" in result["result"]["error"]
