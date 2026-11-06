"""Tests for the internet-gated web tools."""
from __future__ import annotations

from app.agents.tool_executor import ToolExecutor
from app.tools.build import build_default_registry


def _enable_internet(settings_store) -> None:
    settings = settings_store.load()
    settings.internet_enabled = True
    settings_store.save(settings)


def _executor(repository, audit, settings_store, **fns) -> ToolExecutor:
    return ToolExecutor(build_default_registry(**fns), repository, audit, settings_store)


def test_web_search_disabled(repository, audit, settings_store) -> None:
    executor = _executor(repository, audit, settings_store)
    result = executor.submit("web_search", {"query": "test"})
    assert result["status"] == "error"
    assert "disabled" in result["result"]["error"]


def test_web_search_enabled(repository, audit, settings_store) -> None:
    _enable_internet(settings_store)

    def search(query, n):
        return [{"title": "Result", "url": "https://example.com", "snippet": "snippet"}]

    executor = _executor(repository, audit, settings_store, search_fn=search)
    result = executor.submit("web_search", {"query": "anything"})
    assert result["status"] == "success"
    assert result["result"]["data"]["results"][0]["title"] == "Result"


def test_web_fetch_requires_approval(repository, audit, settings_store) -> None:
    _enable_internet(settings_store)

    def fetch(url):
        return {"title": "Page", "url": url, "text": "body text"}

    executor = _executor(repository, audit, settings_store, fetch_fn=fetch)
    pending = executor.submit("web_fetch", {"url": "https://example.com"})
    assert pending["status"] == "awaiting_approval"

    decided = executor.decide(pending["approval_id"], True)
    assert decided["status"] == "success"
    assert decided["result"]["output"] == "body text"


def test_web_download_requires_approval_and_workspace(repository, audit, settings_store, tmp_path) -> None:
    _enable_internet(settings_store)
    ws = tmp_path / "ws"
    ws.mkdir()
    settings = settings_store.load()
    settings.approved_workspaces = [str(ws)]
    settings_store.save(settings)

    def download(url, output):
        return {"path": output, "size_bytes": 5}

    executor = _executor(repository, audit, settings_store, download_fn=download)
    pending = executor.submit("web_download", {"url": "https://example.com/f", "output": str(ws / "f.bin")})
    assert pending["status"] == "awaiting_approval"

    decided = executor.decide(pending["approval_id"], True)
    assert decided["status"] == "success"
    assert decided["result"]["data"]["size_bytes"] == 5
