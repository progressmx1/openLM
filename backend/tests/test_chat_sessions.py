"""Plain chat sessions (no workspace) with optional web lookups, and the web-search parser."""
from __future__ import annotations

import time

import pytest

from app.agents.coding.prompts import PromptLibrary
from app.agents.coding.service import AgentError, AgentService
from app.agents.coding.store import AgentStore
from app.agents.coding.tools import Workspace
from app.models.registry import ModelRegistry
from app.security.audit import AuditLog
from app.tools.web.web import page_text, parse_search_results

from .scripted_adapter import ScriptedAdapter, call, text

DDG_PAGE = """<html><body>
<div class="result results_links results_links_deep web-result">
  <h2 class="result__title"><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp&amp;rut=abc">llama.cpp on GitHub</a></h2>
  <a class="result__snippet" href="#">LLM inference in C/C++</a>
</div>
<div class="result result--ad"><a class="result__a" href="https://ads.example">Ad</a></div>
<div class="result"><h2><a class="result__a" href="https://example.org/page">Example</a></h2></div>
</body></html>"""


def test_search_parser_needs_no_cssselect_and_decodes_redirects() -> None:
    results = parse_search_results(DDG_PAGE, 5)
    assert results == [
        {"title": "llama.cpp on GitHub", "url": "https://github.com/ggml-org/llama.cpp", "snippet": "LLM inference in C/C++"},
        {"title": "Example", "url": "https://example.org/page", "snippet": ""},
    ]
    page = page_text(b"<html><head><title>T</title><script>x()</script></head><body><nav>menu</nav><p>Hello  world</p></body></html>", "u")
    assert page == {"title": "T", "url": "u", "text": "T Hello world"}


@pytest.fixture
def chat_env(repository, settings_store, tmp_path):
    adapter = ScriptedAdapter(native=True)
    registry = ModelRegistry(repository, settings_store, adapter=adapter)
    registry.discover(["/models"])
    model_id = registry.list_models()[0]["id"]
    registry.load(model_id)
    searches: list[str] = []

    def factory(root: str, internet: bool) -> Workspace:
        def search(query, n):
            searches.append(query)
            return [{"title": "Result", "url": "https://example.org/r", "snippet": "the answer is 42"}]

        return Workspace(root, internet_enabled=internet, search_fn=search,
                         fetch_fn=lambda url: {"title": "Page", "text": "full page text"})

    service = AgentService(AgentStore(repository.db), registry, settings_store, AuditLog(repository),
                           PromptLibrary(tmp_path / "prompts"), workspace_factory=factory)
    return service, adapter, model_id, searches


def _wait(service, sid, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(e["type"] == "turn_end" for e in service.events(sid)):
            return service.events(sid)
        time.sleep(0.02)
    raise AssertionError("turn did not end")


def test_chat_needs_no_workspace_and_offers_only_web_tools(chat_env, settings_store) -> None:
    service, adapter, model_id, searches = chat_env
    settings_store.save(settings_store.load().model_copy(update={"internet_enabled": True}))
    session = service.create_session(None, model_id, "chat")
    assert session["workspace"] == "" and session["mode"] == "chat" and session["title"] == "New chat"
    adapter.push(call("web_search", {"query": "meaning of life"}, "c1"), text("It's 42 ([source](https://example.org/r))."))
    service.send(session["id"], "what is the meaning of life?")
    events = _wait(service, session["id"])
    offered = {t["function"]["name"] for t in adapter.requests[0]["options"]["tools"]}
    assert offered == {"web_search", "web_fetch"}
    assert searches == ["meaning of life"]
    system = adapter.requests[0]["messages"][0]["content"]
    assert "web_search" in system and "cite the sources" in system and "workspace" not in system.lower()
    assert [e for e in events if e["type"] == "assistant"][-1]["content"].startswith("It's 42")
    assert service.describe(session["id"])["title"] == "what is the meaning of life?"


def test_chat_without_internet_has_no_tools_and_says_how_to_enable(chat_env) -> None:
    service, adapter, model_id, _ = chat_env
    session = service.create_session(None, model_id, "chat")
    adapter.push(text("Hello!"))
    service.send(session["id"], "hi")
    _wait(service, session["id"])
    assert "tools" not in adapter.requests[0]["options"]
    assert "turn on Web in the header" in adapter.requests[0]["messages"][0]["content"]


def test_chat_streams_even_for_json_variant_models(chat_env) -> None:
    service, adapter, model_id, _ = chat_env
    service.prompts.set("variants.json", '{"version": 1, "variants": [{"match": "", "tool_format": "json"}]}')
    session = service.create_session(None, model_id, "chat")
    adapter.push(text("A normal streamed answer."))
    service.send(session["id"], "hi")
    events = _wait(service, session["id"])
    assert "response_format" not in adapter.requests[0]["options"]
    assert [e for e in events if e["type"] == "assistant"][-1]["content"] == "A normal streamed answer."


def test_chat_and_code_sessions_are_listed_separately_and_cannot_switch(chat_env, tmp_path) -> None:
    service, _, model_id, _ = chat_env
    folder = tmp_path / "proj"
    folder.mkdir()
    chat = service.create_session(None, model_id, "chat")
    code = service.create_session(str(folder), model_id, "auto")
    assert [s["id"] for s in service.list_sessions(kind="chat")] == [chat["id"]]
    assert [s["id"] for s in service.list_sessions(str(folder), kind="code")] == [code["id"]]
    with pytest.raises(AgentError, match="can't become"):
        service.update_session(chat["id"], mode="auto")
    with pytest.raises(AgentError, match="can't become"):
        service.update_session(code["id"], mode="chat")


def test_code_prompt_tells_agent_to_look_things_up(chat_env, settings_store, tmp_path) -> None:
    service, _, model_id, _ = chat_env
    folder = tmp_path / "proj2"
    folder.mkdir()
    session = service.create_session(str(folder), model_id, "auto")
    assert "turn on Web in the header" in service.preview_system_prompt(session["id"])
    settings_store.save(settings_store.load().model_copy(update={"internet_enabled": True}))
    assert "When you are missing information: You can search the web" in service.preview_system_prompt(session["id"])


def test_chat_http_api(client) -> None:
    created = client.post("/api/agent/sessions", json={"mode": "chat"})
    assert created.status_code == 200, created.text
    assert created.json()["workspace"] == ""
    assert [s["id"] for s in client.get("/api/agent/sessions", params={"kind": "chat"}).json()] == [created.json()["id"]]
    assert client.post("/api/agent/sessions", json={"mode": "auto"}).status_code == 400


def test_chat_lookup_for_models_without_tool_calls(chat_env, settings_store) -> None:
    service, adapter, model_id, searches = chat_env
    settings_store.save(settings_store.load().model_copy(update={"internet_enabled": True}))
    service.prompts.set("variants.json", '{"version": 1, "variants": [{"match": "", "tool_format": "json"}]}')
    session = service.create_session(None, model_id, "chat")
    adapter.push(
        text('{"thought": "need current info", "name": "web_search", "arguments": {"query": "llama.cpp latest release"}}'),
        text('{"thought": "enough", "name": "answer", "arguments": {}}'),
        text("The latest release is b9999 ([GitHub](https://example.org/r))."),
    )
    service.send(session["id"], "latest llama.cpp release?")
    events = _wait(service, session["id"])
    assert searches == ["llama.cpp latest release"]
    # Decisions were grammar-forced (search/fetch/answer only); the answer was a normal streamed call.
    names = [o["properties"]["name"]["const"] for o in adapter.requests[0]["options"]["response_format"]["json_schema"]["schema"]["oneOf"]]
    assert set(names) == {"web_search", "web_fetch", "answer"}
    assert "response_format" not in adapter.requests[2]["options"] and "tools" not in adapter.requests[2]["options"]
    assert any("the answer is 42" in str(m.get("content")) for m in adapter.requests[2]["messages"])
    final = [e for e in events if e["type"] == "assistant"][-1]
    assert final["content"].startswith("The latest release is b9999")
    assert any(e["type"] == "tool" and e["name"] == "web_search" for e in events)
