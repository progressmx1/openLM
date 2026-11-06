"""Cloud (paid API) models: key storage, routing, OpenAI-compatible and Anthropic streaming.

No network: OpenAI-compatible calls go to an httpx MockTransport, Anthropic calls to a fake
client with the SDK's stream-event shape.
"""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

import httpx
import pytest
from fastapi.testclient import TestClient

from app.core.secrets import SecretStore, mask
from app.models.adapters.cloud import CloudAdapter, anthropic_messages, anthropic_tools, parse_path


# ------------------------------------------------------------------ fakes
class FakeSecrets(dict):
    def set(self, name, value):
        self[name] = value

    def delete(self, name):
        self.pop(name, None)


def sse(*chunks) -> bytes:
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks).encode() + b"data: [DONE]\n\n"


class Recorder:
    def __init__(self, body: bytes, status: int = 200, models=None):
        self.body, self.status, self.requests = body, status, []
        self.models = models or [{"id": "gpt-5"}, {"id": "deepseek-chat"}]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(200 if self.status == 200 else self.status, json={"data": self.models} if self.status == 200 else {"error": {"message": "bad key"}})
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"message": "nope"}})
        return httpx.Response(200, content=self.body, headers={"content-type": "text/event-stream"})


class FakeStream:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self.events)


def claude_events():
    return [
        NS(type="message_start", message=NS(usage=NS(input_tokens=120, cache_read_input_tokens=0, cache_creation_input_tokens=0))),
        NS(type="content_block_start", index=0, content_block=NS(type="thinking")),
        NS(type="content_block_delta", index=0, delta=NS(type="thinking_delta", thinking="Need to read the file.")),
        NS(type="content_block_start", index=1, content_block=NS(type="text")),
        NS(type="content_block_delta", index=1, delta=NS(type="text_delta", text="Reading it.")),
        NS(type="content_block_start", index=2, content_block=NS(type="tool_use", id="toolu_1", name="read_file")),
        NS(type="content_block_delta", index=2, delta=NS(type="input_json_delta", partial_json='{"path": ')),
        NS(type="content_block_delta", index=2, delta=NS(type="input_json_delta", partial_json='"a.py"}')),
        NS(type="message_delta", delta=NS(stop_reason="tool_use"), usage=NS(output_tokens=42)),
    ]


class FakeAnthropic:
    def __init__(self, events=None, fail_first_with_fallback=False):
        self.calls = []
        self._events = events or claude_events()
        self._fail = fail_first_with_fallback
        outer = self

        class Messages:
            def stream(self, **params):
                outer.calls.append(params)
                if outer._fail and "fallbacks" in params:
                    outer._fail = False
                    err = RuntimeError("fallbacks: not enabled for this account")
                    err.status_code = 400
                    raise err
                return FakeStream(outer._events)

        self.messages = Messages()
        self.beta = NS(messages=Messages())
        self.models = NS(list=lambda limit=100: NS(data=[NS(id="claude-opus-5-5", display_name="Claude Opus 5.5", max_input_tokens=1_000_000)]))


def make_adapter(paths, transport=None, anthropic=None, keys=None):
    secrets = FakeSecrets(keys if keys is not None else {"openai.api_key": "sk-o", "anthropic.api_key": "sk-ant", "deepseek.api_key": "sk-d"})
    return CloudAdapter(
        secrets,
        paths.get,
        http_transport=httpx.MockTransport(transport) if transport else None,
        anthropic_factory=(lambda key, base: anthropic) if anthropic else None,
    )


# ------------------------------------------------------------------ secrets
def test_secret_store_encrypts_and_masks(tmp_path):
    store = SecretStore(tmp_path / "secrets.bin")
    store.set("openai.api_key", "  sk-test-1234567890abcd ")
    assert store.get("openai.api_key") == "sk-test-1234567890abcd"
    assert b"sk-test-1234567890abcd" not in (tmp_path / "secrets.bin").read_bytes()
    assert mask("sk-test-1234567890abcd") == "sk-…abcd"
    store.delete("openai.api_key")
    assert store.names() == []


# ------------------------------------------------------------------ OpenAI-compatible
def responses_stream(*events) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def luna_tool_stream() -> bytes:
    return responses_stream(
        {"type": "response.reasoning_summary_text.delta", "delta": "Need the file."},
        {"type": "response.output_text.delta", "delta": "Reading."},
        {"type": "response.output_item.added", "output_index": 1, "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "read_file", "arguments": ""}},
        {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "output_index": 1, "delta": '{"path":'},
        {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "output_index": 1, "delta": '"a.py"}'},
        {"type": "response.function_call_arguments.done", "item_id": "fc_1", "arguments": '{"path":"a.py"}'},
        {"type": "response.output_item.done", "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "read_file", "arguments": '{"path":"a.py"}'}},
        {"type": "response.completed", "response": {"status": "completed", "usage": {"input_tokens": 10, "output_tokens": 5}}},
    )


def test_openai_uses_responses_api_with_tools_and_reasoning():
    """gpt-6-luna rejects function tools + reasoning in /chat/completions; /responses allows it."""
    rec = Recorder(luna_tool_stream())
    adapter = make_adapter({"m": "cloud://openai/gpt-6-luna"}, rec)
    messages = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "fix a.py"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "list_dir", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_0", "content": "a.py"},
    ]
    tools = [{"type": "function", "function": {"name": "read_file", "description": "Read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]
    events = list(adapter.stream_events("m", messages, {"temperature": 0.3, "max_tokens": 100, "tools": tools, "effort": "high"}))
    assert [e["type"] for e in events] == ["reasoning", "content", "tool_call", "tool_call", "tool_call", "done"]
    args = "".join(e["arguments"] for e in events if e["type"] == "tool_call")
    assert json.loads(args) == {"path": "a.py"}  # .done events don't duplicate the streamed arguments
    assert events[2]["id"] == "call_1" and events[2]["name"] == "read_file"
    assert events[-1]["finish_reason"] == "tool_calls" and events[-1]["usage"] == {"prompt_tokens": 10, "completion_tokens": 5}

    assert str(rec.requests[0].url) == "https://api.openai.com/v1/responses"
    assert rec.requests[0].headers["authorization"] == "Bearer sk-o"
    body = json.loads(rec.requests[0].content)
    assert body["model"] == "gpt-6-luna" and body["store"] is False
    assert body["reasoning"] == {"effort": "high", "summary": "auto"}
    assert "temperature" not in body and body["max_output_tokens"] >= 16_000
    assert body["instructions"] == "You are an agent."
    assert body["tools"] == [{"type": "function", "name": "read_file", "description": "Read", "parameters": tools[0]["function"]["parameters"], "strict": False}]
    assert body["input"] == [
        {"role": "user", "content": "fix a.py"},
        {"type": "function_call", "call_id": "call_0", "name": "list_dir", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_0", "output": "a.py"},
    ]


def test_responses_arguments_only_on_finished_item():
    rec = Recorder(responses_stream(
        {"type": "response.output_item.added", "item": {"type": "function_call", "id": "fc_9", "call_id": "call_9", "name": "grep", "arguments": ""}},
        {"type": "response.output_item.done", "item": {"type": "function_call", "id": "fc_9", "call_id": "call_9", "name": "grep", "arguments": '{"pattern":"x"}'}},
        {"type": "response.completed", "response": {"status": "completed", "usage": {}}},
    ))
    adapter = make_adapter({"m": "cloud://openai/gpt-6-luna"}, rec)
    result = adapter.chat("m", [{"role": "user", "content": "find x"}], {})
    assert result["tool_calls"] == [{"id": "call_9", "type": "function", "function": {"name": "grep", "arguments": '{"pattern":"x"}'}}]


def test_responses_retries_without_reasoning_summary_and_reports_failures():
    calls = []

    def picky(request):
        calls.append(json.loads(request.content))
        if "summary" in (calls[-1].get("reasoning") or {}):
            return httpx.Response(400, json={"error": {"message": "Your organization must be verified to generate reasoning summaries."}})
        return httpx.Response(200, content=responses_stream(
            {"type": "response.output_text.delta", "delta": "ok"},
            {"type": "response.completed", "response": {"status": "completed"}},
        ))

    adapter = make_adapter({"m": "cloud://openai/gpt-6-luna"}, picky)
    assert adapter.chat("m", [{"role": "user", "content": "hi"}], {})["content"] == "ok"
    assert len(calls) == 2 and calls[1]["reasoning"] == {"effort": "high"}

    failing = Recorder(responses_stream({"type": "response.failed", "response": {"error": {"message": "server overloaded"}}}))
    adapter = make_adapter({"m": "cloud://openai/gpt-6-luna"}, failing)
    with pytest.raises(RuntimeError, match="server overloaded"):
        list(adapter.stream_events("m", [{"role": "user", "content": "hi"}], {}))


def test_responses_input_answers_unrun_calls():
    from app.models.adapters.cloud import responses_input

    _, items = responses_input([
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "function": {"name": "x", "arguments": "{}"}}]},
        {"role": "user", "content": "stop"},
    ])
    assert [i.get("type") or i["role"] for i in items] == ["user", "function_call", "function_call_output", "user"]
    assert items[2] == {"type": "function_call_output", "call_id": "a", "output": "(not run)"}


def test_deepseek_reasoning_is_shown_and_uses_its_endpoint():
    rec = Recorder(sse(
        {"choices": [{"delta": {"reasoning_content": "thinking…"}}]},
        {"choices": [{"delta": {"content": "Answer"}, "finish_reason": "stop"}]},
    ))
    adapter = make_adapter({"d": "cloud://deepseek/deepseek-chat"}, rec)
    result = adapter.chat("d", [{"role": "user", "content": "q"}], {"temperature": 0.2})
    assert result["content"] == "Answer"
    events = list(adapter.stream_events("d", [{"role": "user", "content": "q"}], {}))
    assert events[0] == {"type": "reasoning", "text": "thinking…"}
    assert str(rec.requests[0].url) == "https://api.deepseek.com/chat/completions"
    assert json.loads(rec.requests[0].content)["temperature"] == 0.2


def test_bad_key_gives_a_readable_error():
    adapter = make_adapter({"m": "cloud://openai/gpt-5"}, Recorder(b"", status=401))
    with pytest.raises(RuntimeError, match="rejected the API key"):
        list(adapter.stream_events("m", [{"role": "user", "content": "hi"}], {}))


def test_ready_only_with_a_key():
    adapter = make_adapter({"m": "cloud://openai/gpt-5"}, keys={})
    assert not adapter.is_loaded("m")
    status = adapter.get_model_status("m")
    assert status["state"] == "no_key" and "API key" in status["error"]
    adapter.secrets.set("openai.api_key", "sk-x")
    assert adapter.is_loaded("m") and adapter.get_model_status("m")["runtime"] == "cloud"


# ------------------------------------------------------------------ Anthropic
def test_anthropic_stream_events_and_request_shape():
    fake = FakeAnthropic()
    adapter = make_adapter({"c": "cloud://anthropic/claude-opus-5-5"}, anthropic=fake)
    messages = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "fix a.py"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "toolu_0", "type": "function", "function": {"name": "list_dir", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "toolu_0", "content": "a.py"},
    ]
    tools = [{"type": "function", "function": {"name": "read_file", "description": "Read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]
    events = list(adapter.stream_events("c", messages, {"temperature": 0.3, "max_tokens": 4096, "tools": tools, "effort": "medium"}))
    kinds = [e["type"] for e in events]
    assert kinds == ["reasoning", "content", "tool_call", "tool_call", "tool_call", "done"]
    args = "".join(e["arguments"] for e in events if e["type"] == "tool_call")
    assert json.loads(args) == {"path": "a.py"}
    assert events[2]["id"] == "toolu_1" and events[2]["name"] == "read_file"
    assert events[-1]["finish_reason"] == "tool_calls" and events[-1]["usage"] == {"prompt_tokens": 120, "completion_tokens": 42}

    params = fake.calls[0]
    assert "temperature" not in params and "top_p" not in params and "top_k" not in params
    assert params["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert params["output_config"] == {"effort": "medium"}
    assert params["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in params["betas"]
    assert params["system"] == "You are an agent."
    assert params["tools"][0]["eager_input_streaming"] is True
    assert params["tools"][0]["input_schema"]["properties"]["path"] == {"type": "string"}
    assert params["max_tokens"] >= 16_000
    roles = [m["role"] for m in params["messages"]]
    assert roles == ["user", "assistant", "user"]
    assert params["messages"][1]["content"][0] == {"type": "tool_use", "id": "toolu_0", "name": "list_dir", "input": {}}
    assert params["messages"][2]["content"][0]["type"] == "tool_result"
    # Thinking is never sent back.
    assert "thinking" not in json.dumps(params["messages"])


def test_anthropic_retries_without_fallbacks_when_rejected():
    fake = FakeAnthropic(fail_first_with_fallback=True)
    adapter = make_adapter({"c": "cloud://anthropic/claude-opus-5-5"}, anthropic=fake)
    events = list(adapter.stream_events("c", [{"role": "user", "content": "hi"}], {}))
    assert events[-1]["type"] == "done"
    assert len(fake.calls) == 2 and "fallbacks" not in fake.calls[1] and "betas" not in fake.calls[1]


def test_anthropic_load_reads_context_window():
    adapter = make_adapter({"c": "cloud://anthropic/claude-opus-5-5"}, anthropic=FakeAnthropic())
    status = adapter.load_model("c", {})
    assert status["n_ctx_effective"] == 1_000_000


def test_message_conversion_merges_tool_results_and_closes_dangling_calls():
    system, out = anthropic_messages([
        {"role": "system", "content": "S"},
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "ok", "tool_calls": [
            {"id": "a", "function": {"name": "x", "arguments": "{}"}},
            {"id": "b", "function": {"name": "y", "arguments": "not json"}},
        ]},
        {"role": "tool", "tool_call_id": "a", "content": "1"},
        {"role": "user", "content": "continue"},
    ])
    assert system == "S"
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    results = [b for b in out[2]["content"] if b["type"] == "tool_result"]
    assert {r["tool_use_id"] for r in results} == {"a", "b"}
    assert out[2]["content"][-1] == {"type": "text", "text": "continue"}


def test_tool_schema_refs_are_inlined():
    tools = anthropic_tools([{"function": {"name": "t", "parameters": {
        "type": "object", "properties": {"item": {"$ref": "#/$defs/Item"}}, "$defs": {"Item": {"type": "string"}},
    }}}])
    assert tools[0]["input_schema"] == {"type": "object", "properties": {"item": {"type": "string"}}}


def test_parse_path():
    assert parse_path("cloud://deepseek/deepseek-chat") == ("deepseek", "deepseek-chat")
    with pytest.raises(ValueError):
        parse_path("cloud://nope/x")


# ------------------------------------------------------------------ API + routing
@pytest.fixture
def cloud_state(tmp_path, repository, settings_store, fake_adapter, chat_service, tool_executor, agent_loop, retrieval, multi_agent):
    from app.api.deps import AppState
    from app.main import with_cloud
    from app.models.registry import ModelRegistry

    adapter = with_cloud(fake_adapter, repository, settings_store, SecretStore(tmp_path / "secrets.bin"))
    registry = ModelRegistry(repository, settings_store, adapter=adapter)
    return AppState(repository, settings_store, registry, chat_service, tool_executor, agent_loop, retrieval, multi_agent)


@pytest.fixture
def cloud_client(cloud_state):
    from app.main import create_app_with_state

    with TestClient(create_app_with_state(cloud_state)) as c:
        yield c


def test_provider_routes_store_key_encrypted_and_register_models(cloud_client: TestClient, tmp_path):
    client = cloud_client
    providers = client.get("/api/providers").json()
    assert {p["id"] for p in providers} >= {"openai", "anthropic", "deepseek", "custom"}
    assert not next(p for p in providers if p["id"] == "openai")["configured"]

    r = client.put("/api/providers/openai", json={"api_key": "sk-proj-abcdefgh12345678"})
    assert r.status_code == 200
    body = r.json()
    assert body["configured"] and body["masked_key"] == "sk-…5678"
    assert "sk-proj-abcdefgh12345678" not in json.dumps(client.get("/api/providers").json())

    assert client.put("/api/providers/openai", json={"api_key": "sk bad key"}).status_code == 400
    assert client.put("/api/providers/custom", json={"base_url": "ftp://x"}).status_code == 400

    model = client.post("/api/providers/openai/models", json={"model": "gpt-5"}).json()
    assert model["path"] == "cloud://openai/gpt-5" and model["runtime"] == "cloud"
    assert model["loaded"] is True and model["status"]["provider"] == "openai"
    # Adding it again returns the same record.
    assert client.post("/api/providers/openai/models", json={"model": "gpt-5"}).json()["id"] == model["id"]

    client.delete("/api/providers/openai/key")
    assert client.get(f"/api/models/{model['id']}").json()["loaded"] is False


def test_cloud_models_never_unload_local_models(cloud_state):
    app_state = cloud_state
    from app.models.adapters.cloud import CloudAdapter
    from app.models.adapters.router import ProviderRouter

    router = app_state.registry.adapter
    assert isinstance(router, ProviderRouter) and isinstance(router.cloud, CloudAdapter)
    router.cloud.secrets.set("openai.api_key", "sk-x")
    model = app_state.registry.register_cloud("openai", "gpt-5")
    assert router.plan_load(model["id"], {})["physical_devices"] == [f"cloud:{model['id']}"]
    assert model["id"] not in router.get_runtime_info()["loaded_models"]


def test_unsupported_temperature_is_dropped_and_retried():
    def make_picky(ok_body):
        calls = []

        def picky(request):
            calls.append(json.loads(request.content))
            if "temperature" in calls[-1]:
                return httpx.Response(400, json={"error": {"message": "Unsupported parameter: 'temperature'"}})
            return httpx.Response(200, content=ok_body)

        return calls, picky

    # OpenAI (Responses API), a non-reasoning model name that still rejects temperature.
    calls, picky = make_picky(responses_stream({"type": "response.output_text.delta", "delta": "fine"}, {"type": "response.completed", "response": {}}))
    adapter = make_adapter({"n": "cloud://openai/some-new-model"}, picky)
    assert adapter.chat("n", [{"role": "user", "content": "hi"}], {"temperature": 0.3})["content"] == "fine"
    assert len(calls) == 2 and "temperature" not in calls[1]

    # OpenAI-compatible /chat/completions (custom endpoint).
    calls, picky = make_picky(sse({"choices": [{"delta": {"content": "fine"}, "finish_reason": "stop"}]}))
    adapter = make_adapter({"c": "cloud://custom/some-model"}, picky, keys={"custom.api_key": "k"})
    adapter._base_urls = lambda: {"custom": "https://llm.example/v1"}
    assert adapter.chat("c", [{"role": "user", "content": "hi"}], {"temperature": 0.3})["content"] == "fine"
    assert len(calls) == 2 and "temperature" not in calls[1]


def test_suggested_models_include_latest():
    from app.models.adapters.cloud import PROVIDERS

    assert "gpt-6-luna" in PROVIDERS["openai"]["suggested"]
    assert "claude-haiku-5-5" in PROVIDERS["anthropic"]["suggested"]


def test_haiku_gets_no_fallbacks():
    fake = FakeAnthropic()
    adapter = make_adapter({"h": "cloud://anthropic/claude-haiku-5-5"}, anthropic=fake)
    list(adapter.stream_events("h", [{"role": "user", "content": "hi"}], {}))
    assert "fallbacks" not in fake.calls[0] and "temperature" not in fake.calls[0]
    assert fake.calls[0]["thinking"]["type"] == "adaptive"
