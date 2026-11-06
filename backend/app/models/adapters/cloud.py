"""Paid cloud models via API: OpenAI (ChatGPT models), Anthropic (Claude), DeepSeek, and any
OpenAI-compatible endpoint.

A cloud model is registered with the path ``cloud://<provider>/<model>``. There is nothing to
load: a model is ready as soon as its provider has an API key (keys live in the encrypted
:class:`~app.core.secrets.SecretStore`). "Load" only checks the key against the provider.

* OpenAI / DeepSeek / custom: ``/chat/completions`` over SSE, parsed by the same code as
  llama-server (:mod:`.openai_stream`), so native tool calls work unchanged.
* Anthropic: the official ``anthropic`` SDK (Messages API, streaming, adaptive thinking).
  Thinking is shown in the UI but never sent back: the agent edits its history (compaction,
  shortened tool output), and replayed thinking blocks must match the original exactly.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable, Iterable, Iterator, Optional

import httpx

from ...core.logging import get_logger
from .base import ModelAdapter, ModelLoadError, ModelNotLoadedError
from .openai_stream import StreamState, iter_openai_events

log = get_logger(__name__)

PREFIX = "cloud://"

PROVIDERS: dict[str, dict[str, Any]] = {
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "key_url": "https://platform.openai.com/api-keys",
        "context": 128_000,
        "suggested": ["gpt-6-luna", "gpt-6-sol", "gpt-5", "gpt-5-mini"],
    },
    "anthropic": {
        "label": "Anthropic",
        "base_url": "https://api.anthropic.com",
        "key_url": "https://console.anthropic.com/settings/keys",
        "context": 200_000,
        "suggested": ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-1"],
    },
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "key_url": "https://platform.deepseek.com/api_keys",
        "context": 128_000,
        "suggested": ["deepseek-chat", "deepseek-reasoner"],
    },
    "custom": {
        "label": "OpenAI-compatible",
        "base_url": "",
        "key_url": "",
        "context": 32_000,
        "suggested": [],
    },
}

# Claude models that accept server-side fallbacks (another model answers when this one is
# overloaded or refuses), enabled by default.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5")


def parse_path(path: str) -> tuple[str, str]:
    """``cloud://openai/gpt-5`` -> ``("openai", "gpt-5")``."""
    if not path.startswith(PREFIX):
        raise ValueError(f"Not a cloud model path: {path}")
    provider, _, model = path[len(PREFIX):].partition("/")
    if provider not in PROVIDERS or not model:
        raise ValueError(f"Unknown cloud model: {path}")
    return provider, model


def make_path(provider: str, model: str) -> str:
    return f"{PREFIX}{provider}/{model}"


def _openai_reasoning_model(model: str) -> bool:
    name = model.lower()
    # o-series and GPT-5 / GPT-6 (Sol, Luna, …) reason first and reject a custom temperature.
    return name.startswith(("o1", "o3", "o4", "gpt-5", "gpt-6"))


# ---------------------------------------------------------------- Anthropic message mapping
def _inline_refs(schema: Any, defs: Optional[dict[str, Any]] = None) -> Any:
    """Resolve ``$ref``s into ``$defs`` (tool schemas come from pydantic)."""
    if isinstance(schema, dict):
        defs = {**(defs or {}), **schema.get("$defs", {})}
        ref = schema.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/") and ref[8:] in defs:
            return _inline_refs(defs[ref[8:]], defs)
        return {k: _inline_refs(v, defs) for k, v in schema.items() if k != "$defs"}
    if isinstance(schema, list):
        return [_inline_refs(item, defs) for item in schema]
    return schema


def anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for tool in tools:
        fn = tool.get("function") or tool
        out.append({
            "name": fn["name"],
            "description": fn.get("description") or "",
            "input_schema": _inline_refs(fn.get("parameters") or {"type": "object", "properties": {}}),
            # Large inputs (whole files) stream as they are written instead of in one burst.
            "eager_input_streaming": True,
        })
    return out


def anthropic_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """OpenAI-style messages -> (system, Anthropic messages).

    Assistant ``tool_calls`` become ``tool_use`` blocks; consecutive ``tool`` messages become
    one user message of ``tool_result`` blocks; roles are merged so they alternate. Thinking is
    never included.
    """
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []

    def push(role: str, blocks: list[dict[str, Any]]) -> None:
        if not blocks:
            return
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": list(blocks)})

    for message in messages:
        role = message.get("role")
        content = message.get("content")
        text = content if isinstance(content, str) else ("" if content is None else json.dumps(content))
        if role == "system":
            if text.strip():
                system_parts.append(text)
        elif role == "tool":
            push("user", [{
                "type": "tool_result",
                "tool_use_id": message.get("tool_call_id") or "unknown",
                "content": text or "(no output)",
            }])
        elif role == "assistant":
            blocks: list[dict[str, Any]] = []
            if text.strip():
                blocks.append({"type": "text", "text": text})
            for call in message.get("tool_calls") or []:
                fn = call.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                blocks.append({"type": "tool_use", "id": call.get("id"), "name": fn.get("name"), "input": args if isinstance(args, dict) else {}})
            push("assistant", blocks)
        else:
            if text.strip():
                push("user", [{"type": "text", "text": text}])
    # The conversation must start with a user turn and every tool_use needs its result.
    if out and out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": [{"type": "text", "text": "(continue)"}]})
    _close_dangling_tool_uses(out)
    return "\n\n".join(system_parts), out


def _close_dangling_tool_uses(out: list[dict[str, Any]]) -> None:
    for i, message in enumerate(out):
        if message["role"] != "assistant":
            continue
        ids = [b["id"] for b in message["content"] if b.get("type") == "tool_use"]
        if not ids:
            continue
        nxt = out[i + 1] if i + 1 < len(out) else None
        answered = {b.get("tool_use_id") for b in (nxt or {}).get("content", []) if b.get("type") == "tool_result"}
        missing = [{"type": "tool_result", "tool_use_id": t, "content": "(not run)"} for t in ids if t not in answered]
        if not missing:
            continue
        if nxt is not None and nxt["role"] == "user":
            nxt["content"][:0] = missing
        else:
            out.insert(i + 1, {"role": "user", "content": missing})
    # tool_result blocks must come first in their user message.
    for message in out:
        if message["role"] == "user":
            message["content"].sort(key=lambda b: 0 if b.get("type") == "tool_result" else 1)


# ---------------------------------------------------------------------------- the adapter
class CloudAdapter(ModelAdapter):
    runtime_name = "cloud"

    def __init__(
        self,
        secrets: Any,
        resolve_path: Callable[[str], Optional[str]],
        base_urls: Optional[Callable[[], dict[str, str]]] = None,
        http_transport: Optional[httpx.BaseTransport] = None,
        anthropic_factory: Optional[Callable[[str, Optional[str]], Any]] = None,
    ) -> None:
        self.secrets = secrets
        self._resolve = resolve_path
        self._base_urls = base_urls or (lambda: {})
        self._transport = http_transport
        self._anthropic_factory = anthropic_factory
        self._lock = threading.Lock()
        self._timings: dict[str, dict[str, Any]] = {}
        self._context: dict[str, int] = {}
        self._errors: dict[str, str] = {}

    # ------------------------------------------------------------ helpers
    def _model(self, model_id: str) -> tuple[str, str]:
        path = self._resolve(model_id)
        if not path:
            raise ModelNotLoadedError(f"Unknown cloud model: {model_id}")
        return parse_path(path)

    def key(self, provider: str) -> Optional[str]:
        return self.secrets.get(f"{provider}.api_key")

    def base_url(self, provider: str) -> str:
        return (self._base_urls().get(provider) or PROVIDERS[provider]["base_url"]).rstrip("/")

    def _key_or_raise(self, provider: str) -> str:
        key = self.key(provider)
        if not key:
            raise ModelNotLoadedError(f"No API key for {PROVIDERS[provider]['label']}. Add one in Models & GPUs → Cloud models.")
        return key

    def _http(self, provider: str, key: str) -> httpx.Client:
        base = self.base_url(provider)
        if not base:
            raise ModelLoadError("Set the endpoint URL for the OpenAI-compatible provider first.")
        return httpx.Client(
            base_url=base,
            headers={"Authorization": f"Bearer {key}"},
            timeout=httpx.Timeout(20.0, read=300.0),
            transport=self._transport,
        )

    def _anthropic(self, key: str) -> Any:
        base = self._base_urls().get("anthropic") or None
        if self._anthropic_factory is not None:
            return self._anthropic_factory(key, base)
        import anthropic

        return anthropic.Anthropic(api_key=key, base_url=base, max_retries=2)

    # ------------------------------------------------------------ provider API
    def list_provider_models(self, provider: str) -> list[dict[str, Any]]:
        """Models the provider offers for this key (raises with a readable message)."""
        key = self._key_or_raise(provider)
        if provider == "anthropic":
            try:
                page = self._anthropic(key).models.list(limit=100)
            except Exception as exc:  # noqa: BLE001
                raise ModelLoadError(_friendly_error(exc)) from exc
            out = []
            for item in getattr(page, "data", None) or list(page):
                out.append({
                    "id": item.id,
                    "name": getattr(item, "display_name", None) or item.id,
                    "context": getattr(item, "max_input_tokens", None),
                })
            return out
        with self._http(provider, key) as client:
            try:
                response = client.get("/models")
            except httpx.HTTPError as exc:
                raise ModelLoadError(f"Could not reach {PROVIDERS[provider]['label']}: {exc}") from exc
        if response.status_code != 200:
            raise ModelLoadError(_http_error(provider, response))
        data = response.json().get("data") or []
        return sorted(
            ({"id": m["id"], "name": m["id"], "context": m.get("context_length")} for m in data if m.get("id")),
            key=lambda m: m["id"],
        )

    # ------------------------------------------------------------ interface
    def discover_models(self, locations: Iterable[str]) -> list[dict[str, Any]]:
        return []

    def plan_load(self, model_id: str, settings: dict[str, Any]) -> dict[str, Any]:
        # A cloud model uses no local GPU, so loading it never unloads a local model.
        return {"physical_devices": [f"cloud:{model_id}"]}

    def load_model(self, model_id: str, settings: dict[str, Any]) -> dict[str, Any]:
        provider, model = self._model(model_id)
        try:
            self._key_or_raise(provider)
            models = self.list_provider_models(provider)
        except (ModelLoadError, ModelNotLoadedError) as exc:
            self._errors[model_id] = str(exc)
            raise ModelLoadError(str(exc)) from exc
        self._errors.pop(model_id, None)
        match = next((m for m in models if m["id"] == model), None)
        if match and match.get("context"):
            self._context[model_id] = int(match["context"])
        return self.get_model_status(model_id)

    def unload_model(self, model_id: str) -> None:
        return None

    def is_loaded(self, model_id: str) -> bool:
        try:
            provider, _ = self._model(model_id)
        except (ModelNotLoadedError, ValueError):
            return False
        return bool(self.key(provider))

    def supports_native_tools(self, model_id: str) -> bool:
        return True

    def get_model_status(self, model_id: str) -> dict[str, Any]:
        try:
            provider, model = self._model(model_id)
        except (ModelNotLoadedError, ValueError):
            return {"loaded": False}
        ready = bool(self.key(provider))
        status: dict[str, Any] = {
            "loaded": ready,
            "state": "ready" if ready else "no_key",
            "runtime": "cloud",
            "provider": provider,
            "provider_label": PROVIDERS[provider]["label"],
            "remote_model": model,
            "n_ctx_effective": self._context.get(model_id) or PROVIDERS[provider]["context"],
            "native_tools": True,
            "physical_devices": [f"cloud:{model_id}"],
            "last_timings": self._timings.get(model_id),
        }
        if not ready:
            status["error"] = f"No API key for {PROVIDERS[provider]['label']}. Add one in Models & GPUs → Cloud models."
        elif model_id in self._errors:
            status["warning"] = self._errors[model_id]
        return status

    def get_runtime_info(self) -> dict[str, Any]:
        return {"name": "cloud", "providers": [p for p in PROVIDERS if self.key(p)], "loaded_models": []}

    def chat(self, model_id: str, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        content: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        done: dict[str, Any] = {}
        for event in self.stream_events(model_id, messages, options):
            if event["type"] == "content":
                content.append(event["text"])
            elif event["type"] == "tool_call":
                slot = calls.setdefault(event.get("index", 0), {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                slot["id"] = event.get("id") or slot["id"]
                slot["function"]["name"] = event.get("name") or slot["function"]["name"]
                slot["function"]["arguments"] += event.get("arguments") or ""
            elif event["type"] == "done":
                done = event
        return {
            "content": "".join(content),
            "tool_calls": [calls[i] for i in sorted(calls)],
            "finish_reason": done.get("finish_reason"),
            "usage": done.get("usage") or {},
            "timings": done.get("timings"),
        }

    def stream_chat(self, model_id: str, messages: list[dict[str, Any]], options: dict[str, Any]) -> Iterator[str]:
        for event in self.stream_events(model_id, messages, options):
            if event["type"] == "content":
                yield event["text"]

    def stream_events(self, model_id: str, messages: list[dict[str, Any]], options: dict[str, Any]) -> Iterator[dict[str, Any]]:
        provider, model = self._model(model_id)
        key = self._key_or_raise(provider)
        started = time.monotonic()
        if provider == "anthropic":
            events = self._anthropic_events(model, key, messages, options)
        else:
            if provider == "openai":
                # Responses API: GPT-5/6 reasoning models only take function tools with reasoning
                # on there, not in /chat/completions.
                events = self._responses_events(model, key, messages, options)
            else:
                events = self._openai_events(provider, model, key, messages, options)
        done: dict[str, Any] = {"type": "done", "finish_reason": None, "usage": {}, "timings": None}
        for event in events:
            if event["type"] == "done":
                done = event
                continue
            yield event
        usage = done.get("usage") or {}
        elapsed = max(time.monotonic() - started, 1e-6)
        timings = {
            "prompt_n": usage.get("prompt_tokens"),
            "predicted_n": usage.get("completion_tokens"),
            "predicted_per_second": round((usage.get("completion_tokens") or 0) / elapsed, 1) or None,
        }
        self._timings[model_id] = timings
        yield {**done, "timings": timings}

    # ------------------------------------------------------------ OpenAI-compatible
    def _openai_events(
        self, provider: str, model: str, key: str, messages: list[dict[str, Any]], options: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [_openai_message(m) for m in messages],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        max_tokens = int(options.get("max_tokens", 4096))
        if provider == "openai":
            payload["max_completion_tokens"] = max_tokens
            if not _openai_reasoning_model(model):
                payload["temperature"] = float(options.get("temperature", 0.7))
        else:
            payload["max_tokens"] = max_tokens
            if "reasoner" not in model:
                payload["temperature"] = float(options.get("temperature", 0.7))
        if options.get("tools"):
            payload["tools"] = options["tools"]
            payload["tool_choice"] = options.get("tool_choice", "auto")
        if options.get("stop") and not (provider == "openai" and _openai_reasoning_model(model)):
            payload["stop"] = options["stop"]
        state = StreamState()
        with self._http(provider, key) as client:
            try:
                for attempt in range(2):
                    with client.stream("POST", "/chat/completions", json=payload) as response:
                        if response.status_code != 200:
                            response.read()
                            # Newer models may reject sampling parameters: drop them and retry once.
                            unsupported = [k for k in ("temperature", "stop") if k in payload and k in response.text]
                            if attempt == 0 and response.status_code == 400 and unsupported:
                                for k in unsupported:
                                    payload.pop(k)
                                continue
                            raise RuntimeError(_http_error(provider, response))
                        yield from iter_openai_events(response.iter_lines(), state)
                        break
            except httpx.HTTPError as exc:
                raise RuntimeError(f"Could not reach {PROVIDERS[provider]['label']}: {exc}") from exc
        yield {"type": "done", "finish_reason": state.finish_reason, "usage": state.usage, "timings": None}

    # ------------------------------------------------------------ Anthropic
    def anthropic_request(self, model: str, messages: list[dict[str, Any]], options: dict[str, Any], fallbacks: bool = True) -> dict[str, Any]:
        system, converted = anthropic_messages(messages)
        params: dict[str, Any] = {
            "model": model,
            # Thinking shares max_tokens with the answer, so leave room for both.
            "max_tokens": max(int(options.get("max_tokens", 4096)), 16_000),
            "messages": converted,
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": options.get("effort") or "high"},
            "betas": [],
        }
        if system:
            params["system"] = system
        if options.get("tools"):
            params["tools"] = anthropic_tools(options["tools"])
        if fallbacks and model.startswith(_FALLBACK_MODELS):
            params["fallbacks"] = "default"
            params["betas"].append(_FALLBACK_BETA)
        if not params["betas"]:
            params.pop("betas")
        return params

    def _anthropic_events(self, model: str, key: str, messages: list[dict[str, Any]], options: dict[str, Any]) -> Iterator[dict[str, Any]]:
        client = self._anthropic(key)
        params = self.anthropic_request(model, messages, options)
        try:
            yield from self._anthropic_stream(client, params)
        except _RetryWithoutFallbacks:
            params = self.anthropic_request(model, messages, options, fallbacks=False)
            yield from self._anthropic_stream(client, params)

    def _anthropic_stream(self, client: Any, params: dict[str, Any]) -> Iterator[dict[str, Any]]:
        stream_fn = client.beta.messages.stream if "betas" in params else client.messages.stream
        if "betas" not in params:
            params = {k: v for k, v in params.items() if k not in ("fallbacks",)}
        blocks: dict[int, str] = {}
        tool_slot: dict[int, int] = {}
        stop_reason = None
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        yielded = False
        try:
            with stream_fn(**params) as stream:
                try:
                    for event in stream:
                        etype = getattr(event, "type", "")
                        if etype == "message_start":
                            u = getattr(event.message, "usage", None)
                            if u is not None:
                                usage["prompt_tokens"] = (getattr(u, "input_tokens", 0) or 0) + (getattr(u, "cache_read_input_tokens", 0) or 0) + (getattr(u, "cache_creation_input_tokens", 0) or 0)
                        elif etype == "content_block_start":
                            block = event.content_block
                            blocks[event.index] = block.type
                            if block.type == "tool_use":
                                tool_slot[event.index] = len(tool_slot)
                                yielded = True
                                yield {"type": "tool_call", "index": tool_slot[event.index], "id": block.id, "name": block.name, "arguments": ""}
                        elif etype == "content_block_delta":
                            delta = event.delta
                            dtype = getattr(delta, "type", "")
                            if dtype == "text_delta" and delta.text:
                                yielded = True
                                yield {"type": "content", "text": delta.text}
                            elif dtype == "thinking_delta" and delta.thinking:
                                yielded = True
                                yield {"type": "reasoning", "text": delta.thinking}
                            elif dtype == "input_json_delta" and event.index in tool_slot:
                                yield {"type": "tool_call", "index": tool_slot[event.index], "id": None, "name": None, "arguments": delta.partial_json or ""}
                        elif etype == "message_delta":
                            stop_reason = getattr(event.delta, "stop_reason", None) or stop_reason
                            u = getattr(event, "usage", None)
                            if u is not None and getattr(u, "output_tokens", None) is not None:
                                usage["completion_tokens"] = u.output_tokens
                except ValueError:
                    # The SDK could not assemble a (truncated) streamed tool input; the agent
                    # parses the fragments itself and reports invalid arguments to the model.
                    log.warning("Anthropic stream: tool input could not be assembled", exc_info=True)
        except Exception as exc:  # noqa: BLE001
            status = getattr(exc, "status_code", None)
            if status == 400 and not yielded and "fallbacks" in params and "fallback" in str(exc).lower():
                raise _RetryWithoutFallbacks() from exc
            raise RuntimeError(_friendly_error(exc)) from exc
        if stop_reason == "refusal":
            yield {"type": "content", "text": "\n\n(Claude declined to continue this request.)"}
        finish = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length", "refusal": "stop"}.get(stop_reason or "", stop_reason)
        yield {"type": "done", "finish_reason": finish, "usage": usage, "timings": None}


    # ------------------------------------------------------------ OpenAI Responses API
    def responses_request(self, model: str, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        instructions, items = responses_input(messages)
        reasoning_model = _openai_reasoning_model(model)
        max_tokens = int(options.get("max_tokens", 4096))
        payload: dict[str, Any] = {
            "model": model,
            "input": items,
            "stream": True,
            "store": False,
            # Reasoning shares the output budget with the answer, so leave room for both.
            "max_output_tokens": max(max_tokens, 16_000) if reasoning_model else max_tokens,
        }
        if instructions:
            payload["instructions"] = instructions
        if reasoning_model:
            payload["reasoning"] = {"effort": options.get("effort") or "high", "summary": "auto"}
        else:
            payload["temperature"] = float(options.get("temperature", 0.7))
        if options.get("tools"):
            payload["tools"] = [responses_tool(t) for t in options["tools"]]
            payload["tool_choice"] = "auto"
            payload["parallel_tool_calls"] = False
        return payload

    def _responses_events(self, model: str, key: str, messages: list[dict[str, Any]], options: dict[str, Any]) -> Iterator[dict[str, Any]]:
        payload = self.responses_request(model, messages, options)
        slots: dict[str, int] = {}  # output item id -> tool_call index
        sent_args: set[str] = set()
        usage: dict[str, Any] = {}
        status = None
        with self._http("openai", key) as client:
            try:
                for attempt in range(3):
                    with client.stream("POST", "/responses", json=payload) as response:
                        if response.status_code != 200:
                            response.read()
                            if attempt < 2 and response.status_code == 400 and _drop_rejected(payload, response.text):
                                continue
                            raise RuntimeError(_http_error("openai", response))
                        for line in response.iter_lines():
                            if not line.startswith("data:"):
                                continue
                            try:
                                event = json.loads(line[5:].strip())
                            except json.JSONDecodeError:
                                continue
                            for out in self._responses_event(event, slots, sent_args):
                                yield out
                            etype = event.get("type", "")
                            if etype in ("response.completed", "response.incomplete"):
                                body = event.get("response") or {}
                                status = body.get("status") or etype.rsplit(".", 1)[-1]
                                u = body.get("usage") or {}
                                usage = {"prompt_tokens": u.get("input_tokens"), "completion_tokens": u.get("output_tokens")}
                        break
            except httpx.HTTPError as exc:
                raise RuntimeError(f"Could not reach OpenAI: {exc}") from exc
        finish = "tool_calls" if slots else ("length" if status == "incomplete" else "stop")
        yield {"type": "done", "finish_reason": finish, "usage": usage, "timings": None}

    @staticmethod
    def _responses_event(event: dict[str, Any], slots: dict[str, int], sent_args: set[str]) -> Iterator[dict[str, Any]]:
        etype = event.get("type", "")
        if etype == "response.output_text.delta":
            yield {"type": "content", "text": event.get("delta") or ""}
        elif etype in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            yield {"type": "reasoning", "text": event.get("delta") or ""}
        elif etype == "response.reasoning_summary_part.done":
            yield {"type": "reasoning", "text": "\n\n"}
        elif etype in ("response.output_item.added", "response.output_item.done"):
            item = event.get("item") or {}
            if item.get("type") != "function_call":
                return
            item_id = item.get("id") or item.get("call_id")
            if item_id not in slots:
                slots[item_id] = len(slots)
                yield {"type": "tool_call", "index": slots[item_id], "id": item.get("call_id"), "name": item.get("name"), "arguments": ""}
            # Some streams only carry the arguments on the finished item.
            if etype == "response.output_item.done" and item_id not in sent_args and item.get("arguments"):
                sent_args.add(item_id)
                yield {"type": "tool_call", "index": slots[item_id], "id": None, "name": None, "arguments": item["arguments"]}
        elif etype == "response.function_call_arguments.delta":
            item_id = event.get("item_id")
            if item_id not in slots:
                slots[item_id] = len(slots)
            sent_args.add(item_id)
            yield {"type": "tool_call", "index": slots[item_id], "id": None, "name": None, "arguments": event.get("delta") or ""}
        elif etype == "response.function_call_arguments.done":
            item_id = event.get("item_id")
            if item_id in slots and item_id not in sent_args and event.get("arguments"):
                sent_args.add(item_id)
                yield {"type": "tool_call", "index": slots[item_id], "id": None, "name": None, "arguments": event["arguments"]}
        elif etype in ("response.failed", "error"):
            err = (event.get("response") or {}).get("error") or event.get("error") or event
            raise RuntimeError(f"OpenAI error: {err.get('message') if isinstance(err, dict) else err}")


def responses_tool(tool: dict[str, Any]) -> dict[str, Any]:
    fn = tool.get("function") or tool
    # strict=False: the agent's schemas have optional fields, which strict mode does not allow.
    return {
        "type": "function",
        "name": fn["name"],
        "description": fn.get("description") or "",
        "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
        "strict": False,
    }


def responses_input(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """OpenAI chat messages -> (instructions, Responses API input items)."""
    instructions: list[str] = []
    items: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        text = content if isinstance(content, str) else ("" if content is None else json.dumps(content))
        if role == "system":
            if text.strip():
                instructions.append(text)
        elif role == "tool":
            items.append({"type": "function_call_output", "call_id": message.get("tool_call_id") or "unknown", "output": text or "(no output)"})
        elif role == "assistant":
            if text.strip():
                items.append({"role": "assistant", "content": text})
            for call in message.get("tool_calls") or []:
                fn = call.get("function") or {}
                items.append({"type": "function_call", "call_id": call.get("id"), "name": fn.get("name"), "arguments": fn.get("arguments") or "{}"})
        elif text.strip():
            items.append({"role": "user", "content": text})
    # Every function_call needs an output; a missing one is added right after its calls.
    answered = {i["call_id"] for i in items if i.get("type") == "function_call_output"}
    out: list[dict[str, Any]] = []
    pending: list[str] = []
    for item in items:
        if item.get("type") == "function_call":
            out.append(item)
            if item["call_id"] not in answered:
                pending.append(item["call_id"])
            continue
        if pending and item.get("type") != "function_call_output":
            out.extend({"type": "function_call_output", "call_id": c, "output": "(not run)"} for c in pending)
            pending = []
        out.append(item)
    out.extend({"type": "function_call_output", "call_id": c, "output": "(not run)"} for c in pending)
    return "\n\n".join(instructions), out


def _drop_rejected(payload: dict[str, Any], error_text: str) -> bool:
    """Remove a parameter the model rejected; True if something was removed."""
    text = error_text.lower()
    reasoning = payload.get("reasoning") or {}
    if "temperature" in text and "temperature" in payload:
        payload.pop("temperature")
        return True
    if "summar" in text and "summary" in reasoning:
        # Reasoning summaries need a verified organisation; work without them.
        reasoning.pop("summary")
        return True
    if "reasoning" in text and "effort" in text and reasoning.get("effort") not in (None, "medium"):
        reasoning["effort"] = "medium"
        return True
    return False


class _RetryWithoutFallbacks(Exception):
    pass


def _openai_message(message: dict[str, Any]) -> dict[str, Any]:
    allowed = ("role", "content", "tool_calls", "tool_call_id", "name")
    msg = {k: v for k, v in message.items() if k in allowed}
    if msg.get("role") == "assistant" and msg.get("tool_calls") and not msg.get("content"):
        msg["content"] = None
    return msg


def _http_error(provider: str, response: httpx.Response) -> str:
    label = PROVIDERS[provider]["label"]
    try:
        body = response.json()
        err = body.get("error") if isinstance(body, dict) else None
        message = err.get("message") if isinstance(err, dict) else (err or response.text)
    except ValueError:
        message = response.text
    if response.status_code == 401:
        return f"{label} rejected the API key (401). Check the key in Models & GPUs → Cloud models."
    if response.status_code == 402:
        return f"{label}: insufficient balance (402). Add credit to your API account."
    if response.status_code == 429:
        return f"{label} rate limit or quota reached (429): {message}"
    return f"{label} error {response.status_code}: {str(message)[:400]}"


def _friendly_error(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    if name == "AuthenticationError" or status == 401:
        return "Anthropic rejected the API key (401). Check the key in Models & GPUs → Cloud models."
    if name == "PermissionDeniedError" or status == 403:
        return f"Anthropic: this key has no access to that model (403). {getattr(exc, 'message', '')}".strip()
    if name == "NotFoundError" or status == 404:
        return "Anthropic: model not found (404). Pick a model from the list."
    if name == "RateLimitError" or status == 429:
        return "Anthropic rate limit reached (429). Wait a moment and try again."
    if name in ("APIConnectionError", "APITimeoutError"):
        return f"Could not reach Anthropic: {exc}"
    message = getattr(exc, "message", None) or str(exc)
    return f"Anthropic error{f' {status}' if status else ''}: {str(message)[:400]}"
