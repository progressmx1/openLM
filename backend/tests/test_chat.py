"""Tests for the chat service streaming bridge."""
from __future__ import annotations

import asyncio
import time
from typing import Any, Iterator

from app.agents.chat_service import ChatService
from app.core.settings import SettingsStore
from app.models.registry import ModelRegistry
from app.storage.repository import Repository

from .fake_adapter import FakeAdapter


class SlowAdapter(FakeAdapter):
    def stream_chat(self, model_id: str, messages, options) -> Iterator[str]:
        for i in range(5):
            time.sleep(0.02)
            yield f"tok{i}"


def _run(coro):
    return asyncio.run(coro)


def _setup(repository, settings_store, adapter):
    registry = ModelRegistry(repository, settings_store, adapter=adapter)
    chat_service = ChatService(repository, registry, settings_store)
    registry.discover(["x"])
    model_id = registry.list_models()[0]["id"]
    registry.load(model_id)
    conversation = repository.create_conversation(title="t")
    return chat_service, registry, conversation["id"], model_id


def test_stream_persists_messages(repository, settings_store, fake_adapter) -> None:
    chat_service, _registry, conv_id, model_id = _setup(repository, settings_store, fake_adapter)

    async def consume():
        events = []
        async for event in chat_service.stream(conv_id, model_id, "hi"):
            events.append(event)
        return events

    events = _run(consume())
    types = [e["type"] for e in events]
    assert "start" in types and "delta" in types and "done" in types
    text = "".join(e["text"] for e in events if e["type"] == "delta")
    assert text == "Hello!"

    messages = repository.list_messages(conv_id)
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["content"] == "Hello!"


def test_stream_error_returns_error_event(repository, settings_store, fake_adapter) -> None:
    fake_adapter.fail_generate = True
    chat_service, _registry, conv_id, model_id = _setup(repository, settings_store, fake_adapter)

    async def consume():
        return [e async for e in chat_service.stream(conv_id, model_id, "hi")]

    events = _run(consume())
    assert any(e["type"] == "error" for e in events)
    # No assistant message persisted when generation failed with no output.
    messages = repository.list_messages(conv_id)
    assert [m["role"] for m in messages] == ["user"]


def test_stream_unloaded_model_errors(repository, settings_store, fake_adapter) -> None:
    registry = ModelRegistry(repository, settings_store, adapter=fake_adapter)
    chat_service = ChatService(repository, registry, settings_store)
    registry.discover(["x"])
    model_id = registry.list_models()[0]["id"]
    conv_id = repository.create_conversation()["id"]

    async def consume():
        return [e async for e in chat_service.stream(conv_id, model_id, "hi")]

    events = _run(consume())
    assert any(e["type"] == "error" for e in events)


def test_stop_halts_generation(repository, settings_store) -> None:
    slow = SlowAdapter()
    chat_service, _registry, conv_id, model_id = _setup(repository, settings_store, slow)

    async def consume():
        events = []
        async for event in chat_service.stream(conv_id, model_id, "hi"):
            events.append(event)
            if event["type"] == "start":
                chat_service.stop(event["task_id"])
        return events

    events = _run(consume())
    done = [e for e in events if e["type"] == "done"]
    assert done and done[0]["stopped"] is True


def test_stop_unknown_task(repository, settings_store, fake_adapter) -> None:
    registry = ModelRegistry(repository, settings_store, adapter=fake_adapter)
    chat_service = ChatService(repository, registry, settings_store)
    assert chat_service.stop("nope") is False
