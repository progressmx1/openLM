"""Tests for SQLite persistence (conversations, messages, settings)."""
from __future__ import annotations

from app.core.settings import AppSettings, SettingsStore
from app.storage.repository import Repository


def test_conversation_crud(repository: Repository) -> None:
    conv = repository.create_conversation(title="Hello")
    assert conv["id"]
    assert conv["title"] == "Hello"

    got = repository.get_conversation(conv["id"])
    assert got is not None
    assert got["title"] == "Hello"

    repository.update_conversation(conv["id"], title="Renamed", pinned=True)
    got = repository.get_conversation(conv["id"])
    assert got["title"] == "Renamed"
    assert got["pinned"] == 1

    repository.delete_conversation(conv["id"])
    assert repository.get_conversation(conv["id"]) is None


def test_message_persistence_and_ordering(repository: Repository) -> None:
    conv = repository.create_conversation()
    repository.add_message(conv["id"], "user", "first")
    repository.add_message(conv["id"], "assistant", "second")
    repository.add_message(conv["id"], "user", "third")

    messages = repository.list_messages(conv["id"])
    assert [m["content"] for m in messages] == ["first", "second", "third"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]


def test_cascade_delete_messages(repository: Repository) -> None:
    conv = repository.create_conversation()
    repository.add_message(conv["id"], "user", "hi")
    repository.delete_conversation(conv["id"])
    assert repository.list_messages(conv["id"]) == []


def test_settings_roundtrip(repository: Repository) -> None:
    store = SettingsStore(repository)
    settings = store.load()
    assert isinstance(settings, AppSettings)

    settings.theme = "light"
    settings.inference.n_ctx = 4096
    store.save(settings)

    reloaded = SettingsStore(repository).load()
    assert reloaded.theme == "light"
    assert reloaded.inference.n_ctx == 4096


def test_settings_defaults_on_missing(repository: Repository) -> None:
    store = SettingsStore(repository)
    settings = store.load()
    assert settings.inference.n_ctx == 8192
    assert settings.inference.n_gpu_layers == -1  # automatic GPU offload
    assert settings.runtime == "auto"
    assert settings.model_profiles == {}
    assert settings.internet_enabled is False
