"""Tests for model discovery, registration, and lifecycle."""
from __future__ import annotations

import struct

from app.core.settings import SettingsStore
from app.models.registry import ModelRegistry
from app.storage.repository import Repository

from .fake_adapter import FakeAdapter
from .test_gguf import make_gguf, _string


def test_discover_registers_models(registry: ModelRegistry) -> None:
    result = registry.discover(["anywhere"])
    assert result["found"] == 1
    assert result["added"] == 1

    models = registry.list_models()
    assert len(models) == 1
    assert models[0]["format"] == "gguf"
    assert models[0]["loaded"] is False


def test_discover_is_idempotent(registry: ModelRegistry) -> None:
    registry.discover(["a"])
    result = registry.discover(["a"])
    assert result["added"] == 0
    assert result["updated"] == 1
    assert len(registry.list_models()) == 1


def test_unregister_does_not_delete_file(registry: ModelRegistry) -> None:
    registry.discover(["a"])
    model_id = registry.list_models()[0]["id"]
    registry.unregister(model_id)
    assert registry.get_model(model_id) is None
    assert registry.list_models() == []


def test_load_unload_and_status(registry: ModelRegistry) -> None:
    registry.discover(["a"])
    model_id = registry.list_models()[0]["id"]

    assert registry.status(model_id)["loaded"] is False
    status = registry.load(model_id, {"n_ctx": 4096, "n_gpu_layers": 0})
    assert status["loaded"] is True
    assert status["n_ctx"] == 4096

    registry.unload(model_id)
    assert registry.status(model_id)["loaded"] is False


def test_single_model_enforcement(registry: ModelRegistry, fake_adapter: FakeAdapter) -> None:
    fake_adapter.discovered = [
        {"name": "a", "path": "/models/a.gguf", "format": "gguf"},
        {"name": "b", "path": "/models/b.gguf", "format": "gguf"},
    ]
    registry.discover(["x"])
    models = registry.list_models()
    first, second = models[0]["id"], models[1]["id"]

    registry.load(first)
    registry.load(second)

    assert fake_adapter.is_loaded(second) is True
    assert fake_adapter.is_loaded(first) is False  # auto-unloaded


def test_load_unknown_model_raises(registry: ModelRegistry) -> None:
    import pytest

    with pytest.raises(KeyError):
        registry.load("does-not-exist")


def test_register_file_with_real_gguf(repository: Repository, settings_store: SettingsStore, tmp_path) -> None:
    import tempfile

    gguf = make_gguf(
        [
            ("general.name", 8, _string("Real Model")),
            ("general.architecture", 8, _string("llama")),
            ("general.file_type", 4, struct.pack("<I", 2)),
        ]
    )
    path = tmp_path / "real.gguf"
    path.write_bytes(gguf)

    registry = ModelRegistry(repository, settings_store, adapter=FakeAdapter())
    model = registry.register_file(str(path))
    assert model["name"] == "Real Model"
    assert model["quant"] == "Q4_0"  # file_type 2


def test_register_file_rejects_invalid(repository: Repository, settings_store: SettingsStore, tmp_path) -> None:
    import pytest

    path = tmp_path / "not.gguf"
    path.write_bytes(b"junk")
    registry = ModelRegistry(repository, settings_store, adapter=FakeAdapter())
    with pytest.raises(ValueError):
        registry.register_file(str(path))
