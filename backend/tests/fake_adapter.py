"""In-memory adapter used by tests so they never load a real model."""
from __future__ import annotations

from typing import Any, Iterator

from app.models.adapters.base import ModelAdapter, ModelLoadError


class FakeAdapter(ModelAdapter):
    runtime_name = "fake"

    def __init__(self) -> None:
        self.loaded: dict[str, dict[str, Any]] = {}
        self.discovered: list[dict[str, Any]] = [
            {
                "name": "fake-model",
                "path": "/models/fake.gguf",
                "format": "gguf",
                "quant": "Q4_0",
                "size_bytes": 123,
                "architecture": "llama",
                "file_type": 2,
                "chat_template": None,
                "description": None,
            }
        ]
        self.last_options: dict[str, Any] = {}
        self.fail_load = False
        self.fail_generate = False

    def discover_models(self, locations) -> list[dict[str, Any]]:
        return self.discovered

    def load_model(self, model_id: str, settings: dict[str, Any]) -> dict[str, Any]:
        if self.fail_load:
            raise ModelLoadError("mock load failure")
        self.loaded[model_id] = settings
        return {"loaded": True, "n_ctx": settings.get("n_ctx")}

    def unload_model(self, model_id: str) -> None:
        self.loaded.pop(model_id, None)

    def chat(self, model_id: str, messages, options) -> dict[str, Any]:
        if self.fail_generate:
            raise RuntimeError("mock generation failure")
        self.last_options = options
        return {
            "content": "Hello!",
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }

    def stream_chat(self, model_id: str, messages, options) -> Iterator[str]:
        if self.fail_generate:
            raise RuntimeError("mock generation failure")
        self.last_options = options
        yield "Hel"
        yield "lo"
        yield "!"

    def get_model_status(self, model_id: str) -> dict[str, Any]:
        return {"loaded": model_id in self.loaded, **(self.loaded.get(model_id) or {})}

    def get_runtime_info(self) -> dict[str, Any]:
        return {
            "name": "fake",
            "version": "1.0",
            "gpu_offload_supported": True,
            "max_devices": 1,
            "loaded_models": list(self.loaded.keys()),
        }
