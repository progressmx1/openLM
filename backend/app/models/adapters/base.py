"""Runtime adapter interface.

The UI and orchestration layers depend only on this interface, never on
``llama-cpp-python`` internals. New runtimes (Ollama, OpenAI-compatible local
endpoints) are added by implementing the same surface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Iterator


class ModelLoadError(RuntimeError):
    """Raised when a model cannot be loaded; message is user-facing."""


class ModelNotLoadedError(RuntimeError):
    """Raised when a generation is requested for a model that is not loaded."""


class ModelAdapter(ABC):
    runtime_name: str = "base"

    @abstractmethod
    def discover_models(self, locations: Iterable[str]) -> list[dict[str, Any]]:
        """Scan folders for supported model files and return metadata dicts."""

    @abstractmethod
    def load_model(self, model_id: str, settings: dict[str, Any]) -> dict[str, Any]:
        """Load a model into memory and return its runtime status."""

    @abstractmethod
    def unload_model(self, model_id: str) -> None:
        """Release a model from memory."""

    @abstractmethod
    def chat(self, model_id: str, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        """Run a full (non-streaming) completion."""

    @abstractmethod
    def stream_chat(
        self, model_id: str, messages: list[dict[str, Any]], options: dict[str, Any]
    ) -> Iterator[str]:
        """Yield completion text deltas."""

    @abstractmethod
    def get_model_status(self, model_id: str) -> dict[str, Any]:
        """Return runtime status for a single model."""

    @abstractmethod
    def get_runtime_info(self) -> dict[str, Any]:
        """Return adapter/runtime capabilities and versions."""

    # Concrete helpers derived from the interface above.
    def plan_load(self, model_id: str, settings: dict[str, Any]) -> dict[str, Any]:
        """Decide where a model will run before loading it.

        Returns at least ``physical_devices``: the GPUs (or ``"cpu"``) the model will
        occupy. The registry uses it to enforce one model per GPU. Adapters without
        device control report a single shared ``"default"`` device.
        """
        return {"physical_devices": ["default"]}

    def list_devices(self, refresh: bool = False) -> list[dict[str, Any]]:
        """GPUs this adapter can place models on (empty when it has no device control)."""
        return []

    def stream_events(
        self, model_id: str, messages: list[dict[str, Any]], options: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        """Structured stream (content / reasoning / tool_call / done events).

        Runtimes without native tool calling only produce ``content`` events; the agent
        then falls back to its text tool-call protocol.
        """
        for text in self.stream_chat(model_id, messages, options):
            yield {"type": "content", "text": text}
        yield {"type": "done", "finish_reason": "stop", "usage": {}, "timings": None}

    def supports_native_tools(self, model_id: str) -> bool:
        return False

    def process_ids(self) -> dict[int, str]:
        """OS process id -> model id for models running in child processes."""
        return {}

    def is_loaded(self, model_id: str) -> bool:
        return bool(self.get_model_status(model_id).get("loaded"))

    def unload_all(self) -> None:
        for model_id in list(self.get_runtime_info().get("loaded_models", [])):
            self.unload_model(model_id)
