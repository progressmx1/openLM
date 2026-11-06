"""Routes each model to the right runtime.

* ``llama-server`` (preferred): one process per model, per-GPU placement.
* ``llama-cpp-python`` (fallback): in-process, single device; used only when no
  llama-server build is installed or the user forces it in Settings.

The choice is made per load, so installing a llama-server build takes effect on
the next load without restarting the app.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Iterable, Iterator, Optional

from .base import ModelAdapter, ModelLoadError
from .llama_server_adapter import LlamaServerAdapter


class HybridAdapter(ModelAdapter):
    runtime_name = "auto"

    def __init__(
        self,
        server: LlamaServerAdapter,
        inprocess_factory: Optional[Callable[[], ModelAdapter]] = None,
        mode_provider: Optional[Callable[[], str]] = None,
    ) -> None:
        self.server = server
        self._inprocess_factory = inprocess_factory
        self._inprocess: Optional[ModelAdapter] = None
        self._mode = mode_provider or (lambda: "auto")
        self._owner: dict[str, ModelAdapter] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------- routing
    def _get_inprocess(self) -> ModelAdapter:
        with self._lock:
            if self._inprocess is None:
                if self._inprocess_factory is None:
                    from .llama_cpp_adapter import LlamaCppAdapter

                    self._inprocess = LlamaCppAdapter()
                else:
                    self._inprocess = self._inprocess_factory()
            return self._inprocess

    def _choose(self) -> ModelAdapter:
        mode = (self._mode() or "auto").lower()
        if mode == "llama-cpp-python":
            return self._get_inprocess()
        if mode == "llama-server" or self.server.available():
            return self.server
        return self._get_inprocess()

    def _for(self, model_id: str) -> Optional[ModelAdapter]:
        with self._lock:
            return self._owner.get(model_id)

    # ----------------------------------------------------------- interface
    def discover_models(self, locations: Iterable[str]) -> list[dict[str, Any]]:
        return self.server.discover_models(locations)

    def plan_load(self, model_id: str, settings: dict[str, Any]) -> dict[str, Any]:
        owner = self._for(model_id)
        if owner is not None and owner.is_loaded(model_id):
            return owner.plan_load(model_id, settings)
        return self._choose().plan_load(model_id, settings)

    def load_model(self, model_id: str, settings: dict[str, Any]) -> dict[str, Any]:
        previous = self._for(model_id)
        target = self._choose()
        if previous is not None and previous is not target:
            previous.unload_model(model_id)
        try:
            status = target.load_model(model_id, settings)
        except ModelLoadError:
            with self._lock:
                self._owner.pop(model_id, None)
            raise
        with self._lock:
            self._owner[model_id] = target
        return status

    def unload_model(self, model_id: str) -> None:
        owner = self._for(model_id)
        if owner is not None:
            owner.unload_model(model_id)
        with self._lock:
            self._owner.pop(model_id, None)

    def unload_all(self) -> None:
        self.server.unload_all()
        if self._inprocess is not None:
            self._inprocess.unload_all()
        with self._lock:
            self._owner.clear()

    def chat(self, model_id: str, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        owner = self._for(model_id) or self.server
        return owner.chat(model_id, messages, options)

    def stream_chat(self, model_id: str, messages: list[dict[str, Any]], options: dict[str, Any]) -> Iterator[str]:
        owner = self._for(model_id) or self.server
        return owner.stream_chat(model_id, messages, options)

    def stream_events(self, model_id: str, messages: list[dict[str, Any]], options: dict[str, Any]) -> Iterator[dict[str, Any]]:
        owner = self._for(model_id) or self.server
        return owner.stream_events(model_id, messages, options)

    def supports_native_tools(self, model_id: str) -> bool:
        owner = self._for(model_id)
        return owner is not None and owner.supports_native_tools(model_id)

    def process_ids(self) -> dict[int, str]:
        return self.server.process_ids()

    def get_model_status(self, model_id: str) -> dict[str, Any]:
        owner = self._for(model_id)
        if owner is None:
            return {"loaded": False}
        return owner.get_model_status(model_id)

    def is_loaded(self, model_id: str) -> bool:
        owner = self._for(model_id)
        return owner is not None and owner.is_loaded(model_id)

    def list_devices(self, refresh: bool = False) -> list[dict[str, Any]]:
        return self.server.list_devices(refresh=refresh) if self.server.available() else []

    def get_runtime_info(self) -> dict[str, Any]:
        server_info = self.server.get_runtime_info()
        inprocess_info = self._inprocess.get_runtime_info() if self._inprocess is not None else None
        with self._lock:
            owned = dict(self._owner)
        loaded = [m for m, a in owned.items() if a.is_loaded(m)]
        active = self._choose_name()
        gpu = server_info.get("gpu_offload_supported") if active == "llama-server" else (
            (inprocess_info or {}).get("gpu_offload_supported")
        )
        return {
            "name": active,
            "mode": self._mode(),
            "version": server_info.get("version"),
            "gpu_offload_supported": gpu,
            "max_devices": server_info.get("max_devices"),
            "llama_server": server_info,
            "llama_cpp_python": inprocess_info,
            "loaded_models": loaded,
        }

    def _choose_name(self) -> str:
        mode = (self._mode() or "auto").lower()
        if mode == "llama-cpp-python":
            return "llama-cpp-python"
        if mode == "llama-server" or self.server.available():
            return "llama-server"
        return "llama-cpp-python"
