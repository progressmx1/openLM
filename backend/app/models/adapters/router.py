"""Sends each model to its runtime: cloud models (``cloud://…`` paths) to the
:class:`~.cloud.CloudAdapter`, everything else to the local adapter (llama-server / in-process).
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator

from .base import ModelAdapter
from .cloud import CloudAdapter


class ProviderRouter(ModelAdapter):
    runtime_name = "auto"

    def __init__(self, local: ModelAdapter, cloud: CloudAdapter, is_cloud: Callable[[str], bool]) -> None:
        self.local = local
        self.cloud = cloud
        self._is_cloud = is_cloud

    def _for(self, model_id: str) -> ModelAdapter:
        return self.cloud if self._is_cloud(model_id) else self.local

    def discover_models(self, locations: Iterable[str]) -> list[dict[str, Any]]:
        return self.local.discover_models(locations)

    def plan_load(self, model_id: str, settings: dict[str, Any]) -> dict[str, Any]:
        return self._for(model_id).plan_load(model_id, settings)

    def load_model(self, model_id: str, settings: dict[str, Any]) -> dict[str, Any]:
        return self._for(model_id).load_model(model_id, settings)

    def unload_model(self, model_id: str) -> None:
        self._for(model_id).unload_model(model_id)

    def unload_all(self) -> None:
        self.local.unload_all()

    def chat(self, model_id: str, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        return self._for(model_id).chat(model_id, messages, options)

    def stream_chat(self, model_id: str, messages: list[dict[str, Any]], options: dict[str, Any]) -> Iterator[str]:
        return self._for(model_id).stream_chat(model_id, messages, options)

    def stream_events(self, model_id: str, messages: list[dict[str, Any]], options: dict[str, Any]) -> Iterator[dict[str, Any]]:
        return self._for(model_id).stream_events(model_id, messages, options)

    def supports_native_tools(self, model_id: str) -> bool:
        return self._for(model_id).supports_native_tools(model_id)

    def process_ids(self) -> dict[int, str]:
        return self.local.process_ids()

    def get_model_status(self, model_id: str) -> dict[str, Any]:
        return self._for(model_id).get_model_status(model_id)

    def is_loaded(self, model_id: str) -> bool:
        return self._for(model_id).is_loaded(model_id)

    def list_devices(self, refresh: bool = False) -> list[dict[str, Any]]:
        return self.local.list_devices(refresh=refresh)

    def get_runtime_info(self) -> dict[str, Any]:
        # Cloud models hold no local memory, so they are not in ``loaded_models``.
        return {**self.local.get_runtime_info(), "cloud": self.cloud.get_runtime_info()}

    def __getattr__(self, name: str) -> Any:
        # Adapter-specific extras (e.g. llama-server's runtime helpers) stay reachable.
        return getattr(self.local, name)
