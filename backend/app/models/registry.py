"""Model registry: discovery, registration, lifecycle, and status.

Owns the persistence of registered models and delegates loading/generation to a
:class:`~app.models.adapters.base.ModelAdapter`.

GPU rule: models placed on *different* GPUs run side by side. By default only one
model may occupy a given GPU (or the CPU); loading another model onto the same
device unloads the previous one, unless the user enables multi-model loading.

Each model remembers its last explicit load choices (device placement, context
length, GPU layers) in ``settings.model_profiles``.
"""
from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional

from ..core.settings import AppSettings, SettingsStore
from ..storage.repository import Repository
from .adapters.base import ModelAdapter

# Load options that are remembered per model when given explicitly.
PROFILE_KEYS = ("placement", "tensor_split", "n_ctx", "n_gpu_layers", "load_mode")


def clean_path(path: str) -> str:
    """Strip the quotes Explorer's "Copy as path" adds, and stray whitespace."""
    return (path or "").strip().strip('"').strip("'").strip()


class ModelRegistry:
    def __init__(
        self,
        repository: Repository,
        settings_store: SettingsStore,
        adapter: Optional[ModelAdapter] = None,
    ) -> None:
        self.repo = repository
        self.settings_store = settings_store
        if adapter is None:
            from .adapters.llama_cpp_adapter import LlamaCppAdapter

            adapter = LlamaCppAdapter()
        self.adapter = adapter
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------ queries
    def runtime_info(self) -> dict[str, Any]:
        return self.adapter.get_runtime_info()

    def list_models(self) -> list[dict[str, Any]]:
        records = self.repo.list_models()
        for record in records:
            self._decorate(record)
        return records

    def get_model(self, model_id: str) -> Optional[dict[str, Any]]:
        record = self.repo.get_model(model_id)
        if record is None:
            return None
        self._decorate(record)
        return record

    def _decorate(self, record: dict[str, Any]) -> None:
        record["metadata"] = (
            json.loads(record["metadata"]) if record.get("metadata") else None
        )
        status = self.adapter.get_model_status(record["id"])
        record["status"] = status
        record["loaded"] = bool(status.get("loaded"))
        record["profile"] = self.settings_store.load().model_profiles.get(record["id"], {})

    # --------------------------------------------------------- discovery
    def discover(self, locations: Iterable[str]) -> dict[str, Any]:
        locations = [clean_path(loc) for loc in locations if clean_path(loc)]
        discovered = self.adapter.discover_models(locations)
        added = 0
        updated = 0
        errors: list[dict[str, str]] = []
        for item in discovered:
            try:
                if self.repo.get_model_by_path(item["path"]):
                    self.repo.refresh_model(item["path"], item)
                    updated += 1
                else:
                    self.repo.insert_model(self._model_from_discovery(item))
                    added += 1
            except Exception as exc:  # noqa: BLE001
                errors.append({"path": item["path"], "error": str(exc)})
        return {
            "found": len(discovered),
            "added": added,
            "updated": updated,
            "errors": errors,
        }

    def register_file(self, path: str) -> dict[str, Any]:
        path = str(Path(clean_path(path)).resolve())
        if not Path(path).is_file():
            raise ValueError(f"File not found: {path}")
        existing = self.repo.get_model_by_path(path)
        if existing:
            return self.get_model(existing["id"])  # type: ignore[return-value]
        from .gguf import read_gguf_metadata, summarize_metadata

        metadata = read_gguf_metadata(path)
        if metadata is None:
            raise ValueError(f"Not a valid GGUF file: {path}")
        summary = summarize_metadata(Path(path), metadata)
        summary["path"] = path
        summary["format"] = "gguf"
        model = self._model_from_discovery(summary)
        self.repo.insert_model(model)
        return self.get_model(model["id"])  # type: ignore[return-value]

    def register_cloud(self, provider: str, model: str, display_name: Optional[str] = None) -> dict[str, Any]:
        """Add a paid API model (``cloud://<provider>/<model>``); nothing is downloaded."""
        from .adapters.cloud import PROVIDERS, make_path

        model = (model or "").strip()
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")
        if not model or "/" in model and provider != "custom":
            raise ValueError("Enter a model name, e.g. " + (PROVIDERS[provider]["suggested"] or ["my-model"])[0])
        path = make_path(provider, model)
        existing = self.repo.get_model_by_path(path)
        if existing:
            return self.get_model(existing["id"])  # type: ignore[return-value]
        record = {
            "id": uuid.uuid4().hex,
            "name": model,
            "display_name": (display_name or "").strip() or None,
            "description": f"{PROVIDERS[provider]['label']} API model",
            "path": path,
            "format": "cloud",
            "quant": None,
            "size_bytes": None,
            "metadata": {"provider": provider, "remote_model": model},
            "runtime": "cloud",
        }
        self.repo.insert_model(record)
        return self.get_model(record["id"])  # type: ignore[return-value]

    def unregister(self, model_id: str) -> None:
        # Unload even if it crashed, so its process and log handle are released.
        self.adapter.unload_model(model_id)
        self.repo.delete_model(model_id)
        settings = self.settings_store.load()
        if model_id in settings.model_profiles:
            profiles = {k: v for k, v in settings.model_profiles.items() if k != model_id}
            self.settings_store.save(settings.model_copy(update={"model_profiles": profiles}))

    def update_model(self, model_id: str, **fields: Any) -> Optional[dict[str, Any]]:
        allowed = {"display_name", "description"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        updated = self.repo.update_model(model_id, **updates)
        return self.get_model(model_id) if updated else None

    @staticmethod
    def _model_from_discovery(item: dict[str, Any]) -> dict[str, Any]:
        name = item.get("name") or Path(item["path"]).stem
        return {
            "id": uuid.uuid4().hex,
            "name": name,
            "display_name": None,
            "description": item.get("description"),
            "path": item["path"],
            "format": item.get("format", "gguf"),
            "quant": item.get("quant"),
            "size_bytes": item.get("size_bytes"),
            "metadata": item,
            "runtime": "llama.cpp",
        }

    # ---------------------------------------------------------- lifecycle
    def load(self, model_id: str, load_options: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        record = self.repo.get_model(model_id)
        if record is None:
            raise KeyError(f"Unknown model id: {model_id}")

        load_options = {k: v for k, v in (load_options or {}).items() if v is not None}
        remember = bool(load_options.pop("remember", True))
        settings = self.settings_store.load()
        profile = settings.model_profiles.get(model_id, {})

        inference = settings.inference.model_dump()
        inference.update({k: v for k, v in profile.items() if k in PROFILE_KEYS})
        inference.update(load_options)
        inference["path"] = record["path"]
        inference["size_bytes"] = record.get("size_bytes")

        with self._load_lock:
            plan = self.adapter.plan_load(model_id, inference)
            inference["placement_plan"] = plan
            new_devices = set(plan.get("physical_devices") or ["default"])
            if not settings.allow_multi_model:
                for other_id in self.adapter.get_runtime_info()["loaded_models"]:
                    if other_id == model_id:
                        continue
                    other = self.adapter.get_model_status(other_id)
                    if set(other.get("physical_devices") or ["default"]) & new_devices:
                        self.adapter.unload_model(other_id)
            status = self.adapter.load_model(model_id, inference)

        explicit = {k: load_options[k] for k in PROFILE_KEYS if k in load_options}
        if remember and explicit:
            self._remember(model_id, explicit, plan)
        return status

    def _remember(self, model_id: str, explicit: dict[str, Any], plan: dict[str, Any]) -> None:
        placement = plan.get("_placement")
        if "placement" in explicit and explicit["placement"] not in (None, "", "auto") and placement is not None:
            # Save device names too, so the choice survives eGPU re-plugging.
            explicit["placement"] = placement.remembered()
        settings = self.settings_store.load()
        profiles = dict(settings.model_profiles)
        merged = {**profiles.get(model_id, {}), **explicit}
        if merged.get("placement") in ("auto", None):
            merged.pop("placement", None)
        profiles[model_id] = merged
        self.settings_store.save(settings.model_copy(update={"model_profiles": profiles}))

    def devices(self, refresh: bool = False) -> list[dict[str, Any]]:
        return self.adapter.list_devices(refresh=refresh)

    def unload(self, model_id: str) -> dict[str, Any]:
        self.adapter.unload_model(model_id)
        return {"loaded": False, "id": model_id}

    def status(self, model_id: str) -> dict[str, Any]:
        return self.adapter.get_model_status(model_id)
