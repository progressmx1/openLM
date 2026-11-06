"""In-process llama-cpp-python adapter for GGUF models (fallback runtime).

Used when no ``llama-server`` build is installed. It runs inside the backend
process and can only use the single GPU its wheel was built for, so the
llama-server adapter is preferred for multi-GPU setups. ``llama_cpp`` is imported
lazily so a missing or broken wheel never stops the backend from starting.

Inference runs on a background thread (the chat service bridges the synchronous
generator to the async HTTP layer), so the FastAPI event loop is never blocked.
A per-model lock guarantees one generation at a time per loaded model.
"""
from __future__ import annotations

import gc
import threading
from pathlib import Path
from typing import Any, Iterable, Iterator

from .base import ModelAdapter, ModelLoadError, ModelNotLoadedError
from ..gguf import discover_gguf_files


class _LoadedModel:
    def __init__(self, llm: Any, config: dict[str, Any]) -> None:
        self.llm = llm
        self.config = config
        self.lock = threading.Lock()


class LlamaCppAdapter(ModelAdapter):
    runtime_name = "llama-cpp-python"

    def __init__(self) -> None:
        self._loaded: dict[str, _LoadedModel] = {}
        self._runtime_info = self._detect_runtime_info()

    # ------------------------------------------------------------------ info
    @staticmethod
    def _detect_runtime_info() -> dict[str, Any]:
        info: dict[str, Any] = {
            "name": "llama-cpp-python",
            "version": None,
            "gpu_offload_supported": None,
            "max_devices": None,
        }
        try:
            import llama_cpp

            info["version"] = llama_cpp.__version__
        except Exception:
            pass
        try:
            import llama_cpp.llama_cpp as lc

            info["gpu_offload_supported"] = bool(lc.llama_supports_gpu_offload())
            info["max_devices"] = lc.llama_max_devices()
        except Exception:
            pass
        return info

    def get_runtime_info(self) -> dict[str, Any]:
        return {
            **self._runtime_info,
            "loaded_models": list(self._loaded.keys()),
        }

    # ------------------------------------------------------------ discovery
    def discover_models(self, locations: Iterable[str]) -> list[dict[str, Any]]:
        return discover_gguf_files(locations)

    def plan_load(self, model_id: str, settings: dict[str, Any]) -> dict[str, Any]:
        # In-process models share whatever single device the wheel targets.
        return {"physical_devices": ["inprocess"]}

    # ------------------------------------------------------------- loading
    def load_model(self, model_id: str, settings: dict[str, Any]) -> dict[str, Any]:
        if model_id in self._loaded:
            return self.get_model_status(model_id)

        path = settings.get("path")
        if not path or not Path(path).exists():
            raise ModelLoadError(f"Model file not found: {path or '<no path>'}")

        kwargs: dict[str, Any] = {
            "model_path": path,
            "n_ctx": int(settings.get("n_ctx", 8192)),
            "n_batch": int(settings.get("n_batch", 512)),
            "verbose": False,
        }

        requested_gpu = int(settings.get("n_gpu_layers", 0))
        if self._runtime_info.get("gpu_offload_supported"):
            kwargs["n_gpu_layers"] = requested_gpu
        else:
            kwargs["n_gpu_layers"] = 0

        n_threads = settings.get("n_threads")
        if n_threads:
            kwargs["n_threads"] = int(n_threads)

        chat_format = settings.get("chat_format")
        if chat_format:
            kwargs["chat_format"] = chat_format

        try:
            from llama_cpp import Llama
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                "No model runtime is available: llama-cpp-python is not installed and no "
                "llama-server build was found. Run scripts\\install-llama-runtimes.ps1."
            ) from exc
        try:
            llm = Llama(**kwargs)
        except Exception as exc:  # noqa: BLE001 - normalize all loader failures
            raise ModelLoadError(_friendly_load_error(exc)) from exc

        config = {
            "path": path,
            "n_ctx": kwargs["n_ctx"],
            "n_gpu_layers": kwargs["n_gpu_layers"],
            "n_gpu_layers_requested": requested_gpu,
            "n_threads": kwargs.get("n_threads"),
            "n_batch": kwargs["n_batch"],
            "chat_format": kwargs.get("chat_format"),
        }
        self._loaded[model_id] = _LoadedModel(llm, config)
        return self.get_model_status(model_id)

    def unload_model(self, model_id: str) -> None:
        loaded = self._loaded.pop(model_id, None)
        if loaded is None:
            return
        try:
            del loaded.llm
        finally:
            gc.collect()

    def unload_all(self) -> None:
        for model_id in list(self._loaded.keys()):
            self.unload_model(model_id)

    def is_loaded(self, model_id: str) -> bool:
        return model_id in self._loaded

    def _require_loaded(self, model_id: str) -> _LoadedModel:
        loaded = self._loaded.get(model_id)
        if loaded is None:
            raise ModelNotLoadedError(
                f"Model '{model_id}' is not loaded. Load it before generating."
            )
        return loaded

    # ---------------------------------------------------------- generation
    def chat(
        self, model_id: str, messages: list[dict[str, Any]], options: dict[str, Any]
    ) -> dict[str, Any]:
        loaded = self._require_loaded(model_id)
        kwargs = self._completion_kwargs(options, stream=False)
        with loaded.lock:
            result = loaded.llm.create_chat_completion(messages=messages, **kwargs)
        choice = result["choices"][0]
        usage = result.get("usage") or {}
        return {
            "content": choice["message"]["content"],
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
        }

    def stream_chat(
        self, model_id: str, messages: list[dict[str, Any]], options: dict[str, Any]
    ) -> Iterator[str]:
        loaded = self._require_loaded(model_id)
        kwargs = self._completion_kwargs(options, stream=True)
        with loaded.lock:
            completion = loaded.llm.create_chat_completion(messages=messages, **kwargs)
            for chunk in completion:
                delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                text = delta.get("content")
                if text:
                    yield text

    def get_model_status(self, model_id: str) -> dict[str, Any]:
        loaded = self._loaded.get(model_id)
        if loaded is None:
            return {"loaded": False}
        return {"loaded": True, "runtime": "llama-cpp-python", "physical_devices": ["inprocess"], **loaded.config}

    @staticmethod
    def _completion_kwargs(options: dict[str, Any], stream: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "temperature": float(options.get("temperature", 0.7)),
            "top_p": float(options.get("top_p", 0.95)),
            "top_k": int(options.get("top_k", 40)),
            "repeat_penalty": float(options.get("repeat_penalty", 1.1)),
            "max_tokens": int(options.get("max_tokens", 2048)),
            "stream": stream,
        }
        seed = options.get("seed", -1)
        if seed is not None and int(seed) >= 0:
            kwargs["seed"] = int(seed)
        stop = options.get("stop")
        if stop:
            kwargs["stop"] = stop
        return kwargs


def _friendly_load_error(exc: Exception) -> str:
    text = str(exc).lower()
    if "memory" in text or "out of memory" in text:
        return (
            "Not enough memory to load this model. Try a smaller quantization, "
            "reduce the context length, or close other applications."
        )
    if "gguf" in text or "magic" in text or "invalid" in text:
        return f"The file does not appear to be a valid GGUF model: {exc}"
    if "no such file" in text:
        return f"Model file not found: {exc}"
    return str(exc)
