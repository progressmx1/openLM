"""Application settings: typed model, defaults, and SQLite persistence.

Settings are stored in the SQLite ``settings`` table (never in source or committed
files). Secrets such as external-provider API keys are stored separately through
the OS credential store (Milestone 4); this file only holds non-secret configuration.
"""
from __future__ import annotations

import threading
from typing import Any, Optional

from pydantic import BaseModel, Field


class InferenceDefaults(BaseModel):
    # Context length. With llama-server, 0 = the largest context that fits in VRAM.
    n_ctx: int = 8192
    # llama-server: -1 or 0 = ALL layers on the chosen GPU(s); N > 0 = cap at N layers.
    # 0 never means "CPU": choose the CPU GPU-mode to keep a model off the GPU.
    # In-process llama-cpp-python fallback (CPU-only wheel): ignored.
    n_gpu_layers: int = -1
    # Weight loading: "auto" = llama.cpp default (mmap); or "mmap", "none", "mlock".
    load_mode: str = "auto"
    n_threads: Optional[int] = None
    n_batch: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int = 4096
    seed: int = -1
    stop: list[str] = Field(default_factory=list)
    chat_template: Optional[str] = None


class AppSettings(BaseModel):
    theme: str = "dark"  # "dark" | "light" | "system"
    default_model_id: Optional[str] = None
    model_directories: list[str] = Field(default_factory=list)
    inference: InferenceDefaults = Field(default_factory=InferenceDefaults)
    max_history_messages: int = 40
    internet_enabled: bool = False
    # Models on *different* GPUs always run side by side. This allows more than one
    # model on the *same* GPU (or several CPU models), which risks running out of memory.
    allow_multi_model: bool = False
    # Model runtime: "auto" (llama-server when installed, else llama-cpp-python),
    # "llama-server", or "llama-cpp-python".
    runtime: str = "auto"
    # Extra llama-server executables or folders (runtimes/llama.cpp/* is always scanned).
    llama_server_paths: list[str] = Field(default_factory=list)
    # Seconds to wait for a model to finish loading (eGPU loads can be slow).
    load_timeout_seconds: int = 600
    # Per-model remembered load choices: {model_id: {"placement": ..., "n_ctx": ...}}.
    model_profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # Approved workspace directories (Milestone 2 security boundary).
    approved_workspaces: list[str] = Field(default_factory=list)
    # Cloud providers: endpoint overrides ({"custom": "https://…/v1"}). API keys are not stored
    # here but encrypted in data/secrets.bin.
    provider_base_urls: dict[str, str] = Field(default_factory=dict)


class SettingsStore:
    """Thread-safe load/save of :class:`AppSettings` backed by SQLite."""

    _KEY = "app_settings"

    def __init__(self, repository: Any) -> None:
        self._repo = repository
        self._lock = threading.Lock()
        self._settings: AppSettings | None = None

    def load(self) -> AppSettings:
        with self._lock:
            if self._settings is not None:
                return self._settings
            raw = self._repo.get_setting(self._KEY)
            if isinstance(raw, dict):
                try:
                    self._settings = AppSettings.model_validate(raw)
                except Exception:
                    self._settings = AppSettings()
            else:
                self._settings = AppSettings()
                self._repo.set_setting(self._KEY, self._settings.model_dump())
            return self._settings

    def save(self, settings: AppSettings) -> AppSettings:
        with self._lock:
            self._repo.set_setting(self._KEY, settings.model_dump())
            self._settings = settings
            return self._settings
