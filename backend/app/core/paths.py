"""Filesystem paths and runtime-directory helpers.

Paths are configurable through environment variables so the application never
hard-codes a user's absolute paths:

- ``LOCAL_AI_DATA_DIR``  -> runtime data (SQLite DB, settings, logs)
- ``LOCAL_AI_MODELS_DIR`` -> default model directory (discovery root)
"""
from __future__ import annotations

import os
from pathlib import Path

# backend/app/core/paths.py -> offline_models/ is four levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default.resolve()


DATA_DIR = _env_path("LOCAL_AI_DATA_DIR", PROJECT_ROOT / "data")
MODELS_DIR = _env_path("LOCAL_AI_MODELS_DIR", PROJECT_ROOT / "models")

DB_PATH = DATA_DIR / "local_ai.db"
SETTINGS_PATH = DATA_DIR / "settings.json"
LOGS_DIR = DATA_DIR / "logs"
DOCUMENTS_DIR = DATA_DIR / "documents"


def get_documents_dir() -> Path:
    """Return the documents directory, honoring ``LOCAL_AI_DOCUMENTS_DIR``.

    A function (not a module constant) so tests can redirect it to a temp dir.
    """
    raw = os.environ.get("LOCAL_AI_DOCUMENTS_DIR")
    directory = Path(raw).expanduser().resolve() if raw else DOCUMENTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_runtime_dirs() -> None:
    """Create runtime directories if they do not exist."""
    for directory in (DATA_DIR, MODELS_DIR, LOGS_DIR, DOCUMENTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
