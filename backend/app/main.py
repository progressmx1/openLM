"""FastAPI application factory for the local AI desktop backend.

Run directly with::

    python -m app.main

or via uvicorn::

    uvicorn app.main:app --host 127.0.0.1 --port 8765
"""
from __future__ import annotations

import contextlib
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .agents.chat_service import ChatService
from .agents.multi_agent import MultiAgentCoordinator
from .agents.orchestrator import AgentLoop
from .agents.tool_executor import ToolExecutor
from .api import (
    routes_agents,
    routes_chat,
    routes_coding_agent,
    routes_conversations,
    routes_documents,
    routes_models,
    routes_providers,
    routes_rag,
    routes_system,
    routes_tools,
)
from .api.deps import AppState
from .core.logging import configure, get_logger
from .agents.coding.prompts import PromptLibrary
from .agents.coding.service import AgentService
from .agents.coding.store import AgentStore
from .core.paths import DATA_DIR, DB_PATH, LOGS_DIR, ensure_runtime_dirs
from .core.secrets import SecretStore
from .core.settings import SettingsStore
from .models.adapters.base import ModelAdapter
from .models.adapters.cloud import PREFIX as CLOUD_PREFIX, CloudAdapter
from .models.adapters.hybrid import HybridAdapter
from .models.adapters.router import ProviderRouter
from .models.adapters.llama_server_adapter import LlamaServerAdapter
from .models.devices import DeviceCatalog, find_server_binaries
from .models.gpu_telemetry import GpuTelemetry
from .models.registry import ModelRegistry
from .retrieval.service import RetrievalService
from .security.audit import AuditLog
from .storage.db import Database
from .storage.repository import Repository
from .tools.build import build_default_registry
from .tools.registry import ToolRegistry

log = get_logger(__name__)


def build_state(
    adapter: ModelAdapter | None = None,
    db_path: Path | None = None,
    tool_registry: ToolRegistry | None = None,
) -> AppState:
    """Construct the shared application state (database + services)."""
    ensure_runtime_dirs()
    database = Database(db_path or DB_PATH)
    database.initialize()
    repository = Repository(database)
    settings_store = SettingsStore(repository)
    if adapter is None:
        adapter = build_default_adapter(settings_store)
    secrets = SecretStore(Path(db_path or DB_PATH).parent / "secrets.bin")
    adapter = with_cloud(adapter, repository, settings_store, secrets)
    registry = ModelRegistry(repository, settings_store, adapter=adapter)
    chat_service = ChatService(repository, registry, settings_store)
    audit = AuditLog(repository)
    tools = tool_registry or build_default_registry()
    tool_executor = ToolExecutor(tools, repository, audit, settings_store)
    agent_loop = AgentLoop(registry, tools, tool_executor, repository, settings_store)
    retrieval = RetrievalService(repository)
    multi_agent = MultiAgentCoordinator(registry, settings_store)
    return AppState(
        repository, settings_store, registry, chat_service, tool_executor, agent_loop, retrieval, multi_agent
    )


def with_cloud(local: ModelAdapter, repository: Repository, settings_store: SettingsStore, secrets: SecretStore) -> ProviderRouter:
    """Cloud models (``cloud://`` paths) go to the API adapter, the rest to ``local``."""

    def path_of(model_id: str) -> str | None:
        record = repository.get_model(model_id)
        return record["path"] if record else None

    cloud = CloudAdapter(secrets, path_of, base_urls=lambda: settings_store.load().provider_base_urls)
    return ProviderRouter(local, cloud, lambda model_id: (path_of(model_id) or "").startswith(CLOUD_PREFIX))


def build_default_adapter(settings_store: SettingsStore) -> ModelAdapter:
    """llama-server (multi-GPU, process per model) with in-process fallback."""
    catalog = DeviceCatalog(
        lambda: find_server_binaries(settings_store.load().llama_server_paths)
    )
    server = LlamaServerAdapter(
        catalog,
        LOGS_DIR,
        load_timeout_provider=lambda: settings_store.load().load_timeout_seconds,
        telemetry=GpuTelemetry(),
    )
    return HybridAdapter(server, mode_provider=lambda: settings_store.load().runtime)


def build_agent_service(state: AppState) -> AgentService:
    return AgentService(
        AgentStore(state.repository.db),
        state.registry,
        state.settings_store,
        state.tool_executor.audit,
        PromptLibrary(DATA_DIR / "prompts"),
    )


def create_app_with_state(state: AppState) -> FastAPI:
    if state.agent_service is None:
        state.agent_service = build_agent_service(state)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        for session in state.agent_service.list_sessions():
            state.agent_service.interrupt(session["id"])
        # Release model memory on shutdown.
        state.registry.adapter.unload_all()

    app = FastAPI(title="Local AI Desktop", version="0.1.0", lifespan=lifespan)
    app.state.app_state = state

    # The desktop UI may be served from a Vite dev server during development.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(routes_system.router)
    app.include_router(routes_models.router)
    app.include_router(routes_providers.router)
    app.include_router(routes_conversations.router)
    app.include_router(routes_chat.router)
    app.include_router(routes_tools.router)
    app.include_router(routes_documents.router)
    app.include_router(routes_rag.router)
    app.include_router(routes_agents.router)
    app.include_router(routes_coding_agent.router)
    app.include_router(routes_coding_agent.workspace_router)

    _mount_frontend_if_built(app)

    return app


def create_app() -> FastAPI:
    configure()
    return create_app_with_state(build_state())


def _mount_frontend_if_built(app: FastAPI) -> None:
    """Serve the built React app when present (convenience, not the primary shell)."""
    # backend/app/main.py -> project root is two levels up.
    dist = Path(__file__).resolve().parents[2] / "desktop" / "frontend" / "dist"
    if not (dist / "index.html").exists():
        return

    from fastapi.staticfiles import StaticFiles

    # Mounted last, so every /api route wins; serves index.html, /assets/* and the
    # other files in dist/ (e.g. theme-init.js), all from disk — nothing from a CDN.
    app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8765, reload=False)
