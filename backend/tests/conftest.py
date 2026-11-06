"""Shared fixtures."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.agents.chat_service import ChatService
from app.agents.multi_agent import MultiAgentCoordinator
from app.agents.orchestrator import AgentLoop
from app.agents.tool_executor import ToolExecutor
from app.api.deps import AppState
from app.core.settings import SettingsStore
from app.main import create_app_with_state
from app.models.registry import ModelRegistry
from app.retrieval.service import RetrievalService
from app.security.audit import AuditLog
from app.storage.db import Database
from app.storage.repository import Repository
from app.tools.build import build_default_registry
from app.tools.registry import ToolRegistry

from .fake_adapter import FakeAdapter


@pytest.fixture
def fake_adapter() -> FakeAdapter:
    return FakeAdapter()


@pytest.fixture
def repository(tmp_path) -> Repository:
    database = Database(tmp_path / "test.db")
    database.initialize()
    return Repository(database)


@pytest.fixture
def settings_store(repository) -> SettingsStore:
    return SettingsStore(repository)


@pytest.fixture
def registry(repository, settings_store, fake_adapter) -> ModelRegistry:
    return ModelRegistry(repository, settings_store, adapter=fake_adapter)


@pytest.fixture
def chat_service(repository, registry, settings_store) -> ChatService:
    return ChatService(repository, registry, settings_store)


@pytest.fixture
def tool_registry() -> ToolRegistry:
    return build_default_registry()


@pytest.fixture
def audit(repository) -> AuditLog:
    return AuditLog(repository)


@pytest.fixture
def tool_executor(tool_registry, repository, audit, settings_store) -> ToolExecutor:
    return ToolExecutor(tool_registry, repository, audit, settings_store)


@pytest.fixture
def agent_loop(registry, tool_registry, tool_executor, repository, settings_store) -> AgentLoop:
    return AgentLoop(registry, tool_registry, tool_executor, repository, settings_store)


@pytest.fixture
def retrieval(repository) -> RetrievalService:
    return RetrievalService(repository)


@pytest.fixture
def multi_agent(registry, settings_store) -> MultiAgentCoordinator:
    return MultiAgentCoordinator(registry, settings_store)


@pytest.fixture
def client(repository, settings_store, registry, chat_service, tool_executor, agent_loop, retrieval, multi_agent) -> TestClient:
    state = AppState(repository, settings_store, registry, chat_service, tool_executor, agent_loop, retrieval, multi_agent)
    app = create_app_with_state(state)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def approved_workspace(client, tmp_path) -> str:
    """Configure an approved workspace and return its path."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    settings = client.get("/api/settings").json()
    settings["approved_workspaces"] = [str(workspace)]
    client.put("/api/settings", json={"settings": settings})
    return str(workspace)
