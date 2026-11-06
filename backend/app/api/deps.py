"""FastAPI dependencies: access to shared application state."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import Request

from ..agents.chat_service import ChatService
from ..agents.multi_agent import MultiAgentCoordinator
from ..agents.orchestrator import AgentLoop
from ..agents.tool_executor import ToolExecutor
from ..core.settings import SettingsStore
from ..models.registry import ModelRegistry
from ..retrieval.service import RetrievalService
from ..storage.repository import Repository


class AppState:
    def __init__(
        self,
        repository: Repository,
        settings_store: SettingsStore,
        registry: ModelRegistry,
        chat_service: ChatService,
        tool_executor: ToolExecutor,
        agent_loop: AgentLoop,
        retrieval: RetrievalService,
        multi_agent: MultiAgentCoordinator,
        agent_service: Optional[Any] = None,
    ) -> None:
        self.repository = repository
        self.settings_store = settings_store
        self.registry = registry
        self.chat_service = chat_service
        self.tool_executor = tool_executor
        self.agent_loop = agent_loop
        self.retrieval = retrieval
        self.multi_agent = multi_agent
        # Coding-agent service; built from the other services when not supplied.
        self.agent_service = agent_service


def get_state(request: Request) -> AppState:
    return request.app.state.app_state
