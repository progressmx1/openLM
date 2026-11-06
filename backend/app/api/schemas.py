"""Typed request schemas for the HTTP API."""
from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import BaseModel, Field


class ChatStreamRequest(BaseModel):
    conversation_id: str
    model_id: str
    content: str = Field(min_length=1)
    options: Optional[dict[str, Any]] = None
    context: Optional[str] = None


class ChatStopRequest(BaseModel):
    task_id: str


class AgentRunRequest(BaseModel):
    conversation_id: str
    model_id: str
    content: str = Field(min_length=1)
    context: Optional[str] = None
    options: Optional[dict[str, Any]] = None
    max_steps: int = Field(default=6, ge=1, le=20)


class ModelLoadRequest(BaseModel):
    n_ctx: Optional[int] = None
    n_gpu_layers: Optional[int] = None
    n_threads: Optional[int] = None
    n_batch: Optional[int] = None
    chat_format: Optional[str] = None
    # Where to run the model: "auto", "cpu", "split", a device key such as
    # "vulkan:Vulkan0", or {"mode": "split", "devices": [...], "tensor_split": [2, 1]}.
    placement: Optional[Union[str, dict[str, Any]]] = None
    tensor_split: Optional[list[float]] = None
    # Weight loading: "auto" (llama.cpp default = mmap), "mmap", "none", "mlock".
    load_mode: Optional[str] = None
    # Remember placement/context choices for this model (default true).
    remember: Optional[bool] = None


class ModelUpdateRequest(BaseModel):
    display_name: Optional[str] = None
    description: Optional[str] = None


class DiscoverRequest(BaseModel):
    locations: list[str] = Field(min_length=1)


class RegisterFileRequest(BaseModel):
    path: str


class ConversationCreateRequest(BaseModel):
    title: str = "New chat"
    model_id: Optional[str] = None
    system_prompt: Optional[str] = None


class ConversationUpdateRequest(BaseModel):
    title: Optional[str] = None
    model_id: Optional[str] = None
    system_prompt: Optional[str] = None
    pinned: Optional[bool] = None
    archived: Optional[bool] = None


class SettingsUpdateRequest(BaseModel):
    settings: dict[str, Any]


class ToolExecuteRequest(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class DecideRequest(BaseModel):
    approved: bool


class RagIndexRequest(BaseModel):
    folder: str


class RagSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class AgentRunWorkflowRequest(BaseModel):
    task: str = Field(min_length=1)
    agents: list[dict[str, str]]
    workflow: str = "sequential"  # sequential | review | parallel
    max_rounds: int = Field(default=1, ge=1, le=10)
    options: Optional[dict[str, Any]] = None
