"""Conversation CRUD routes."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from .deps import AppState, get_state
from .schemas import ConversationCreateRequest, ConversationUpdateRequest

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.get("")
def list_conversations(
    search: Optional[str] = Query(default=None),
    include_archived: bool = Query(default=False),
    state: AppState = Depends(get_state),
) -> list[dict]:
    return state.repository.list_conversations(
        search=search, include_archived=include_archived
    )


@router.post("")
def create_conversation(
    payload: ConversationCreateRequest, state: AppState = Depends(get_state)
) -> dict:
    return state.repository.create_conversation(
        title=payload.title, model_id=payload.model_id, system_prompt=payload.system_prompt
    )


@router.get("/{conversation_id}")
def get_conversation(conversation_id: str, state: AppState = Depends(get_state)) -> dict:
    conversation = state.repository.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    conversation["messages"] = state.repository.list_messages(conversation_id)
    return conversation


@router.patch("/{conversation_id}")
def update_conversation(
    conversation_id: str,
    payload: ConversationUpdateRequest,
    state: AppState = Depends(get_state),
) -> dict:
    updates = payload.model_dump(exclude_none=True)
    updated = state.repository.update_conversation(conversation_id, **updates)
    if updated is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return updated


@router.delete("/{conversation_id}")
def delete_conversation(conversation_id: str, state: AppState = Depends(get_state)) -> dict:
    state.repository.delete_conversation(conversation_id)
    return {"deleted": True, "id": conversation_id}
