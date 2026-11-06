"""Chat streaming and control routes."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from .deps import AppState, get_state
from .schemas import AgentRunRequest, ChatStopRequest, ChatStreamRequest

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/stream")
async def stream_chat(payload: ChatStreamRequest, state: AppState = Depends(get_state)) -> StreamingResponse:
    conversation = state.repository.get_conversation(payload.conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if state.registry.get_model(payload.model_id) is None:
        raise HTTPException(status_code=404, detail="Model not found")

    async def event_source():
        async for event in state.chat_service.stream(
            payload.conversation_id,
            payload.model_id,
            payload.content,
            payload.options,
            payload.context,
        ):
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post("/stop")
def stop_chat(payload: ChatStopRequest, state: AppState = Depends(get_state)) -> dict:
    stopped = state.chat_service.stop(payload.task_id)
    if not stopped:
        raise HTTPException(status_code=404, detail="No such active generation")
    return {"stopped": True, "task_id": payload.task_id}


@router.post("/agent")
def run_agent(payload: AgentRunRequest, state: AppState = Depends(get_state)) -> dict:
    if state.repository.get_conversation(payload.conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if state.registry.get_model(payload.model_id) is None:
        raise HTTPException(status_code=404, detail="Model not found")
    # Synchronous endpoint: FastAPI runs it in the threadpool so the loop never
    # blocks the event loop.
    return state.agent_loop.run(
        payload.conversation_id,
        payload.model_id,
        payload.content,
        payload.context,
        payload.max_steps,
        payload.options,
    )
