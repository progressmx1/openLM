"""Multi-agent workflow routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from .deps import AppState, get_state
from .schemas import AgentRunWorkflowRequest

router = APIRouter(prefix="/api/agents", tags=["agents"])


@router.post("/run")
def run_workflow(payload: AgentRunWorkflowRequest, state: AppState = Depends(get_state)) -> dict:
    result = state.multi_agent.run(
        payload.task, payload.agents, payload.workflow, payload.max_rounds, payload.options
    )
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("error", "Workflow error"))
    return result
