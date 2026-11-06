"""Tool execution and approval routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..tools.registry import InvalidToolArgumentsError, UnknownToolError
from .deps import AppState, get_state
from .schemas import DecideRequest, ToolExecuteRequest

router = APIRouter(prefix="/api", tags=["tools"])


@router.get("/tools")
def list_tools(state: AppState = Depends(get_state)) -> list[dict]:
    return state.tool_executor.list_tools()


@router.post("/tools/execute")
def execute_tool(payload: ToolExecuteRequest, state: AppState = Depends(get_state)) -> dict:
    try:
        return state.tool_executor.submit(payload.tool, payload.args)
    except UnknownToolError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except InvalidToolArgumentsError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/approvals")
def list_approvals(state: AppState = Depends(get_state)) -> list[dict]:
    return state.tool_executor.pending_approvals()


@router.post("/approvals/{approval_id}/decide")
def decide_approval(
    approval_id: str, payload: DecideRequest, state: AppState = Depends(get_state)
) -> dict:
    result = state.tool_executor.decide(approval_id, payload.approved)
    if result.get("status") == "error":
        raise HTTPException(status_code=404, detail=result.get("error", "Approval error"))
    return result


@router.get("/tools/logs")
def tool_logs(state: AppState = Depends(get_state)) -> list[dict]:
    return state.tool_executor.recent_logs()
