"""Coding-agent API: sessions, live event stream, approvals, file changes, prompts,
and workspace browsing for the file tree."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..agents.coding.environment import IGNORED_DIRS
from ..agents.coding.service import MODES, SLASH_COMMANDS, AgentError
from ..security.workspace import WorkspacePolicy, canonicalize
from .deps import AppState, get_state

router = APIRouter(prefix="/api/agent", tags=["coding-agent"])
workspace_router = APIRouter(prefix="/api/workspace", tags=["workspace"])


class SessionCreate(BaseModel):
    workspace: str = ""  # empty for plain chat sessions
    model_id: Optional[str] = None
    mode: str = "auto"


class SessionUpdate(BaseModel):
    title: Optional[str] = None
    model_id: Optional[str] = None
    mode: Optional[str] = None
    max_steps: Optional[int] = Field(default=None, ge=1, le=200)


class MessageIn(BaseModel):
    text: str = Field(min_length=1)


class DecisionIn(BaseModel):
    call_id: str
    approve: bool
    remember: bool = False


class FileDecisionIn(BaseModel):
    turn: int
    path: str
    accept: bool


class PromptIn(BaseModel):
    content: str


def _agent(state: AppState):
    return state.agent_service


def _err(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail="Session not found")
    return HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------- sessions
@router.get("/meta")
def meta() -> dict:
    return {"modes": list(MODES), "commands": SLASH_COMMANDS}


@router.get("/sessions")
def list_sessions(workspace: Optional[str] = None, kind: Optional[str] = None, state: AppState = Depends(get_state)) -> list[dict]:
    return _agent(state).list_sessions(workspace, kind)


@router.post("/sessions")
def create_session(payload: SessionCreate, state: AppState = Depends(get_state)) -> dict:
    try:
        return _agent(state).create_session(payload.workspace, payload.model_id, payload.mode)
    except AgentError as exc:
        raise _err(exc)


@router.get("/sessions/{session_id}")
def get_session(session_id: str, state: AppState = Depends(get_state)) -> dict:
    try:
        return _agent(state).describe(session_id)
    except KeyError as exc:
        raise _err(exc)


@router.patch("/sessions/{session_id}")
def update_session(session_id: str, payload: SessionUpdate, state: AppState = Depends(get_state)) -> dict:
    agent = _agent(state)
    try:
        fields: dict[str, Any] = payload.model_dump(exclude_none=True, exclude={"max_steps"})
        if payload.max_steps is not None:
            session = agent.describe(session_id)
            fields["options"] = {**session.get("options", {}), "max_steps": payload.max_steps}
        return agent.update_session(session_id, **fields)
    except (AgentError, KeyError) as exc:
        raise _err(exc)


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, state: AppState = Depends(get_state)) -> dict:
    _agent(state).delete_session(session_id)
    return {"deleted": True}


@router.get("/sessions/{session_id}/events")
def session_events(session_id: str, after: int = 0, state: AppState = Depends(get_state)) -> list[dict]:
    return _agent(state).events(session_id, after)


@router.get("/sessions/{session_id}/stream")
async def stream(session_id: str, request: Request, after: int = 0, state: AppState = Depends(get_state)) -> StreamingResponse:
    agent = _agent(state)
    if agent.store.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")

    async def source():
        last = after
        idle = 0.0
        while True:
            if await request.is_disconnected():
                break
            events = agent.hub.live_after(session_id, last)
            if events:
                idle = 0.0
                for event in events:
                    last = event["seq"]
                    yield f"data: {json.dumps(event)}\n\n"
                continue
            await asyncio.sleep(0.05)
            idle += 0.05
            if idle >= 15:
                idle = 0.0
                yield ": keep-alive\n\n"

    return StreamingResponse(
        source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@router.post("/sessions/{session_id}/messages")
def send_message(session_id: str, payload: MessageIn, state: AppState = Depends(get_state)) -> dict:
    try:
        return _agent(state).send(session_id, payload.text)
    except (AgentError, KeyError) as exc:
        raise _err(exc)


@router.post("/sessions/{session_id}/interrupt")
def interrupt(session_id: str, state: AppState = Depends(get_state)) -> dict:
    return {"interrupted": _agent(state).interrupt(session_id)}


@router.post("/sessions/{session_id}/decision")
def decide(session_id: str, payload: DecisionIn, state: AppState = Depends(get_state)) -> dict:
    ok = _agent(state).decide(session_id, payload.call_id, payload.approve, payload.remember)
    if not ok:
        raise HTTPException(status_code=409, detail="That action is no longer waiting for approval.")
    return {"ok": True}


@router.get("/sessions/{session_id}/changes")
def changes(session_id: str, state: AppState = Depends(get_state)) -> list[dict]:
    return _agent(state).changes(session_id)


@router.post("/sessions/{session_id}/changes")
def decide_file(session_id: str, payload: FileDecisionIn, state: AppState = Depends(get_state)) -> dict:
    agent = _agent(state)
    try:
        if payload.accept:
            return agent.accept_file(session_id, payload.turn, payload.path)
        return agent.reject_file(session_id, payload.turn, payload.path)
    except AgentError as exc:
        raise _err(exc)


@router.post("/sessions/{session_id}/undo")
def undo(session_id: str, state: AppState = Depends(get_state)) -> dict:
    try:
        return {"reverted": _agent(state).undo_turn(session_id)}
    except AgentError as exc:
        raise _err(exc)


@router.get("/sessions/{session_id}/system-prompt")
def system_prompt(session_id: str, state: AppState = Depends(get_state)) -> dict:
    try:
        return {"prompt": _agent(state).preview_system_prompt(session_id)}
    except KeyError as exc:
        raise _err(exc)


# ----------------------------------------------------------------- prompts
@router.get("/prompts")
def prompts(state: AppState = Depends(get_state)) -> list[dict]:
    return _agent(state).prompts.describe()


@router.put("/prompts/{name}")
def set_prompt(name: str, payload: PromptIn, state: AppState = Depends(get_state)) -> dict:
    try:
        _agent(state).prompts.set(name, payload.content)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid JSON: {exc}")
    return {"ok": True}


@router.delete("/prompts/{name}")
def reset_prompt(name: str, state: AppState = Depends(get_state)) -> dict:
    try:
        _agent(state).prompts.reset(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"ok": True}


# --------------------------------------------------------------- workspace
def _inside_approved(state: AppState, path: str) -> Path:
    target = canonicalize(path)
    policy = WorkspacePolicy(state.settings_store.load().approved_workspaces)
    if not policy.is_within(target):
        raise HTTPException(status_code=403, detail="That folder is not an open workspace.")
    return target


@workspace_router.get("/tree")
def tree(path: str, state: AppState = Depends(get_state)) -> dict:
    folder = _inside_approved(state, path)
    if not folder.is_dir():
        raise HTTPException(status_code=404, detail="Folder not found")
    entries = []
    try:
        children = sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    for child in children[:1000]:
        if child.name in IGNORED_DIRS:
            continue
        is_dir = child.is_dir()
        try:
            size = None if is_dir else child.stat().st_size
        except OSError:
            size = None
        entries.append({"name": child.name, "path": str(child), "dir": is_dir, "size": size})
    return {"path": str(folder), "entries": entries}


@workspace_router.get("/file")
def read_file(path: str, state: AppState = Depends(get_state)) -> dict:
    target = _inside_approved(state, path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if target.stat().st_size > 1_000_000:
        raise HTTPException(status_code=413, detail="File is larger than 1 MB.")
    data = target.read_bytes()
    if b"\x00" in data[:8192]:
        return {"path": str(target), "binary": True, "content": ""}
    return {"path": str(target), "binary": False, "content": data.decode("utf-8", errors="replace")}


@workspace_router.get("/recent")
def recent(state: AppState = Depends(get_state)) -> list[dict]:
    """Open workspaces (approved folders) with their latest session time."""
    sessions = _agent(state).list_sessions()
    latest: dict[str, str] = {}
    for s in sessions:
        latest.setdefault(s["workspace"], s["updated_at"])
    out = []
    for root in state.settings_store.load().approved_workspaces:
        key = str(canonicalize(root))
        out.append({"path": key, "name": os.path.basename(key) or key, "exists": os.path.isdir(key), "last_used": latest.get(key)})
    out.sort(key=lambda w: w["last_used"] or "", reverse=True)
    return out


@workspace_router.post("/open")
def open_workspace(payload: SessionCreate, state: AppState = Depends(get_state)) -> dict:
    """Approve a folder as a workspace (the user picked it in the UI)."""
    try:
        root = _agent(state).resolve_folder(payload.workspace)
    except AgentError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    _agent(state)._approve_workspace(str(root))
    return {"path": str(root), "name": root.name or str(root)}


@workspace_router.get("/browse")
def browse(path: Optional[str] = None, ext: Optional[str] = None, state: AppState = Depends(get_state)) -> dict:
    """Folder names for the folder/model pickers (works without Electron's dialog).

    Read-only: folder names, plus — only when ``ext`` is given (e.g. ``.gguf``) — names and
    sizes of files with that extension. Never file contents. With no path: the home folder
    and the drive roots.
    """
    if not path:
        home = Path.home()
        roots = [{"name": f"Home ({home.name})", "path": str(home)}]
        if os.name == "nt":
            import string

            roots += [{"name": f"{d}:\\", "path": f"{d}:\\"} for d in string.ascii_uppercase if os.path.isdir(f"{d}:\\")]
        else:
            roots.append({"name": "/", "path": "/"})
        return {"path": None, "parent": None, "dirs": roots, "files": []}
    try:
        folder = _agent(state).resolve_folder(path)
    except AgentError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    suffix = ("." + ext.lower().lstrip(".")) if ext else None
    dirs: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    try:
        for child in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
            if child.name.startswith((".", "$")) or child.name in IGNORED_DIRS:
                continue
            try:
                if child.is_dir():
                    dirs.append({"name": child.name, "path": str(child)})
                elif suffix and child.suffix.lower() == suffix:
                    files.append({"name": child.name, "path": str(child), "size": child.stat().st_size})
            except OSError:
                continue
            if len(dirs) + len(files) >= 1000:
                break
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Windows denied access to {folder}.")
    parent = folder.parent
    return {"path": str(folder), "parent": str(parent) if parent != folder else None, "dirs": dirs, "files": files}
