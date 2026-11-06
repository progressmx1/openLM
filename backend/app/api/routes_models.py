"""Model discovery, registration, and lifecycle routes."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from ..models.adapters.base import ModelLoadError
from .deps import AppState, get_state
from .schemas import DiscoverRequest, ModelLoadRequest, ModelUpdateRequest, RegisterFileRequest

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("")
def list_models(state: AppState = Depends(get_state)) -> list[dict]:
    return state.registry.list_models()


@router.get("/{model_id}")
def get_model(model_id: str, state: AppState = Depends(get_state)) -> dict:
    model = state.registry.get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.post("/discover")
def discover(payload: DiscoverRequest, state: AppState = Depends(get_state)) -> dict:
    from ..models.registry import clean_path

    result = state.registry.discover(payload.locations)
    if not result["found"]:
        missing = [clean_path(p) for p in payload.locations if clean_path(p) and not Path(clean_path(p)).exists()]
        if missing:
            raise HTTPException(status_code=400, detail=f"Folder not found: {missing[0]}")
    return result


@router.post("/register")
def register(payload: RegisterFileRequest, state: AppState = Depends(get_state)) -> dict:
    try:
        return state.registry.register_file(payload.path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch("/{model_id}")
def update_model(
    model_id: str, payload: ModelUpdateRequest, state: AppState = Depends(get_state)
) -> dict:
    updated = state.registry.update_model(
        model_id, display_name=payload.display_name, description=payload.description
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return updated


@router.delete("/{model_id}")
def unregister(model_id: str, state: AppState = Depends(get_state)) -> dict:
    if state.registry.get_model(model_id) is None:
        raise HTTPException(status_code=404, detail="Model not found")
    state.registry.unregister(model_id)
    return {"deleted": True, "id": model_id}


@router.post("/{model_id}/load")
def load_model(
    model_id: str, payload: ModelLoadRequest | None = None, state: AppState = Depends(get_state)
) -> dict:
    options = payload.model_dump(exclude_none=True) if payload else {}
    try:
        return state.registry.load(model_id, options)
    except KeyError:
        raise HTTPException(status_code=404, detail="Model not found")
    except ModelLoadError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/{model_id}/unload")
def unload_model(model_id: str, state: AppState = Depends(get_state)) -> dict:
    return state.registry.unload(model_id)


@router.get("/{model_id}/status")
def model_status(model_id: str, state: AppState = Depends(get_state)) -> dict:
    if state.registry.get_model(model_id) is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return state.registry.status(model_id)


@router.get("/{model_id}/log")
def model_log(model_id: str, tail: int = 200, state: AppState = Depends(get_state)) -> dict:
    """Last lines of the model's runtime log (llama-server output)."""
    if state.registry.get_model(model_id) is None:
        raise HTTPException(status_code=404, detail="Model not found")
    status = state.registry.status(model_id)
    path = status.get("log_path")
    if not path:
        return {"lines": [], "path": None}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return {"lines": [], "path": path}
    tail = max(1, min(int(tail), 2000))
    return {"lines": lines[-tail:], "path": path}
