"""Local retrieval (RAG) routes: index, search, and manage sources."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from .deps import AppState, get_state
from .schemas import RagIndexRequest, RagSearchRequest

router = APIRouter(prefix="/api/rag", tags=["rag"])


@router.post("/index")
def index_folder(payload: RagIndexRequest, state: AppState = Depends(get_state)) -> dict:
    try:
        return state.retrieval.index_folder(payload.folder)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/search")
def search(payload: RagSearchRequest, state: AppState = Depends(get_state)) -> dict:
    matches = state.retrieval.search(payload.query, payload.top_k)
    return {"query": payload.query, "results": matches}


@router.get("/sources")
def list_sources(state: AppState = Depends(get_state)) -> list[dict]:
    return state.retrieval.list_sources()


@router.delete("/sources/{source_id}")
def delete_source(source_id: str, state: AppState = Depends(get_state)) -> dict:
    state.retrieval.delete_source(source_id)
    return {"deleted": True, "source_id": source_id}
