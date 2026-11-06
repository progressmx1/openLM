"""Cloud model providers: API keys (stored encrypted, never returned), model lists, and
registering API models."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..core.secrets import mask
from ..models.adapters.base import ModelLoadError, ModelNotLoadedError
from ..models.adapters.cloud import PROVIDERS
from .deps import AppState, get_state

router = APIRouter(prefix="/api/providers", tags=["providers"])


class ProviderUpdate(BaseModel):
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class CloudModelCreate(BaseModel):
    model: str
    display_name: Optional[str] = None


def _cloud(state: AppState):
    cloud = getattr(state.registry.adapter, "cloud", None)
    if cloud is None:
        raise HTTPException(status_code=503, detail="Cloud models are not available in this build.")
    return cloud


def _provider(provider: str) -> dict:
    if provider not in PROVIDERS:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider}")
    return PROVIDERS[provider]


def _describe(state: AppState, provider: str) -> dict:
    info = PROVIDERS[provider]
    cloud = _cloud(state)
    key = cloud.key(provider)
    return {
        "id": provider,
        "label": info["label"],
        "configured": bool(key),
        "masked_key": mask(key) if key else None,
        "base_url": cloud.base_url(provider),
        "default_base_url": info["base_url"],
        "key_url": info["key_url"],
        "suggested": info["suggested"],
    }


@router.get("")
def list_providers(state: AppState = Depends(get_state)) -> list[dict]:
    return [_describe(state, p) for p in PROVIDERS]


@router.put("/{provider}")
def update_provider(provider: str, payload: ProviderUpdate, state: AppState = Depends(get_state)) -> dict:
    _provider(provider)
    cloud = _cloud(state)
    if payload.api_key is not None:
        key = payload.api_key.strip()
        if not key:
            raise HTTPException(status_code=400, detail="The API key is empty.")
        if any(c.isspace() for c in key):
            raise HTTPException(status_code=400, detail="An API key has no spaces; paste just the key.")
        cloud.secrets.set(f"{provider}.api_key", key)
    if payload.base_url is not None:
        url = payload.base_url.strip().rstrip("/")
        if url and not url.startswith(("https://", "http://")):
            raise HTTPException(status_code=400, detail="The endpoint must start with https:// (or http:// for a local server).")
        settings = state.settings_store.load()
        urls = {k: v for k, v in settings.provider_base_urls.items() if k != provider}
        if url and url != PROVIDERS[provider]["base_url"]:
            urls[provider] = url
        state.settings_store.save(settings.model_copy(update={"provider_base_urls": urls}))
    return _describe(state, provider)


@router.delete("/{provider}/key")
def delete_key(provider: str, state: AppState = Depends(get_state)) -> dict:
    _provider(provider)
    _cloud(state).secrets.delete(f"{provider}.api_key")
    return _describe(state, provider)


@router.get("/{provider}/models")
def provider_models(provider: str, state: AppState = Depends(get_state)) -> dict:
    """Models this key can use, straight from the provider (the only call that goes online)."""
    _provider(provider)
    try:
        return {"models": _cloud(state).list_provider_models(provider)}
    except (ModelLoadError, ModelNotLoadedError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{provider}/models")
def add_model(provider: str, payload: CloudModelCreate, state: AppState = Depends(get_state)) -> dict:
    _provider(provider)
    try:
        return state.registry.register_cloud(provider, payload.model, payload.display_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
