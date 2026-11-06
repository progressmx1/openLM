"""System, settings, and diagnostics routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from ..core.diagnostics import collect_diagnostics
from ..core.settings import AppSettings
from .deps import AppState, get_state
from .schemas import SettingsUpdateRequest

router = APIRouter(prefix="/api", tags=["system"])

APP_VERSION = "0.1.0"


@router.get("/health")
def health(state: AppState = Depends(get_state)) -> dict:
    return {
        "status": "ok",
        "version": APP_VERSION,
        "loaded_models": state.registry.runtime_info().get("loaded_models", []),
    }


@router.get("/system/diagnostics")
def diagnostics(state: AppState = Depends(get_state)) -> dict:
    data = collect_diagnostics()
    data["backend_version"] = APP_VERSION
    runtime = state.registry.runtime_info()
    data["loaded_models"] = runtime.get("loaded_models", [])
    data["runtime"] = {
        "name": runtime.get("name"),
        "mode": runtime.get("mode"),
        "gpu_offload_supported": runtime.get("gpu_offload_supported"),
    }
    try:
        data["gpu_devices"] = state.registry.devices()
    except Exception:  # noqa: BLE001 - diagnostics must never fail
        data["gpu_devices"] = []
    return data


@router.get("/system/devices")
async def devices(refresh: bool = False, state: AppState = Depends(get_state)) -> dict:
    """GPUs the model runtimes can use, and which models occupy each one."""
    found = await run_in_threadpool(state.registry.devices, refresh)
    runtime = state.registry.runtime_info()
    server = runtime.get("llama_server") or {}
    return {
        "runtime": runtime.get("name"),
        "mode": runtime.get("mode"),
        "devices": found,
        "binaries": server.get("binaries", []),
        "errors": server.get("device_errors", []),
    }


@router.get("/system/gpu-telemetry")
async def gpu_telemetry(state: AppState = Depends(get_state)) -> dict:
    """Live per-GPU memory and utilisation, with each loaded model's own usage.

    Uses the Windows "GPU Process Memory"/"GPU Engine" counters (vendor-neutral) and
    nvidia-smi totals for NVIDIA cards.
    """
    adapter = state.registry.adapter
    server = getattr(adapter, "server", adapter)
    telemetry = getattr(server, "telemetry", None)
    if telemetry is None:
        return {"supported": False, "reason": "Telemetry is not available for this runtime.", "adapters": []}
    pids = adapter.process_ids()
    return await run_in_threadpool(telemetry.snapshot, pids)


@router.get("/system/runtimes")
def runtimes(state: AppState = Depends(get_state)) -> dict:
    return state.registry.runtime_info()


@router.get("/settings")
def get_settings(state: AppState = Depends(get_state)) -> dict:
    return state.settings_store.load().model_dump()


@router.put("/settings")
def update_settings(payload: SettingsUpdateRequest, state: AppState = Depends(get_state)) -> dict:
    current = state.settings_store.load()
    try:
        updated = AppSettings.model_validate({**current.model_dump(), **payload.settings})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Invalid settings: {exc}") from exc
    return state.settings_store.save(updated).model_dump()
