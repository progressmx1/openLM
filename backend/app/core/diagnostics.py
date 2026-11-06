"""Best-effort system diagnostics.

Every probe is wrapped so that an unsupported or denied query never crashes the
application; unknown values are reported as ``None``/``"unknown"`` instead.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
import shutil
import subprocess
import sys
from typing import Any

from .paths import DATA_DIR


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _windows_memory() -> dict[str, int | None]:
    """Return total/available physical memory in bytes via the Win32 API."""
    try:
        if sys.platform != "win32":
            return {"total_bytes": None, "available_bytes": None}

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return {"total_bytes": None, "available_bytes": None}
        return {
            "total_bytes": int(status.ullTotalPhys),
            "available_bytes": int(status.ullAvailPhys),
        }
    except Exception:
        return {"total_bytes": None, "available_bytes": None}


def _gpu_info() -> dict[str, Any]:
    """Probe NVIDIA via nvidia-smi; otherwise report unknown (best-effort)."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return {"vendor": None, "name": None, "vram_total_mb": None, "driver": None}
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 4:
                continue
            name, total_mb, free_mb, driver = parts[:4]
            gpus.append(
                {
                    "vendor": "nvidia",
                    "name": name,
                    "vram_total_mb": int(float(total_mb)),
                    "vram_free_mb": int(float(free_mb)),
                    "driver": driver,
                }
            )
        if not gpus:
            return {"vendor": None, "name": None, "vram_total_mb": None, "driver": None}
        return {**gpus[0], "count": len(gpus)}
    except Exception:
        return {"vendor": None, "name": None, "vram_total_mb": None, "driver": None}


def _video_controllers() -> list[dict[str, Any]]:
    """All display adapters Windows knows about (NVIDIA, AMD eGPU, Intel iGPU)."""
    if sys.platform != "win32":
        return []
    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-CimInstance Win32_VideoController | "
                "Select-Object Name,DriverVersion,Status | ConvertTo-Json -Compress",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=0x08000000,  # CREATE_NO_WINDOW
        )
        import json

        data = json.loads(result.stdout or "[]")
        if isinstance(data, dict):
            data = [data]
        return [
            {"name": item.get("Name"), "driver": item.get("DriverVersion"), "status": item.get("Status")}
            for item in data
            if isinstance(item, dict)
        ]
    except Exception:
        return []


def collect_diagnostics() -> dict[str, Any]:
    memory = _safe(_windows_memory, {"total_bytes": None, "available_bytes": None})
    gpu = _safe(_gpu_info)

    try:
        disk = shutil.disk_usage(str(DATA_DIR))
        disk_info = {
            "total_bytes": disk.total,
            "free_bytes": disk.free,
            "used_bytes": disk.used,
        }
    except Exception:
        disk_info = {"total_bytes": None, "free_bytes": None, "used_bytes": None}

    llama_gpu_offload = None
    try:
        import llama_cpp.llama_cpp as _lc

        llama_gpu_offload = bool(_lc.llama_supports_gpu_offload())
    except Exception:
        llama_gpu_offload = None

    python_info = {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
    }

    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "cpu": {
            "brand": platform.processor() or "unknown",
            "logical_count": os.cpu_count(),
        },
        "memory": memory,
        "gpu": gpu,
        "video_controllers": _safe(_video_controllers, []),
        "disk": disk_info,
        "python": python_info,
        "llama_cpp": {
            "version": _safe(lambda: __import__("llama_cpp").__version__),
            "gpu_offload_supported": llama_gpu_offload,
        },
    }
