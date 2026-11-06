"""Live GPU memory/utilisation telemetry, per adapter and per process.

Why not just nvidia-smi: on Windows (WDDM) ``nvidia-smi --query-compute-apps`` reports
per-process memory as ``[N/A]``, and it knows nothing about AMD or Intel GPUs. Windows
exposes vendor-neutral performance counters instead:

* ``\\GPU Process Memory(pid_<pid>_luid_<luid>_phys_0)\\Dedicated Usage`` / ``Shared Usage``
* ``\\GPU Adapter Memory(luid_<luid>_phys_0)\\Dedicated Usage`` / ``Shared Usage``
* ``\\GPU Engine(pid_<pid>_luid_<luid>_phys_0_eng_<n>_engtype_<type>)\\Utilization Percentage``

Counter instances are keyed by adapter LUID, so adapters are enumerated through DXGI
(``IDXGIFactory1::EnumAdapters1``) to map each LUID to a name and its dedicated VRAM.
Everything is read through ctypes (PDH + DXGI): no PowerShell processes, ~ms per sample.

On other platforms (or if anything fails) :meth:`GpuTelemetry.snapshot` returns
``{"supported": False, ...}`` — telemetry is best-effort and must never break the app.
"""
from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from ..core.logging import get_logger

log = get_logger(__name__)

MIB = 1024 * 1024

_PROCESS_INSTANCE = re.compile(r"pid_(\d+)_luid_(0x[0-9a-fA-F]+_0x[0-9a-fA-F]+)_phys_\d+", re.I)
_ADAPTER_INSTANCE = re.compile(r"luid_(0x[0-9a-fA-F]+_0x[0-9a-fA-F]+)_phys_\d+", re.I)
_ENGINE_INSTANCE = re.compile(
    r"pid_(\d+)_luid_(0x[0-9a-fA-F]+_0x[0-9a-fA-F]+)_phys_\d+_eng_\d+_engtype_(.+)$", re.I
)


def normalize_gpu_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.replace("(R)", "").replace("(TM)", "")).strip().lower()


@dataclass
class Adapter:
    luid: str  # "0x00000000_0x00015117" (high_low, as in counter instance names)
    name: str
    vendor_id: int
    dedicated_mib: int
    software: bool = False

    @property
    def vendor(self) -> str:
        return {0x10DE: "nvidia", 0x1002: "amd", 0x1022: "amd", 0x8086: "intel", 0x1414: "microsoft"}.get(
            self.vendor_id, "unknown"
        )


# ---------------------------------------------------------------- DXGI (Windows)
def enumerate_adapters() -> list[Adapter]:
    """List hardware adapters with their LUIDs. Empty list off Windows or on failure."""
    if sys.platform != "win32":
        return []
    try:
        return _dxgi_adapters()
    except Exception as exc:  # noqa: BLE001
        log.warning("DXGI adapter enumeration failed: %s", exc)
        return []


def _dxgi_adapters() -> list[Adapter]:
    import ctypes
    from ctypes import wintypes

    class LUID(ctypes.Structure):
        _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

    class DESC1(ctypes.Structure):
        _fields_ = [
            ("Description", wintypes.WCHAR * 128),
            ("VendorId", wintypes.UINT),
            ("DeviceId", wintypes.UINT),
            ("SubSysId", wintypes.UINT),
            ("Revision", wintypes.UINT),
            ("DedicatedVideoMemory", ctypes.c_size_t),
            ("DedicatedSystemMemory", ctypes.c_size_t),
            ("SharedSystemMemory", ctypes.c_size_t),
            ("AdapterLuid", LUID),
            ("Flags", wintypes.UINT),
        ]

    class GUID(ctypes.Structure):
        _fields_ = [("a", ctypes.c_uint32), ("b", ctypes.c_uint16), ("c", ctypes.c_uint16), ("d", ctypes.c_ubyte * 8)]

    # IID_IDXGIFactory1 {770aae78-f26f-4dba-a829-253c83d1b387}
    iid = GUID(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))
    dxgi = ctypes.WinDLL("dxgi")
    factory = ctypes.c_void_p()
    if dxgi.CreateDXGIFactory1(ctypes.byref(iid), ctypes.byref(factory)) != 0:
        return []

    def method(obj: ctypes.c_void_p, index: int, restype: Any, *argtypes: Any) -> Any:
        vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtable[index])

    adapters: list[Adapter] = []
    try:
        index = 0
        while index < 32:
            adapter = ctypes.c_void_p()
            # IDXGIFactory1::EnumAdapters1 is vtable slot 12.
            hr = method(factory, 12, ctypes.c_long, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(
                factory, index, ctypes.byref(adapter)
            )
            if hr != 0:
                break
            desc = DESC1()
            # IDXGIAdapter1::GetDesc1 is vtable slot 10.
            method(adapter, 10, ctypes.c_long, ctypes.POINTER(DESC1))(adapter, ctypes.byref(desc))
            method(adapter, 2, ctypes.c_ulong)(adapter)  # Release
            luid = f"0x{desc.AdapterLuid.HighPart & 0xFFFFFFFF:08x}_0x{desc.AdapterLuid.LowPart:08x}"
            adapters.append(
                Adapter(
                    luid=luid,
                    name=desc.Description,
                    vendor_id=desc.VendorId,
                    dedicated_mib=int(desc.DedicatedVideoMemory // MIB),
                    software=bool(desc.Flags & 2),  # DXGI_ADAPTER_FLAG_SOFTWARE
                )
            )
            index += 1
    finally:
        method(factory, 2, ctypes.c_ulong)(factory)
    return adapters


# ------------------------------------------------------------------ PDH (Windows)
class _PdhQuery:
    """A persistent PDH query with wildcard counters (instances re-expand every collect)."""

    PATHS = {
        "proc_dedicated": r"\GPU Process Memory(*)\Dedicated Usage",
        "proc_shared": r"\GPU Process Memory(*)\Shared Usage",
        "adapter_dedicated": r"\GPU Adapter Memory(*)\Dedicated Usage",
        "adapter_shared": r"\GPU Adapter Memory(*)\Shared Usage",
        "engine_util": r"\GPU Engine(*)\Utilization Percentage",
    }

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self.ctypes = ctypes
        self.pdh = ctypes.WinDLL("pdh")
        self.query = ctypes.c_void_p()
        status = self.pdh.PdhOpenQueryW(None, None, ctypes.byref(self.query))
        if status != 0:
            raise OSError(f"PdhOpenQueryW failed: 0x{status & 0xFFFFFFFF:08x}")
        self.counters: dict[str, Any] = {}
        for key, path in self.PATHS.items():
            handle = ctypes.c_void_p()
            status = self.pdh.PdhAddEnglishCounterW(self.query, wintypes.LPCWSTR(path), None, ctypes.byref(handle))
            if status == 0:
                self.counters[key] = handle
        if not self.counters:
            raise OSError("No GPU performance counters are available on this system.")

        class VALUE(ctypes.Structure):
            _fields_ = [("CStatus", wintypes.DWORD), ("pad", wintypes.DWORD), ("value", ctypes.c_double)]

        class LARGE(ctypes.Structure):
            _fields_ = [("CStatus", wintypes.DWORD), ("pad", wintypes.DWORD), ("value", ctypes.c_longlong)]

        class ITEM_D(ctypes.Structure):
            _fields_ = [("name", wintypes.LPWSTR), ("v", VALUE)]

        class ITEM_L(ctypes.Structure):
            _fields_ = [("name", wintypes.LPWSTR), ("v", LARGE)]

        self.ITEM_D, self.ITEM_L = ITEM_D, ITEM_L
        self.pdh.PdhCollectQueryData(self.query)  # prime rate counters
        self.primed_at = time.monotonic()

    def collect(self) -> dict[str, dict[str, float]]:
        ctypes = self.ctypes
        self.pdh.PdhCollectQueryData(self.query)
        out: dict[str, dict[str, float]] = {}
        for key, handle in self.counters.items():
            is_rate = key == "engine_util"
            fmt = 0x00000200 if is_rate else 0x00000400  # PDH_FMT_DOUBLE / PDH_FMT_LARGE
            item_type = self.ITEM_D if is_rate else self.ITEM_L
            size = ctypes.c_ulong(0)
            count = ctypes.c_ulong(0)
            status = self.pdh.PdhGetFormattedCounterArrayW(handle, fmt, ctypes.byref(size), ctypes.byref(count), None)
            if (status & 0xFFFFFFFF) != 0x800007D2 or size.value == 0:  # PDH_MORE_DATA
                out[key] = {}
                continue
            buffer = (ctypes.c_byte * size.value)()
            status = self.pdh.PdhGetFormattedCounterArrayW(
                handle, fmt, ctypes.byref(size), ctypes.byref(count), buffer
            )
            if status != 0:
                out[key] = {}
                continue
            items = ctypes.cast(buffer, ctypes.POINTER(item_type))
            values: dict[str, float] = {}
            for i in range(count.value):
                item = items[i]
                if item.v.CStatus in (0, 1) and item.name:  # PDH_CSTATUS_VALID_DATA / NEW_DATA
                    values[item.name] = float(item.v.value)
            out[key] = values
        return out

    def close(self) -> None:
        try:
            self.pdh.PdhCloseQuery(self.query)
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------- nvidia-smi
def query_nvidia_smi(runner: Optional[Callable[[list[str]], str]] = None) -> list[dict[str, Any]]:
    """Whole-GPU totals from nvidia-smi (per-process memory is N/A under WDDM)."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        text = runner(cmd) if runner else _run(cmd)
    except Exception:  # noqa: BLE001 - no NVIDIA driver, or nvidia-smi missing
        return []
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        gpus.append(
            {
                "index": _int(parts[0]),
                "name": parts[1],
                "used_mib": _int(parts[2]),
                "total_mib": _int(parts[3]),
                "utilization_pct": _int(parts[4]),
            }
        )
    return gpus


def _run(cmd: list[str]) -> str:
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, stdin=subprocess.DEVNULL, **kwargs)
    if result.returncode != 0:
        raise OSError(result.stderr.strip() or f"exit {result.returncode}")
    return result.stdout


def _int(value: str) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------- aggregation
def build_snapshot(
    adapters: list[Adapter],
    counters: dict[str, dict[str, float]],
    pids: dict[int, str],
    nvidia: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Combine raw counter values into a per-adapter view. Pure function (tested directly).

    ``pids`` maps llama-server process ids to model ids, so each adapter lists how much
    memory every loaded model uses on it.
    """
    by_luid: dict[str, dict[str, Any]] = {}
    for adapter in adapters:
        if adapter.software:
            continue
        by_luid[adapter.luid.lower()] = {
            "luid": adapter.luid,
            "name": adapter.name,
            "vendor": adapter.vendor,
            "integrated": adapter.vendor == "intel" and adapter.dedicated_mib <= 512,
            "dedicated_total_mib": adapter.dedicated_mib,
            "dedicated_used_mib": None,
            "shared_used_mib": None,
            "utilization": {},
            "processes": [],
        }

    for instance, value in counters.get("adapter_dedicated", {}).items():
        match = _ADAPTER_INSTANCE.search(instance)
        if match and match.group(1).lower() in by_luid:
            by_luid[match.group(1).lower()]["dedicated_used_mib"] = round(value / MIB)
    for instance, value in counters.get("adapter_shared", {}).items():
        match = _ADAPTER_INSTANCE.search(instance)
        if match and match.group(1).lower() in by_luid:
            by_luid[match.group(1).lower()]["shared_used_mib"] = round(value / MIB)

    procs: dict[tuple[int, str], dict[str, Any]] = {}
    for key, field_name in (("proc_dedicated", "dedicated_mib"), ("proc_shared", "shared_mib")):
        for instance, value in counters.get(key, {}).items():
            match = _PROCESS_INSTANCE.search(instance)
            if not match:
                continue
            pid, luid = int(match.group(1)), match.group(2).lower()
            if pid not in pids or luid not in by_luid:
                continue
            entry = procs.setdefault((pid, luid), {"pid": pid, "model_id": pids[pid], "dedicated_mib": 0, "shared_mib": 0})
            entry[field_name] = round(value / MIB)

    util: dict[str, dict[str, float]] = {}
    proc_util: dict[tuple[int, str], float] = {}
    for instance, value in counters.get("engine_util", {}).items():
        match = _ENGINE_INSTANCE.search(instance)
        if not match:
            continue
        pid, luid, engine = int(match.group(1)), match.group(2).lower(), match.group(3).lower()
        if luid not in by_luid:
            continue
        # Task Manager style: per engine type, sum over processes, capped at 100.
        bucket = util.setdefault(luid, {})
        bucket[engine] = min(100.0, bucket.get(engine, 0.0) + value)
        if pid in pids:
            proc_util[(pid, luid)] = max(proc_util.get((pid, luid), 0.0), value)

    for luid, entry in by_luid.items():
        engines = util.get(luid, {})
        entry["utilization"] = {k: round(v, 1) for k, v in sorted(engines.items()) if v >= 0.05}
        entry["busiest_engine_pct"] = round(max(engines.values()), 1) if engines else 0.0
    for (pid, luid), entry in procs.items():
        entry["utilization_pct"] = round(proc_util.get((pid, luid), 0.0), 1)
        if entry["dedicated_mib"] or entry["shared_mib"]:
            by_luid[luid]["processes"].append(entry)

    # Prefer nvidia-smi totals when available (they include other processes' contexts).
    nvidia_list = list(nvidia)
    for entry in by_luid.values():
        if entry["vendor"] != "nvidia":
            continue
        for gpu in nvidia_list:
            if normalize_gpu_name(gpu["name"]) == normalize_gpu_name(entry["name"]):
                entry["nvidia_smi"] = gpu
                break

    return {"supported": True, "adapters": list(by_luid.values())}


class GpuTelemetry:
    """Thread-safe, cached sampler. ``snapshot()`` is cheap enough to poll every second."""

    def __init__(self, min_interval: float = 0.8, nvidia_interval: float = 3.0) -> None:
        self._lock = threading.Lock()
        self._query: Optional[_PdhQuery] = None
        self._adapters: list[Adapter] = []
        self._adapters_at = 0.0
        self._last: Optional[dict[str, Any]] = None
        self._last_at = 0.0
        self._last_pids: dict[int, str] = {}
        self._nvidia: list[dict[str, Any]] = []
        self._nvidia_at = 0.0
        self._min_interval = min_interval
        self._nvidia_interval = nvidia_interval
        self._error: Optional[str] = None

    @property
    def supported(self) -> bool:
        return sys.platform == "win32"

    def snapshot(self, pids: Optional[dict[int, str]] = None) -> dict[str, Any]:
        pids = dict(pids or {})
        if not self.supported:
            return {"supported": False, "reason": "Per-process GPU counters are only read on Windows.", "adapters": []}
        with self._lock:
            now = time.monotonic()
            if self._last is not None and now - self._last_at < self._min_interval and pids == self._last_pids:
                return self._last
            try:
                if not self._adapters or now - self._adapters_at > 30:
                    self._adapters = enumerate_adapters()
                    self._adapters_at = now
                if self._query is None:
                    self._query = _PdhQuery()
                counters = self._query.collect()
                if now - self._nvidia_at > self._nvidia_interval:
                    self._nvidia = query_nvidia_smi()
                    self._nvidia_at = now
                snap = build_snapshot(self._adapters, counters, pids, self._nvidia)
                self._error = None
            except Exception as exc:  # noqa: BLE001
                self._error = str(exc)
                log.warning("GPU telemetry failed: %s", exc)
                snap = {"supported": False, "reason": f"GPU counters unavailable: {exc}", "adapters": []}
            snap["sampled_at"] = time.time()
            self._last, self._last_at, self._last_pids = snap, now, pids
            return snap

    def close(self) -> None:
        with self._lock:
            if self._query is not None:
                self._query.close()
                self._query = None
