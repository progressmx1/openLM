"""GPU device discovery and model placement for llama.cpp ``llama-server`` runtimes.

A *runtime* is one ``llama-server`` build (CUDA, Vulkan, ROCm, CPU, ...). Each build
can see a different set of devices: a CUDA build only sees NVIDIA cards, while a
Vulkan build sees every GPU (NVIDIA, AMD and Intel) at once. Devices are discovered
by running ``llama-server --list-devices``, whose output looks like::

    Available devices:
      Vulkan0: AMD Radeon RX 9060 XT (16304 MiB, 15800 MiB free)
      Vulkan1: NVIDIA GeForce RTX 4070 Laptop GPU (8188 MiB, 7900 MiB free)

The same physical card can appear under several runtimes (the RTX 4070 is both
``CUDA0`` and ``Vulkan1``). Placement decisions and the "one model per GPU" rule
use a *physical key* derived from the device name so those entries are treated as
one card.

Device indices can change when an eGPU is plugged in or removed, so saved
placements remember the device *name* as well as its id, and are re-resolved by
name first.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from ..core.paths import PROJECT_ROOT

RUNTIMES_DIR = PROJECT_ROOT / "runtimes" / "llama.cpp"

# Preferred runtime per physical card, fastest first. NVIDIA cards prefer CUDA when
# a CUDA build is installed; every other card falls through to Vulkan.
RUNTIME_PREFERENCE = ["cuda", "rocm", "vulkan", "sycl", "openvino", "opencl", "cpu"]
_KNOWN_RUNTIMES = set(RUNTIME_PREFERENCE) | {"hip", "metal"}

SERVER_NAMES = ("llama-server.exe", "llama-server", "llama_server.py")

_DEVICE_LINE = re.compile(
    r"^\s*(?P<id>[A-Za-z][\w.-]*\d+):\s+(?P<name>.+?)\s+\((?P<total>\d+) MiB, (?P<free>\d+) MiB free\)\s*$"
)

# Rough VRAM headroom for KV cache + compute buffers on top of the weights. The
# server's own ``--fit`` logic makes the final call; this only guides placement.
_OVERHEAD_MIB = 1024
_WEIGHT_FACTOR = 1.10


class PlacementError(ValueError):
    """Raised when a requested placement cannot be satisfied; message is user-facing."""


# --------------------------------------------------------------------- binaries
@dataclass(frozen=True)
class ServerBinary:
    path: str
    runtime: str

    def command(self) -> list[str]:
        # A ``.py`` "binary" is only used by the test-suite's fake server.
        if self.path.endswith(".py"):
            return [sys.executable, self.path]
        return [self.path]

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "runtime": self.runtime}


def guess_runtime(path: str) -> str:
    """Infer the runtime from folder names like ``llama-b11960-bin-win-vulkan-x64``."""
    parts = [p.lower() for p in Path(path).parts[:-1]]
    for part in reversed(parts):
        for token in re.split(r"[^a-z0-9]+", part):
            if token in _KNOWN_RUNTIMES:
                return "rocm" if token == "hip" else token
    return "unknown"


def _find_in_folder(folder: Path) -> list[Path]:
    found: list[Path] = []
    for name in SERVER_NAMES:
        direct = folder / name
        if direct.is_file():
            found.append(direct)
    if found:
        return found
    # Release zips extract into one nested folder; look one and two levels down.
    for pattern in ("*/", "*/*/"):
        for sub in sorted(folder.glob(pattern)):
            for name in SERVER_NAMES:
                candidate = sub / name
                if candidate.is_file():
                    found.append(candidate)
    return found


def find_server_binaries(extra_paths: Iterable[str] = (), runtimes_dir: Path | None = None) -> list[ServerBinary]:
    """Locate ``llama-server`` builds from settings, the environment, and ``runtimes/``."""
    candidates: list[str] = []
    env = os.environ.get("LOCAL_AI_LLAMA_SERVER")
    if env:
        candidates += [p for p in env.split(os.pathsep) if p.strip()]
    candidates += [p for p in extra_paths if p and p.strip()]

    paths: list[Path] = []
    for raw in candidates:
        path = Path(raw.strip().strip('"')).expanduser()
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths += _find_in_folder(path)

    root = runtimes_dir if runtimes_dir is not None else RUNTIMES_DIR
    if root.is_dir():
        for child in sorted(p for p in root.iterdir() if p.is_dir()):
            paths += _find_in_folder(child)

    binaries: list[ServerBinary] = []
    seen: set[str] = set()
    for path in paths:
        resolved = str(path.resolve())
        key = resolved.lower() if sys.platform == "win32" else resolved
        if key in seen:
            continue
        seen.add(key)
        binaries.append(ServerBinary(resolved, guess_runtime(resolved)))
    return binaries


# ---------------------------------------------------------------------- devices
@dataclass
class GpuDevice:
    id: str  # e.g. "Vulkan0" — what ``--device`` expects
    runtime: str  # e.g. "vulkan"
    name: str
    total_mib: int
    free_mib: int
    binary: ServerBinary
    physical_key: str = ""
    integrated: bool = False

    @property
    def key(self) -> str:
        return f"{self.runtime}:{self.id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "id": self.id,
            "runtime": self.runtime,
            "name": self.name,
            "total_mib": self.total_mib,
            "free_mib": self.free_mib,
            "physical_key": self.physical_key,
            "integrated": self.integrated,
            "binary": self.binary.path,
        }


def runtime_from_device_id(device_id: str) -> str:
    base = re.sub(r"\d+$", "", device_id).lower()
    return "rocm" if base in ("hip", "rocm") else base or "unknown"


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.replace("(R)", "").replace("(TM)", "")).strip().lower()


def is_integrated(name: str) -> bool:
    """Best-effort check for integrated GPUs (auto placement skips them)."""
    n = normalize_name(name)
    if "intel" in n:
        # Discrete Arc cards carry a model number (A770, B580, Pro B60...).
        return not re.search(r"\b[ab]\d{2,3}\b", n)
    if "amd" in n or "radeon" in n:
        return bool(re.search(r"radeon (?:\(tm\) )?graphics$|\b\d{3}m\b|vega \d+ graphics", n))
    return False


def parse_list_devices(text: str, binary: ServerBinary) -> list[GpuDevice]:
    devices: list[GpuDevice] = []
    for line in text.splitlines():
        match = _DEVICE_LINE.match(line)
        if not match:
            continue
        dev_id = match.group("id")
        runtime = runtime_from_device_id(dev_id)
        if runtime == "cpu":
            continue
        name = match.group("name").strip()
        devices.append(
            GpuDevice(
                id=dev_id,
                runtime=runtime,
                name=name,
                total_mib=int(match.group("total")),
                free_mib=int(match.group("free")),
                binary=binary,
                integrated=is_integrated(name),
            )
        )
    return devices


def assign_physical_keys(devices: list[GpuDevice]) -> None:
    """Give each device a key shared by the same card across runtimes."""
    by_runtime: dict[str, list[GpuDevice]] = {}
    for device in devices:
        by_runtime.setdefault(device.runtime, []).append(device)
    for group in by_runtime.values():
        counts: dict[str, int] = {}
        for device in group:
            base = normalize_name(device.name)
            n = counts.get(base, 0)
            counts[base] = n + 1
            # Two identical cards in one runtime get #0, #1 suffixes.
            device.physical_key = base if n == 0 else f"{base}#{n}"
        # Retro-fit the first duplicate so both carry a suffix.
        for base, total in counts.items():
            if total > 1:
                for device in group:
                    if device.physical_key == base:
                        device.physical_key = f"{base}#0"


Runner = Callable[[list[str], float], tuple[int, str]]


def _default_runner(cmd: list[str], timeout: float) -> tuple[int, str]:
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        stdin=subprocess.DEVNULL,
        **kwargs,
    )
    return result.returncode, (result.stdout or "") + "\n" + (result.stderr or "")


class DeviceCatalog:
    """Caches the devices each installed ``llama-server`` build can see."""

    def __init__(
        self,
        binaries_provider: Callable[[], list[ServerBinary]],
        runner: Optional[Runner] = None,
        max_age_seconds: float = 15.0,
    ) -> None:
        self._binaries_provider = binaries_provider
        self._runner = runner or _default_runner
        self._max_age = max_age_seconds
        self._lock = threading.Lock()
        self._devices: list[GpuDevice] = []
        self._binaries: list[ServerBinary] = []
        self._errors: list[dict[str, str]] = []
        self._refreshed_at = 0.0

    def binaries(self) -> list[ServerBinary]:
        try:
            return self._binaries_provider()
        except Exception:  # noqa: BLE001 - discovery must never crash the app
            return []

    def refresh(self) -> list[GpuDevice]:
        binaries = self.binaries()
        devices: list[GpuDevice] = []
        errors: list[dict[str, str]] = []
        for binary in binaries:
            try:
                code, output = self._runner(binary.command() + ["--list-devices"], 60.0)
            except subprocess.TimeoutExpired:
                errors.append({"binary": binary.path, "error": "Timed out listing devices."})
                continue
            except OSError as exc:
                errors.append({"binary": binary.path, "error": str(exc)})
                continue
            found = parse_list_devices(output, binary)
            if code != 0 and not found:
                tail = " ".join(output.strip().splitlines()[-3:])
                errors.append({"binary": binary.path, "error": f"Exit code {code}. {tail}".strip()})
            if binary.runtime == "unknown" and found:
                runtime = found[0].runtime
                binary = ServerBinary(binary.path, runtime)
                for device in found:
                    device.binary = binary
            devices += found
        assign_physical_keys(devices)
        with self._lock:
            self._devices = devices
            self._binaries = binaries
            self._errors = errors
            self._refreshed_at = time.monotonic()
        return list(devices)

    def devices(self, refresh: bool = False, max_age: Optional[float] = None) -> list[GpuDevice]:
        age_limit = self._max_age if max_age is None else max_age
        with self._lock:
            fresh = self._refreshed_at and (time.monotonic() - self._refreshed_at) < age_limit
            cached = list(self._devices)
        if refresh or not fresh:
            return self.refresh()
        return cached

    def cached_devices(self) -> list[GpuDevice]:
        """Last scan result without triggering a new scan (cheap; for status polling)."""
        with self._lock:
            return list(self._devices)

    def errors(self) -> list[dict[str, str]]:
        with self._lock:
            return list(self._errors)




# -------------------------------------------------------------------- placement
# User-facing GPU modes (per model):
#   auto   - best free discrete GPU that holds the model; split if none does; CPU if no GPU
#   gpu    - "Single GPU": one chosen card, all layers on it
#   split  - "Multi-GPU": one model with layers on several cards (one runtime, Vulkan
#            for NVIDIA + AMD); falls back to a single GPU when only one is available
#   cpu    - no GPU at all
MODE_ALIASES = {"single": "gpu", "multi": "split", "multi-gpu": "split", "single-gpu": "gpu"}


@dataclass
class Placement:
    mode: str  # "gpu" | "split" | "cpu"
    binary: ServerBinary
    devices: list[GpuDevice] = field(default_factory=list)
    tensor_split: Optional[list[float]] = None
    note: Optional[str] = None
    requested_mode: str = "auto"
    # True when the model is known not to fit and llama.cpp should keep some layers on
    # the CPU (auto mode only). Otherwise every layer is forced onto the chosen GPU(s).
    partial_offload: bool = False

    @property
    def runtime(self) -> str:
        return self.devices[0].runtime if self.devices else self.binary.runtime

    def physical_keys(self) -> list[str]:
        return [d.physical_key for d in self.devices] or ["cpu"]

    def label(self) -> str:
        if not self.devices:
            return "CPU"
        return " + ".join(d.name for d in self.devices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "requested_mode": self.requested_mode,
            "runtime": self.runtime,
            "binary": self.binary.path,
            "devices": [d.key for d in self.devices],
            "device_ids": [d.id for d in self.devices],
            "device_names": [d.name for d in self.devices],
            "physical_devices": self.physical_keys(),
            "tensor_split": self.tensor_split,
            "partial_offload": self.partial_offload,
            "note": self.note,
        }

    def remembered(self) -> dict[str, Any]:
        """Compact form saved in the model's profile (survives index changes)."""
        if self.requested_mode == "cpu":
            return {"mode": "cpu"}
        if self.requested_mode == "auto":
            return {"mode": "auto"}
        return {
            "mode": self.requested_mode,
            "devices": [{"key": d.key, "name": d.name} for d in self.devices],
            "tensor_split": self.tensor_split if self.requested_mode == "split" else None,
        }


def normalize_request(request: Any) -> dict[str, Any]:
    """Accept ``"auto"``, ``"cpu"``, ``"split"``/``"multi"``, a device key, or a dict."""
    if request is None or request == "" or request == "auto":
        return {"mode": "auto", "devices": [], "tensor_split": None}
    if isinstance(request, str):
        mode = MODE_ALIASES.get(request, request)
        if mode in ("cpu", "split", "gpu"):
            return {"mode": mode, "devices": [], "tensor_split": None}
        return {"mode": "gpu", "devices": [{"key": request}], "tensor_split": None}
    if isinstance(request, dict):
        req = dict(request)
        mode = req.get("mode") or ("gpu" if req.get("device") or req.get("devices") else "auto")
        mode = MODE_ALIASES.get(mode, mode)
        if mode not in ("auto", "gpu", "split", "cpu"):
            raise PlacementError(f"Unknown GPU mode '{mode}'. Use auto, single, multi or cpu.")
        devices = req.get("devices")
        if devices is None and req.get("device"):
            devices = [req["device"]]
        norm_devices = []
        for item in devices or []:
            if isinstance(item, str):
                norm_devices.append({"key": item})
            elif isinstance(item, dict):
                norm_devices.append({"key": item.get("key"), "name": item.get("name")})
        split = req.get("tensor_split")
        return {"mode": mode, "devices": norm_devices, "tensor_split": list(split) if split else None}
    raise PlacementError(f"Unrecognized device selection: {request!r}")


def _lookup(spec: dict[str, Any], devices: list[GpuDevice]) -> Optional[GpuDevice]:
    """Find a requested device. Integrated GPUs are only returned when named explicitly.

    Device indices shift when the eGPU is plugged in or removed (``Vulkan1`` is the RTX
    4070 with the eGPU attached and the Intel iGPU without it), so the name wins over
    the key, and a key-only request never silently lands on an integrated GPU.
    """
    key = spec.get("key")
    name = spec.get("name")
    runtime = key.split(":", 1)[0] if key and ":" in key else None
    if name:
        same = [d for d in devices if d.name == name and (runtime is None or d.runtime == runtime)]
        if not same:
            same = [d for d in devices if normalize_name(d.name) == normalize_name(name)]
        if same:
            exact = [d for d in same if d.key == key]
            return (exact or same)[0]
        return None
    if key:
        for device in devices:
            if device.key == key:
                return None if device.integrated else device
    return None


def _pick_binary_for_cpu(binaries: list[ServerBinary]) -> ServerBinary:
    for runtime in ("cpu", "vulkan", "cuda"):
        for binary in binaries:
            if binary.runtime == runtime:
                return binary
    return binaries[0]


def _preferred_entries(devices: list[GpuDevice]) -> list[GpuDevice]:
    """One entry per physical card, using the fastest runtime that sees it."""
    best: dict[str, GpuDevice] = {}
    for device in devices:
        current = best.get(device.physical_key)
        if current is None or _rank(device.runtime) < _rank(current.runtime):
            best[device.physical_key] = device
    return list(best.values())


def _preferred_for(device: GpuDevice, devices: list[GpuDevice]) -> GpuDevice:
    """The fastest runtime's entry for the same physical card (CUDA over Vulkan for NVIDIA)."""
    same = [d for d in devices if d.physical_key == device.physical_key]
    return min(same, key=lambda d: _rank(d.runtime)) if same else device


def _rank(runtime: str) -> int:
    return RUNTIME_PREFERENCE.index(runtime) if runtime in RUNTIME_PREFERENCE else len(RUNTIME_PREFERENCE)


def _binary_groups(devices: list[GpuDevice]) -> dict[str, list[GpuDevice]]:
    groups: dict[str, list[GpuDevice]] = {}
    for device in devices:
        groups.setdefault(device.binary.path, []).append(device)
    return groups


def _split_group(devices: list[GpuDevice], need_mib: float) -> Optional[list[GpuDevice]]:
    """The single runtime whose discrete devices together hold the most memory."""
    best: Optional[list[GpuDevice]] = None
    for group in _binary_groups(devices).values():
        unique = list({d.physical_key: d for d in group if not d.integrated}.values())
        if len(unique) < 2:
            continue
        if best is None or sum(d.total_mib for d in unique) > sum(d.total_mib for d in best):
            best = unique
    if best and sum(d.total_mib for d in best) >= need_mib:
        return sorted(best, key=lambda d: -d.total_mib)
    return None


def required_mib(model_size_bytes: Optional[int]) -> float:
    if not model_size_bytes:
        return 0.0
    return model_size_bytes / (1024 * 1024) * _WEIGHT_FACTOR + _OVERHEAD_MIB


def default_split(devices: list[GpuDevice]) -> list[float]:
    """Default Multi-GPU ratio: proportional to each card's total VRAM."""
    return [float(d.total_mib) for d in devices]


def resolve_placement(
    request: Any,
    devices: list[GpuDevice],
    binaries: list[ServerBinary],
    model_size_bytes: Optional[int] = None,
    busy: Iterable[str] = (),
) -> Placement:
    if not binaries:
        raise PlacementError(
            "No llama.cpp runtime is installed. Run scripts\\install-llama-runtimes.ps1 "
            "or set its path under Settings → GPU runtimes."
        )
    req = normalize_request(request)
    mode = req["mode"]
    busy_keys = set(busy)
    need = required_mib(model_size_bytes)

    if mode == "cpu":
        return Placement("cpu", _pick_binary_for_cpu(binaries), requested_mode="cpu")

    if mode == "gpu":
        if req["devices"]:
            found = _lookup(req["devices"][0], devices)
            if found is not None:
                return Placement("gpu", found.binary, [found], requested_mode="gpu")
        placement = _auto(devices, binaries, need, busy_keys, single_only=True)
        placement.requested_mode = "gpu"
        if req["devices"]:
            wanted = req["devices"][0].get("name") or req["devices"][0].get("key")
            placement.note = (
                f"The selected GPU '{wanted}' is not available (eGPU disconnected?). "
                f"Loaded on {placement.label()} instead."
            )
        return placement

    if mode == "split":
        return _multi(req, devices, binaries, need, busy_keys)

    placement = _auto(devices, binaries, need, busy_keys)
    placement.requested_mode = "auto"
    return placement


def _multi(
    req: dict[str, Any], devices: list[GpuDevice], binaries: list[ServerBinary], need: float, busy_keys: set[str]
) -> Placement:
    chosen: list[GpuDevice] = []
    missing: list[str] = []
    if req["devices"]:
        for spec in req["devices"]:
            found = _lookup(spec, devices)
            if found is None:
                missing.append(spec.get("name") or spec.get("key") or "?")
            elif found.physical_key not in {d.physical_key for d in chosen}:
                chosen.append(found)
        # Every chosen card must be visible to one runtime (Vulkan sees NVIDIA + AMD).
        if len(chosen) > 1 and len({d.binary.path for d in chosen}) != 1:
            keys = [d.physical_key for d in chosen]
            for group in _binary_groups(devices).values():
                by_key = {d.physical_key: d for d in group}
                if all(k in by_key for k in keys):
                    chosen = [by_key[k] for k in keys]
                    break
            else:
                raise PlacementError(
                    "Multi-GPU needs one runtime that sees every chosen GPU "
                    "(the Vulkan build sees both NVIDIA and AMD cards)."
                )
    else:
        chosen = _split_group(devices, 0) or []

    if len(chosen) >= 2:
        split = req.get("tensor_split")
        if split and len(split) == len(chosen) and not missing:
            ratio = [float(x) for x in split]
        else:
            ratio = default_split(chosen)
        if any(x < 0 for x in ratio) or sum(ratio) <= 0:
            raise PlacementError("The Multi-GPU split ratio must be positive numbers.")
        note = None
        if missing:
            note = f"GPU {', '.join(missing)} is not available; splitting across the remaining GPUs."
        return Placement("split", chosen[0].binary, chosen, ratio, note=note, requested_mode="split")

    # Only one (or no) GPU available: fall back instead of failing.
    if len(chosen) == 1:
        single = _preferred_for(chosen[0], devices)
        reason = f"{', '.join(missing)} is not available" if missing else "only one GPU is available"
        return Placement(
            "gpu",
            single.binary,
            [single],
            note=f"Multi-GPU fell back to {single.name} alone ({reason}).",
            requested_mode="split",
        )
    placement = _auto(devices, binaries, need, busy_keys, single_only=True)
    placement.requested_mode = "split"
    placement.note = (
        f"Multi-GPU fell back to {placement.label()} alone (only one GPU is available)."
        if placement.devices
        else "Multi-GPU fell back to the CPU (no GPU is available)."
    )
    return placement


def _auto(
    devices: list[GpuDevice],
    binaries: list[ServerBinary],
    need: float,
    busy_keys: set[str],
    single_only: bool = False,
) -> Placement:
    """Prefer a free, discrete card that holds the whole model; never an integrated GPU."""
    candidates = [d for d in _preferred_entries(devices) if not d.integrated]
    if not candidates:
        return Placement("cpu", _pick_binary_for_cpu(binaries), note="No discrete GPU detected; running on the CPU.")
    candidates.sort(key=lambda d: (d.physical_key in busy_keys, -d.total_mib, _rank(d.runtime)))
    for device in candidates:
        if device.total_mib >= need:
            return Placement("gpu", device.binary, [device])
    if not single_only:
        group = _split_group(devices, need)
        if group:
            return Placement(
                "split",
                group[0].binary,
                group,
                default_split(group),
                note="The model is larger than any single GPU, so it is split across GPUs.",
            )
    biggest = max(candidates, key=lambda d: d.total_mib)
    return Placement(
        "gpu",
        biggest.binary,
        [biggest],
        note="The model is larger than the GPU's memory; some layers will run on the CPU.",
        partial_offload=True,
    )
