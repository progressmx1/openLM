"""Proof of where a model actually runs, parsed from the llama-server log.

llama-server is started with ``-lv 4`` so libllama's loader lines are written (at the
default verbosity 3 they are suppressed — which is why the app previously had no proof
of placement). Real lines from b11960 on this machine::

    llama_prepare_model_devices: using device CUDA0 (NVIDIA GeForce RTX 4070 Laptop GPU) (0000:01:00.0) - 7068 MiB free
    load_tensors: offloaded 29/29 layers to GPU
    load_tensors:   CPU_Mapped model buffer size =   445.12 MiB
    load_tensors:        CUDA0 model buffer size =  2944.68 MiB
    llama_kv_cache:      CUDA0 KV buffer size =   777.00 MiB
    sched_reserve:      CUDA0 compute buffer size =    92.26 MiB

``--fit`` runs a dry-run load first that prints the same lines with ``0.00 MiB``, so only
the *last* load block counts.

Per-layer device assignment is only logged at debug level (thousands of lines per
request), so layers per device are computed with llama.cpp's own rule (cumulative,
normalised ``tensor_split``; layer ``il`` goes to the first device whose cumulative share
exceeds ``il / n_gpu_layers``) and cross-checked by the measured buffer sizes.
"""
from __future__ import annotations

import bisect
import re
from typing import Any, Optional

_OFFLOADED = re.compile(r"load_tensors: offloaded (\d+)/(\d+) layers to GPU")
_BUFFER = re.compile(
    r"(load_tensors|llama_kv_cache|sched_reserve|llama_context):\s+(\S+?)\s+(model|KV|compute|output) buffer size =\s+([\d.]+) MiB"
)
_USING = re.compile(r"llama_prepare_model_devices: using device (\S+) \((.*)\) \([^()]*\) - \d+ MiB free")
_N_CTX = re.compile(r"llama_context: n_ctx\s+=\s+(\d+)")
_BLOCK_COUNT = re.compile(r"\.block_count\s+u32\s+=\s+(\d+)")
_LOAD_START = "load_tensors: loading model tensors"

_HOST_BUFFERS = ("cpu", "cpu_mapped", "cuda_host", "vulkan_host", "rocm_host", "cpu_repack")


def _is_host(name: str) -> bool:
    lowered = name.lower()
    return lowered in _HOST_BUFFERS or lowered.endswith("_host") or lowered.startswith("cpu")


def parse_offload_log(text: str) -> Optional[dict[str, Any]]:
    """Extract the final load's offload facts; ``None`` when the log has none yet."""
    starts = [m.start() for m in re.finditer(re.escape(_LOAD_START), text)]
    tail = text[starts[-1]:] if starts else text
    offloaded = list(_OFFLOADED.finditer(tail))
    if not offloaded:
        return None
    layers_on_gpu, layers_total = int(offloaded[-1].group(1)), int(offloaded[-1].group(2))

    devices: dict[str, dict[str, float]] = {}
    host: dict[str, float] = {"model_mib": 0.0, "kv_mib": 0.0, "compute_mib": 0.0}
    for match in _BUFFER.finditer(tail):
        name, kind, mib = match.group(2), match.group(3), float(match.group(4))
        field = {"model": "model_mib", "KV": "kv_mib", "compute": "compute_mib", "output": "compute_mib"}[kind]
        if _is_host(name):
            host[field] += mib
        else:
            entry = devices.setdefault(name, {"model_mib": 0.0, "kv_mib": 0.0, "compute_mib": 0.0})
            entry[field] += mib

    using = {}
    for match in _USING.finditer(text):
        using[match.group(1)] = match.group(2)
    n_ctx = _N_CTX.findall(tail)
    blocks = _BLOCK_COUNT.findall(text)
    return {
        "layers_on_gpu": layers_on_gpu,
        "layers_total": layers_total,
        "n_layer": int(blocks[-1]) if blocks else max(layers_total - 1, 0),
        "devices": {
            k: {**{f: round(v, 1) for f, v in d.items()}, "total_mib": round(sum(d.values()), 1), "name": using.get(k)}
            for k, d in devices.items()
        },
        "host": {**{f: round(v, 1) for f, v in host.items()}, "total_mib": round(sum(host.values()), 1)},
        "n_ctx": int(n_ctx[-1]) if n_ctx else None,
        "devices_used": using,
    }


def layers_per_device(n_gpu_layers: int, n_layer: int, split: list[float]) -> list[int]:
    """Replicates llama.cpp's layer→device assignment for ``--split-mode layer``."""
    if not split or n_gpu_layers <= 0:
        return [0] * len(split)
    total = sum(split)
    if total <= 0:
        split = [1.0] * len(split)
        total = float(len(split))
    cumulative = []
    acc = 0.0
    for value in split:
        acc += value
        cumulative.append(acc / total)
    act = min(n_gpu_layers, n_layer + 1)
    start = max(n_layer + 1 - n_gpu_layers, 0)
    counts = [0] * len(split)
    for il in range(start, n_layer + 1):
        idx = bisect.bisect_right(cumulative, float(il - start) / act)
        counts[min(idx, len(split) - 1)] += 1
    return counts


def evaluate(report: Optional[dict[str, Any]], placement: dict[str, Any], log_ready: bool = True) -> dict[str, Any]:
    """Compare what the user asked for with what llama.cpp did.

    Returns ``{"level": "ok"|"warning"|"error"|"unknown", "message": str, "per_device": [...]}``.
    """
    mode = placement.get("mode")
    ids = placement.get("device_ids") or []
    names = placement.get("device_names") or []
    split = placement.get("tensor_split") or ([1.0] if ids else [])
    if report is None:
        if mode == "cpu":
            return {"level": "ok", "message": "Running on the CPU (as selected).", "per_device": []}
        return {
            "level": "unknown",
            "message": "The runtime did not report layer placement, so GPU use could not be verified."
            if log_ready
            else "Waiting for the runtime to report layer placement…",
            "per_device": [],
        }

    on_gpu, total = report["layers_on_gpu"], report["layers_total"]
    counts = layers_per_device(on_gpu, report["n_layer"], split) if len(ids) > 1 else ([on_gpu] if ids else [])
    per_device = []
    for i, dev_id in enumerate(ids):
        measured = report["devices"].get(dev_id, {})
        per_device.append(
            {
                "id": dev_id,
                "name": names[i] if i < len(names) else measured.get("name") or dev_id,
                "layers": counts[i] if i < len(counts) else 0,
                "model_mib": measured.get("model_mib", 0.0),
                "kv_mib": measured.get("kv_mib", 0.0),
                "compute_mib": measured.get("compute_mib", 0.0),
                "total_mib": measured.get("total_mib", 0.0),
            }
        )
    unexpected = [d for d in report["devices"] if d not in ids and report["devices"][d]["total_mib"] > 0]
    where = " + ".join(names) or "GPU"

    if mode == "cpu":
        if on_gpu > 0 or unexpected:
            return {"level": "warning", "message": f"CPU was selected but {on_gpu}/{total} layers went to a GPU.", "per_device": per_device}
        return {"level": "ok", "message": "Running on the CPU (as selected).", "per_device": per_device}
    if unexpected:
        return {
            "level": "error",
            "message": f"Memory was allocated on an unexpected device ({', '.join(unexpected)}) instead of only {where}.",
            "per_device": per_device,
        }
    if on_gpu == 0:
        return {
            "level": "error",
            "message": f"0/{total} layers are on {where} — the model is running on the CPU. "
            "Reload with a lower context length or a smaller quantization.",
            "per_device": per_device,
        }
    if on_gpu < total:
        return {
            "level": "warning",
            "message": f"Only {on_gpu}/{total} layers fit on {where}; the rest run on the CPU (slower). "
            "Lower the context length, use a smaller quantization, or split across GPUs.",
            "per_device": per_device,
        }
    missing = [d["name"] for d in per_device if d["model_mib"] <= 0]
    if missing and len(per_device) > 1:
        return {
            "level": "warning",
            "message": f"All {total} layers are on the GPU, but nothing was placed on {', '.join(missing)}.",
            "per_device": per_device,
        }
    if len(per_device) > 1:
        detail = ", ".join(f"{d['layers']} on {d['name']}" for d in per_device)
        return {"level": "ok", "message": f"{on_gpu}/{total} layers on GPU ({detail}).", "per_device": per_device}
    return {"level": "ok", "message": f"{on_gpu}/{total} layers on {where}.", "per_device": per_device}
