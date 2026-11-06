"""Real-hardware check of GPU placement (sections 1 and 2 of the takeover brief).

Loads a model through the app's own runtime code (LlamaServerAdapter + placement +
offload parsing + Windows GPU telemetry), measures VRAM per GPU before/after, runs one
generation, prints a report, and unloads. Nothing in models/ or the app database is
touched; llama-server logs go to data/logs/verify/.

Usage (from the project root):
    python scripts/verify_gpu.py --model models/DeepSeek-R1-Distill-Qwen-1.5B-f16.gguf --mode single --gpu "RTX 4070"
    python scripts/verify_gpu.py --model ... --mode multi [--split 2,1]
    python scripts/verify_gpu.py --model ... --mode auto|cpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.models.adapters.llama_server_adapter import LlamaServerAdapter  # noqa: E402
from app.models.devices import DeviceCatalog, find_server_binaries  # noqa: E402
from app.models.gpu_telemetry import GpuTelemetry  # noqa: E402


def adapters_summary(snap: dict) -> dict:
    return {
        a["name"]: {"dedicated_used_mib": a["dedicated_used_mib"], "shared_used_mib": a["shared_used_mib"]}
        for a in snap.get("adapters", [])
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", default="single", choices=["auto", "single", "multi", "cpu"])
    parser.add_argument("--gpu", default="RTX 4070", help="substring of the GPU name for --mode single")
    parser.add_argument("--split", default=None, help="Multi-GPU ratio, e.g. 2,1")
    parser.add_argument("--devices", default=None, help="Multi-GPU: comma-separated name substrings, e.g. 'RTX 4070,9060'")
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--load-mode", default="auto", choices=["auto", "mmap", "none"])
    parser.add_argument("--prompt", default="Write a Python function that reverses a string. Answer briefly.")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    model_path = (ROOT / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model)
    telemetry = GpuTelemetry()
    catalog = DeviceCatalog(lambda: find_server_binaries())
    adapter = LlamaServerAdapter(catalog, ROOT / "data" / "logs" / "verify", telemetry=telemetry)

    devices = catalog.devices(refresh=True)
    print("Devices:")
    for d in devices:
        print(f"  {d.key:16} {d.name} ({d.total_mib} MiB){' [integrated]' if d.integrated else ''}")

    if args.mode == "single":
        match = [d for d in devices if args.gpu.lower() in d.name.lower()]
        if not match:
            print(f"No GPU matching '{args.gpu}'.")
            return 2
        placement = {"mode": "single", "devices": [{"name": match[0].name}]}
    elif args.mode == "multi":
        placement = {"mode": "multi"}
        if args.devices:
            wanted = [w.strip().lower() for w in args.devices.split(",")]
            names = []
            for w in wanted:
                hit = next((d.name for d in devices if w in d.name.lower() and d.runtime == "vulkan"), None)
                names.append({"name": hit or w})
            placement["devices"] = names
        if args.split:
            placement["tensor_split"] = [float(x) for x in args.split.split(",")]
    else:
        placement = args.mode

    telemetry.snapshot({})
    time.sleep(1.0)
    before = telemetry.snapshot({})
    settings = {
        "path": str(model_path),
        "size_bytes": model_path.stat().st_size,
        "placement": placement,
        "n_ctx": args.ctx,
        "n_gpu_layers": 0,  # the saved value from the bug report: must still mean "all layers"
        "n_batch": 512,
        "load_mode": args.load_mode,
    }
    started = time.monotonic()
    status = adapter.load_model("verify", settings)
    load_s = time.monotonic() - started
    try:
        time.sleep(1.5)
        telemetry.snapshot(adapter.process_ids())
        time.sleep(1.0)
        loaded = telemetry.snapshot(adapter.process_ids())
        reply = []
        gen_start = time.monotonic()
        for event in adapter.stream_events(
            "verify", [{"role": "user", "content": args.prompt}], {"max_tokens": args.max_tokens, "temperature": 0.2}
        ):
            if event["type"] in ("content", "reasoning"):
                reply.append(event["text"])
            if event["type"] == "done":
                timings = event["timings"]
        gen_s = time.monotonic() - gen_start
        during = telemetry.snapshot(adapter.process_ids())
        status = adapter.get_model_status("verify")
    finally:
        adapter.unload_all()
    time.sleep(1.5)
    telemetry.snapshot({})
    time.sleep(1.0)
    after = telemetry.snapshot({})

    report = {
        "model": model_path.name,
        "requested": placement,
        "command": status.get("command") or adapter_command(status),
        "placement": status["placement"],
        "placement_note": status.get("placement_note"),
        "placement_check": status["placement_check"],
        "offload": status["offload"],
        "llama_server_vram_per_adapter": status.get("vram"),
        "adapters_before": adapters_summary(before),
        "adapters_loaded": adapters_summary(loaded),
        "adapters_after_unload": adapters_summary(after),
        "nvidia_smi_loaded": [a.get("nvidia_smi") for a in loaded.get("adapters", []) if a.get("nvidia_smi")],
        "load_seconds": round(load_s, 1),
        "generation_seconds": round(gen_s, 1),
        "timings": timings,
        "reply_preview": "".join(reply)[:300],
        "busiest_engine_during_generation": {
            a["name"]: a.get("utilization") for a in during.get("adapters", [])
        },
    }
    print(json.dumps(report, indent=2))
    out = ROOT / "data" / "logs" / "verify" / f"verify-{args.mode}-{int(time.time())}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


def adapter_command(status: dict) -> list:
    return status.get("config", {}).get("command", [])


if __name__ == "__main__":
    sys.exit(main())
