"""End-to-end check of the coding agent on a real local model, through the real HTTP API.

Starts the backend (with a throwaway data folder, so the app's real data/ is untouched),
loads a model on the chosen GPU, opens a scratch workspace, sends a task, streams the
agent's events, and plays the user: approves non-destructive actions (as clicking
"Approve" would) and denies destructive ones. Afterwards it re-runs the tests itself,
hard-kills the backend, and checks that no llama-server process was left behind.

    python scripts/agent_e2e.py --model Qwen2.5-Coder-7B --workspace <empty folder> [--mode auto]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
TASK = (
    "Create a small Python command-line tool `wordstats.py` that reads a text file given as an argument and prints "
    "the number of lines, words and characters, plus the 3 most common words (case-insensitive). Put the logic in "
    "functions so it can be tested. Write pytest tests in tests/test_wordstats.py covering the functions and the CLI, "
    "run the tests with `python -m pytest -q`, and fix any failures until all tests pass."
)


def llama_server_pids() -> set[int]:
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq llama-server.exe", "/FO", "CSV", "/NH"], capture_output=True, text=True).stdout
    pids = set()
    for line in out.splitlines():
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) > 1 and parts[1].isdigit():
            pids.add(int(parts[1]))
    return pids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen2.5-Coder-7B")
    parser.add_argument("--gpu", default="RTX 4070")
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--mode", default="auto")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--port", type=int, default=8799)
    parser.add_argument("--timeout", type=int, default=1500)
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    data_dir = Path(tempfile.mkdtemp(prefix="agent-e2e-data-"))
    env = dict(os.environ, LOCAL_AI_DATA_DIR=str(data_dir), LOCAL_AI_MODELS_DIR=str(ROOT / "models"))
    before_pids = llama_server_pids()
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(args.port)],
        cwd=ROOT / "backend", env=env, stdout=open(data_dir / "backend.out", "w"), stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{args.port}"
    client = httpx.Client(base_url=base, timeout=900, trust_env=False)
    report: dict = {"task": args.task, "workspace": str(workspace)}
    try:
        for _ in range(120):
            try:
                if client.get("/api/health").status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.5)
        client.post("/api/models/discover", json={"locations": [str(ROOT / "models")]}).raise_for_status()
        wanted = args.model.lower().replace(" ", "-")
        model = next(
            m for m in client.get("/api/models").json()
            if wanted in Path(m["path"]).name.lower() or wanted in m["name"].lower().replace(" ", "-")
        )
        devices = client.get("/api/system/devices", params={"refresh": True}).json()["devices"]
        gpu = next(d for d in devices if args.gpu.lower() in d["name"].lower())
        t0 = time.monotonic()
        status = client.post(
            f"/api/models/{model['id']}/load",
            json={"placement": {"mode": "single", "devices": [{"name": gpu["name"]}]}, "n_ctx": args.ctx},
        )
        status.raise_for_status()
        status = status.json()
        report["load"] = {
            "seconds": round(time.monotonic() - t0, 1),
            "placement_check": status["placement_check"]["message"],
            "vram": status.get("vram"),
            "native_tools": status.get("native_tools"),
            "n_ctx": status.get("n_ctx_effective"),
        }
        print("LOADED:", json.dumps(report["load"]))
        session = client.post("/api/agent/sessions", json={"workspace": str(workspace), "model_id": model["id"], "mode": args.mode}).json()
        sid = session["id"]
        client.post(f"/api/agent/sessions/{sid}/messages", json={"text": args.task}).raise_for_status()

        approvals = []
        tools_used = []
        usage = []
        deadline = time.monotonic() + args.timeout
        ended = None
        with client.stream("GET", f"/api/agent/sessions/{sid}/stream", params={"after": 0}, timeout=None) as stream:
            for line in stream.iter_lines():
                if not line.startswith("data:"):
                    if time.monotonic() > deadline:
                        break
                    continue
                event = json.loads(line[5:])
                kind = event["type"]
                if kind == "tool":
                    tools_used.append(event["name"])
                    label = event.get("command") or event.get("path") or json.dumps(event.get("args"))[:120]
                    print(f"  TOOL {event['name']}: {label} [{event['status']}]")
                    if event["status"] == "awaiting_approval":
                        approve = event.get("risk") != "destructive"
                        approvals.append({"tool": event["name"], "what": label, "risk": event.get("risk"), "approved": approve})
                        print(f"    -> {'APPROVE' if approve else 'DENY'} ({event.get('risk')})")
                        client.post(f"/api/agent/sessions/{sid}/decision", json={"call_id": event["call_id"], "approve": approve, "remember": approve and event.get("risk") == "command"})
                elif kind == "tool_update" and event["status"] not in ("running",):
                    first = (event.get("output") or "").strip().splitlines()[:1]
                    print(f"    = {event['status']}: {first[0][:150] if first else ''}")
                elif kind == "assistant":
                    print("  ASSISTANT:", (event.get("content") or "").strip()[:400].replace("\n", " | "))
                elif kind == "usage":
                    usage.append(event)
                elif kind in ("notice", "error"):
                    print(f"  {kind.upper()}: {event.get('text') or event.get('message')}")
                elif kind == "turn_end":
                    ended = event["reason"]
                    break
                if time.monotonic() > deadline:
                    client.post(f"/api/agent/sessions/{sid}/interrupt")
                    ended = "timeout"
                    break
        report.update({
            "turn_end": ended,
            "tool_calls": len(tools_used),
            "tools": tools_used,
            "approvals": approvals,
            "steps": len(usage),
            "tokens_per_second": [round(u["tokens_per_second"], 1) for u in usage if u.get("tokens_per_second")],
            "max_context_used": max((u.get("context_used") or 0) for u in usage) if usage else None,
            "changes": [(c["path"], c["added"], c["removed"]) for c in client.get(f"/api/agent/sessions/{sid}/changes").json()],
        })
        verify = subprocess.run([sys.executable, "-m", "pytest"], cwd=workspace, capture_output=True, text=True, timeout=300)
        report["independent_pytest"] = {"exit_code": verify.returncode, "tail": verify.stdout.strip().splitlines()[-3:]}
        tele = client.get("/api/system/gpu-telemetry").json()
        report["telemetry_during"] = [
            {"name": a["name"], "dedicated_used_mib": a["dedicated_used_mib"], "processes": a["processes"]} for a in tele.get("adapters", [])
        ]
    finally:
        loaded_pids = llama_server_pids() - before_pids
        backend.kill()  # hard kill: the Job Object must take llama-server down with it
        backend.wait(30)
        time.sleep(3)
        leftover = (llama_server_pids() - before_pids) & loaded_pids
        report["cleanup"] = {"llama_server_pids_while_running": sorted(loaded_pids), "left_after_backend_kill": sorted(leftover)}
    print(json.dumps(report, indent=2))
    out = ROOT / "data" / "logs" / "verify" / f"agent-e2e-{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved", out)
    return 0 if report.get("independent_pytest", {}).get("exit_code") == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
