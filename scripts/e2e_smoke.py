"""End-to-end smoke test against a running backend with a real GGUF model.

Usage (backend must be running on 127.0.0.1:8765):

    python scripts/e2e_smoke.py
"""
from __future__ import annotations

import json
import sys

import httpx

BASE = "http://127.0.0.1:8765"


def main() -> int:
    health = httpx.get(f"{BASE}/api/health", timeout=5)
    if health.status_code != 200:
        print("Backend not reachable at", BASE)
        return 1

    models = httpx.get(f"{BASE}/api/models", timeout=10).json()
    if not models:
        print("No registered models. Register some first.")
        return 1

    # Prefer the smallest model for a quick smoke test.
    model = min((m for m in models if m.get("size_bytes")), key=lambda m: m["size_bytes"])
    model_id = model["id"]
    print(f"Using model: {model['name']} ({model['quant']}, {model['size_bytes']} bytes)")

    load = httpx.post(
        f"{BASE}/api/models/{model_id}/load", json={"n_ctx": 1024}, timeout=120
    )
    print("Load:", load.json())

    conv = httpx.post(f"{BASE}/api/conversations", json={"title": "e2e smoke"}).json()
    conversation_id = conv["id"]

    assembled = ""
    with httpx.stream(
        "POST",
        f"{BASE}/api/chat/stream",
        json={
            "conversation_id": conversation_id,
            "model_id": model_id,
            "content": "Reply with exactly one word.",
            "options": {"max_tokens": 64, "temperature": 0},
        },
        timeout=180,
    ) as resp:
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event["type"] == "delta":
                assembled += event["text"]
            elif event["type"] == "error":
                print("ERROR:", event.get("message"))
                return 1

    print("Assistant:", repr(assembled[:200]))

    detail = httpx.get(f"{BASE}/api/conversations/{conversation_id}", timeout=10).json()
    roles = [m["role"] for m in detail["messages"]]
    print("Persisted messages:", roles)

    httpx.post(f"{BASE}/api/models/{model_id}/unload", timeout=60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
