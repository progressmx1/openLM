"""End-to-end API tests using the fake adapter (no real model)."""
from __future__ import annotations

import json


def test_health(client) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_diagnostics_and_runtimes(client) -> None:
    assert client.get("/api/system/diagnostics").status_code == 200
    runtime = client.get("/api/system/runtimes").json()
    assert runtime["name"] == "fake"


def test_model_discover_load_unload(client) -> None:
    result = client.post("/api/models/discover", json={"locations": ["x"]}).json()
    assert result["added"] == 1

    models = client.get("/api/models").json()
    assert len(models) == 1
    model_id = models[0]["id"]
    assert models[0]["loaded"] is False

    loaded = client.post(f"/api/models/{model_id}/load", json={"n_ctx": 2048}).json()
    assert loaded["loaded"] is True

    status = client.get(f"/api/models/{model_id}/status").json()
    assert status["loaded"] is True

    unloaded = client.post(f"/api/models/{model_id}/unload").json()
    assert unloaded["loaded"] is False


def test_model_not_found_404(client) -> None:
    assert client.get("/api/models/nope").status_code == 404
    assert client.post("/api/models/nope/load").status_code == 404


def test_conversation_lifecycle(client) -> None:
    conv = client.post("/api/conversations", json={"title": "A"}).json()
    conv_id = conv["id"]

    listed = client.get("/api/conversations").json()
    assert any(c["id"] == conv_id for c in listed)

    updated = client.patch(f"/api/conversations/{conv_id}", json={"title": "B", "pinned": True}).json()
    assert updated["title"] == "B"

    deleted = client.delete(f"/api/conversations/{conv_id}").json()
    assert deleted["deleted"] is True
    assert client.get(f"/api/conversations/{conv_id}").status_code == 404


def test_chat_stream_endpoint(client) -> None:
    conv = client.post("/api/conversations", json={"title": "chat"}).json()
    client.post("/api/models/discover", json={"locations": ["x"]})
    model_id = client.get("/api/models").json()[0]["id"]
    client.post(f"/api/models/{model_id}/load")

    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"conversation_id": conv["id"], "model_id": model_id, "content": "hi"},
    ) as response:
        assert response.status_code == 200
        lines = [line for line in response.iter_lines() if line.startswith("data: ")]

    events = [json.loads(line[6:]) for line in lines if line[6:].strip() != "[DONE]"]
    assert any(e["type"] == "delta" for e in events)
    assert any(e["type"] == "done" for e in events)

    detail = client.get(f"/api/conversations/{conv['id']}").json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]


def test_settings_roundtrip(client) -> None:
    current = client.get("/api/settings").json()
    current["theme"] = "light"
    updated = client.put("/api/settings", json={"settings": current}).json()
    assert updated["theme"] == "light"


def test_devices_endpoint_and_placement_passthrough(client) -> None:
    body = client.get("/api/system/devices").json()
    assert body["devices"] == [] and "binaries" in body

    client.post("/api/models/discover", json={"locations": ["x"]})
    model_id = client.get("/api/models").json()[0]["id"]
    client.post(f"/api/models/{model_id}/load", json={"placement": "cpu", "n_ctx": 2048})
    model = client.get(f"/api/models/{model_id}").json()
    assert model["status"]["placement"] == "cpu"
    assert model["profile"] == {"placement": "cpu", "n_ctx": 2048}
