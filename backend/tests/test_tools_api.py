"""API tests for the tool framework and approval flow."""
from __future__ import annotations

import os


def test_list_tools_api(client) -> None:
    response = client.get("/api/tools")
    assert response.status_code == 200
    names = {t["name"] for t in response.json()}
    assert "read_file" in names
    assert "run_command" in names


def test_execute_filesystem_via_api(client, approved_workspace) -> None:
    target = os.path.join(approved_workspace, "a.txt")
    response = client.post(
        "/api/tools/execute",
        json={"tool": "write_file", "args": {"path": target, "content": "hi"}},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_execute_outside_workspace_and_approve(client, approved_workspace, tmp_path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    response = client.post(
        "/api/tools/execute",
        json={"tool": "write_file", "args": {"path": str(outside / "x.txt"), "content": "hi"}},
    )
    data = response.json()
    assert data["status"] == "awaiting_approval"

    decided = client.post(
        f"/api/approvals/{data['approval_id']}/decide", json={"approved": True}
    )
    assert decided.json()["status"] == "success"
    assert (outside / "x.txt").read_text() == "hi"


def test_approval_list_and_deny(client, approved_workspace) -> None:
    target = os.path.join(approved_workspace, "f.txt")
    client.post("/api/tools/execute", json={"tool": "write_file", "args": {"path": target, "content": "x"}})

    pending = client.post("/api/tools/execute", json={"tool": "delete", "args": {"path": target}}).json()
    assert pending["status"] == "awaiting_approval"
    assert pending["level"] == "high_impact"

    approvals = client.get("/api/approvals").json()
    assert any(a["id"] == pending["approval_id"] for a in approvals)

    denied = client.post(f"/api/approvals/{pending['approval_id']}/decide", json={"approved": False}).json()
    assert denied["status"] == "denied"


def test_execute_unknown_tool_404(client) -> None:
    response = client.post("/api/tools/execute", json={"tool": "nope", "args": {}})
    assert response.status_code == 404


def test_execute_malformed_args_422(client) -> None:
    response = client.post("/api/tools/execute", json={"tool": "read_file", "args": {"path": 123}})
    assert response.status_code == 422


def test_tool_logs_recorded(client, approved_workspace) -> None:
    target = os.path.join(approved_workspace, "b.txt")
    client.post("/api/tools/execute", json={"tool": "write_file", "args": {"path": target, "content": "y"}})
    logs = client.get("/api/tools/logs").json()
    assert any(log["tool_name"] == "write_file" for log in logs)
