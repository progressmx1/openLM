"""Tests for workspace boundaries and the permission decision engine."""
from __future__ import annotations

import pytest

from app.security.permissions import PermissionLevel, PermissionManager, ToolContext
from app.security.workspace import WorkspacePolicy, WorkspaceViolation


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


def test_within_workspace(workspace) -> None:
    policy = WorkspacePolicy([str(workspace)])
    assert policy.is_within(workspace / "file.txt") is True
    assert policy.is_within(workspace / "sub" / "deep" / "file.txt") is True
    assert policy.is_within(workspace) is True


def test_outside_workspace(workspace, tmp_path) -> None:
    policy = WorkspacePolicy([str(workspace)])
    assert policy.is_within(tmp_path / "other.txt") is False


def test_path_traversal_blocked(workspace) -> None:
    policy = WorkspacePolicy([str(workspace)])
    with pytest.raises(WorkspaceViolation):
        policy.resolve_in_workspace(workspace, "..", "escape.txt")
    # A path literally containing ".." resolves outside -> blocked.
    assert policy.is_within(workspace / ".." / "escape.txt") is False


def test_symlink_escape_blocked(workspace, tmp_path) -> None:
    try:
        outside = tmp_path / "outside"
        outside.mkdir()
        link = workspace / "link"
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks not permitted on this system")

    policy = WorkspacePolicy([str(workspace)])
    assert policy.is_within(str(link)) is False
    with pytest.raises(WorkspaceViolation):
        policy.assert_within(str(link))


def test_resolve_in_workspace_ok(workspace) -> None:
    policy = WorkspacePolicy([str(workspace)])
    resolved = policy.resolve_in_workspace(workspace, "sub", "file.txt")
    assert resolved == (workspace / "sub" / "file.txt").resolve()


def test_no_workspaces_means_everything_outside(workspace) -> None:
    policy = WorkspacePolicy([])
    assert policy.is_within(workspace / "x.txt") is False
    with pytest.raises(WorkspaceViolation):
        policy.assert_within(workspace / "x.txt")


# ----------------------------------------------------------------- permissions


def test_read_inside_workspace_auto_allowed(workspace) -> None:
    pm = PermissionManager(WorkspacePolicy([str(workspace)]))
    decision = pm.evaluate(ToolContext(
        tool_name="read_file",
        level=PermissionLevel.READ_ONLY,
        target_paths=[str(workspace / "a.txt")],
        within_workspace=True,
        auto_allowed_in_workspace=True,
    ))
    assert decision.requires_approval is False
    assert decision.allowed is True


def test_write_inside_workspace_auto_allowed(workspace) -> None:
    pm = PermissionManager(WorkspacePolicy([str(workspace)]))
    decision = pm.evaluate(ToolContext(
        tool_name="write_file",
        level=PermissionLevel.WORKSPACE_WRITE,
        target_paths=[str(workspace / "a.txt")],
        within_workspace=True,
        auto_allowed_in_workspace=True,
    ))
    assert decision.requires_approval is False


def test_write_outside_workspace_requires_approval(workspace, tmp_path) -> None:
    pm = PermissionManager(WorkspacePolicy([str(workspace)]))
    decision = pm.evaluate(ToolContext(
        tool_name="write_file",
        level=PermissionLevel.WORKSPACE_WRITE,
        target_paths=[str(tmp_path / "b.txt")],
        within_workspace=False,
        auto_allowed_in_workspace=True,
    ))
    assert decision.requires_approval is True


def test_system_action_requires_approval(workspace) -> None:
    pm = PermissionManager(WorkspacePolicy([str(workspace)]))
    decision = pm.evaluate(ToolContext(
        tool_name="run_command",
        level=PermissionLevel.SYSTEM_ACTION,
        target_paths=[],
        within_workspace=True,
        auto_allowed_in_workspace=False,
    ))
    assert decision.requires_approval is True


def test_high_impact_always_requires_approval(workspace) -> None:
    pm = PermissionManager(WorkspacePolicy([str(workspace)]))
    decision = pm.evaluate(ToolContext(
        tool_name="delete",
        level=PermissionLevel.HIGH_IMPACT,
        target_paths=[str(workspace / "c.txt")],
        within_workspace=True,
        auto_allowed_in_workspace=False,
    ))
    assert decision.requires_approval is True


def test_is_within_workspace_all_paths(workspace, tmp_path) -> None:
    pm = PermissionManager(WorkspacePolicy([str(workspace)]))
    assert pm.is_within_workspace([str(workspace / "a.txt"), str(workspace / "b.txt")]) is True
    assert pm.is_within_workspace([str(workspace / "a.txt"), str(tmp_path / "b.txt")]) is False
