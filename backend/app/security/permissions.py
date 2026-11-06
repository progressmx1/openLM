"""Permission classification and approval decisions.

Four levels (mirroring the product spec):

1. ``READ_ONLY``      — list/search/read within approved locations.
2. ``WORKSPACE_WRITE`` — create/edit/copy/move/rename inside approved workspaces.
3. ``SYSTEM_ACTION``   — run commands, launch apps, or touch paths outside workspaces.
4. ``HIGH_IMPACT``     — delete/overwrite important files, modify system state, etc.

A model-generated request is *never* user approval. The decision engine only
auto-allows low-risk work strictly inside approved workspaces; everything else
requires explicit approval, and high-impact actions always require a separate
confirmation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .workspace import WorkspacePolicy


class PermissionLevel(str, Enum):
    READ_ONLY = "read_only"
    WORKSPACE_WRITE = "workspace_write"
    SYSTEM_ACTION = "system_action"
    HIGH_IMPACT = "high_impact"


# Levels that may run without approval when the target is inside an approved workspace.

@dataclass
class PermissionDecision:
    allowed: bool
    level: PermissionLevel
    requires_approval: bool
    reason: str
    auto: bool = False


@dataclass
class ToolContext:
    """Facts needed to classify a tool invocation."""

    tool_name: str
    level: PermissionLevel
    target_paths: list[str] = field(default_factory=list)
    within_workspace: bool = False
    # True for tools that are safe to auto-run *only when* the target is inside
    # an approved workspace (read/write/open/reveal). False for system commands.
    auto_allowed_in_workspace: bool = False
    summary: str = ""


class PermissionManager:
    def __init__(self, workspace_policy: WorkspacePolicy) -> None:
        self.workspace = workspace_policy

    def evaluate(self, context: ToolContext) -> PermissionDecision:
        level = context.level

        if level == PermissionLevel.HIGH_IMPACT:
            return PermissionDecision(
                allowed=False,
                level=level,
                requires_approval=True,
                reason="High-impact action requires explicit confirmation.",
            )

        if context.auto_allowed_in_workspace and context.within_workspace:
            return PermissionDecision(
                allowed=True,
                level=level,
                requires_approval=False,
                auto=True,
                reason="Action is inside an approved workspace.",
            )

        return PermissionDecision(
            allowed=False,
            level=level,
            requires_approval=True,
            reason="Action requires explicit approval"
            + (" (target is outside all approved workspaces)." if not context.within_workspace else "."),
        )

    def is_within_workspace(self, paths: list[str]) -> bool:
        if not paths:
            return False
        return all(self.workspace.is_within(p) for p in paths if p)
