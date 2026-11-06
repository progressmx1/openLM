"""Tool framework primitives: result, definition, and execution context."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from pydantic import BaseModel

from ..security.audit import AuditLog
from ..security.permissions import PermissionLevel
from ..security.workspace import WorkspacePolicy, WorkspaceViolation, canonicalize


class ToolResult:
    """Structured outcome of a tool execution."""

    def __init__(
        self,
        status: str,
        output: str = "",
        data: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        # status: success | error | denied | cancelled | timeout
        self.status = status
        self.output = output
        self.data = data or {}
        self.error = error

    @property
    def ok(self) -> bool:
        return self.status == "success"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "output": self.output,
            "data": self.data,
            "error": self.error,
        }


@dataclass
class ExecutionContext:
    workspace: WorkspacePolicy
    audit: AuditLog
    approved_workspaces: list[str] = field(default_factory=list)
    # Paths explicitly approved for this single execution (out-of-workspace targets
    # that the user confirmed). Handlers use :meth:`resolve_path` to enforce both
    # workspace membership and these one-off approvals.
    approved_paths: list[str] = field(default_factory=list)
    # Whether internet access is enabled (gate for web tools).
    internet_enabled: bool = False

    def resolve_path(self, path: str) -> str:
        """Canonicalize ``path`` and verify it is allowed (workspace or approved).

        Raises :class:`WorkspaceViolation` otherwise. Returns the canonical path.
        """
        target = canonicalize(path)
        if self.workspace.is_within(target):
            return str(target)
        approved = {canonicalize(p) for p in self.approved_paths}
        if target in approved:
            return str(target)
        raise WorkspaceViolation(
            f"Path '{target}' is outside all approved workspaces and was not explicitly approved."
        )


@dataclass
class Tool:
    name: str
    description: str
    permission_level: PermissionLevel
    input_model: type[BaseModel]
    # Names of input fields that contain filesystem paths (for boundary checks).
    path_arg_names: tuple[str, ...] = ()
    # Optional function to compute an effective (possibly escalated) level from
    # validated arguments (e.g. destructive-command detection).
    dynamic_level_fn: Optional[Callable[[BaseModel], PermissionLevel]] = None
    # handler(ctx, validated_input) -> ToolResult
    handler: Callable[[ExecutionContext, BaseModel], ToolResult] = lambda ctx, args: ToolResult(
        "error", error="unimplemented"
    )

    def validate(self, args: dict[str, Any]) -> BaseModel:
        return self.input_model.model_validate(args)

    def effective_level(self, validated: BaseModel) -> PermissionLevel:
        if self.dynamic_level_fn is not None:
            return self.dynamic_level_fn(validated)
        return self.permission_level

    def target_paths(self, validated: BaseModel) -> list[str]:
        paths: list[str] = []
        for name in self.path_arg_names:
            value = getattr(validated, name, None)
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                paths.extend(str(v) for v in value)
            else:
                paths.append(str(value))
        return paths

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "permission_level": self.permission_level.value,
            "schema": self.input_model.model_json_schema(),
        }
