"""Tool executor: validates, classifies, approves, executes, and audits tools.

Execution flow:

1. Validate arguments against the tool's Pydantic schema.
2. Compute the effective permission level (dynamic escalation for destructive
   commands / executables) and workspace containment.
3. Auto-run low-risk actions strictly inside approved workspaces; otherwise
   create a *pending approval* and do not execute until the user decides.
4. Record every request, decision, and execution in the audit log.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

from ..core.settings import SettingsStore
from ..security.audit import AuditLog
from ..security.permissions import (
    PermissionLevel,
    PermissionManager,
    ToolContext,
)
from ..security.workspace import WorkspacePolicy
from ..storage.repository import Repository
from ..tools.base import ExecutionContext, Tool, ToolResult
from ..tools.registry import ToolRegistry

_AUTO_LEVELS = {PermissionLevel.READ_ONLY, PermissionLevel.WORKSPACE_WRITE}


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        repository: Repository,
        audit: AuditLog,
        settings_store: SettingsStore,
    ) -> None:
        self.registry = registry
        self.repo = repository
        self.audit = audit
        self.settings_store = settings_store

    # ------------------------------------------------------------------ util
    def _policy(self) -> WorkspacePolicy:
        return WorkspacePolicy(self.settings_store.load().approved_workspaces)

    def _manager(self) -> PermissionManager:
        return PermissionManager(self._policy())

    @staticmethod
    def _summary(tool: Tool, validated: Any) -> str:
        fields = {k: v for k, v in validated.model_dump().items() if v is not None}
        return f"{tool.name}: {fields}"

    def _run(
        self,
        tool: Tool,
        validated: Any,
        level: PermissionLevel,
        approved_paths: Optional[list[str]] = None,
    ) -> ToolResult:
        policy = self._policy()
        settings = self.settings_store.load()
        ctx = ExecutionContext(
            workspace=policy,
            audit=self.audit,
            approved_workspaces=policy.roots,
            approved_paths=approved_paths or [],
            internet_enabled=settings.internet_enabled,
        )
        try:
            result = tool.handler(ctx, validated)
        except Exception as exc:  # noqa: BLE001
            result = ToolResult("error", error=str(exc))
        self.audit.record_execution(
            tool_name=tool.name,
            status=result.status,
            permission_level=level.value,
            target=self._target(tool, validated),
            result_summary=(result.error or result.output or "")[:500],
        )
        return result

    @staticmethod
    def _target(tool: Tool, validated: Any) -> Optional[str]:
        paths = tool.target_paths(validated)
        return paths[0] if paths else None

    # ------------------------------------------------------------------ list
    def list_tools(self) -> list[dict[str, Any]]:
        return self.registry.list()

    # -------------------------------------------------------------- submit
    def submit(self, tool_name: str, args: dict[str, Any], task_id: Optional[str] = None) -> dict[str, Any]:
        tool, validated = self.registry.validate(tool_name, args)
        level = tool.effective_level(validated)
        paths = tool.target_paths(validated)
        policy = self._policy()
        within = all(policy.is_within(p) for p in paths) if paths else True
        summary = self._summary(tool, validated)

        decision = self._manager().evaluate(
            ToolContext(
                tool_name=tool_name,
                level=level,
                target_paths=paths,
                within_workspace=within,
                auto_allowed_in_workspace=(level in _AUTO_LEVELS),
                summary=summary,
            )
        )

        if decision.requires_approval:
            approval_id = uuid.uuid4().hex
            self.audit.record_request(
                approval_id=approval_id,
                tool_name=tool_name,
                permission_level=level.value,
                args=args,
                summary=summary,
            )
            return {
                "status": "awaiting_approval",
                "approval_id": approval_id,
                "tool": tool_name,
                "level": level.value,
                "summary": summary,
                "args": args,
                "reason": decision.reason,
            }

        result = self._run(tool, validated, level)
        return {
            "status": result.status,
            "tool": tool_name,
            "level": level.value,
            "approved": True,
            "auto": True,
            "result": result.to_dict(),
        }

    # -------------------------------------------------------------- decide
    def decide(self, approval_id: str, approved: bool, task_id: Optional[str] = None) -> dict[str, Any]:
        approval = self.repo.get_approval(approval_id)
        if approval is None or approval["status"] != "pending":
            return {"status": "error", "error": "No such pending approval."}

        self.audit.record_decision(approval_id, approved)
        if not approved:
            return {"status": "denied", "approval_id": approval_id}

        tool, validated = self.registry.validate(approval["tool_name"], approval["args"])
        level = tool.effective_level(validated)
        result = self._run(tool, validated, level, approved_paths=tool.target_paths(validated))
        return {
            "status": result.status,
            "approval_id": approval_id,
            "approved": True,
            "tool": approval["tool_name"],
            "result": result.to_dict(),
        }

    # -------------------------------------------------------------- queries
    def pending_approvals(self) -> list[dict[str, Any]]:
        return self.repo.list_approvals(status="pending")

    def recent_logs(self, limit: int = 200) -> list[dict[str, Any]]:
        return self.audit.recent(limit=limit)
