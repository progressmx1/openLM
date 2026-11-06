"""Audit trail: records approvals and tool execution results locally."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from ..storage.repository import Repository


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuditLog:
    def __init__(self, repository: Repository) -> None:
        self.repo = repository

    def record_request(
        self,
        *,
        approval_id: str,
        tool_name: str,
        permission_level: str,
        args: dict[str, Any],
        summary: str,
    ) -> None:
        self.repo.create_approval(
            approval_id=approval_id,
            tool_name=tool_name,
            permission_level=permission_level,
            args=args,
            summary=summary,
        )

    def record_decision(self, approval_id: str, approved: bool) -> None:
        self.repo.decide_approval(approval_id, approved)

    def record_execution(
        self,
        *,
        tool_name: str,
        status: str,
        permission_level: str,
        target: Optional[str] = None,
        result_summary: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> None:
        self.repo.add_tool_log(
            tool_name=tool_name,
            status=status,
            permission_level=permission_level,
            target=target,
            result_summary=result_summary,
            task_id=task_id,
        )

    def recent(self, limit: int = 200) -> list[dict[str, Any]]:
        return self.repo.list_tool_logs(limit=limit)
