"""Coding-assistant tools: content search and read-only Git inspection.

Git tools run a fixed allowlist of commands (status/diff/log) — never arbitrary
shell — and the runner is injectable so tests never depend on git or the network.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from ..base import ExecutionContext, Tool, ToolResult
from ..registry import ToolRegistry
from ...security.permissions import PermissionLevel


class SearchContentRequest(BaseModel):
    directory: str
    pattern: str = Field(min_length=1)
    file_glob: str = "*"
    max_results: int = Field(default=100, ge=1, le=1000)


class GitRequest(BaseModel):
    path: str
    count: int = Field(default=20, ge=1, le=200)


def search_content(
    directory: str, pattern: str, file_glob: str = "*", max_results: int = 100
) -> list[dict[str, Any]]:
    regex = re.compile(pattern)
    matches: list[dict[str, Any]] = []
    for file in Path(directory).rglob(file_glob):
        if not file.is_file():
            continue
        try:
            text = file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                matches.append({"path": str(file), "line": line_no, "text": line.strip()[:200]})
                if len(matches) >= max_results:
                    return matches
    return matches


def _real_git_runner(args: list[str], timeout: int = 30) -> dict[str, Any]:
    proc = subprocess.run(
        ["git"] + args, capture_output=True, text=True, errors="replace", timeout=timeout
    )
    return {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def _git(args: list[str], runner, timeout: int = 30) -> dict[str, Any]:
    return runner(args, timeout)


def make_coding_tools(
    runner: Optional[Callable[[list[str], int], dict[str, Any]]] = None,
) -> list[Tool]:
    git_runner = runner or _real_git_runner

    def _search(ctx: ExecutionContext, args: SearchContentRequest) -> ToolResult:
        try:
            directory = ctx.resolve_path(args.directory)
            matches = search_content(directory, args.pattern, args.file_glob, args.max_results)
            return ToolResult("success", output=f"{len(matches)} matches", data={"matches": matches})
        except Exception as exc:  # noqa: BLE001
            return ToolResult("error", error=str(exc))

    def _git_status(ctx: ExecutionContext, args: GitRequest) -> ToolResult:
        try:
            path = ctx.resolve_path(args.path)
            result = _git(["-C", path, "status", "--short"], git_runner)
            return _git_result(result)
        except Exception as exc:  # noqa: BLE001
            return ToolResult("error", error=str(exc))

    def _git_diff(ctx: ExecutionContext, args: GitRequest) -> ToolResult:
        try:
            path = ctx.resolve_path(args.path)
            result = _git(["-C", path, "diff"], git_runner)
            return _git_result(result)
        except Exception as exc:  # noqa: BLE001
            return ToolResult("error", error=str(exc))

    def _git_log(ctx: ExecutionContext, args: GitRequest) -> ToolResult:
        try:
            path = ctx.resolve_path(args.path)
            result = _git(["-C", path, "log", "--oneline", f"-n{args.count}"], git_runner)
            return _git_result(result)
        except Exception as exc:  # noqa: BLE001
            return ToolResult("error", error=str(exc))

    return [
        Tool(
            "search_content",
            "Search file contents by regex inside an approved workspace.",
            PermissionLevel.READ_ONLY,
            SearchContentRequest,
            ("directory",),
            handler=_search,
        ),
        Tool(
            "git_status",
            "Show Git working-tree status (read-only).",
            PermissionLevel.READ_ONLY,
            GitRequest,
            ("path",),
            handler=_git_status,
        ),
        Tool(
            "git_diff",
            "Show uncommitted Git changes (read-only).",
            PermissionLevel.READ_ONLY,
            GitRequest,
            ("path",),
            handler=_git_diff,
        ),
        Tool(
            "git_log",
            "Show recent Git commits (read-only).",
            PermissionLevel.READ_ONLY,
            GitRequest,
            ("path",),
            handler=_git_log,
        ),
    ]


def _git_result(result: dict[str, Any]) -> ToolResult:
    exit_code = result.get("exit_code")
    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")
    if exit_code != 0:
        return ToolResult(
            "error", output=stdout, data={"exit_code": exit_code}, error=(stderr or f"git exited with code {exit_code}").strip()
        )
    return ToolResult("success", output=stdout, data={"exit_code": 0})


def register_coding_tools(
    registry: ToolRegistry,
    runner: Optional[Callable[[list[str], int], dict[str, Any]]] = None,
) -> None:
    for tool in make_coding_tools(runner):
        registry.register(tool)
