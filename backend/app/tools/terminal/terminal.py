"""Controlled command execution.

Commands are run through the permission system (always require approval) and are
never executed with elevation. Destructive command patterns are detected and
escalated to the HIGH_IMPACT level, which requires a separate confirmation.
"""
from __future__ import annotations

import re
import subprocess
import time
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from ..base import ExecutionContext, Tool, ToolResult
from ..registry import ToolRegistry
from ...security.permissions import PermissionLevel

MAX_OUTPUT_CHARS = 50_000

# Patterns that escalate a command to HIGH_IMPACT (always require confirmation).
DESTRUCTIVE_PATTERNS = [
    r"\brm\s+(-[a-z]*r[a-z]*f[a-z]*|-rf)\b",
    r"\brmdir\s+/s\b",
    r"\brd\s+/s\b",
    r"\bdel\s+/[fsq]+\b",
    r"\bformat\b",
    r"\bdiskpart\b",
    r"\bchkdsk\s+/f\b",
    r"\breg\s+(add|delete)\b",
    r"\bregedit\b",
    r"\bnetsh\b",
    r"\b(cacls|icacls)\b",
    r"\bchmod\s+-R\s+777\b",
    r"\b(chown|chmod)\s+-R\b",
    r"\bshutdown\b",
    r"\b(restart|stop)-computer\b",
    r"\bmkfs\b",
    r"\bdd\s+if=\b",
    r"\bschtasks\s+/delete\b",
    r"\bbcdedit\b",
]

_DESTRUCTIVE_RE = [re.compile(p, re.IGNORECASE) for p in DESTRUCTIVE_PATTERNS]


class RunCommandRequest(BaseModel):
    command: str = Field(min_length=1)
    cwd: Optional[str] = None
    timeout: int = Field(default=60, ge=1, le=3600)


def classify_command(command: str) -> PermissionLevel:
    """Return SYSTEM_ACTION, or HIGH_IMPACT if the command looks destructive."""
    if any(regex.search(command) for regex in _DESTRUCTIVE_RE):
        return PermissionLevel.HIGH_IMPACT
    return PermissionLevel.SYSTEM_ACTION


def _real_runner(command: str, cwd: Optional[str], timeout: int) -> dict[str, Any]:
    started = time.monotonic()
    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        raise TimeoutError(f"Command timed out after {timeout}s.")
    elapsed = time.monotonic() - started
    return {
        "exit_code": proc.returncode,
        "stdout": stdout or "",
        "stderr": stderr or "",
        "elapsed": elapsed,
    }


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        proc.kill()
    except Exception:
        pass
    # On Windows, also kill the child tree via taskkill.
    try:
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
            capture_output=True,
            timeout=10,
        )
    except Exception:
        pass


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n… [truncated {len(text) - MAX_OUTPUT_CHARS} chars]"


def make_run_command_tool(
    runner: Optional[Callable[[str, Optional[str], int], dict[str, Any]]] = None,
) -> Tool:
    run = runner or _real_runner

    def handler(ctx: ExecutionContext, args: RunCommandRequest) -> ToolResult:
        level = classify_command(args.command)
        try:
            result = run(args.command, args.cwd, args.timeout)
        except TimeoutError as exc:
            return ToolResult("timeout", error=str(exc))
        except OSError as exc:
            return ToolResult("error", error=str(exc))

        exit_code = result.get("exit_code")
        stdout = _truncate(result.get("stdout", ""))
        stderr = _truncate(result.get("stderr", ""))
        output = stdout
        if stderr:
            output += ("\n" if stdout else "") + f"[stderr]\n{stderr}"

        status = "success" if exit_code == 0 else "error"
        data = {
            "exit_code": exit_code,
            "elapsed": result.get("elapsed"),
            "stdout": stdout,
            "stderr": stderr,
            "command": args.command,
            "cwd": args.cwd,
        }
        if exit_code != 0:
            return ToolResult(
                status, output=output, data=data, error=f"Exited with code {exit_code}."
            )
        return ToolResult(status, output=output, data=data)

    return Tool(
        name="run_command",
        description="Run a shell command in a chosen working directory. Always requires approval.",
        permission_level=PermissionLevel.SYSTEM_ACTION,
        input_model=RunCommandRequest,
        path_arg_names=("cwd",),
        dynamic_level_fn=lambda args: classify_command(args.command),
        handler=handler,
    )


def register_terminal_tools(
    registry: ToolRegistry,
    runner: Optional[Callable[[str, Optional[str], int], dict[str, Any]]] = None,
) -> None:
    registry.register(make_run_command_tool(runner))
