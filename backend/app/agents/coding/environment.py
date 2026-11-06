"""Environment context injected into the agent's system prompt.

OS, shell, workspace path, date, git branch/status, a shallow file tree, and per-project
instructions read from ``AGENTS.md`` (or ``CLAUDE.md`` / ``.agents.md``) in the workspace.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, Optional

INSTRUCTION_FILES = ("AGENTS.md", "CLAUDE.md", ".agents.md")
IGNORED_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", "dist", "build", ".next", ".idea", ".vscode", "target", ".tox", ".cache",
}
MAX_INSTRUCTIONS_CHARS = 12_000


def shell_name() -> str:
    return "PowerShell" if sys.platform == "win32" else "bash"


def os_name() -> str:
    if sys.platform == "win32":
        release = platform.release()
        build = platform.version()
        return f"Windows {release} (build {build})"
    return f"{platform.system()} {platform.release()}"


def run_git(workspace: str, args: list[str], timeout: float = 5.0) -> Optional[str]:
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=workspace,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            **kwargs,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def git_summary(workspace: str) -> str:
    branch = run_git(workspace, ["rev-parse", "--abbrev-ref", "HEAD"])
    if branch is None:
        return "not a git repository"
    status = run_git(workspace, ["status", "--porcelain"]) or ""
    changed = [line for line in status.splitlines() if line.strip()]
    summary = f"branch {branch.strip()}, "
    if not changed:
        return summary + "working tree clean"
    shown = "; ".join(line.strip() for line in changed[:15])
    more = f" (+{len(changed) - 15} more)" if len(changed) > 15 else ""
    return summary + f"{len(changed)} changed: {shown}{more}"


def file_tree(workspace: str, max_entries: int = 60, depth: int = 2) -> str:
    root = Path(workspace)
    lines: list[str] = []

    def walk(folder: Path, level: int, prefix: str) -> None:
        if len(lines) >= max_entries:
            return
        try:
            entries = sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return
        for entry in entries:
            if len(lines) >= max_entries:
                lines.append(prefix + "…")
                return
            if entry.name in IGNORED_DIRS or (entry.name.startswith(".") and entry.name not in (".gitignore", ".env.example")):
                continue
            if entry.is_dir():
                lines.append(f"{prefix}{entry.name}/")
                if level < depth:
                    walk(entry, level + 1, prefix + "  ")
            else:
                lines.append(prefix + entry.name)

    walk(root, 1, "")
    return "\n".join(lines) if lines else "(empty folder)"


def project_instructions(workspace: str) -> tuple[Optional[str], str]:
    """Returns (file name, instructions block) or (None, "")."""
    for name in INSTRUCTION_FILES:
        path = Path(workspace) / name
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if len(text) > MAX_INSTRUCTIONS_CHARS:
                text = text[:MAX_INSTRUCTIONS_CHARS] + "\n… [truncated]"
            block = (
                f"# Project instructions (from {name})\n"
                "Follow these instructions from the project's maintainers; they override the general guidance above "
                "where they conflict.\n\n" + text.strip()
            )
            return name, block
    return None, ""


def environment_values(workspace: str) -> dict[str, Any]:
    name, instructions = project_instructions(workspace)
    return {
        "os": os_name(),
        "shell": shell_name(),
        "workspace": os.path.abspath(workspace),
        "date": date.today().isoformat(),
        "git": git_summary(workspace),
        "tree": file_tree(workspace),
        "project_instructions": instructions,
        "instructions_file": name,
    }
