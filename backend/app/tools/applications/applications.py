"""Tools for opening files and revealing them in the OS file manager."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel

from ..base import ExecutionContext, Tool, ToolResult
from ..registry import ToolRegistry
from ...security.permissions import PermissionLevel

EXECUTABLE_EXTENSIONS = {
    ".exe", ".bat", ".cmd", ".com", ".msi", ".ps1", ".vbs", ".scr", ".pif", ".lnk", ".jar", ".reg",
}


def _is_executable(path: str) -> bool:
    return Path(path).suffix.lower() in EXECUTABLE_EXTENSIONS


def _default_open(path: str) -> None:
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


def _default_reveal(path: str) -> None:
    if os.name == "nt":
        subprocess.run(["explorer", "/select,", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(Path(path).parent)], check=False)


class OpenFileRequest(BaseModel):
    path: str


class RevealRequest(BaseModel):
    path: str


def make_application_tools(
    opener: Optional[Callable[[str], None]] = None,
    revealer: Optional[Callable[[str], None]] = None,
) -> tuple[Tool, Tool]:
    open_fn = opener or _default_open
    reveal_fn = revealer or _default_reveal

    def _open(ctx: ExecutionContext, args: OpenFileRequest) -> ToolResult:
        path = Path(ctx.resolve_path(args.path))
        if not path.exists():
            return ToolResult("error", error=f"File not found: {path}")
        try:
            open_fn(str(path))
        except OSError as exc:
            return ToolResult("error", error=str(exc))
        return ToolResult("success", output=f"Opened {path}", data={"path": str(path)})

    def _reveal(ctx: ExecutionContext, args: RevealRequest) -> ToolResult:
        path = Path(ctx.resolve_path(args.path))
        if not path.exists():
            return ToolResult("error", error=f"Path not found: {path}")
        try:
            reveal_fn(str(path))
        except OSError as exc:
            return ToolResult("error", error=str(exc))
        return ToolResult("success", output=f"Revealed {path}", data={"path": str(path)})

    open_tool = Tool(
        name="open_file",
        description="Open a file with the operating system's default application.",
        permission_level=PermissionLevel.WORKSPACE_WRITE,
        input_model=OpenFileRequest,
        path_arg_names=("path",),
        dynamic_level_fn=lambda args: (
            PermissionLevel.SYSTEM_ACTION if _is_executable(args.path) else PermissionLevel.WORKSPACE_WRITE
        ),
        handler=_open,
    )
    reveal_tool = Tool(
        name="reveal_in_folder",
        description="Reveal a file or folder in the file manager.",
        permission_level=PermissionLevel.WORKSPACE_WRITE,
        input_model=RevealRequest,
        path_arg_names=("path",),
        handler=_reveal,
    )
    return open_tool, reveal_tool


def register_application_tools(
    registry: ToolRegistry,
    opener: Optional[Callable[[str], None]] = None,
    revealer: Optional[Callable[[str], None]] = None,
) -> None:
    open_tool, reveal_tool = make_application_tools(opener, revealer)
    registry.register(open_tool)
    registry.register(reveal_tool)
