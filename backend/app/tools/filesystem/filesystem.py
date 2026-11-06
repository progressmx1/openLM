"""Workspace-aware filesystem tools.

Every handler enforces the workspace boundary itself (defense in depth) in
addition to the permission check performed by the executor.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from ..base import ExecutionContext, Tool, ToolResult
from ..registry import ToolRegistry
from ...security.permissions import PermissionLevel
from ...security.workspace import WorkspaceViolation

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".json", ".yaml", ".yml", ".csv", ".tsv",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs", ".html", ".htm",
    ".css", ".scss", ".sh", ".bat", ".ps1", ".c", ".h", ".cpp", ".hpp",
    ".java", ".go", ".rs", ".rb", ".php", ".sql", ".xml", ".toml", ".ini",
    ".cfg", ".log", ".gitignore", ".dockerfile", ".env",
}

MAX_READ_BYTES = 1_000_000
MAX_LIST_ENTRIES = 500
MAX_SEARCH_RESULTS = 200


class ListDirectoryRequest(BaseModel):
    path: str


class ReadFileRequest(BaseModel):
    path: str


class WriteFileRequest(BaseModel):
    path: str
    content: str
    overwrite: bool = False


class CreateFolderRequest(BaseModel):
    path: str


class SearchFilesRequest(BaseModel):
    directory: str
    pattern: str = "*"


class CopyFileRequest(BaseModel):
    source: str
    destination: str


class MoveFileRequest(BaseModel):
    source: str
    destination: str


class DeleteRequest(BaseModel):
    path: str


def _resolve(ctx: ExecutionContext, path: str) -> Path:
    return Path(ctx.resolve_path(path))


def _list_directory(ctx: ExecutionContext, args: ListDirectoryRequest) -> ToolResult:
    try:
        directory = _resolve(ctx, args.path)
    except WorkspaceViolation as exc:
        return ToolResult("error", error=str(exc))
    if not directory.is_dir():
        return ToolResult("error", error=f"Not a directory: {directory}")
    entries: list[dict[str, Any]] = []
    for item in sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if len(entries) >= MAX_LIST_ENTRIES:
            break
        try:
            entries.append(
                {
                    "name": item.name,
                    "is_dir": item.is_dir(),
                    "size_bytes": item.stat().st_size if not item.is_dir() else None,
                    "modified": item.stat().st_mtime,
                }
            )
        except OSError:
            continue
    return ToolResult("success", output=f"{len(entries)} entries", data={"entries": entries})


def _read_file(ctx: ExecutionContext, args: ReadFileRequest) -> ToolResult:
    try:
        path = _resolve(ctx, args.path)
    except WorkspaceViolation as exc:
        return ToolResult("error", error=str(exc))
    if not path.is_file():
        return ToolResult("error", error=f"Not a file: {path}")
    if path.suffix.lower() not in TEXT_EXTENSIONS:
        return ToolResult("error", error=f"Unsupported file type: {path.suffix or '<none>'}")
    size = path.stat().st_size
    if size > MAX_READ_BYTES:
        return ToolResult(
            "error",
            error=f"File too large to read ({size} bytes; limit {MAX_READ_BYTES}).",
        )
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return ToolResult("error", error=str(exc))
    return ToolResult(
        "success",
        output=content,
        data={"path": str(path), "size_bytes": size},
    )


def _write_file(ctx: ExecutionContext, args: WriteFileRequest) -> ToolResult:
    try:
        path = _resolve(ctx, args.path)
    except WorkspaceViolation as exc:
        return ToolResult("error", error=str(exc))
    if path.exists() and not args.overwrite:
        return ToolResult(
            "error",
            error=f"File already exists: {path}. Set overwrite=true to replace it.",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(args.content, encoding="utf-8")
    except OSError as exc:
        return ToolResult("error", error=str(exc))
    return ToolResult("success", output=f"Wrote {path}", data={"path": str(path)})


def _create_folder(ctx: ExecutionContext, args: CreateFolderRequest) -> ToolResult:
    try:
        path = _resolve(ctx, args.path)
    except WorkspaceViolation as exc:
        return ToolResult("error", error=str(exc))
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return ToolResult("error", error=str(exc))
    return ToolResult("success", output=f"Created {path}", data={"path": str(path)})


def _search_files(ctx: ExecutionContext, args: SearchFilesRequest) -> ToolResult:
    try:
        directory = _resolve(ctx, args.directory)
    except WorkspaceViolation as exc:
        return ToolResult("error", error=str(exc))
    if not directory.is_dir():
        return ToolResult("error", error=f"Not a directory: {directory}")
    matches = []
    for item in directory.rglob(args.pattern):
        if len(matches) >= MAX_SEARCH_RESULTS:
            break
        if item.is_file():
            matches.append({"path": str(item), "name": item.name, "size_bytes": item.stat().st_size})
    return ToolResult("success", output=f"{len(matches)} matches", data={"matches": matches})


def _copy_file(ctx: ExecutionContext, args: CopyFileRequest) -> ToolResult:
    try:
        source = _resolve(ctx, args.source)
        destination = _resolve(ctx, args.destination)
    except WorkspaceViolation as exc:
        return ToolResult("error", error=str(exc))
    if not source.is_file():
        return ToolResult("error", error=f"Source not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and source.resolve() == destination.resolve():
        return ToolResult("error", error="Source and destination are the same file.")
    try:
        shutil.copy2(source, destination)
    except OSError as exc:
        return ToolResult("error", error=str(exc))
    return ToolResult("success", output=f"Copied to {destination}", data={"path": str(destination)})


def _move_file(ctx: ExecutionContext, args: MoveFileRequest) -> ToolResult:
    try:
        source = _resolve(ctx, args.source)
        destination = _resolve(ctx, args.destination)
    except WorkspaceViolation as exc:
        return ToolResult("error", error=str(exc))
    if not source.exists():
        return ToolResult("error", error=f"Source not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(source), str(destination))
    except OSError as exc:
        return ToolResult("error", error=str(exc))
    return ToolResult("success", output=f"Moved to {destination}", data={"path": str(destination)})


def _delete(ctx: ExecutionContext, args: DeleteRequest) -> ToolResult:
    try:
        path = _resolve(ctx, args.path)
    except WorkspaceViolation as exc:
        return ToolResult("error", error=str(exc))
    if not path.exists():
        return ToolResult("error", error=f"Path not found: {path}")
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except OSError as exc:
        return ToolResult("error", error=str(exc))
    return ToolResult("success", output=f"Deleted {path}", data={"path": str(path)})


def register_filesystem_tools(registry: ToolRegistry) -> None:
    registry.register(Tool(
        name="list_directory",
        description="List files and folders in a directory inside an approved workspace.",
        permission_level=PermissionLevel.READ_ONLY,
        input_model=ListDirectoryRequest,
        path_arg_names=("path",),
        handler=_list_directory,
    ))
    registry.register(Tool(
        name="read_file",
        description="Read a text file inside an approved workspace.",
        permission_level=PermissionLevel.READ_ONLY,
        input_model=ReadFileRequest,
        path_arg_names=("path",),
        handler=_read_file,
    ))
    registry.register(Tool(
        name="write_file",
        description="Create or overwrite a text file inside an approved workspace.",
        permission_level=PermissionLevel.WORKSPACE_WRITE,
        input_model=WriteFileRequest,
        path_arg_names=("path",),
        handler=_write_file,
    ))
    registry.register(Tool(
        name="create_folder",
        description="Create a folder inside an approved workspace.",
        permission_level=PermissionLevel.WORKSPACE_WRITE,
        input_model=CreateFolderRequest,
        path_arg_names=("path",),
        handler=_create_folder,
    ))
    registry.register(Tool(
        name="search_files",
        description="Search file names by pattern inside an approved workspace.",
        permission_level=PermissionLevel.READ_ONLY,
        input_model=SearchFilesRequest,
        path_arg_names=("directory",),
        handler=_search_files,
    ))
    registry.register(Tool(
        name="copy_file",
        description="Copy a file within an approved workspace.",
        permission_level=PermissionLevel.WORKSPACE_WRITE,
        input_model=CopyFileRequest,
        path_arg_names=("source", "destination"),
        handler=_copy_file,
    ))
    registry.register(Tool(
        name="move_file",
        description="Move or rename a file within an approved workspace.",
        permission_level=PermissionLevel.WORKSPACE_WRITE,
        input_model=MoveFileRequest,
        path_arg_names=("source", "destination"),
        handler=_move_file,
    ))
    registry.register(Tool(
        name="delete",
        description="Delete a file or folder. Always requires explicit confirmation.",
        permission_level=PermissionLevel.HIGH_IMPACT,
        input_model=DeleteRequest,
        path_arg_names=("path",),
        handler=_delete,
    ))
