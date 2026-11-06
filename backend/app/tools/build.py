"""Assemble the default tool registry."""
from __future__ import annotations

from typing import Any, Callable, Optional

from .applications.applications import register_application_tools
from .coding.coding import register_coding_tools
from .documents.docx import register_docx_tools
from .documents.pdf import register_pdf_tools
from .filesystem.filesystem import register_filesystem_tools
from .registry import ToolRegistry
from .terminal.terminal import register_terminal_tools
from .web.web import register_web_tools


def build_default_registry(
    runner: Optional[Callable[[str, Optional[str], int], dict[str, Any]]] = None,
    opener: Optional[Callable[[str], None]] = None,
    revealer: Optional[Callable[[str], None]] = None,
    git_runner: Optional[Callable[[list[str], int], dict[str, Any]]] = None,
    search_fn: Optional[Callable[[str, int], list[dict[str, str]]]] = None,
    fetch_fn: Optional[Callable[[str], dict[str, str]]] = None,
    download_fn: Optional[Callable[[str, str], dict[str, Any]]] = None,
) -> ToolRegistry:
    """Build a registry with all built-in tools.

    ``runner``/``opener``/``revealer``/``git_runner``/``search_fn``/``fetch_fn``/
    ``download_fn`` are injectable for tests.
    """
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    register_terminal_tools(registry, runner=runner)
    register_application_tools(registry, opener=opener, revealer=revealer)
    register_pdf_tools(registry)
    register_docx_tools(registry)
    register_web_tools(registry, search=search_fn, fetch=fetch_fn, download=download_fn)
    register_coding_tools(registry, runner=git_runner)
    return registry
