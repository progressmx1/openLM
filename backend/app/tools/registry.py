"""Central tool registry with validated input schemas."""
from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from .base import Tool


class UnknownToolError(KeyError):
    pass


class InvalidToolArgumentsError(ValueError):
    pass


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise UnknownToolError(f"Unknown tool: {name}")
        return tool

    def list(self) -> list[dict[str, Any]]:
        return [tool.to_dict() for tool in self._tools.values()]

    def validate(self, name: str, args: dict[str, Any]):
        """Return (tool, validated_input); raises on unknown tool or bad args."""
        tool = self.get(name)
        try:
            validated = tool.validate(args)
        except ValidationError as exc:
            raise InvalidToolArgumentsError(str(exc)) from exc
        return tool, validated
