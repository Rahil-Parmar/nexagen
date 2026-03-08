"""ToolRegistry — manages tool instances and filters by availability."""

from __future__ import annotations

from nexagen.tools.base import BaseTool


class ToolRegistry:
    """A registry that holds tool instances and exposes only available ones."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a single tool by its name."""
        self._tools[tool.name] = tool

    def register_many(self, tools: list[BaseTool]) -> None:
        """Register multiple tools at once."""
        for t in tools:
            self.register(t)

    def get(self, name: str) -> BaseTool | None:
        """Return the tool with the given *name*, or ``None``."""
        return self._tools.get(name)

    def list_available(self) -> list[BaseTool]:
        """Return all registered tools whose ``is_available()`` returns True."""
        return [t for t in self._tools.values() if t.is_available()]

    def get_tool_schemas(self) -> list[dict]:
        """Return provider-agnostic schemas for every available tool."""
        return [t.to_tool_schema() for t in self.list_available()]

    def get_tool_names(self) -> list[str]:
        """Return the names of every available tool."""
        return [t.name for t in self.list_available()]
