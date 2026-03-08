"""Tests for ToolRegistry."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from nexagen.tools.base import BaseTool, tool
from nexagen.tools.registry import ToolRegistry


# -- Fixtures / helpers -------------------------------------------------------

class DummyInput(BaseModel):
    text: str


@tool("dummy", "A dummy tool", input_model=DummyInput)
async def dummy_tool(args: DummyInput) -> str:
    return args.text


@tool("another", "Another dummy tool", input_model=DummyInput)
async def another_tool(args: DummyInput) -> str:
    return args.text.upper()


class UnavailableTool(BaseTool):
    """A tool that is never available."""

    def is_available(self) -> bool:
        return False


def _make_unavailable_tool() -> UnavailableTool:
    return UnavailableTool(
        name="unavailable",
        description="never available",
        input_model=DummyInput,
        handler=dummy_tool._handler,
    )


# -- Tests --------------------------------------------------------------------


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        registry.register(dummy_tool)
        assert registry.get("dummy") is dummy_tool

    def test_get_unknown_returns_none(self):
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_list_available(self):
        registry = ToolRegistry()
        registry.register(dummy_tool)
        registry.register(another_tool)
        available = registry.list_available()
        assert len(available) == 2
        assert dummy_tool in available
        assert another_tool in available

    def test_list_available_filters_unavailable(self):
        registry = ToolRegistry()
        registry.register(dummy_tool)
        unavailable = _make_unavailable_tool()
        registry.register(unavailable)
        available = registry.list_available()
        assert len(available) == 1
        assert dummy_tool in available
        assert unavailable not in available

    def test_get_schemas(self):
        registry = ToolRegistry()
        registry.register(dummy_tool)
        schemas = registry.get_tool_schemas()
        assert len(schemas) == 1
        schema = schemas[0]
        assert schema["name"] == "dummy"
        assert schema["description"] == "A dummy tool"
        assert "parameters" in schema

    def test_register_many(self):
        registry = ToolRegistry()
        registry.register_many([dummy_tool, another_tool])
        assert registry.get("dummy") is dummy_tool
        assert registry.get("another") is another_tool

    def test_get_tool_names(self):
        registry = ToolRegistry()
        registry.register_many([dummy_tool, another_tool])
        names = registry.get_tool_names()
        assert sorted(names) == ["another", "dummy"]
