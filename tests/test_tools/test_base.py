"""Tests for nexagen.tools.base — BaseTool class and @tool decorator."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from nexagen.models import ToolResult
from nexagen.tools.base import BaseTool, tool


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

class GreetInput(BaseModel):
    name: str
    formal: bool = False


async def greet_handler(inp: GreetInput) -> str:
    if inp.formal:
        return f"Good day, {inp.name}."
    return f"Hi, {inp.name}!"


@tool(name="greet", description="Greets a user", input_model=GreetInput)
async def greet_tool(inp: GreetInput) -> str:
    if inp.formal:
        return f"Good day, {inp.name}."
    return f"Hi, {inp.name}!"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_tool_decorator_creates_tool():
    """@tool decorator should return a BaseTool with correct name/description."""
    assert isinstance(greet_tool, BaseTool)
    assert greet_tool.name == "greet"
    assert greet_tool.description == "Greets a user"


def test_tool_has_json_schema():
    """input_schema should contain 'properties' with the input model fields."""
    schema = greet_tool.input_schema
    assert "properties" in schema
    assert "name" in schema["properties"]
    assert "formal" in schema["properties"]


def test_tool_is_available():
    """Default is_available() should return True."""
    assert greet_tool.is_available() is True


def test_tool_execute_success():
    """Execute with valid args should return ToolResult(is_error=False)."""
    result = asyncio.run(greet_tool.execute({"name": "Alice", "formal": False}))
    assert isinstance(result, ToolResult)
    assert result.is_error is False
    assert "Alice" in result.output


def test_tool_execute_validation_error():
    """Execute with invalid args should return is_error=True with validation message."""
    result = asyncio.run(greet_tool.execute({"formal": "not_a_bool"}))
    assert result.is_error is True
    assert "ValidationError" in result.output


def test_tool_execute_runtime_error():
    """Handler that raises RuntimeError should return is_error=True with error type."""

    async def bad_handler(inp: GreetInput) -> str:
        raise RuntimeError("something broke")

    bad_tool = BaseTool(
        name="bad",
        description="A tool that fails",
        input_model=GreetInput,
        handler=bad_handler,
    )
    result = asyncio.run(bad_tool.execute({"name": "Bob"}))
    assert result.is_error is True
    assert "RuntimeError" in result.output
    assert "something broke" in result.output


def test_tool_to_schema():
    """to_tool_schema() should return dict with name, description, parameters."""
    schema = greet_tool.to_tool_schema()
    assert schema["name"] == "greet"
    assert schema["description"] == "Greets a user"
    assert "parameters" in schema
    assert "properties" in schema["parameters"]


def test_tool_connect_disconnect():
    """connect() and disconnect() should be no-op coroutines that don't raise."""
    asyncio.run(greet_tool.connect())
    asyncio.run(greet_tool.disconnect())
