"""Tests for ParallelExecutor — parallel tool execution."""

from __future__ import annotations

import asyncio
import time

import pytest
from pydantic import BaseModel

from nexagen.execution import ParallelExecutor
from nexagen.models import ToolCall, ToolResult
from nexagen.permissions import Allow, Deny, PermissionManager
from nexagen.tools.base import BaseTool, tool
from nexagen.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Test tools
# ---------------------------------------------------------------------------

class AddInput(BaseModel):
    a: int
    b: int


@tool(name="add", description="Add two numbers", input_model=AddInput)
async def add_tool(inp: AddInput) -> str:
    return str(inp.a + inp.b)


class MultiplyInput(BaseModel):
    x: int
    y: int


@tool(name="multiply", description="Multiply two numbers", input_model=MultiplyInput)
async def multiply_tool(inp: MultiplyInput) -> str:
    return str(inp.x * inp.y)


class SlowInput(BaseModel):
    delay: float
    label: str


@tool(name="slow", description="Sleeps then returns label", input_model=SlowInput)
async def slow_tool(inp: SlowInput) -> str:
    await asyncio.sleep(inp.delay)
    return inp.label


class FailInput(BaseModel):
    pass


@tool(name="fail", description="Always raises", input_model=FailInput)
async def fail_tool(inp: FailInput) -> str:
    raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry(*tools: BaseTool) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register_many(list(tools))
    return reg


def _make_permissions(mode: str = "full") -> PermissionManager:
    return PermissionManager(mode=mode)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParallelExecutor:
    async def test_single_tool_call(self):
        """A single tool call executes and returns a correct ToolResult."""
        registry = _make_registry(add_tool)
        permissions = _make_permissions()
        executor = ParallelExecutor()

        calls = [ToolCall(id="tc1", name="add", arguments={"a": 3, "b": 4})]
        results = await executor.execute_batch(calls, registry, permissions)

        assert len(results) == 1
        assert results[0].tool_call_id == "tc1"
        assert results[0].output == "7"
        assert results[0].is_error is False

    async def test_multiple_tools_run_in_parallel(self):
        """Multiple slow tools run concurrently, not sequentially."""
        registry = _make_registry(slow_tool)
        permissions = _make_permissions()
        executor = ParallelExecutor()

        # Three tools each sleeping 0.2s. Sequential = ~0.6s, parallel < 0.4s.
        calls = [
            ToolCall(id="tc1", name="slow", arguments={"delay": 0.2, "label": "a"}),
            ToolCall(id="tc2", name="slow", arguments={"delay": 0.2, "label": "b"}),
            ToolCall(id="tc3", name="slow", arguments={"delay": 0.2, "label": "c"}),
        ]

        start = time.monotonic()
        results = await executor.execute_batch(calls, registry, permissions)
        elapsed = time.monotonic() - start

        assert len(results) == 3
        assert all(r.is_error is False for r in results)
        # If parallel, should complete in ~0.2s, not ~0.6s
        assert elapsed < 0.5, f"Took {elapsed:.2f}s — tools likely ran sequentially"

    async def test_permission_denied(self):
        """Permission-denied tool calls return ToolResult with is_error=True."""
        registry = _make_registry(add_tool)
        # readonly mode only allows file_read, grep, glob — not "add"
        permissions = _make_permissions(mode="readonly")
        executor = ParallelExecutor()

        calls = [ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})]
        results = await executor.execute_batch(calls, registry, permissions)

        assert len(results) == 1
        assert results[0].tool_call_id == "tc1"
        assert results[0].is_error is True
        assert "denied" in results[0].output.lower() or "not allowed" in results[0].output.lower()

    async def test_unknown_tool(self):
        """Calling an unregistered tool returns ToolResult with is_error=True."""
        registry = _make_registry(add_tool)
        permissions = _make_permissions()
        executor = ParallelExecutor()

        calls = [ToolCall(id="tc1", name="nonexistent", arguments={})]
        results = await executor.execute_batch(calls, registry, permissions)

        assert len(results) == 1
        assert results[0].tool_call_id == "tc1"
        assert results[0].is_error is True
        assert "unknown" in results[0].output.lower() or "not found" in results[0].output.lower()

    async def test_one_failure_does_not_cancel_others(self):
        """A failing tool does not prevent other tools from completing."""
        registry = _make_registry(add_tool, fail_tool)
        permissions = _make_permissions()
        executor = ParallelExecutor()

        calls = [
            ToolCall(id="tc1", name="add", arguments={"a": 5, "b": 5}),
            ToolCall(id="tc2", name="fail", arguments={}),
            ToolCall(id="tc3", name="add", arguments={"a": 10, "b": 20}),
        ]
        results = await executor.execute_batch(calls, registry, permissions)

        assert len(results) == 3
        # First and third succeed
        assert results[0].tool_call_id == "tc1"
        assert results[0].output == "10"
        assert results[0].is_error is False

        # Second is an error
        assert results[1].tool_call_id == "tc2"
        assert results[1].is_error is True

        # Third still succeeds
        assert results[2].tool_call_id == "tc3"
        assert results[2].output == "30"
        assert results[2].is_error is False

    async def test_preserves_order(self):
        """Results are returned in the same order as the input tool calls."""
        registry = _make_registry(slow_tool)
        permissions = _make_permissions()
        executor = ParallelExecutor()

        # Varying delays — order should still match input, not completion order
        calls = [
            ToolCall(id="tc1", name="slow", arguments={"delay": 0.15, "label": "first"}),
            ToolCall(id="tc2", name="slow", arguments={"delay": 0.05, "label": "second"}),
            ToolCall(id="tc3", name="slow", arguments={"delay": 0.10, "label": "third"}),
        ]
        results = await executor.execute_batch(calls, registry, permissions)

        assert [r.tool_call_id for r in results] == ["tc1", "tc2", "tc3"]
        assert [r.output for r in results] == ["first", "second", "third"]

    async def test_empty_batch(self):
        """An empty batch returns an empty list."""
        registry = _make_registry()
        permissions = _make_permissions()
        executor = ParallelExecutor()

        results = await executor.execute_batch([], registry, permissions)
        assert results == []

    async def test_excess_tool_calls_rejected(self):
        """Tool calls beyond max_concurrent are rejected with error results."""
        registry = _make_registry(add_tool)
        permissions = _make_permissions()
        executor = ParallelExecutor(max_concurrent=2)

        calls = [
            ToolCall(id=f"tc{i}", name="add", arguments={"a": i, "b": 1})
            for i in range(5)
        ]
        results = await executor.execute_batch(calls, registry, permissions)

        assert len(results) == 5
        # First 2 accepted
        assert results[0].is_error is False
        assert results[1].is_error is False
        # Last 3 rejected
        assert all(r.is_error for r in results[2:])
        assert all("too many" in r.output.lower() for r in results[2:])

    async def test_custom_max_concurrent(self):
        """max_concurrent parameter is respected."""
        executor = ParallelExecutor(max_concurrent=3)
        assert executor.max_concurrent == 3
