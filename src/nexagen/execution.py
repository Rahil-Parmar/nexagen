"""Parallel tool execution engine."""

from __future__ import annotations

import asyncio
import logging

from nexagen.constants import DEFAULT_MAX_PARALLEL_TOOLS
from nexagen.models import ToolCall, ToolResult
from nexagen.permissions import Deny, PermissionManager
from nexagen.tools.registry import ToolRegistry

logger = logging.getLogger("nexagen.execution")


class ParallelExecutor:
    """Executes multiple tool calls concurrently via asyncio.gather.

    Each call goes through: permission check -> tool lookup -> execute.
    One failing tool does NOT cancel others. Results preserve input order.
    """

    def __init__(self, max_concurrent: int = DEFAULT_MAX_PARALLEL_TOOLS):
        self.max_concurrent = max_concurrent

    async def execute_batch(
        self,
        tool_calls: list[ToolCall],
        tool_registry: ToolRegistry,
        permissions: PermissionManager,
    ) -> list[ToolResult]:
        """Run all *tool_calls* in parallel and return results in the same order.

        Caps concurrency at ``max_concurrent`` via an asyncio.Semaphore.
        Excess tool calls beyond the cap are rejected with an error result.
        """
        if not tool_calls:
            return []

        # Hard cap: reject excess tool calls entirely
        if len(tool_calls) > self.max_concurrent:
            logger.warning(
                "Tool call batch of %d exceeds max_concurrent=%d. "
                "Executing first %d, rejecting rest.",
                len(tool_calls),
                self.max_concurrent,
                self.max_concurrent,
            )
            accepted = tool_calls[: self.max_concurrent]
            rejected = tool_calls[self.max_concurrent :]
        else:
            accepted = tool_calls
            rejected = []

        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = [
            self._execute_single(call, tool_registry, permissions, semaphore)
            for call in accepted
        ]
        results = list(await asyncio.gather(*tasks))

        # Append error results for rejected calls
        for tc in rejected:
            results.append(
                ToolResult(
                    tool_call_id=tc.id,
                    output=f"Rejected: too many parallel tool calls (max {self.max_concurrent})",
                    is_error=True,
                )
            )

        return results

    async def _execute_single(
        self,
        call: ToolCall,
        tool_registry: ToolRegistry,
        permissions: PermissionManager,
        semaphore: asyncio.Semaphore,
    ) -> ToolResult:
        """Execute a single tool call with permission checking and concurrency limit."""
        async with semaphore:
            # Permission check
            verdict = await permissions.check(call.name, call.arguments)
            if isinstance(verdict, Deny):
                return ToolResult(
                    tool_call_id=call.id,
                    output=f"Permission denied: {verdict.message}",
                    is_error=True,
                )

            # Tool lookup
            tool = tool_registry.get(call.name)
            if tool is None:
                return ToolResult(
                    tool_call_id=call.id,
                    output=f"Unknown tool: '{call.name}' not found in registry",
                    is_error=True,
                )

            # Execute
            result = await tool.execute(call.arguments)
            # BaseTool.execute sets tool_call_id="" by default; patch it
            result.tool_call_id = call.id
            return result
