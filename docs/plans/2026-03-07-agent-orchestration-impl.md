# Agent Orchestration Improvements — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 7 frontier agent orchestration improvements to nexagen: parallel tool execution, observation masking, self-reflection, structured planning, episodic memory, rich supervisor feedback, and KV-cache-aware context design.

**Architecture:** Layered Enhancement — each improvement is a composable module that plugs into the existing `Agent` class. New modules: `execution.py`, `context.py`, `reflection.py`, `planning.py`, `memory.py`. Enhanced modules: `supervisor.py`, `conversation.py`, `agent.py`, `constants.py`.

**Tech Stack:** Python 3.11+, Pydantic 2.x, asyncio, pytest + pytest-asyncio

---

### Task 1: Add New Constants

**Files:**
- Modify: `src/nexagen/constants.py:1-19`

**Step 1: Add the new default constants**

```python
DEFAULT_MODEL = "ollama/qwen3"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096
DEFAULT_CONTEXT_THRESHOLD = 0.80
DEFAULT_COMPRESS_TARGET = 0.50
DEFAULT_MAX_TOOL_ERRORS = 3
DEFAULT_SUPERVISOR_MODEL = "ollama/phi3"
DEFAULT_SUPERVISOR_CHECK_INTERVAL = 5
DEFAULT_PERMISSION_MODE = "safe"
DEFAULT_MAX_ITERATIONS = 100  # max agent loop iterations to prevent runaway loops
CHARS_PER_TOKEN = 4

# --- New constants for orchestration improvements ---
DEFAULT_MAX_REFLECTIONS = 2
DEFAULT_MAX_EPISODES = 50
DEFAULT_RECENT_TOOL_RESULTS_TO_KEEP = 3

OPENAI_COMPAT_DEFAULT_URLS: dict[str, str] = {
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
    "lmstudio": "http://localhost:1234/v1",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
}
```

**Step 2: Verify no tests break**

Run: `pytest tests/ -v --tb=short`
Expected: All existing tests PASS (constants are additive)

**Step 3: Commit**

```bash
git add src/nexagen/constants.py
git commit -m "feat: add constants for orchestration improvements"
```

---

### Task 2: Parallel Tool Execution

**Files:**
- Create: `src/nexagen/execution.py`
- Test: `tests/test_execution.py`

**Step 1: Write the failing tests**

```python
"""Tests for ParallelExecutor."""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock

from nexagen.execution import ParallelExecutor
from nexagen.models import NexagenMessage, ToolCall, ToolResult
from nexagen.tools.base import BaseTool, tool
from nexagen.tools.registry import ToolRegistry
from nexagen.permissions import PermissionManager, Allow, Deny
from pydantic import BaseModel


class AddInput(BaseModel):
    a: int
    b: int


@tool(name="add", description="Add two numbers", input_model=AddInput)
async def add_tool(inp: AddInput) -> str:
    return str(inp.a + inp.b)


class SlowInput(BaseModel):
    delay: float


@tool(name="slow", description="Slow tool", input_model=SlowInput)
async def slow_tool(inp: SlowInput) -> str:
    await asyncio.sleep(inp.delay)
    return "done"


class FailInput(BaseModel):
    pass


@tool(name="fail_tool", description="Always fails", input_model=FailInput)
async def fail_tool(inp: FailInput) -> str:
    raise RuntimeError("boom")


def _make_registry(*tools) -> ToolRegistry:
    registry = ToolRegistry()
    for t in tools:
        registry.register(t)
    return registry


class TestParallelExecutor:
    async def test_single_tool_call(self):
        """Single tool call executes and returns result."""
        executor = ParallelExecutor()
        registry = _make_registry(add_tool)
        permissions = PermissionManager(mode="full")
        calls = [ToolCall(id="tc1", name="add", arguments={"a": 2, "b": 3})]

        results = await executor.execute_batch(calls, registry, permissions)

        assert len(results) == 1
        assert results[0].output == "5"
        assert results[0].is_error is False
        assert results[0].tool_call_id == "tc1"

    async def test_multiple_tools_run_in_parallel(self):
        """Multiple tool calls run concurrently, not sequentially."""
        executor = ParallelExecutor()
        registry = _make_registry(slow_tool)
        permissions = PermissionManager(mode="full")
        # Two 0.1s tasks should complete in ~0.1s if parallel, ~0.2s if sequential
        calls = [
            ToolCall(id="tc1", name="slow", arguments={"delay": 0.1}),
            ToolCall(id="tc2", name="slow", arguments={"delay": 0.1}),
        ]

        start = asyncio.get_event_loop().time()
        results = await executor.execute_batch(calls, registry, permissions)
        elapsed = asyncio.get_event_loop().time() - start

        assert len(results) == 2
        assert all(r.output == "done" for r in results)
        assert elapsed < 0.18  # parallel: ~0.1s, sequential would be ~0.2s

    async def test_permission_denied(self):
        """Permission-denied tool call returns error without executing."""
        executor = ParallelExecutor()
        registry = _make_registry(add_tool)
        permissions = PermissionManager(mode="readonly")  # add not in readonly
        calls = [ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})]

        results = await executor.execute_batch(calls, registry, permissions)

        assert len(results) == 1
        assert results[0].is_error is True
        assert "Permission denied" in results[0].output

    async def test_unknown_tool(self):
        """Unknown tool returns error result."""
        executor = ParallelExecutor()
        registry = _make_registry()  # empty
        permissions = PermissionManager(mode="full")
        calls = [ToolCall(id="tc1", name="nonexistent", arguments={})]

        results = await executor.execute_batch(calls, registry, permissions)

        assert len(results) == 1
        assert results[0].is_error is True
        assert "Unknown tool" in results[0].output

    async def test_one_failure_does_not_cancel_others(self):
        """A failing tool does not prevent other tools from completing."""
        executor = ParallelExecutor()
        registry = _make_registry(add_tool, fail_tool)
        permissions = PermissionManager(mode="full")
        calls = [
            ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
            ToolCall(id="tc2", name="fail_tool", arguments={}),
        ]

        results = await executor.execute_batch(calls, registry, permissions)

        assert len(results) == 2
        assert results[0].output == "3"
        assert results[0].is_error is False
        assert results[1].is_error is True

    async def test_preserves_order(self):
        """Results are returned in same order as input tool calls."""
        executor = ParallelExecutor()
        registry = _make_registry(add_tool)
        permissions = PermissionManager(mode="full")
        calls = [
            ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 1}),
            ToolCall(id="tc2", name="add", arguments={"a": 2, "b": 2}),
            ToolCall(id="tc3", name="add", arguments={"a": 3, "b": 3}),
        ]

        results = await executor.execute_batch(calls, registry, permissions)

        assert [r.tool_call_id for r in results] == ["tc1", "tc2", "tc3"]
        assert [r.output for r in results] == ["2", "4", "6"]

    async def test_empty_batch(self):
        """Empty batch returns empty list."""
        executor = ParallelExecutor()
        registry = _make_registry()
        permissions = PermissionManager(mode="full")

        results = await executor.execute_batch([], registry, permissions)

        assert results == []
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_execution.py -v`
Expected: FAIL (module does not exist yet)

**Step 3: Write the implementation**

```python
"""Parallel tool execution for the nexagen agent loop."""

from __future__ import annotations

import asyncio
import logging

from nexagen.models import ToolCall, ToolResult
from nexagen.tools.registry import ToolRegistry
from nexagen.permissions import PermissionManager, Allow, Deny

logger = logging.getLogger("nexagen.execution")


class ParallelExecutor:
    """Executes tool calls concurrently using asyncio.gather."""

    async def _execute_one(
        self,
        tc: ToolCall,
        tool_registry: ToolRegistry,
        permissions: PermissionManager,
    ) -> ToolResult:
        """Execute a single tool call with permission check. Never raises."""
        # Permission check
        try:
            perm = await permissions.check(tc.name, tc.arguments)
        except Exception as e:
            logger.error("Permission check failed for %s: %s", tc.name, e)
            return ToolResult(
                tool_call_id=tc.id,
                output=f"Permission check error: {type(e).__name__}",
                is_error=True,
            )

        if isinstance(perm, Deny):
            return ToolResult(
                tool_call_id=tc.id,
                output=f"Permission denied: {perm.message}",
                is_error=True,
            )

        # Lookup tool
        tool_instance = tool_registry.get(tc.name)
        if tool_instance is None:
            return ToolResult(
                tool_call_id=tc.id,
                output=f"Unknown tool: {tc.name}",
                is_error=True,
            )

        # Execute
        try:
            result = await tool_instance.execute(tc.arguments)
            result.tool_call_id = tc.id
            return result
        except Exception as e:
            logger.error("Tool '%s' crashed: %s", tc.name, e, exc_info=True)
            return ToolResult(
                tool_call_id=tc.id,
                output=f"Tool execution error: {type(e).__name__}: {e}",
                is_error=True,
            )

    async def execute_batch(
        self,
        tool_calls: list[ToolCall],
        tool_registry: ToolRegistry,
        permissions: PermissionManager,
    ) -> list[ToolResult]:
        """Run all tool calls in parallel, return results in same order."""
        if not tool_calls:
            return []

        tasks = [
            self._execute_one(tc, tool_registry, permissions)
            for tc in tool_calls
        ]
        return list(await asyncio.gather(*tasks))
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_execution.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/nexagen/execution.py tests/test_execution.py
git commit -m "feat: add ParallelExecutor for concurrent tool execution"
```

---

### Task 3: Episodic Memory

**Files:**
- Create: `src/nexagen/memory.py`
- Test: `tests/test_memory.py`

**Step 1: Write the failing tests**

```python
"""Tests for EpisodicMemory."""

from __future__ import annotations

import time
import pytest

from nexagen.memory import Episode, EpisodicMemory


class TestEpisode:
    def test_episode_creation(self):
        ep = Episode(
            task="Fix the login bug",
            outcome="success",
            tools_used=["file_read", "file_edit"],
            errors_encountered=[],
            reflections=[],
            timestamp=time.time(),
        )
        assert ep.task == "Fix the login bug"
        assert ep.outcome == "success"

    def test_episode_with_errors(self):
        ep = Episode(
            task="Deploy service",
            outcome="failure",
            tools_used=["bash"],
            errors_encountered=["ConnectionError: host unreachable"],
            reflections=["Need to check if service is running first"],
            timestamp=time.time(),
        )
        assert ep.outcome == "failure"
        assert len(ep.errors_encountered) == 1
        assert len(ep.reflections) == 1


class TestEpisodicMemory:
    def test_record_and_retrieve(self):
        mem = EpisodicMemory(max_episodes=10)
        ep = Episode(
            task="Parse CSV files",
            outcome="success",
            tools_used=["file_read", "bash"],
            errors_encountered=[],
            reflections=[],
            timestamp=time.time(),
        )
        mem.record(ep)
        results = mem.retrieve("CSV parsing", k=1)
        assert len(results) == 1
        assert results[0].task == "Parse CSV files"

    def test_retrieve_relevance_scoring(self):
        """More relevant episodes rank higher."""
        mem = EpisodicMemory(max_episodes=10)
        mem.record(Episode(
            task="Fix database connection",
            outcome="success",
            tools_used=["bash"],
            errors_encountered=[],
            reflections=[],
            timestamp=time.time() - 100,
        ))
        mem.record(Episode(
            task="Parse CSV files and validate data",
            outcome="success",
            tools_used=["file_read"],
            errors_encountered=[],
            reflections=[],
            timestamp=time.time() - 50,
        ))
        mem.record(Episode(
            task="Write unit tests for CSV parser",
            outcome="failure",
            tools_used=["file_write"],
            errors_encountered=["AssertionError"],
            reflections=["Check CSV header format"],
            timestamp=time.time(),
        ))

        results = mem.retrieve("CSV", k=2)
        # Both CSV-related episodes should be returned
        assert len(results) == 2
        assert all("CSV" in r.task or "csv" in r.task.lower() for r in results)

    def test_eviction_when_over_capacity(self):
        mem = EpisodicMemory(max_episodes=3)
        for i in range(5):
            mem.record(Episode(
                task=f"Task {i}",
                outcome="success",
                tools_used=[],
                errors_encountered=[],
                reflections=[],
                timestamp=time.time() + i,
            ))
        assert len(mem._episodes) == 3
        # Oldest should be evicted
        assert mem._episodes[0].task == "Task 2"

    def test_retrieve_empty_memory(self):
        mem = EpisodicMemory()
        results = mem.retrieve("anything", k=3)
        assert results == []

    def test_retrieve_k_larger_than_stored(self):
        mem = EpisodicMemory()
        mem.record(Episode(
            task="Only task",
            outcome="success",
            tools_used=[],
            errors_encountered=[],
            reflections=[],
            timestamp=time.time(),
        ))
        results = mem.retrieve("task", k=5)
        assert len(results) == 1

    def test_format_for_context(self):
        mem = EpisodicMemory()
        ep = Episode(
            task="Fix login bug",
            outcome="failure",
            tools_used=["file_read", "bash"],
            errors_encountered=["TypeError: None has no attribute 'id'"],
            reflections=["Need to check for None before accessing .id"],
            timestamp=time.time(),
        )
        mem.record(ep)
        text = mem.format_for_context([ep])
        assert "Fix login bug" in text
        assert "failure" in text
        assert "TypeError" in text

    def test_format_for_context_empty(self):
        mem = EpisodicMemory()
        text = mem.format_for_context([])
        assert text == ""
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_memory.py -v`
Expected: FAIL (module does not exist yet)

**Step 3: Write the implementation**

```python
"""Episodic memory for the nexagen agent — stores and retrieves past task experiences."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from nexagen.constants import DEFAULT_MAX_EPISODES


@dataclass
class Episode:
    """A structured record of a completed task."""

    task: str
    outcome: str  # "success" | "failure" | "partial"
    tools_used: list[str] = field(default_factory=list)
    errors_encountered: list[str] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class EpisodicMemory:
    """In-memory store of past task episodes with scored retrieval."""

    def __init__(self, max_episodes: int = DEFAULT_MAX_EPISODES):
        self._episodes: list[Episode] = []
        self.max_episodes = max_episodes

    def record(self, episode: Episode):
        """Store an episode. Evicts oldest if over capacity."""
        self._episodes.append(episode)
        while len(self._episodes) > self.max_episodes:
            self._episodes.pop(0)

    def _keyword_relevance(self, query: str, episode: Episode) -> float:
        """Score relevance by word overlap between query and episode fields."""
        query_words = set(query.lower().split())
        if not query_words:
            return 0.0

        episode_text = " ".join([
            episode.task,
            " ".join(episode.tools_used),
            " ".join(episode.errors_encountered),
            " ".join(episode.reflections),
        ]).lower()
        episode_words = set(episode_text.split())

        if not episode_words:
            return 0.0

        overlap = query_words & episode_words
        return len(overlap) / len(query_words)

    def _recency_score(self, episode: Episode) -> float:
        """Score recency: 1.0 for now, decays toward 0.0 for older episodes."""
        if not self._episodes:
            return 0.0
        newest = max(e.timestamp for e in self._episodes)
        oldest = min(e.timestamp for e in self._episodes)
        span = newest - oldest
        if span == 0:
            return 1.0
        return (episode.timestamp - oldest) / span

    def retrieve(self, query: str, k: int = 3) -> list[Episode]:
        """Return the k most relevant episodes.

        Scoring: recency (0.3) x keyword_relevance (0.7).
        """
        if not self._episodes:
            return []

        scored = []
        for ep in self._episodes:
            relevance = self._keyword_relevance(query, ep)
            recency = self._recency_score(ep)
            score = 0.7 * relevance + 0.3 * recency
            scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:k]]

    def format_for_context(self, episodes: list[Episode]) -> str:
        """Format retrieved episodes for injection into system prompt."""
        if not episodes:
            return ""

        lines = []
        for ep in episodes:
            parts = [f"- Task: {ep.task} (outcome: {ep.outcome})"]
            if ep.errors_encountered:
                parts.append(f"  Errors: {'; '.join(ep.errors_encountered[:2])}")
            if ep.reflections:
                parts.append(f"  Learned: {'; '.join(ep.reflections[:2])}")
            lines.append("\n".join(parts))
        return "\n".join(lines)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_memory.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/nexagen/memory.py tests/test_memory.py
git commit -m "feat: add EpisodicMemory with scored retrieval"
```

---

### Task 4: Rich Supervisor Feedback

**Files:**
- Modify: `src/nexagen/supervisor/supervisor.py:1-143`
- Modify: `tests/test_supervisor.py:1-129`

**Step 1: Write new failing tests (append to existing test file)**

```python
# --- NEW: SupervisorFeedback tests ---

from nexagen.supervisor.supervisor import SupervisorFeedback


class TestParseFeedback:
    def setup_method(self):
        self.provider = MockSupervisorProvider("")
        self.agent = SupervisorAgent(provider=self.provider)

    def test_parse_feedback_continue(self):
        result = self.agent._parse_feedback('{"decision": "continue"}')
        assert result.decision == "continue"
        assert result.diagnosis is None
        assert result.suggestion is None

    def test_parse_feedback_stop(self):
        result = self.agent._parse_feedback('{"decision": "stop", "diagnosis": "stuck in loop"}')
        assert result.decision == "stop"
        assert result.diagnosis == "stuck in loop"

    def test_parse_feedback_redirect(self):
        text = '{"decision": "redirect", "diagnosis": "wrong approach", "suggestion": "try grep instead"}'
        result = self.agent._parse_feedback(text)
        assert result.decision == "redirect"
        assert result.diagnosis == "wrong approach"
        assert result.suggestion == "try grep instead"

    def test_parse_feedback_json_in_code_block(self):
        text = '```json\n{"decision": "redirect", "suggestion": "use file_read"}\n```'
        result = self.agent._parse_feedback(text)
        assert result.decision == "redirect"
        assert result.suggestion == "use file_read"

    def test_parse_feedback_fallback_text_continue(self):
        result = self.agent._parse_feedback("The agent should continue working")
        assert result.decision == "continue"

    def test_parse_feedback_fallback_text_stop(self):
        result = self.agent._parse_feedback("The agent should stop immediately")
        assert result.decision == "stop"

    def test_parse_feedback_fallback_gibberish(self):
        result = self.agent._parse_feedback("asdfghjkl")
        assert result.decision == "stop"  # safe default

    def test_parse_feedback_invalid_decision_defaults_stop(self):
        result = self.agent._parse_feedback('{"decision": "maybe"}')
        assert result.decision == "stop"


class TestCheckProgressFeedback:
    @pytest.mark.asyncio
    async def test_check_progress_returns_feedback(self):
        provider = MockSupervisorProvider(
            '{"decision": "redirect", "diagnosis": "reading same file", "suggestion": "try grep"}'
        )
        agent = SupervisorAgent(provider=provider)
        action_log = [ActionEntry("Read config", ["file_read"])]
        result = await agent.check_progress("Fix the bug", action_log)
        assert isinstance(result, SupervisorFeedback)
        assert result.decision == "redirect"
        assert result.suggestion == "try grep"

    @pytest.mark.asyncio
    async def test_check_progress_provider_failure_returns_continue(self):
        """If the supervisor LLM fails, default to continue."""

        class FailingProvider:
            async def chat(self, messages, tools=None):
                raise ConnectionError("LLM down")
            def supports_tool_calling(self):
                return False
            def supports_vision(self):
                return False

        agent = SupervisorAgent(provider=FailingProvider())
        result = await agent.check_progress("Fix bug", [])
        assert isinstance(result, SupervisorFeedback)
        assert result.decision == "continue"
```

**Step 2: Run tests to verify new tests fail**

Run: `pytest tests/test_supervisor.py -v`
Expected: New tests FAIL (SupervisorFeedback doesn't exist yet)

**Step 3: Update the implementation**

Replace the full `src/nexagen/supervisor/supervisor.py` with:

```python
"""Supervisor agent that monitors worker progress and compresses context."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from nexagen.models import NexagenMessage, NexagenResponse, ToolCall
from nexagen.providers.base import LLMProvider

logger = logging.getLogger("nexagen.supervisor")


class ActionEntry:
    """One step in the worker's action log."""

    def __init__(self, summary: str, tool_names: list[str]):
        self.summary = summary
        self.tool_names = tool_names

    def __str__(self) -> str:
        tools = ", ".join(self.tool_names)
        return f"{self.summary} [{tools}]"


@dataclass
class SupervisorFeedback:
    """Structured feedback from the supervisor."""

    decision: str  # "continue" | "stop" | "redirect"
    diagnosis: str | None = None
    suggestion: str | None = None


class SupervisorAgent:
    """Monitors worker agent progress and handles context compression."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def _build_progress_prompt(
        self, original_task: str, action_log: list[ActionEntry]
    ) -> str:
        log_text = "\n".join(
            f"  Step {i + 1}: {entry}" for i, entry in enumerate(action_log)
        )
        return (
            f"You are a supervisor monitoring an AI agent's progress.\n\n"
            f"Original task: {original_task}\n\n"
            f"Actions taken so far:\n{log_text}\n\n"
            f"Is the agent making progress toward completing the original task?\n"
            f"Respond with JSON:\n"
            f'{{"decision": "continue"}} — if the agent is on track\n'
            f'{{"decision": "stop", "diagnosis": "reason"}} — if the agent should stop\n'
            f'{{"decision": "redirect", "diagnosis": "what is wrong", '
            f'"suggestion": "what to try instead"}} — if the agent needs course correction'
        )

    def _parse_feedback(self, response_text: str) -> SupervisorFeedback:
        """Parse supervisor response into structured feedback.

        Try JSON first, fall back to text search, default to stop.
        """
        # Try direct JSON parse
        data = self._try_parse_json(response_text)
        if data:
            decision = data.get("decision", "").lower()
            if decision in ("continue", "stop", "redirect"):
                return SupervisorFeedback(
                    decision=decision,
                    diagnosis=data.get("diagnosis"),
                    suggestion=data.get("suggestion"),
                )

        # Fallback: text search
        text_lower = response_text.lower()
        if "continue" in text_lower:
            return SupervisorFeedback(decision="continue")
        if "stop" in text_lower:
            return SupervisorFeedback(decision="stop")

        # Default to stop (safe)
        return SupervisorFeedback(decision="stop")

    def _try_parse_json(self, text: str) -> dict | None:
        """Try to extract JSON from text. Handles raw JSON and code blocks."""
        # Try direct parse
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try to find JSON in text (might be wrapped in code blocks)
        json_match = re.search(r"\{[^}]+\}", text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except (json.JSONDecodeError, TypeError):
                pass

        return None

    async def check_progress(
        self, original_task: str, action_log: list[ActionEntry]
    ) -> SupervisorFeedback:
        """Returns structured feedback: continue, stop, or redirect.

        If the supervisor LLM call fails, defaults to continue
        (non-fatal — let the worker keep going).
        """
        try:
            prompt = self._build_progress_prompt(original_task, action_log)
            messages = [NexagenMessage(role="user", text=prompt)]
            response = await self.provider.chat(messages)
            text = response.message.text if response and response.message else ""
            return self._parse_feedback(text or "")
        except Exception as e:
            logger.warning(
                "Supervisor check_progress failed: %s. Defaulting to 'continue'.", e
            )
            return SupervisorFeedback(decision="continue")

    def _build_compress_prompt(self, messages: list[NexagenMessage]) -> str:
        summaries = []
        for msg in messages:
            if msg.role == "assistant" and msg.summary:
                summaries.append(f"- {msg.summary}")
            elif msg.role == "assistant" and msg.text:
                summaries.append(f"- {msg.text[:100]}")
            elif msg.role == "tool":
                status = "error" if msg.is_error else "success"
                summaries.append(
                    f"- Tool result ({status}): {(msg.text or '')[:80]}"
                )

        content = "\n".join(summaries)
        return (
            f"Summarize these agent actions into a single concise paragraph "
            f"(2-3 sentences max) that captures the key findings and progress:\n\n"
            f"{content}"
        )

    async def compress_history(self, messages: list[NexagenMessage]) -> str:
        """Compress a list of messages into a summary string.

        If the LLM call fails, returns a basic concatenation of summaries
        rather than crashing.
        """
        try:
            prompt = self._build_compress_prompt(messages)
            response = await self.provider.chat(
                [NexagenMessage(role="user", text=prompt)]
            )
            text = response.message.text if response and response.message else None
            return text or "Summary unavailable."
        except Exception as e:
            logger.warning(
                "Supervisor compress_history failed: %s. Using fallback.", e
            )
            # Fallback: concatenate available summaries
            parts = []
            for msg in messages:
                if msg.summary:
                    parts.append(msg.summary)
                elif msg.role == "assistant" and msg.text:
                    parts.append(msg.text[:50])
            return "; ".join(parts) if parts else "Summary unavailable."
```

**Step 4: Update existing tests for backward compatibility**

The old `check_progress` tests that expected a `str` return now get `SupervisorFeedback`. Update:

In `tests/test_supervisor.py`, change `TestCheckProgress`:

```python
class TestCheckProgress:
    @pytest.mark.asyncio
    async def test_check_progress_calls_provider(self):
        provider = MockSupervisorProvider('{"decision": "continue"}')
        agent = SupervisorAgent(provider=provider)
        action_log = [ActionEntry("Read config", ["file_read"])]
        result = await agent.check_progress("Fix the bug", action_log)
        assert isinstance(result, SupervisorFeedback)
        assert result.decision == "continue"
        assert provider.last_messages is not None
        assert len(provider.last_messages) == 1
        assert provider.last_messages[0].role == "user"
```

**Step 5: Run all tests**

Run: `pytest tests/test_supervisor.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/nexagen/supervisor/supervisor.py tests/test_supervisor.py
git commit -m "feat: upgrade supervisor to structured feedback (continue/stop/redirect)"
```

---

### Task 5: Reflection Engine

**Files:**
- Create: `src/nexagen/reflection.py`
- Test: `tests/test_reflection.py`

**Step 1: Write the failing tests**

```python
"""Tests for ReflectionEngine."""

from __future__ import annotations

import pytest

from nexagen.models import NexagenMessage, NexagenResponse
from nexagen.reflection import ReflectionEngine, ReflectionResult


class MockReflectionProvider:
    """Mock provider that returns predefined text."""

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.call_count = 0

    async def chat(self, messages, tools=None):
        self.call_count += 1
        return NexagenResponse(
            message=NexagenMessage(role="assistant", text=self.response_text)
        )

    def supports_tool_calling(self):
        return False

    def supports_vision(self):
        return False


class TestReflectionResult:
    def test_creation(self):
        r = ReflectionResult(
            diagnosis="File path was wrong",
            strategy="Search with glob first",
            should_retry=True,
        )
        assert r.diagnosis == "File path was wrong"
        assert r.should_retry is True


class TestReflectionEngine:
    async def test_reflect_returns_structured_result(self):
        provider = MockReflectionProvider(
            '{"diagnosis": "wrong file path", "strategy": "use glob to find it", "should_retry": true}'
        )
        engine = ReflectionEngine(provider=provider, max_reflections=2)
        result = await engine.reflect(
            original_task="Fix the bug",
            failed_action="file_read('/wrong/path.py')",
            error="FileNotFoundError: No such file",
            past_reflections=[],
        )
        assert isinstance(result, ReflectionResult)
        assert result.should_retry is True
        assert "glob" in result.strategy.lower() or "find" in result.strategy.lower() or len(result.strategy) > 0

    async def test_reflect_with_past_reflections(self):
        provider = MockReflectionProvider(
            '{"diagnosis": "still wrong", "strategy": "ask user", "should_retry": false}'
        )
        engine = ReflectionEngine(provider=provider, max_reflections=2)
        result = await engine.reflect(
            original_task="Fix the bug",
            failed_action="file_read('/also/wrong.py')",
            error="FileNotFoundError",
            past_reflections=["Tried glob but no match found"],
        )
        assert result.should_retry is False

    async def test_reflect_provider_failure_returns_no_retry(self):
        """If reflection LLM fails, don't retry (safe default)."""

        class FailingProvider:
            async def chat(self, messages, tools=None):
                raise ConnectionError("LLM down")
            def supports_tool_calling(self):
                return False
            def supports_vision(self):
                return False

        engine = ReflectionEngine(provider=FailingProvider(), max_reflections=2)
        result = await engine.reflect(
            original_task="Fix bug",
            failed_action="bash('make test')",
            error="Error",
            past_reflections=[],
        )
        assert result.should_retry is False
        assert "unavailable" in result.diagnosis.lower() or len(result.diagnosis) > 0

    async def test_reflect_malformed_json_still_works(self):
        """Non-JSON response still produces a usable result."""
        provider = MockReflectionProvider(
            "The file path is wrong. Try searching with glob. You should retry."
        )
        engine = ReflectionEngine(provider=provider, max_reflections=2)
        result = await engine.reflect(
            original_task="Fix bug",
            failed_action="file_read",
            error="FileNotFoundError",
            past_reflections=[],
        )
        assert isinstance(result, ReflectionResult)
        # Non-JSON should still produce a result (fallback parsing)
        assert len(result.diagnosis) > 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_reflection.py -v`
Expected: FAIL (module does not exist yet)

**Step 3: Write the implementation**

```python
"""Self-reflection engine for the nexagen agent — diagnoses failures and suggests corrections."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from nexagen.models import NexagenMessage, NexagenResponse
from nexagen.providers.base import LLMProvider
from nexagen.constants import DEFAULT_MAX_REFLECTIONS

logger = logging.getLogger("nexagen.reflection")


@dataclass
class ReflectionResult:
    """Structured diagnosis and corrective strategy."""

    diagnosis: str
    strategy: str
    should_retry: bool


class ReflectionEngine:
    """Diagnoses failures and generates corrective strategies."""

    def __init__(
        self,
        provider: LLMProvider,
        max_reflections: int = DEFAULT_MAX_REFLECTIONS,
    ):
        self.provider = provider
        self.max_reflections = max_reflections

    def _build_reflection_prompt(
        self,
        original_task: str,
        failed_action: str,
        error: str,
        past_reflections: list[str],
    ) -> str:
        past = ""
        if past_reflections:
            past = "\n\nPrevious reflection attempts:\n" + "\n".join(
                f"  - {r}" for r in past_reflections
            )

        return (
            f"You are a self-reflection module for an AI agent.\n\n"
            f"Original task: {original_task}\n"
            f"Failed action: {failed_action}\n"
            f"Error: {error}\n"
            f"{past}\n\n"
            f"Analyze what went wrong and suggest a different approach.\n"
            f"Respond with JSON:\n"
            f'{{"diagnosis": "what went wrong", '
            f'"strategy": "what to try instead", '
            f'"should_retry": true/false}}'
        )

    def _parse_reflection(self, response_text: str) -> ReflectionResult:
        """Parse LLM response into ReflectionResult. Handles JSON and free text."""
        # Try JSON parse
        try:
            data = json.loads(response_text)
            return ReflectionResult(
                diagnosis=str(data.get("diagnosis", "Unknown error")),
                strategy=str(data.get("strategy", "Try a different approach")),
                should_retry=bool(data.get("should_retry", False)),
            )
        except (json.JSONDecodeError, TypeError):
            pass

        # Try to find JSON in text
        json_match = re.search(r"\{[^}]+\}", response_text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return ReflectionResult(
                    diagnosis=str(data.get("diagnosis", "Unknown error")),
                    strategy=str(data.get("strategy", "Try a different approach")),
                    should_retry=bool(data.get("should_retry", False)),
                )
            except (json.JSONDecodeError, TypeError):
                pass

        # Fallback: use the raw text as diagnosis
        should_retry = "retry" in response_text.lower() or "try" in response_text.lower()
        return ReflectionResult(
            diagnosis=response_text[:200] if response_text else "Unknown error",
            strategy=response_text[200:400] if len(response_text) > 200 else "Try a different approach",
            should_retry=should_retry,
        )

    async def reflect(
        self,
        original_task: str,
        failed_action: str,
        error: str,
        past_reflections: list[str],
    ) -> ReflectionResult:
        """Diagnose a failure and suggest a corrective strategy.

        If the LLM call fails, returns a safe default (don't retry).
        """
        try:
            prompt = self._build_reflection_prompt(
                original_task, failed_action, error, past_reflections
            )
            messages = [NexagenMessage(role="user", text=prompt)]
            response = await self.provider.chat(messages)
            text = response.message.text if response and response.message else ""
            return self._parse_reflection(text or "")
        except Exception as e:
            logger.warning("Reflection failed: %s. Defaulting to no retry.", e)
            return ReflectionResult(
                diagnosis=f"Reflection unavailable: {type(e).__name__}",
                strategy="Cannot reflect — falling back to default behavior",
                should_retry=False,
            )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_reflection.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/nexagen/reflection.py tests/test_reflection.py
git commit -m "feat: add ReflectionEngine for self-diagnosis on failures"
```

---

### Task 6: Context Manager (Observation Masking + Two-Tier Compression)

**Files:**
- Create: `src/nexagen/context.py`
- Test: `tests/test_context.py`

**Step 1: Write the failing tests**

```python
"""Tests for ContextManager (observation masking + two-tier compression)."""

from __future__ import annotations

import pytest

from nexagen.context import ContextManager
from nexagen.models import NexagenMessage, NexagenResponse, ToolCall
from nexagen.conversation import Conversation
from nexagen.supervisor.supervisor import SupervisorAgent


class TestObservationMasking:
    def test_keeps_all_assistant_messages(self):
        cm = ContextManager(recent_tool_results_to_keep=2)
        messages = [
            NexagenMessage(role="system", text="You are helpful."),
            NexagenMessage(role="user", text="Do something"),
            NexagenMessage(role="assistant", text="I'll read the file."),
            NexagenMessage(role="tool", text="file contents here..."),
            NexagenMessage(role="assistant", text="Now I'll edit it."),
            NexagenMessage(role="tool", text="edit result here..."),
            NexagenMessage(role="assistant", text="Done!"),
        ]
        masked = cm.mask_observations(messages)
        assistant_msgs = [m for m in masked if m.role == "assistant"]
        assert len(assistant_msgs) == 3
        assert all(m.text is not None and len(m.text) > 0 for m in assistant_msgs)

    def test_keeps_recent_tool_results_verbatim(self):
        cm = ContextManager(recent_tool_results_to_keep=2)
        messages = [
            NexagenMessage(role="tool", text="old result 1", tool_call_id="t1"),
            NexagenMessage(role="tool", text="old result 2", tool_call_id="t2"),
            NexagenMessage(role="tool", text="recent result 1", tool_call_id="t3"),
            NexagenMessage(role="tool", text="recent result 2", tool_call_id="t4"),
        ]
        masked = cm.mask_observations(messages)
        tool_msgs = [m for m in masked if m.role == "tool"]
        # Last 2 should be verbatim
        assert tool_msgs[-1].text == "recent result 2"
        assert tool_msgs[-2].text == "recent result 1"
        # First 2 should be stubs
        assert "old result 1" not in tool_msgs[0].text
        assert "old result 2" not in tool_msgs[1].text

    def test_masks_old_tool_results_as_stubs(self):
        cm = ContextManager(recent_tool_results_to_keep=1)
        messages = [
            NexagenMessage(role="tool", text="a" * 500, tool_call_id="t1", is_error=False),
            NexagenMessage(role="tool", text="recent", tool_call_id="t2", is_error=False),
        ]
        masked = cm.mask_observations(messages)
        tool_msgs = [m for m in masked if m.role == "tool"]
        # Old one should be short stub
        assert len(tool_msgs[0].text) < 100
        assert "success" in tool_msgs[0].text.lower() or "result" in tool_msgs[0].text.lower()
        # Recent one untouched
        assert tool_msgs[1].text == "recent"

    def test_never_touches_system_or_user_messages(self):
        cm = ContextManager(recent_tool_results_to_keep=0)
        messages = [
            NexagenMessage(role="system", text="system prompt"),
            NexagenMessage(role="user", text="user question"),
            NexagenMessage(role="tool", text="should be masked", tool_call_id="t1"),
        ]
        masked = cm.mask_observations(messages)
        assert masked[0].text == "system prompt"
        assert masked[1].text == "user question"

    def test_handles_error_tool_results(self):
        cm = ContextManager(recent_tool_results_to_keep=0)
        messages = [
            NexagenMessage(role="tool", text="FileNotFoundError: /bad/path", tool_call_id="t1", is_error=True),
        ]
        masked = cm.mask_observations(messages)
        assert "error" in masked[0].text.lower()

    def test_empty_messages(self):
        cm = ContextManager()
        assert cm.mask_observations([]) == []

    def test_no_tool_messages_returns_unchanged(self):
        cm = ContextManager()
        messages = [
            NexagenMessage(role="system", text="sys"),
            NexagenMessage(role="user", text="hi"),
            NexagenMessage(role="assistant", text="hello"),
        ]
        masked = cm.mask_observations(messages)
        assert len(masked) == 3
        assert masked[0].text == "sys"
        assert masked[1].text == "hi"
        assert masked[2].text == "hello"


class TestEstimateTokens:
    def test_estimate_messages_tokens(self):
        cm = ContextManager()
        messages = [
            NexagenMessage(role="user", text="a" * 100),
        ]
        tokens = cm.estimate_tokens(messages)
        assert tokens == 25  # 100 / 4


class TestShapeContext:
    async def test_no_shaping_needed_under_threshold(self):
        """Messages under threshold are returned unchanged."""
        cm = ContextManager()
        messages = [
            NexagenMessage(role="user", text="short message"),
        ]
        result = await cm.shape_context(messages, context_window=8192, supervisor=None)
        assert result == messages

    async def test_masking_applied_when_over_threshold(self):
        """Over threshold triggers observation masking."""
        cm = ContextManager(recent_tool_results_to_keep=1)
        # Create messages that exceed threshold
        messages = [
            NexagenMessage(role="user", text="task"),
            NexagenMessage(role="tool", text="x" * 5000, tool_call_id="t1"),
            NexagenMessage(role="tool", text="recent", tool_call_id="t2"),
        ]
        result = await cm.shape_context(messages, context_window=200, supervisor=None)
        tool_msgs = [m for m in result if m.role == "tool"]
        # Old tool result should be masked (much shorter)
        assert len(tool_msgs[0].text) < 100
        assert tool_msgs[1].text == "recent"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_context.py -v`
Expected: FAIL (module does not exist yet)

**Step 3: Write the implementation**

```python
"""Two-tier context management: observation masking + LLM compression."""

from __future__ import annotations

import logging

from nexagen.constants import (
    CHARS_PER_TOKEN,
    DEFAULT_CONTEXT_THRESHOLD,
    DEFAULT_RECENT_TOOL_RESULTS_TO_KEEP,
)
from nexagen.models import NexagenMessage

logger = logging.getLogger("nexagen.context")


class ContextManager:
    """Two-tier context management: observation masking first, LLM compression second."""

    def __init__(
        self,
        recent_tool_results_to_keep: int = DEFAULT_RECENT_TOOL_RESULTS_TO_KEEP,
    ):
        self.recent_tool_results_to_keep = recent_tool_results_to_keep

    def estimate_tokens(self, messages: list[NexagenMessage]) -> int:
        """Estimate total tokens for a list of messages."""
        try:
            total_chars = sum(
                len(msg.text or "")
                + sum(
                    len(str(tc.arguments)) + len(tc.name)
                    for tc in (msg.tool_calls or [])
                )
                for msg in messages
            )
            return max(0, total_chars // max(1, CHARS_PER_TOKEN))
        except (TypeError, AttributeError):
            return len(messages) * 100

    def mask_observations(
        self, messages: list[NexagenMessage]
    ) -> list[NexagenMessage]:
        """Tier 1: Strip old tool result content, keep reasoning intact.

        Rules:
        - Keep all assistant messages intact
        - Keep the last N tool results verbatim
        - Replace older tool results with a short stub
        - Never touch system or user messages
        """
        if not messages:
            return []

        # Find indices of tool messages
        tool_indices = [
            i for i, m in enumerate(messages) if m.role == "tool"
        ]

        if not tool_indices:
            return list(messages)

        # Indices of tool messages to keep verbatim (the most recent N)
        keep_indices = set(tool_indices[-self.recent_tool_results_to_keep:])

        result = []
        for i, msg in enumerate(messages):
            if msg.role != "tool" or i in keep_indices:
                result.append(msg)
            else:
                # Mask this tool result
                status = "error" if msg.is_error else "success"
                stub = f"[Tool result: {status}]"
                result.append(NexagenMessage(
                    role="tool",
                    text=stub,
                    tool_call_id=msg.tool_call_id,
                    is_error=msg.is_error,
                ))

        return result

    async def shape_context(
        self,
        messages: list[NexagenMessage],
        context_window: int,
        supervisor=None,
    ) -> list[NexagenMessage]:
        """Shape context using two-tier strategy.

        Tier 1: Observation masking (free, instant)
        Tier 2: LLM compression via supervisor (expensive, if still over limit)
        """
        threshold = int(context_window * DEFAULT_CONTEXT_THRESHOLD)

        # Check if shaping is needed
        if self.estimate_tokens(messages) < threshold:
            return messages

        # Tier 1: observation masking
        masked = self.mask_observations(messages)

        # Check if masking was sufficient
        if self.estimate_tokens(masked) < threshold:
            return masked

        # Tier 2: LLM compression (only if supervisor available)
        if supervisor is None:
            logger.warning("Context still over threshold after masking, but no supervisor for compression.")
            return masked

        try:
            # Get compressible messages (everything except first + last 3)
            if len(masked) <= 4:
                return masked

            compressible = masked[1:-3]
            summary = await supervisor.compress_history(compressible)
            summary_msg = NexagenMessage(
                role="assistant", text=summary, summary=summary
            )
            return [masked[0], summary_msg] + masked[-3:]
        except Exception as e:
            logger.warning("Tier 2 compression failed: %s. Using masked context.", e)
            return masked
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_context.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/nexagen/context.py tests/test_context.py
git commit -m "feat: add ContextManager with observation masking and two-tier compression"
```

---

### Task 7: Planning Phase

**Files:**
- Create: `src/nexagen/planning.py`
- Test: `tests/test_planning.py`

**Step 1: Write the failing tests**

```python
"""Tests for PlanningPhase."""

from __future__ import annotations

import pytest

from nexagen.models import NexagenMessage, NexagenResponse
from nexagen.planning import PlanningPhase, Plan, Subtask
from nexagen.supervisor.supervisor import ActionEntry


class MockPlanProvider:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.call_count = 0
        self.last_messages = None

    async def chat(self, messages, tools=None):
        self.last_messages = messages
        self.call_count += 1
        return NexagenResponse(
            message=NexagenMessage(role="assistant", text=self.response_text)
        )

    def supports_tool_calling(self):
        return False

    def supports_vision(self):
        return False


class TestSubtask:
    def test_default_status(self):
        st = Subtask(description="Read the config file")
        assert st.status == "pending"

    def test_custom_status(self):
        st = Subtask(description="Done", status="completed")
        assert st.status == "completed"


class TestPlan:
    def test_plan_creation(self):
        plan = Plan(subtasks=[
            Subtask(description="Step 1"),
            Subtask(description="Step 2"),
        ])
        assert len(plan.subtasks) == 2
        assert plan.current_step == 0

    def test_advance_step(self):
        plan = Plan(subtasks=[
            Subtask(description="Step 1"),
            Subtask(description="Step 2"),
        ])
        plan.advance()
        assert plan.current_step == 1
        assert plan.subtasks[0].status == "completed"

    def test_advance_does_not_exceed_bounds(self):
        plan = Plan(subtasks=[Subtask(description="Only step")])
        plan.advance()
        plan.advance()  # should not crash
        assert plan.current_step == 1

    def test_is_complete(self):
        plan = Plan(subtasks=[Subtask(description="Only step")])
        assert plan.is_complete is False
        plan.advance()
        assert plan.is_complete is True


class TestPlanFormatContext:
    def test_format_plan_context(self):
        plan = Plan(subtasks=[
            Subtask(description="Read config", status="completed"),
            Subtask(description="Fix bug", status="in_progress"),
            Subtask(description="Run tests", status="pending"),
        ], current_step=1)
        phase = PlanningPhase(provider=MockPlanProvider(""))
        text = phase.format_plan_context(plan)
        assert "Read config" in text
        assert "Fix bug" in text
        assert "Run tests" in text


class TestClassifyComplexity:
    async def test_classify_simple(self):
        provider = MockPlanProvider('{"complexity": "simple"}')
        phase = PlanningPhase(provider=provider)
        result = await phase.classify_complexity("What time is it?")
        assert result == "simple"

    async def test_classify_complex(self):
        provider = MockPlanProvider('{"complexity": "complex"}')
        phase = PlanningPhase(provider=provider)
        result = await phase.classify_complexity("Refactor the auth module to use JWT tokens")
        assert result == "complex"

    async def test_classify_fallback_on_bad_json(self):
        provider = MockPlanProvider("This is a complex task")
        phase = PlanningPhase(provider=provider)
        result = await phase.classify_complexity("Build a REST API")
        assert result in ("simple", "complex")

    async def test_classify_provider_failure_defaults_simple(self):
        class FailingProvider:
            async def chat(self, messages, tools=None):
                raise ConnectionError("down")
            def supports_tool_calling(self):
                return False
            def supports_vision(self):
                return False

        phase = PlanningPhase(provider=FailingProvider())
        result = await phase.classify_complexity("anything")
        assert result == "simple"


class TestGeneratePlan:
    async def test_generate_plan(self):
        provider = MockPlanProvider(
            '{"subtasks": ["Read the config file", "Identify the bug", "Write fix", "Run tests"]}'
        )
        phase = PlanningPhase(provider=provider)
        plan = await phase.generate_plan("Fix the config bug")
        assert isinstance(plan, Plan)
        assert len(plan.subtasks) >= 1

    async def test_generate_plan_provider_failure_returns_single_step(self):
        class FailingProvider:
            async def chat(self, messages, tools=None):
                raise ConnectionError("down")
            def supports_tool_calling(self):
                return False
            def supports_vision(self):
                return False

        phase = PlanningPhase(provider=FailingProvider())
        plan = await phase.generate_plan("Fix something")
        assert isinstance(plan, Plan)
        assert len(plan.subtasks) == 1  # fallback single-step plan
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_planning.py -v`
Expected: FAIL (module does not exist yet)

**Step 3: Write the implementation**

```python
"""Structured planning phase — auto-detects complexity and generates execution plans."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from nexagen.models import NexagenMessage, NexagenResponse
from nexagen.providers.base import LLMProvider

logger = logging.getLogger("nexagen.planning")


@dataclass
class Subtask:
    """A single step in an execution plan."""

    description: str
    status: str = "pending"  # "pending" | "in_progress" | "completed" | "failed"


@dataclass
class Plan:
    """An ordered list of subtasks with progress tracking."""

    subtasks: list[Subtask] = field(default_factory=list)
    current_step: int = 0

    def advance(self):
        """Mark current step as completed and move to the next."""
        if self.current_step < len(self.subtasks):
            self.subtasks[self.current_step].status = "completed"
            self.current_step = min(self.current_step + 1, len(self.subtasks))

    @property
    def is_complete(self) -> bool:
        return self.current_step >= len(self.subtasks)


class PlanningPhase:
    """Auto-detects task complexity and generates execution plans."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    async def classify_complexity(self, prompt: str) -> str:
        """Returns 'simple' or 'complex'.

        Uses a short LLM call. Defaults to 'simple' on failure.
        """
        try:
            classification_prompt = (
                "Classify this task as simple or complex.\n"
                "Simple: single step, direct answer, no file changes, quick lookup.\n"
                "Complex: multiple steps, research, file changes, debugging.\n\n"
                f"Task: {prompt}\n\n"
                'Respond with JSON: {"complexity": "simple"} or {"complexity": "complex"}'
            )
            messages = [NexagenMessage(role="user", text=classification_prompt)]
            response = await self.provider.chat(messages)
            text = response.message.text if response and response.message else ""
            return self._parse_complexity(text or "")
        except Exception as e:
            logger.warning("Complexity classification failed: %s. Defaulting to simple.", e)
            return "simple"

    def _parse_complexity(self, text: str) -> str:
        """Parse complexity from LLM response."""
        # Try JSON
        try:
            data = json.loads(text)
            c = data.get("complexity", "").lower()
            if c in ("simple", "complex"):
                return c
        except (json.JSONDecodeError, TypeError):
            pass

        # Try JSON in code blocks
        json_match = re.search(r"\{[^}]+\}", text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                c = data.get("complexity", "").lower()
                if c in ("simple", "complex"):
                    return c
            except (json.JSONDecodeError, TypeError):
                pass

        # Text fallback
        text_lower = text.lower()
        if "complex" in text_lower:
            return "complex"
        return "simple"

    async def generate_plan(self, prompt: str) -> Plan:
        """Decompose a task into ordered subtasks.

        Returns a single-step fallback plan on failure.
        """
        try:
            plan_prompt = (
                "Break this task into 2-6 ordered subtasks.\n\n"
                f"Task: {prompt}\n\n"
                'Respond with JSON: {"subtasks": ["step 1 description", "step 2 description", ...]}'
            )
            messages = [NexagenMessage(role="user", text=plan_prompt)]
            response = await self.provider.chat(messages)
            text = response.message.text if response and response.message else ""
            return self._parse_plan(text or "", prompt)
        except Exception as e:
            logger.warning("Plan generation failed: %s. Using single-step fallback.", e)
            return Plan(subtasks=[Subtask(description=prompt)])

    def _parse_plan(self, text: str, original_prompt: str) -> Plan:
        """Parse plan from LLM response."""
        # Try JSON
        for candidate in [text, None]:
            source = candidate
            if source is None:
                match = re.search(r"\{[^}]*\[.*?\][^}]*\}", text, re.DOTALL)
                if match:
                    source = match.group()
                else:
                    break
            try:
                data = json.loads(source)
                steps = data.get("subtasks", [])
                if isinstance(steps, list) and len(steps) > 0:
                    return Plan(
                        subtasks=[Subtask(description=str(s)) for s in steps]
                    )
            except (json.JSONDecodeError, TypeError):
                continue

        # Fallback: single-step plan
        return Plan(subtasks=[Subtask(description=original_prompt)])

    def format_plan_context(self, plan: Plan) -> str:
        """Format current plan state for injection into system prompt."""
        lines = []
        for i, st in enumerate(plan.subtasks):
            if st.status == "completed":
                marker = "[x]"
            elif i == plan.current_step:
                marker = "[>]"
            else:
                marker = "[ ]"
            lines.append(f"{marker} Step {i + 1}: {st.description}")
        return "\n".join(lines)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_planning.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/nexagen/planning.py tests/test_planning.py
git commit -m "feat: add PlanningPhase with auto-detect complexity and plan generation"
```

---

### Task 8: Update Conversation for Append-Only + KV-Cache Awareness

**Files:**
- Modify: `src/nexagen/conversation.py:1-82`
- Modify: `tests/test_conversation.py:1-131`

**Step 1: Write new failing tests**

Append these to `tests/test_conversation.py`:

```python
from nexagen.context import ContextManager


class TestGetMessagesForLLM:
    def test_returns_messages_without_mutation(self):
        """get_messages_for_llm does not mutate the source list."""
        conv = Conversation()
        conv.add_message(NexagenMessage(role="user", text="hello"))
        conv.add_message(NexagenMessage(role="tool", text="x" * 5000, tool_call_id="t1"))
        conv.add_message(NexagenMessage(role="tool", text="recent", tool_call_id="t2"))

        original_count = len(conv.messages)
        cm = ContextManager(recent_tool_results_to_keep=1)
        result = conv.get_messages_for_llm(
            system_prompt="sys",
            context_manager=cm,
            context_window=200,
        )
        # Source messages unchanged
        assert len(conv.messages) == original_count
        assert conv.messages[1].text == "x" * 5000  # not masked

    def test_without_context_manager_returns_raw(self):
        """Without a context manager, returns raw messages (backward compat)."""
        conv = Conversation()
        conv.add_message(NexagenMessage(role="user", text="hello"))
        result = conv.get_messages_for_llm(system_prompt="sys")
        assert len(result) == 2  # system + user
        assert result[0].role == "system"
        assert result[1].text == "hello"

    def test_append_only_messages_never_mutated(self):
        """Adding messages never modifies existing entries."""
        conv = Conversation()
        msg1 = NexagenMessage(role="user", text="first")
        conv.add_message(msg1)
        msg2 = NexagenMessage(role="assistant", text="second")
        conv.add_message(msg2)
        assert conv.messages[0] is msg1  # same reference, not copied
        assert conv.messages[1] is msg2
```

**Step 2: Run tests to verify new tests fail**

Run: `pytest tests/test_conversation.py::TestGetMessagesForLLM -v`
Expected: FAIL (method doesn't exist yet)

**Step 3: Update the Conversation class**

Add the `get_messages_for_llm` method to `src/nexagen/conversation.py`. Keep all existing methods intact for backward compatibility:

```python
"""Conversation management for the nexagen SDK."""

from __future__ import annotations

from nexagen.constants import CHARS_PER_TOKEN, DEFAULT_CONTEXT_THRESHOLD, DEFAULT_COMPRESS_TARGET
from nexagen.models import NexagenMessage


class Conversation:
    """Manages message history, token estimation, context compression, and task summaries."""

    def __init__(self, context_window: int = 8192):
        self.messages: list[NexagenMessage] = []
        self.task_summaries: list[str] = []
        self.context_window = context_window  # in tokens

    def add_message(self, message: NexagenMessage):
        """Add a single message to the conversation (append-only)."""
        self.messages.append(message)

    def add_messages(self, messages: list[NexagenMessage]):
        """Add multiple messages to the conversation (append-only)."""
        self.messages.extend(messages)

    def estimate_tokens(self) -> int:
        """Estimate the total token count of all messages using chars / CHARS_PER_TOKEN."""
        try:
            total_chars = sum(
                len(msg.text or "")
                + sum(
                    len(str(tc.arguments)) + len(tc.name)
                    for tc in (msg.tool_calls or [])
                )
                for msg in self.messages
            )
            return max(0, total_chars // max(1, CHARS_PER_TOKEN))
        except (TypeError, AttributeError):
            # Malformed messages — estimate conservatively
            return len(self.messages) * 100

    def needs_compression(self) -> bool:
        """Return True if estimated tokens exceed the context threshold."""
        try:
            return self.estimate_tokens() >= int(self.context_window * DEFAULT_CONTEXT_THRESHOLD)
        except (TypeError, ValueError):
            return False

    def get_compressible_messages(self) -> list[NexagenMessage]:
        """Returns messages that can be compressed (everything except first user msg + last 3)."""
        if len(self.messages) <= 4:
            return []
        return self.messages[1:-3]

    def compress(self, summary: str):
        """Replace compressible messages with a summary message.

        NOTE: Legacy method kept for backward compatibility.
        Prefer using ContextManager.shape_context() for new code.
        """
        if len(self.messages) <= 4:
            return
        first = self.messages[0]
        last_three = self.messages[-3:]
        summary_msg = NexagenMessage(role="assistant", text=summary, summary=summary)
        self.messages = [first, summary_msg] + last_three

    def complete_task(self, summary: str):
        """Called when a task completes. Stores summary and resets messages."""
        self.task_summaries.append(summary)
        self.messages = []

    def get_messages_with_history(self, system_prompt: str | None = None) -> list[NexagenMessage]:
        """Get messages including task summaries from previous tasks.

        NOTE: Legacy method. Prefer get_messages_for_llm() for new code.
        """
        result: list[NexagenMessage] = []
        if system_prompt:
            result.append(NexagenMessage(role="system", text=system_prompt))
        for summary in self.task_summaries:
            result.append(NexagenMessage(role="assistant", text=summary, summary=summary))
        result.extend(self.messages)
        return result

    def get_messages_for_llm(
        self,
        system_prompt: str | None = None,
        context_manager=None,
        context_window: int | None = None,
        supervisor=None,
    ) -> list[NexagenMessage]:
        """Build a shaped view of messages for the LLM. Never mutates the source.

        If context_manager is provided, applies observation masking.
        This is the KV-cache-aware path: the source messages list stays append-only
        and shaping happens at read time only.
        """
        result: list[NexagenMessage] = []
        if system_prompt:
            result.append(NexagenMessage(role="system", text=system_prompt))
        for summary in self.task_summaries:
            result.append(NexagenMessage(role="assistant", text=summary, summary=summary))
        result.extend(self.messages)

        if context_manager is not None:
            window = context_window or self.context_window
            threshold = int(window * DEFAULT_CONTEXT_THRESHOLD)
            tokens = context_manager.estimate_tokens(result)
            if tokens >= threshold:
                result = context_manager.mask_observations(result)

        return result

    def clear(self):
        """Reset all messages and task summaries."""
        self.messages = []
        self.task_summaries = []
```

**Step 4: Run all conversation tests**

Run: `pytest tests/test_conversation.py -v`
Expected: All PASS (old + new)

**Step 5: Commit**

```bash
git add src/nexagen/conversation.py tests/test_conversation.py
git commit -m "feat: add append-only get_messages_for_llm with context shaping"
```

---

### Task 9: Wire Everything into Agent

**Files:**
- Modify: `src/nexagen/agent.py:1-303`
- Modify: `tests/test_agent.py:1-269`

This is the integration task that brings all 7 improvements together.

**Step 1: Write new failing tests**

Append these to `tests/test_agent.py`:

```python
from nexagen.supervisor.supervisor import SupervisorFeedback


class TestParallelExecution:
    async def test_parallel_tool_calls(self):
        """Multiple tool calls in one response execute in parallel."""
        provider = MockProvider([
            NexagenResponse(
                message=NexagenMessage(
                    role="assistant",
                    text="Adding both.",
                    tool_calls=[
                        ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
                        ToolCall(id="tc2", name="add", arguments={"a": 3, "b": 4}),
                    ],
                )
            ),
            _text_response("Results are 3 and 7."),
        ])
        agent = Agent(
            provider=provider,
            custom_tools=[add_tool],
            permission_mode="full",
        )

        messages = []
        async for msg in agent.run("Add 1+2 and 3+4"):
            messages.append(msg)

        tool_results = [m for m in messages if m.role == "tool"]
        assert len(tool_results) == 2
        outputs = {m.text for m in tool_results}
        assert outputs == {"3", "7"}


class TestSupervisorRedirect:
    async def test_supervisor_redirect_injects_hint(self):
        """Supervisor 'redirect' injects a hint and continues the loop."""

        call_count = 0

        class RedirectSupervisorProvider:
            async def chat(self, messages, tools=None):
                nonlocal call_count
                call_count += 1
                return NexagenResponse(
                    message=NexagenMessage(
                        role="assistant",
                        text='{"decision": "redirect", "diagnosis": "wrong approach", "suggestion": "try grep"}',
                    )
                )
            def supports_tool_calling(self):
                return True
            def supports_vision(self):
                return False

        provider = MockProvider([
            _tool_call_response("Trying.", "add", "tc1", {"a": 1, "b": 1}),
            _tool_call_response("Trying.", "add", "tc2", {"a": 1, "b": 1}),
            _tool_call_response("Trying.", "add", "tc3", {"a": 1, "b": 1}),
            _tool_call_response("Trying.", "add", "tc4", {"a": 1, "b": 1}),
            _tool_call_response("Trying.", "add", "tc5", {"a": 1, "b": 1}),
            _text_response("Done after redirect."),
        ])

        agent = Agent(
            provider=provider,
            custom_tools=[add_tool],
            permission_mode="full",
            supervisor=RedirectSupervisorProvider(),
            supervisor_check_interval=5,
        )

        messages = []
        async for msg in agent.run("Keep trying"):
            messages.append(msg)

        # Should NOT have stopped — redirect means continue with hint
        last = messages[-1]
        assert "Done after redirect" in last.text


class TestAgentMemory:
    async def test_episodic_memory_records_episodes(self):
        """After a run, the agent records an episode in memory."""
        provider = MockProvider([_text_response("Done.")])
        agent = Agent(provider=provider, permission_mode="full")

        async for _ in agent.run("Test task"):
            pass

        assert len(agent.memory._episodes) == 1
        assert agent.memory._episodes[0].task == "Test task"
        assert agent.memory._episodes[0].outcome == "success"

    async def test_memory_persists_across_runs(self):
        """Episodes accumulate across multiple run() calls."""
        provider = MockProvider([_text_response("Done.")])
        agent = Agent(provider=provider, permission_mode="full")

        async for _ in agent.run("First task"):
            pass
        async for _ in agent.run("Second task"):
            pass

        assert len(agent.memory._episodes) == 2
```

**Step 2: Run tests to verify new tests fail**

Run: `pytest tests/test_agent.py::TestParallelExecution tests/test_agent.py::TestSupervisorRedirect tests/test_agent.py::TestAgentMemory -v`
Expected: FAIL

**Step 3: Rewrite agent.py with all improvements integrated**

```python
"""Core Agent loop for the nexagen SDK."""

from __future__ import annotations

import json
import logging
import time
from typing import AsyncIterator, Callable, Awaitable

import httpx

from nexagen.models import NexagenMessage, NexagenResponse, ProviderConfig, ToolResult
from nexagen.providers.registry import ProviderRegistry
from nexagen.providers.base import LLMProvider
from nexagen.tools.base import BaseTool
from nexagen.tools.registry import ToolRegistry
from nexagen.tools.builtin import BUILTIN_TOOLS
from nexagen.permissions import PermissionManager, Allow, Deny
from nexagen.conversation import Conversation
from nexagen.execution import ParallelExecutor
from nexagen.context import ContextManager
from nexagen.reflection import ReflectionEngine, ReflectionResult
from nexagen.planning import PlanningPhase, Plan
from nexagen.memory import EpisodicMemory, Episode
from nexagen.supervisor.supervisor import SupervisorAgent, SupervisorFeedback, ActionEntry
from nexagen.constants import (
    DEFAULT_PERMISSION_MODE,
    DEFAULT_SUPERVISOR_CHECK_INTERVAL,
    DEFAULT_MAX_TOOL_ERRORS,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_REFLECTIONS,
    DEFAULT_MAX_EPISODES,
    DEFAULT_RECENT_TOOL_RESULTS_TO_KEEP,
)

logger = logging.getLogger("nexagen.agent")


class Agent:
    """Agentic loop that orchestrates LLM calls, tool execution, permissions, and supervision."""

    def __init__(
        self,
        provider: str | ProviderConfig | LLMProvider,
        tools: list[str] | None = None,
        custom_tools: list[BaseTool] | None = None,
        system_prompt: str | None = None,
        permission_mode: str = DEFAULT_PERMISSION_MODE,
        allowed_tools: list[str] | None = None,
        can_use_tool: Callable[[str, dict], Awaitable[Allow | Deny]] | None = None,
        supervisor: LLMProvider | None = None,
        supervisor_check_interval: int = DEFAULT_SUPERVISOR_CHECK_INTERVAL,
        max_tool_errors: int = DEFAULT_MAX_TOOL_ERRORS,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        max_reflections: int = DEFAULT_MAX_REFLECTIONS,
        max_episodes: int = DEFAULT_MAX_EPISODES,
        recent_tool_results_to_keep: int = DEFAULT_RECENT_TOOL_RESULTS_TO_KEEP,
    ):
        # Resolve provider
        if isinstance(provider, (str, ProviderConfig)):
            registry = ProviderRegistry()
            from nexagen.providers.openai_compat import OpenAICompatProvider

            for backend in ["ollama", "vllm", "lmstudio", "groq", "together", "openai_compat"]:
                registry.register(backend, OpenAICompatProvider)
            try:
                from nexagen.providers.anthropic_provider import AnthropicProvider
                registry.register("anthropic", AnthropicProvider)
            except ImportError:
                pass
            try:
                from nexagen.providers.openai_native import OpenAINativeProvider
                registry.register("openai", OpenAINativeProvider)
            except ImportError:
                pass
            try:
                from nexagen.providers.google_provider import GoogleProvider
                registry.register("google", GoogleProvider)
            except ImportError:
                pass
            self.provider = registry.resolve(provider)
        else:
            self.provider = provider

        # Setup tool registry
        self.tool_registry = ToolRegistry()
        builtin_names = tools or []
        for name in builtin_names:
            if name in BUILTIN_TOOLS:
                self.tool_registry.register(BUILTIN_TOOLS[name])
        if custom_tools:
            self.tool_registry.register_many(custom_tools)

        # Frozen system prompt (KV-cache-aware: never mutated after init)
        self._frozen_system_prompt = (
            (system_prompt or "You are a helpful AI assistant.")
            + "\n\nWhen you use tools, always include a brief one-sentence summary "
            "of what you're trying to accomplish in your response text."
        )

        self.permissions = PermissionManager(
            mode=permission_mode,
            allowed_tools=allowed_tools,
            can_use_tool=can_use_tool,
        )

        # Supervisor
        self.supervisor = SupervisorAgent(supervisor) if supervisor else None
        self.supervisor_check_interval = supervisor_check_interval
        self.max_tool_errors = max_tool_errors
        self.max_iterations = max_iterations

        # New modules
        self.executor = ParallelExecutor()
        self.context_manager = ContextManager(
            recent_tool_results_to_keep=recent_tool_results_to_keep,
        )
        self.reflection_engine = (
            ReflectionEngine(provider=supervisor, max_reflections=max_reflections)
            if supervisor
            else None
        )
        self.planning = (
            PlanningPhase(provider=supervisor)
            if supervisor
            else None
        )
        self.memory = EpisodicMemory(max_episodes=max_episodes)

    async def run(
        self, prompt: str, conversation: Conversation | None = None
    ) -> AsyncIterator[NexagenMessage]:
        """Run the agent loop, yielding messages as they are produced."""
        conv = conversation or Conversation()

        # --- EPISODIC MEMORY: retrieve relevant past experiences ---
        system_prompt = self._frozen_system_prompt
        relevant_episodes = self.memory.retrieve(prompt, k=3)
        if relevant_episodes:
            memory_context = self.memory.format_for_context(relevant_episodes)
            if memory_context:
                system_prompt = (
                    system_prompt
                    + "\n\n## Relevant Past Experience\n"
                    + memory_context
                )

        # --- PLANNING: auto-detect complexity and generate plan ---
        plan: Plan | None = None
        if self.planning:
            complexity = await self.planning.classify_complexity(prompt)
            if complexity == "complex":
                plan = await self.planning.generate_plan(prompt)
                plan_context = self.planning.format_plan_context(plan)
                system_prompt = (
                    system_prompt + "\n\n## Execution Plan\n" + plan_context
                )

        # --- BUILD CONTEXT (KV-cache-aware: frozen prefix + append-only) ---
        messages = conv.get_messages_for_llm(
            system_prompt=system_prompt,
            context_manager=self.context_manager,
            context_window=conv.context_window,
        )

        # Add user message
        user_msg = NexagenMessage(role="user", text=prompt)
        messages.append(user_msg)
        conv.add_message(user_msg)

        tool_schemas = self.tool_registry.get_tool_schemas()
        action_log: list[ActionEntry] = []
        total_tool_calls = 0
        consecutive_errors: dict[str, int] = {}
        reflection_counts: dict[str, int] = {}
        all_reflections: list[str] = []
        errors_encountered: list[str] = []
        tools_used_set: set[str] = set()

        iterations = 0

        while True:
            iterations += 1
            if iterations > self.max_iterations:
                yield NexagenMessage(
                    role="assistant",
                    text=f"Stopping: reached maximum iteration limit ({self.max_iterations}).",
                )
                self._record_episode(prompt, "partial", tools_used_set, errors_encountered, all_reflections)
                conv.complete_task("Task stopped: max iterations reached")
                return

            # --- CONTEXT SHAPING (two-tier: mask then compress) ---
            messages = conv.get_messages_for_llm(
                system_prompt=system_prompt,
                context_manager=self.context_manager,
                context_window=conv.context_window,
            )

            # Call LLM
            try:
                response = await self.provider.chat(
                    messages, tool_schemas if tool_schemas else None
                )
            except httpx.HTTPStatusError as e:
                error_text = f"LLM request failed: HTTP {e.response.status_code}"
                logger.error("Provider HTTP error: %s", e)
                yield NexagenMessage(role="assistant", text=error_text)
                self._record_episode(prompt, "failure", tools_used_set, errors_encountered, all_reflections)
                conv.complete_task(f"Task failed: {error_text}")
                return
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                error_text = f"LLM connection failed: {type(e).__name__}"
                logger.error("Provider connection error: %s", e)
                yield NexagenMessage(role="assistant", text=error_text)
                self._record_episode(prompt, "failure", tools_used_set, errors_encountered, all_reflections)
                conv.complete_task(f"Task failed: {error_text}")
                return
            except Exception as e:
                error_text = f"LLM error: {type(e).__name__}: {e}"
                logger.error("Unexpected provider error: %s", e, exc_info=True)
                yield NexagenMessage(role="assistant", text=error_text)
                self._record_episode(prompt, "failure", tools_used_set, errors_encountered, all_reflections)
                conv.complete_task(f"Task failed: {error_text}")
                return

            assistant_msg = response.message
            conv.add_message(assistant_msg)

            # Yield the assistant message
            yield assistant_msg

            # No tool calls = done
            if not response.has_tool_calls:
                break

            # --- PARALLEL TOOL EXECUTION ---
            results = await self.executor.execute_batch(
                assistant_msg.tool_calls, self.tool_registry, self.permissions
            )

            tool_names_this_cycle: list[str] = []
            for tc, result in zip(assistant_msg.tool_calls, results):
                tool_names_this_cycle.append(tc.name)
                tools_used_set.add(tc.name)
                total_tool_calls += 1

                # Track consecutive errors
                if result.is_error:
                    consecutive_errors[tc.name] = consecutive_errors.get(tc.name, 0) + 1
                    errors_encountered.append(f"{tc.name}: {result.output[:100]}")
                else:
                    consecutive_errors[tc.name] = 0

                # --- SELF-REFLECTION on repeated errors ---
                if (
                    consecutive_errors.get(tc.name, 0) >= self.max_tool_errors
                    and self.reflection_engine
                ):
                    tool_reflection_count = reflection_counts.get(tc.name, 0)
                    if tool_reflection_count < self.reflection_engine.max_reflections:
                        reflection = await self.reflection_engine.reflect(
                            original_task=prompt,
                            failed_action=f"{tc.name}({json.dumps(tc.arguments, sort_keys=True)})",
                            error=result.output,
                            past_reflections=all_reflections,
                        )
                        reflection_counts[tc.name] = tool_reflection_count + 1
                        all_reflections.append(reflection.diagnosis)

                        if reflection.should_retry:
                            # Inject reflection so LLM can read its own diagnosis
                            reflection_msg = NexagenMessage(
                                role="assistant",
                                text=(
                                    f"[Reflection] {reflection.diagnosis} "
                                    f"New strategy: {reflection.strategy}"
                                ),
                            )
                            conv.add_message(reflection_msg)
                            yield reflection_msg
                            consecutive_errors[tc.name] = 0  # reset for retry
                        else:
                            # Reflection says don't retry — escalate to supervisor
                            if self.supervisor:
                                feedback = await self.supervisor.check_progress(prompt, action_log)
                                if feedback.decision == "stop":
                                    yield NexagenMessage(
                                        role="assistant",
                                        text=(
                                            f"Stopping: tool '{tc.name}' failed "
                                            f"{self.max_tool_errors} times. "
                                            f"Diagnosis: {reflection.diagnosis}"
                                        ),
                                    )
                                    self._record_episode(prompt, "failure", tools_used_set, errors_encountered, all_reflections)
                                    conv.complete_task(f"Task stopped: {reflection.diagnosis}")
                                    return
                    else:
                        # Max reflections exhausted — escalate to supervisor
                        if self.supervisor:
                            feedback = await self.supervisor.check_progress(prompt, action_log)
                            if feedback.decision == "stop":
                                yield NexagenMessage(
                                    role="assistant",
                                    text=(
                                        f"Stopping: tool '{tc.name}' failed repeatedly, "
                                        f"reflections exhausted."
                                    ),
                                )
                                self._record_episode(prompt, "failure", tools_used_set, errors_encountered, all_reflections)
                                conv.complete_task(f"Task stopped due to repeated errors with {tc.name}")
                                return

                # Add tool result to conversation
                tool_msg = result.to_message()
                conv.add_message(tool_msg)
                yield tool_msg

            # Record action log entry
            summary_text = assistant_msg.summary or assistant_msg.text or "No summary"
            if len(summary_text) > 100:
                summary_text = summary_text[:100] + "..."
            action_log.append(
                ActionEntry(summary=summary_text, tool_names=tool_names_this_cycle)
            )

            # --- PLAN PROGRESS ---
            if plan and not plan.is_complete:
                plan.advance()

            # --- SUPERVISOR CHECK every N tool calls (with redirect support) ---
            if (
                self.supervisor
                and total_tool_calls % self.supervisor_check_interval == 0
                and total_tool_calls > 0
            ):
                try:
                    feedback = await self.supervisor.check_progress(prompt, action_log)
                except Exception as e:
                    logger.warning("Supervisor progress check failed: %s. Continuing.", e)
                    feedback = SupervisorFeedback(decision="continue")

                if feedback.decision == "stop":
                    yield NexagenMessage(
                        role="assistant",
                        text="Stopping: supervisor determined insufficient progress.",
                    )
                    self._record_episode(prompt, "partial", tools_used_set, errors_encountered, all_reflections)
                    conv.complete_task("Task stopped by supervisor")
                    return
                elif feedback.decision == "redirect" and feedback.suggestion:
                    hint_msg = NexagenMessage(
                        role="assistant",
                        text=f"[Supervisor hint: {feedback.suggestion}]",
                    )
                    conv.add_message(hint_msg)
                    yield hint_msg

        # --- TASK COMPLETE ---
        final_text = assistant_msg.text or "Task completed"
        if len(final_text) > 150:
            final_text = final_text[:150] + "..."
        conv.complete_task(final_text)
        self._record_episode(prompt, "success", tools_used_set, errors_encountered, all_reflections)

    def _record_episode(
        self,
        task: str,
        outcome: str,
        tools_used: set[str],
        errors: list[str],
        reflections: list[str],
    ):
        """Record a completed task as an episodic memory."""
        self.memory.record(Episode(
            task=task,
            outcome=outcome,
            tools_used=list(tools_used),
            errors_encountered=errors[:5],  # cap to avoid bloat
            reflections=reflections[:5],
            timestamp=time.time(),
        ))
```

**Step 4: Update existing test for supervisor that now returns SupervisorFeedback**

In `tests/test_agent.py`, update `TestAgent.test_supervisor_stops`:

The `MockSupervisorProvider` returns `'{"decision": "stop"}'` which will be parsed into `SupervisorFeedback(decision="stop")`. The test should still pass since the agent checks `feedback.decision == "stop"`.

**Step 5: Run all agent tests**

Run: `pytest tests/test_agent.py -v`
Expected: All PASS (old + new)

**Step 6: Run the full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All PASS

**Step 7: Commit**

```bash
git add src/nexagen/agent.py tests/test_agent.py
git commit -m "feat: integrate all 7 orchestration improvements into Agent loop

- Parallel tool execution via ParallelExecutor
- Two-tier context management (observation masking + LLM compression)
- Self-reflection engine for error diagnosis and retry
- Auto-detect planning phase for complex tasks
- Episodic memory for cross-task learning
- Rich supervisor feedback with redirect support
- KV-cache-aware frozen system prompt and append-only context"
```

---

### Task 10: Update Public API Exports

**Files:**
- Modify: `src/nexagen/__init__.py:1-38`

**Step 1: Update exports**

```python
# src/nexagen/__init__.py
"""nexagen — Universal LLM Agent SDK."""

from nexagen.agent import Agent
from nexagen.models import (
    NexagenMessage,
    NexagenResponse,
    ProviderConfig,
    ToolCall,
    ToolResult,
)
from nexagen.tools.base import tool, BaseTool
from nexagen.tools.registry import ToolRegistry
from nexagen.tools.mcp import MCPServerConfig, MCPTool, MCPManager
from nexagen.conversation import Conversation
from nexagen.permissions import Allow, Deny, PermissionManager
from nexagen.supervisor import SupervisorAgent, ActionEntry
from nexagen.supervisor.supervisor import SupervisorFeedback
from nexagen.execution import ParallelExecutor
from nexagen.context import ContextManager
from nexagen.reflection import ReflectionEngine, ReflectionResult
from nexagen.planning import PlanningPhase, Plan, Subtask
from nexagen.memory import EpisodicMemory, Episode

__all__ = [
    "Agent",
    "NexagenMessage",
    "NexagenResponse",
    "ProviderConfig",
    "ToolCall",
    "ToolResult",
    "tool",
    "BaseTool",
    "ToolRegistry",
    "MCPServerConfig",
    "MCPTool",
    "MCPManager",
    "Conversation",
    "Allow",
    "Deny",
    "PermissionManager",
    "SupervisorAgent",
    "SupervisorFeedback",
    "ActionEntry",
    "ParallelExecutor",
    "ContextManager",
    "ReflectionEngine",
    "ReflectionResult",
    "PlanningPhase",
    "Plan",
    "Subtask",
    "EpisodicMemory",
    "Episode",
]
```

**Step 2: Run public API tests**

Run: `pytest tests/test_public_api.py -v`
Expected: PASS

**Step 3: Run the full test suite one final time**

Run: `pytest tests/ -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add src/nexagen/__init__.py
git commit -m "feat: export new orchestration modules from public API"
```

---

### Task 11: Final Verification

**Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=long`
Expected: All PASS

**Step 2: Run linter**

Run: `ruff check src/ tests/`
Expected: No errors

**Step 3: Verify import works**

Run: `python -c "from nexagen import Agent, ParallelExecutor, ContextManager, ReflectionEngine, PlanningPhase, EpisodicMemory, SupervisorFeedback; print('All imports OK')"`
Expected: `All imports OK`
