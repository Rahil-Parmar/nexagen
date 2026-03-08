"""Resilience tests — verify the SDK handles failures gracefully without crashing.

These tests simulate real-world failure modes:
- LLM provider goes down mid-conversation
- Provider returns garbage responses
- Tool execution crashes unexpectedly
- Supervisor is unreachable
- Malformed API responses
- Network timeouts
"""

import asyncio

import httpx
import pytest

from nexagen import Agent, NexagenMessage, NexagenResponse, ToolCall, ToolResult, Conversation
from nexagen.tools.base import tool, BaseTool
from nexagen.supervisor.supervisor import SupervisorAgent, ActionEntry
from pydantic import BaseModel


# ── Helpers ───────────────────────────────────────────────────


class FailingProvider:
    """Provider that raises on the Nth call."""

    def __init__(self, fail_on: int, error: Exception, responses_before: list[NexagenResponse] | None = None):
        self.fail_on = fail_on
        self.error = error
        self.responses_before = responses_before or []
        self.call_count = 0

    async def chat(self, messages, tools=None):
        self.call_count += 1
        if self.call_count == self.fail_on:
            raise self.error
        idx = min(self.call_count - 1, len(self.responses_before) - 1)
        if idx >= 0:
            return self.responses_before[idx]
        return NexagenResponse(message=NexagenMessage(role="assistant", text="ok"))

    def supports_tool_calling(self):
        return True

    def supports_vision(self):
        return False


class GarbageProvider:
    """Provider that returns malformed responses."""

    def __init__(self, response: NexagenResponse):
        self.response = response

    async def chat(self, messages, tools=None):
        return self.response

    def supports_tool_calling(self):
        return True

    def supports_vision(self):
        return False


class MockProvider:
    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0

    async def chat(self, messages, tools=None):
        r = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return r

    def supports_tool_calling(self):
        return True

    def supports_vision(self):
        return False


class CrashingInput(BaseModel):
    value: str


@tool("crasher", "A tool that always crashes", input_model=CrashingInput)
async def crasher_tool(args: CrashingInput) -> str:
    raise RuntimeError("Unexpected internal error!")


class TimeoutInput(BaseModel):
    seconds: int = 1


@tool("slow_tool", "A tool that takes forever", input_model=TimeoutInput)
async def slow_tool(args: TimeoutInput) -> str:
    await asyncio.sleep(args.seconds)
    return "done"


# ── Test: Provider HTTP failure mid-conversation ──────────────


async def test_agent_survives_http_error():
    """Agent yields error message and stops gracefully when provider returns HTTP error."""
    provider = FailingProvider(
        fail_on=1,
        error=httpx.HTTPStatusError(
            "Server Error",
            request=httpx.Request("POST", "http://test"),
            response=httpx.Response(500),
        ),
    )
    agent = Agent(provider=provider)
    messages = [msg async for msg in agent.run("Hello")]

    assert len(messages) >= 1
    assert any("failed" in (m.text or "").lower() or "error" in (m.text or "").lower() for m in messages)


async def test_agent_survives_connection_error():
    """Agent handles connection refused gracefully."""
    provider = FailingProvider(
        fail_on=1,
        error=httpx.ConnectError("Connection refused"),
    )
    agent = Agent(provider=provider)
    messages = [msg async for msg in agent.run("Hello")]

    assert len(messages) >= 1
    assert any("connection" in (m.text or "").lower() or "error" in (m.text or "").lower() for m in messages)


async def test_agent_survives_timeout():
    """Agent handles LLM timeout gracefully."""
    provider = FailingProvider(
        fail_on=1,
        error=httpx.ReadTimeout("Request timed out"),
    )
    agent = Agent(provider=provider)
    messages = [msg async for msg in agent.run("Hello")]

    assert len(messages) >= 1
    assert any("failed" in (m.text or "").lower() or "timeout" in (m.text or "").lower() for m in messages)


# ── Test: Provider fails mid-tool-loop ────────────────────────


async def test_agent_survives_failure_after_tool_call(tmp_path):
    """Provider works for first call (tool call), crashes on second (after tool result)."""
    f = tmp_path / "test.txt"
    f.write_text("hello")

    provider = FailingProvider(
        fail_on=2,
        error=httpx.ConnectError("Server went away"),
        responses_before=[
            NexagenResponse(message=NexagenMessage(
                role="assistant",
                text="Reading file",
                tool_calls=[ToolCall(id="1", name="file_read", arguments={"file_path": str(f)})],
            )),
        ],
    )
    agent = Agent(provider=provider, tools=["file_read"], permission_mode="full")
    messages = [msg async for msg in agent.run("Read the file")]

    # Should have: assistant msg, tool result, then error message
    roles = [m.role for m in messages]
    assert "assistant" in roles
    assert "tool" in roles
    # Agent should have gracefully stopped
    assert any("error" in (m.text or "").lower() or "failed" in (m.text or "").lower() for m in messages)


# ── Test: Tool crashes don't kill the agent ───────────────────


async def test_agent_survives_crashing_tool():
    """A tool that throws an exception doesn't crash the agent loop."""
    provider = MockProvider([
        NexagenResponse(message=NexagenMessage(
            role="assistant",
            text="Using crasher",
            tool_calls=[ToolCall(id="1", name="crasher", arguments={"value": "test"})],
        )),
        NexagenResponse(message=NexagenMessage(
            role="assistant",
            text="The tool failed, but I'm still here.",
        )),
    ])
    agent = Agent(provider=provider, custom_tools=[crasher_tool], permission_mode="full")
    messages = [msg async for msg in agent.run("Use the crasher")]

    tool_results = [m for m in messages if m.role == "tool"]
    assert any(m.is_error for m in tool_results)
    # Agent should continue and return a final message
    assistant_msgs = [m for m in messages if m.role == "assistant"]
    assert len(assistant_msgs) >= 2


# ── Test: Supervisor failure doesn't crash agent ──────────────


async def test_agent_continues_when_supervisor_fails():
    """If supervisor LLM is down, agent continues without it."""
    failing_supervisor = FailingProvider(
        fail_on=1,
        error=httpx.ConnectError("Supervisor unreachable"),
    )

    # Worker provider returns tool calls to trigger supervisor check
    worker = MockProvider([
        NexagenResponse(message=NexagenMessage(
            role="assistant", text="Working",
            tool_calls=[ToolCall(id=str(i), name="crasher", arguments={"value": "x"})],
        )) for i in range(6)  # enough to trigger supervisor check
    ] + [
        NexagenResponse(message=NexagenMessage(role="assistant", text="Done")),
    ])

    agent = Agent(
        provider=worker,
        custom_tools=[crasher_tool],
        permission_mode="full",
        supervisor=failing_supervisor,
        supervisor_check_interval=3,
        max_iterations=10,
    )
    messages = [msg async for msg in agent.run("Do something")]

    # Should NOT crash — supervisor failure is non-fatal for progress checks
    assert len(messages) > 0


# ── Test: Conversation survives malformed messages ────────────


def test_conversation_handles_none_text():
    """Token estimation handles messages with None text."""
    conv = Conversation()
    conv.add_message(NexagenMessage(role="assistant", text=None, tool_calls=None))
    conv.add_message(NexagenMessage(role="user", text=None))
    tokens = conv.estimate_tokens()
    assert tokens >= 0


def test_conversation_handles_empty():
    """Empty conversation doesn't crash."""
    conv = Conversation()
    assert conv.estimate_tokens() == 0
    assert not conv.needs_compression()
    assert conv.get_compressible_messages() == []


# ── Test: Supervisor parse handles garbage ────────────────────


def test_supervisor_handles_none_response():
    """Supervisor doesn't crash on None text."""
    sup = SupervisorAgent.__new__(SupervisorAgent)
    assert sup._parse_feedback("").decision == "stop"
    assert sup._parse_feedback(None or "").decision == "stop"


async def test_supervisor_check_progress_handles_crash():
    """check_progress returns 'continue' when LLM crashes."""
    failing = FailingProvider(fail_on=1, error=RuntimeError("boom"))
    sup = SupervisorAgent(failing)
    result = await sup.check_progress("task", [ActionEntry("step", ["tool"])])
    assert result.decision == "continue"  # non-fatal fallback


async def test_supervisor_compress_handles_crash():
    """compress_history returns fallback when LLM crashes."""
    failing = FailingProvider(fail_on=1, error=RuntimeError("boom"))
    sup = SupervisorAgent(failing)
    messages = [
        NexagenMessage(role="assistant", text="Did something", summary="Read a file"),
    ]
    result = await sup.compress_history(messages)
    assert len(result) > 0  # fallback should return something
    assert "Read a file" in result  # should use available summaries


# ── Test: Agent handles max iterations ────────────────────────


async def test_agent_stops_at_max_iterations():
    """Agent with infinite tool calls stops at max_iterations."""
    infinite_provider = MockProvider([
        NexagenResponse(message=NexagenMessage(
            role="assistant", text="Again",
            tool_calls=[ToolCall(id="1", name="crasher", arguments={"value": "x"})],
        )),
    ] * 100)

    agent = Agent(
        provider=infinite_provider,
        custom_tools=[crasher_tool],
        permission_mode="full",
        max_iterations=5,
    )
    messages = [msg async for msg in agent.run("Loop forever")]

    # Should have stopped — not 100 iterations
    assert any("maximum iteration" in (m.text or "").lower() for m in messages)


# ── Test: MCP disconnect cascade ──────────────────────────────


async def test_mcp_disconnect_survives_failure():
    """disconnect_all continues even if one tool fails to disconnect."""
    from nexagen.tools.mcp import MCPManager, MCPTool, MCPServerConfig

    class FailingMCPTool(MCPTool):
        async def disconnect(self):
            raise RuntimeError("Disconnect failed!")

    manager = MCPManager()
    config = MCPServerConfig(command="test")
    good_tool = MCPTool("good", "Good tool", {}, config)
    bad_tool = FailingMCPTool("bad", "Bad tool", {}, config)
    await good_tool.connect()
    await bad_tool.connect()
    manager.register_tool("server", good_tool)
    manager.register_tool("server", bad_tool)

    # Should NOT raise — continues disconnecting remaining tools
    await manager.disconnect_all()
    assert not good_tool.is_available()


# ── Test: Provider config validation ──────────────────────────


def test_provider_config_rejects_empty():
    from nexagen.models import ProviderConfig
    with pytest.raises(ValueError):
        ProviderConfig.from_string("")


def test_provider_config_rejects_no_slash():
    from nexagen.models import ProviderConfig
    with pytest.raises(ValueError):
        ProviderConfig.from_string("just-a-string")


def test_provider_config_rejects_file_protocol():
    """file:// URLs should not be usable as base URLs."""
    from nexagen.models import ProviderConfig
    # When host doesn't start with http, it gets http:// prefix
    # The resulting URL http://file:///etc/passwd is nonsensical but not dangerous
    # (it would fail to connect). More important: direct file:// should be blocked.
    # Test that using file:// as a properly-formed base_url is rejected.
    with pytest.raises(ValueError):
        ProviderConfig(backend="evil", model="model", base_url="file:///etc/passwd")


def test_provider_config_rejects_ftp_protocol_direct():
    """Direct ftp:// base_url should be rejected."""
    from nexagen.models import ProviderConfig
    with pytest.raises(ValueError):
        ProviderConfig(backend="evil", model="model", base_url="ftp://evil.com")
