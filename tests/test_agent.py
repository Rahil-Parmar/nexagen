"""Tests for the Agent class and its agentic loop."""

from __future__ import annotations

import pytest

from nexagen.agent import Agent
from nexagen.models import NexagenMessage, NexagenResponse, ToolCall, ToolResult
from nexagen.tools.base import tool, BaseTool
from nexagen.conversation import Conversation
from nexagen.permissions import Allow, Deny
from pydantic import BaseModel


class MockProvider:
    """Mock LLM provider that returns predefined responses in sequence."""

    def __init__(self, responses: list[NexagenResponse]):
        self.responses = responses
        self.call_count = 0
        self.last_messages = None
        self.last_tools = None

    async def chat(self, messages, tools=None):
        self.last_messages = messages
        self.last_tools = tools
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return response

    def supports_tool_calling(self):
        return True

    def supports_vision(self):
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_response(text: str) -> NexagenResponse:
    """Create a simple text-only NexagenResponse."""
    return NexagenResponse(
        message=NexagenMessage(role="assistant", text=text)
    )


def _tool_call_response(
    text: str | None, tool_name: str, tool_id: str, arguments: dict
) -> NexagenResponse:
    """Create a NexagenResponse that contains a single tool call."""
    return NexagenResponse(
        message=NexagenMessage(
            role="assistant",
            text=text,
            tool_calls=[ToolCall(id=tool_id, name=tool_name, arguments=arguments)],
        )
    )


# A simple custom tool for testing
class AddInput(BaseModel):
    a: int
    b: int


@tool(name="add", description="Add two numbers", input_model=AddInput)
async def add_tool(inp: AddInput) -> str:
    return str(inp.a + inp.b)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgent:
    async def test_simple_response(self):
        """Provider returns text only -- agent yields one message, loop ends."""
        provider = MockProvider([_text_response("Hello there!")])
        agent = Agent(provider=provider, permission_mode="full")

        messages = []
        async for msg in agent.run("Hi"):
            messages.append(msg)

        assert len(messages) == 1
        assert messages[0].role == "assistant"
        assert messages[0].text == "Hello there!"
        assert provider.call_count == 1

    async def test_tool_call_and_response(self):
        """Provider returns tool call, then text. Agent executes tool, yields results."""
        provider = MockProvider(
            [
                _tool_call_response("Let me add those.", "add", "tc1", {"a": 2, "b": 3}),
                _text_response("The answer is 5."),
            ]
        )
        agent = Agent(
            provider=provider,
            custom_tools=[add_tool],
            permission_mode="full",
        )

        messages = []
        async for msg in agent.run("What is 2 + 3?"):
            messages.append(msg)

        # Expected: assistant (tool call), tool result, assistant (final)
        assert len(messages) == 3
        assert messages[0].role == "assistant"
        assert messages[0].tool_calls is not None
        assert messages[1].role == "tool"
        assert messages[1].text == "5"
        assert messages[1].is_error is False
        assert messages[2].role == "assistant"
        assert messages[2].text == "The answer is 5."
        assert provider.call_count == 2

    async def test_permission_denied(self):
        """Tool call blocked by permission mode yields error tool result."""
        provider = MockProvider(
            [
                _tool_call_response("Running bash.", "bash", "tc1", {"command": "ls"}),
                _text_response("I cannot run that."),
            ]
        )
        # readonly mode only allows file_read, grep, glob
        agent = Agent(
            provider=provider,
            custom_tools=[add_tool],
            permission_mode="readonly",
        )

        messages = []
        async for msg in agent.run("Run bash"):
            messages.append(msg)

        # assistant (tool call), tool result (denied), assistant (final)
        assert len(messages) == 3
        assert messages[1].role == "tool"
        assert messages[1].is_error is True
        assert "Permission denied" in messages[1].text

    async def test_unknown_tool(self):
        """LLM calls a tool that doesn't exist -- yields error tool result."""
        provider = MockProvider(
            [
                _tool_call_response("Using magic.", "nonexistent", "tc1", {}),
                _text_response("That didn't work."),
            ]
        )
        agent = Agent(provider=provider, permission_mode="full")

        messages = []
        async for msg in agent.run("Do magic"):
            messages.append(msg)

        assert len(messages) == 3
        assert messages[1].role == "tool"
        assert messages[1].is_error is True
        assert "Unknown tool" in messages[1].text

    async def test_custom_tool(self):
        """Register a custom @tool, LLM calls it, verify execution and result."""
        provider = MockProvider(
            [
                _tool_call_response("Adding.", "add", "tc1", {"a": 10, "b": 20}),
                _text_response("Result is 30."),
            ]
        )
        agent = Agent(
            provider=provider,
            custom_tools=[add_tool],
            permission_mode="full",
        )

        messages = []
        async for msg in agent.run("Add 10 and 20"):
            messages.append(msg)

        assert messages[1].role == "tool"
        assert messages[1].text == "30"
        assert messages[1].is_error is False

    async def test_conversation_continuity(self):
        """Run two prompts with same Conversation, verify task summaries carry over."""
        conv = Conversation()

        # First run
        provider1 = MockProvider([_text_response("First task done.")])
        agent1 = Agent(provider=provider1, permission_mode="full")
        async for _ in agent1.run("First task", conversation=conv):
            pass

        assert len(conv.task_summaries) == 1
        assert conv.task_summaries[0] == "First task done."

        # Second run -- provider should receive messages including prior task summary
        provider2 = MockProvider([_text_response("Second task done.")])
        agent2 = Agent(provider=provider2, permission_mode="full")
        async for _ in agent2.run("Second task", conversation=conv):
            pass

        assert len(conv.task_summaries) == 2
        assert conv.task_summaries[1] == "Second task done."

        # Verify the provider received the prior summary in messages
        assert provider2.last_messages is not None
        summaries_in_messages = [
            m for m in provider2.last_messages
            if m.role == "assistant" and m.summary is not None
        ]
        assert len(summaries_in_messages) == 1
        assert summaries_in_messages[0].summary == "First task done."

    async def test_supervisor_stops(self):
        """Supervisor returns 'stop', agent stops early."""

        class MockSupervisorProvider:
            """Always tells the agent to stop."""

            async def chat(self, messages, tools=None):
                return NexagenResponse(
                    message=NexagenMessage(
                        role="assistant",
                        text='{"decision": "stop"}',
                    )
                )

            def supports_tool_calling(self):
                return True

            def supports_vision(self):
                return False

        # Provider always returns tool calls so the loop would never end
        # without supervisor intervention.
        provider = MockProvider(
            [
                _tool_call_response("Trying.", "add", "tc1", {"a": 1, "b": 1}),
                _tool_call_response("Trying again.", "add", "tc2", {"a": 1, "b": 1}),
                _tool_call_response("Trying more.", "add", "tc3", {"a": 1, "b": 1}),
                _tool_call_response("Trying yet more.", "add", "tc4", {"a": 1, "b": 1}),
                _tool_call_response("Still trying.", "add", "tc5", {"a": 1, "b": 1}),
                _text_response("Should not reach here."),
            ]
        )

        supervisor_provider = MockSupervisorProvider()
        agent = Agent(
            provider=provider,
            custom_tools=[add_tool],
            permission_mode="full",
            supervisor=supervisor_provider,
            supervisor_check_interval=5,  # check after 5 tool calls
        )

        messages = []
        async for msg in agent.run("Keep adding"):
            messages.append(msg)

        # The agent should have been stopped by supervisor
        last_msg = messages[-1]
        assert "Stopping" in last_msg.text
        assert "supervisor" in last_msg.text.lower()
