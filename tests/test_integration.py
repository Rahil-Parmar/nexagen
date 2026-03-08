"""End-to-end integration tests using a MockProvider."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from nexagen.agent import Agent
from nexagen.conversation import Conversation
from nexagen.models import NexagenMessage, NexagenResponse, ToolCall, ToolResult
from nexagen.tools.base import tool


# ---------------------------------------------------------------------------
# MockProvider
# ---------------------------------------------------------------------------

class MockProvider:
    def __init__(self, responses: list[NexagenResponse]):
        self.responses = responses
        self.call_count = 0

    async def chat(self, messages, tools=None):
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
    return NexagenResponse(message=NexagenMessage(role="assistant", text=text))


def _tool_call_response(
    tool_calls: list[ToolCall], text: str | None = None
) -> NexagenResponse:
    return NexagenResponse(
        message=NexagenMessage(role="assistant", text=text, tool_calls=tool_calls)
    )


async def _collect(agent: Agent, prompt: str, conversation: Conversation | None = None) -> list[NexagenMessage]:
    msgs: list[NexagenMessage] = []
    async for msg in agent.run(prompt, conversation=conversation):
        msgs.append(msg)
    return msgs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_agent_reads_file(tmp_path):
    """Agent uses file_read tool to read a temp file and returns the contents."""
    # Create temp file with known content
    test_file = tmp_path / "hello.txt"
    test_file.write_text("Hello, nexagen!")

    provider = MockProvider([
        # First response: LLM asks to call file_read
        _tool_call_response([
            ToolCall(id="tc1", name="file_read", arguments={"file_path": str(test_file)})
        ]),
        # Second response: LLM returns text mentioning the content
        _text_response("The file contains: Hello, nexagen!"),
    ])

    agent = Agent(
        provider=provider,
        tools=["file_read"],
        permission_mode="full",
    )

    msgs = await _collect(agent, "Read the file")

    # Should yield: assistant(tool_call) -> tool(result) -> assistant(text)
    assert len(msgs) == 3
    # First message is the assistant with tool call
    assert msgs[0].tool_calls is not None
    assert msgs[0].tool_calls[0].name == "file_read"
    # Second message is the tool result
    assert msgs[1].role == "tool"
    assert "Hello, nexagen!" in msgs[1].text
    assert not msgs[1].is_error
    # Third message is the final text
    assert msgs[2].role == "assistant"
    assert "Hello, nexagen!" in msgs[2].text


async def test_agent_permission_blocks_bash():
    """Agent with permission_mode='readonly' tries bash, gets denied."""
    provider = MockProvider([
        _tool_call_response([
            ToolCall(id="tc1", name="bash", arguments={"command": "echo hi"})
        ]),
        _text_response("I could not run the command."),
    ])

    agent = Agent(
        provider=provider,
        tools=["bash", "file_read"],
        permission_mode="readonly",
    )

    msgs = await _collect(agent, "Run echo hi")

    # assistant(tool_call) -> tool(error) -> assistant(text)
    assert len(msgs) == 3
    tool_result = msgs[1]
    assert tool_result.role == "tool"
    assert tool_result.is_error is True
    assert "Permission denied" in tool_result.text


async def test_agent_custom_tool():
    """Agent with a custom @tool processes input correctly."""

    class CalcInput(BaseModel):
        a: int
        b: int

    @tool(name="calculator", description="Add two numbers", input_model=CalcInput)
    async def calculator(args: CalcInput) -> str:
        return str(args.a + args.b)

    provider = MockProvider([
        _tool_call_response([
            ToolCall(id="tc1", name="calculator", arguments={"a": 17, "b": 25})
        ]),
        _text_response("The sum is 42."),
    ])

    agent = Agent(
        provider=provider,
        custom_tools=[calculator],
        permission_mode="full",
    )

    msgs = await _collect(agent, "Add 17 and 25")

    assert len(msgs) == 3
    tool_result = msgs[1]
    assert tool_result.role == "tool"
    assert "42" in tool_result.text
    assert not tool_result.is_error


async def test_conversation_across_tasks():
    """Run two prompts with same Conversation, verify history carries over."""
    conv = Conversation()

    # First run
    provider1 = MockProvider([_text_response("First task done.")])
    agent1 = Agent(provider=provider1, permission_mode="full")
    msgs1 = await _collect(agent1, "Do the first task", conversation=conv)
    assert len(msgs1) == 1
    assert msgs1[0].text == "First task done."

    # After first run, conversation should have a task summary
    assert len(conv.task_summaries) == 1

    # Second run
    provider2 = MockProvider([_text_response("Second task done.")])
    agent2 = Agent(provider=provider2, permission_mode="full")

    # Verify that get_messages_with_history includes the summary
    history = conv.get_messages_with_history("You are helpful.")
    summary_msgs = [m for m in history if m.summary is not None]
    assert len(summary_msgs) >= 1
    assert "First task done." in summary_msgs[0].summary

    msgs2 = await _collect(agent2, "Do the second task", conversation=conv)
    assert len(msgs2) == 1
    assert msgs2[0].text == "Second task done."

    # Now conv should have two summaries
    assert len(conv.task_summaries) == 2


async def test_agent_handles_tool_error():
    """Tool raises exception on nonexistent file; error is fed back to LLM."""
    provider = MockProvider([
        # LLM tries to read a file that doesn't exist
        _tool_call_response([
            ToolCall(
                id="tc1",
                name="file_read",
                arguments={"file_path": "/tmp/nexagen_nonexistent_99999.txt"},
            )
        ]),
        # LLM responds after seeing the error
        _text_response("The file does not exist."),
    ])

    agent = Agent(
        provider=provider,
        tools=["file_read"],
        permission_mode="full",
    )

    msgs = await _collect(agent, "Read a missing file")

    assert len(msgs) == 3
    tool_result = msgs[1]
    assert tool_result.role == "tool"
    assert tool_result.is_error is True


async def test_agent_with_multiple_tools(tmp_path):
    """Agent uses multiple tools in sequence (file_read then grep)."""
    # Create temp files
    test_file = tmp_path / "data.txt"
    test_file.write_text("line1: alpha\nline2: beta\nline3: alpha\n")

    provider = MockProvider([
        # LLM calls file_read first, then grep
        _tool_call_response([
            ToolCall(id="tc1", name="file_read", arguments={"file_path": str(test_file)}),
            ToolCall(
                id="tc2",
                name="grep",
                arguments={"pattern": "alpha", "path": str(test_file)},
            ),
        ]),
        _text_response("Found 2 lines with alpha."),
    ])

    agent = Agent(
        provider=provider,
        tools=["file_read", "grep"],
        permission_mode="full",
    )

    msgs = await _collect(agent, "Read and search file")

    # assistant(tool_calls) -> tool(file_read result) -> tool(grep result) -> assistant(text)
    assert len(msgs) == 4

    # First is assistant with two tool calls
    assert msgs[0].tool_calls is not None
    assert len(msgs[0].tool_calls) == 2

    # Second is file_read result
    assert msgs[1].role == "tool"
    assert "alpha" in msgs[1].text
    assert not msgs[1].is_error

    # Third is grep result
    assert msgs[2].role == "tool"
    assert "alpha" in msgs[2].text
    assert not msgs[2].is_error

    # Fourth is final text
    assert msgs[3].role == "assistant"


async def test_full_workflow(tmp_path):
    """Complete workflow: read file -> edit file -> read again -> text response."""
    test_file = tmp_path / "config.txt"
    test_file.write_text("setting=old_value\n")

    provider = MockProvider([
        # Step 1: read file
        _tool_call_response([
            ToolCall(id="tc1", name="file_read", arguments={"file_path": str(test_file)})
        ]),
        # Step 2: edit file
        _tool_call_response([
            ToolCall(
                id="tc2",
                name="file_edit",
                arguments={
                    "file_path": str(test_file),
                    "old_string": "setting=old_value",
                    "new_string": "setting=new_value",
                },
            )
        ]),
        # Step 3: read file again to verify
        _tool_call_response([
            ToolCall(id="tc3", name="file_read", arguments={"file_path": str(test_file)})
        ]),
        # Step 4: final text
        _text_response("Done. Updated setting to new_value."),
    ])

    agent = Agent(
        provider=provider,
        tools=["file_read", "file_edit"],
        permission_mode="full",
    )

    msgs = await _collect(agent, "Update the config")

    # 3 cycles of (assistant + tool_result) + 1 final assistant = 7 messages
    assert len(msgs) == 7

    # Verify file was actually edited on disk
    final_content = test_file.read_text()
    assert "setting=new_value" in final_content
    assert "old_value" not in final_content

    # Verify the last read got the updated content
    last_tool_result = msgs[5]  # tool result from the third file_read
    assert last_tool_result.role == "tool"
    assert "new_value" in last_tool_result.text
