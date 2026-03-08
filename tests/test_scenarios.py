"""
Sanity check scenarios for nexagen SDK.
Run with: uv run python -m pytest tests/test_scenarios.py -v
"""

import asyncio
import os
import tempfile

import pytest
from pydantic import BaseModel

from nexagen import (
    Agent,
    NexagenMessage,
    NexagenResponse,
    ProviderConfig,
    ToolCall,
    ToolResult,
    tool,
    BaseTool,
    ToolRegistry,
    MCPServerConfig,
    MCPTool,
    MCPManager,
    Conversation,
    Allow,
    Deny,
    PermissionManager,
    SupervisorAgent,
    ActionEntry,
)
from nexagen.tools.builtin import BUILTIN_TOOLS


# ── Helpers ──────────────────────────────────────────────────


class MockProvider:
    """Mock LLM provider returning predefined responses in sequence."""

    def __init__(self, responses: list[NexagenResponse]):
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


class CalcInput(BaseModel):
    a: int
    b: int
    op: str = "add"


@tool("calc", "Calculator", input_model=CalcInput)
async def calc_tool(args: CalcInput) -> str:
    if args.op == "add":
        return str(args.a + args.b)
    if args.op == "mul":
        return str(args.a * args.b)
    return "unknown op"


# ── Scenario 1: Public API imports ───────────────────────────


def test_scenario_public_api_imports():
    """All public symbols are importable."""
    assert Agent is not None
    assert NexagenMessage is not None
    assert NexagenResponse is not None
    assert ProviderConfig is not None
    assert ToolCall is not None
    assert ToolResult is not None
    assert tool is not None
    assert BaseTool is not None
    assert ToolRegistry is not None
    assert MCPServerConfig is not None
    assert MCPTool is not None
    assert MCPManager is not None
    assert Conversation is not None
    assert Allow is not None
    assert Deny is not None
    assert PermissionManager is not None
    assert SupervisorAgent is not None
    assert ActionEntry is not None


# ── Scenario 2: Provider string parsing ──────────────────────


@pytest.mark.parametrize(
    "provider_str, expected_backend, expected_model, expected_url",
    [
        ("ollama/qwen3", "ollama", "qwen3", None),
        ("openai/gpt-4o", "openai", "gpt-4o", None),
        ("anthropic/claude-sonnet-4-20250514", "anthropic", "claude-sonnet-4-20250514", None),
        ("ollama/qwen3@192.168.1.5:11434", "ollama", "qwen3", "http://192.168.1.5:11434"),
        ("vllm/mistral@http://myserver:8000", "vllm", "mistral", "http://myserver:8000"),
    ],
)
def test_scenario_provider_string_parsing(provider_str, expected_backend, expected_model, expected_url):
    config = ProviderConfig.from_string(provider_str)
    assert config.backend == expected_backend
    assert config.model == expected_model
    assert config.base_url == expected_url


# ── Scenario 3: Custom tool with Pydantic ────────────────────


async def test_scenario_custom_tool_valid_input():
    result = await calc_tool.execute({"a": 3, "b": 7})
    assert not result.is_error
    assert result.output == "10"


async def test_scenario_custom_tool_multiply():
    result = await calc_tool.execute({"a": 3, "b": 7, "op": "mul"})
    assert not result.is_error
    assert result.output == "21"


async def test_scenario_custom_tool_validation_error():
    result = await calc_tool.execute({"wrong": "field"})
    assert result.is_error


# ── Scenario 4: Permission system layers ─────────────────────


async def test_scenario_permissions_safe_mode_with_allowlist():
    pm = PermissionManager(mode="safe", allowed_tools=["file_read", "grep"])
    assert isinstance(await pm.check("file_read", {}), Allow)
    assert isinstance(await pm.check("bash", {}), Deny)  # blocked by mode
    assert isinstance(await pm.check("file_write", {}), Deny)  # blocked by allowlist


async def test_scenario_permissions_callback():
    async def block_rm(tool_name, args):
        if tool_name == "bash" and "rm" in args.get("command", ""):
            return Deny("Destructive command")
        return Allow()

    pm = PermissionManager(mode="full", can_use_tool=block_rm)
    assert isinstance(await pm.check("bash", {"command": "ls"}), Allow)
    assert isinstance(await pm.check("bash", {"command": "rm -rf /"}), Deny)


# ── Scenario 5: Conversation management ──────────────────────


def test_scenario_conversation_task_summaries():
    conv = Conversation(context_window=1000)
    conv.add_message(NexagenMessage(role="user", text="Hello"))
    conv.add_message(NexagenMessage(role="assistant", text="Hi there"))
    assert conv.estimate_tokens() > 0
    assert not conv.needs_compression()

    conv.complete_task("Greeted the user")
    assert len(conv.task_summaries) == 1
    assert conv.messages == []

    msgs = conv.get_messages_with_history("You are helpful")
    assert msgs[0].role == "system"
    assert msgs[1].text == "Greeted the user"


def test_scenario_conversation_compression():
    conv = Conversation(context_window=100)  # tiny window
    # Add enough messages to trigger compression
    for i in range(10):
        conv.add_message(NexagenMessage(role="user", text=f"Message {i} " * 20))
    assert conv.needs_compression()
    compressible = conv.get_compressible_messages()
    assert len(compressible) > 0


# ── Scenario 6: Tool registry with availability ──────────────


async def test_scenario_tool_registry_availability():
    reg = ToolRegistry()
    reg.register(calc_tool)
    mcp_tool = MCPTool("github_search", "Search GitHub", {"type": "object"}, MCPServerConfig(command="npx"))
    reg.register(mcp_tool)

    available = reg.list_available()
    assert len(available) == 1  # MCP tool not connected
    assert available[0].name == "calc"

    await mcp_tool.connect()
    assert len(reg.list_available()) == 2

    await mcp_tool.disconnect()
    assert len(reg.list_available()) == 1


# ── Scenario 7: Supervisor decision parsing ───────────────────


def test_scenario_supervisor_json_decisions():
    sup = SupervisorAgent.__new__(SupervisorAgent)
    assert sup._parse_decision('{"decision": "continue"}') == "continue"
    assert sup._parse_decision('{"decision": "stop"}') == "stop"


def test_scenario_supervisor_text_fallback():
    sup = SupervisorAgent.__new__(SupervisorAgent)
    assert sup._parse_decision("I think you should continue working") == "continue"
    assert sup._parse_decision("Please stop, you are going in circles") == "stop"


def test_scenario_supervisor_safe_default():
    sup = SupervisorAgent.__new__(SupervisorAgent)
    assert sup._parse_decision("random gibberish xyz") == "stop"


# ── Scenario 8: Built-in tools ────────────────────────────────


def test_scenario_builtin_tools_registered():
    assert set(BUILTIN_TOOLS.keys()) == {"file_read", "file_write", "file_edit", "bash", "grep", "glob"}


async def test_scenario_file_read(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("line one\nline two\nline three\n")
    result = await BUILTIN_TOOLS["file_read"].execute({"file_path": str(f)})
    assert not result.is_error
    assert "line one" in result.output
    assert "line two" in result.output


async def test_scenario_file_write_and_read(tmp_path):
    target = str(tmp_path / "output.txt")
    w_result = await BUILTIN_TOOLS["file_write"].execute({"file_path": target, "content": "hello nexagen"})
    assert not w_result.is_error
    r_result = await BUILTIN_TOOLS["file_read"].execute({"file_path": target})
    assert "hello nexagen" in r_result.output


async def test_scenario_file_edit(tmp_path):
    f = tmp_path / "edit_me.txt"
    f.write_text("old value here")
    result = await BUILTIN_TOOLS["file_edit"].execute({
        "file_path": str(f),
        "old_string": "old value",
        "new_string": "new value",
    })
    assert not result.is_error
    assert f.read_text() == "new value here"


async def test_scenario_bash():
    result = await BUILTIN_TOOLS["bash"].execute({"command": "echo hello world"})
    assert not result.is_error
    assert "hello world" in result.output


async def test_scenario_grep(tmp_path):
    f = tmp_path / "searchme.txt"
    f.write_text("apple\nbanana\napricot\ncherry\n")
    result = await BUILTIN_TOOLS["grep"].execute({"pattern": "ap", "path": str(f)})
    assert not result.is_error
    assert "apple" in result.output
    assert "apricot" in result.output


async def test_scenario_glob(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.py").write_text("")
    (tmp_path / "c.txt").write_text("")
    result = await BUILTIN_TOOLS["glob"].execute({"pattern": "*.py", "path": str(tmp_path)})
    assert not result.is_error
    assert "a.py" in result.output
    assert "b.py" in result.output
    assert "c.txt" not in result.output


# ── Scenario 9: Tool error formatting ─────────────────────────


async def test_scenario_tool_error_format():
    result = await BUILTIN_TOOLS["file_read"].execute({"file_path": "/nonexistent/path/file.txt"})
    assert result.is_error
    assert "FileNotFoundError" in result.output or "No such file" in result.output


async def test_scenario_tool_error_on_invalid_edit(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("some content")
    result = await BUILTIN_TOOLS["file_edit"].execute({
        "file_path": str(f),
        "old_string": "nonexistent string",
        "new_string": "replacement",
    })
    assert "not found" in result.output.lower() or "error" in result.output.lower()


# ── Scenario 10: End-to-end agent with mock provider ──────────


async def test_scenario_agent_simple_response():
    provider = MockProvider([
        NexagenResponse(message=NexagenMessage(role="assistant", text="Hello! I am nexagen."))
    ])
    agent = Agent(provider=provider)
    msgs = [msg async for msg in agent.run("Say hello")]
    assert len(msgs) == 1
    assert msgs[0].text == "Hello! I am nexagen."


async def test_scenario_agent_tool_call_flow(tmp_path):
    """Agent calls file_read, gets result, then responds."""
    f = tmp_path / "data.txt"
    f.write_text("secret: 42")

    provider = MockProvider([
        NexagenResponse(message=NexagenMessage(
            role="assistant",
            text="Let me read the file",
            tool_calls=[ToolCall(id="1", name="file_read", arguments={"file_path": str(f)})],
            summary="Reading data file",
        )),
        NexagenResponse(message=NexagenMessage(
            role="assistant",
            text="The secret is 42.",
        )),
    ])
    agent = Agent(provider=provider, tools=["file_read"], permission_mode="full")

    msgs = [msg async for msg in agent.run("What's in the file?")]
    texts = [m.text for m in msgs if m.text]
    assert any("42" in t for t in texts)


async def test_scenario_agent_permission_denied():
    """Agent tries bash in readonly mode — gets denied."""
    provider = MockProvider([
        NexagenResponse(message=NexagenMessage(
            role="assistant",
            text="Running command",
            tool_calls=[ToolCall(id="1", name="bash", arguments={"command": "ls"})],
        )),
        NexagenResponse(message=NexagenMessage(role="assistant", text="Done")),
    ])
    agent = Agent(provider=provider, tools=["bash"], permission_mode="readonly")

    msgs = [msg async for msg in agent.run("List files")]
    tool_results = [m for m in msgs if m.role == "tool"]
    assert any(m.is_error and "Permission denied" in (m.text or "") for m in tool_results)


async def test_scenario_agent_conversation_continuity():
    """Two runs on same conversation — second sees task summary."""
    provider1 = MockProvider([
        NexagenResponse(message=NexagenMessage(role="assistant", text="Task 1 done"))
    ])
    provider2 = MockProvider([
        NexagenResponse(message=NexagenMessage(role="assistant", text="Task 2 done"))
    ])

    conv = Conversation()
    agent1 = Agent(provider=provider1)
    async for _ in agent1.run("Do task 1", conversation=conv):
        pass

    assert len(conv.task_summaries) == 1

    agent2 = Agent(provider=provider2)
    async for _ in agent2.run("Do task 2", conversation=conv):
        pass

    assert len(conv.task_summaries) == 2


async def test_scenario_agent_with_custom_tool():
    """Agent uses a custom @tool."""
    provider = MockProvider([
        NexagenResponse(message=NexagenMessage(
            role="assistant",
            text="Calculating",
            tool_calls=[ToolCall(id="1", name="calc", arguments={"a": 5, "b": 3, "op": "mul"})],
            summary="Multiplying 5 * 3",
        )),
        NexagenResponse(message=NexagenMessage(role="assistant", text="The answer is 15")),
    ])
    agent = Agent(provider=provider, custom_tools=[calc_tool], permission_mode="full")

    msgs = [msg async for msg in agent.run("What is 5 * 3?")]
    tool_results = [m for m in msgs if m.role == "tool"]
    assert any("15" in (m.text or "") for m in tool_results)
