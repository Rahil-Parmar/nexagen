"""Tests for nexagen core data models."""

from __future__ import annotations

from nexagen.models import (
    NexagenMessage,
    NexagenResponse,
    ProviderConfig,
    ToolCall,
    ToolResult,
)


def test_user_message():
    msg = NexagenMessage(role="user", text="Hello, world!")
    assert msg.role == "user"
    assert msg.text == "Hello, world!"
    assert msg.tool_calls is None
    assert msg.summary is None
    assert msg.tool_call_id is None
    assert msg.is_error is False


def test_assistant_message_with_tool_calls():
    tool_calls = [
        ToolCall(id="tc_1", name="get_weather", arguments={"city": "London"}),
        ToolCall(id="tc_2", name="get_time", arguments={"timezone": "UTC"}),
    ]
    msg = NexagenMessage(
        role="assistant",
        text=None,
        tool_calls=tool_calls,
        summary="Called weather and time tools",
    )
    assert msg.role == "assistant"
    assert msg.text is None
    assert len(msg.tool_calls) == 2
    assert msg.tool_calls[0].name == "get_weather"
    assert msg.tool_calls[1].arguments == {"timezone": "UTC"}
    assert msg.summary == "Called weather and time tools"


def test_tool_result_success():
    result = ToolResult(tool_call_id="tc_1", output="72°F and sunny")
    assert result.tool_call_id == "tc_1"
    assert result.output == "72°F and sunny"
    assert result.is_error is False


def test_tool_result_error():
    result = ToolResult(
        tool_call_id="tc_1", output="City not found", is_error=True
    )
    assert result.tool_call_id == "tc_1"
    assert result.output == "City not found"
    assert result.is_error is True


def test_tool_result_to_message():
    result = ToolResult(
        tool_call_id="tc_1", output="72°F and sunny", is_error=False
    )
    msg = result.to_message()
    assert isinstance(msg, NexagenMessage)
    assert msg.role == "tool"
    assert msg.text == "72°F and sunny"
    assert msg.tool_call_id == "tc_1"
    assert msg.is_error is False


def test_nexagen_response_has_tool_calls():
    msg = NexagenMessage(
        role="assistant",
        tool_calls=[
            ToolCall(id="tc_1", name="search", arguments={"q": "test"})
        ],
    )
    response = NexagenResponse(message=msg)
    assert response.has_tool_calls is True


def test_nexagen_response_no_tools():
    msg = NexagenMessage(role="assistant", text="Just a text reply")
    response = NexagenResponse(message=msg)
    assert response.has_tool_calls is False


def test_provider_config_from_string():
    config = ProviderConfig.from_string("ollama/qwen3")
    assert config.backend == "ollama"
    assert config.model == "qwen3"
    assert config.base_url is None


def test_provider_config_from_string_with_host():
    config = ProviderConfig.from_string("ollama/qwen3@192.168.1.5:11434")
    assert config.backend == "ollama"
    assert config.model == "qwen3"
    assert config.base_url == "http://192.168.1.5:11434"


def test_provider_config_cloud():
    config = ProviderConfig.from_string("openai/gpt-4o")
    assert config.backend == "openai"
    assert config.model == "gpt-4o"
    assert config.base_url is None
