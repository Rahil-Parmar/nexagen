"""Tests for the Anthropic provider."""

from __future__ import annotations

import pytest

from nexagen.models import (
    NexagenMessage,
    ProviderConfig,
    ToolCall,
)
from nexagen.providers.base import LLMProvider, ToolSchema
from nexagen.providers.anthropic_provider import AnthropicProvider


@pytest.fixture
def config() -> ProviderConfig:
    return ProviderConfig(backend="anthropic", model="claude-sonnet-4-20250514", api_key="test-key-123")


@pytest.fixture
def provider(config: ProviderConfig) -> AnthropicProvider:
    return AnthropicProvider(config)


class TestAnthropicProviderInit:
    """Tests for provider initialization and protocol compliance."""

    def test_implements_protocol(self, provider: AnthropicProvider) -> None:
        """AnthropicProvider satisfies the LLMProvider protocol."""
        assert isinstance(provider, LLMProvider)

    def test_api_key_from_config(self) -> None:
        """API key is read from ProviderConfig.api_key."""
        config = ProviderConfig(backend="anthropic", model="claude-sonnet-4-20250514", api_key="cfg-key")
        p = AnthropicProvider(config)
        assert p._api_key == "cfg-key"

    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """API key falls back to ANTHROPIC_API_KEY env var."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        config = ProviderConfig(backend="anthropic", model="claude-sonnet-4-20250514")
        p = AnthropicProvider(config)
        assert p._api_key == "env-key"

    def test_no_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ValueError is raised when no API key is available."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config = ProviderConfig(backend="anthropic", model="claude-sonnet-4-20250514")
        with pytest.raises(ValueError, match="API key"):
            AnthropicProvider(config)


class TestConvertMessages:
    """Tests for _convert_messages()."""

    def test_convert_messages_extracts_system(self, provider: AnthropicProvider) -> None:
        """System message is extracted and returned separately."""
        messages = [
            NexagenMessage(role="system", text="You are helpful."),
            NexagenMessage(role="user", text="Hello"),
        ]
        system, converted = provider._convert_messages(messages)
        assert system == "You are helpful."
        assert len(converted) == 1
        assert converted[0]["role"] == "user"

    def test_convert_messages_user(self, provider: AnthropicProvider) -> None:
        """User messages are converted correctly."""
        messages = [NexagenMessage(role="user", text="Hi there")]
        system, converted = provider._convert_messages(messages)
        assert system is None
        assert len(converted) == 1
        assert converted[0] == {"role": "user", "content": "Hi there"}

    def test_convert_messages_tool_result(self, provider: AnthropicProvider) -> None:
        """Tool result messages become user role with tool_result content block."""
        messages = [
            NexagenMessage(role="tool", text="42", tool_call_id="call_123"),
        ]
        system, converted = provider._convert_messages(messages)
        assert system is None
        assert len(converted) == 1
        msg = converted[0]
        assert msg["role"] == "user"
        assert len(msg["content"]) == 1
        block = msg["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "call_123"
        assert block["content"] == "42"

    def test_convert_messages_assistant_with_tools(self, provider: AnthropicProvider) -> None:
        """Assistant messages with tool_calls become tool_use content blocks."""
        messages = [
            NexagenMessage(
                role="assistant",
                text=None,
                tool_calls=[
                    ToolCall(id="tc_1", name="get_weather", arguments={"city": "London"}),
                    ToolCall(id="tc_2", name="get_time", arguments={"tz": "UTC"}),
                ],
            ),
        ]
        system, converted = provider._convert_messages(messages)
        assert system is None
        assert len(converted) == 1
        msg = converted[0]
        assert msg["role"] == "assistant"
        content = msg["content"]
        assert len(content) == 2
        assert content[0] == {
            "type": "tool_use",
            "id": "tc_1",
            "name": "get_weather",
            "input": {"city": "London"},
        }
        assert content[1] == {
            "type": "tool_use",
            "id": "tc_2",
            "name": "get_time",
            "input": {"tz": "UTC"},
        }


class TestConvertTools:
    """Tests for _convert_tools()."""

    def test_convert_tools(self, provider: AnthropicProvider) -> None:
        """Tool schemas are converted to Anthropic format with input_schema."""
        tools: list[ToolSchema] = [
            ToolSchema(
                {
                    "name": "get_weather",
                    "description": "Get the weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                        },
                        "required": ["city"],
                    },
                }
            ),
        ]
        converted = provider._convert_tools(tools)
        assert len(converted) == 1
        tool = converted[0]
        assert tool["name"] == "get_weather"
        assert tool["description"] == "Get the weather for a city"
        assert "input_schema" in tool
        assert "parameters" not in tool
        assert tool["input_schema"]["type"] == "object"
        assert "city" in tool["input_schema"]["properties"]


class TestCapabilities:
    """Tests for capability flags."""

    def test_supports_tool_calling(self, provider: AnthropicProvider) -> None:
        assert provider.supports_tool_calling() is True

    def test_supports_vision(self, provider: AnthropicProvider) -> None:
        assert provider.supports_vision() is True
