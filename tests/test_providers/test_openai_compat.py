"""Tests for the OpenAI-compatible provider."""

from __future__ import annotations

import json

import pytest

from nexagen.constants import (
    OPENAI_COMPAT_DEFAULT_URLS,
)
from nexagen.models import (
    NexagenMessage,
    ProviderConfig,
    ToolCall,
)
from nexagen.providers.base import LLMProvider
from nexagen.providers.openai_compat import OpenAICompatProvider


class TestOpenAICompatProtocol:
    """Verify the provider satisfies the LLMProvider protocol."""

    def test_implements_protocol(self):
        """OpenAICompatProvider is recognized as an LLMProvider."""
        config = ProviderConfig(backend="ollama", model="qwen3")
        provider = OpenAICompatProvider(config)
        assert isinstance(provider, LLMProvider)


class TestBaseURLResolution:
    """Verify default and custom base URL handling."""

    def test_default_base_url_ollama(self):
        """Ollama backend uses the default localhost:11434/v1 URL."""
        config = ProviderConfig(backend="ollama", model="qwen3")
        provider = OpenAICompatProvider(config)
        assert provider.base_url == OPENAI_COMPAT_DEFAULT_URLS["ollama"]

    def test_default_base_url_vllm(self):
        """vLLM backend uses the default localhost:8000/v1 URL."""
        config = ProviderConfig(backend="vllm", model="mistral")
        provider = OpenAICompatProvider(config)
        assert provider.base_url == OPENAI_COMPAT_DEFAULT_URLS["vllm"]

    def test_custom_base_url(self):
        """A custom base_url without /v1 gets /v1 appended."""
        config = ProviderConfig(
            backend="ollama",
            model="qwen3",
            base_url="http://myhost:9999",
        )
        provider = OpenAICompatProvider(config)
        assert provider.base_url == "http://myhost:9999/v1"

    def test_custom_base_url_with_v1(self):
        """A custom base_url already ending in /v1 is not double-appended."""
        config = ProviderConfig(
            backend="ollama",
            model="qwen3",
            base_url="http://myhost:9999/v1",
        )
        provider = OpenAICompatProvider(config)
        assert provider.base_url == "http://myhost:9999/v1"


class TestConvertMessages:
    """Verify _convert_messages produces correct OpenAI-format dicts."""

    def test_convert_messages_basic(self):
        """System + user messages convert to role/content dicts."""
        config = ProviderConfig(backend="ollama", model="qwen3")
        provider = OpenAICompatProvider(config)

        messages = [
            NexagenMessage(role="system", text="You are helpful."),
            NexagenMessage(role="user", text="Hello"),
        ]
        result = provider._convert_messages(messages)

        assert result == [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

    def test_convert_messages_tool_result(self):
        """A tool-role message includes tool_call_id."""
        config = ProviderConfig(backend="ollama", model="qwen3")
        provider = OpenAICompatProvider(config)

        messages = [
            NexagenMessage(
                role="tool",
                text="42",
                tool_call_id="call_123",
            ),
        ]
        result = provider._convert_messages(messages)

        assert result == [
            {
                "role": "tool",
                "content": "42",
                "tool_call_id": "call_123",
            },
        ]

    def test_convert_messages_assistant_with_tools(self):
        """Assistant message with tool_calls converts correctly."""
        config = ProviderConfig(backend="ollama", model="qwen3")
        provider = OpenAICompatProvider(config)

        messages = [
            NexagenMessage(
                role="assistant",
                text=None,
                tool_calls=[
                    ToolCall(
                        id="call_abc",
                        name="get_weather",
                        arguments={"city": "London"},
                    ),
                ],
            ),
        ]
        result = provider._convert_messages(messages)

        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert msg["content"] is None
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["id"] == "call_abc"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "London"}


class TestParseToolCalls:
    """Verify _parse_tool_calls handles different argument formats."""

    def test_parse_tool_calls_dict_args(self):
        """Ollama-style: arguments arrive as a dict."""
        config = ProviderConfig(backend="ollama", model="qwen3")
        provider = OpenAICompatProvider(config)

        raw = [
            {
                "id": "call_1",
                "function": {
                    "name": "search",
                    "arguments": {"query": "test"},
                },
            },
        ]
        result = provider._parse_tool_calls(raw)

        assert len(result) == 1
        assert isinstance(result[0], ToolCall)
        assert result[0].id == "call_1"
        assert result[0].name == "search"
        assert result[0].arguments == {"query": "test"}

    def test_parse_tool_calls_string_args(self):
        """vLLM/OpenAI-style: arguments arrive as a JSON string."""
        config = ProviderConfig(backend="vllm", model="mistral")
        provider = OpenAICompatProvider(config)

        raw = [
            {
                "id": "call_2",
                "function": {
                    "name": "calculate",
                    "arguments": '{"expression": "2+2"}',
                },
            },
        ]
        result = provider._parse_tool_calls(raw)

        assert len(result) == 1
        assert result[0].id == "call_2"
        assert result[0].name == "calculate"
        assert result[0].arguments == {"expression": "2+2"}


class TestConvertTools:
    """Verify _convert_tools wraps schemas in OpenAI function format."""

    def test_convert_tools(self):
        """Tool schemas are wrapped with type=function and nested under function key."""
        config = ProviderConfig(backend="ollama", model="qwen3")
        provider = OpenAICompatProvider(config)

        tools = [
            {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
            },
        ]
        result = provider._convert_tools(tools)

        assert result is not None
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"] == tools[0]

    def test_convert_tools_none(self):
        """Passing None returns None."""
        config = ProviderConfig(backend="ollama", model="qwen3")
        provider = OpenAICompatProvider(config)

        assert provider._convert_tools(None) is None
