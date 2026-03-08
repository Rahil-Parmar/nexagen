"""Tests for the Google/Gemini provider."""

from __future__ import annotations

import pytest

from nexagen.models import NexagenMessage, ProviderConfig
from nexagen.providers.base import LLMProvider, ToolSchema
from nexagen.providers.google_provider import GoogleProvider


class TestGoogleProvider:
    """Unit tests for GoogleProvider."""

    def test_implements_protocol(self):
        """GoogleProvider satisfies the LLMProvider protocol."""
        config = ProviderConfig(
            backend="google", model="gemini-2.0-flash", api_key="test-key"
        )
        provider = GoogleProvider(config)
        assert isinstance(provider, LLMProvider)

    def test_api_key_from_env(self, monkeypatch):
        """Provider picks up the API key from GOOGLE_API_KEY env var."""
        monkeypatch.setenv("GOOGLE_API_KEY", "env-google-key")
        config = ProviderConfig(backend="google", model="gemini-2.0-flash")
        provider = GoogleProvider(config)
        assert provider.api_key == "env-google-key"

    def test_no_api_key_raises(self, monkeypatch):
        """Provider raises ValueError when no API key is available."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        config = ProviderConfig(backend="google", model="gemini-2.0-flash")
        with pytest.raises(ValueError, match="API key"):
            GoogleProvider(config)

    def test_convert_messages_to_google_format(self):
        """_convert_messages maps roles and wraps text in parts."""
        config = ProviderConfig(
            backend="google", model="gemini-2.0-flash", api_key="test-key"
        )
        provider = GoogleProvider(config)

        messages = [
            NexagenMessage(role="user", text="Hello"),
            NexagenMessage(role="assistant", text="Hi there!"),
        ]

        converted = provider._convert_messages(messages)

        assert len(converted) == 2
        assert converted[0] == {
            "role": "user",
            "parts": [{"text": "Hello"}],
        }
        # assistant -> model in Google format
        assert converted[1] == {
            "role": "model",
            "parts": [{"text": "Hi there!"}],
        }

    def test_convert_messages_system_extracted(self):
        """System messages are extracted separately for Google API."""
        config = ProviderConfig(
            backend="google", model="gemini-2.0-flash", api_key="test-key"
        )
        provider = GoogleProvider(config)

        messages = [
            NexagenMessage(role="system", text="You are helpful."),
            NexagenMessage(role="user", text="Hello"),
        ]

        converted = provider._convert_messages(messages)

        # System messages should not appear in the contents array
        assert len(converted) == 1
        assert converted[0]["role"] == "user"

    def test_extract_system_instruction(self):
        """_extract_system_instruction pulls out system message text."""
        config = ProviderConfig(
            backend="google", model="gemini-2.0-flash", api_key="test-key"
        )
        provider = GoogleProvider(config)

        messages = [
            NexagenMessage(role="system", text="Be concise."),
            NexagenMessage(role="user", text="Hello"),
        ]

        system_text = provider._extract_system_instruction(messages)
        assert system_text == "Be concise."

    def test_convert_tools_to_google_format(self):
        """_convert_tools wraps tools in function_declarations."""
        config = ProviderConfig(
            backend="google", model="gemini-2.0-flash", api_key="test-key"
        )
        provider = GoogleProvider(config)

        tools = [
            ToolSchema(
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                            },
                            "required": ["city"],
                        },
                    },
                }
            ),
        ]

        converted = provider._convert_tools(tools)

        assert len(converted) == 1
        assert "function_declarations" in converted[0]
        decls = converted[0]["function_declarations"]
        assert len(decls) == 1
        assert decls[0]["name"] == "get_weather"
        assert decls[0]["description"] == "Get weather for a city"
        assert "parameters" in decls[0]

    def test_supports_tool_calling(self):
        """Google provider supports tool calling."""
        config = ProviderConfig(
            backend="google", model="gemini-2.0-flash", api_key="test-key"
        )
        provider = GoogleProvider(config)
        assert provider.supports_tool_calling() is True

    def test_supports_vision(self):
        """Google provider supports vision."""
        config = ProviderConfig(
            backend="google", model="gemini-2.0-flash", api_key="test-key"
        )
        provider = GoogleProvider(config)
        assert provider.supports_vision() is True
