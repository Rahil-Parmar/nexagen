"""Tests for the native OpenAI provider."""

from __future__ import annotations

import pytest

from nexagen.models import NexagenMessage, ProviderConfig
from nexagen.providers.base import LLMProvider
from nexagen.providers.openai_native import OpenAINativeProvider


class TestOpenAINativeProvider:
    """Unit tests for OpenAINativeProvider."""

    def test_implements_protocol(self):
        """OpenAINativeProvider satisfies the LLMProvider protocol."""
        config = ProviderConfig(
            backend="openai", model="gpt-4o", api_key="test-key"
        )
        provider = OpenAINativeProvider(config)
        assert isinstance(provider, LLMProvider)

    def test_api_key_from_env(self, monkeypatch):
        """Provider picks up the API key from OPENAI_API_KEY env var."""
        monkeypatch.setenv("OPENAI_API_KEY", "env-test-key")
        config = ProviderConfig(backend="openai", model="gpt-4o")
        provider = OpenAINativeProvider(config)
        assert provider.api_key == "env-test-key"

    def test_no_api_key_raises(self, monkeypatch):
        """Provider raises ValueError when no API key is available."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = ProviderConfig(backend="openai", model="gpt-4o")
        with pytest.raises(ValueError, match="API key"):
            OpenAINativeProvider(config)

    def test_convert_messages(self):
        """_convert_messages produces OpenAI-format dicts."""
        config = ProviderConfig(
            backend="openai", model="gpt-4o", api_key="test-key"
        )
        provider = OpenAINativeProvider(config)

        messages = [
            NexagenMessage(role="system", text="You are helpful."),
            NexagenMessage(role="user", text="Hello"),
            NexagenMessage(role="assistant", text="Hi there!"),
        ]

        converted = provider._convert_messages(messages)

        assert len(converted) == 3
        assert converted[0] == {"role": "system", "content": "You are helpful."}
        assert converted[1] == {"role": "user", "content": "Hello"}
        assert converted[2] == {"role": "assistant", "content": "Hi there!"}

    def test_supports_vision(self):
        """OpenAI native provider supports vision."""
        config = ProviderConfig(
            backend="openai", model="gpt-4o", api_key="test-key"
        )
        provider = OpenAINativeProvider(config)
        assert provider.supports_vision() is True

    def test_supports_tool_calling(self):
        """OpenAI native provider supports tool calling."""
        config = ProviderConfig(
            backend="openai", model="gpt-4o", api_key="test-key"
        )
        provider = OpenAINativeProvider(config)
        assert provider.supports_tool_calling() is True
