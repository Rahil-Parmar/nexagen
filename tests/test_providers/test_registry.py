"""Tests for the provider registry and LLMProvider protocol."""

from __future__ import annotations

import pytest

from nexagen.models import NexagenMessage, NexagenResponse, ProviderConfig
from nexagen.providers.base import LLMProvider, ToolSchema
from nexagen.providers.registry import ProviderRegistry


class MockProvider:
    """A mock provider that satisfies the LLMProvider protocol."""

    def __init__(self, config: ProviderConfig):
        self.config = config

    async def chat(
        self,
        messages: list[NexagenMessage],
        tools: list[ToolSchema] | None = None,
    ) -> NexagenResponse:
        return NexagenResponse(
            message=NexagenMessage(role="assistant", text="mock")
        )

    def supports_tool_calling(self) -> bool:
        return True

    def supports_vision(self) -> bool:
        return False


class AnotherMockProvider:
    """A second mock provider for multi-backend tests."""

    def __init__(self, config: ProviderConfig):
        self.config = config

    async def chat(
        self,
        messages: list[NexagenMessage],
        tools: list[ToolSchema] | None = None,
    ) -> NexagenResponse:
        return NexagenResponse(
            message=NexagenMessage(role="assistant", text="another")
        )

    def supports_tool_calling(self) -> bool:
        return False

    def supports_vision(self) -> bool:
        return True


class TestLLMProviderProtocol:
    """Verify that the LLMProvider protocol works at runtime."""

    def test_llm_provider_protocol(self):
        """A class implementing all required methods is recognized as LLMProvider."""
        config = ProviderConfig(backend="mock", model="test")
        provider = MockProvider(config)
        assert isinstance(provider, LLMProvider)

    def test_non_provider_not_recognized(self):
        """An object missing protocol methods is not an LLMProvider."""

        class NotAProvider:
            pass

        assert not isinstance(NotAProvider(), LLMProvider)


class TestProviderRegistry:
    """Tests for the ProviderRegistry class."""

    def test_registry_register_and_resolve(self):
        """Register a mock provider class and resolve it by string."""
        registry = ProviderRegistry()
        registry.register("mock", MockProvider)

        provider = registry.resolve("mock/test-model")

        assert isinstance(provider, MockProvider)
        assert provider.config.backend == "mock"
        assert provider.config.model == "test-model"

    def test_registry_resolve_with_config(self):
        """Resolve a provider using a ProviderConfig object directly."""
        registry = ProviderRegistry()
        registry.register("mock", MockProvider)

        config = ProviderConfig(backend="mock", model="gpt-4")
        provider = registry.resolve(config)

        assert isinstance(provider, MockProvider)
        assert provider.config.backend == "mock"
        assert provider.config.model == "gpt-4"

    def test_registry_unknown_backend_raises(self):
        """Resolving an unknown backend raises ValueError."""
        registry = ProviderRegistry()

        with pytest.raises(ValueError, match="Unknown backend: 'nonexistent'"):
            registry.resolve("nonexistent/model")

    def test_registry_unknown_backend_lists_available(self):
        """The error message for unknown backends lists available options."""
        registry = ProviderRegistry()
        registry.register("alpha", MockProvider)

        with pytest.raises(ValueError, match="alpha"):
            registry.resolve("beta/model")

    def test_registry_multiple_backends(self):
        """Register multiple backends and resolve each independently."""
        registry = ProviderRegistry()
        registry.register("mock", MockProvider)
        registry.register("another", AnotherMockProvider)

        provider_a = registry.resolve("mock/model-a")
        provider_b = registry.resolve("another/model-b")

        assert isinstance(provider_a, MockProvider)
        assert isinstance(provider_b, AnotherMockProvider)
        assert provider_a.config.model == "model-a"
        assert provider_b.config.model == "model-b"

    def test_registry_resolve_with_host(self):
        """Resolve a provider string that includes a host."""
        registry = ProviderRegistry()
        registry.register("mock", MockProvider)

        provider = registry.resolve("mock/model@localhost:8080")

        assert provider.config.base_url == "http://localhost:8080"
        assert provider.config.model == "model"
