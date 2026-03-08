"""Provider registry for mapping backend names to provider classes."""

from __future__ import annotations

from nexagen.models import ProviderConfig
from nexagen.providers.base import LLMProvider


class ProviderRegistry:
    """Registry that maps backend names to provider classes.

    Providers are registered by name and resolved from either a string
    like ``"ollama/qwen3@host"`` or a :class:`ProviderConfig` instance.
    """

    def __init__(self) -> None:
        self._backends: dict[str, type] = {}

    def register(self, backend_name: str, provider_class: type) -> None:
        """Register a provider class under a backend name."""
        self._backends[backend_name] = provider_class

    def resolve(self, provider: str | ProviderConfig) -> LLMProvider:
        """Create a provider instance from a string or config.

        Args:
            provider: Either a provider string (e.g. ``"ollama/qwen3"``)
                or a :class:`ProviderConfig` object.

        Returns:
            An instantiated provider.

        Raises:
            ValueError: If the backend is not registered.
        """
        if isinstance(provider, str):
            config = ProviderConfig.from_string(provider)
        else:
            config = provider

        if config.backend not in self._backends:
            raise ValueError(
                f"Unknown backend: '{config.backend}'. "
                f"Available: {list(self._backends.keys())}"
            )

        provider_class = self._backends[config.backend]
        return provider_class(config)
