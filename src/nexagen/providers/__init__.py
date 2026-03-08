"""Provider subsystem for nexagen."""

from __future__ import annotations

from nexagen.models import ProviderConfig
from nexagen.providers.base import LLMProvider, ToolSchema
from nexagen.providers.registry import ProviderRegistry

_registry = ProviderRegistry()


def get_provider(provider: str | ProviderConfig) -> LLMProvider:
    """Resolve a provider from the global registry.

    Args:
        provider: A provider string (e.g. ``"ollama/qwen3"``) or config.

    Returns:
        An instantiated :class:`LLMProvider`.
    """
    return _registry.resolve(provider)


__all__ = [
    "LLMProvider",
    "ProviderRegistry",
    "ToolSchema",
    "get_provider",
]
