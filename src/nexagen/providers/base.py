"""Base protocol for LLM providers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from nexagen.models import NexagenMessage, NexagenResponse


class ToolSchema(dict):
    """JSON schema dict representing a tool for the LLM."""

    pass


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol that all LLM provider backends must implement."""

    async def chat(
        self,
        messages: list[NexagenMessage],
        tools: list[ToolSchema] | None = None,
    ) -> NexagenResponse: ...

    def supports_tool_calling(self) -> bool: ...

    def supports_vision(self) -> bool: ...
