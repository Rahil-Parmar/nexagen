"""Anthropic provider for the nexagen SDK.

Uses the filename ``anthropic_provider.py`` to avoid collisions with the
``anthropic`` package.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from nexagen.http_retry import request_with_retry

from nexagen.models import (
    NexagenMessage,
    NexagenResponse,
    ProviderConfig,
    ToolCall,
)
from nexagen.providers.base import ToolSchema

_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider:
    """Native Anthropic API provider.

    Implements the :class:`LLMProvider` protocol using ``httpx`` to
    communicate with the Anthropic Messages API.
    """

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Anthropic API key not found. Pass it via ProviderConfig.api_key "
                "or set the ANTHROPIC_API_KEY environment variable."
            )
        base_url = config.base_url or "https://api.anthropic.com"
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": _ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
        )
        self._model = config.model

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def _convert_messages(
        self, messages: list[NexagenMessage]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert nexagen messages to Anthropic API format.

        The Anthropic API requires the ``system`` prompt as a top-level
        parameter rather than a message, so we extract it here.

        Returns:
            A tuple of ``(system_prompt, converted_messages)``.
        """
        system: str | None = None
        converted: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                system = msg.text
                continue

            if msg.role == "tool":
                # Tool results are sent as user messages with a
                # ``tool_result`` content block.
                converted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.text,
                            }
                        ],
                    }
                )
                continue

            if msg.role == "assistant" and msg.tool_calls:
                # Convert tool calls into ``tool_use`` content blocks.
                content: list[dict[str, Any]] = []
                if msg.text:
                    content.append({"type": "text", "text": msg.text})
                for tc in msg.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                converted.append({"role": "assistant", "content": content})
                continue

            # Default: simple user or assistant text message.
            converted.append({"role": msg.role, "content": msg.text})

        return system, converted

    def _convert_tools(self, tools: list[ToolSchema]) -> list[dict[str, Any]]:
        """Convert nexagen tool schemas to Anthropic format.

        Anthropic uses ``input_schema`` instead of ``parameters``.
        """
        result: list[dict[str, Any]] = []
        for tool in tools:
            result.append(
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": tool["parameters"],
                }
            )
        return result

    # ------------------------------------------------------------------
    # LLMProvider protocol
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[NexagenMessage],
        tools: list[ToolSchema] | None = None,
    ) -> NexagenResponse:
        """Send a chat request to the Anthropic Messages API."""
        system, converted = self._convert_messages(messages)

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": converted,
            "max_tokens": self._config.max_tokens or 4096,
        }
        if system is not None:
            payload["system"] = system
        if self._config.temperature is not None:
            payload["temperature"] = self._config.temperature
        if tools:
            payload["tools"] = self._convert_tools(tools)

        response = await request_with_retry(
            self._client, "POST", "/v1/messages", json=payload,
        )

        try:
            data = response.json()
        except (ValueError, Exception) as e:
            raise ValueError(f"Invalid JSON in Anthropic response: {e}") from e

        # Check for API error responses
        if "error" in data:
            error_msg = data["error"].get("message", str(data["error"]))
            raise ValueError(f"Anthropic API error: {error_msg}")

        return self._parse_response(data)

    def _parse_response(self, data: dict[str, Any]) -> NexagenResponse:
        """Parse the raw Anthropic API response into a NexagenResponse."""
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in data.get("content", []):
            block_type = block.get("type")
            try:
                if block_type == "text":
                    text_parts.append(block.get("text", ""))
                elif block_type == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=block.get("id", "unknown"),
                            name=block.get("name", "unknown"),
                            arguments=block.get("input", {}),
                        )
                    )
            except (KeyError, TypeError) as e:
                import logging
                logging.getLogger("nexagen.provider").warning(
                    "Skipping malformed content block: %s", e
                )
                continue

        message = NexagenMessage(
            role="assistant",
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls if tool_calls else None,
        )
        return NexagenResponse(message=message)

    def supports_tool_calling(self) -> bool:
        """Anthropic supports tool calling."""
        return True

    def supports_vision(self) -> bool:
        """Anthropic supports vision inputs."""
        return True
