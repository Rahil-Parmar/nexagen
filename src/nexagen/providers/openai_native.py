"""Native OpenAI provider using httpx directly."""

from __future__ import annotations

import json
import os
import uuid

import httpx

from nexagen.http_retry import request_with_retry

from nexagen.models import (
    NexagenMessage,
    NexagenResponse,
    ProviderConfig,
    ToolCall,
)
from nexagen.providers.base import ToolSchema


class OpenAINativeProvider:
    """LLM provider that talks directly to the OpenAI API via httpx.

    Uses ``https://api.openai.com/v1`` as the default base URL.
    The API key is read from *config.api_key* first, falling back to
    the ``OPENAI_API_KEY`` environment variable.
    """

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set config.api_key or the "
                "OPENAI_API_KEY environment variable."
            )
        self.base_url = config.base_url or "https://api.openai.com/v1"

    # ------------------------------------------------------------------
    # Protocol methods
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[NexagenMessage],
        tools: list[ToolSchema] | None = None,
    ) -> NexagenResponse:
        """Send a chat completion request to the OpenAI API."""
        payload: dict = {
            "model": self.config.model,
            "messages": self._convert_messages(messages),
        }
        if tools:
            payload["tools"] = self._convert_tools(tools)
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.max_tokens is not None:
            payload["max_tokens"] = self.config.max_tokens

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await request_with_retry(
                client,
                "POST",
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

        data = response.json()
        choice = data["choices"][0]
        msg = choice["message"]

        tool_calls = None
        if msg.get("tool_calls"):
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=json.loads(tc["function"]["arguments"]),
                )
                for tc in msg["tool_calls"]
            ]

        return NexagenResponse(
            message=NexagenMessage(
                role="assistant",
                text=msg.get("content"),
                tool_calls=tool_calls,
            )
        )

    def supports_tool_calling(self) -> bool:  # noqa: D102
        return True

    def supports_vision(self) -> bool:  # noqa: D102
        return True

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def _convert_messages(
        self, messages: list[NexagenMessage]
    ) -> list[dict]:
        """Convert NexagenMessages to OpenAI-format dicts."""
        converted: list[dict] = []
        for msg in messages:
            entry: dict = {"role": msg.role, "content": msg.text or ""}

            if msg.role == "tool" and msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id

            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]

            converted.append(entry)
        return converted

    def _convert_tools(self, tools: list[ToolSchema]) -> list[dict]:
        """Pass through — ToolSchema is already in OpenAI format."""
        return [dict(t) for t in tools]
