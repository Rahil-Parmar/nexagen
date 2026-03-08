"""OpenAI-compatible provider for Ollama, vLLM, LM Studio, Groq, Together."""

from __future__ import annotations

import json

import httpx

from nexagen.constants import OPENAI_COMPAT_DEFAULT_URLS
from nexagen.http_retry import request_with_retry, RetryConfig
from nexagen.models import (
    NexagenMessage,
    NexagenResponse,
    ProviderConfig,
    ToolCall,
)
from nexagen.providers.base import ToolSchema


class OpenAICompatProvider:
    """Provider that speaks the OpenAI chat-completions API.

    Works with any backend that exposes an OpenAI-compatible
    ``/chat/completions`` endpoint (Ollama, vLLM, LM Studio, Groq,
    Together AI, etc.).
    """

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.base_url = self._resolve_base_url(config)
        headers: dict[str, str] = {}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        self._client = httpx.AsyncClient(headers=headers, timeout=300.0)

    # ------------------------------------------------------------------
    # Base URL resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_base_url(config: ProviderConfig) -> str:
        """Determine the base URL from config or backend defaults."""
        if config.base_url:
            url = config.base_url.rstrip("/")
            if not url.endswith("/v1"):
                url += "/v1"
            return url
        return OPENAI_COMPAT_DEFAULT_URLS.get(
            config.backend,
            OPENAI_COMPAT_DEFAULT_URLS["ollama"],
        )

    # ------------------------------------------------------------------
    # Message / tool conversion helpers
    # ------------------------------------------------------------------

    def _convert_messages(self, messages: list[NexagenMessage]) -> list[dict]:
        """Convert nexagen messages to OpenAI chat-completions format."""
        converted: list[dict] = []
        for msg in messages:
            entry: dict = {"role": msg.role, "content": msg.text}

            if msg.role == "tool" and msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id

            if msg.role == "assistant" and msg.tool_calls:
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

    def _convert_tools(self, tools: list[ToolSchema] | None) -> list[dict] | None:
        """Wrap tool schemas in OpenAI function-calling format."""
        if tools is None:
            return None
        return [{"type": "function", "function": t} for t in tools]

    def _parse_tool_calls(self, raw_tool_calls: list[dict]) -> list[ToolCall]:
        """Parse raw tool call dicts into ToolCall models.

        Handles both dict arguments (Ollama) and JSON-string arguments
        (vLLM / OpenAI).
        """
        parsed: list[ToolCall] = []
        for raw in raw_tool_calls:
            func = raw["function"]
            args = func["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            parsed.append(
                ToolCall(
                    id=raw["id"],
                    name=func["name"],
                    arguments=args,
                )
            )
        return parsed

    # ------------------------------------------------------------------
    # Core chat method
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[NexagenMessage],
        tools: list[ToolSchema] | None = None,
    ) -> NexagenResponse:
        """Send a chat-completions request and return a NexagenResponse."""
        payload: dict = {
            "model": self.config.model,
            "messages": self._convert_messages(messages),
        }
        converted_tools = self._convert_tools(tools)
        if converted_tools:
            payload["tools"] = converted_tools

        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.max_tokens is not None:
            payload["max_tokens"] = self.config.max_tokens

        response = await request_with_retry(
            self._client,
            "POST",
            f"{self.base_url}/chat/completions",
            json=payload,
        )

        try:
            data = response.json()
        except (ValueError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid JSON in LLM response: {e}") from e

        # Defensive parsing — handle malformed responses
        choices = data.get("choices")
        if not choices or not isinstance(choices, list):
            raise ValueError(f"LLM returned no choices: {data.get('error', data)}")

        choice = choices[0].get("message", {})
        tool_calls = None
        try:
            if choice.get("tool_calls"):
                tool_calls = self._parse_tool_calls(choice["tool_calls"])
        except (KeyError, TypeError, json.JSONDecodeError) as e:
            import logging
            logging.getLogger("nexagen.provider").warning("Failed to parse tool calls: %s", e)
            # Continue without tool calls rather than crashing

        return NexagenResponse(
            message=NexagenMessage(
                role="assistant",
                text=choice.get("content"),
                tool_calls=tool_calls,
            )
        )

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    def supports_tool_calling(self) -> bool:
        """Return True — OpenAI-compatible endpoints support tools."""
        return True

    def supports_vision(self) -> bool:
        """Return False — vision is not universally supported."""
        return False
