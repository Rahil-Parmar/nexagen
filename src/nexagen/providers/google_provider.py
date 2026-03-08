"""Google/Gemini provider using httpx directly."""

from __future__ import annotations

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


class GoogleProvider:
    """LLM provider that talks to the Google Generative Language API.

    Uses ``https://generativelanguage.googleapis.com/v1beta`` as the
    base URL.  The API key is read from *config.api_key* first, falling
    back to the ``GOOGLE_API_KEY`` environment variable.
    """

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.api_key = config.api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key is required. Set config.api_key or the "
                "GOOGLE_API_KEY environment variable."
            )
        self.base_url = (
            config.base_url
            or "https://generativelanguage.googleapis.com/v1beta"
        )

    # ------------------------------------------------------------------
    # Protocol methods
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[NexagenMessage],
        tools: list[ToolSchema] | None = None,
    ) -> NexagenResponse:
        """Send a generateContent request to the Gemini API."""
        payload: dict = {
            "contents": self._convert_messages(messages),
        }

        system_text = self._extract_system_instruction(messages)
        if system_text:
            payload["system_instruction"] = {
                "parts": [{"text": system_text}],
            }

        if tools:
            payload["tools"] = self._convert_tools(tools)

        generation_config: dict = {}
        if self.config.temperature is not None:
            generation_config["temperature"] = self.config.temperature
        if self.config.max_tokens is not None:
            generation_config["maxOutputTokens"] = self.config.max_tokens
        if generation_config:
            payload["generationConfig"] = generation_config

        url = f"{self.base_url}/models/{self.config.model}:generateContent"

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await request_with_retry(
                client,
                "POST",
                url,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key,
                },
                json=payload,
            )

        data = response.json()
        candidate = data["candidates"][0]
        parts = candidate["content"]["parts"]

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append(
                    ToolCall(
                        id=str(uuid.uuid4()),
                        name=fc["name"],
                        arguments=fc.get("args", {}),
                    )
                )

        return NexagenResponse(
            message=NexagenMessage(
                role="assistant",
                text="\n".join(text_parts) if text_parts else None,
                tool_calls=tool_calls if tool_calls else None,
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
        """Convert NexagenMessages to Google/Gemini format.

        - Skips ``system`` messages (handled via system_instruction).
        - Maps ``assistant`` role to ``model``.
        - Wraps text in ``parts: [{text: ...}]``.
        """
        converted: list[dict] = []
        for msg in messages:
            if msg.role == "system":
                continue

            role = "model" if msg.role == "assistant" else msg.role
            parts: list[dict] = []

            if msg.text:
                parts.append({"text": msg.text})

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    parts.append(
                        {
                            "functionCall": {
                                "name": tc.name,
                                "args": tc.arguments,
                            }
                        }
                    )

            converted.append({"role": role, "parts": parts})
        return converted

    def _extract_system_instruction(
        self, messages: list[NexagenMessage]
    ) -> str | None:
        """Return concatenated text from all system messages, or None."""
        system_texts = [
            msg.text for msg in messages if msg.role == "system" and msg.text
        ]
        return "\n".join(system_texts) if system_texts else None

    def _convert_tools(self, tools: list[ToolSchema]) -> list[dict]:
        """Convert OpenAI-format tool schemas to Google format.

        Google expects::

            [{"function_declarations": [{"name": ..., "description": ...,
              "parameters": ...}]}]
        """
        declarations: list[dict] = []
        for tool in tools:
            func = tool.get("function", tool)
            declarations.append(
                {
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                }
            )
        return [{"function_declarations": declarations}]
