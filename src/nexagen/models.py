"""Core data models for the nexagen SDK."""

from __future__ import annotations

from pydantic import BaseModel


class ToolCall(BaseModel):
    """Represents a tool call made by the assistant."""

    id: str
    name: str
    arguments: dict


class ToolResult(BaseModel):
    """Represents the result of executing a tool call."""

    tool_call_id: str
    output: str
    is_error: bool = False

    def to_message(self) -> NexagenMessage:
        """Convert this tool result into a NexagenMessage with role='tool'."""
        return NexagenMessage(
            role="tool",
            text=self.output,
            tool_call_id=self.tool_call_id,
            is_error=self.is_error,
        )


class NexagenMessage(BaseModel):
    """A unified message type used across all providers."""

    role: str  # "system" | "user" | "assistant" | "tool"
    text: str | None = None
    tool_calls: list[ToolCall] | None = None
    summary: str | None = None
    tool_call_id: str | None = None
    is_error: bool = False


class NexagenResponse(BaseModel):
    """Wraps a NexagenMessage returned by a provider."""

    message: NexagenMessage

    @property
    def has_tool_calls(self) -> bool:
        """Return True if the response contains tool calls."""
        return bool(self.message.tool_calls)


class ProviderConfig(BaseModel):
    """Configuration for connecting to an LLM provider."""

    backend: str
    model: str
    base_url: str | None = None
    api_key: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None

    def model_post_init(self, __context):
        if self.base_url and not self.base_url.startswith(("http://", "https://")):
            raise ValueError(
                f"Invalid base_url protocol: '{self.base_url}'. "
                f"Only http:// and https:// are allowed."
            )

    @classmethod
    def from_string(cls, provider_string: str) -> ProviderConfig:
        """Parse a provider string like 'ollama/qwen3@192.168.1.5:11434'.

        Format: backend/model[@host]

        Raises ValueError if the string is malformed.
        """
        if not provider_string or not isinstance(provider_string, str):
            raise ValueError("Provider string must be a non-empty string")

        base_url = None
        if "@" in provider_string:
            provider_part, host = provider_string.split("@", 1)
            if not host:
                raise ValueError(f"Invalid provider string: host is empty in '{provider_string}'")
            if not host.startswith("http"):
                base_url = f"http://{host}"
            else:
                base_url = host
            # Block file:// and other dangerous protocols
            if base_url and not base_url.startswith(("http://", "https://")):
                raise ValueError(f"Invalid base URL protocol: only http:// and https:// are allowed")
        else:
            provider_part = provider_string

        if "/" not in provider_part:
            raise ValueError(
                f"Invalid provider string: '{provider_string}'. "
                f"Expected format: 'backend/model' (e.g., 'ollama/qwen3')"
            )

        backend, model = provider_part.split("/", 1)
        if not backend or not model:
            raise ValueError(f"Invalid provider string: backend and model must not be empty")

        return cls(backend=backend, model=model, base_url=base_url)
