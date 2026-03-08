"""Conversation management for the nexagen SDK."""

from __future__ import annotations

from nexagen.constants import CHARS_PER_TOKEN, DEFAULT_CONTEXT_THRESHOLD
from nexagen.models import NexagenMessage


class Conversation:
    """Manages message history, token estimation, context compression, and task summaries."""

    def __init__(self, context_window: int = 8192):
        self.messages: list[NexagenMessage] = []
        self.task_summaries: list[str] = []
        self.context_window = context_window  # in tokens

    def add_message(self, message: NexagenMessage):
        """Add a single message to the conversation."""
        self.messages.append(message)

    def add_messages(self, messages: list[NexagenMessage]):
        """Add multiple messages to the conversation."""
        self.messages.extend(messages)

    def estimate_tokens(self) -> int:
        """Estimate the total token count of all messages using chars / CHARS_PER_TOKEN."""
        try:
            total_chars = sum(
                len(msg.text or "")
                + sum(
                    len(str(tc.arguments)) + len(tc.name)
                    for tc in (msg.tool_calls or [])
                )
                for msg in self.messages
            )
            return max(0, total_chars // max(1, CHARS_PER_TOKEN))
        except (TypeError, AttributeError):
            # Malformed messages — estimate conservatively
            return len(self.messages) * 100

    def needs_compression(self) -> bool:
        """Return True if estimated tokens exceed the context threshold."""
        try:
            return self.estimate_tokens() >= int(self.context_window * DEFAULT_CONTEXT_THRESHOLD)
        except (TypeError, ValueError):
            return False

    def get_compressible_messages(self) -> list[NexagenMessage]:
        """Returns messages that can be compressed (everything except first user msg + last 3)."""
        if len(self.messages) <= 4:
            return []
        return self.messages[1:-3]

    def compress(self, summary: str):
        """Replace compressible messages with a summary message."""
        if len(self.messages) <= 4:
            return
        first = self.messages[0]
        last_three = self.messages[-3:]
        summary_msg = NexagenMessage(role="assistant", text=summary, summary=summary)
        self.messages = [first, summary_msg] + last_three

    def complete_task(self, summary: str):
        """Called when a task completes. Stores summary and resets messages."""
        self.task_summaries.append(summary)
        self.messages = []

    def get_messages_for_llm(
        self,
        system_prompt: str | None = None,
        context_manager=None,
        context_window: int | None = None,
    ) -> list[NexagenMessage]:
        """Build a shaped view of messages for the LLM. Never mutates the source.

        If context_manager is provided, applies observation masking when over threshold.
        This is the KV-cache-aware path: source messages stay append-only,
        shaping happens at read time only.
        """
        result: list[NexagenMessage] = []
        if system_prompt:
            result.append(NexagenMessage(role="system", text=system_prompt))
        for summary in self.task_summaries:
            result.append(NexagenMessage(role="assistant", text=summary, summary=summary))
        result.extend(self.messages)

        if context_manager is not None:
            window = context_window or self.context_window
            threshold = int(window * DEFAULT_CONTEXT_THRESHOLD)
            tokens = context_manager.estimate_tokens(result)
            if tokens >= threshold:
                result = context_manager.mask_observations(result)

        return result

    def get_messages_with_history(self, system_prompt: str | None = None) -> list[NexagenMessage]:
        """Get messages including task summaries from previous tasks."""
        result: list[NexagenMessage] = []
        if system_prompt:
            result.append(NexagenMessage(role="system", text=system_prompt))
        for summary in self.task_summaries:
            result.append(NexagenMessage(role="assistant", text=summary, summary=summary))
        result.extend(self.messages)
        return result

    def clear(self):
        """Reset all messages and task summaries."""
        self.messages = []
        self.task_summaries = []
