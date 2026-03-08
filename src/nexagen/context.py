"""Two-tier context management: observation masking (free) then LLM compression (expensive fallback)."""

from __future__ import annotations

from nexagen.constants import (
    CHARS_PER_TOKEN,
    DEFAULT_CONTEXT_THRESHOLD,
    DEFAULT_RECENT_TOOL_RESULTS_TO_KEEP,
)
from nexagen.models import NexagenMessage


class ContextManager:
    """Manages context window pressure via observation masking and optional LLM compression."""

    def __init__(
        self,
        recent_tool_results_to_keep: int = DEFAULT_RECENT_TOOL_RESULTS_TO_KEEP,
    ):
        self.recent_tool_results_to_keep = recent_tool_results_to_keep

    # ------------------------------------------------------------------
    # Token estimation (mirrors Conversation.estimate_tokens)
    # ------------------------------------------------------------------

    def estimate_tokens(self, messages: list[NexagenMessage]) -> int:
        """Estimate token count: sum(len(text) + tool_call args/names) // CHARS_PER_TOKEN."""
        total_chars = sum(
            len(msg.text or "")
            + sum(
                len(str(tc.arguments)) + len(tc.name)
                for tc in (msg.tool_calls or [])
            )
            for msg in messages
        )
        return max(0, total_chars // max(1, CHARS_PER_TOKEN))

    # ------------------------------------------------------------------
    # Tier 1 — Observation masking
    # ------------------------------------------------------------------

    def mask_observations(self, messages: list[NexagenMessage]) -> list[NexagenMessage]:
        """Replace older tool results with short stubs; keep recent N verbatim.

        - All assistant / system / user messages are kept intact.
        - The last ``recent_tool_results_to_keep`` tool messages are kept verbatim.
        - Older tool messages are replaced with ``[Tool result: success]`` or
          ``[Tool result: error]`` depending on ``is_error``.
        - Never mutates the input list or its messages.
        - Returns an empty list for empty input.
        """
        if not messages:
            return []

        # Identify indices of tool messages (in original order)
        tool_indices: list[int] = [
            i for i, msg in enumerate(messages) if msg.role == "tool"
        ]

        # Determine which tool indices are "recent" (last N)
        keep_count = self.recent_tool_results_to_keep
        if keep_count >= len(tool_indices):
            recent_set: set[int] = set(tool_indices)
        else:
            recent_set = set(tool_indices[-keep_count:])

        result: list[NexagenMessage] = []
        for i, msg in enumerate(messages):
            if msg.role != "tool" or i in recent_set:
                # Keep as-is (shallow copy via model_copy for safety)
                result.append(msg.model_copy())
            else:
                # Mask old tool result
                stub = "[Tool result: error]" if msg.is_error else "[Tool result: success]"
                result.append(
                    NexagenMessage(
                        role="tool",
                        text=stub,
                        tool_call_id=msg.tool_call_id,
                        is_error=msg.is_error,
                    )
                )
        return result

    # ------------------------------------------------------------------
    # Orchestrator — shape_context
    # ------------------------------------------------------------------

    async def shape_context(
        self,
        messages: list[NexagenMessage],
        context_window: int,
        supervisor=None,
    ) -> list[NexagenMessage]:
        """Orchestrate two-tier context management.

        1. If under threshold (80% of *context_window*) -> return unchanged.
        2. Tier 1: mask observations.
        3. If still over threshold and *supervisor* is available -> Tier 2 LLM
           compression via ``supervisor.compress_history``.
        4. If Tier 2 fails or no supervisor -> return masked result.
        """
        threshold = int(context_window * DEFAULT_CONTEXT_THRESHOLD)

        # Under threshold — no work needed
        if self.estimate_tokens(messages) < threshold:
            return messages

        # Tier 1 — observation masking
        masked = self.mask_observations(messages)

        # Check if masking was enough
        if self.estimate_tokens(masked) < threshold:
            return masked

        # Tier 2 — LLM compression (best-effort)
        if supervisor is not None and len(masked) > 4:
            try:
                compressible = masked[1:-3]
                summary = await supervisor.compress_history(compressible)
                if summary:
                    first = masked[0]
                    tail = masked[-3:]
                    summary_msg = NexagenMessage(
                        role="assistant", text=summary, summary=summary
                    )
                    return [first, summary_msg] + tail
            except Exception:
                pass  # Fall through to return masked

        return masked
