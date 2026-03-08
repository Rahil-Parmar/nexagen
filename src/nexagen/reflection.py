"""Self-reflection engine for error diagnosis and retry decisions."""

from __future__ import annotations

import json
from dataclasses import dataclass

from nexagen.constants import DEFAULT_MAX_REFLECTIONS
from nexagen.models import NexagenMessage, NexagenResponse
from nexagen.providers.base import LLMProvider


@dataclass
class ReflectionResult:
    """Result of a reflection on a failed action."""

    diagnosis: str
    strategy: str
    should_retry: bool


class ReflectionEngine:
    """Analyzes failures using an LLM and suggests retry strategies."""

    def __init__(
        self,
        provider: LLMProvider,
        max_reflections: int = DEFAULT_MAX_REFLECTIONS,
    ) -> None:
        self.provider = provider
        self.max_reflections = max_reflections

    async def reflect(
        self,
        original_task: str,
        failed_action: str,
        error: str,
        past_reflections: list[ReflectionResult],
    ) -> ReflectionResult:
        """Reflect on a failure and produce a diagnosis with retry recommendation."""
        prompt = self._build_prompt(original_task, failed_action, error, past_reflections)
        messages = [NexagenMessage(role="user", text=prompt)]

        try:
            response: NexagenResponse = await self.provider.chat(messages)
        except Exception:
            return ReflectionResult(
                diagnosis="Reflection failed: unable to reach the LLM provider.",
                strategy="No strategy available.",
                should_retry=False,
            )

        return self._parse_response(response.message.text or "")

    def _build_prompt(
        self,
        original_task: str,
        failed_action: str,
        error: str,
        past_reflections: list[ReflectionResult],
    ) -> str:
        # Truncate inputs to prevent prompt injection and context bloat
        safe_task = original_task[:500]
        safe_action = failed_action[:200]
        safe_error = error[:500]

        parts = [
            "You are a self-reflection module for an AI agent. Analyze the following failure and suggest a correction.",
            "",
            f"Original task: {safe_task}",
            "",
            "<failed_action>",
            safe_action,
            "</failed_action>",
            "",
            "<error_output>",
            safe_error,
            "</error_output>",
        ]

        if past_reflections:
            parts.append("")
            parts.append("Previous reflection attempts:")
            for i, ref in enumerate(past_reflections, 1):
                diag = ref.diagnosis[:200] if isinstance(ref.diagnosis, str) else str(ref.diagnosis)[:200]
                parts.append(f"  Attempt {i}: diagnosis={diag!r}, should_retry={ref.should_retry}")

        parts.append("")
        parts.append(
            'Respond with JSON: {"diagnosis": "...", "strategy": "...", "should_retry": true/false}'
        )

        return "\n".join(parts)

    def _parse_response(self, text: str) -> ReflectionResult:
        """Parse the LLM response, falling back to free-text extraction."""
        # Try JSON parsing first
        try:
            data = json.loads(text)
            return ReflectionResult(
                diagnosis=str(data.get("diagnosis", "")),
                strategy=str(data.get("strategy", "")),
                should_retry=bool(data.get("should_retry", False)),
            )
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

        # Fallback: use raw text as diagnosis, heuristic for should_retry
        lower = text.lower()
        should_retry = "retry" in lower or "try" in lower

        return ReflectionResult(
            diagnosis=text[:500],
            strategy="",
            should_retry=should_retry,
        )
