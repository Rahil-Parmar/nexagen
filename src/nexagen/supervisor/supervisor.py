"""Supervisor agent that monitors worker progress and compresses context."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from nexagen.models import NexagenMessage
from nexagen.providers.base import LLMProvider


@dataclass
class SupervisorFeedback:
    decision: str  # "continue" | "stop" | "redirect"
    diagnosis: str | None = None
    suggestion: str | None = None


class ActionEntry:
    """One step in the worker's action log."""

    def __init__(self, summary: str, tool_names: list[str]):
        self.summary = summary
        self.tool_names = tool_names

    def __str__(self) -> str:
        tools = ", ".join(self.tool_names)
        return f"{self.summary} [{tools}]"


class SupervisorAgent:
    """Monitors worker agent progress and handles context compression."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def _build_progress_prompt(
        self, original_task: str, action_log: list[ActionEntry]
    ) -> str:
        log_text = "\n".join(
            f"  Step {i + 1}: {entry}" for i, entry in enumerate(action_log)
        )
        return (
            f"You are a supervisor monitoring an AI agent's progress.\n\n"
            f"Original task: {original_task}\n\n"
            f"Actions taken so far:\n{log_text}\n\n"
            f"Is the agent making progress toward completing the original task?\n"
            f'Respond with JSON: {{"decision": "continue"}}, {{"decision": "stop"}}, '
            f'or {{"decision": "redirect", "diagnosis": "what went wrong", "suggestion": "what to try instead"}}'
        )

    def _parse_feedback(self, response_text: str) -> SupervisorFeedback:
        """Parse supervisor response into structured feedback.

        Try JSON first, fall back to text search, default to stop (safe).
        """
        valid_decisions = ("continue", "stop", "redirect")

        def _feedback_from_data(data: dict) -> SupervisorFeedback | None:
            decision = data.get("decision", "").lower()
            if decision in valid_decisions:
                return SupervisorFeedback(
                    decision=decision,
                    diagnosis=data.get("diagnosis"),
                    suggestion=data.get("suggestion"),
                )
            return None

        # Try direct JSON parse
        try:
            data = json.loads(response_text)
            fb = _feedback_from_data(data)
            if fb is not None:
                return fb
        except (json.JSONDecodeError, AttributeError):
            pass

        # Try to find JSON in text (might be wrapped in code blocks)
        json_match = re.search(r"\{[^}]+\}", response_text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                fb = _feedback_from_data(data)
                if fb is not None:
                    return fb
            except (json.JSONDecodeError, AttributeError):
                pass

        # Fallback: text search
        text_lower = response_text.lower()
        if "continue" in text_lower:
            return SupervisorFeedback(decision="continue")
        if "stop" in text_lower:
            return SupervisorFeedback(decision="stop")

        # Default to stop (safe)
        return SupervisorFeedback(decision="stop")

    async def check_progress(
        self, original_task: str, action_log: list[ActionEntry]
    ) -> SupervisorFeedback:
        """Returns a SupervisorFeedback with decision and optional diagnosis/suggestion.

        If the supervisor LLM call fails, defaults to continue
        (non-fatal — let the worker keep going).
        """
        try:
            prompt = self._build_progress_prompt(original_task, action_log)
            messages = [NexagenMessage(role="user", text=prompt)]
            response = await self.provider.chat(messages)
            text = response.message.text if response and response.message else ""
            return self._parse_feedback(text or "")
        except Exception as e:
            import logging
            logging.getLogger("nexagen.supervisor").warning(
                "Supervisor check_progress failed: %s. Defaulting to 'continue'.", e
            )
            return SupervisorFeedback(decision="continue")

    def _build_compress_prompt(self, messages: list[NexagenMessage]) -> str:
        summaries = []
        for msg in messages:
            if msg.role == "assistant" and msg.summary:
                summaries.append(f"- {msg.summary}")
            elif msg.role == "assistant" and msg.text:
                summaries.append(f"- {msg.text[:100]}")
            elif msg.role == "tool":
                status = "error" if msg.is_error else "success"
                summaries.append(
                    f"- Tool result ({status}): {(msg.text or '')[:80]}"
                )

        content = "\n".join(summaries)
        return (
            f"Summarize these agent actions into a single concise paragraph "
            f"(2-3 sentences max) that captures the key findings and progress:\n\n"
            f"{content}"
        )

    async def compress_history(self, messages: list[NexagenMessage]) -> str:
        """Compress a list of messages into a summary string.

        If the LLM call fails, returns a basic concatenation of summaries
        rather than crashing.
        """
        try:
            prompt = self._build_compress_prompt(messages)
            response = await self.provider.chat(
                [NexagenMessage(role="user", text=prompt)]
            )
            text = response.message.text if response and response.message else None
            return text or "Summary unavailable."
        except Exception as e:
            import logging
            logging.getLogger("nexagen.supervisor").warning(
                "Supervisor compress_history failed: %s. Using fallback.", e
            )
            # Fallback: concatenate available summaries
            parts = []
            for msg in messages:
                if msg.summary:
                    parts.append(msg.summary)
                elif msg.role == "assistant" and msg.text:
                    parts.append(msg.text[:50])
            return "; ".join(parts) if parts else "Summary unavailable."
