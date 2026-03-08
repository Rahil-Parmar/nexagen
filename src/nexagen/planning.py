"""Planning module for auto-detecting complexity and generating structured execution plans."""

from __future__ import annotations

import json
from dataclasses import dataclass

from nexagen.models import NexagenMessage, NexagenResponse
from nexagen.providers.base import LLMProvider


@dataclass
class Subtask:
    """A single step in an execution plan."""

    description: str
    status: str = "pending"  # "pending" | "in_progress" | "completed" | "failed"


@dataclass
class Plan:
    """An ordered list of subtasks to execute."""

    subtasks: list[Subtask]
    current_step: int = 0

    def advance(self) -> None:
        """Mark the current subtask as completed and move to the next step."""
        if self.current_step < len(self.subtasks):
            self.subtasks[self.current_step].status = "completed"
            self.current_step += 1

    @property
    def is_complete(self) -> bool:
        """Return True when all subtasks have been processed."""
        return self.current_step >= len(self.subtasks)


class PlanningPhase:
    """Orchestrates complexity classification and plan generation via an LLM provider."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    async def classify_complexity(self, prompt: str) -> str:
        """Classify *prompt* as ``"simple"`` or ``"complex"`` using a short LLM call."""
        try:
            response: NexagenResponse = await self.provider.chat(
                [
                    NexagenMessage(
                        role="system",
                        text=(
                            "Classify the following user request as simple or complex. "
                            'Respond with JSON: {"complexity": "simple"} or {"complexity": "complex"}'
                        ),
                    ),
                    NexagenMessage(role="user", text=prompt),
                ]
            )
            text = (response.message.text or "").strip()

            # Try JSON parse first
            try:
                data = json.loads(text)
                value = data.get("complexity", "").lower()
                if value in ("simple", "complex"):
                    return value
            except (json.JSONDecodeError, AttributeError):
                pass

            # Fallback: search for keywords in raw text
            lower = text.lower()
            if "complex" in lower:
                return "complex"
            if "simple" in lower:
                return "simple"

            return "simple"
        except Exception:
            return "simple"

    async def generate_plan(self, prompt: str) -> Plan:
        """Generate a :class:`Plan` of subtasks for the given *prompt*."""
        try:
            response: NexagenResponse = await self.provider.chat(
                [
                    NexagenMessage(
                        role="system",
                        text=(
                            "Break the following user request into a list of subtasks. "
                            'Respond with JSON: {"subtasks": ["step 1", "step 2", ...]}'
                        ),
                    ),
                    NexagenMessage(role="user", text=prompt),
                ]
            )
            text = (response.message.text or "").strip()

            try:
                data = json.loads(text)
                steps = data.get("subtasks", [])
                if isinstance(steps, list) and steps:
                    return Plan(subtasks=[Subtask(description=str(s)[:200]) for s in steps])
            except (json.JSONDecodeError, AttributeError):
                pass

            # Fallback: single-step plan with the original prompt
            return Plan(subtasks=[Subtask(description=prompt)])
        except Exception:
            return Plan(subtasks=[Subtask(description=prompt)])

    def format_plan_context(self, plan: Plan) -> str:
        """Format *plan* as a human-readable checklist string.

        Markers:
        - ``[x]`` completed
        - ``[>]`` current (in_progress or the step at ``current_step``)
        - ``[ ]`` pending
        """
        lines: list[str] = []
        for i, subtask in enumerate(plan.subtasks):
            if subtask.status == "completed":
                marker = "[x]"
            elif i == plan.current_step and not plan.is_complete:
                marker = "[>]"
            else:
                marker = "[ ]"
            lines.append(f"{marker} {subtask.description}")
        return "\n".join(lines)
