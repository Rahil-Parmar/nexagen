"""Tests for the planning module."""

from __future__ import annotations

import pytest

from nexagen.models import NexagenMessage, NexagenResponse
from nexagen.planning import Plan, PlanningPhase, Subtask


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------

class MockPlanProvider:
    """Minimal mock that satisfies the LLMProvider protocol."""

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.call_count = 0

    async def chat(self, messages, tools=None):
        self.call_count += 1
        return NexagenResponse(
            message=NexagenMessage(role="assistant", text=self.response_text)
        )

    def supports_tool_calling(self):
        return False

    def supports_vision(self):
        return False


class FailingProvider:
    """Provider that always raises."""

    async def chat(self, messages, tools=None):
        raise RuntimeError("provider down")

    def supports_tool_calling(self):
        return False

    def supports_vision(self):
        return False


# ---------------------------------------------------------------------------
# TestSubtask
# ---------------------------------------------------------------------------

class TestSubtask:
    def test_default_status(self):
        s = Subtask(description="do something")
        assert s.status == "pending"

    def test_custom_status(self):
        s = Subtask(description="do something", status="completed")
        assert s.status == "completed"


# ---------------------------------------------------------------------------
# TestPlan
# ---------------------------------------------------------------------------

class TestPlan:
    def test_creation(self):
        plan = Plan(subtasks=[Subtask(description="a"), Subtask(description="b")])
        assert len(plan.subtasks) == 2
        assert plan.current_step == 0

    def test_advance_step(self):
        plan = Plan(subtasks=[Subtask(description="a"), Subtask(description="b")])
        plan.advance()
        assert plan.subtasks[0].status == "completed"
        assert plan.current_step == 1

    def test_advance_does_not_exceed_bounds(self):
        plan = Plan(subtasks=[Subtask(description="only")])
        plan.advance()  # completes the only subtask
        plan.advance()  # should not crash or exceed
        assert plan.current_step == 1
        assert plan.is_complete

    def test_is_complete(self):
        plan = Plan(subtasks=[Subtask(description="a")])
        assert not plan.is_complete
        plan.advance()
        assert plan.is_complete


# ---------------------------------------------------------------------------
# TestPlanFormatContext
# ---------------------------------------------------------------------------

class TestPlanFormatContext:
    def test_format_shows_all_markers(self):
        plan = Plan(
            subtasks=[
                Subtask(description="first", status="completed"),
                Subtask(description="second", status="in_progress"),
                Subtask(description="third", status="pending"),
            ],
            current_step=1,
        )
        provider = MockPlanProvider("")
        phase = PlanningPhase(provider)
        text = phase.format_plan_context(plan)

        assert "[x]" in text  # completed
        assert "[>]" in text  # current
        assert "[ ]" in text  # pending
        assert "first" in text
        assert "second" in text
        assert "third" in text


# ---------------------------------------------------------------------------
# TestClassifyComplexity
# ---------------------------------------------------------------------------

class TestClassifyComplexity:
    @pytest.mark.asyncio
    async def test_simple(self):
        provider = MockPlanProvider('{"complexity": "simple"}')
        phase = PlanningPhase(provider)
        result = await phase.classify_complexity("hello")
        assert result == "simple"

    @pytest.mark.asyncio
    async def test_complex(self):
        provider = MockPlanProvider('{"complexity": "complex"}')
        phase = PlanningPhase(provider)
        result = await phase.classify_complexity("build a website")
        assert result == "complex"

    @pytest.mark.asyncio
    async def test_bad_json_fallback_complex(self):
        provider = MockPlanProvider("this is complex task")
        phase = PlanningPhase(provider)
        result = await phase.classify_complexity("do many things")
        assert result == "complex"

    @pytest.mark.asyncio
    async def test_bad_json_fallback_simple(self):
        provider = MockPlanProvider("this is simple task")
        phase = PlanningPhase(provider)
        result = await phase.classify_complexity("say hi")
        assert result == "simple"

    @pytest.mark.asyncio
    async def test_provider_failure_defaults_simple(self):
        provider = FailingProvider()
        phase = PlanningPhase(provider)
        result = await phase.classify_complexity("anything")
        assert result == "simple"


# ---------------------------------------------------------------------------
# TestGeneratePlan
# ---------------------------------------------------------------------------

class TestGeneratePlan:
    @pytest.mark.asyncio
    async def test_parse_subtasks(self):
        provider = MockPlanProvider('{"subtasks": ["step 1", "step 2", "step 3"]}')
        phase = PlanningPhase(provider)
        plan = await phase.generate_plan("build something")
        assert len(plan.subtasks) == 3
        assert plan.subtasks[0].description == "step 1"
        assert plan.subtasks[2].description == "step 3"

    @pytest.mark.asyncio
    async def test_bad_json_returns_single_step(self):
        provider = MockPlanProvider("not json at all")
        phase = PlanningPhase(provider)
        plan = await phase.generate_plan("do the thing")
        assert len(plan.subtasks) == 1
        assert plan.subtasks[0].description == "do the thing"

    @pytest.mark.asyncio
    async def test_provider_failure_returns_single_step(self):
        provider = FailingProvider()
        phase = PlanningPhase(provider)
        plan = await phase.generate_plan("do the thing")
        assert len(plan.subtasks) == 1
        assert plan.subtasks[0].description == "do the thing"
