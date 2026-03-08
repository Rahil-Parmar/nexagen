"""Tests for the supervisor agent."""

from __future__ import annotations

import pytest

from nexagen.models import NexagenMessage, NexagenResponse
from nexagen.supervisor.supervisor import ActionEntry, SupervisorAgent, SupervisorFeedback


class MockSupervisorProvider:
    """Mock provider that returns predefined responses."""

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_messages: list[NexagenMessage] | None = None

    async def chat(self, messages, tools=None):
        self.last_messages = messages
        return NexagenResponse(
            message=NexagenMessage(role="assistant", text=self.response_text)
        )

    def supports_tool_calling(self):
        return False

    def supports_vision(self):
        return False


class FailingProvider:
    """Mock provider that always raises."""

    async def chat(self, messages, tools=None):
        raise RuntimeError("provider down")

    def supports_tool_calling(self):
        return False

    def supports_vision(self):
        return False


# --- _parse_feedback tests ---


class TestParseFeedback:
    def setup_method(self):
        self.provider = MockSupervisorProvider("")
        self.agent = SupervisorAgent(provider=self.provider)

    def test_parse_feedback_json_continue(self):
        result = self.agent._parse_feedback('{"decision": "continue"}')
        assert isinstance(result, SupervisorFeedback)
        assert result.decision == "continue"
        assert result.diagnosis is None
        assert result.suggestion is None

    def test_parse_feedback_json_stop(self):
        result = self.agent._parse_feedback('{"decision": "stop"}')
        assert result.decision == "stop"

    def test_parse_feedback_json_redirect(self):
        text = '{"decision": "redirect", "diagnosis": "stuck in loop", "suggestion": "try a different approach"}'
        result = self.agent._parse_feedback(text)
        assert result.decision == "redirect"
        assert result.diagnosis == "stuck in loop"
        assert result.suggestion == "try a different approach"

    def test_parse_feedback_code_block_json(self):
        text = '```json\n{"decision": "continue"}\n```'
        result = self.agent._parse_feedback(text)
        assert result.decision == "continue"

    def test_parse_feedback_text_fallback_continue(self):
        result = self.agent._parse_feedback("I think you should continue working")
        assert result.decision == "continue"

    def test_parse_feedback_text_fallback_stop(self):
        result = self.agent._parse_feedback("You should stop now")
        assert result.decision == "stop"

    def test_parse_feedback_gibberish_defaults_to_stop(self):
        result = self.agent._parse_feedback("gibberish text with no relevant keywords")
        assert result.decision == "stop"

    def test_parse_feedback_invalid_decision_defaults_to_stop(self):
        result = self.agent._parse_feedback('{"decision": "banana"}')
        assert result.decision == "stop"


# --- check_progress tests ---


class TestCheckProgress:
    @pytest.mark.asyncio
    async def test_check_progress_calls_provider(self):
        provider = MockSupervisorProvider('{"decision": "continue"}')
        agent = SupervisorAgent(provider=provider)
        action_log = [ActionEntry("Read config", ["file_read"])]
        result = await agent.check_progress("Fix the bug", action_log)
        assert isinstance(result, SupervisorFeedback)
        assert result.decision == "continue"
        assert provider.last_messages is not None
        assert len(provider.last_messages) == 1
        assert provider.last_messages[0].role == "user"

    @pytest.mark.asyncio
    async def test_check_progress_returns_redirect(self):
        text = '{"decision": "redirect", "diagnosis": "looping", "suggestion": "reset"}'
        provider = MockSupervisorProvider(text)
        agent = SupervisorAgent(provider=provider)
        action_log = [ActionEntry("Retry same step", ["file_read"])]
        result = await agent.check_progress("Fix the bug", action_log)
        assert isinstance(result, SupervisorFeedback)
        assert result.decision == "redirect"
        assert result.diagnosis == "looping"
        assert result.suggestion == "reset"

    @pytest.mark.asyncio
    async def test_check_progress_provider_failure_returns_continue(self):
        agent = SupervisorAgent(provider=FailingProvider())
        action_log = [ActionEntry("Read config", ["file_read"])]
        result = await agent.check_progress("Fix the bug", action_log)
        assert isinstance(result, SupervisorFeedback)
        assert result.decision == "continue"


# --- compress_history tests ---


class TestCompressHistory:
    @pytest.mark.asyncio
    async def test_compress_history(self):
        provider = MockSupervisorProvider("The agent read a config file and found the bug.")
        agent = SupervisorAgent(provider=provider)
        messages = [
            NexagenMessage(role="assistant", text="I will read the config file."),
            NexagenMessage(role="tool", text="config contents here", is_error=False),
            NexagenMessage(role="assistant", summary="Found the bug in config"),
        ]
        result = await agent.compress_history(messages)
        assert result == "The agent read a config file and found the bug."
        assert provider.last_messages is not None


# --- ActionEntry tests ---


class TestActionEntry:
    def test_action_entry_str(self):
        entry = ActionEntry("Reading file", ["file_read"])
        assert str(entry) == "Reading file [file_read]"

    def test_action_entry_str_multiple_tools(self):
        entry = ActionEntry("Searching codebase", ["grep", "glob"])
        assert str(entry) == "Searching codebase [grep, glob]"


# --- _build_progress_prompt tests ---


class TestBuildProgressPrompt:
    def test_build_progress_prompt(self):
        provider = MockSupervisorProvider("")
        agent = SupervisorAgent(provider=provider)
        action_log = [
            ActionEntry("Read file", ["file_read"]),
            ActionEntry("Wrote fix", ["file_write"]),
        ]
        prompt = agent._build_progress_prompt("Fix the bug", action_log)
        assert "Fix the bug" in prompt
        assert "Step 1:" in prompt
        assert "Step 2:" in prompt
        assert "Read file [file_read]" in prompt
        assert "Wrote fix [file_write]" in prompt
        assert "redirect" in prompt
