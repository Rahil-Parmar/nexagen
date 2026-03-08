"""Tests for the ContextManager class."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nexagen.constants import (
    CHARS_PER_TOKEN,
    DEFAULT_CONTEXT_THRESHOLD,
    DEFAULT_RECENT_TOOL_RESULTS_TO_KEEP,
)
from nexagen.context import ContextManager
from nexagen.models import NexagenMessage, ToolCall


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tool_msg(text: str, tool_call_id: str = "tc1", is_error: bool = False) -> NexagenMessage:
    return NexagenMessage(role="tool", text=text, tool_call_id=tool_call_id, is_error=is_error)


def _assistant_msg(text: str, tool_calls: list[ToolCall] | None = None) -> NexagenMessage:
    return NexagenMessage(role="assistant", text=text, tool_calls=tool_calls)


def _user_msg(text: str) -> NexagenMessage:
    return NexagenMessage(role="user", text=text)


def _system_msg(text: str) -> NexagenMessage:
    return NexagenMessage(role="system", text=text)


# ===========================================================================
# TestObservationMasking
# ===========================================================================


class TestObservationMasking:
    """Tier-1 observation masking tests."""

    def test_keeps_all_assistant_messages(self):
        cm = ContextManager()
        msgs = [
            _assistant_msg("Hello"),
            _assistant_msg("World"),
        ]
        result = cm.mask_observations(msgs)
        assert len(result) == 2
        assert result[0].text == "Hello"
        assert result[1].text == "World"

    def test_keeps_recent_tool_results_verbatim(self):
        cm = ContextManager(recent_tool_results_to_keep=2)
        msgs = [
            _tool_msg("old result", tool_call_id="tc1"),
            _tool_msg("recent1", tool_call_id="tc2"),
            _tool_msg("recent2", tool_call_id="tc3"),
        ]
        result = cm.mask_observations(msgs)
        # Last 2 tool results kept verbatim
        assert result[1].text == "recent1"
        assert result[2].text == "recent2"
        # First one is masked
        assert result[0].text == "[Tool result: success]"

    def test_masks_old_tool_results_as_stubs(self):
        cm = ContextManager(recent_tool_results_to_keep=1)
        msgs = [
            _tool_msg("first", tool_call_id="tc1"),
            _tool_msg("second", tool_call_id="tc2"),
            _tool_msg("third", tool_call_id="tc3"),
        ]
        result = cm.mask_observations(msgs)
        assert result[0].text == "[Tool result: success]"
        assert result[1].text == "[Tool result: success]"
        assert result[2].text == "third"  # last one kept

    def test_never_touches_system_or_user_messages(self):
        cm = ContextManager()
        msgs = [
            _system_msg("You are a helpful assistant."),
            _user_msg("Do something"),
            _tool_msg("old tool output", tool_call_id="tc1"),
        ]
        result = cm.mask_observations(msgs)
        assert result[0].text == "You are a helpful assistant."
        assert result[0].role == "system"
        assert result[1].text == "Do something"
        assert result[1].role == "user"

    def test_handles_error_tool_results(self):
        cm = ContextManager(recent_tool_results_to_keep=1)
        msgs = [
            _tool_msg("traceback: something broke", tool_call_id="tc1", is_error=True),
            _tool_msg("good result", tool_call_id="tc2"),
        ]
        result = cm.mask_observations(msgs)
        # Old error tool result gets error stub
        assert result[0].text == "[Tool result: error]"
        assert result[0].is_error is True
        # Recent one kept verbatim
        assert result[1].text == "good result"

    def test_empty_messages(self):
        cm = ContextManager()
        result = cm.mask_observations([])
        assert result == []

    def test_no_tool_messages_returns_unchanged(self):
        cm = ContextManager()
        msgs = [
            _system_msg("sys"),
            _user_msg("hi"),
            _assistant_msg("hello"),
        ]
        result = cm.mask_observations(msgs)
        assert len(result) == 3
        assert result[0].text == "sys"
        assert result[1].text == "hi"
        assert result[2].text == "hello"

    def test_does_not_mutate_input(self):
        cm = ContextManager(recent_tool_results_to_keep=1)
        original_tool = _tool_msg("old output", tool_call_id="tc1")
        msgs = [original_tool, _tool_msg("new", tool_call_id="tc2")]
        cm.mask_observations(msgs)
        # Original message should be unchanged
        assert msgs[0].text == "old output"


# ===========================================================================
# TestEstimateTokens
# ===========================================================================


class TestEstimateTokens:
    def test_estimate_messages_tokens(self):
        cm = ContextManager()
        msgs = [
            _user_msg("Hello world"),  # 11 chars
            _assistant_msg(
                "Sure",
                tool_calls=[ToolCall(id="tc1", name="read_file", arguments={"path": "/tmp/f"})],
            ),
        ]
        result = cm.estimate_tokens(msgs)
        text_chars = len("Hello world") + len("Sure")
        tc_chars = len("read_file") + len(str({"path": "/tmp/f"}))
        expected = (text_chars + tc_chars) // CHARS_PER_TOKEN
        assert result == expected

    def test_estimate_empty(self):
        cm = ContextManager()
        assert cm.estimate_tokens([]) == 0


# ===========================================================================
# TestShapeContext
# ===========================================================================


class TestShapeContext:
    async def test_no_shaping_needed_under_threshold(self):
        cm = ContextManager()
        msgs = [_user_msg("short")]
        result = await cm.shape_context(msgs, context_window=100_000)
        assert result == msgs

    async def test_masking_applied_when_over_threshold(self):
        cm = ContextManager(recent_tool_results_to_keep=1)
        big_text = "x" * 400
        msgs = [
            _tool_msg(big_text, tool_call_id="tc1"),
            _tool_msg(big_text, tool_call_id="tc2"),
            _tool_msg("keep", tool_call_id="tc3"),
        ]
        result = await cm.shape_context(msgs, context_window=200)
        assert result[0].text == "[Tool result: success]"
        assert result[1].text == "[Tool result: success]"
        assert result[2].text == "keep"

    async def test_tier2_compression_called_when_still_over(self):
        cm = ContextManager(recent_tool_results_to_keep=0)
        big_text = "x" * 400
        # Need > 4 messages for tier 2 compression to apply
        msgs = [
            _system_msg("system"),
            _user_msg(big_text),
            _assistant_msg(big_text),
            _tool_msg(big_text, tool_call_id="tc1"),
            _assistant_msg("latest"),
            _tool_msg("recent", tool_call_id="tc2"),
            _assistant_msg("end"),
        ]
        supervisor = MagicMock()
        supervisor.compress_history = AsyncMock(return_value="compressed summary")
        result = await cm.shape_context(msgs, context_window=50, supervisor=supervisor)
        # Tier 2 should have been called and compressed
        supervisor.compress_history.assert_called_once()
        # Should have: first + summary + last 3
        assert len(result) == 5
        assert result[1].summary == "compressed summary"

    async def test_no_supervisor_returns_masked(self):
        cm = ContextManager(recent_tool_results_to_keep=1)
        big_text = "x" * 400
        msgs = [
            _user_msg(big_text),
            _tool_msg(big_text, tool_call_id="tc1"),
            _tool_msg("keep", tool_call_id="tc2"),
        ]
        result = await cm.shape_context(msgs, context_window=50, supervisor=None)
        assert result[1].text == "[Tool result: success]"
        assert result[2].text == "keep"
