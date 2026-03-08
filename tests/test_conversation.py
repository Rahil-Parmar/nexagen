"""Tests for the Conversation class."""

from __future__ import annotations

import pytest

from nexagen.conversation import Conversation
from nexagen.models import NexagenMessage, ToolCall
from nexagen.constants import CHARS_PER_TOKEN, DEFAULT_CONTEXT_THRESHOLD


class TestConversation:
    def test_add_message(self):
        conv = Conversation()
        msg = NexagenMessage(role="user", text="hello")
        conv.add_message(msg)
        assert len(conv.messages) == 1
        assert conv.messages[0] is msg

    def test_add_messages(self):
        conv = Conversation()
        msgs = [
            NexagenMessage(role="user", text="hello"),
            NexagenMessage(role="assistant", text="hi there"),
        ]
        conv.add_messages(msgs)
        assert len(conv.messages) == 2
        assert conv.messages[0].text == "hello"
        assert conv.messages[1].text == "hi there"

    def test_estimate_tokens(self):
        conv = Conversation()
        # 20 chars of text -> 20 // 4 = 5 tokens
        conv.add_message(NexagenMessage(role="user", text="a" * 20))
        assert conv.estimate_tokens() == 20 // CHARS_PER_TOKEN

    def test_estimate_tokens_with_tool_calls(self):
        conv = Conversation()
        tc = ToolCall(id="1", name="mytool", arguments={"key": "val"})
        conv.add_message(NexagenMessage(role="assistant", text="ok", tool_calls=[tc]))
        # text "ok" = 2 chars, tool name "mytool" = 6 chars, str(arguments) = str({"key": "val"})
        arg_str = str({"key": "val"})
        expected = (2 + len(arg_str) + 6) // CHARS_PER_TOKEN
        assert conv.estimate_tokens() == expected

    def test_needs_compression_false(self):
        conv = Conversation(context_window=8192)
        conv.add_message(NexagenMessage(role="user", text="short"))
        assert conv.needs_compression() is False

    def test_needs_compression_true(self):
        conv = Conversation(context_window=100)
        # 100 * 0.80 = 80 token threshold -> 80 * 4 = 320 chars needed
        conv.add_message(NexagenMessage(role="user", text="x" * 400))
        assert conv.needs_compression() is True

    def test_get_compressible_messages(self):
        conv = Conversation()
        msgs = [NexagenMessage(role="user", text=f"msg{i}") for i in range(6)]
        conv.add_messages(msgs)
        compressible = conv.get_compressible_messages()
        # Should be messages[1:-3] -> indices 1, 2
        assert len(compressible) == 2
        assert compressible[0].text == "msg1"
        assert compressible[1].text == "msg2"

    def test_get_compressible_messages_short(self):
        conv = Conversation()
        msgs = [NexagenMessage(role="user", text=f"msg{i}") for i in range(4)]
        conv.add_messages(msgs)
        assert conv.get_compressible_messages() == []

    def test_compress(self):
        conv = Conversation()
        msgs = [NexagenMessage(role="user", text=f"msg{i}") for i in range(6)]
        conv.add_messages(msgs)
        conv.compress("summary of middle messages")
        assert len(conv.messages) == 5  # first + summary + last 3
        assert conv.messages[0].text == "msg0"
        assert conv.messages[1].summary == "summary of middle messages"
        assert conv.messages[1].role == "assistant"
        assert conv.messages[2].text == "msg3"
        assert conv.messages[3].text == "msg4"
        assert conv.messages[4].text == "msg5"

    def test_compress_short_conversation_noop(self):
        conv = Conversation()
        msgs = [NexagenMessage(role="user", text=f"msg{i}") for i in range(3)]
        conv.add_messages(msgs)
        conv.compress("should not apply")
        assert len(conv.messages) == 3

    def test_complete_task(self):
        conv = Conversation()
        conv.add_message(NexagenMessage(role="user", text="do something"))
        conv.add_message(NexagenMessage(role="assistant", text="done"))
        conv.complete_task("Task completed successfully")
        assert len(conv.messages) == 0
        assert len(conv.task_summaries) == 1
        assert conv.task_summaries[0] == "Task completed successfully"

    def test_get_messages_with_history(self):
        conv = Conversation()
        conv.complete_task("Previous task summary")
        conv.add_message(NexagenMessage(role="user", text="new task"))

        result = conv.get_messages_with_history(system_prompt="You are helpful.")
        assert len(result) == 3
        assert result[0].role == "system"
        assert result[0].text == "You are helpful."
        assert result[1].role == "assistant"
        assert result[1].summary == "Previous task summary"
        assert result[2].role == "user"
        assert result[2].text == "new task"

    def test_get_messages_with_history_no_system_prompt(self):
        conv = Conversation()
        conv.add_message(NexagenMessage(role="user", text="hi"))
        result = conv.get_messages_with_history()
        assert len(result) == 1
        assert result[0].text == "hi"

    def test_clear(self):
        conv = Conversation()
        conv.add_message(NexagenMessage(role="user", text="hello"))
        conv.complete_task("summary")
        conv.add_message(NexagenMessage(role="user", text="another"))
        conv.clear()
        assert len(conv.messages) == 0
        assert len(conv.task_summaries) == 0
