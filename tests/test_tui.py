"""Tests for the nexagen TUI application."""

import pytest
from nexagen.tui.app import NexagenApp, MessageDisplay, StepProgress, run_tui


def test_nexagen_app_creation():
    app = NexagenApp(provider="ollama/qwen3")
    assert app.provider == "ollama/qwen3"


def test_nexagen_app_custom_config():
    app = NexagenApp(provider="openai/gpt-4o", tools=["file_read"], permission_mode="readonly")
    assert app.provider == "openai/gpt-4o"
    assert app.tools == ["file_read"]
    assert app.permission_mode == "readonly"


def test_message_display_user():
    msg = MessageDisplay("user", "Hello")
    rendered = msg.render()
    assert "Hello" in rendered.plain


def test_message_display_assistant():
    msg = MessageDisplay("assistant", "I can help")
    rendered = msg.render()
    assert "I can help" in rendered.plain


def test_message_display_tool_error():
    msg = MessageDisplay("tool", "Failed", is_error=True)
    rendered = msg.render()
    assert "Failed" in rendered.plain


def test_step_progress_completed():
    step = StepProgress(1, "Reading auth.py", completed=True)
    rendered = step.render()
    assert "Step 1" in rendered.plain
    assert "Reading auth.py" in rendered.plain


def test_step_progress_in_progress():
    step = StepProgress(2, "Searching for bugs", completed=False)
    rendered = step.render()
    assert "Step 2" in rendered.plain


def test_run_tui_is_callable():
    assert callable(run_tui)
