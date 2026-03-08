"""Tests for the ReflectionEngine and ReflectionResult."""

from __future__ import annotations

import json

import pytest

from nexagen.models import NexagenMessage, NexagenResponse
from nexagen.reflection import ReflectionEngine, ReflectionResult


class MockReflectionProvider:
    """Mock LLM provider for reflection tests."""

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
    """Mock provider that raises an exception."""

    async def chat(self, messages, tools=None):
        raise RuntimeError("Provider is down")

    def supports_tool_calling(self):
        return False

    def supports_vision(self):
        return False


@pytest.mark.asyncio
async def test_reflect_returns_structured_result():
    """ReflectionEngine should parse a valid JSON response into a ReflectionResult."""
    response_json = json.dumps(
        {
            "diagnosis": "The tool failed because the file path was incorrect.",
            "strategy": "Use the absolute path instead of relative.",
            "should_retry": True,
        }
    )
    provider = MockReflectionProvider(response_json)
    engine = ReflectionEngine(provider=provider, max_reflections=3)

    result = await engine.reflect(
        original_task="Summarize the document",
        failed_action="read_file",
        error="FileNotFoundError: no such file",
        past_reflections=[],
    )

    assert isinstance(result, ReflectionResult)
    assert result.diagnosis == "The tool failed because the file path was incorrect."
    assert result.strategy == "Use the absolute path instead of relative."
    assert result.should_retry is True
    assert provider.call_count == 1


@pytest.mark.asyncio
async def test_reflect_with_past_reflections():
    """Past reflections should be included in the prompt context."""
    response_json = json.dumps(
        {
            "diagnosis": "Second attempt also failed due to permissions.",
            "strategy": "Request elevated permissions before retrying.",
            "should_retry": False,
        }
    )
    provider = MockReflectionProvider(response_json)
    engine = ReflectionEngine(provider=provider)

    past = [
        ReflectionResult(
            diagnosis="File not found",
            strategy="Try absolute path",
            should_retry=True,
        )
    ]

    result = await engine.reflect(
        original_task="Summarize the document",
        failed_action="read_file",
        error="PermissionError: access denied",
        past_reflections=past,
    )

    assert result.diagnosis == "Second attempt also failed due to permissions."
    assert result.strategy == "Request elevated permissions before retrying."
    assert result.should_retry is False
    assert provider.call_count == 1


@pytest.mark.asyncio
async def test_reflect_provider_failure_returns_no_retry():
    """If the provider itself fails, return a safe default with should_retry=False."""
    provider = FailingProvider()
    engine = ReflectionEngine(provider=provider)

    result = await engine.reflect(
        original_task="Summarize the document",
        failed_action="read_file",
        error="FileNotFoundError",
        past_reflections=[],
    )

    assert isinstance(result, ReflectionResult)
    assert result.should_retry is False
    # diagnosis should contain some indication of the reflection failure
    assert result.diagnosis != ""


@pytest.mark.asyncio
async def test_reflect_malformed_json_still_works():
    """If the LLM returns free text instead of JSON, fall back to text parsing."""
    free_text = (
        "The action failed because the API rate limit was exceeded. "
        "You should retry after waiting a moment."
    )
    provider = MockReflectionProvider(free_text)
    engine = ReflectionEngine(provider=provider)

    result = await engine.reflect(
        original_task="Fetch weather data",
        failed_action="call_api",
        error="RateLimitError: 429",
        past_reflections=[],
    )

    assert isinstance(result, ReflectionResult)
    # Free text becomes the diagnosis
    assert "rate limit" in result.diagnosis.lower()
    # should_retry True because text contains "retry"
    assert result.should_retry is True
