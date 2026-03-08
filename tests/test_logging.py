"""Tests for nexagen structured JSON logging."""
from __future__ import annotations

import json
import logging
import io

import pytest

from nexagen.agent_logging import (
    JSONFormatter,
    get_logger,
    log_tool_call,
    log_tool_result,
    log_supervisor_decision,
    log_error,
)


def _capture_log_output(logger: logging.Logger) -> io.StringIO:
    """Attach a StringIO handler with JSONFormatter to the logger and return the stream."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    return stream


def _make_fresh_logger(name: str) -> logging.Logger:
    """Return a fresh logger with no handlers, suitable for testing."""
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    return logger


class TestJSONFormatter:
    def test_json_formatter(self):
        """Format a log record and verify output is valid JSON with timestamp, level, event."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test_event",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)

        assert "timestamp" in data
        assert data["level"] == "INFO"
        assert data["event"] == "test_event"


class TestGetLogger:
    def test_get_logger(self):
        """get_logger returns a Logger with a JSONFormatter handler."""
        # Clear any existing logger state
        name = "nexagen_test_get_logger"
        logger = logging.getLogger(name)
        logger.handlers.clear()

        logger = get_logger(name)
        assert isinstance(logger, logging.Logger)
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0].formatter, JSONFormatter)

    def test_get_logger_idempotent(self):
        """Calling get_logger twice doesn't add duplicate handlers."""
        name = "nexagen_test_idempotent"
        logger = logging.getLogger(name)
        logger.handlers.clear()

        logger1 = get_logger(name)
        logger2 = get_logger(name)
        assert logger1 is logger2
        assert len(logger2.handlers) == 1


class TestLogToolCall:
    def test_log_tool_call(self):
        """Capture log output and verify JSON has tool, args, cycle fields."""
        logger = _make_fresh_logger("test_tool_call")
        stream = _capture_log_output(logger)

        log_tool_call(logger, tool_name="search", args={"query": "hello"}, cycle=1)

        output = stream.getvalue().strip()
        data = json.loads(output)

        assert data["event"] == "tool_call"
        assert data["tool"] == "search"
        assert data["tool_args"] == {"query": "hello"}
        assert data["cycle"] == 1
        assert data["level"] == "INFO"


class TestLogToolResult:
    def test_log_tool_result(self):
        """Verify JSON has is_error, output_preview fields."""
        logger = _make_fresh_logger("test_tool_result")
        stream = _capture_log_output(logger)

        log_tool_result(logger, tool_name="search", is_error=False, output="found it", cycle=2)

        output = stream.getvalue().strip()
        data = json.loads(output)

        assert data["event"] == "tool_result"
        assert data["tool"] == "search"
        assert data["is_error"] is False
        assert data["output_preview"] == "found it"
        assert data["cycle"] == 2

    def test_log_tool_result_truncates(self):
        """Output longer than 100 chars gets truncated with '...'."""
        logger = _make_fresh_logger("test_tool_result_truncate")
        stream = _capture_log_output(logger)

        long_output = "x" * 150
        log_tool_result(logger, tool_name="search", is_error=False, output=long_output, cycle=3)

        output = stream.getvalue().strip()
        data = json.loads(output)

        assert data["output_preview"] == "x" * 100 + "..."
        assert len(data["output_preview"]) == 103


class TestLogSupervisorDecision:
    def test_log_supervisor_decision(self):
        """Verify JSON has decision field."""
        logger = _make_fresh_logger("test_supervisor")
        stream = _capture_log_output(logger)

        log_supervisor_decision(logger, decision="continue", cycle=4)

        output = stream.getvalue().strip()
        data = json.loads(output)

        assert data["event"] == "supervisor_check"
        assert data["decision"] == "continue"
        assert data["cycle"] == 4


class TestLogError:
    def test_log_error(self):
        """Verify JSON has error level and output_preview."""
        logger = _make_fresh_logger("test_error")
        stream = _capture_log_output(logger)

        log_error(logger, error="something broke", cycle=5)

        output = stream.getvalue().strip()
        data = json.loads(output)

        assert data["event"] == "agent_error"
        assert data["level"] == "ERROR"
        assert data["output_preview"] == "something broke"
        assert data["cycle"] == 5
