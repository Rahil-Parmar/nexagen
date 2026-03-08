"""Structured JSON logging for nexagen agent events.

Includes automatic redaction of sensitive data (API keys, tokens, passwords).
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

# Patterns to redact from log output
_SENSITIVE_PATTERNS = [
    (re.compile(r'(Bearer\s+)\S+', re.IGNORECASE), r'\1***REDACTED***'),
    (re.compile(r'(Authorization\s*[:=]\s*)(\S+(\s+\S+)?)', re.IGNORECASE), r'\1***REDACTED***'),
    (re.compile(r'(api[_-]?key|token|secret|password)\s*[:=]\s*\S+', re.IGNORECASE), r'\1=***REDACTED***'),
    (re.compile(r'(sk-[a-zA-Z0-9]{20,})'), '***API_KEY***'),
    (re.compile(r'(ghp_[a-zA-Z0-9]{36,})'), '***GITHUB_TOKEN***'),
    (re.compile(r'(xoxb-[a-zA-Z0-9-]+)'), '***SLACK_TOKEN***'),
    (re.compile(r'(AIza[a-zA-Z0-9_-]{35})'), '***GOOGLE_KEY***'),
    (re.compile(r'(AKIA[A-Z0-9]{16})'), '***AWS_KEY***'),
]


def _redact(text: str) -> str:
    """Redact sensitive patterns from text."""
    for pattern, replacement in _SENSITIVE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _redact_dict(d: dict) -> dict:
    """Redact sensitive values from a dict (shallow copy)."""
    redacted = {}
    for k, v in d.items():
        key_lower = k.lower()
        if any(s in key_lower for s in ("key", "token", "secret", "password", "auth", "credential")):
            redacted[k] = "***REDACTED***"
        elif isinstance(v, str):
            redacted[k] = _redact(v)
        elif isinstance(v, dict):
            redacted[k] = _redact_dict(v)
        else:
            redacted[k] = v
    return redacted


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": getattr(record, "event", record.getMessage()),
        }
        # Add extra fields if present
        for key in ("tool", "tool_args", "cycle", "decision", "is_error", "output_preview"):
            val = getattr(record, key, None)
            if val is not None:
                if isinstance(val, dict):
                    val = _redact_dict(val)
                elif isinstance(val, str):
                    val = _redact(val)
                log_data[key] = val
        return json.dumps(log_data)


def get_logger(name: str = "nexagen") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def log_tool_call(logger: logging.Logger, tool_name: str, args: dict, cycle: int):
    logger.info(
        "tool_call",
        extra={"event": "tool_call", "tool": tool_name, "tool_args": _redact_dict(args), "cycle": cycle},
    )


def log_tool_result(logger: logging.Logger, tool_name: str, is_error: bool, output: str, cycle: int):
    preview = output[:100] + "..." if len(output) > 100 else output
    logger.info(
        "tool_result",
        extra={
            "event": "tool_result",
            "tool": tool_name,
            "is_error": is_error,
            "output_preview": preview,
            "cycle": cycle,
        },
    )


def log_supervisor_decision(logger: logging.Logger, decision: str, cycle: int):
    logger.info(
        "supervisor_check",
        extra={"event": "supervisor_check", "decision": decision, "cycle": cycle},
    )


def log_error(logger: logging.Logger, error: str, cycle: int):
    logger.error(
        "agent_error",
        extra={"event": "agent_error", "output_preview": error, "cycle": cycle},
    )
