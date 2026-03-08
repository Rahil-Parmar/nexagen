"""BaseTool class and @tool decorator for the nexagen tool system."""

from __future__ import annotations

import traceback
from typing import Any, Callable, Awaitable

from pydantic import BaseModel, ValidationError

from nexagen.models import ToolResult


class BaseTool:
    """A tool that validates input via a Pydantic model and delegates to an async handler."""

    def __init__(
        self,
        name: str,
        description: str,
        input_model: type[BaseModel],
        handler: Callable[[Any], Awaitable[str]],
    ):
        self.name = name
        self.description = description
        self._input_model = input_model
        self._handler = handler

    @property
    def input_schema(self) -> dict:
        """Return the JSON Schema for the input model."""
        return self._input_model.model_json_schema()

    async def connect(self) -> None:
        """Lifecycle hook — called when the tool is attached. No-op by default."""
        pass

    async def disconnect(self) -> None:
        """Lifecycle hook — called when the tool is detached. No-op by default."""
        pass

    def is_available(self) -> bool:
        """Return True if this tool is ready to execute."""
        return True

    async def execute(self, args: dict) -> ToolResult:
        """Validate *args* against the input model and run the handler.

        Returns a ``ToolResult`` — never raises.
        """
        try:
            validated = self._input_model.model_validate(args)
        except ValidationError as e:
            return ToolResult(
                tool_call_id="",
                output=f"ValidationError: {e.errors()[0]['msg']}\n  in {self.name}",
                is_error=True,
            )

        try:
            result = await self._handler(validated)
            return ToolResult(tool_call_id="", output=str(result), is_error=False)
        except Exception as e:
            # Return error type + message for the LLM, but strip internal paths
            error_type = type(e).__name__
            error_msg_str = str(e)
            # Strip absolute paths from error messages to prevent info leakage
            import re
            # Match paths like /Users/foo/bar, /home/user/file.py, /tmp/x
            error_msg_str = re.sub(r'(/[a-zA-Z0-9._-]+){2,}', '<path-redacted>', error_msg_str)
            # Strip potential API keys or tokens (long hex/base64 strings)
            error_msg_str = re.sub(r'[a-zA-Z0-9+/=_-]{32,}', '<redacted>', error_msg_str)
            error_msg = f"{error_type}: {error_msg_str}"
            # Log the full traceback internally (not returned to LLM)
            import logging
            logging.getLogger("nexagen.tools").debug(
                f"Tool '{self.name}' error: {error_type}: {e}",
                exc_info=True,
            )
            return ToolResult(tool_call_id="", output=error_msg, is_error=True)

    def to_tool_schema(self) -> dict:
        """Return a provider-agnostic tool schema dict."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
        }


def tool(
    name: str, description: str, input_model: type[BaseModel]
) -> Callable[[Callable[[Any], Awaitable[str]]], BaseTool]:
    """Decorator that turns an async function into a ``BaseTool``.

    Usage::

        @tool(name="greet", description="Greets a user", input_model=GreetInput)
        async def greet(inp: GreetInput) -> str:
            return f"Hello, {inp.name}!"
    """

    def decorator(func: Callable[[Any], Awaitable[str]]) -> BaseTool:
        return BaseTool(
            name=name,
            description=description,
            input_model=input_model,
            handler=func,
        )

    return decorator
