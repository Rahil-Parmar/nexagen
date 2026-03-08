"""Core Agent loop for the nexagen SDK."""

from __future__ import annotations

import logging
from typing import AsyncIterator, Callable, Awaitable

import httpx

from nexagen.models import NexagenMessage, NexagenResponse, ProviderConfig, ToolResult
from nexagen.providers.registry import ProviderRegistry
from nexagen.providers.base import LLMProvider
from nexagen.tools.base import BaseTool
from nexagen.tools.registry import ToolRegistry
from nexagen.tools.builtin import BUILTIN_TOOLS
from nexagen.permissions import PermissionManager, Allow, Deny
from nexagen.conversation import Conversation
from nexagen.supervisor.supervisor import SupervisorAgent, ActionEntry
from nexagen.constants import (
    DEFAULT_PERMISSION_MODE,
    DEFAULT_SUPERVISOR_CHECK_INTERVAL,
    DEFAULT_MAX_TOOL_ERRORS,
    DEFAULT_MAX_ITERATIONS,
)

logger = logging.getLogger("nexagen.agent")


class Agent:
    """Agentic loop that orchestrates LLM calls, tool execution, permissions, and supervision."""

    def __init__(
        self,
        provider: str | ProviderConfig | LLMProvider,
        tools: list[str] | None = None,
        custom_tools: list[BaseTool] | None = None,
        system_prompt: str | None = None,
        permission_mode: str = DEFAULT_PERMISSION_MODE,
        allowed_tools: list[str] | None = None,
        can_use_tool: Callable[[str, dict], Awaitable[Allow | Deny]] | None = None,
        supervisor: LLMProvider | None = None,
        supervisor_check_interval: int = DEFAULT_SUPERVISOR_CHECK_INTERVAL,
        max_tool_errors: int = DEFAULT_MAX_TOOL_ERRORS,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ):
        # Resolve provider
        if isinstance(provider, (str, ProviderConfig)):
            registry = ProviderRegistry()
            # Register all known backends
            from nexagen.providers.openai_compat import OpenAICompatProvider

            for backend in ["ollama", "vllm", "lmstudio", "groq", "together", "openai_compat"]:
                registry.register(backend, OpenAICompatProvider)
            try:
                from nexagen.providers.anthropic_provider import AnthropicProvider

                registry.register("anthropic", AnthropicProvider)
            except ImportError:
                pass
            try:
                from nexagen.providers.openai_native import OpenAINativeProvider

                registry.register("openai", OpenAINativeProvider)
            except ImportError:
                pass
            try:
                from nexagen.providers.google_provider import GoogleProvider

                registry.register("google", GoogleProvider)
            except ImportError:
                pass
            self.provider = registry.resolve(provider)
        else:
            self.provider = provider

        # Setup tool registry
        self.tool_registry = ToolRegistry()
        builtin_names = tools or []
        for name in builtin_names:
            if name in BUILTIN_TOOLS:
                self.tool_registry.register(BUILTIN_TOOLS[name])
        if custom_tools:
            self.tool_registry.register_many(custom_tools)

        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        self.permissions = PermissionManager(
            mode=permission_mode,
            allowed_tools=allowed_tools,
            can_use_tool=can_use_tool,
        )

        # Supervisor
        self.supervisor = SupervisorAgent(supervisor) if supervisor else None
        self.supervisor_check_interval = supervisor_check_interval
        self.max_tool_errors = max_tool_errors
        self.max_iterations = max_iterations

    async def run(
        self, prompt: str, conversation: Conversation | None = None
    ) -> AsyncIterator[NexagenMessage]:
        """Run the agent loop, yielding messages as they are produced.

        The loop continues until the LLM produces a response with no tool calls.
        """
        conv = conversation or Conversation()

        # Build initial messages
        messages = conv.get_messages_with_history(self.system_prompt)

        # Add the summary instruction to system prompt
        system_with_summary = (
            self.system_prompt
            + "\n\nWhen you use tools, always include a brief one-sentence summary "
            "of what you're trying to accomplish in your response text."
        )
        # Replace system message
        if messages and messages[0].role == "system":
            messages[0] = NexagenMessage(role="system", text=system_with_summary)
        else:
            messages.insert(0, NexagenMessage(role="system", text=system_with_summary))

        # Add user message
        user_msg = NexagenMessage(role="user", text=prompt)
        messages.append(user_msg)
        conv.add_message(user_msg)

        tool_schemas = self.tool_registry.get_tool_schemas()
        action_log: list[ActionEntry] = []
        total_tool_calls = 0
        consecutive_errors: dict[str, int] = {}

        iterations = 0

        while True:
            iterations += 1
            if iterations > self.max_iterations:
                yield NexagenMessage(
                    role="assistant",
                    text=f"Stopping: reached maximum iteration limit ({self.max_iterations}).",
                )
                conv.complete_task("Task stopped: max iterations reached")
                return

            # Check context compression (non-fatal — skip if supervisor fails)
            if conv.needs_compression() and self.supervisor:
                try:
                    compressible = conv.get_compressible_messages()
                    if compressible:
                        summary = await self.supervisor.compress_history(compressible)
                        conv.compress(summary)
                        messages = conv.get_messages_with_history(system_with_summary)
                except Exception as e:
                    logger.warning("Context compression failed, continuing without: %s", e)

            # Call LLM
            try:
                response = await self.provider.chat(
                    messages, tool_schemas if tool_schemas else None
                )
            except httpx.HTTPStatusError as e:
                error_text = f"LLM request failed: HTTP {e.response.status_code}"
                logger.error("Provider HTTP error: %s", e)
                yield NexagenMessage(role="assistant", text=error_text)
                conv.complete_task(f"Task failed: {error_text}")
                return
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                error_text = f"LLM connection failed: {type(e).__name__}"
                logger.error("Provider connection error: %s", e)
                yield NexagenMessage(role="assistant", text=error_text)
                conv.complete_task(f"Task failed: {error_text}")
                return
            except Exception as e:
                error_text = f"LLM error: {type(e).__name__}: {e}"
                logger.error("Unexpected provider error: %s", e, exc_info=True)
                yield NexagenMessage(role="assistant", text=error_text)
                conv.complete_task(f"Task failed: {error_text}")
                return

            assistant_msg = response.message
            conv.add_message(assistant_msg)
            messages.append(assistant_msg)

            # Yield the assistant message
            yield assistant_msg

            # No tool calls = done
            if not response.has_tool_calls:
                break

            # Execute tool calls sequentially
            tool_names_this_cycle: list[str] = []
            for tc in assistant_msg.tool_calls:
                tool_names_this_cycle.append(tc.name)
                total_tool_calls += 1

                # Permission check
                try:
                    perm = await self.permissions.check(tc.name, tc.arguments)
                except Exception as e:
                    logger.error("Permission check failed for %s: %s", tc.name, e)
                    perm = Deny(f"Permission check error: {type(e).__name__}")

                if isinstance(perm, Deny):
                    result = ToolResult(
                        tool_call_id=tc.id,
                        output=f"Permission denied: {perm.message}",
                        is_error=True,
                    )
                else:
                    # Execute tool
                    tool_instance = self.tool_registry.get(tc.name)
                    if tool_instance is None:
                        result = ToolResult(
                            tool_call_id=tc.id,
                            output=f"Unknown tool: {tc.name}",
                            is_error=True,
                        )
                    else:
                        try:
                            result = await tool_instance.execute(tc.arguments)
                            result.tool_call_id = tc.id
                        except Exception as e:
                            logger.error("Tool '%s' crashed: %s", tc.name, e, exc_info=True)
                            result = ToolResult(
                                tool_call_id=tc.id,
                                output=f"Tool execution error: {type(e).__name__}: {e}",
                                is_error=True,
                            )

                # Track consecutive errors
                if result.is_error:
                    consecutive_errors[tc.name] = (
                        consecutive_errors.get(tc.name, 0) + 1
                    )
                else:
                    consecutive_errors[tc.name] = 0

                # Error escalation -- repeated failures on same tool
                if (
                    consecutive_errors.get(tc.name, 0) >= self.max_tool_errors
                    and self.supervisor
                ):
                    try:
                        decision = await self.supervisor.check_progress(prompt, action_log)
                    except Exception as e:
                        logger.warning("Supervisor error escalation failed: %s", e)
                        decision = "stop"  # safe default when supervisor is down

                    if decision == "stop":
                        yield NexagenMessage(
                            role="assistant",
                            text=(
                                f"Stopping: tool '{tc.name}' failed "
                                f"{self.max_tool_errors} times consecutively."
                            ),
                        )
                        conv.complete_task(
                            f"Task stopped due to repeated errors with {tc.name}"
                        )
                        return

                # Add tool result to conversation
                tool_msg = result.to_message()
                conv.add_message(tool_msg)
                messages.append(tool_msg)

                # Yield tool result
                yield tool_msg

            # Record action log entry
            summary_text = assistant_msg.summary or assistant_msg.text or "No summary"
            if len(summary_text) > 100:
                summary_text = summary_text[:100] + "..."
            action_log.append(
                ActionEntry(summary=summary_text, tool_names=tool_names_this_cycle)
            )

            # Supervisor progress check every N tool calls
            if (
                self.supervisor
                and total_tool_calls % self.supervisor_check_interval == 0
                and total_tool_calls > 0
            ):
                try:
                    decision = await self.supervisor.check_progress(prompt, action_log)
                except Exception as e:
                    logger.warning("Supervisor progress check failed: %s. Continuing.", e)
                    decision = "continue"  # if supervisor is down, keep going

                if decision == "stop":
                    yield NexagenMessage(
                        role="assistant",
                        text="Stopping: supervisor determined insufficient progress.",
                    )
                    conv.complete_task("Task stopped by supervisor")
                    return

        # Task complete -- generate summary
        final_text = assistant_msg.text or "Task completed"
        if len(final_text) > 150:
            final_text = final_text[:150] + "..."
        conv.complete_task(final_text)
