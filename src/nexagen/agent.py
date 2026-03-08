"""Core Agent loop for the nexagen SDK."""

from __future__ import annotations

import json
import logging
import time
from typing import AsyncIterator, Callable, Awaitable

import httpx

from nexagen.models import NexagenMessage, ProviderConfig
from nexagen.providers.registry import ProviderRegistry
from nexagen.providers.base import LLMProvider
from nexagen.tools.base import BaseTool
from nexagen.tools.registry import ToolRegistry
from nexagen.tools.builtin import BUILTIN_TOOLS
from nexagen.permissions import PermissionManager, Allow, Deny
from nexagen.conversation import Conversation
from nexagen.supervisor.supervisor import SupervisorAgent, SupervisorFeedback, ActionEntry
from nexagen.execution import ParallelExecutor
from nexagen.context import ContextManager
from nexagen.reflection import ReflectionEngine
from nexagen.planning import PlanningPhase
from nexagen.memory import EpisodicMemory, Episode
from nexagen.constants import (
    DEFAULT_PERMISSION_MODE,
    DEFAULT_SUPERVISOR_CHECK_INTERVAL,
    DEFAULT_MAX_TOOL_ERRORS,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_REFLECTIONS,
    DEFAULT_MAX_EPISODES,
    DEFAULT_RECENT_TOOL_RESULTS_TO_KEEP,
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
        max_reflections: int = DEFAULT_MAX_REFLECTIONS,
        max_episodes: int = DEFAULT_MAX_EPISODES,
        recent_tool_results_to_keep: int = DEFAULT_RECENT_TOOL_RESULTS_TO_KEEP,
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

        # Frozen system prompt (built once at init time)
        base_prompt = system_prompt or "You are a helpful AI assistant."
        self._frozen_system_prompt = (
            base_prompt
            + "\n\nWhen you use tools, always include a brief one-sentence summary "
            "of what you're trying to accomplish in your response text."
        )

        # New orchestration modules
        self.executor = ParallelExecutor()
        self.context_manager = ContextManager(
            recent_tool_results_to_keep=recent_tool_results_to_keep,
        )
        self.reflection_engine = (
            ReflectionEngine(provider=supervisor, max_reflections=max_reflections)
            if supervisor
            else None
        )
        self.planning = PlanningPhase(provider=supervisor) if supervisor else None
        self.memory = EpisodicMemory(max_episodes=max_episodes)

    async def run(
        self, prompt: str, conversation: Conversation | None = None
    ) -> AsyncIterator[NexagenMessage]:
        """Run the agent loop, yielding messages as they are produced.

        The loop continues until the LLM produces a response with no tool calls.
        """
        conv = conversation or Conversation()

        # EPISODIC MEMORY: retrieve relevant past experiences
        system_prompt = self._frozen_system_prompt
        relevant = self.memory.retrieve(prompt, k=3)
        if relevant:
            ctx = self.memory.format_for_context(relevant)
            if ctx:
                system_prompt = system_prompt + "\n\n## Relevant Past Experience\n" + ctx

        # PLANNING: auto-detect complexity, generate plan if complex
        plan = None
        if self.planning:
            complexity = await self.planning.classify_complexity(prompt)
            if complexity == "complex":
                plan = await self.planning.generate_plan(prompt)
                system_prompt = system_prompt + "\n\n## Execution Plan\n" + self.planning.format_plan_context(plan)

        # BUILD CONTEXT (KV-cache-aware)
        messages = conv.get_messages_for_llm(
            system_prompt=system_prompt,
            context_manager=self.context_manager,
            context_window=conv.context_window,
        )

        # Add user message
        user_msg = NexagenMessage(role="user", text=prompt)
        messages.append(user_msg)
        conv.add_message(user_msg)

        tool_schemas = self.tool_registry.get_tool_schemas()
        action_log: list[ActionEntry] = []
        total_tool_calls = 0
        consecutive_errors: dict[str, int] = {}
        reflection_counts: dict[str, int] = {}
        all_reflections: list = []  # list of ReflectionResult objects
        errors_encountered: list[str] = []
        tools_used_set: set[str] = set()

        iterations = 0

        while True:
            iterations += 1
            if iterations > self.max_iterations:
                yield NexagenMessage(
                    role="assistant",
                    text=f"Stopping: reached maximum iteration limit ({self.max_iterations}).",
                )
                self._record_episode(prompt, "partial", tools_used_set, errors_encountered, all_reflections)
                conv.complete_task("Task stopped: max iterations reached")
                return

            # CONTEXT SHAPING (rebuild from conversation each iteration)
            messages = conv.get_messages_for_llm(
                system_prompt=system_prompt,
                context_manager=self.context_manager,
                context_window=conv.context_window,
            )

            # Call LLM
            try:
                response = await self.provider.chat(
                    messages, tool_schemas if tool_schemas else None
                )
            except httpx.HTTPStatusError as e:
                error_text = f"LLM request failed: HTTP {e.response.status_code}"
                logger.error("Provider HTTP error: %s", e)
                yield NexagenMessage(role="assistant", text=error_text)
                self._record_episode(prompt, "failure", tools_used_set, errors_encountered, all_reflections)
                conv.complete_task(f"Task failed: {error_text}")
                return
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                error_text = f"LLM connection failed: {type(e).__name__}"
                logger.error("Provider connection error: %s", e)
                yield NexagenMessage(role="assistant", text=error_text)
                self._record_episode(prompt, "failure", tools_used_set, errors_encountered, all_reflections)
                conv.complete_task(f"Task failed: {error_text}")
                return
            except Exception as e:
                error_text = f"LLM error: {type(e).__name__}: {e}"
                logger.error("Unexpected provider error: %s", e, exc_info=True)
                yield NexagenMessage(role="assistant", text=error_text)
                self._record_episode(prompt, "failure", tools_used_set, errors_encountered, all_reflections)
                conv.complete_task(f"Task failed: {error_text}")
                return

            assistant_msg = response.message
            conv.add_message(assistant_msg)

            # Yield the assistant message
            yield assistant_msg

            # No tool calls = done
            if not response.has_tool_calls:
                break

            # PARALLEL TOOL EXECUTION
            results = await self.executor.execute_batch(
                assistant_msg.tool_calls, self.tool_registry, self.permissions
            )

            tool_names_this_cycle: list[str] = []
            for tc, result in zip(assistant_msg.tool_calls, results):
                tool_names_this_cycle.append(tc.name)
                tools_used_set.add(tc.name)
                total_tool_calls += 1

                # Track consecutive errors
                if result.is_error:
                    consecutive_errors[tc.name] = (
                        consecutive_errors.get(tc.name, 0) + 1
                    )
                    errors_encountered.append(f"{tc.name}: {result.output[:100]}")
                else:
                    consecutive_errors[tc.name] = 0

                # SELF-REFLECTION on repeated errors
                if (
                    consecutive_errors.get(tc.name, 0) >= self.max_tool_errors
                    and self.reflection_engine
                ):
                    tool_refl_count = reflection_counts.get(tc.name, 0)
                    if tool_refl_count < self.reflection_engine.max_reflections:
                        safe_args = json.dumps(tc.arguments, sort_keys=True)[:200]
                        reflection = await self.reflection_engine.reflect(
                            original_task=prompt,
                            failed_action=f"{tc.name}({safe_args})",
                            error=result.output[:500],
                            past_reflections=all_reflections,
                        )
                        reflection_counts[tc.name] = tool_refl_count + 1
                        all_reflections.append(reflection)

                        if reflection.should_retry:
                            reflection_msg = NexagenMessage(
                                role="assistant",
                                text=f"[Reflection] {reflection.diagnosis} New strategy: {reflection.strategy}",
                            )
                            conv.add_message(reflection_msg)
                            yield reflection_msg
                            consecutive_errors[tc.name] = 0  # reset for retry
                        else:
                            if self.supervisor:
                                feedback = await self.supervisor.check_progress(prompt, action_log)
                                if feedback.decision == "stop":
                                    yield NexagenMessage(
                                        role="assistant",
                                        text=(
                                            f"Stopping: tool '{tc.name}' failed "
                                            f"{self.max_tool_errors} times. Diagnosis: {reflection.diagnosis}"
                                        ),
                                    )
                                    self._record_episode(prompt, "failure", tools_used_set, errors_encountered, all_reflections)
                                    conv.complete_task(f"Task stopped: {reflection.diagnosis}")
                                    return
                    else:
                        if self.supervisor:
                            feedback = await self.supervisor.check_progress(prompt, action_log)
                            if feedback.decision == "stop":
                                yield NexagenMessage(
                                    role="assistant",
                                    text=f"Stopping: tool '{tc.name}' failed repeatedly, reflections exhausted.",
                                )
                                self._record_episode(prompt, "failure", tools_used_set, errors_encountered, all_reflections)
                                conv.complete_task(f"Task stopped due to repeated errors with {tc.name}")
                                return

                # Error escalation -- repeated failures on same tool (no reflection engine)
                elif (
                    consecutive_errors.get(tc.name, 0) >= self.max_tool_errors
                    and not self.reflection_engine
                    and self.supervisor
                ):
                    try:
                        feedback = await self.supervisor.check_progress(prompt, action_log)
                    except Exception as e:
                        logger.warning("Supervisor error escalation failed: %s", e)
                        feedback = SupervisorFeedback(decision="stop")

                    if feedback.decision == "stop":
                        yield NexagenMessage(
                            role="assistant",
                            text=(
                                f"Stopping: tool '{tc.name}' failed "
                                f"{self.max_tool_errors} times consecutively."
                            ),
                        )
                        self._record_episode(prompt, "failure", tools_used_set, errors_encountered, all_reflections)
                        conv.complete_task(
                            f"Task stopped due to repeated errors with {tc.name}"
                        )
                        return

                # Add tool result to conversation
                tool_msg = result.to_message()
                conv.add_message(tool_msg)

                # Yield tool result
                yield tool_msg

            # Record action log entry
            summary_text = assistant_msg.summary or assistant_msg.text or "No summary"
            if len(summary_text) > 100:
                summary_text = summary_text[:100] + "..."
            action_log.append(
                ActionEntry(summary=summary_text, tool_names=tool_names_this_cycle)
            )

            # PLAN PROGRESS
            if plan and not plan.is_complete:
                plan.advance()

            # Supervisor progress check every N tool calls
            if (
                self.supervisor
                and total_tool_calls % self.supervisor_check_interval == 0
                and total_tool_calls > 0
            ):
                try:
                    feedback = await self.supervisor.check_progress(prompt, action_log)
                except Exception as e:
                    logger.warning("Supervisor progress check failed: %s. Continuing.", e)
                    feedback = SupervisorFeedback(decision="continue")

                if feedback.decision == "stop":
                    yield NexagenMessage(
                        role="assistant",
                        text="Stopping: supervisor determined insufficient progress.",
                    )
                    self._record_episode(prompt, "partial", tools_used_set, errors_encountered, all_reflections)
                    conv.complete_task("Task stopped by supervisor")
                    return
                elif feedback.decision == "redirect" and feedback.suggestion:
                    safe_suggestion = feedback.suggestion[:200]
                    hint_msg = NexagenMessage(
                        role="assistant",
                        text=f"[Supervisor hint: {safe_suggestion}]",
                    )
                    conv.add_message(hint_msg)
                    yield hint_msg

        # Task complete -- generate summary
        final_text = assistant_msg.text or "Task completed"
        if len(final_text) > 150:
            final_text = final_text[:150] + "..."
        conv.complete_task(final_text)
        self._record_episode(prompt, "success", tools_used_set, errors_encountered, all_reflections)

    def _record_episode(
        self,
        task: str,
        outcome: str,
        tools_used: set[str],
        errors: list[str],
        reflections: list,
    ) -> None:
        """Record an episode in episodic memory."""
        # Extract diagnosis strings from ReflectionResult objects
        refl_strings = []
        for r in reflections[:5]:
            if hasattr(r, "diagnosis"):
                refl_strings.append(r.diagnosis)
            else:
                refl_strings.append(str(r))

        self.memory.record(Episode(
            task=task,
            outcome=outcome,
            tools_used=list(tools_used),
            errors_encountered=errors[:5],
            reflections=refl_strings,
            timestamp=time.time(),
        ))
