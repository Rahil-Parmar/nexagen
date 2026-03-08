# Agent Orchestration Improvements — Design Document

**Date:** 2026-03-07
**Status:** Approved
**Approach:** Layered Enhancement (composable modules added to existing Agent)

---

## Summary

Seven improvements to nexagen's agent lifecycle, bringing it in line with frontier agent orchestration patterns (Claude Code, Manus, Codex, SWE-Agent). All changes are backward-compatible — existing users are not broken.

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Planning phase trigger | Auto-detect complexity | Smart routing: simple tasks stay fast, complex tasks get structured |
| Episodic memory storage | In-memory only | Zero deps; resets per process; upgradeable to persistence later |
| Max reflections | Configurable, default 2 | Covers common cascading fixes; users can tune per use case |
| Context management | Two-tier (mask first, LLM compress second) | Masking handles 90% of cases free; LLM handles the rest |

---

## Improvement 1: Parallel Tool Execution

**New module:** `src/nexagen/execution.py`

### Design

```python
class ParallelExecutor:
    """Executes tool calls concurrently using asyncio.gather."""

    async def execute_batch(
        self,
        tool_calls: list[ToolCall],
        tool_registry: ToolRegistry,
        permissions: PermissionManager,
    ) -> list[ToolResult]:
        """Run all tool calls in parallel, return results in same order.

        Each tool call gets its own asyncio.Task.
        Permission checks happen inside each task (before execution).
        One failing tool does not cancel the others.
        """
```

### Integration

Replaces the sequential `for tc in assistant_msg.tool_calls` loop in `agent.py` with:

```python
results = await self.executor.execute_batch(
    assistant_msg.tool_calls, self.tool_registry, self.permissions
)
```

### Edge Cases

- All tool calls in a single LLM response are treated as independent (matches how Claude/GPT/Gemini emit parallel tool calls).
- If one tool call depends on another's output, the LLM should emit them in separate response cycles.

---

## Improvement 2: Two-Tier Context Management (Observation Masking + LLM Compression)

**New module:** `src/nexagen/context.py`

### Design

```python
class ContextManager:
    """Two-tier context management: observation masking first, LLM compression second."""

    def __init__(self, recent_tool_results_to_keep: int = 3):
        self.recent_tool_results_to_keep = recent_tool_results_to_keep

    def mask_observations(self, messages: list[NexagenMessage]) -> list[NexagenMessage]:
        """Tier 1: Strip tool result content from old messages, keep reasoning.

        Rules:
        - Keep all assistant messages (reasoning + tool call decisions) intact
        - Keep the last N tool results verbatim (default 3)
        - Replace older tool results with a stub: "[tool_name result: success/error]"
        - Never touch system or user messages
        """

    async def compress_if_needed(
        self,
        messages: list[NexagenMessage],
        conversation: Conversation,
        supervisor: SupervisorAgent | None,
    ) -> list[NexagenMessage]:
        """Tier 2: If still over threshold after masking, use LLM compression."""
```

### Two-Tier Chain (in agent loop, before each LLM call)

1. Estimate tokens
2. If over 80% threshold -> `mask_observations()` (free, instant)
3. Re-estimate tokens
4. If STILL over 80% -> `compress_if_needed()` (LLM call, expensive)
5. Proceed with `chat()`

### Research Basis

JetBrains NeurIPS 2025: observation masking halves cost and matches or beats LLM summarization for software engineering agents.

---

## Improvement 3: Self-Reflection Engine

**New module:** `src/nexagen/reflection.py`

### Design

```python
@dataclass
class ReflectionResult:
    diagnosis: str       # what went wrong
    strategy: str        # what to try differently
    should_retry: bool   # whether retrying is worthwhile

class ReflectionEngine:
    """Diagnoses failures and generates corrective strategies."""

    def __init__(self, provider: LLMProvider, max_reflections: int = 2):
        self.provider = provider
        self.max_reflections = max_reflections

    async def reflect(
        self,
        original_task: str,
        failed_action: str,
        error: str,
        past_reflections: list[str],
    ) -> ReflectionResult:
        """Structured diagnosis + suggested next action."""
```

### Integration

```
Tool fails ->
  if reflection_count < max_reflections:
      reflection = await self.reflection_engine.reflect(...)
      if reflection.should_retry:
          inject reflection as assistant message
          continue loop (LLM sees diagnosis and adjusts)
      else:
          proceed to supervisor escalation
  else:
      fall through to existing error counting / supervisor stop
```

### Research Basis

Reflexion pattern (Shinn et al.): converts environmental feedback into linguistic self-reflection, stored in context for the LLM to learn from within the conversation.

---

## Improvement 4: Structured Planning Phase

**New module:** `src/nexagen/planning.py`

### Design

```python
@dataclass
class Subtask:
    description: str
    status: str = "pending"  # "pending" | "in_progress" | "completed" | "failed"

@dataclass
class Plan:
    subtasks: list[Subtask]
    current_step: int = 0

    def update_progress(self, action_log: list[ActionEntry]):
        """Mark subtasks as completed based on action log."""

class PlanningPhase:
    """Auto-detects task complexity and generates execution plans."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    async def classify_complexity(self, prompt: str) -> str:
        """Returns 'simple' or 'complex'.
        Short LLM call with constrained JSON output. ~200 tokens overhead."""

    async def generate_plan(self, prompt: str) -> Plan:
        """Decomposes a complex task into ordered subtasks."""

    def format_plan_context(self, plan: Plan) -> str:
        """Formats current plan state for injection into system prompt.
        Shows completed steps with checkmarks and current step highlighted."""
```

### Integration (top of `run()`)

```python
complexity = await self.planning.classify_complexity(prompt)
plan = None
if complexity == "complex":
    plan = await self.planning.generate_plan(prompt)
    # Inject plan into context

# During loop: after each tool cycle, update plan progress
if plan:
    plan.update_progress(action_log)
```

### Complexity Classification

Single-sentence prompt: "Is this task simple (single step, direct answer) or complex (multiple steps, research, file changes)?" with JSON output `{"complexity": "simple"|"complex"}`.

---

## Improvement 5: Episodic Memory

**New module:** `src/nexagen/memory.py`

### Design

```python
@dataclass
class Episode:
    task: str
    outcome: str              # "success" | "failure" | "partial"
    tools_used: list[str]
    errors_encountered: list[str]
    reflections: list[str]
    timestamp: float

class EpisodicMemory:
    """In-memory store of past task episodes with scored retrieval."""

    def __init__(self, max_episodes: int = 50):
        self._episodes: list[Episode] = []
        self.max_episodes = max_episodes

    def record(self, episode: Episode):
        """Store an episode. Evicts oldest if over capacity."""

    def retrieve(self, query: str, k: int = 3) -> list[Episode]:
        """Return k most relevant episodes.
        Scoring: recency (0.3) x keyword_relevance (0.7)
        Uses word overlap -- no vector DB needed."""

    def format_for_context(self, episodes: list[Episode]) -> str:
        """Format episodes for injection into system prompt."""
```

### Integration

```python
# At start of run():
relevant = self.memory.retrieve(prompt, k=3)
if relevant:
    system_prompt += "\n\n## Relevant Past Experience\n" + self.memory.format_for_context(relevant)

# At end of run():
self.memory.record(Episode(...))
```

### Storage

In-memory only. Lives on the Agent instance. Persists across `run()` calls within the same process, resets when process ends. Max 50 episodes with LRU eviction.

---

## Improvement 6: Rich Supervisor Feedback

**Enhanced module:** `src/nexagen/supervisor/supervisor.py`

### Design

```python
@dataclass
class SupervisorFeedback:
    decision: str           # "continue" | "stop" | "redirect"
    diagnosis: str | None   # what's going wrong
    suggestion: str | None  # what to try instead

class SupervisorAgent:
    async def check_progress(self, original_task, action_log) -> SupervisorFeedback:
        """Returns structured feedback instead of binary string.

        Prompt asks for JSON:
        {
            "decision": "continue|stop|redirect",
            "diagnosis": "...",
            "suggestion": "..."
        }
        """
```

### New "redirect" Decision

When supervisor returns "redirect", the suggestion is injected as a hint:

```python
if feedback.decision == "redirect":
    hint = NexagenMessage(role="assistant", text=f"[Supervisor hint: {feedback.suggestion}]")
    messages.append(hint)
    # Continue the loop
```

### Backward Compatibility

If JSON parsing fails, falls back to current binary text-search behavior ("continue" / "stop").

---

## Improvement 7: KV-Cache-Aware Context Design

**Changes across:** `conversation.py`, `context.py`, `agent.py`

### 7a. Stable System Prompt Prefix

System prompt is frozen at `__init__` time. Dynamic content (plan, memory, supervisor hints) appended as separate messages, never modifying the prefix.

```python
self._frozen_system_prompt = system_prompt + "\n\nWhen you use tools..."
# Dynamic content injected as early assistant/system messages
```

### 7b. Append-Only Conversation

`Conversation.messages` is append-only. Context shaping happens at read time, not write time:

```python
class Conversation:
    def get_messages_for_llm(self, context_manager: ContextManager) -> list[NexagenMessage]:
        """Returns a shaped view of messages. Never mutates the source list."""
```

### 7c. Deterministic Serialization

Stable JSON key ordering in tool schemas and tool results:

```python
json.dumps(args, sort_keys=True)
```

### Research Basis

Manus blog: cached input tokens cost 10x less than uncached ($0.30 vs $3.00/MTok on Claude Sonnet). Append-only + stable prefix maximizes KV-cache hit rate.

---

## Enhanced Agent Loop (All 7 Together)

```
User prompt arrives
    |
    v
[Memory Retrieval] -- retrieve relevant episodes (#5)
    |
    v
[Complexity Classification] -- "simple" or "complex"? (#4)
    |
    +-- complex -> [Generate Plan] -> inject into context
    |
    v
[Build Context] -- frozen system prefix + memory + plan (#7)
    |
    v
+-> [LLM Call] -- with append-only context (#7)
|       |
|       v
|   [Response] -- no tool calls? -> done -> record episode (#5)
|       |
|       v (has tool calls)
|   [Parallel Execute] -- asyncio.gather all tools (#1)
|       |
|       v
|   [Check for Errors]
|       |
|       +-- errors -> [Reflect] -- diagnose + suggest (#3)
|       |       |
|       |       +-- should_retry -> inject reflection -> continue
|       |       +-- give up -> supervisor escalation
|       |
|       v
|   [Context Shaping] -- mask old observations (#2, tier 1)
|       |               +-- still too big? -> LLM compress (tier 2)
|       |
|       v
|   [Supervisor Check] -- every N calls (#6)
|       |
|       +-- "continue" -> loop
|       +-- "redirect" -> inject hint -> loop
|       +-- "stop" -> end
|       |
+-------+
```

## New Files

| File | Purpose |
|------|---------|
| `src/nexagen/execution.py` | ParallelExecutor |
| `src/nexagen/context.py` | ContextManager (observation masking + LLM compression) |
| `src/nexagen/reflection.py` | ReflectionEngine |
| `src/nexagen/planning.py` | PlanningPhase (auto-detect + plan generation) |
| `src/nexagen/memory.py` | EpisodicMemory |

## Modified Files

| File | Changes |
|------|---------|
| `src/nexagen/agent.py` | Wire all new modules; parallel execution; new run() flow |
| `src/nexagen/supervisor/supervisor.py` | SupervisorFeedback dataclass; structured JSON parsing; "redirect" decision |
| `src/nexagen/conversation.py` | Append-only; `get_messages_for_llm()` method; remove mutation in `compress()` |
| `src/nexagen/constants.py` | New defaults: `DEFAULT_MAX_REFLECTIONS`, `DEFAULT_MAX_EPISODES`, `DEFAULT_RECENT_TOOL_RESULTS` |
| `src/nexagen/models.py` | No changes expected |

## Test Files

| File | Covers |
|------|--------|
| `tests/test_execution.py` | Parallel execution, error isolation, permission checks |
| `tests/test_context.py` | Observation masking, two-tier compression, token estimation |
| `tests/test_reflection.py` | Reflection prompts, retry logic, max reflection cap |
| `tests/test_planning.py` | Complexity classification, plan generation, progress tracking |
| `tests/test_memory.py` | Episode storage, retrieval scoring, eviction, formatting |
| `tests/test_supervisor.py` | Updated for SupervisorFeedback, redirect logic, backward compat |
| `tests/test_agent.py` | Updated for new run() flow with all improvements integrated |
