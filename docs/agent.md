# Agent Architecture

The `Agent` class is the core of nexagen. It orchestrates the loop between the LLM and tools, manages permissions, handles context compression, and optionally uses a supervisor to monitor progress.

---

## The Agent Loop

The agent follows a plan-act-observe cycle:

```
                    +---------------------+
                    |   User sends prompt  |
                    +----------+----------+
                               |
                               v
                    +----------+----------+
                    |    Call LLM          |
                    +----------+----------+
                               |
                   +-----------+-----------+
                   |                       |
           Has tool calls?            No tool calls
                   |                       |
                   v                       v
          +--------+--------+     +--------+--------+
          | Check permissions|     | Return final    |
          +--------+--------+     | response        |
                   |              +-----------------+
                   v
          +--------+--------+
          | Execute tools    |
          +--------+--------+
                   |
                   v
          +--------+--------+
          | Yield results    |
          +--------+--------+
                   |
                   v
          +--------+---------+
          | Supervisor check |  (every N calls)
          | if configured    |
          +--------+---------+
                   |
                   v
              Loop back to
              "Call LLM"
```

### Step by step

1. **User prompt** is added to the conversation
2. **LLM is called** with the conversation history and available tool schemas
3. If the LLM responds **without tool calls**, the loop ends and the response is yielded
4. If the LLM responds **with tool calls**:
   a. Each tool call goes through the **permission system**
   b. Permitted tools are **executed** and results are collected
   c. Tool results are added to the conversation
   d. Everything is **yielded** back to the caller
   e. The **supervisor** checks progress (if configured and interval reached)
   f. The loop repeats from step 2

### Streaming results

`agent.run()` is an async generator. You consume results as they are produced:

```python
async for message in agent.run("Do something"):
    if message.role == "assistant":
        # LLM's response (may include tool_calls)
        if message.text:
            print(f"Agent: {message.text}")
        if message.tool_calls:
            for tc in message.tool_calls:
                print(f"Calling: {tc.name}({tc.arguments})")
    elif message.role == "tool":
        # Tool execution result
        print(f"Result: {message.text[:100]}")
```

---

## Agent Constructor

```python
Agent(
    provider: str | ProviderConfig | LLMProvider,
    tools: list[str] | None = None,
    custom_tools: list[BaseTool] | None = None,
    system_prompt: str | None = None,
    permission_mode: str = "safe",
    allowed_tools: list[str] | None = None,
    can_use_tool: Callable | None = None,
    supervisor: LLMProvider | None = None,
    supervisor_check_interval: int = 5,
    max_tool_errors: int = 3,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `str \| ProviderConfig \| LLMProvider` | required | The LLM backend to use |
| `tools` | `list[str] \| None` | `None` | Built-in tool names to enable |
| `custom_tools` | `list[BaseTool] \| None` | `None` | Custom tool instances |
| `system_prompt` | `str \| None` | `"You are a helpful AI assistant."` | System prompt |
| `permission_mode` | `str` | `"safe"` | Permission mode: `readonly`, `safe`, `full` |
| `allowed_tools` | `list[str] \| None` | `None` | Additional allowlist filter |
| `can_use_tool` | `Callable \| None` | `None` | Async permission callback |
| `supervisor` | `LLMProvider \| None` | `None` | Supervisor LLM for progress monitoring |
| `supervisor_check_interval` | `int` | `5` | Check supervisor every N tool calls |
| `max_tool_errors` | `int` | `3` | Max consecutive errors before supervisor escalation |

---

## Supervisor Agent

The supervisor is an optional second LLM that monitors the worker agent's progress. It serves two purposes:

### 1. Progress checking

Every `supervisor_check_interval` tool calls, the supervisor reviews the action log and decides whether the agent is making progress or is stuck:

```python
from nexagen import Agent
from nexagen.providers.openai_compat import OpenAICompatProvider
from nexagen import ProviderConfig

supervisor = OpenAICompatProvider(ProviderConfig(
    backend="ollama",
    model="phi3",
))

agent = Agent(
    provider="ollama/qwen3",
    tools=["file_read", "bash"],
    supervisor=supervisor,
    supervisor_check_interval=5,  # Check every 5 tool calls
)
```

The supervisor responds with `{"decision": "continue"}` or `{"decision": "stop"}`. If it says stop, the agent loop ends gracefully.

### 2. Context compression

When the conversation grows too long (exceeds 80% of the context window), the supervisor compresses older messages into a summary:

```
Before compression:
  [system] [msg1] [msg2] [msg3] [msg4] [msg5] [msg6] [msg7]

After compression:
  [system] [summary of msg1-msg4] [msg5] [msg6] [msg7]
```

This keeps the agent running within its context window for long tasks.

### Error escalation

If a tool fails `max_tool_errors` times consecutively, the supervisor is consulted immediately (regardless of the check interval). This prevents the agent from endlessly retrying a broken tool.

---

## Permission System

Permissions are enforced through three layers, evaluated in order:

```
Layer 1: Mode  -->  Layer 2: Allowlist  -->  Layer 3: Callback
```

The first `Deny` at any layer stops evaluation. If all layers pass, the tool is allowed.

### Layer 1: Permission modes

| Mode | Allowed Tools |
|------|---------------|
| `readonly` | `file_read`, `grep`, `glob` |
| `safe` | `file_read`, `file_write`, `file_edit`, `grep`, `glob` |
| `full` | All tools (no restrictions) |

```python
agent = Agent(
    provider="ollama/qwen3",
    tools=["file_read", "bash"],
    permission_mode="readonly",  # bash will be denied
)
```

### Layer 2: Allowlist

An explicit set of tool names that further narrows access:

```python
agent = Agent(
    provider="ollama/qwen3",
    tools=["file_read", "file_write", "grep"],
    permission_mode="safe",
    allowed_tools=["file_read", "grep"],  # file_write denied
)
```

### Layer 3: Callback

An async function for fine-grained, argument-level control:

```python
from nexagen import Agent, Allow, Deny

async def check_permission(tool_name: str, args: dict) -> Allow | Deny:
    if tool_name == "bash":
        cmd = args.get("command", "")
        if "rm" in cmd or "sudo" in cmd:
            return Deny("Dangerous command blocked")
    return Allow()

agent = Agent(
    provider="ollama/qwen3",
    tools=["file_read", "bash"],
    permission_mode="full",
    can_use_tool=check_permission,
)
```

---

## Conversation Management

The `Conversation` class manages message history across multiple agent runs.

### Basic usage

```python
from nexagen import Agent, Conversation

conv = Conversation(context_window=8192)

agent = Agent(provider="ollama/qwen3")

# First task
async for msg in agent.run("What files are in /tmp?", conversation=conv):
    pass

# Second task -- agent remembers the first task's summary
async for msg in agent.run("Now count those files", conversation=conv):
    pass
```

### Task summaries

When `agent.run()` completes, a summary of the task is stored in the conversation. On the next run, these summaries are included in the system prompt, giving the agent context about previous work.

### Context compression

The conversation estimates token usage (using a chars-per-token heuristic) and triggers compression when it exceeds 80% of the context window:

```
Token estimate = total characters / 4
Compression threshold = context_window * 0.80
```

When compression is needed and a supervisor is configured, older messages are summarized and replaced with a compact summary message.

### Conversation API

| Method | Description |
|--------|-------------|
| `add_message(msg)` | Add a message to the conversation |
| `add_messages(msgs)` | Add multiple messages |
| `estimate_tokens()` | Estimate total token count |
| `needs_compression()` | Check if context is too long |
| `compress(summary)` | Replace old messages with a summary |
| `complete_task(summary)` | Store task summary and reset messages |
| `get_messages_with_history(prompt)` | Get messages including past task summaries |
| `clear()` | Reset all messages and summaries |
