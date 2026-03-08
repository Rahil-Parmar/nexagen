# API Reference

Complete reference for all public classes and functions in nexagen.

---

## Agent

```python
from nexagen import Agent
```

### Constructor

```python
Agent(
    provider: str | ProviderConfig | LLMProvider,
    tools: list[str] | None = None,
    custom_tools: list[BaseTool] | None = None,
    system_prompt: str | None = None,
    permission_mode: str = "safe",
    allowed_tools: list[str] | None = None,
    can_use_tool: Callable[[str, dict], Awaitable[Allow | Deny]] | None = None,
    supervisor: LLMProvider | None = None,
    supervisor_check_interval: int = 5,
    max_tool_errors: int = 3,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `str \| ProviderConfig \| LLMProvider` | required | LLM backend |
| `tools` | `list[str] \| None` | `None` | Built-in tool names |
| `custom_tools` | `list[BaseTool] \| None` | `None` | Custom tools |
| `system_prompt` | `str \| None` | `"You are a helpful AI assistant."` | System prompt |
| `permission_mode` | `str` | `"safe"` | `readonly` / `safe` / `full` |
| `allowed_tools` | `list[str] \| None` | `None` | Allowlist filter |
| `can_use_tool` | `Callable \| None` | `None` | Permission callback |
| `supervisor` | `LLMProvider \| None` | `None` | Supervisor LLM |
| `supervisor_check_interval` | `int` | `5` | Supervisor check frequency |
| `max_tool_errors` | `int` | `3` | Max consecutive errors before escalation |

### Methods

#### `run(prompt, conversation=None) -> AsyncIterator[NexagenMessage]`

Run the agent loop. Yields messages as they are produced.

```python
async for message in agent.run("Hello"):
    print(message.role, message.text)
```

**Parameters:**
- `prompt` (str) -- The user's input
- `conversation` (Conversation | None) -- Optional conversation for history

**Yields:** `NexagenMessage` -- assistant responses and tool results

---

## NexagenMessage

```python
from nexagen import NexagenMessage
```

A unified message type used across all providers.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `role` | `str` | required | `"system"`, `"user"`, `"assistant"`, or `"tool"` |
| `text` | `str \| None` | `None` | Message content |
| `tool_calls` | `list[ToolCall] \| None` | `None` | Tool calls (assistant only) |
| `summary` | `str \| None` | `None` | Summary text for compressed messages |
| `tool_call_id` | `str \| None` | `None` | ID linking tool results to calls |
| `is_error` | `bool` | `False` | Whether this is an error result |

```python
msg = NexagenMessage(role="user", text="Hello")
msg = NexagenMessage(role="assistant", text="Hi!", tool_calls=[...])
msg = NexagenMessage(role="tool", text="result", tool_call_id="abc", is_error=False)
```

---

## NexagenResponse

```python
from nexagen import NexagenResponse
```

Wraps a `NexagenMessage` returned by a provider.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `message` | `NexagenMessage` | The response message |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `has_tool_calls` | `bool` | `True` if the message contains tool calls |

---

## ToolCall

```python
from nexagen import ToolCall
```

Represents a tool call made by the assistant.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique identifier for this call |
| `name` | `str` | Tool name |
| `arguments` | `dict` | Arguments to pass to the tool |

---

## ToolResult

```python
from nexagen import ToolResult
```

Result of executing a tool call.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tool_call_id` | `str` | required | ID of the original tool call |
| `output` | `str` | required | Result text |
| `is_error` | `bool` | `False` | Whether execution failed |

### Methods

#### `to_message() -> NexagenMessage`

Convert this result into a `NexagenMessage` with `role="tool"`.

---

## ProviderConfig

```python
from nexagen import ProviderConfig
```

Configuration for connecting to an LLM provider.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backend` | `str` | required | Backend name |
| `model` | `str` | required | Model name |
| `base_url` | `str \| None` | `None` | Custom API URL |
| `api_key` | `str \| None` | `None` | API key |
| `temperature` | `float \| None` | `None` | Sampling temperature |
| `max_tokens` | `int \| None` | `None` | Max output tokens |

### Class Methods

#### `from_string(provider_string) -> ProviderConfig`

Parse a string like `"ollama/qwen3@192.168.1.5:11434"`.

```python
config = ProviderConfig.from_string("ollama/qwen3")
config = ProviderConfig.from_string("anthropic/claude-sonnet-4-20250514")
config = ProviderConfig.from_string("ollama/llama3@10.0.0.1:11434")
```

---

## BaseTool

```python
from nexagen import BaseTool
```

A tool that validates input via a Pydantic model and delegates to an async handler.

### Constructor

```python
BaseTool(
    name: str,
    description: str,
    input_model: type[BaseModel],
    handler: Callable[[Any], Awaitable[str]],
)
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `input_schema` | `dict` | JSON Schema for the input model |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `connect()` | `None` | Lifecycle hook: tool attached |
| `disconnect()` | `None` | Lifecycle hook: tool detached |
| `is_available()` | `bool` | Whether the tool is ready |
| `execute(args)` | `ToolResult` | Validate args and run handler |
| `to_tool_schema()` | `dict` | Provider-agnostic schema |

---

## @tool Decorator

```python
from nexagen import tool
```

Turns an async function into a `BaseTool`.

```python
@tool(name="my_tool", description="Does something", input_model=MyInput)
async def my_tool(args: MyInput) -> str:
    return "result"
```

**Parameters:**
- `name` (str) -- Tool name
- `description` (str) -- Description sent to the LLM
- `input_model` (type[BaseModel]) -- Pydantic model for input validation

**Returns:** `BaseTool` instance

---

## ToolRegistry

```python
from nexagen import ToolRegistry
```

Manages tool instances.

### Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `register(tool)` | `BaseTool` | `None` | Register a single tool |
| `register_many(tools)` | `list[BaseTool]` | `None` | Register multiple tools |
| `get(name)` | `str` | `BaseTool \| None` | Get tool by name |
| `list_available()` | -- | `list[BaseTool]` | All available tools |
| `get_tool_schemas()` | -- | `list[dict]` | Schemas for all available tools |
| `get_tool_names()` | -- | `list[str]` | Names of all available tools |

---

## MCPServerConfig

```python
from nexagen import MCPServerConfig
```

Configuration for an MCP server.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `command` | `str` | required | Command to start the server |
| `args` | `list[str]` | `[]` | Command arguments |
| `env` | `dict[str, str]` | `{}` | Environment variables |

---

## MCPTool

```python
from nexagen import MCPTool
```

A tool backed by an MCP server.

### Constructor

```python
MCPTool(
    name: str,
    description: str,
    input_schema: dict,
    server_config: MCPServerConfig,
)
```

### Methods

Same as `BaseTool`: `connect()`, `disconnect()`, `is_available()`, `execute(args)`, `to_tool_schema()`.

---

## MCPManager

```python
from nexagen import MCPManager
```

Manages multiple MCP server connections.

### Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `add_server(name, config)` | `str, MCPServerConfig` | `None` | Add a server |
| `add_server_from_dict(name, dict)` | `str, dict` | `None` | Add from dict |
| `register_tool(server_name, tool)` | `str, MCPTool` | `None` | Register a tool |
| `connect_all()` | -- | `None` | Connect all servers |
| `disconnect_all()` | -- | `None` | Disconnect all servers |
| `get_tools()` | -- | `list[MCPTool]` | All tools |
| `get_available_tools()` | -- | `list[MCPTool]` | Available tools |

---

## Conversation

```python
from nexagen import Conversation
```

### Constructor

```python
Conversation(context_window: int = 8192)
```

### Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `add_message(msg)` | `NexagenMessage` | `None` | Add a message |
| `add_messages(msgs)` | `list[NexagenMessage]` | `None` | Add multiple |
| `estimate_tokens()` | -- | `int` | Estimate token count |
| `needs_compression()` | -- | `bool` | Check threshold |
| `get_compressible_messages()` | -- | `list[NexagenMessage]` | Messages to compress |
| `compress(summary)` | `str` | `None` | Replace old with summary |
| `complete_task(summary)` | `str` | `None` | Store summary, reset |
| `get_messages_with_history(prompt)` | `str \| None` | `list[NexagenMessage]` | Full history |
| `clear()` | -- | `None` | Reset everything |

---

## PermissionManager

```python
from nexagen import PermissionManager
```

### Constructor

```python
PermissionManager(
    mode: str = "safe",
    allowed_tools: list[str] | None = None,
    can_use_tool: Callable[[str, dict], Awaitable[Allow | Deny]] | None = None,
)
```

### Methods

#### `check(tool_name, args) -> Allow | Deny`

Check whether a tool may be invoked. Evaluates mode, allowlist, and callback in order.

---

## Allow / Deny

```python
from nexagen import Allow, Deny
```

### Allow

Dataclass with no fields. Signals permission granted.

### Deny

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `message` | `str` | `"Permission denied"` | Reason for denial |

---

## SupervisorAgent

```python
from nexagen import SupervisorAgent
```

### Constructor

```python
SupervisorAgent(provider: LLMProvider)
```

### Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `check_progress(task, log)` | `str, list[ActionEntry]` | `str` | `"continue"` or `"stop"` |
| `compress_history(messages)` | `list[NexagenMessage]` | `str` | Summary string |

---

## ActionEntry

```python
from nexagen import ActionEntry
```

One step in the worker's action log, used by the supervisor.

### Constructor

```python
ActionEntry(summary: str, tool_names: list[str])
```

---

## LLMProvider Protocol

```python
from nexagen.providers.base import LLMProvider
```

Protocol that all provider backends must implement:

```python
class LLMProvider(Protocol):
    async def chat(
        self,
        messages: list[NexagenMessage],
        tools: list[ToolSchema] | None = None,
    ) -> NexagenResponse: ...

    def supports_tool_calling(self) -> bool: ...
    def supports_vision(self) -> bool: ...
```

Any class that implements these three methods satisfies the protocol (structural typing, no inheritance needed).
