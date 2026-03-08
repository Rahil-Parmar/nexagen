# Architecture Overview

This document describes the architecture of nexagen, its component relationships, design patterns, and data flow.

---

## System Architecture

```
+-----------------------------------------------------------------------+
|                            User / CLI / TUI                           |
+----------------------------------+------------------------------------+
                                   |
                                   v
+----------------------------------+------------------------------------+
|                              Agent                                    |
|                                                                       |
|  +------------------+  +------------------+  +---------------------+  |
|  | PermissionManager|  |  Conversation    |  | SupervisorAgent     |  |
|  |                  |  |                  |  | (optional)          |  |
|  | - mode check     |  | - message history|  | - progress check   |  |
|  | - allowlist      |  | - token estimate |  | - context compress  |  |
|  | - callback       |  | - task summaries |  |                     |  |
|  +------------------+  +------------------+  +---------------------+  |
|                                                                       |
|  +------------------+  +------------------+                           |
|  |  ToolRegistry    |  |  ProviderRegistry|                           |
|  |                  |  |                  |                           |
|  | - register()     |  | - register()     |                           |
|  | - get()          |  | - resolve()      |                           |
|  | - list_available |  |                  |                           |
|  | - get_schemas()  |  |                  |                           |
|  +--------+---------+  +--------+---------+                           |
|           |                      |                                    |
+-----------+----------------------+------------------------------------+
            |                      |
            v                      v
+---------------------+  +---------------------+
|       Tools         |  |     Providers        |
|                     |  |                      |
| +------+ +-------+  |  | +-----------------+  |
| |Built-| | Custom|  |  | |OpenAICompat     |  |
| | in   | | Tools |  |  | | (Ollama, vLLM,  |  |
| +------+ +-------+  |  | |  LM Studio,     |  |
| +------+            |  | |  Groq, Together) |  |
| | MCP  |            |  | +-----------------+  |
| |Tools |            |  | |Anthropic        |  |
| +------+            |  | +-----------------+  |
|                     |  | |OpenAI Native    |  |
|                     |  | +-----------------+  |
|                     |  | |Google           |  |
|                     |  | +-----------------+  |
+---------------------+  +---------------------+
```

---

## Component Relationships

### Agent (orchestrator)

The `Agent` is the central orchestrator. It:
- Owns a `ToolRegistry` with all registered tools
- Delegates LLM calls to a provider (resolved via `ProviderRegistry`)
- Enforces permissions via `PermissionManager`
- Manages conversation state via `Conversation`
- Optionally uses a `SupervisorAgent` for monitoring

### ProviderRegistry + Providers

```
ProviderRegistry
    |
    +-- "ollama"    --> OpenAICompatProvider
    +-- "vllm"      --> OpenAICompatProvider
    +-- "lmstudio"  --> OpenAICompatProvider
    +-- "groq"      --> OpenAICompatProvider
    +-- "together"  --> OpenAICompatProvider
    +-- "openai"    --> OpenAINativeProvider
    +-- "anthropic" --> AnthropicProvider
    +-- "google"    --> GoogleProvider
```

All providers implement the `LLMProvider` protocol. The registry maps backend names to provider classes and resolves them from a string or `ProviderConfig`.

### ToolRegistry + Tools

```
ToolRegistry
    |
    +-- "file_read"  --> BaseTool (built-in)
    +-- "file_write" --> BaseTool (built-in)
    +-- "bash"       --> BaseTool (built-in)
    +-- "my_tool"    --> BaseTool (custom, via @tool)
    +-- "mcp__fs__read" --> MCPTool (from MCP server)
```

Tools are registered by name. The registry filters by `is_available()` when generating schemas for the LLM.

---

## Design Patterns

### Strategy Pattern (Providers)

Each LLM provider is a strategy that implements the same interface (`LLMProvider` protocol). The agent is decoupled from any specific provider -- switching from Ollama to OpenAI requires only changing the provider string.

```
LLMProvider (Protocol)
    |
    +-- chat(messages, tools) -> NexagenResponse
    +-- supports_tool_calling() -> bool
    +-- supports_vision() -> bool
```

**Why this fits:** Different LLM APIs have different request formats, authentication schemes, and response structures. The Strategy pattern lets us encapsulate each one behind a uniform interface without the agent needing to know the details.

### Protocol (structural typing)

nexagen uses Python's `Protocol` (from `typing`) instead of abstract base classes for the `LLMProvider` interface. This means any class with the right methods qualifies -- no inheritance required.

```python
@runtime_checkable
class LLMProvider(Protocol):
    async def chat(self, messages, tools) -> NexagenResponse: ...
    def supports_tool_calling(self) -> bool: ...
    def supports_vision(self) -> bool: ...
```

**Why this fits:** Users can bring their own provider implementations without inheriting from a base class. Just implement the methods and pass it in.

### Registry Pattern (Providers and Tools)

Both providers and tools use registries that map names to instances/classes. This decouples discovery from usage.

### Decorator Pattern (Tools)

The `@tool` decorator transforms a plain async function into a `BaseTool` instance with input validation, error handling, and schema generation.

---

## Data Flow

### Message types

All communication flows through `NexagenMessage`:

```python
NexagenMessage(
    role="system" | "user" | "assistant" | "tool",
    text="...",                    # Message content
    tool_calls=[ToolCall(...)],    # Tools the LLM wants to call
    summary="...",                 # Compressed summary
    tool_call_id="...",            # For tool results
    is_error=False,                # For tool errors
)
```

### Request flow

```
User prompt (str)
    |
    v
NexagenMessage(role="user", text=prompt)
    |
    v
Conversation.get_messages_with_history()
    |  (includes system prompt + past task summaries + current messages)
    v
Provider.chat(messages, tool_schemas)
    |  (converts to provider-specific format)
    v
LLM API call (HTTP)
    |
    v
Raw API response
    |
    v
Provider parses into NexagenResponse
    |
    v
NexagenResponse.message (NexagenMessage)
    |
    +-- No tool calls --> yield message, loop ends
    |
    +-- Has tool calls --> for each ToolCall:
            |
            v
        PermissionManager.check(name, args)
            |
            +-- Deny --> ToolResult(is_error=True)
            |
            +-- Allow --> tool.execute(args) --> ToolResult
                    |
                    v
                ToolResult.to_message() --> NexagenMessage(role="tool")
                    |
                    v
                yield tool message, add to conversation
            |
            v
        Loop back to Provider.chat()
```

### Provider message conversion

Each provider converts `NexagenMessage` to its native format:

| Provider | System | User | Assistant + Tools | Tool Result |
|----------|--------|------|-------------------|-------------|
| OpenAI-compat | `{"role": "system"}` | `{"role": "user"}` | `tool_calls: [...]` | `{"role": "tool", "tool_call_id": ...}` |
| Anthropic | Top-level `system` param | `{"role": "user"}` | `tool_use` content blocks | `tool_result` in user message |
| Google | `system_instruction` | `{"role": "user"}` | `functionCall` parts | `functionResponse` parts |

---

## Module Structure

```
src/nexagen/
    __init__.py            # Public API exports
    agent.py               # Agent loop orchestrator
    models.py              # Data models (NexagenMessage, ProviderConfig, etc.)
    constants.py           # Default values and configuration constants
    permissions.py         # Three-layer permission system
    conversation.py        # Conversation state management
    agent_logging.py       # JSON-structured logging
    providers/
        __init__.py
        base.py            # LLMProvider protocol
        registry.py        # ProviderRegistry
        openai_compat.py   # Ollama, vLLM, LM Studio, Groq, Together
        openai_native.py   # OpenAI native API
        anthropic_provider.py  # Anthropic Messages API
        google_provider.py     # Google Generative Language API
    tools/
        __init__.py
        base.py            # BaseTool class and @tool decorator
        registry.py        # ToolRegistry
        mcp.py             # MCPServerConfig, MCPTool, MCPManager
        builtin/
            __init__.py    # BUILTIN_TOOLS dict
            file_read.py
            file_write.py
            file_edit.py
            bash.py
            grep_tool.py
            glob_tool.py
    supervisor/
        __init__.py
        supervisor.py      # SupervisorAgent and ActionEntry
    cli/
        __init__.py
        app.py             # Click CLI commands
    tui/
        __init__.py
        app.py             # Textual TUI application
```
