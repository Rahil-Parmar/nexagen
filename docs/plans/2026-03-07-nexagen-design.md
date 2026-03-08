# nexagen — Design Document

## Overview

nexagen is a Python SDK that connects to any LLM provider (local and cloud), implements an agent loop with tool use, and provides built-in file operations, shell commands, and MCP support. It is a foundational SDK — other solutions (e.g., A2A multi-agent systems) build on top of it.

## Goals

- Connect to any LLM: Ollama, vLLM, LM Studio, OpenAI, Anthropic, Google, Groq, and any OpenAI-compatible endpoint
- Agent loop with multi-step tool use and reasoning
- Built-in developer tools: file read/write/edit, bash, grep, glob
- Custom tool definitions via `@tool` decorator with Pydantic models
- MCP server integration for external tools
- CLI and TUI interfaces
- Designed as a building block for multi-agent frameworks (A2A, etc.)

## Architecture

```
+-------------------------------------------------------------+
|                         nexagen                             |
|                                                             |
|  +---------+  +---------+  +----------------------------+  |
|  |   CLI   |  |   TUI   |  |     Python SDK (library)   |  |
|  | (click) |  |(textual)|  |  from nexagen import Agent  |  |
|  +----+----+  +----+----+  +------------+---------------+  |
|       +-------------+-------------------+                   |
|                     |                                       |
|  +------------------------------------------------------+  |
|  |                 Agent Engine                          |  |
|  |                                                      |  |
|  |  +------------------+    +------------------------+  |  |
|  |  |   Agent Loop     |<-->|   Supervisor Agent     |  |  |
|  |  | prompt>act>observe|    | - continue/stop        |  |  |
|  |  | >repeat          |    | - context compression   |  |  |
|  |  +--------+---------+    | - error escalation      |  |  |
|  |           |              +------------------------+  |  |
|  |  +--------v---------+                                |  |
|  |  |  Conversation    |    +------------------------+  |  |
|  |  | - message history|    |  Permission System     |  |  |
|  |  | - task summaries |    | - modes (ro/safe/full) |  |  |
|  |  | - token tracking |    | - allowlist            |  |  |
|  |  +------------------+    | - callback             |  |  |
|  |                          +------------------------+  |  |
|  +------------------------------------------------------+  |
|                     |                                       |
|  +------------------------------------------------------+  |
|  |            Provider Layer (Strategy Pattern)          |  |
|  |  +----------+ +--------+ +-------+ +------------+   |  |
|  |  |Anthropic | | OpenAI | |Google | |OAI-compat  |   |  |
|  |  | native   | | native | |native | |(Ollama,    |   |  |
|  |  |          | |        | |       | | vLLM, etc) |   |  |
|  |  +----------+ +--------+ +-------+ +------------+   |  |
|  +------------------------------------------------------+  |
|                     |                                       |
|  +------------------------------------------------------+  |
|  |              Tool Layer                               |  |
|  |  +----------+ +--------------+ +----------------+    |  |
|  |  | Built-in | | Custom @tool | |  MCP Servers   |    |  |
|  |  |file_read | |  (Pydantic)  | | (stdio, SSE)   |    |  |
|  |  |file_write| |              | |                |    |  |
|  |  |file_edit | |              | | connect()      |    |  |
|  |  |bash      | |              | | disconnect()   |    |  |
|  |  |grep      | |              | | is_available() |    |  |
|  |  |glob      | |              | |                |    |  |
|  |  +----------+ +--------------+ +----------------+    |  |
|  +------------------------------------------------------+  |
+-------------------------------------------------------------+
```

## Design Decisions

### 1. Provider Pattern: Strategy

Each provider is a self-contained class implementing the `LLMProvider` protocol. The provider handles both the API call and response translation internally.

```python
class LLMProvider(Protocol):
    async def chat(self, messages: list[NexagenMessage], tools: list[Tool]) -> NexagenResponse: ...
    async def stream(self, messages: list[NexagenMessage], tools: list[Tool]) -> AsyncIterator[NexagenChunk]: ...
    def supports_tool_calling(self) -> bool: ...
    def supports_vision(self) -> bool: ...
```

**Why Strategy over Adapter:** For LLM providers, the API call and response translation are coupled — if you know the provider, you know both. One class per provider is simpler and easier to extend.

Providers:
- `AnthropicProvider` — native Anthropic Messages API
- `OpenAIProvider` — native OpenAI Chat Completions API
- `GoogleProvider` — native Google Gemini API
- `OpenAICompatProvider` — any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio, Groq, Together, etc.)

### 2. Common Message Format: Separated Fields

```python
class NexagenMessage:
    role: str               # "system" | "user" | "assistant" | "tool"
    text: str | None        # text content
    tool_calls: list[ToolCall] | None  # tool invocations (assistant only)
    summary: str | None     # 1-line intent statement (assistant only, when tool_calls present)
    tool_call_id: str | None  # links tool result to its call (tool role only)
    is_error: bool          # whether this tool result is an error (tool role only)
```

**Why separated over content blocks:** The agent loop's hot path is checking for tool calls and extracting them. Separate fields make this a direct attribute access rather than filtering a list. Most providers (OpenAI-compat) naturally produce this format.

### 3. Agent Loop: Supervisor Pattern

A separate supervisor agent (small/fast model) monitors the worker agent's progress.

**Worker agent:** Executes the task using tools.
**Supervisor agent:** Judges progress every N tool calls, returns `{"decision": "continue"}` or `{"decision": "stop"}`.

Supervisor receives:
- Original user task
- Action log: each step's 1-line summary + tool call names
- Does NOT receive full tool outputs (stays lightweight)

Supervisor output parsing:
1. Try JSON parse → extract `decision` field
2. Fallback: search text for "continue" or "stop"
3. Neither found → default to "stop" (safe)

### 4. Tool Execution: Sequential

Tool calls are executed one after another, never in parallel. This prevents dependency issues (e.g., write then read same file) and keeps behavior predictable.

### 5. Tool Errors

Format: error type + error message (last line) + first 2 lines of traceback.

```
CommandNotFoundError: 'setup.py' not found
  in bash, line 15
```

Each tool result carries an `is_error: bool` flag (inspired by Anthropic).

Error escalation: 3 consecutive errors on the same tool → flag to supervisor for continue/stop decision.

### 6. @tool Decorator: Pydantic Input Models

```python
class SearchInput(BaseModel):
    query: str
    max_results: int = 10

@tool("search", "Search the codebase", input_model=SearchInput)
async def search(args: SearchInput) -> str:
    ...
```

Pydantic provides:
- `model_json_schema()` → JSON schema sent to LLM as tool definition
- `model_validate(raw_dict)` → validates LLM output before calling the function
- IDE autocomplete and type checking

### 7. Tool Interface: Lifecycle Methods

```python
class Tool(Protocol):
    name: str
    description: str
    input_schema: dict

    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    def is_available(self) -> bool: ...
    async def execute(self, args: dict) -> ToolResult: ...
```

Built-in and custom tools: `connect`/`disconnect` are no-ops, `is_available` always returns True.
MCP tools: `connect` starts the server, `disconnect` stops it, `is_available` checks connection health.

The tool registry checks `is_available()` before each LLM call and only sends available tools.

### 8. Provider Configuration: String + Config Object

```python
# String shorthand
agent = Agent(provider="ollama/qwen3")

# Full control
agent = Agent(provider=ProviderConfig(
    backend="ollama",
    model="qwen3",
    base_url="http://192.168.1.5:11434",
    temperature=0.7,
    max_tokens=4096,
))
```

API keys: read from environment variables only (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY).

### 9. Context Management: Compress at 80%

When conversation reaches 80% of the model's context window:
- Supervisor compresses older messages into a summary
- Keeps: original user message (always) + last 3 messages
- Compresses down to ~50% of context window
- Token estimation: 1 token ≈ 4 characters (approximate, no per-model tokenizer needed)

### 10. Permission System: Three Layers

```
Layer 1: Permission modes (presets)
  "readonly"  → file_read, grep, glob only
  "safe"      → read + write tools, no bash
  "full"      → everything allowed

Layer 2: Tool allowlist
  allowed_tools=["file_read", "grep"]  → narrows from mode

Layer 3: Permission callback
  can_use_tool(tool_name, args) → Allow | Deny
  Fine-grained runtime checks
```

Layers stack: mode sets baseline → allowlist narrows → callback does final check.

### 11. Conversation Continuity: Task Summaries

Each completed task collapses into a 1-statement summary. Cross-task history grows by one line per task:

```
[system_prompt, task_1_summary, task_2_summary, ..., current_user_message]
```

### 12. Streaming: Complete Messages

The SDK waits for the full LLM response before returning it. No token-by-token streaming. Simplifies the agent loop and tool execution flow.

### 13. Logging: Structured JSON Logs

Python `logging` module with JSON formatter. Each entry captures:
- timestamp, level, event type
- tool name, args, cycle number
- supervisor decisions
- error details

### 14. Configuration: Constants File

```python
# src/nexagen/constants.py
DEFAULT_MODEL = "ollama/qwen3"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096
DEFAULT_CONTEXT_THRESHOLD = 0.80
DEFAULT_COMPRESS_TARGET = 0.50
DEFAULT_MAX_TOOL_ERRORS = 3
DEFAULT_SUPERVISOR_MODEL = "ollama/phi3"
DEFAULT_SUPERVISOR_CHECK_INTERVAL = 5
DEFAULT_PERMISSION_MODE = "safe"
```

### 15. Frontend: TUI

Interactive terminal UI using Textual or Rich. No web GUI.

## Project Structure

```
nexagen/
├── pyproject.toml
├── README.md
├── src/
│   └── nexagen/
│       ├── __init__.py              # Public API exports
│       ├── agent.py                 # Agent class + agent loop
│       ├── conversation.py          # Message history, context management
│       ├── constants.py             # Shared defaults and configuration
│       ├── permissions.py           # Permission system (modes, allowlist, callback)
│       ├── logging.py               # Structured JSON logging setup
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── base.py              # LLMProvider protocol + NexagenMessage/Response
│       │   ├── openai_compat.py     # OpenAI-compatible (Ollama, vLLM, Groq, etc.)
│       │   ├── anthropic.py         # Native Anthropic
│       │   ├── openai.py            # Native OpenAI
│       │   ├── google.py            # Native Google/Gemini
│       │   └── registry.py          # Provider discovery & string parsing
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── base.py              # Tool protocol + @tool decorator + ToolResult
│       │   ├── registry.py          # Tool registry (availability checks)
│       │   ├── mcp.py               # MCP server integration
│       │   └── builtin/
│       │       ├── __init__.py
│       │       ├── file_read.py
│       │       ├── file_write.py
│       │       ├── file_edit.py
│       │       ├── bash.py
│       │       ├── grep.py
│       │       └── glob.py
│       ├── supervisor/
│       │   ├── __init__.py
│       │   └── supervisor.py        # Supervisor agent (progress check + compression)
│       ├── cli/
│       │   ├── __init__.py
│       │   └── app.py               # CLI entry point (click + rich)
│       └── tui/
│           ├── __init__.py
│           └── app.py               # Interactive TUI (textual)
├── tests/
│   ├── test_agent.py
│   ├── test_providers/
│   ├── test_tools/
│   └── test_supervisor.py
└── docs/
    └── plans/
```

## Public API

```python
# Core
from nexagen import Agent, AgentOptions, ProviderConfig

# Tools
from nexagen import tool
from nexagen.tools import ToolRegistry, ToolResult

# Providers
from nexagen.providers import get_provider

# MCP
from nexagen.tools.mcp import MCPServer

# Types
from nexagen.providers.base import NexagenMessage, NexagenResponse, ToolCall
```

## Dependencies

```
httpx          — HTTP client for providers
pydantic       — Data models, tool schemas, config
click          — CLI framework
rich           — CLI formatting
textual        — TUI framework
mcp            — MCP Python SDK (client for tool servers)
```

## Usage Examples

### Basic Agent

```python
from nexagen import Agent

agent = Agent(provider="ollama/qwen3")

async for message in agent.run("What files are in this directory?"):
    print(message)
```

### Agent with Tools

```python
from nexagen import Agent

agent = Agent(
    provider="ollama/qwen3",
    tools=["file_read", "file_write", "bash", "grep"],
    permission_mode="safe",
)

async for message in agent.run("Find and fix the bug in auth.py"):
    print(message)
```

### Custom Tools

```python
from nexagen import Agent, tool
from pydantic import BaseModel

class SearchInput(BaseModel):
    query: str
    max_results: int = 10

@tool("search_docs", "Search documentation", input_model=SearchInput)
async def search_docs(args: SearchInput) -> str:
    results = my_search_index.search(args.query, limit=args.max_results)
    return str(results)

agent = Agent(
    provider="anthropic/claude-sonnet-4-20250514",
    custom_tools=[search_docs],
)

async for message in agent.run("Find docs about authentication"):
    print(message)
```

### MCP Integration

```python
from nexagen import Agent
from nexagen.tools.mcp import MCPServer

agent = Agent(
    provider="openai/gpt-4o",
    mcp_servers={
        "github": MCPServer(command="npx", args=["-y", "@modelcontextprotocol/server-github"]),
    },
)

async for message in agent.run("List open issues in my repo"):
    print(message)
```

### Conversation Continuity

```python
from nexagen import Agent, Conversation

agent = Agent(provider="ollama/qwen3", tools=["file_read", "file_write", "bash"])
conv = Conversation()

async for msg in agent.run("Fix the bug in auth.py", conversation=conv):
    print(msg)

# conv now contains a 1-line summary of the completed task

async for msg in agent.run("Write tests for that fix", conversation=conv):
    print(msg)
```

### Permission Callback

```python
from nexagen import Agent, Allow, Deny

async def check(tool_name, args):
    if tool_name == "bash" and "rm" in args.get("command", ""):
        return Deny("Destructive command blocked")
    return Allow()

agent = Agent(
    provider="ollama/qwen3",
    tools=["file_read", "bash"],
    permission_mode="full",
    can_use_tool=check,
)
```
