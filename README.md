# nexagen

**Universal LLM Agent SDK.** Connect to any provider. Run agents with tools. Support MCP integration. All from a single, clean Python API.

nexagen lets you build AI agents that work with any LLM backend -- local models via Ollama/vLLM/LM Studio, or cloud APIs like OpenAI, Anthropic, Google, Groq, and Together. Switch providers by changing a single string.

---

## Features

- **Universal provider support** -- Ollama, vLLM, LM Studio, OpenAI, Anthropic, Google Gemini, Groq, Together AI
- **Built-in tools** -- file read/write/edit, bash, grep, glob
- **Custom tools** -- `@tool` decorator with Pydantic validation
- **MCP integration** -- connect to any MCP server for external tools
- **Permission system** -- three-layer security (mode, allowlist, callback)
- **Supervisor agent** -- progress monitoring and context compression
- **Conversation management** -- task summaries, context window handling
- **CLI and TUI** -- command-line interface and interactive terminal UI
- **Async-first** -- built on async/await for non-blocking execution
- **Zero lock-in** -- Protocol-based provider interface, no inheritance required

---

## Install

```bash
pip install nexagen
```

Or with uv:

```bash
uv add nexagen
```

### Optional extras

```bash
pip install nexagen[anthropic]  # Anthropic Claude
pip install nexagen[openai]     # OpenAI GPT
pip install nexagen[google]     # Google Gemini
pip install nexagen[mcp]        # MCP support
pip install nexagen[all]        # Everything
```

---

## Quick Start

### Simple agent

```python
import asyncio
from nexagen import Agent

async def main():
    agent = Agent(provider="ollama/qwen3")

    async for msg in agent.run("Explain Python decorators"):
        if msg.role == "assistant" and msg.text:
            print(msg.text)

asyncio.run(main())
```

### Agent with tools

```python
import asyncio
from nexagen import Agent

async def main():
    agent = Agent(
        provider="ollama/qwen3",
        tools=["file_read", "grep", "glob"],
        permission_mode="readonly",
    )

    async for msg in agent.run("Find all TODO comments in this project"):
        if msg.role == "assistant" and msg.text:
            print(msg.text)

asyncio.run(main())
```

### Custom tools

```python
from pydantic import BaseModel
from nexagen import Agent, tool

class GreetInput(BaseModel):
    name: str

@tool(name="greet", description="Greet a user by name", input_model=GreetInput)
async def greet(args: GreetInput) -> str:
    return f"Hello, {args.name}!"

agent = Agent(provider="ollama/qwen3", custom_tools=[greet])
```

### CLI

```bash
# One-shot
nexagen run "What is the capital of France?" -p ollama/qwen3

# Interactive chat
nexagen chat -p ollama/qwen3 -t file_read,grep
```

---

## Supported Providers

| Provider | Type | Provider String |
|----------|------|----------------|
| Ollama | Local | `ollama/qwen3` |
| vLLM | Local | `vllm/mistral-7b` |
| LM Studio | Local | `lmstudio/my-model` |
| OpenAI | Cloud | `openai/gpt-4o` |
| Anthropic | Cloud | `anthropic/claude-sonnet-4-20250514` |
| Google | Cloud | `google/gemini-2.0-flash` |
| Groq | Cloud | via `ProviderConfig` |
| Together | Cloud | via `ProviderConfig` |

Custom host: `ollama/qwen3@192.168.1.5:11434`

---

## Architecture

```
User / CLI / TUI
       |
       v
+------+-------+
|    Agent      |  Orchestrates the plan-act-observe loop
|               |
| +Permission+  |  Three-layer security
| +Conversat.+  |  History + compression
| +Supervisor+  |  Progress monitoring (optional)
+------+-------+
       |
  +----+----+
  |         |
  v         v
Tools    Providers
(6 built-in,   (8 backends,
 custom,        unified
 MCP)           protocol)
```

The agent loop: call LLM -> execute tool calls -> feed results back -> repeat until done.

---

## Documentation

Full documentation is in the [`docs/`](docs/) directory:

- [Getting Started](docs/getting-started.md) -- installation and first agent
- [Providers](docs/providers.md) -- configuring LLM backends
- [Tools](docs/tools.md) -- built-in and custom tools
- [MCP Integration](docs/mcp.md) -- connecting to MCP servers
- [Agent Architecture](docs/agent.md) -- the agent loop, supervisor, permissions
- [CLI and TUI](docs/cli-tui.md) -- command-line and terminal UI
- [Architecture Overview](docs/architecture.md) -- system design and data flow
- [API Reference](docs/api-reference.md) -- all public classes and methods
- [Examples](docs/examples.md) -- practical recipes

---

## Environment Variables

| Variable | Provider |
|----------|----------|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `GOOGLE_API_KEY` | Google Gemini |

---

## Requirements

- Python 3.11+
- At least one LLM backend (Ollama for local, or an API key for cloud)

---

## Contributing

Contributions are welcome. To set up a development environment:

```bash
git clone https://github.com/your-org/nexagen.git
cd nexagen
uv sync --extra dev

# Run tests
pytest

# Lint
ruff check src/
```

### Guidelines

- Write tests for new features
- Follow the existing code style (ruff-enforced)
- Use async/await for all I/O operations
- Add type hints to all function signatures
- Keep tools stateless when possible
- Use Pydantic models for tool inputs

### Project structure

```
src/nexagen/
    agent.py            # Core agent loop
    models.py           # Data models
    constants.py        # Defaults
    permissions.py      # Permission system
    conversation.py     # Conversation management
    providers/          # LLM provider backends
    tools/              # Tool system (base, registry, MCP, builtins)
    supervisor/         # Supervisor agent
    cli/                # CLI (Click)
    tui/                # TUI (Textual)
```

---

## License

MIT
