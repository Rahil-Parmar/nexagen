# nexagen Documentation

**nexagen** is a universal LLM Agent SDK that connects to any provider, runs agents with tool use, and supports MCP integration. Build AI agents that work with Ollama, vLLM, LM Studio, OpenAI, Anthropic, Google, Groq, Together, and more.

---

## Guides

| Guide | Description |
|-------|-------------|
| [Getting Started](getting-started.md) | Installation, quick start, and running your first agent |
| [Providers](providers.md) | Configuring LLM backends (Ollama, OpenAI, Anthropic, Google, etc.) |
| [Tools](tools.md) | Built-in tools, creating custom tools, and the tool lifecycle |
| [MCP Integration](mcp.md) | Connecting to MCP servers and using external tools |
| [Agent Architecture](agent.md) | The agent loop, supervisor, permissions, and conversation management |
| [CLI and TUI](cli-tui.md) | Command-line interface, interactive chat, and the terminal UI |
| [Architecture Overview](architecture.md) | System design, component relationships, and data flow |
| [API Reference](api-reference.md) | All public classes, methods, parameters, and return types |
| [Examples](examples.md) | Practical recipes for common use cases |

---

## Quick Links

- **Install:** `pip install nexagen` or `uv add nexagen`
- **Run an agent:** `nexagen run "summarize this file" -t file_read`
- **Interactive chat:** `nexagen chat -p ollama/qwen3 -t file_read,bash`
- **Python API:**

```python
import asyncio
from nexagen import Agent

async def main():
    agent = Agent(provider="ollama/qwen3")
    async for msg in agent.run("What is 2 + 2?"):
        if msg.role == "assistant" and msg.text:
            print(msg.text)

asyncio.run(main())
```

---

## Requirements

- Python 3.11+
- At least one LLM backend (Ollama for local, or an API key for cloud providers)

## License

MIT
