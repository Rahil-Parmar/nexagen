# Getting Started

This guide walks you through installing nexagen and running your first agent.

---

## Installation

### Using uv (recommended)

```bash
uv add nexagen
```

### Using pip

```bash
pip install nexagen
```

### Optional dependencies

nexagen has optional extras for cloud providers and MCP:

```bash
# Individual providers
pip install nexagen[anthropic]   # Anthropic Claude
pip install nexagen[openai]      # OpenAI GPT
pip install nexagen[google]      # Google Gemini

# MCP support
pip install nexagen[mcp]

# Everything
pip install nexagen[all]
```

---

## Prerequisites

You need at least one LLM backend running. The easiest way to get started is with **Ollama** (free, runs locally):

```bash
# Install Ollama: https://ollama.com
ollama pull qwen3
```

For cloud providers, set the appropriate environment variable:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Google
export GOOGLE_API_KEY="AIza..."
```

---

## Quick Start: Simple Agent

The simplest possible agent -- ask a question, get an answer:

```python
import asyncio
from nexagen import Agent

async def main():
    agent = Agent(provider="ollama/qwen3")

    async for message in agent.run("Explain what a Python decorator is"):
        if message.role == "assistant" and message.text:
            print(message.text)

asyncio.run(main())
```

The `agent.run()` method is an async generator that yields `NexagenMessage` objects. Each message has a `role` ("assistant", "tool", "user") and optional `text`.

---

## Quick Start: Agent with Tools

Give the agent tools to interact with the file system:

```python
import asyncio
from nexagen import Agent

async def main():
    agent = Agent(
        provider="ollama/qwen3",
        tools=["file_read", "file_write", "grep", "glob"],
        permission_mode="safe",
    )

    async for message in agent.run("List all Python files in the current directory"):
        if message.role == "assistant" and message.text:
            print(f"Agent: {message.text}")
        elif message.role == "tool":
            print(f"Tool: {message.text[:200]}")

asyncio.run(main())
```

The agent will autonomously call tools, observe results, and continue until it has a final answer.

---

## Running with Different Providers

### Ollama (local, default)

```python
agent = Agent(provider="ollama/qwen3")
```

### Ollama on a remote machine

```python
agent = Agent(provider="ollama/qwen3@192.168.1.100:11434")
```

### OpenAI

```python
# Requires: OPENAI_API_KEY env var or api_key in ProviderConfig
agent = Agent(provider="openai/gpt-4o")
```

### Anthropic

```python
# Requires: ANTHROPIC_API_KEY env var
agent = Agent(provider="anthropic/claude-sonnet-4-20250514")
```

### Google Gemini

```python
# Requires: GOOGLE_API_KEY env var
agent = Agent(provider="google/gemini-2.0-flash")
```

### Groq (cloud, fast inference)

```python
from nexagen import Agent, ProviderConfig

agent = Agent(
    provider=ProviderConfig(
        backend="groq",
        model="llama-3.3-70b-versatile",
        api_key="gsk_...",
    )
)
```

### Together AI

```python
from nexagen import Agent, ProviderConfig

agent = Agent(
    provider=ProviderConfig(
        backend="together",
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        api_key="...",
    )
)
```

---

## Using the CLI

Run a one-shot prompt:

```bash
nexagen run "What is the capital of France?" -p ollama/qwen3
```

Start an interactive chat session:

```bash
nexagen chat -p ollama/qwen3 -t file_read,bash
```

See [CLI and TUI](cli-tui.md) for full CLI documentation.

---

## Next Steps

- [Providers](providers.md) -- detailed provider configuration
- [Tools](tools.md) -- built-in tools and creating custom ones
- [Agent Architecture](agent.md) -- how the agent loop works
- [Examples](examples.md) -- practical recipes
