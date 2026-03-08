# Examples

Practical recipes for common use cases.

---

## Basic Agent

The simplest agent with no tools -- just an LLM conversation:

```python
import asyncio
from nexagen import Agent

async def main():
    agent = Agent(provider="ollama/qwen3")

    async for msg in agent.run("What are the SOLID principles?"):
        if msg.role == "assistant" and msg.text:
            print(msg.text)

asyncio.run(main())
```

---

## Agent with File Tools

An agent that can read, search, and navigate files:

```python
import asyncio
from nexagen import Agent

async def main():
    agent = Agent(
        provider="ollama/qwen3",
        tools=["file_read", "grep", "glob"],
        permission_mode="readonly",
    )

    async for msg in agent.run("Find all Python files and summarize what each one does"):
        if msg.role == "assistant" and msg.text:
            print(f"\nAgent: {msg.text}")
        elif msg.role == "tool" and not msg.is_error:
            # Show truncated tool output
            print(f"  [tool] {msg.text[:100]}...")

asyncio.run(main())
```

---

## Custom Tool Agent

Build an agent with your own tools:

```python
import asyncio
import httpx
from pydantic import BaseModel
from nexagen import Agent, tool

class FetchURLInput(BaseModel):
    url: str

@tool(
    name="fetch_url",
    description="Fetch the contents of a URL",
    input_model=FetchURLInput,
)
async def fetch_url(args: FetchURLInput) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(args.url, timeout=10.0)
        response.raise_for_status()
        # Return first 2000 chars
        return response.text[:2000]

class WordCountInput(BaseModel):
    text: str

@tool(
    name="word_count",
    description="Count words in a text",
    input_model=WordCountInput,
)
async def word_count(args: WordCountInput) -> str:
    count = len(args.text.split())
    return f"Word count: {count}"

async def main():
    agent = Agent(
        provider="ollama/qwen3",
        custom_tools=[fetch_url, word_count],
        permission_mode="full",
    )

    async for msg in agent.run("Fetch example.com and count the words"):
        if msg.role == "assistant" and msg.text:
            print(msg.text)

asyncio.run(main())
```

---

## MCP Integration

Connect to an MCP server and use its tools:

```python
import asyncio
from nexagen import Agent, MCPManager, MCPServerConfig

async def main():
    manager = MCPManager()

    # Add a filesystem MCP server
    manager.add_server("filesystem", MCPServerConfig(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/home/user/documents"],
    ))

    try:
        # Connect and discover tools
        await manager.connect_all()

        agent = Agent(
            provider="ollama/qwen3",
            custom_tools=manager.get_available_tools(),
            permission_mode="full",
        )

        async for msg in agent.run("List all markdown files in my documents"):
            if msg.role == "assistant" and msg.text:
                print(msg.text)

    finally:
        await manager.disconnect_all()

asyncio.run(main())
```

---

## Permission Callback

Fine-grained permission control with a callback:

```python
import asyncio
from nexagen import Agent, Allow, Deny

async def permission_check(tool_name: str, args: dict) -> Allow | Deny:
    """Custom permission logic."""

    # Block dangerous bash commands
    if tool_name == "bash":
        command = args.get("command", "")
        dangerous = ["rm -rf", "sudo", "chmod 777", "mkfs", "> /dev"]
        for pattern in dangerous:
            if pattern in command:
                return Deny(f"Blocked dangerous command: {pattern}")

    # Block writes outside /tmp
    if tool_name == "file_write":
        path = args.get("file_path", "")
        if not path.startswith("/tmp/"):
            return Deny("Writes only allowed in /tmp")

    return Allow()

async def main():
    agent = Agent(
        provider="ollama/qwen3",
        tools=["file_read", "file_write", "bash"],
        permission_mode="full",
        can_use_tool=permission_check,
    )

    async for msg in agent.run("Create a test file and run a command"):
        if msg.role == "assistant" and msg.text:
            print(msg.text)
        elif msg.role == "tool" and msg.is_error:
            print(f"DENIED: {msg.text}")

asyncio.run(main())
```

---

## Conversation Continuity

Maintain context across multiple agent runs:

```python
import asyncio
from nexagen import Agent, Conversation

async def main():
    agent = Agent(
        provider="ollama/qwen3",
        tools=["file_read", "grep", "glob"],
    )

    # Shared conversation persists across runs
    conv = Conversation(context_window=8192)

    # First task
    print("=== Task 1 ===")
    async for msg in agent.run("Find all Python files in /tmp", conversation=conv):
        if msg.role == "assistant" and msg.text:
            print(f"Agent: {msg.text}")

    # Second task -- agent remembers the first
    print("\n=== Task 2 ===")
    async for msg in agent.run("Which of those files is the largest?", conversation=conv):
        if msg.role == "assistant" and msg.text:
            print(f"Agent: {msg.text}")

    # Third task
    print("\n=== Task 3 ===")
    async for msg in agent.run("Show me the first 10 lines of that file", conversation=conv):
        if msg.role == "assistant" and msg.text:
            print(f"Agent: {msg.text}")

asyncio.run(main())
```

---

## Using as a Library

Embed nexagen in your own application:

```python
import asyncio
from nexagen import Agent, ProviderConfig, Conversation

class CodeReviewBot:
    """A code review bot built on nexagen."""

    def __init__(self, provider_string: str = "ollama/qwen3"):
        self.agent = Agent(
            provider=provider_string,
            tools=["file_read", "grep", "glob"],
            system_prompt=(
                "You are a senior code reviewer. When asked to review code, "
                "read the files, identify issues, and provide actionable feedback. "
                "Categorize issues as: must-fix, should-fix, or nit."
            ),
            permission_mode="readonly",
        )

    async def review(self, file_path: str) -> str:
        """Review a file and return feedback."""
        responses = []
        async for msg in self.agent.run(f"Review the code in {file_path}"):
            if msg.role == "assistant" and msg.text:
                responses.append(msg.text)
        return "\n".join(responses)

async def main():
    bot = CodeReviewBot()
    feedback = await bot.review("/path/to/your/code.py")
    print(feedback)

asyncio.run(main())
```

---

## Supervised Agent

Use a supervisor LLM to monitor progress:

```python
import asyncio
from nexagen import Agent, ProviderConfig
from nexagen.providers.openai_compat import OpenAICompatProvider

async def main():
    # Supervisor uses a smaller, faster model
    supervisor = OpenAICompatProvider(ProviderConfig(
        backend="ollama",
        model="phi3",
    ))

    agent = Agent(
        provider="ollama/qwen3",
        tools=["file_read", "file_write", "bash", "grep", "glob"],
        permission_mode="full",
        supervisor=supervisor,
        supervisor_check_interval=5,  # Check every 5 tool calls
        max_tool_errors=3,            # Escalate after 3 consecutive failures
    )

    async for msg in agent.run("Refactor all Python files to use type hints"):
        if msg.role == "assistant" and msg.text:
            print(f"Agent: {msg.text}")

asyncio.run(main())
```

---

## Building an A2A Agent on Top of nexagen

Use nexagen as the foundation for an Agent-to-Agent (A2A) system:

```python
import asyncio
from nexagen import Agent, Conversation
from pydantic import BaseModel
from nexagen import tool

# Define a tool that delegates to another agent
class SubTaskInput(BaseModel):
    task: str
    context: str = ""

@tool(
    name="delegate_research",
    description="Delegate a research task to a specialized research agent",
    input_model=SubTaskInput,
)
async def delegate_research(args: SubTaskInput) -> str:
    """A tool that runs a sub-agent to handle research tasks."""
    research_agent = Agent(
        provider="ollama/qwen3",
        tools=["file_read", "grep", "glob"],
        system_prompt="You are a research assistant. Be thorough and cite sources.",
        permission_mode="readonly",
    )

    results = []
    prompt = f"{args.task}\n\nContext: {args.context}" if args.context else args.task
    async for msg in research_agent.run(prompt):
        if msg.role == "assistant" and msg.text:
            results.append(msg.text)

    return "\n".join(results)

@tool(
    name="delegate_coding",
    description="Delegate a coding task to a specialized coding agent",
    input_model=SubTaskInput,
)
async def delegate_coding(args: SubTaskInput) -> str:
    """A tool that runs a sub-agent to handle coding tasks."""
    coding_agent = Agent(
        provider="ollama/qwen3",
        tools=["file_read", "file_write", "file_edit", "bash"],
        system_prompt="You are a senior software engineer. Write clean, tested code.",
        permission_mode="safe",
    )

    results = []
    prompt = f"{args.task}\n\nContext: {args.context}" if args.context else args.task
    async for msg in coding_agent.run(prompt):
        if msg.role == "assistant" and msg.text:
            results.append(msg.text)

    return "\n".join(results)

async def main():
    # Orchestrator agent that delegates to specialized sub-agents
    orchestrator = Agent(
        provider="ollama/qwen3",
        custom_tools=[delegate_research, delegate_coding],
        system_prompt=(
            "You are a project manager. Break down complex tasks and delegate "
            "to the appropriate specialist: use delegate_research for information "
            "gathering and delegate_coding for implementation tasks."
        ),
        permission_mode="full",
    )

    async for msg in orchestrator.run(
        "Research best practices for Python error handling, "
        "then create a utility module that implements them"
    ):
        if msg.role == "assistant" and msg.text:
            print(f"Orchestrator: {msg.text}")

asyncio.run(main())
```

This pattern lets you compose agents hierarchically -- an orchestrator agent delegates to specialist agents, each with their own tools, permissions, and system prompts.
