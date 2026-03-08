# Tools

Tools give agents the ability to interact with the world -- read files, run commands, search codebases, and more. nexagen includes a set of built-in tools and makes it straightforward to create custom ones.

---

## Built-in Tools

nexagen ships with six built-in tools:

| Tool | Name | Description |
|------|------|-------------|
| File Read | `file_read` | Read file contents with line numbers |
| File Write | `file_write` | Write content to a file |
| File Edit | `file_edit` | Edit a file with find-and-replace |
| Bash | `bash` | Execute shell commands |
| Grep | `grep` | Search file contents with regex |
| Glob | `glob` | Find files by pattern |

### Enabling built-in tools

Pass tool names to the `tools` parameter:

```python
from nexagen import Agent

# Enable specific tools
agent = Agent(
    provider="ollama/qwen3",
    tools=["file_read", "grep", "glob"],
)

# Enable all file tools + bash
agent = Agent(
    provider="ollama/qwen3",
    tools=["file_read", "file_write", "file_edit", "bash", "grep", "glob"],
)
```

### file_read

Reads file contents with line numbers.

**Parameters:**
- `file_path` (str, required) -- Absolute path to the file
- `offset` (int, optional) -- Line number to start reading from (1-based)
- `limit` (int, optional) -- Number of lines to read

### file_write

Writes content to a file, creating it if it does not exist.

**Parameters:**
- `file_path` (str, required) -- Absolute path to the file
- `content` (str, required) -- Content to write

### file_edit

Edits a file by replacing a string with another.

**Parameters:**
- `file_path` (str, required) -- Absolute path to the file
- `old_string` (str, required) -- Text to find
- `new_string` (str, required) -- Text to replace it with

### bash

Executes a shell command and returns stdout/stderr.

**Parameters:**
- `command` (str, required) -- Shell command to execute

### grep

Searches file contents using regex patterns.

**Parameters:**
- `pattern` (str, required) -- Regex pattern to search for
- `path` (str, optional) -- Directory or file to search in
- `include` (str, optional) -- Glob pattern to filter files (e.g., `"*.py"`)

### glob

Finds files matching a glob pattern.

**Parameters:**
- `pattern` (str, required) -- Glob pattern (e.g., `"**/*.py"`)
- `path` (str, optional) -- Base directory to search from

---

## Creating Custom Tools

Custom tools are created using the `@tool` decorator and a Pydantic model for input validation.

### Step 1: Define the input model

```python
from pydantic import BaseModel

class WeatherInput(BaseModel):
    city: str
    units: str = "celsius"  # default value
```

The Pydantic model serves double duty:
1. It validates input from the LLM (catching type errors, missing fields)
2. Its JSON Schema is sent to the LLM so it knows the tool's parameters

### Step 2: Write the handler

```python
from nexagen import tool

@tool(
    name="get_weather",
    description="Get current weather for a city",
    input_model=WeatherInput,
)
async def get_weather(args: WeatherInput) -> str:
    # Your implementation here
    return f"Weather in {args.city}: 22 {args.units}"
```

Key rules:
- The handler must be an **async function**
- It receives a validated Pydantic model instance
- It must return a **string** (the tool result shown to the LLM)
- Exceptions are caught automatically and returned as error results

### Step 3: Register with the agent

```python
from nexagen import Agent

agent = Agent(
    provider="ollama/qwen3",
    custom_tools=[get_weather],
)
```

### Complete example

```python
import asyncio
from pydantic import BaseModel
from nexagen import Agent, tool

class WordCountInput(BaseModel):
    text: str

@tool(
    name="word_count",
    description="Count the number of words in a text",
    input_model=WordCountInput,
)
async def word_count(args: WordCountInput) -> str:
    count = len(args.text.split())
    return f"Word count: {count}"

async def main():
    agent = Agent(
        provider="ollama/qwen3",
        custom_tools=[word_count],
    )

    async for msg in agent.run("How many words are in 'the quick brown fox'?"):
        if msg.role == "assistant" and msg.text:
            print(msg.text)

asyncio.run(main())
```

---

## Tool Lifecycle

Every tool (including custom tools) has lifecycle hooks:

```
connect()  -->  is_available()  -->  execute()  -->  disconnect()
```

| Method | When Called | Default Behavior |
|--------|-----------|------------------|
| `connect()` | When the tool is attached to an agent | No-op |
| `disconnect()` | When the tool is detached | No-op |
| `is_available()` | Before including the tool in schemas | Returns `True` |
| `execute(args)` | When the LLM calls the tool | Validates input, runs handler |

### Overriding lifecycle hooks

For tools that need setup (database connections, API clients):

```python
from nexagen.tools.base import BaseTool
from pydantic import BaseModel

class DBQueryInput(BaseModel):
    query: str

class DatabaseTool(BaseTool):
    def __init__(self, connection_string: str):
        self._conn_string = connection_string
        self._db = None
        super().__init__(
            name="db_query",
            description="Run a database query",
            input_model=DBQueryInput,
            handler=self._handle,
        )

    async def connect(self):
        # Called when tool is registered
        self._db = await create_connection(self._conn_string)

    async def disconnect(self):
        if self._db:
            await self._db.close()

    def is_available(self) -> bool:
        return self._db is not None

    async def _handle(self, args: DBQueryInput) -> str:
        result = await self._db.execute(args.query)
        return str(result)
```

---

## Tool Schemas

When tools are registered, nexagen converts them to a provider-agnostic schema format:

```python
{
    "name": "file_read",
    "description": "Read contents of a file with line numbers",
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "offset": {"type": "integer", "default": null},
            "limit": {"type": "integer", "default": null},
        },
        "required": ["file_path"],
    },
}
```

Each provider adapter converts this schema into the format expected by the LLM API:
- **OpenAI-compatible:** Wrapped in `{"type": "function", "function": {...}}`
- **Anthropic:** Uses `input_schema` instead of `parameters`
- **Google:** Wrapped in `{"function_declarations": [...]}`

You never need to worry about these differences -- nexagen handles the conversion.

---

## Error Handling

Tool errors are handled gracefully at two levels:

### Input validation errors

If the LLM sends invalid arguments, Pydantic catches it:

```
ValidationError: field required
  in file_read
```

### Runtime errors

If your handler raises an exception, nexagen catches it and returns a structured error:

```
ValueError: File not found: /nonexistent/path
  File "tools.py", line 15, in my_handler
```

The error is sent back to the LLM as a tool result with `is_error=True`, allowing the agent to retry or adjust its approach.

### Consecutive error tracking

The agent tracks consecutive errors per tool. If a tool fails `max_tool_errors` times in a row (default: 3), and a supervisor is configured, the supervisor is consulted on whether to continue or stop.

---

## Tool Registry

Under the hood, tools are managed by a `ToolRegistry`:

```python
from nexagen import ToolRegistry

registry = ToolRegistry()
registry.register(my_tool)
registry.register_many([tool_a, tool_b])

# Get a tool by name
tool = registry.get("my_tool")

# List all available tools
available = registry.list_available()

# Get schemas for LLM
schemas = registry.get_tool_schemas()
```

You typically do not interact with the registry directly -- the `Agent` manages it for you.
