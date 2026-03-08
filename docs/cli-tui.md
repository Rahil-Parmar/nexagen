# CLI and TUI

nexagen provides both a command-line interface (CLI) and a terminal user interface (TUI) for interacting with agents.

---

## CLI Commands

### `nexagen run`

Run a single prompt through the agent:

```bash
nexagen run "Explain how Python generators work"
```

#### Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--provider` | `-p` | `ollama/qwen3` | LLM provider string |
| `--tools` | `-t` | none | Comma-separated tool names |
| `--permission-mode` | `-m` | `safe` | Permission mode: `readonly`, `safe`, `full` |
| `--system-prompt` | `-s` | none | Custom system prompt |

#### Examples

```bash
# Simple question
nexagen run "What is the capital of France?"

# Use a specific provider
nexagen run "Summarize this code" -p openai/gpt-4o

# Enable tools
nexagen run "Find all TODO comments in this project" -t grep,glob

# Full access with bash
nexagen run "Install requests and test it" -t bash -m full

# Custom system prompt
nexagen run "Review this code" -s "You are a senior Python developer"
```

### `nexagen chat`

Start an interactive chat session:

```bash
nexagen chat
```

#### Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--provider` | `-p` | `ollama/qwen3` | LLM provider string |
| `--tools` | `-t` | none | Comma-separated tool names |
| `--permission-mode` | `-m` | `safe` | Permission mode |

#### Examples

```bash
# Interactive chat with default settings
nexagen chat

# Chat with tools enabled
nexagen chat -t file_read,grep,glob

# Chat with a cloud provider
nexagen chat -p anthropic/claude-sonnet-4-20250514

# Full access mode
nexagen chat -p ollama/qwen3 -t file_read,file_write,bash -m full
```

#### Chat session

In chat mode, the conversation is persistent across prompts:

```
nexagen interactive chat
Provider: ollama/qwen3
Tools: file_read,grep
Mode: safe

Type 'exit' or 'quit' to end.

You: What Python files are in the current directory?
Agent: Let me check...
Agent: I found 5 Python files: main.py, utils.py, ...

You: Which one is the largest?
Agent: Based on my earlier search, main.py is the largest at 245 lines.

You: exit
Goodbye!
```

### `nexagen --version`

Print the version number:

```bash
nexagen --version
# nexagen, version 0.1.0
```

---

## TUI (Terminal User Interface)

nexagen includes a rich terminal UI built with [Textual](https://textual.textualize.io/) that provides a more visual experience.

### Launching the TUI

```python
from nexagen.tui.app import run_tui

run_tui(
    provider="ollama/qwen3",
    tools=["file_read", "grep"],
    permission_mode="safe",
)
```

### TUI Features

#### Step-by-step progress

When the agent uses tools, each step is displayed with a progress indicator:

```
[check] Step 1: Searching for Python files
[check] Step 2: Reading main.py
[check] Step 3: Analyzing the code
Agent: Here's my analysis of main.py...
```

#### Spinner

While the agent is thinking, a spinner animation is displayed:

```
[spinner] Thinking...
```

#### Keyboard shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+C` | Quit the application |
| `Ctrl+L` | Clear the chat log |

#### UI Layout

```
+------------------------------------------+
| Header                          12:34:56  |
+------------------------------------------+
|                                          |
|  You: Find all Python files              |
|  [check] Step 1: Searching...            |
|  [check] Step 2: Counting results        |
|  Agent: Found 12 Python files...         |
|                                          |
|  You: Which is the largest?              |
|  Agent: main.py at 450 lines             |
|                                          |
+------------------------------------------+
| Provider: ollama/qwen3 | Mode: safe     |
+------------------------------------------+
| Type your message...                     |
+------------------------------------------+
| Ctrl+C Quit  Ctrl+L Clear               |
+------------------------------------------+
```

#### Message styling

- **User messages:** Bold blue
- **Agent responses:** Bold green
- **Tool results:** Dimmed (truncated to 200 chars)
- **Tool errors:** Bold red
- **Status:** Yellow with spinner

---

## Output Formatting

### CLI run mode

In `nexagen run` mode, output is formatted with Rich panels:

- Agent responses are displayed in a green-bordered panel
- Tool results are shown with dimmed text (truncated to 500 chars)
- Tool errors are shown in red

### CLI chat mode

In `nexagen chat` mode, output is inline:

- `You:` prefix for user messages (bold blue)
- `Agent:` prefix for agent responses (bold green)
- Errors shown in red
