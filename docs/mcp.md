# MCP Integration

nexagen supports the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) for connecting to external tool servers.

---

## What is MCP?

MCP is an open protocol that allows LLM applications to connect to external tool servers. Instead of building tools directly into your application, you can connect to MCP servers that expose tools over a standardized JSON-RPC interface.

Benefits:
- **Reuse tools** across different LLM applications
- **Isolate tool execution** in separate processes
- **Use community tools** from the MCP ecosystem
- **Separate concerns** between agent logic and tool implementation

---

## MCPServerConfig

Configure an MCP server connection:

```python
from nexagen import MCPServerConfig

config = MCPServerConfig(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/home/user/documents"],
    env={"NODE_ENV": "production"},
)
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `command` | `str` | required | Command to start the MCP server |
| `args` | `list[str]` | `[]` | Arguments passed to the command |
| `env` | `dict[str, str]` | `{}` | Environment variables for the server process |

---

## Adding MCP Servers to an Agent

### Using MCPManager

The `MCPManager` handles multiple MCP server connections:

```python
from nexagen import MCPManager, MCPServerConfig

manager = MCPManager()

# Add servers
manager.add_server("filesystem", MCPServerConfig(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
))

manager.add_server("github", MCPServerConfig(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-github"],
    env={"GITHUB_TOKEN": "ghp_..."},
))

# Connect to all servers and discover tools
await manager.connect_all()

# Get discovered tools for use with an agent
tools = manager.get_available_tools()
```

### Adding from a dict

You can also configure servers from a dictionary (useful for config files):

```python
manager.add_server_from_dict("filesystem", {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    "env": {},
})
```

### Using MCP tools with an Agent

Pass discovered MCP tools as custom tools:

```python
from nexagen import Agent, MCPManager, MCPServerConfig

manager = MCPManager()
manager.add_server("fs", MCPServerConfig(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
))
await manager.connect_all()

agent = Agent(
    provider="ollama/qwen3",
    custom_tools=manager.get_available_tools(),
)

async for msg in agent.run("List files in /tmp"):
    if msg.role == "assistant" and msg.text:
        print(msg.text)

# Clean up
await manager.disconnect_all()
```

---

## Tool Namespacing

MCP tools are namespaced to avoid collisions between servers. The naming convention is:

```
mcp__{server_name}__{tool_name}
```

For example, if you register a server named `"filesystem"` and it exposes a tool called `"read_file"`, the tool will be available as:

```
mcp__filesystem__read_file
```

This means you can connect multiple MCP servers that expose tools with the same name without conflicts.

---

## Lifecycle Management

MCP servers run as child processes. Proper lifecycle management is important:

```python
manager = MCPManager()
manager.add_server("my_server", config)

try:
    # Start servers and discover tools
    await manager.connect_all()

    # Use tools...
    agent = Agent(
        provider="ollama/qwen3",
        custom_tools=manager.get_available_tools(),
    )
    async for msg in agent.run("Do something"):
        pass

finally:
    # Always disconnect to clean up child processes
    await manager.disconnect_all()
```

### Individual tool lifecycle

Each `MCPTool` also has its own lifecycle:

```python
from nexagen import MCPTool

tool = MCPTool(
    name="read_file",
    description="Read a file",
    input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
    server_config=config,
)

await tool.connect()       # Start the server connection
tool.is_available()        # True if connected
result = await tool.execute({"path": "/tmp/test.txt"})
await tool.disconnect()    # Clean up
```

---

## MCPManager API

| Method | Description |
|--------|-------------|
| `add_server(name, config)` | Register an MCP server configuration |
| `add_server_from_dict(name, dict)` | Register from a dictionary |
| `connect_all()` | Connect to all servers and discover tools |
| `disconnect_all()` | Disconnect from all servers |
| `get_tools()` | Return all discovered tools |
| `get_available_tools()` | Return only connected/available tools |
| `register_tool(server_name, tool)` | Manually register a tool for a server |
