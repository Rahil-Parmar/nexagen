"""MCP (Model Context Protocol) integration for the nexagen tool system."""

from __future__ import annotations

from dataclasses import dataclass, field

from nexagen.models import ToolResult
from nexagen.tools.base import BaseTool


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""

    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


class MCPTool(BaseTool):
    """A tool backed by an MCP server. Implements the full lifecycle protocol."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict,
        server_config: MCPServerConfig,
    ):
        # Don't call super().__init__ — we manage our own state
        self.name = name
        self.description = description
        self._schema = input_schema
        self.server_config = server_config
        self._connected = False
        self._client = None

    @property
    def input_schema(self) -> dict:
        return self._schema

    async def connect(self) -> None:
        """Connect to the MCP server.

        In production, this starts the subprocess and initializes the MCP client.
        """
        # TODO: Implement actual MCP client connection using the mcp Python SDK
        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        self._connected = False
        self._client = None

    def is_available(self) -> bool:
        return self._connected

    async def execute(self, args: dict) -> ToolResult:
        if not self._connected:
            return ToolResult(
                tool_call_id="",
                output="Error: MCP server not connected. Call connect() first.",
                is_error=True,
            )
        # TODO: In production, forward the call to the MCP server via JSON-RPC
        return ToolResult(
            tool_call_id="",
            output=f"MCP tool '{self.name}' called with args: {args}",
            is_error=False,
        )

    def to_tool_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._schema,
        }


class MCPManager:
    """Manages multiple MCP server connections and their tools."""

    def __init__(self):
        self._servers: dict[str, MCPServerConfig] = {}
        self._tools: dict[str, MCPTool] = {}

    def add_server(self, name: str, config: MCPServerConfig):
        self._servers[name] = config

    def add_server_from_dict(self, name: str, config_dict: dict):
        """Convenience method to add server from dict config."""
        if not isinstance(config_dict, dict):
            raise TypeError(f"Expected dict for MCP server config, got {type(config_dict).__name__}")
        if "command" not in config_dict:
            raise ValueError("MCP server config must include 'command' key")
        config = MCPServerConfig(
            command=config_dict["command"],
            args=config_dict.get("args", []),
            env=config_dict.get("env", {}),
        )
        self.add_server(name, config)

    def register_tool(self, server_name: str, tool: MCPTool):
        """Register a discovered tool from an MCP server."""
        self._tools[f"mcp__{server_name}__{tool.name}"] = tool

    async def connect_all(self):
        """Connect to all registered MCP servers and discover their tools."""
        for name, config in self._servers.items():
            # TODO: In production, start MCP server subprocess, initialize client,
            # call tools/list to discover tools, wrap each as MCPTool
            pass

    async def disconnect_all(self):
        """Disconnect from all MCP servers. Continues even if individual disconnects fail."""
        import logging
        for name, tool in self._tools.items():
            try:
                await tool.disconnect()
            except Exception as e:
                logging.getLogger("nexagen.mcp").warning(
                    "Failed to disconnect MCP tool '%s': %s", name, e
                )

    def get_tools(self) -> list[MCPTool]:
        """Return all discovered MCP tools."""
        return list(self._tools.values())

    def get_available_tools(self) -> list[MCPTool]:
        """Return only connected/available MCP tools."""
        return [t for t in self._tools.values() if t.is_available()]
