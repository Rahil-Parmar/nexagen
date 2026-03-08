"""Tests for MCP (Model Context Protocol) tool integration."""

from __future__ import annotations

import pytest

from nexagen.tools.mcp import MCPServerConfig, MCPTool, MCPManager


# --- MCPServerConfig ---

class TestMCPServerConfig:
    def test_mcp_server_config(self):
        config = MCPServerConfig(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem"],
            env={"HOME": "/tmp"},
        )
        assert config.command == "npx"
        assert config.args == ["-y", "@modelcontextprotocol/server-filesystem"]
        assert config.env == {"HOME": "/tmp"}

    def test_mcp_server_config_defaults(self):
        config = MCPServerConfig(command="python")
        assert config.args == []
        assert config.env == {}


# --- MCPTool ---

class TestMCPTool:
    def _make_tool(self) -> MCPTool:
        return MCPTool(
            name="read_file",
            description="Read a file from disk",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
            server_config=MCPServerConfig(command="npx"),
        )

    def test_mcp_tool_is_available(self):
        tool = self._make_tool()
        assert tool.is_available() is False

    @pytest.mark.asyncio
    async def test_mcp_tool_not_connected(self):
        tool = self._make_tool()
        result = await tool.execute({"path": "/tmp/foo"})
        assert result.is_error is True
        assert "not connected" in result.output.lower()

    @pytest.mark.asyncio
    async def test_mcp_tool_connect_disconnect(self):
        tool = self._make_tool()
        assert tool.is_available() is False

        await tool.connect()
        assert tool.is_available() is True

        await tool.disconnect()
        assert tool.is_available() is False

    @pytest.mark.asyncio
    async def test_mcp_tool_execute_connected(self):
        tool = self._make_tool()
        await tool.connect()
        result = await tool.execute({"path": "/tmp/foo"})
        assert result.is_error is False
        assert "read_file" in result.output
        assert "/tmp/foo" in result.output

    def test_mcp_tool_to_schema(self):
        tool = self._make_tool()
        schema = tool.to_tool_schema()
        assert schema["name"] == "read_file"
        assert schema["description"] == "Read a file from disk"
        assert schema["parameters"] == {
            "type": "object",
            "properties": {"path": {"type": "string"}},
        }

    def test_mcp_tool_input_schema_property(self):
        tool = self._make_tool()
        assert tool.input_schema == {
            "type": "object",
            "properties": {"path": {"type": "string"}},
        }


# --- MCPManager ---

class TestMCPManager:
    def test_mcp_manager_add_server(self):
        manager = MCPManager()
        config = MCPServerConfig(command="npx", args=["server"])
        manager.add_server("filesystem", config)
        assert "filesystem" in manager._servers
        assert manager._servers["filesystem"] is config

    def test_mcp_manager_add_server_from_dict(self):
        manager = MCPManager()
        manager.add_server_from_dict("filesystem", {
            "command": "npx",
            "args": ["-y", "server"],
            "env": {"KEY": "val"},
        })
        config = manager._servers["filesystem"]
        assert config.command == "npx"
        assert config.args == ["-y", "server"]
        assert config.env == {"KEY": "val"}

    def test_mcp_manager_register_tool(self):
        manager = MCPManager()
        tool = MCPTool(
            name="read_file",
            description="Read a file",
            input_schema={},
            server_config=MCPServerConfig(command="npx"),
        )
        manager.register_tool("filesystem", tool)
        assert "mcp__filesystem__read_file" in manager._tools
        assert manager._tools["mcp__filesystem__read_file"] is tool

    @pytest.mark.asyncio
    async def test_mcp_manager_get_available_tools(self):
        manager = MCPManager()
        tool_a = MCPTool(
            name="tool_a", description="A", input_schema={},
            server_config=MCPServerConfig(command="x"),
        )
        tool_b = MCPTool(
            name="tool_b", description="B", input_schema={},
            server_config=MCPServerConfig(command="x"),
        )
        manager.register_tool("srv", tool_a)
        manager.register_tool("srv", tool_b)

        # Neither connected yet
        assert manager.get_available_tools() == []

        # Connect only tool_a
        await tool_a.connect()
        available = manager.get_available_tools()
        assert len(available) == 1
        assert available[0] is tool_a

    @pytest.mark.asyncio
    async def test_mcp_manager_disconnect_all(self):
        manager = MCPManager()
        tool_a = MCPTool(
            name="tool_a", description="A", input_schema={},
            server_config=MCPServerConfig(command="x"),
        )
        tool_b = MCPTool(
            name="tool_b", description="B", input_schema={},
            server_config=MCPServerConfig(command="x"),
        )
        manager.register_tool("srv", tool_a)
        manager.register_tool("srv", tool_b)

        await tool_a.connect()
        await tool_b.connect()
        assert len(manager.get_available_tools()) == 2

        await manager.disconnect_all()
        assert manager.get_available_tools() == []
        assert tool_a.is_available() is False
        assert tool_b.is_available() is False
