# src/nexagen/__init__.py
"""nexagen — Universal LLM Agent SDK."""

from nexagen.agent import Agent
from nexagen.models import (
    NexagenMessage,
    NexagenResponse,
    ProviderConfig,
    ToolCall,
    ToolResult,
)
from nexagen.tools.base import tool, BaseTool
from nexagen.tools.registry import ToolRegistry
from nexagen.tools.mcp import MCPServerConfig, MCPTool, MCPManager
from nexagen.conversation import Conversation
from nexagen.permissions import Allow, Deny, PermissionManager
from nexagen.supervisor import SupervisorAgent, ActionEntry

__all__ = [
    "Agent",
    "NexagenMessage",
    "NexagenResponse",
    "ProviderConfig",
    "ToolCall",
    "ToolResult",
    "tool",
    "BaseTool",
    "ToolRegistry",
    "MCPServerConfig",
    "MCPTool",
    "MCPManager",
    "Conversation",
    "Allow",
    "Deny",
    "PermissionManager",
    "SupervisorAgent",
    "ActionEntry",
]
