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
from nexagen.supervisor.supervisor import SupervisorFeedback
from nexagen.execution import ParallelExecutor
from nexagen.context import ContextManager
from nexagen.reflection import ReflectionEngine, ReflectionResult
from nexagen.planning import PlanningPhase, Plan, Subtask
from nexagen.memory import EpisodicMemory, Episode

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
    "SupervisorFeedback",
    "ActionEntry",
    "ParallelExecutor",
    "ContextManager",
    "ReflectionEngine",
    "ReflectionResult",
    "PlanningPhase",
    "Plan",
    "Subtask",
    "EpisodicMemory",
    "Episode",
]
