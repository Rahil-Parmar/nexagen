# tests/test_public_api.py

def test_import_agent():
    from nexagen import Agent
    assert Agent is not None

def test_import_models():
    from nexagen import NexagenMessage, NexagenResponse, ProviderConfig, ToolCall, ToolResult
    assert all(cls is not None for cls in [NexagenMessage, NexagenResponse, ProviderConfig, ToolCall, ToolResult])

def test_import_tool_decorator():
    from nexagen import tool, BaseTool
    assert callable(tool)
    assert BaseTool is not None

def test_import_tool_registry():
    from nexagen import ToolRegistry
    assert ToolRegistry is not None

def test_import_mcp():
    from nexagen import MCPServerConfig, MCPTool, MCPManager
    assert all(cls is not None for cls in [MCPServerConfig, MCPTool, MCPManager])

def test_import_conversation():
    from nexagen import Conversation
    assert Conversation is not None

def test_import_permissions():
    from nexagen import Allow, Deny, PermissionManager
    assert all(cls is not None for cls in [Allow, Deny, PermissionManager])

def test_import_supervisor():
    from nexagen import SupervisorAgent, ActionEntry
    assert all(cls is not None for cls in [SupervisorAgent, ActionEntry])

def test_all_exports():
    import nexagen
    assert hasattr(nexagen, "__all__")
    assert len(nexagen.__all__) >= 16
