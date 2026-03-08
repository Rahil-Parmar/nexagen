# nexagen Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Python SDK that connects to any LLM provider and implements an agent loop with tool use, MCP support, and a TUI.

**Architecture:** Strategy pattern for providers, supervisor agent for loop safety, Pydantic-based tool system, three-layer permissions. The agent loop is the core — providers and tools plug into it via protocols.

**Tech Stack:** Python 3.11+, pydantic, httpx, click, rich, textual, mcp (Python SDK)

**Design doc:** `docs/plans/2026-03-07-nexagen-design.md`

---

## Phase 1: Project Scaffolding & Core Types

### Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `src/nexagen/__init__.py`
- Create: `src/nexagen/constants.py`
- Create: `README.md`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "nexagen"
version = "0.1.0"
description = "Universal LLM Agent SDK — connect to any provider, run agents locally"
readme = "README.md"
requires-python = ">=3.11"
license = "MIT"
dependencies = [
    "pydantic>=2.0",
    "httpx>=0.27",
    "click>=8.0",
    "rich>=13.0",
    "textual>=0.50",
]

[project.optional-dependencies]
anthropic = ["anthropic>=0.40"]
openai = ["openai>=1.50"]
google = ["google-genai>=1.0"]
mcp = ["mcp>=1.0"]
all = ["nexagen[anthropic,openai,google,mcp]"]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "ruff>=0.8",
]

[project.scripts]
nexagen = "nexagen.cli.app:main"
```

**Step 2: Create constants.py**

```python
# src/nexagen/constants.py

DEFAULT_MODEL = "ollama/qwen3"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096
DEFAULT_CONTEXT_THRESHOLD = 0.80
DEFAULT_COMPRESS_TARGET = 0.50
DEFAULT_MAX_TOOL_ERRORS = 3
DEFAULT_SUPERVISOR_MODEL = "ollama/phi3"
DEFAULT_SUPERVISOR_CHECK_INTERVAL = 5
DEFAULT_PERMISSION_MODE = "safe"
CHARS_PER_TOKEN = 4
```

**Step 3: Create __init__.py (empty for now)**

```python
# src/nexagen/__init__.py
"""nexagen — Universal LLM Agent SDK."""
```

**Step 4: Create README.md**

```markdown
# nexagen

Universal LLM Agent SDK. Connect to any provider. Run agents locally.

## Install

pip install nexagen

## Quick Start

from nexagen import Agent

agent = Agent(provider="ollama/qwen3")
```

**Step 5: Initialize git and commit**

```bash
cd /Users/rahilparmar/Projects/nexagen
git init
git add pyproject.toml src/ README.md docs/
git commit -m "feat: initial project scaffolding with pyproject.toml and constants"
```

---

### Task 2: Core Data Models

**Files:**
- Create: `src/nexagen/models.py`
- Create: `tests/__init__.py`
- Create: `tests/test_models.py`

**Step 1: Write the failing tests**

```python
# tests/test_models.py
import pytest
from nexagen.models import (
    NexagenMessage,
    ToolCall,
    ToolResult,
    NexagenResponse,
    ProviderConfig,
)


def test_user_message():
    msg = NexagenMessage(role="user", text="Hello")
    assert msg.role == "user"
    assert msg.text == "Hello"
    assert msg.tool_calls is None
    assert msg.summary is None


def test_assistant_message_with_tool_calls():
    tc = ToolCall(id="1", name="read_file", arguments={"path": "main.py"})
    msg = NexagenMessage(
        role="assistant",
        text="I'll read that file",
        tool_calls=[tc],
        summary="Reading main.py to understand the code",
    )
    assert msg.tool_calls[0].name == "read_file"
    assert msg.summary is not None


def test_tool_result_success():
    result = ToolResult(tool_call_id="1", output="file contents here", is_error=False)
    assert result.is_error is False


def test_tool_result_error():
    result = ToolResult(tool_call_id="1", output="FileNotFoundError: not found\n  in read_file, line 12", is_error=True)
    assert result.is_error is True


def test_tool_result_to_message():
    result = ToolResult(tool_call_id="1", output="file contents", is_error=False)
    msg = result.to_message()
    assert msg.role == "tool"
    assert msg.tool_call_id == "1"
    assert msg.text == "file contents"
    assert msg.is_error is False


def test_nexagen_response():
    tc = ToolCall(id="1", name="bash", arguments={"command": "ls"})
    msg = NexagenMessage(role="assistant", text="Running ls", tool_calls=[tc], summary="Listing files")
    resp = NexagenResponse(message=msg)
    assert resp.has_tool_calls is True


def test_nexagen_response_no_tools():
    msg = NexagenMessage(role="assistant", text="Here is the answer")
    resp = NexagenResponse(message=msg)
    assert resp.has_tool_calls is False


def test_provider_config_from_string():
    config = ProviderConfig.from_string("ollama/qwen3")
    assert config.backend == "ollama"
    assert config.model == "qwen3"


def test_provider_config_from_string_with_host():
    config = ProviderConfig.from_string("ollama/qwen3@192.168.1.5:11434")
    assert config.backend == "ollama"
    assert config.model == "qwen3"
    assert config.base_url == "http://192.168.1.5:11434"


def test_provider_config_cloud():
    config = ProviderConfig.from_string("openai/gpt-4o")
    assert config.backend == "openai"
    assert config.model == "gpt-4o"
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/rahilparmar/Projects/nexagen && python -m pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nexagen.models'`

**Step 3: Implement models.py**

```python
# src/nexagen/models.py
from __future__ import annotations
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict


class ToolResult(BaseModel):
    tool_call_id: str
    output: str
    is_error: bool = False

    def to_message(self) -> NexagenMessage:
        return NexagenMessage(
            role="tool",
            text=self.output,
            tool_call_id=self.tool_call_id,
            is_error=self.is_error,
        )


class NexagenMessage(BaseModel):
    role: str  # "system" | "user" | "assistant" | "tool"
    text: str | None = None
    tool_calls: list[ToolCall] | None = None
    summary: str | None = None
    tool_call_id: str | None = None
    is_error: bool = False


class NexagenResponse(BaseModel):
    message: NexagenMessage

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.message.tool_calls)


class ProviderConfig(BaseModel):
    backend: str
    model: str
    base_url: str | None = None
    api_key: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None

    @classmethod
    def from_string(cls, provider_string: str) -> ProviderConfig:
        base_url = None

        if "@" in provider_string:
            provider_part, host = provider_string.split("@", 1)
            if not host.startswith("http"):
                base_url = f"http://{host}"
            else:
                base_url = host
        else:
            provider_part = provider_string

        backend, model = provider_part.split("/", 1)
        return cls(backend=backend, model=model, base_url=base_url)
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/rahilparmar/Projects/nexagen && pip install -e ".[dev]" && python -m pytest tests/test_models.py -v`
Expected: All 10 tests PASS

**Step 5: Commit**

```bash
git add src/nexagen/models.py tests/
git commit -m "feat: core data models — NexagenMessage, ToolCall, ToolResult, ProviderConfig"
```

---

## Phase 2: Provider Layer

### Task 3: Provider Protocol & Registry

**Files:**
- Create: `src/nexagen/providers/__init__.py`
- Create: `src/nexagen/providers/base.py`
- Create: `src/nexagen/providers/registry.py`
- Create: `tests/test_providers/__init__.py`
- Create: `tests/test_providers/test_registry.py`

**Step 1: Write the failing tests**

```python
# tests/test_providers/test_registry.py
import pytest
from nexagen.providers.registry import ProviderRegistry
from nexagen.providers.base import LLMProvider
from nexagen.models import ProviderConfig


def test_registry_resolves_ollama():
    registry = ProviderRegistry()
    provider = registry.resolve("ollama/qwen3")
    assert isinstance(provider, LLMProvider)


def test_registry_resolves_provider_config():
    registry = ProviderRegistry()
    config = ProviderConfig(backend="ollama", model="qwen3")
    provider = registry.resolve(config)
    assert isinstance(provider, LLMProvider)


def test_registry_unknown_backend():
    registry = ProviderRegistry()
    with pytest.raises(ValueError, match="Unknown backend"):
        registry.resolve("unknown/model")
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_providers/test_registry.py -v`
Expected: FAIL

**Step 3: Implement base.py**

```python
# src/nexagen/providers/base.py
from __future__ import annotations
from typing import Protocol, AsyncIterator, runtime_checkable
from nexagen.models import NexagenMessage, NexagenResponse, ToolCall


class ToolSchema(dict):
    """JSON schema dict representing a tool for the LLM."""
    pass


@runtime_checkable
class LLMProvider(Protocol):
    async def chat(
        self, messages: list[NexagenMessage], tools: list[ToolSchema] | None = None
    ) -> NexagenResponse: ...

    def supports_tool_calling(self) -> bool: ...

    def supports_vision(self) -> bool: ...
```

**Step 4: Implement registry.py**

```python
# src/nexagen/providers/registry.py
from __future__ import annotations
from nexagen.models import ProviderConfig
from nexagen.providers.base import LLMProvider


class ProviderRegistry:
    def __init__(self):
        self._backends: dict[str, type] = {}
        self._register_defaults()

    def _register_defaults(self):
        from nexagen.providers.openai_compat import OpenAICompatProvider
        self._backends["ollama"] = OpenAICompatProvider
        self._backends["vllm"] = OpenAICompatProvider
        self._backends["lmstudio"] = OpenAICompatProvider
        self._backends["groq"] = OpenAICompatProvider
        self._backends["together"] = OpenAICompatProvider

    def register(self, backend_name: str, provider_class: type):
        self._backends[backend_name] = provider_class

    def resolve(self, provider: str | ProviderConfig) -> LLMProvider:
        if isinstance(provider, str):
            config = ProviderConfig.from_string(provider)
        else:
            config = provider

        if config.backend not in self._backends:
            raise ValueError(f"Unknown backend: '{config.backend}'. Available: {list(self._backends.keys())}")

        provider_class = self._backends[config.backend]
        return provider_class(config)
```

**Step 5: Create providers __init__.py**

```python
# src/nexagen/providers/__init__.py
from nexagen.providers.registry import ProviderRegistry

_registry = ProviderRegistry()

def get_provider(provider: str):
    return _registry.resolve(provider)
```

**Step 6: Run tests (will still fail — OpenAICompatProvider doesn't exist yet)**

Expected: ImportError — this is fine, Task 4 creates it.

**Step 7: Commit (partial — the protocol and registry structure)**

```bash
git add src/nexagen/providers/
git commit -m "feat: provider protocol and registry with backend resolution"
```

---

### Task 4: OpenAI-Compatible Provider

**Files:**
- Create: `src/nexagen/providers/openai_compat.py`
- Create: `tests/test_providers/test_openai_compat.py`

**Step 1: Write the failing tests**

```python
# tests/test_providers/test_openai_compat.py
import pytest
import json
from nexagen.providers.openai_compat import OpenAICompatProvider
from nexagen.providers.base import LLMProvider
from nexagen.models import ProviderConfig, NexagenMessage


def test_implements_protocol():
    config = ProviderConfig(backend="ollama", model="qwen3")
    provider = OpenAICompatProvider(config)
    assert isinstance(provider, LLMProvider)


def test_default_base_url_ollama():
    config = ProviderConfig(backend="ollama", model="qwen3")
    provider = OpenAICompatProvider(config)
    assert provider.base_url == "http://localhost:11434/v1"


def test_custom_base_url():
    config = ProviderConfig(backend="ollama", model="qwen3", base_url="http://192.168.1.5:11434")
    provider = OpenAICompatProvider(config)
    assert provider.base_url == "http://192.168.1.5:11434/v1"


def test_supports_tool_calling():
    config = ProviderConfig(backend="ollama", model="qwen3")
    provider = OpenAICompatProvider(config)
    assert provider.supports_tool_calling() is True


def test_supports_vision():
    config = ProviderConfig(backend="ollama", model="qwen3")
    provider = OpenAICompatProvider(config)
    assert provider.supports_vision() is False


def test_convert_messages():
    config = ProviderConfig(backend="ollama", model="qwen3")
    provider = OpenAICompatProvider(config)
    messages = [
        NexagenMessage(role="system", text="You are helpful"),
        NexagenMessage(role="user", text="Hello"),
    ]
    converted = provider._convert_messages(messages)
    assert converted[0] == {"role": "system", "content": "You are helpful"}
    assert converted[1] == {"role": "user", "content": "Hello"}


def test_convert_tool_result_message():
    config = ProviderConfig(backend="ollama", model="qwen3")
    provider = OpenAICompatProvider(config)
    msg = NexagenMessage(role="tool", text="file contents", tool_call_id="call_1")
    converted = provider._convert_messages([msg])
    assert converted[0]["role"] == "tool"
    assert converted[0]["tool_call_id"] == "call_1"


def test_parse_tool_calls_dict_args():
    """Ollama returns arguments as dict."""
    config = ProviderConfig(backend="ollama", model="qwen3")
    provider = OpenAICompatProvider(config)
    raw_tool_calls = [
        {"id": "1", "function": {"name": "read_file", "arguments": {"path": "main.py"}}}
    ]
    parsed = provider._parse_tool_calls(raw_tool_calls)
    assert parsed[0].name == "read_file"
    assert parsed[0].arguments == {"path": "main.py"}


def test_parse_tool_calls_string_args():
    """vLLM/OpenAI returns arguments as JSON string."""
    config = ProviderConfig(backend="vllm", model="mistral")
    provider = OpenAICompatProvider(config)
    raw_tool_calls = [
        {"id": "1", "function": {"name": "read_file", "arguments": '{"path": "main.py"}'}}
    ]
    parsed = provider._parse_tool_calls(raw_tool_calls)
    assert parsed[0].name == "read_file"
    assert parsed[0].arguments == {"path": "main.py"}
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_providers/test_openai_compat.py -v`
Expected: FAIL

**Step 3: Implement OpenAICompatProvider**

```python
# src/nexagen/providers/openai_compat.py
from __future__ import annotations
import json
import uuid
import httpx
from nexagen.models import (
    NexagenMessage,
    NexagenResponse,
    ProviderConfig,
    ToolCall,
)
from nexagen.providers.base import LLMProvider, ToolSchema
from nexagen.constants import DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS


_DEFAULT_URLS = {
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
    "lmstudio": "http://localhost:1234/v1",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
}


class OpenAICompatProvider:
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.model = config.model

        if config.base_url:
            base = config.base_url.rstrip("/")
            self.base_url = base if base.endswith("/v1") else f"{base}/v1"
        else:
            self.base_url = _DEFAULT_URLS.get(config.backend, "http://localhost:11434/v1")

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {config.api_key}"} if config.api_key else {},
            timeout=120.0,
        )

    def supports_tool_calling(self) -> bool:
        return True

    def supports_vision(self) -> bool:
        return False

    def _convert_messages(self, messages: list[NexagenMessage]) -> list[dict]:
        converted = []
        for msg in messages:
            if msg.role == "tool":
                converted.append({
                    "role": "tool",
                    "content": msg.text or "",
                    "tool_call_id": msg.tool_call_id or "",
                })
            elif msg.role == "assistant" and msg.tool_calls:
                entry = {
                    "role": "assistant",
                    "content": msg.text or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
                converted.append(entry)
            else:
                converted.append({
                    "role": msg.role,
                    "content": msg.text or "",
                })
        return converted

    def _convert_tools(self, tools: list[ToolSchema] | None) -> list[dict] | None:
        if not tools:
            return None
        return [
            {"type": "function", "function": tool}
            for tool in tools
        ]

    def _parse_tool_calls(self, raw_tool_calls: list[dict]) -> list[ToolCall]:
        parsed = []
        for tc in raw_tool_calls:
            func = tc.get("function", {})
            args = func.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            parsed.append(ToolCall(
                id=tc.get("id", str(uuid.uuid4())),
                name=func.get("name", ""),
                arguments=args,
            ))
        return parsed

    async def chat(
        self, messages: list[NexagenMessage], tools: list[ToolSchema] | None = None
    ) -> NexagenResponse:
        payload = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "temperature": self.config.temperature or DEFAULT_TEMPERATURE,
            "max_tokens": self.config.max_tokens or DEFAULT_MAX_TOKENS,
        }

        converted_tools = self._convert_tools(tools)
        if converted_tools:
            payload["tools"] = converted_tools

        response = await self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        message = choice["message"]

        tool_calls = None
        raw_tc = message.get("tool_calls")
        if raw_tc:
            tool_calls = self._parse_tool_calls(raw_tc)

        return NexagenResponse(
            message=NexagenMessage(
                role="assistant",
                text=message.get("content"),
                tool_calls=tool_calls,
            )
        )
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_providers/ -v`
Expected: All tests PASS (including registry tests from Task 3)

**Step 5: Commit**

```bash
git add src/nexagen/providers/openai_compat.py tests/test_providers/
git commit -m "feat: OpenAI-compatible provider — supports Ollama, vLLM, LM Studio, Groq"
```

---

### Task 5: Anthropic Provider

**Files:**
- Create: `src/nexagen/providers/anthropic.py`
- Create: `tests/test_providers/test_anthropic.py`

**Step 1: Write the failing tests**

```python
# tests/test_providers/test_anthropic.py
import pytest
import os
from nexagen.providers.anthropic import AnthropicProvider
from nexagen.providers.base import LLMProvider
from nexagen.models import ProviderConfig, NexagenMessage


def test_implements_protocol():
    config = ProviderConfig(backend="anthropic", model="claude-sonnet-4-20250514", api_key="test")
    provider = AnthropicProvider(config)
    assert isinstance(provider, LLMProvider)


def test_api_key_from_config():
    config = ProviderConfig(backend="anthropic", model="claude-sonnet-4-20250514", api_key="sk-test-123")
    provider = AnthropicProvider(config)
    assert provider.api_key == "sk-test-123"


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env-456")
    config = ProviderConfig(backend="anthropic", model="claude-sonnet-4-20250514")
    provider = AnthropicProvider(config)
    assert provider.api_key == "sk-env-456"


def test_no_api_key_raises():
    config = ProviderConfig(backend="anthropic", model="claude-sonnet-4-20250514")
    with pytest.raises(ValueError, match="API key"):
        AnthropicProvider(config)


def test_convert_messages():
    config = ProviderConfig(backend="anthropic", model="claude-sonnet-4-20250514", api_key="test")
    provider = AnthropicProvider(config)
    messages = [
        NexagenMessage(role="system", text="You are helpful"),
        NexagenMessage(role="user", text="Hello"),
    ]
    system, converted = provider._convert_messages(messages)
    assert system == "You are helpful"
    assert len(converted) == 1
    assert converted[0] == {"role": "user", "content": "Hello"}


def test_supports_tool_calling():
    config = ProviderConfig(backend="anthropic", model="claude-sonnet-4-20250514", api_key="test")
    provider = AnthropicProvider(config)
    assert provider.supports_tool_calling() is True
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_providers/test_anthropic.py -v`
Expected: FAIL

**Step 3: Implement AnthropicProvider**

```python
# src/nexagen/providers/anthropic.py
from __future__ import annotations
import os
import httpx
from nexagen.models import (
    NexagenMessage,
    NexagenResponse,
    ProviderConfig,
    ToolCall,
)
from nexagen.providers.base import LLMProvider, ToolSchema
from nexagen.constants import DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS


class AnthropicProvider:
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.model = config.model
        self.api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError(
                "API key required for Anthropic. Set ANTHROPIC_API_KEY env var "
                "or pass api_key in ProviderConfig."
            )

        self.client = httpx.AsyncClient(
            base_url="https://api.anthropic.com",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=120.0,
        )

    def supports_tool_calling(self) -> bool:
        return True

    def supports_vision(self) -> bool:
        return True

    def _convert_messages(self, messages: list[NexagenMessage]) -> tuple[str | None, list[dict]]:
        system = None
        converted = []
        for msg in messages:
            if msg.role == "system":
                system = msg.text
            elif msg.role == "tool":
                converted.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.text or "",
                            "is_error": msg.is_error,
                        }
                    ],
                })
            elif msg.role == "assistant" and msg.tool_calls:
                content = []
                if msg.text:
                    content.append({"type": "text", "text": msg.text})
                for tc in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                converted.append({"role": "assistant", "content": content})
            else:
                converted.append({"role": msg.role, "content": msg.text or ""})
        return system, converted

    def _convert_tools(self, tools: list[ToolSchema] | None) -> list[dict] | None:
        if not tools:
            return None
        return [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("parameters", {}),
            }
            for tool in tools
        ]

    async def chat(
        self, messages: list[NexagenMessage], tools: list[ToolSchema] | None = None
    ) -> NexagenResponse:
        system, converted = self._convert_messages(messages)

        payload = {
            "model": self.model,
            "messages": converted,
            "temperature": self.config.temperature or DEFAULT_TEMPERATURE,
            "max_tokens": self.config.max_tokens or DEFAULT_MAX_TOKENS,
        }

        if system:
            payload["system"] = system

        converted_tools = self._convert_tools(tools)
        if converted_tools:
            payload["tools"] = converted_tools

        response = await self.client.post("/v1/messages", json=payload)
        response.raise_for_status()
        data = response.json()

        text_parts = []
        tool_calls = []

        for block in data.get("content", []):
            if block["type"] == "text":
                text_parts.append(block["text"])
            elif block["type"] == "tool_use":
                tool_calls.append(ToolCall(
                    id=block["id"],
                    name=block["name"],
                    arguments=block.get("input", {}),
                ))

        return NexagenResponse(
            message=NexagenMessage(
                role="assistant",
                text="\n".join(text_parts) if text_parts else None,
                tool_calls=tool_calls if tool_calls else None,
            )
        )
```

**Step 4: Register Anthropic in registry**

Add to `registry.py` in `_register_defaults`:
```python
from nexagen.providers.anthropic import AnthropicProvider
self._backends["anthropic"] = AnthropicProvider
```

**Step 5: Run tests**

Run: `python -m pytest tests/test_providers/ -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/nexagen/providers/anthropic.py tests/test_providers/test_anthropic.py
git commit -m "feat: Anthropic native provider with Messages API"
```

---

### Task 6: OpenAI and Google Providers

**Files:**
- Create: `src/nexagen/providers/openai.py`
- Create: `src/nexagen/providers/google.py`
- Create: `tests/test_providers/test_openai.py`
- Create: `tests/test_providers/test_google.py`

Follow the same TDD pattern as Task 5:
1. Write tests for protocol compliance, API key handling, message conversion
2. Implement the provider (similar structure to AnthropicProvider)
3. Register in registry
4. Run all tests, commit

OpenAI provider wraps the native `openai` SDK.
Google provider wraps the native `google-genai` SDK.

Both follow the same Strategy pattern — implement `chat()` returning `NexagenResponse`.

**Step (final): Commit**

```bash
git commit -m "feat: OpenAI and Google native providers"
```

---

## Phase 3: Tool System

### Task 7: Tool Protocol & @tool Decorator

**Files:**
- Create: `src/nexagen/tools/__init__.py`
- Create: `src/nexagen/tools/base.py`
- Create: `tests/test_tools/__init__.py`
- Create: `tests/test_tools/test_base.py`

**Step 1: Write the failing tests**

```python
# tests/test_tools/test_base.py
import pytest
import asyncio
from pydantic import BaseModel
from nexagen.tools.base import tool, BaseTool


class GreetInput(BaseModel):
    name: str
    formal: bool = False


def test_tool_decorator_creates_tool():
    @tool("greet", "Greet a user", input_model=GreetInput)
    async def greet(args: GreetInput) -> str:
        return f"Hello, {args.name}!"

    assert isinstance(greet, BaseTool)
    assert greet.name == "greet"
    assert greet.description == "Greet a user"


def test_tool_has_json_schema():
    @tool("greet", "Greet a user", input_model=GreetInput)
    async def greet(args: GreetInput) -> str:
        return f"Hello, {args.name}!"

    schema = greet.input_schema
    assert "properties" in schema
    assert "name" in schema["properties"]


def test_tool_is_available():
    @tool("greet", "Greet a user", input_model=GreetInput)
    async def greet(args: GreetInput) -> str:
        return f"Hello, {args.name}!"

    assert greet.is_available() is True


def test_tool_execute():
    @tool("greet", "Greet a user", input_model=GreetInput)
    async def greet(args: GreetInput) -> str:
        return f"Hello, {args.name}!"

    result = asyncio.run(greet.execute({"name": "Alice"}))
    assert result.output == "Hello, Alice!"
    assert result.is_error is False


def test_tool_execute_validation_error():
    @tool("greet", "Greet a user", input_model=GreetInput)
    async def greet(args: GreetInput) -> str:
        return f"Hello, {args.name}!"

    result = asyncio.run(greet.execute({"wrong_field": "Alice"}))
    assert result.is_error is True
    assert "validation" in result.output.lower() or "name" in result.output.lower()


def test_tool_execute_runtime_error():
    @tool("fail", "Always fails", input_model=GreetInput)
    async def fail(args: GreetInput) -> str:
        raise RuntimeError("Something broke")

    result = asyncio.run(fail.execute({"name": "Alice"}))
    assert result.is_error is True
    assert "RuntimeError" in result.output


def test_tool_to_schema():
    @tool("greet", "Greet a user", input_model=GreetInput)
    async def greet(args: GreetInput) -> str:
        return f"Hello, {args.name}!"

    schema = greet.to_tool_schema()
    assert schema["name"] == "greet"
    assert schema["description"] == "Greet a user"
    assert "parameters" in schema
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tools/test_base.py -v`
Expected: FAIL

**Step 3: Implement base.py**

```python
# src/nexagen/tools/base.py
from __future__ import annotations
import traceback
from typing import Any, Callable, Awaitable
from pydantic import BaseModel, ValidationError
from nexagen.models import ToolResult


class BaseTool:
    def __init__(
        self,
        name: str,
        description: str,
        input_model: type[BaseModel],
        handler: Callable[[Any], Awaitable[str]],
    ):
        self.name = name
        self.description = description
        self._input_model = input_model
        self._handler = handler

    @property
    def input_schema(self) -> dict:
        return self._input_model.model_json_schema()

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    def is_available(self) -> bool:
        return True

    async def execute(self, args: dict) -> ToolResult:
        try:
            validated = self._input_model.model_validate(args)
        except ValidationError as e:
            return ToolResult(
                tool_call_id="",
                output=f"ValidationError: {e.errors()[0]['msg']}\n  in {self.name}",
                is_error=True,
            )

        try:
            result = await self._handler(validated)
            return ToolResult(tool_call_id="", output=str(result), is_error=False)
        except Exception as e:
            tb_lines = traceback.format_tb(e.__traceback__)
            short_tb = "".join(tb_lines[:2]).strip() if tb_lines else ""
            error_msg = f"{type(e).__name__}: {e}"
            if short_tb:
                error_msg += f"\n  {short_tb.splitlines()[-1].strip()}"
            return ToolResult(tool_call_id="", output=error_msg, is_error=True)

    def to_tool_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
        }


def tool(name: str, description: str, input_model: type[BaseModel]):
    def decorator(func: Callable[[Any], Awaitable[str]]) -> BaseTool:
        return BaseTool(
            name=name,
            description=description,
            input_model=input_model,
            handler=func,
        )
    return decorator
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_tools/test_base.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/nexagen/tools/ tests/test_tools/
git commit -m "feat: tool protocol with @tool decorator and Pydantic validation"
```

---

### Task 8: Tool Registry

**Files:**
- Create: `src/nexagen/tools/registry.py`
- Create: `tests/test_tools/test_registry.py`

**Step 1: Write the failing tests**

```python
# tests/test_tools/test_registry.py
import pytest
import asyncio
from pydantic import BaseModel
from nexagen.tools.registry import ToolRegistry
from nexagen.tools.base import tool, BaseTool


class DummyInput(BaseModel):
    text: str


@tool("dummy", "A dummy tool", input_model=DummyInput)
async def dummy_tool(args: DummyInput) -> str:
    return args.text


def test_register_and_get():
    registry = ToolRegistry()
    registry.register(dummy_tool)
    assert registry.get("dummy") is dummy_tool


def test_get_unknown_returns_none():
    registry = ToolRegistry()
    assert registry.get("nonexistent") is None


def test_list_available():
    registry = ToolRegistry()
    registry.register(dummy_tool)
    available = registry.list_available()
    assert len(available) == 1
    assert available[0].name == "dummy"


def test_get_schemas():
    registry = ToolRegistry()
    registry.register(dummy_tool)
    schemas = registry.get_tool_schemas()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "dummy"


def test_register_multiple():
    @tool("other", "Another tool", input_model=DummyInput)
    async def other_tool(args: DummyInput) -> str:
        return "other"

    registry = ToolRegistry()
    registry.register(dummy_tool)
    registry.register(other_tool)
    assert len(registry.list_available()) == 2
```

**Step 2: Run tests — FAIL**

**Step 3: Implement registry.py**

```python
# src/nexagen/tools/registry.py
from __future__ import annotations
from nexagen.tools.base import BaseTool


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_available(self) -> list[BaseTool]:
        return [t for t in self._tools.values() if t.is_available()]

    def get_tool_schemas(self) -> list[dict]:
        return [t.to_tool_schema() for t in self.list_available()]
```

**Step 4: Run tests — all PASS**

**Step 5: Commit**

```bash
git add src/nexagen/tools/registry.py tests/test_tools/test_registry.py
git commit -m "feat: tool registry with availability filtering"
```

---

### Task 9: Built-in Tools (file_read, file_write, file_edit, bash, grep, glob)

**Files:**
- Create: `src/nexagen/tools/builtin/__init__.py`
- Create: `src/nexagen/tools/builtin/file_read.py`
- Create: `src/nexagen/tools/builtin/file_write.py`
- Create: `src/nexagen/tools/builtin/file_edit.py`
- Create: `src/nexagen/tools/builtin/bash.py`
- Create: `src/nexagen/tools/builtin/grep.py`
- Create: `src/nexagen/tools/builtin/glob_tool.py`
- Create: `tests/test_tools/test_builtin.py`

For each tool, follow TDD:
1. Write test for the tool's core behavior
2. Implement using `@tool` decorator with Pydantic input model
3. Run tests, pass, commit

**Example — file_read:**

```python
# src/nexagen/tools/builtin/file_read.py
from pydantic import BaseModel
from nexagen.tools.base import tool


class FileReadInput(BaseModel):
    file_path: str
    offset: int | None = None
    limit: int | None = None


@tool("file_read", "Read contents of a file", input_model=FileReadInput)
async def file_read(args: FileReadInput) -> str:
    with open(args.file_path, "r") as f:
        lines = f.readlines()

    start = (args.offset or 1) - 1
    end = start + args.limit if args.limit else len(lines)
    selected = lines[start:end]

    numbered = [f"{i + start + 1:>4} | {line.rstrip()}" for i, line in enumerate(selected)]
    return "\n".join(numbered)
```

**Example — bash:**

```python
# src/nexagen/tools/builtin/bash.py
import asyncio
from pydantic import BaseModel, Field
from nexagen.tools.base import tool


class BashInput(BaseModel):
    command: str
    timeout: int = Field(default=120, description="Timeout in seconds")


@tool("bash", "Execute a shell command", input_model=BashInput)
async def bash(args: BashInput) -> str:
    proc = await asyncio.create_subprocess_shell(
        args.command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=args.timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return f"Error: Command timed out after {args.timeout}s"

    output = stdout.decode() if stdout else ""
    errors = stderr.decode() if stderr else ""

    if proc.returncode != 0:
        return f"Exit code: {proc.returncode}\n{errors}\n{output}".strip()
    return output.strip()
```

**Example — __init__.py to collect all built-ins:**

```python
# src/nexagen/tools/builtin/__init__.py
from nexagen.tools.builtin.file_read import file_read
from nexagen.tools.builtin.file_write import file_write
from nexagen.tools.builtin.file_edit import file_edit
from nexagen.tools.builtin.bash import bash
from nexagen.tools.builtin.grep import grep
from nexagen.tools.builtin.glob_tool import glob_tool

BUILTIN_TOOLS = {
    "file_read": file_read,
    "file_write": file_write,
    "file_edit": file_edit,
    "bash": bash,
    "grep": grep,
    "glob": glob_tool,
}
```

**Commit after all built-ins pass:**

```bash
git commit -m "feat: built-in tools — file_read, file_write, file_edit, bash, grep, glob"
```

---

## Phase 4: Agent Engine

### Task 10: Permission System

**Files:**
- Create: `src/nexagen/permissions.py`
- Create: `tests/test_permissions.py`

**Step 1: Write failing tests**

```python
# tests/test_permissions.py
import pytest
import asyncio
from nexagen.permissions import PermissionManager, Allow, Deny


def test_readonly_mode():
    pm = PermissionManager(mode="readonly")
    assert asyncio.run(pm.check("file_read", {})) == Allow()
    assert asyncio.run(pm.check("grep", {})) == Allow()
    result = asyncio.run(pm.check("bash", {}))
    assert isinstance(result, Deny)


def test_safe_mode():
    pm = PermissionManager(mode="safe")
    assert asyncio.run(pm.check("file_read", {})) == Allow()
    assert asyncio.run(pm.check("file_write", {})) == Allow()
    result = asyncio.run(pm.check("bash", {}))
    assert isinstance(result, Deny)


def test_full_mode():
    pm = PermissionManager(mode="full")
    assert asyncio.run(pm.check("bash", {})) == Allow()
    assert asyncio.run(pm.check("file_write", {})) == Allow()


def test_allowlist_narrows():
    pm = PermissionManager(mode="full", allowed_tools=["file_read"])
    assert asyncio.run(pm.check("file_read", {})) == Allow()
    result = asyncio.run(pm.check("bash", {}))
    assert isinstance(result, Deny)


def test_callback_overrides():
    async def block_rm(tool_name, args):
        if tool_name == "bash" and "rm" in args.get("command", ""):
            return Deny("Destructive command")
        return Allow()

    pm = PermissionManager(mode="full", can_use_tool=block_rm)
    assert asyncio.run(pm.check("bash", {"command": "ls"})) == Allow()
    result = asyncio.run(pm.check("bash", {"command": "rm -rf /"}))
    assert isinstance(result, Deny)
```

**Step 2: Run — FAIL**

**Step 3: Implement**

```python
# src/nexagen/permissions.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Awaitable


@dataclass
class Allow:
    pass


@dataclass
class Deny:
    message: str = "Permission denied"


_MODE_TOOLS = {
    "readonly": {"file_read", "grep", "glob"},
    "safe": {"file_read", "file_write", "file_edit", "grep", "glob"},
    "full": None,  # None means all tools allowed
}


class PermissionManager:
    def __init__(
        self,
        mode: str = "safe",
        allowed_tools: list[str] | None = None,
        can_use_tool: Callable[[str, dict], Awaitable[Allow | Deny]] | None = None,
    ):
        self.mode = mode
        self.allowed_tools = set(allowed_tools) if allowed_tools else None
        self.can_use_tool = can_use_tool

    async def check(self, tool_name: str, args: dict) -> Allow | Deny:
        # Layer 1: mode
        mode_tools = _MODE_TOOLS.get(self.mode)
        if mode_tools is not None and tool_name not in mode_tools:
            return Deny(f"Tool '{tool_name}' not allowed in '{self.mode}' mode")

        # Layer 2: allowlist
        if self.allowed_tools is not None and tool_name not in self.allowed_tools:
            return Deny(f"Tool '{tool_name}' not in allowed_tools")

        # Layer 3: callback
        if self.can_use_tool:
            return await self.can_use_tool(tool_name, args)

        return Allow()
```

**Step 4: Run — PASS**

**Step 5: Commit**

```bash
git commit -m "feat: three-layer permission system — modes, allowlist, callback"
```

---

### Task 11: Conversation Manager

**Files:**
- Create: `src/nexagen/conversation.py`
- Create: `tests/test_conversation.py`

Implement:
- Message list management
- Token estimation (chars / 4)
- Threshold check (80%)
- Task summary storage
- Method to get compressible messages (everything except first user msg + last 3)

Follow TDD. Commit.

```bash
git commit -m "feat: conversation manager with token tracking and task summaries"
```

---

### Task 12: Supervisor Agent

**Files:**
- Create: `src/nexagen/supervisor/__init__.py`
- Create: `src/nexagen/supervisor/supervisor.py`
- Create: `tests/test_supervisor.py`

Implement:
- `SupervisorAgent` class takes a provider config (small model)
- `check_progress(original_task, action_log) -> "continue" | "stop"`
- `compress_history(messages) -> list[NexagenMessage]`
- Structured output with text fallback parsing
- Action log formatting (summary + tool names per step)

Follow TDD. Commit.

```bash
git commit -m "feat: supervisor agent — progress checking and context compression"
```

---

### Task 13: Agent Loop

**Files:**
- Create: `src/nexagen/agent.py`
- Create: `tests/test_agent.py`

This is the core. Implement the `Agent` class:

```python
class Agent:
    def __init__(
        self,
        provider: str | ProviderConfig,
        tools: list[str] | None = None,
        custom_tools: list[BaseTool] | None = None,
        mcp_servers: dict | None = None,
        system_prompt: str | None = None,
        permission_mode: str = "safe",
        allowed_tools: list[str] | None = None,
        can_use_tool: Callable | None = None,
        supervisor_provider: str | ProviderConfig | None = None,
    ): ...

    async def run(self, prompt: str, conversation: Conversation | None = None) -> AsyncIterator[NexagenMessage]:
        """Main agent loop."""
        ...
```

The agent loop:
1. Build messages (system prompt + conversation history + user prompt)
2. Get available tools from registry
3. Call provider.chat()
4. If no tool calls → yield final message, summarize task, done
5. If tool calls → for each tool call sequentially:
   a. Check permissions
   b. Execute tool
   c. Track errors (3 consecutive → supervisor)
   d. Append results to conversation
6. Every N calls → supervisor check (continue/stop)
7. At 80% context → supervisor compress
8. Loop back to step 3

Tests should use a mock provider that returns predefined responses.

Follow TDD. Commit.

```bash
git commit -m "feat: agent loop with supervisor, permissions, and context management"
```

---

## Phase 5: MCP Integration

### Task 14: MCP Tool Wrapper

**Files:**
- Create: `src/nexagen/tools/mcp.py`
- Create: `tests/test_tools/test_mcp.py`

Implement:
- `MCPServer` config class (command, args, env)
- `MCPTool` class implementing Tool protocol with lifecycle (connect/disconnect/is_available)
- `MCPManager` that starts MCP servers, discovers tools, wraps them as `MCPTool` instances
- Uses the `mcp` Python SDK as client

Follow TDD. Commit.

```bash
git commit -m "feat: MCP server integration with lifecycle management"
```

---

## Phase 6: Logging

### Task 15: Structured Logging

**Files:**
- Create: `src/nexagen/logging.py`
- Create: `tests/test_logging.py`

Implement:
- JSON formatter for Python logging
- Helper functions: `log_tool_call()`, `log_tool_result()`, `log_supervisor_decision()`, `log_error()`
- Each entry includes: timestamp, level, event, cycle number, relevant data

Follow TDD. Commit.

```bash
git commit -m "feat: structured JSON logging for agent events"
```

---

## Phase 7: CLI & TUI

### Task 16: CLI

**Files:**
- Create: `src/nexagen/cli/__init__.py`
- Create: `src/nexagen/cli/app.py`

Implement using click + rich:
- `nexagen run "prompt"` — run a single prompt
- `nexagen chat` — enter interactive mode
- `--provider` flag (default from constants)
- `--tools` flag (comma-separated)
- `--permission-mode` flag

Follow TDD. Commit.

```bash
git commit -m "feat: CLI with run and chat commands"
```

---

### Task 17: TUI

**Files:**
- Create: `src/nexagen/tui/__init__.py`
- Create: `src/nexagen/tui/app.py`

Implement using Textual:
- Input area for user messages
- Output area showing agent responses
- Step-by-step progress display (summary + spinner per cycle)
- Status bar with provider info, cycle count, token usage

Follow TDD. Commit.

```bash
git commit -m "feat: interactive TUI with progress display"
```

---

## Phase 8: Public API & Packaging

### Task 18: Public API Exports

**Files:**
- Modify: `src/nexagen/__init__.py`

Export all public API:

```python
from nexagen.agent import Agent
from nexagen.models import ProviderConfig, NexagenMessage, NexagenResponse, ToolCall, ToolResult
from nexagen.tools.base import tool, BaseTool
from nexagen.tools.registry import ToolRegistry
from nexagen.tools.mcp import MCPServer
from nexagen.conversation import Conversation
from nexagen.permissions import Allow, Deny
```

**Step: Commit**

```bash
git commit -m "feat: public API exports"
```

---

### Task 19: Integration Tests

**Files:**
- Create: `tests/test_integration.py`

Write end-to-end tests using a mock provider:
1. Agent runs a prompt with file_read tool → reads a temp file → returns answer
2. Agent with permission_mode="readonly" blocks bash
3. Agent with custom @tool executes it correctly
4. Conversation carries task summaries across two runs
5. Agent stops when supervisor says stop (mock supervisor)

```bash
git commit -m "test: integration tests for full agent loop"
```

---

## Task Dependency Graph

```
Task 1 (scaffolding)
  └→ Task 2 (models)
       ├→ Task 3 (provider registry)
       │    └→ Task 4 (OpenAI-compat provider)
       │    └→ Task 5 (Anthropic provider)
       │    └→ Task 6 (OpenAI + Google providers)
       ├→ Task 7 (tool base + decorator)
       │    └→ Task 8 (tool registry)
       │    └→ Task 9 (built-in tools)
       │    └→ Task 14 (MCP integration)
       ├→ Task 10 (permissions)
       ├→ Task 11 (conversation)
       └→ Task 12 (supervisor)
            └→ Task 13 (agent loop) ← depends on 4,8,9,10,11,12
                 └→ Task 15 (logging)
                 └→ Task 16 (CLI)
                 └→ Task 17 (TUI)
                 └→ Task 18 (public API)
                 └→ Task 19 (integration tests)
```

**Parallelizable groups:**
- Tasks 4, 5, 6 (providers — independent of each other)
- Tasks 7-9 and 10-12 (tools and engine components — independent groups)
- Tasks 15, 16, 17 (post-agent-loop — independent of each other)
