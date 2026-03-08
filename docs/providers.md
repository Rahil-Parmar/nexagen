# Provider Configuration

nexagen supports multiple LLM backends through a unified provider interface. Every provider implements the same `LLMProvider` protocol, so switching between backends requires only changing the provider string.

---

## String Shorthand Format

The fastest way to specify a provider:

```
backend/model
```

Examples:

```python
"ollama/qwen3"
"openai/gpt-4o"
"anthropic/claude-sonnet-4-20250514"
"google/gemini-2.0-flash"
"groq/llama-3.3-70b-versatile"
"together/meta-llama/Llama-3.3-70B-Instruct-Turbo"
```

### Custom host

Use `@` to specify a custom host:

```
backend/model@host:port
```

Examples:

```python
"ollama/qwen3@192.168.1.5:11434"
"vllm/mistral@10.0.0.1:8000"
"ollama/llama3@https://my-ollama.example.com"
```

If the host does not start with `http`, nexagen prepends `http://` automatically.

---

## ProviderConfig for Full Control

For fine-grained configuration, use `ProviderConfig`:

```python
from nexagen import Agent, ProviderConfig

config = ProviderConfig(
    backend="ollama",
    model="qwen3",
    base_url="http://192.168.1.5:11434/v1",
    api_key=None,            # Not needed for Ollama
    temperature=0.3,
    max_tokens=8192,
)

agent = Agent(provider=config)
```

### ProviderConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backend` | `str` | required | Backend name: `ollama`, `vllm`, `lmstudio`, `openai`, `anthropic`, `google`, `groq`, `together` |
| `model` | `str` | required | Model name as expected by the backend |
| `base_url` | `str \| None` | `None` | Override the default API URL |
| `api_key` | `str \| None` | `None` | API key (falls back to env vars for cloud providers) |
| `temperature` | `float \| None` | `None` | Sampling temperature |
| `max_tokens` | `int \| None` | `None` | Maximum output tokens |

---

## Provider Details

### Ollama (local)

- **Backend name:** `ollama`
- **Default URL:** `http://localhost:11434/v1`
- **API key:** Not required
- **Protocol:** OpenAI-compatible
- **Tool calling:** Yes (model-dependent)
- **Vision:** No

```python
agent = Agent(provider="ollama/qwen3")
```

### vLLM (local/server)

- **Backend name:** `vllm`
- **Default URL:** `http://localhost:8000/v1`
- **API key:** Not required
- **Protocol:** OpenAI-compatible
- **Tool calling:** Yes (model-dependent)
- **Vision:** No

```python
agent = Agent(provider="vllm/mistral-7b")
```

### LM Studio (local)

- **Backend name:** `lmstudio`
- **Default URL:** `http://localhost:1234/v1`
- **API key:** Not required
- **Protocol:** OpenAI-compatible
- **Tool calling:** Yes (model-dependent)
- **Vision:** No

```python
agent = Agent(provider="lmstudio/my-model")
```

### OpenAI (cloud)

- **Backend name:** `openai`
- **Default URL:** `https://api.openai.com/v1`
- **API key:** Required (`OPENAI_API_KEY` env var or `ProviderConfig.api_key`)
- **Protocol:** Native OpenAI API
- **Tool calling:** Yes
- **Vision:** Yes

```python
agent = Agent(provider="openai/gpt-4o")
```

### Anthropic (cloud)

- **Backend name:** `anthropic`
- **Default URL:** `https://api.anthropic.com`
- **API key:** Required (`ANTHROPIC_API_KEY` env var or `ProviderConfig.api_key`)
- **Protocol:** Anthropic Messages API
- **Tool calling:** Yes
- **Vision:** Yes

```python
agent = Agent(provider="anthropic/claude-sonnet-4-20250514")
```

### Google Gemini (cloud)

- **Backend name:** `google`
- **Default URL:** `https://generativelanguage.googleapis.com/v1beta`
- **API key:** Required (`GOOGLE_API_KEY` env var or `ProviderConfig.api_key`)
- **Protocol:** Google Generative Language API
- **Tool calling:** Yes
- **Vision:** Yes

```python
agent = Agent(provider="google/gemini-2.0-flash")
```

### Groq (cloud, fast inference)

- **Backend name:** `groq`
- **Default URL:** `https://api.groq.com/openai/v1`
- **API key:** Required (via `ProviderConfig.api_key`)
- **Protocol:** OpenAI-compatible
- **Tool calling:** Yes
- **Vision:** No

```python
from nexagen import Agent, ProviderConfig

agent = Agent(provider=ProviderConfig(
    backend="groq",
    model="llama-3.3-70b-versatile",
    api_key="gsk_...",
))
```

### Together AI (cloud)

- **Backend name:** `together`
- **Default URL:** `https://api.together.xyz/v1`
- **API key:** Required (via `ProviderConfig.api_key`)
- **Protocol:** OpenAI-compatible
- **Tool calling:** Yes
- **Vision:** No

```python
from nexagen import Agent, ProviderConfig

agent = Agent(provider=ProviderConfig(
    backend="together",
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    api_key="...",
))
```

---

## Provider Comparison

| Provider | Local | API Key | Tool Calling | Vision | Protocol |
|----------|-------|---------|--------------|--------|----------|
| Ollama | Yes | No | Yes* | No | OpenAI-compat |
| vLLM | Yes | No | Yes* | No | OpenAI-compat |
| LM Studio | Yes | No | Yes* | No | OpenAI-compat |
| OpenAI | No | Yes | Yes | Yes | Native |
| Anthropic | No | Yes | Yes | Yes | Native |
| Google | No | Yes | Yes | Yes | Native |
| Groq | No | Yes | Yes | No | OpenAI-compat |
| Together | No | Yes | Yes | No | OpenAI-compat |

*Tool calling support depends on the specific model.

---

## Environment Variables

| Variable | Provider |
|----------|----------|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `GOOGLE_API_KEY` | Google Gemini |

For Groq and Together, pass the API key directly via `ProviderConfig.api_key`.

---

## Passing a Custom Provider

You can also pass a pre-instantiated provider object that implements the `LLMProvider` protocol:

```python
from nexagen.providers.openai_compat import OpenAICompatProvider
from nexagen import Agent, ProviderConfig

provider = OpenAICompatProvider(ProviderConfig(
    backend="ollama",
    model="qwen3",
))

agent = Agent(provider=provider)
```

This is useful when you need to share a provider instance across multiple agents or customize behavior beyond what `ProviderConfig` offers.
