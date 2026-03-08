DEFAULT_MODEL = "ollama/qwen3"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096
DEFAULT_CONTEXT_THRESHOLD = 0.80
DEFAULT_COMPRESS_TARGET = 0.50
DEFAULT_MAX_TOOL_ERRORS = 3
DEFAULT_SUPERVISOR_MODEL = "ollama/phi3"
DEFAULT_SUPERVISOR_CHECK_INTERVAL = 5
DEFAULT_PERMISSION_MODE = "safe"
DEFAULT_MAX_ITERATIONS = 100  # max agent loop iterations to prevent runaway loops
CHARS_PER_TOKEN = 4

OPENAI_COMPAT_DEFAULT_URLS: dict[str, str] = {
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
    "lmstudio": "http://localhost:1234/v1",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
}
