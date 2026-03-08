import asyncio
import glob as globlib
import os
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field
from nexagen.tools.base import tool
from nexagen.tools.builtin.path_security import validate_path

_MAX_RESULTS = 500
_GLOB_TIMEOUT = 10  # seconds — prevent DoS from expensive patterns
_MAX_PATTERN_DEPTH = 10  # max ** segments in a pattern
_thread_pool = ThreadPoolExecutor(max_workers=1)


def _run_glob(search_pattern: str) -> list[str]:
    """Run glob in a thread so we can apply a timeout."""
    return sorted(globlib.glob(search_pattern, recursive=True))


class GlobInput(BaseModel):
    pattern: str = Field(description="Glob pattern like '**/*.py'")
    path: str = Field(default=".", description="Root directory to search from")


@tool("glob", "Find files matching a glob pattern", input_model=GlobInput)
async def glob_tool(args: GlobInput) -> str:
    # Validate base path
    safe_path = validate_path(args.path)

    # Block path traversal in pattern
    if ".." in args.pattern:
        return "Error: glob pattern must not contain '..'"

    # Block patterns with excessive ** depth (DoS prevention)
    double_star_count = args.pattern.count("**")
    if double_star_count > _MAX_PATTERN_DEPTH:
        return f"Error: glob pattern has too many '**' segments (max {_MAX_PATTERN_DEPTH})"

    # Block excessively long patterns
    if len(args.pattern) > 500:
        return "Error: glob pattern too long (max 500 characters)"

    search_pattern = os.path.join(safe_path, args.pattern)

    # Run glob with a timeout to prevent DoS
    loop = asyncio.get_event_loop()
    try:
        matches = await asyncio.wait_for(
            loop.run_in_executor(_thread_pool, _run_glob, search_pattern),
            timeout=_GLOB_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return f"Error: glob search timed out after {_GLOB_TIMEOUT}s (pattern may be too broad)"

    if not matches:
        return f"No files matching '{args.pattern}'"

    # Filter results to only safe paths (glob might follow symlinks)
    from nexagen.tools.builtin.path_security import _is_safe_path
    safe_matches = [m for m in matches if _is_safe_path(str(os.path.realpath(m)))]

    # Limit results to prevent memory exhaustion
    if len(safe_matches) > _MAX_RESULTS:
        safe_matches = safe_matches[:_MAX_RESULTS]
        return "\n".join(safe_matches) + f"\n... ({_MAX_RESULTS} results shown, more available)"

    return "\n".join(safe_matches)
