import os
import re
import signal
import threading

from pydantic import BaseModel, Field
from nexagen.tools.base import tool
from nexagen.tools.builtin.path_security import validate_path, _is_safe_path

_MAX_FILES = 1000
_MAX_LINE_LENGTH = 10_000
_REGEX_TIMEOUT = 5  # seconds per file


class _RegexTimeoutError(Exception):
    pass


def _compile_regex_safe(pattern: str) -> re.Pattern:
    """Compile a regex with ReDoS protection."""
    if len(pattern) > 500:
        raise ValueError("Regex pattern too long (max 500 characters)")

    # Detect common ReDoS patterns: nested quantifiers like (a+)+, (.*)*
    redos_patterns = [
        r'\([^)]*[+*][^)]*\)[+*]',  # (x+)+ or (x*)*
        r'\([^)]*\|[^)]*\)[+*]{2,}',  # (a|b)++ etc.
        r'(\.\*){3,}',  # excessive .* chains
        r'([+*?]){2,}',  # consecutive quantifiers
    ]
    for rdp in redos_patterns:
        if re.search(rdp, pattern):
            raise ValueError(f"Potentially unsafe regex pattern (ReDoS risk)")

    return re.compile(pattern)


def _regex_search_with_timeout(regex: re.Pattern, text: str, timeout: float = 2.0):
    """Run a regex search with a timeout to prevent ReDoS."""
    result = [None]
    exception = [None]

    def search():
        try:
            result[0] = regex.search(text)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=search, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        # Thread is still running — regex is catastrophically backtracking
        raise _RegexTimeoutError(f"Regex search timed out after {timeout}s")

    if exception[0]:
        raise exception[0]

    return result[0]


class GrepInput(BaseModel):
    pattern: str = Field(description="Regex pattern to search for")
    path: str = Field(description="File or directory to search in")
    max_results: int = Field(default=50, description="Maximum number of matches", le=500)


@tool("grep", "Search file contents using regex", input_model=GrepInput)
async def grep_tool(args: GrepInput) -> str:
    safe_path = validate_path(args.path)

    try:
        regex = _compile_regex_safe(args.pattern)
    except (re.error, ValueError) as e:
        return f"Error: Invalid regex pattern: {e}"

    results = []
    timed_out_files = 0

    if os.path.isfile(safe_path):
        files = [safe_path]
    elif os.path.isdir(safe_path):
        files = []
        for root, _, filenames in os.walk(safe_path):
            # Validate each directory is still in safe territory
            if not _is_safe_path(str(os.path.realpath(root))):
                continue
            for f in filenames:
                files.append(os.path.join(root, f))
                if len(files) >= _MAX_FILES:
                    break
            if len(files) >= _MAX_FILES:
                break
    else:
        return f"Error: {args.path} not found"

    for filepath in files:
        # Skip symlinks pointing outside safe areas
        real_path = str(os.path.realpath(filepath))
        if not _is_safe_path(real_path):
            continue

        try:
            with open(filepath, "r", errors="ignore") as f:
                for i, line in enumerate(f, 1):
                    # Truncate very long lines to prevent ReDoS on content
                    search_line = line[:_MAX_LINE_LENGTH]
                    try:
                        if _regex_search_with_timeout(regex, search_line, timeout=2.0):
                            results.append(f"{filepath}:{i}: {line.rstrip()[:200]}")
                            if len(results) >= args.max_results:
                                break
                    except _RegexTimeoutError:
                        timed_out_files += 1
                        break  # Skip this file
        except (OSError, UnicodeDecodeError):
            continue
        if len(results) >= args.max_results:
            break

    if not results:
        msg = f"No matches found for '{args.pattern}'"
        if timed_out_files > 0:
            msg += f" ({timed_out_files} files skipped due to regex timeout)"
        return msg

    output = "\n".join(results)
    if timed_out_files > 0:
        output += f"\n({timed_out_files} files skipped due to regex timeout)"
    return output
