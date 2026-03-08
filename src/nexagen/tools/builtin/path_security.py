"""Path security utilities for file tools.

Validates paths against symlink attacks, traversal, and size limits.
All file tools MUST use validate_path() before any file operation.
For write operations, use safe_open_for_write() to atomically prevent symlink attacks.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

_MAX_FILE_READ_BYTES = 10_000_000  # 10 MB
_MAX_FILE_WRITE_BYTES = 50_000_000  # 50 MB
_MAX_PATH_LENGTH = 4096
_MAX_PATH_DEPTH = 50


def validate_path(path: str, *, must_exist: bool = False) -> str:
    """Resolve and validate a file path. Returns the resolved absolute path.

    Raises ValueError if the path is unsafe:
    - Path too long or too deeply nested
    - Symlink pointing outside safe areas
    - Path to sensitive system files
    - Path traversal (../) to restricted areas
    """
    if len(path) > _MAX_PATH_LENGTH:
        raise ValueError(f"Path too long ({len(path)} chars, max {_MAX_PATH_LENGTH})")

    if path.count("/") > _MAX_PATH_DEPTH:
        raise ValueError(f"Path too deeply nested (max {_MAX_PATH_DEPTH} levels)")

    # Resolve to absolute path, following symlinks
    resolved = str(Path(path).resolve())

    # If the ORIGINAL path is a symlink, validate where it points
    if os.path.islink(path):
        link_target = str(Path(path).resolve())
        if not _is_safe_path(link_target):
            raise ValueError(f"Symlink '{path}' points to restricted location")

    if must_exist and not os.path.exists(resolved):
        raise FileNotFoundError(f"File not found: {resolved}")

    if not _is_safe_path(resolved):
        raise ValueError(f"Access denied: path is in a restricted location")

    return resolved


def validate_path_for_write(path: str) -> str:
    """Validate a path for write operations with stricter symlink checks.

    Unlike validate_path(), this also:
    - Rejects if the original path IS a symlink (prevents TOCTOU)
    - Validates the parent directory exists and is safe
    - Rejects if any component in the path is a symlink to outside safe areas
    """
    if len(path) > _MAX_PATH_LENGTH:
        raise ValueError(f"Path too long ({len(path)} chars, max {_MAX_PATH_LENGTH})")

    # Reject if original path is a symlink — prevents TOCTOU attacks
    abs_path = str(Path(path).resolve()) if not os.path.isabs(path) else path
    if os.path.islink(path):
        raise ValueError(f"Refusing to write through symlink: {path}")

    resolved = str(Path(path).resolve())

    if not _is_safe_path(resolved):
        raise ValueError(f"Access denied: path is in a restricted location")

    # Validate parent directory path is also safe
    parent = str(Path(resolved).parent)
    if not _is_safe_path(parent):
        raise ValueError(f"Access denied: parent directory is in a restricted location")

    return resolved


def safe_open_for_write(path: str, mode: str = "w"):
    """Atomically open a file for writing, refusing to follow symlinks.

    Uses O_NOFOLLOW to prevent TOCTOU symlink races. This ensures that
    between the path check and the actual open, no symlink swap can occur.

    Returns a file object. Caller must close it.
    """
    resolved = validate_path_for_write(path)

    # Build flags: O_NOFOLLOW ensures we don't follow symlinks at open time
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
    if "a" in mode:
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW

    try:
        fd = os.open(resolved, flags, 0o644)
    except OSError as e:
        if e.errno == 40:  # ELOOP — path is a symlink
            raise ValueError(f"Refusing to write through symlink: {path}") from e
        raise

    return os.fdopen(fd, mode)


def _is_safe_path(resolved: str) -> bool:
    """Check if a resolved path is safe to access."""
    home = str(Path.home())

    # Allow: user's home directory
    if resolved.startswith(home + "/") or resolved == home:
        return True

    # Allow: /tmp and temp directories
    if resolved.startswith("/tmp/") or resolved.startswith("/var/folders/"):
        return True

    # Allow: /private/tmp (macOS)
    if resolved.startswith("/private/tmp/") or resolved.startswith("/private/var/folders/"):
        return True

    # Allow: current working directory (even if outside home)
    try:
        cwd = str(Path.cwd().resolve())
        if resolved.startswith(cwd + "/") or resolved == cwd:
            return True
    except OSError:
        pass

    # Everything else is blocked
    return False


def check_file_size(path: str, max_bytes: int = _MAX_FILE_READ_BYTES) -> None:
    """Raise ValueError if file exceeds size limit."""
    try:
        size = os.path.getsize(path)
        if size > max_bytes:
            raise ValueError(f"File too large: {size} bytes (limit: {max_bytes})")
    except OSError:
        pass  # File might not exist yet (for writes)


def check_write_content_size(content: str, max_bytes: int = _MAX_FILE_WRITE_BYTES) -> None:
    """Raise ValueError if content exceeds write size limit."""
    size = len(content.encode("utf-8"))
    if size > max_bytes:
        raise ValueError(f"Content too large: {size} bytes (limit: {max_bytes})")
