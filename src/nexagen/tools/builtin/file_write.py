import os

from pydantic import BaseModel
from nexagen.tools.base import tool
from nexagen.tools.builtin.path_security import (
    validate_path_for_write,
    check_write_content_size,
    safe_open_for_write,
    _is_safe_path,
)


class FileWriteInput(BaseModel):
    file_path: str
    content: str


@tool("file_write", "Create or overwrite a file with content", input_model=FileWriteInput)
async def file_write(args: FileWriteInput) -> str:
    # Validate path with strict write-mode checks (rejects symlinks, validates parent)
    safe_path = validate_path_for_write(args.file_path)
    check_write_content_size(args.content)

    # Validate that parent directory creation stays within safe boundaries
    parent_dir = os.path.dirname(safe_path) or "."
    parent_resolved = str(os.path.realpath(parent_dir))
    if os.path.exists(parent_dir) and not _is_safe_path(parent_resolved):
        return f"Error: parent directory is in a restricted location"

    os.makedirs(os.path.dirname(safe_path) or ".", exist_ok=True)

    # Use safe_open_for_write with O_NOFOLLOW to prevent TOCTOU symlink attacks
    try:
        with safe_open_for_write(safe_path) as f:
            f.write(args.content)
    except ValueError as e:
        return f"Error: {e}"

    return f"File written: {args.file_path}"
