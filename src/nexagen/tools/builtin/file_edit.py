import os

from pydantic import BaseModel, Field
from nexagen.tools.base import tool
from nexagen.tools.builtin.path_security import (
    validate_path_for_write,
    check_file_size,
    check_write_content_size,
    safe_open_for_write,
)


class FileEditInput(BaseModel):
    file_path: str
    old_string: str = Field(description="Exact string to find and replace")
    new_string: str = Field(description="Replacement string")


@tool("file_edit", "Edit a file by replacing an exact string match", input_model=FileEditInput)
async def file_edit(args: FileEditInput) -> str:
    # Use strict write-mode validation (rejects symlinks, validates parent)
    safe_path = validate_path_for_write(args.file_path)

    if not os.path.exists(safe_path):
        return f"Error: file not found: {args.file_path}"

    check_file_size(safe_path)

    # Read current content (safe_path is resolved and validated, not a symlink)
    with open(safe_path, "r") as f:
        content = f.read()

    if args.old_string not in content:
        return f"Error: old_string not found in {args.file_path}"
    count = content.count(args.old_string)
    if count > 1:
        return f"Error: old_string found {count} times in {args.file_path}. Must be unique."

    new_content = content.replace(args.old_string, args.new_string, 1)
    check_write_content_size(new_content)

    # Use safe_open_for_write with O_NOFOLLOW to prevent TOCTOU symlink attacks
    try:
        with safe_open_for_write(safe_path) as f:
            f.write(new_content)
    except ValueError as e:
        return f"Error: {e}"

    return f"File edited: {args.file_path}"
