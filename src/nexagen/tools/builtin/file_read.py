from pydantic import BaseModel
from nexagen.tools.base import tool
from nexagen.tools.builtin.path_security import validate_path, check_file_size


class FileReadInput(BaseModel):
    file_path: str
    offset: int | None = None
    limit: int | None = None


@tool("file_read", "Read contents of a file with line numbers", input_model=FileReadInput)
async def file_read(args: FileReadInput) -> str:
    safe_path = validate_path(args.file_path, must_exist=True)
    check_file_size(safe_path)

    with open(safe_path, "r") as f:
        lines = f.readlines()
    start = (args.offset or 1) - 1
    end = start + args.limit if args.limit else len(lines)
    selected = lines[start:end]
    numbered = [f"{i + start + 1:>4} | {line.rstrip()}" for i, line in enumerate(selected)]
    return "\n".join(numbered)
