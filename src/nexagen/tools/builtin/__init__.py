from nexagen.tools.builtin.file_read import file_read
from nexagen.tools.builtin.file_write import file_write
from nexagen.tools.builtin.file_edit import file_edit
from nexagen.tools.builtin.bash import bash
from nexagen.tools.builtin.grep_tool import grep_tool
from nexagen.tools.builtin.glob_tool import glob_tool

BUILTIN_TOOLS = {
    "file_read": file_read,
    "file_write": file_write,
    "file_edit": file_edit,
    "bash": bash,
    "grep": grep_tool,
    "glob": glob_tool,
}
