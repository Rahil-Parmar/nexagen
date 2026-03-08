import asyncio
import os
import resource as _resource

from pydantic import BaseModel, Field
from nexagen.tools.base import tool

# Limits applied to child processes
_MAX_OUTPUT_BYTES = 1_000_000  # 1 MB max output
_MAX_CHILD_MEMORY = 512 * 1024 * 1024  # 512 MB
_MAX_CHILD_PROCS = 64  # prevent fork bombs
_MAX_TIMEOUT = 300  # 5 min hard ceiling


def _set_child_limits():
    """Set resource limits on the child process (called via preexec_fn)."""
    try:
        _resource.setrlimit(_resource.RLIMIT_AS, (_MAX_CHILD_MEMORY, _MAX_CHILD_MEMORY))
    except (ValueError, OSError):
        pass
    try:
        _resource.setrlimit(_resource.RLIMIT_NPROC, (_MAX_CHILD_PROCS, _MAX_CHILD_PROCS))
    except (ValueError, OSError):
        pass


class BashInput(BaseModel):
    command: str
    timeout: int = Field(default=120, description="Timeout in seconds", le=_MAX_TIMEOUT)


@tool("bash", "Execute a shell command", input_model=BashInput)
async def bash(args: BashInput) -> str:
    timeout = min(args.timeout, _MAX_TIMEOUT)

    proc = await asyncio.create_subprocess_shell(
        args.command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        preexec_fn=_set_child_limits,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return f"Error: Command timed out after {timeout}s"

    output = (stdout[:_MAX_OUTPUT_BYTES].decode(errors="replace")) if stdout else ""
    errors = (stderr[:_MAX_OUTPUT_BYTES].decode(errors="replace")) if stderr else ""

    if len(stdout or b"") > _MAX_OUTPUT_BYTES:
        output += "\n... (output truncated)"

    if proc.returncode != 0:
        return f"Exit code: {proc.returncode}\n{errors}\n{output}".strip()
    return output.strip()
