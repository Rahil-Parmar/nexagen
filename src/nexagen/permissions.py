"""Three-layer permission system with system-level guardrails.

Layer 0: System guardrails (always enforced, cannot be bypassed)
Layer 1: Mode presets (readonly / safe / full)
Layer 2: Allowlist (narrows from mode)
Layer 3: User callback (fine-grained control)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable


@dataclass
class Allow:
    pass


@dataclass
class Deny:
    message: str = "Permission denied"


_MODE_TOOLS: dict[str, set[str] | None] = {
    "readonly": {"file_read", "grep", "glob"},
    "safe": {"file_read", "file_write", "file_edit", "grep", "glob"},
    "full": None,  # None means all tools allowed
}

# Directories that tools must never read from or write to
_BLOCKED_PATHS = {
    "/etc/shadow", "/etc/sudoers", "/etc/master.passwd", "/etc/passwd",
    "/etc/gshadow", "/etc/security/opasswd",
    "/private/etc/shadow", "/private/etc/sudoers",
    # SSH keys
    "authorized_keys", "id_rsa", "id_ed25519", "id_ecdsa",
}

_BLOCKED_PREFIXES = (
    "/System/", "/usr/sbin/", "/sbin/",
    "/private/var/root/",
    "/proc/", "/sys/",  # Linux kernel interfaces
    "/boot/",
    "/dev/",
)

# Dangerous bash patterns that are always blocked (checked after normalization)
_BLOCKED_COMMANDS = (
    # Privilege escalation
    "sudo ", "su ", "doas ", "pkexec ",
    "chown root", "chmod u+s", "chmod 4", "chmod 777",
    # System destruction
    "mkfs", "dd if=",
    "> /dev/", ">> /dev/",
    "shutdown", "reboot", "halt", "poweroff", "init 0", "init 6",
    "rm -rf /", "rm -rf /*", "rm -rf ~",
    # Fork bombs and process abuse
    ":(){ ",  # bash fork bomb
    ".() {", # alternative fork bomb syntax
    # Sensitive file access
    "/etc/shadow", "/etc/sudoers", "/etc/passwd",
    "authorized_keys",
    # Remote code execution — piping into shells
    "curl|sh", "curl|bash", "wget|sh", "wget|bash",
    "|sh", "|bash", "|zsh", "|ksh", "|dash",
    "| sh", "| bash", "| zsh", "| ksh", "| dash",
    # Dynamic code execution that can bypass other checks
    "python -c", "python3 -c", "perl -e", "ruby -e", "node -e",
    # Eval and exec in shell
    "eval ", "exec ",
    # Reverse shells
    "/dev/tcp/", "/dev/udp/",
    "nc -e", "ncat -e", "netcat -e",
    "bash -i >",
    # File exfiltration
    "curl -d @", "curl --data @", "curl -F file=@",
    # Crontab manipulation
    "crontab ", "/etc/cron",
    # User/group manipulation
    "useradd", "usermod", "groupadd", "passwd ",
    # Service manipulation
    "systemctl ", "service ", "launchctl ",
    # Disk/partition manipulation
    "fdisk", "parted", "mount ", "umount ",
    # Network configuration changes
    "iptables", "ip route", "ifconfig",
    # Package manager operations (can install malware)
    "apt install", "apt-get install", "yum install", "dnf install",
    "brew install", "pip install", "npm install -g",
)

# Regex patterns for harder-to-evade checks
_BLOCKED_COMMAND_PATTERNS = [
    # Piping to any shell (handles whitespace variations)
    re.compile(r'\|\s*(ba)?sh\b'),
    re.compile(r'\|\s*zsh\b'),
    re.compile(r'\|\s*/bin/(ba)?sh\b'),
    re.compile(r'\|\s*/usr/bin/env\s+(ba)?sh\b'),
    # Downloading and executing
    re.compile(r'curl\s.*\|\s*\w*sh\b'),
    re.compile(r'wget\s.*\|\s*\w*sh\b'),
    re.compile(r'curl\s.*-o\s*/'), # curl writing to system paths
    re.compile(r'wget\s.*-O\s*/'), # wget writing to system paths
    # find -exec can run arbitrary commands
    re.compile(r'find\s.*-exec\s'),
    re.compile(r'find\s.*-execdir\s'),
    # xargs can construct arbitrary commands
    re.compile(r'xargs\s.*-I'),
    # Backgrounding suspicious commands
    re.compile(r'nohup\s.*(curl|wget|nc|python|perl|ruby)'),
    # Process substitution to bypass pipes
    re.compile(r'<\(.*\)'),
    re.compile(r'>\(.*\)'),
    # Backtick command substitution (can hide commands)
    re.compile(r'`[^`]*`'),
    # Hex/octal/base64 encoded command execution
    re.compile(r'\\x[0-9a-fA-F]{2}'),
    re.compile(r'echo\s.*\|\s*base64\s.*-d'),
    re.compile(r'base64\s.*-d\s*\|'),
    # Environment variable manipulation for escalation
    re.compile(r'export\s+(PATH|LD_PRELOAD|LD_LIBRARY_PATH)\s*='),
    # chmod with setuid/setgid bits
    re.compile(r'chmod\s+[0-7]*[4-7][0-7]{2}\s'),  # setuid/setgid
]


def _normalize_command(command: str) -> str:
    """Normalize a command for security checking.

    Collapses whitespace, strips comments, lowercases for consistent matching.
    """
    cmd = command.lower().strip()
    # Collapse multiple spaces/tabs to single space
    cmd = re.sub(r'\s+', ' ', cmd)
    # Remove shell comments (but not inside quotes — best effort)
    cmd = re.sub(r'#[^\'"]*$', '', cmd)
    # Normalize pipe spacing: ` | ` → `|` (for substring matching)
    # Keep original for regex patterns that handle whitespace
    return cmd


def _get_user_home() -> str:
    return str(Path.home())


def _is_running_as_root() -> bool:
    return os.getuid() == 0


def _resolve_path(path: str) -> str:
    """Resolve a path to absolute, following symlinks."""
    try:
        return str(Path(path).resolve())
    except (OSError, ValueError):
        return path


def _check_path_guardrail(path: str) -> Deny | None:
    """Check if a file path violates system guardrails. Returns Deny or None."""
    resolved = _resolve_path(path)

    # Block access to sensitive system files (exact match and basename match)
    if resolved in _BLOCKED_PATHS:
        return Deny(f"System guardrail: access to this path is blocked")

    basename = os.path.basename(resolved)
    if basename in _BLOCKED_PATHS:
        return Deny(f"System guardrail: access to '{basename}' files is blocked")

    # Block access to system directories
    for prefix in _BLOCKED_PREFIXES:
        if resolved.startswith(prefix):
            return Deny(f"System guardrail: access to system paths is blocked")

    # Block access to other users' home directories
    home = _get_user_home()
    home_parent = str(Path(home).parent)  # e.g., /Users or /home

    if resolved.startswith(home_parent + "/") and not resolved.startswith(home + "/"):
        return Deny("System guardrail: cannot access other user's home directory")

    # Block dotfiles that contain secrets (in any directory)
    sensitive_dotfiles = {
        ".env", ".env.local", ".env.production",
        ".netrc", ".npmrc", ".pypirc",
        ".aws/credentials", ".docker/config.json",
    }
    for dotfile in sensitive_dotfiles:
        if resolved.endswith("/" + dotfile):
            return Deny(f"System guardrail: access to '{dotfile}' is blocked (may contain secrets)")

    return None


def _check_command_guardrail(command: str) -> Deny | None:
    """Check if a bash command violates system guardrails. Returns Deny or None."""
    normalized = _normalize_command(command)

    # Substring matching on normalized command
    for blocked in _BLOCKED_COMMANDS:
        if blocked in normalized:
            return Deny(f"System guardrail: command contains blocked pattern")

    # Regex matching on the original (lowered) command for evasion resistance
    cmd_lower = command.lower()
    for pattern in _BLOCKED_COMMAND_PATTERNS:
        if pattern.search(cmd_lower):
            return Deny(f"System guardrail: command matches blocked pattern")

    # Block commands with an excessive number of semicolons or && chains
    # (likely attempting to sneak past simple checks)
    separator_count = cmd_lower.count(";") + cmd_lower.count("&&") + cmd_lower.count("||")
    if separator_count > 10:
        return Deny("System guardrail: too many chained commands (max 10)")

    return None


def check_system_guardrails(tool_name: str, args: dict) -> Deny | None:
    """Layer 0: System-level guardrails that are always enforced.

    - Blocks running as root
    - Blocks access to sensitive system files
    - Blocks access to other users' home directories
    - Blocks dangerous shell commands (sudo, rm -rf /, etc.)
    """
    # Never run as root
    if _is_running_as_root():
        return Deny("System guardrail: nexagen must not run as root")

    # Check file paths for file tools
    if tool_name in ("file_read", "file_write", "file_edit"):
        path = args.get("file_path", "")
        if path:
            denial = _check_path_guardrail(path)
            if denial:
                return denial

    # Check grep/glob paths
    if tool_name in ("grep", "glob"):
        path = args.get("path", "")
        if path:
            denial = _check_path_guardrail(path)
            if denial:
                return denial

    # Check bash commands
    if tool_name == "bash":
        command = args.get("command", "")
        if command:
            denial = _check_command_guardrail(command)
            if denial:
                return denial

    return None


class PermissionManager:
    """Enforces a four-layer permission check:

    Layer 0: System guardrails (always enforced, cannot be bypassed)
    Layer 1: Mode presets (readonly / safe / full)
    Layer 2: Allowlist (narrows from mode)
    Layer 3: User callback (fine-grained control)
    """

    def __init__(
        self,
        mode: str = "safe",
        allowed_tools: list[str] | None = None,
        can_use_tool: Callable[[str, dict], Awaitable[Allow | Deny]] | None = None,
    ):
        if mode not in _MODE_TOOLS:
            raise ValueError(f"Unknown permission mode: '{mode}'. Must be one of: {list(_MODE_TOOLS.keys())}")
        self.mode = mode
        self.allowed_tools = set(allowed_tools) if allowed_tools else None
        self.can_use_tool = can_use_tool

    async def check(self, tool_name: str, args: dict) -> Allow | Deny:
        """Check whether *tool_name* may be invoked with *args*.

        Layers are evaluated in order; the first Deny wins.
        """
        # Layer 0: system guardrails (always enforced)
        guardrail = check_system_guardrails(tool_name, args)
        if guardrail:
            return guardrail

        # Layer 1: mode
        mode_tools = _MODE_TOOLS.get(self.mode)
        if mode_tools is not None and tool_name not in mode_tools:
            return Deny(f"Tool '{tool_name}' not allowed in '{self.mode}' mode")

        # Layer 2: allowlist
        if self.allowed_tools is not None and tool_name not in self.allowed_tools:
            return Deny(f"Tool '{tool_name}' not in allowed_tools")

        # Layer 3: callback
        if self.can_use_tool:
            return await self.can_use_tool(tool_name, args)

        return Allow()
