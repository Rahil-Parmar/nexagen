"""Tests for the four-layer permission system (including system guardrails)."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from nexagen.permissions import (
    Allow, Deny, PermissionManager,
    check_system_guardrails, _check_path_guardrail, _check_command_guardrail,
)


def _run(coro):
    """Helper to run async check() calls."""
    return asyncio.run(coro)


# ── Layer 1: Mode tests ──────────────────────────────────────────────


class TestReadonlyMode:
    def test_file_read_allowed(self):
        pm = PermissionManager(mode="readonly")
        result = _run(pm.check("file_read", {}))
        assert isinstance(result, Allow)

    def test_bash_denied(self):
        pm = PermissionManager(mode="readonly")
        result = _run(pm.check("bash", {}))
        assert isinstance(result, Deny)
        assert "readonly" in result.message


class TestSafeMode:
    def test_file_read_allowed(self):
        pm = PermissionManager(mode="safe")
        result = _run(pm.check("file_read", {}))
        assert isinstance(result, Allow)

    def test_file_write_allowed(self):
        pm = PermissionManager(mode="safe")
        result = _run(pm.check("file_write", {}))
        assert isinstance(result, Allow)

    def test_bash_denied(self):
        pm = PermissionManager(mode="safe")
        result = _run(pm.check("bash", {}))
        assert isinstance(result, Deny)
        assert "safe" in result.message


class TestFullMode:
    def test_everything_allowed(self):
        pm = PermissionManager(mode="full")
        for tool in ("file_read", "file_write", "bash", "anything_goes"):
            result = _run(pm.check(tool, {}))
            assert isinstance(result, Allow), f"{tool} should be allowed in full mode"


# ── Layer 2: Allowlist ────────────────────────────────────────────────


class TestAllowlist:
    def test_allowlist_narrows_full_mode(self):
        pm = PermissionManager(mode="full", allowed_tools=["file_read"])
        assert isinstance(_run(pm.check("file_read", {})), Allow)
        result = _run(pm.check("bash", {}))
        assert isinstance(result, Deny)
        assert "allowed_tools" in result.message


# ── Layer 3: Callback ────────────────────────────────────────────────


class TestCallback:
    def test_callback_allows(self):
        async def always_allow(tool_name: str, args: dict) -> Allow | Deny:
            return Allow()

        pm = PermissionManager(mode="full", can_use_tool=always_allow)
        result = _run(pm.check("bash", {"command": "ls"}))
        assert isinstance(result, Allow)

    def test_callback_denies_dangerous_pattern(self):
        async def block_curl(tool_name: str, args: dict) -> Allow | Deny:
            if tool_name == "bash" and "curl" in args.get("command", ""):
                return Deny("Network access blocked")
            return Allow()

        pm = PermissionManager(mode="full", can_use_tool=block_curl)
        result = _run(pm.check("bash", {"command": "curl http://evil.com"}))
        assert isinstance(result, Deny)
        assert "Network access" in result.message

        # Non-curl bash should still pass
        result = _run(pm.check("bash", {"command": "ls"}))
        assert isinstance(result, Allow)


# ── All layers stacked ───────────────────────────────────────────────


class TestLayersStack:
    def test_mode_safe_allowlist_file_read_callback_allow(self):
        """mode=safe + allowlist=[file_read] + permissive callback => only file_read passes."""

        async def permissive(tool_name: str, args: dict) -> Allow | Deny:
            return Allow()

        pm = PermissionManager(
            mode="safe",
            allowed_tools=["file_read"],
            can_use_tool=permissive,
        )
        # file_read is in safe mode AND in allowlist => Allow
        assert isinstance(_run(pm.check("file_read", {})), Allow)

        # file_write is in safe mode but NOT in allowlist => Deny
        result = _run(pm.check("file_write", {}))
        assert isinstance(result, Deny)
        assert "allowed_tools" in result.message

        # bash is NOT in safe mode => Deny (blocked at layer 1)
        result = _run(pm.check("bash", {}))
        assert isinstance(result, Deny)
        assert "safe" in result.message


# ── Edge case: invalid/unknown mode ──────────────────────────────────


class TestInvalidMode:
    def test_unknown_mode_raises(self):
        """Unknown mode must raise ValueError — silently allowing everything is dangerous."""
        import pytest
        with pytest.raises(ValueError, match="Unknown permission mode"):
            PermissionManager(mode="yolo")


# ── Layer 0: System Guardrails ──────────────────────────────────


class TestSystemGuardrailsRoot:
    def test_blocks_running_as_root(self):
        """Agent must never run as root."""
        with patch("nexagen.permissions._is_running_as_root", return_value=True):
            result = check_system_guardrails("file_read", {"file_path": "/tmp/test"})
            assert isinstance(result, Deny)
            assert "root" in result.message

    def test_allows_normal_user(self):
        """Normal user should pass the root check."""
        with patch("nexagen.permissions._is_running_as_root", return_value=False):
            result = check_system_guardrails("file_read", {"file_path": "/tmp/test"})
            assert result is None  # No denial


class TestSystemGuardrailsPaths:
    def test_blocks_etc_shadow(self):
        result = _check_path_guardrail("/etc/shadow")
        assert isinstance(result, Deny)
        assert "blocked" in result.message

    def test_blocks_etc_sudoers(self):
        result = _check_path_guardrail("/etc/sudoers")
        assert isinstance(result, Deny)

    def test_blocks_system_directories(self):
        result = _check_path_guardrail("/System/Library/something")
        assert isinstance(result, Deny)
        assert "System" in result.message

    def test_allows_user_home(self):
        home = str(Path.home())
        result = _check_path_guardrail(os.path.join(home, "projects", "test.py"))
        assert result is None

    def test_allows_tmp(self):
        result = _check_path_guardrail("/tmp/nexagen_test/file.txt")
        assert result is None

    def test_blocks_other_users_home(self):
        home = str(Path.home())
        home_parent = str(Path(home).parent)
        other_user_path = os.path.join(home_parent, "otheruser", "secrets.txt")
        result = _check_path_guardrail(other_user_path)
        assert isinstance(result, Deny)
        assert "home directory" in result.message

    def test_allows_current_user_home(self):
        home = str(Path.home())
        result = _check_path_guardrail(os.path.join(home, "Documents", "file.txt"))
        assert result is None


class TestSystemGuardrailsCommands:
    def test_blocks_sudo(self):
        result = _check_command_guardrail("sudo rm -rf /")
        assert isinstance(result, Deny)
        assert "blocked" in result.message

    def test_blocks_su(self):
        result = _check_command_guardrail("su - root")
        assert isinstance(result, Deny)

    def test_blocks_rm_rf_root(self):
        result = _check_command_guardrail("rm -rf /")
        assert isinstance(result, Deny)

    def test_blocks_rm_rf_wildcard(self):
        result = _check_command_guardrail("rm -rf /*")
        assert isinstance(result, Deny)

    def test_blocks_shutdown(self):
        result = _check_command_guardrail("shutdown -h now")
        assert isinstance(result, Deny)

    def test_blocks_reboot(self):
        result = _check_command_guardrail("reboot")
        assert isinstance(result, Deny)

    def test_blocks_dd(self):
        result = _check_command_guardrail("dd if=/dev/zero of=/dev/sda")
        assert isinstance(result, Deny)

    def test_blocks_chmod_777(self):
        result = _check_command_guardrail("chmod 777 /etc/passwd")
        assert isinstance(result, Deny)

    def test_allows_safe_commands(self):
        for cmd in ["echo hello", "ls -la", "cat file.txt", "python main.py", "git status"]:
            result = _check_command_guardrail(cmd)
            assert result is None, f"Safe command '{cmd}' should be allowed"

    def test_allows_rm_in_project_dir(self):
        """rm on specific files (not rm -rf /) should be allowed by guardrails."""
        result = _check_command_guardrail("rm /tmp/nexagen_test/temp.txt")
        assert result is None


class TestGuardrailsIntegration:
    def test_guardrails_enforced_in_full_mode(self):
        """Even in full mode, system guardrails block dangerous operations."""
        pm = PermissionManager(mode="full")
        result = _run(pm.check("bash", {"command": "sudo rm -rf /"}))
        assert isinstance(result, Deny)
        assert "guardrail" in result.message.lower()

    def test_guardrails_block_shadow_in_full_mode(self):
        pm = PermissionManager(mode="full")
        result = _run(pm.check("file_read", {"file_path": "/etc/shadow"}))
        assert isinstance(result, Deny)

    def test_guardrails_allow_normal_in_full_mode(self):
        pm = PermissionManager(mode="full")
        home = str(Path.home())
        result = _run(pm.check("file_read", {"file_path": os.path.join(home, "test.py")}))
        assert isinstance(result, Allow)

    def test_guardrails_block_other_user_home_in_full_mode(self):
        pm = PermissionManager(mode="full")
        home = str(Path.home())
        home_parent = str(Path(home).parent)
        result = _run(pm.check("file_read", {"file_path": os.path.join(home_parent, "otheruser", "file")}))
        assert isinstance(result, Deny)

    def test_guardrails_block_grep_in_system_dir(self):
        pm = PermissionManager(mode="full")
        result = _run(pm.check("grep", {"pattern": "password", "path": "/System/Library"}))
        assert isinstance(result, Deny)

    def test_guardrails_with_root_user(self):
        pm = PermissionManager(mode="full")
        with patch("nexagen.permissions._is_running_as_root", return_value=True):
            result = _run(pm.check("file_read", {"file_path": "/tmp/test"}))
            assert isinstance(result, Deny)
            assert "root" in result.message
