"""Tests for nexagen built-in tools."""

import asyncio
import os

import pytest


def run(coro):
    """Helper to run an async coroutine synchronously."""
    return asyncio.run(coro)


# ── file_read ──────────────────────────────────────────────────────────────


class TestFileRead:
    def test_file_read(self, tmp_path):
        from nexagen.tools.builtin.file_read import file_read

        p = tmp_path / "sample.txt"
        p.write_text("alpha\nbeta\ngamma\n")

        result = run(file_read.execute({"file_path": str(p)}))
        assert not result.is_error
        assert "   1 | alpha" in result.output
        assert "   2 | beta" in result.output
        assert "   3 | gamma" in result.output

    def test_file_read_with_offset_limit(self, tmp_path):
        from nexagen.tools.builtin.file_read import file_read

        p = tmp_path / "lines.txt"
        p.write_text("line1\nline2\nline3\nline4\nline5\n")

        result = run(file_read.execute({"file_path": str(p), "offset": 2, "limit": 2}))
        assert not result.is_error
        assert "   2 | line2" in result.output
        assert "   3 | line3" in result.output
        assert "line1" not in result.output
        assert "line4" not in result.output


# ── file_write ─────────────────────────────────────────────────────────────


class TestFileWrite:
    def test_file_write(self, tmp_path):
        from nexagen.tools.builtin.file_write import file_write

        p = tmp_path / "out.txt"
        result = run(file_write.execute({"file_path": str(p), "content": "hello world"}))
        assert not result.is_error
        assert p.read_text() == "hello world"

    def test_file_write_creates_dirs(self, tmp_path):
        from nexagen.tools.builtin.file_write import file_write

        p = tmp_path / "a" / "b" / "c" / "deep.txt"
        result = run(
            file_write.execute({"file_path": str(p), "content": "nested content"})
        )
        assert not result.is_error
        assert p.exists()
        assert p.read_text() == "nested content"


# ── file_edit ──────────────────────────────────────────────────────────────


class TestFileEdit:
    def test_file_edit(self, tmp_path):
        from nexagen.tools.builtin.file_edit import file_edit

        p = tmp_path / "edit_me.txt"
        p.write_text("foo bar baz")

        result = run(
            file_edit.execute(
                {"file_path": str(p), "old_string": "bar", "new_string": "qux"}
            )
        )
        assert not result.is_error
        assert p.read_text() == "foo qux baz"

    def test_file_edit_not_found(self, tmp_path):
        from nexagen.tools.builtin.file_edit import file_edit

        p = tmp_path / "edit_me.txt"
        p.write_text("foo bar baz")

        result = run(
            file_edit.execute(
                {"file_path": str(p), "old_string": "MISSING", "new_string": "x"}
            )
        )
        assert "Error" in result.output

    def test_file_edit_multiple_matches(self, tmp_path):
        from nexagen.tools.builtin.file_edit import file_edit

        p = tmp_path / "dup.txt"
        p.write_text("aaa bbb aaa")

        result = run(
            file_edit.execute(
                {"file_path": str(p), "old_string": "aaa", "new_string": "ccc"}
            )
        )
        assert "Error" in result.output
        assert "2" in result.output


# ── bash ───────────────────────────────────────────────────────────────────


class TestBash:
    def test_bash_success(self):
        from nexagen.tools.builtin.bash import bash

        result = run(bash.execute({"command": "echo hello"}))
        assert not result.is_error
        assert "hello" in result.output

    def test_bash_failure(self):
        from nexagen.tools.builtin.bash import bash

        result = run(bash.execute({"command": "false"}))
        assert "Exit code: 1" in result.output


# ── grep ───────────────────────────────────────────────────────────────────


class TestGrep:
    def test_grep_file(self, tmp_path):
        from nexagen.tools.builtin.grep_tool import grep_tool

        p = tmp_path / "data.txt"
        p.write_text("apple\nbanana\napricot\ncherry\n")

        result = run(grep_tool.execute({"pattern": "ap", "path": str(p)}))
        assert not result.is_error
        assert "apple" in result.output
        assert "apricot" in result.output
        assert "banana" not in result.output

    def test_grep_no_matches(self, tmp_path):
        from nexagen.tools.builtin.grep_tool import grep_tool

        p = tmp_path / "data.txt"
        p.write_text("apple\nbanana\n")

        result = run(grep_tool.execute({"pattern": "zzz", "path": str(p)}))
        assert "No matches" in result.output


# ── glob ───────────────────────────────────────────────────────────────────


class TestGlob:
    def test_glob_matches(self, tmp_path):
        from nexagen.tools.builtin.glob_tool import glob_tool

        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        (tmp_path / "c.py").write_text("c")

        result = run(
            glob_tool.execute({"pattern": "*.txt", "path": str(tmp_path)})
        )
        assert not result.is_error
        assert "a.txt" in result.output
        assert "b.txt" in result.output
        assert "c.py" not in result.output


# ── BUILTIN_TOOLS dict ────────────────────────────────────────────────────


class TestBuiltinToolsDict:
    def test_builtin_tools_dict(self):
        from nexagen.tools.builtin import BUILTIN_TOOLS

        assert len(BUILTIN_TOOLS) == 6
        expected_keys = {"file_read", "file_write", "file_edit", "bash", "grep", "glob"}
        assert set(BUILTIN_TOOLS.keys()) == expected_keys
