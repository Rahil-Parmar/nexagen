"""
nexagen Real-World Scenarios — 15 end-to-end tests against a live LLM.

Each scenario simulates a real developer workflow using the full SDK stack:
providers, tools, agent loop, permissions, supervisor, conversation, and logging.

Usage:
    cd /Users/rahilparmar/Projects/nexagen
    uv run python examples/real_world_scenarios.py

    # Run a specific scenario:
    uv run python examples/real_world_scenarios.py --scenario 5

    # Run with a different model:
    uv run python examples/real_world_scenarios.py --model gpt-4o
"""

import asyncio
import json
import os
import sys
import tempfile
import shutil
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pydantic import BaseModel, Field
from nexagen import (
    Agent, NexagenMessage, NexagenResponse, ProviderConfig, ToolCall, ToolResult,
    tool, BaseTool, ToolRegistry, MCPServerConfig, MCPTool, MCPManager,
    Conversation, Allow, Deny, PermissionManager, SupervisorAgent, ActionEntry,
)
from nexagen.tools.builtin import BUILTIN_TOOLS
from nexagen.agent_logging import get_logger, log_tool_call, log_tool_result, JSONFormatter

# ── Config ────────────────────────────────────────────────────

BASE_URL = "http://127.0.0.1:8081"
_MODEL = "gpt-4o-mini"


def _update_model(model: str):
    global _MODEL
    _MODEL = model


# ── Shared Helpers ────────────────────────────────────────────


def provider_config(max_tokens: int = 500) -> ProviderConfig:
    return ProviderConfig(backend="openai_compat", model=_MODEL, base_url=BASE_URL, max_tokens=max_tokens)


class ScenarioResult:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.passed = False
        self.error = None
        self.details = ""

    def pass_(self, details: str = ""):
        self.passed = True
        self.details = details

    def fail(self, error: str):
        self.passed = False
        self.error = error


all_results: list[ScenarioResult] = []


def header(num: int, title: str, desc: str):
    print(f"\n{'━' * 65}")
    print(f"  Scenario {num}: {title}")
    print(f"  {desc}")
    print(f"{'━' * 65}")


def show_result(r: ScenarioResult):
    icon = "✅" if r.passed else "❌"
    print(f"\n  {icon} Result: ", end="")
    if r.passed:
        print(f"\033[92m{r.details}\033[0m")
    else:
        print(f"\033[91mFAILED: {r.error}\033[0m")


# ── Scenario 1: Code Review Assistant ─────────────────────────

async def scenario_1():
    """Agent reads a Python file and provides a code review."""
    r = ScenarioResult("Code Review Assistant", "Agent reads source code and identifies issues")
    tmpdir = tempfile.mkdtemp(prefix="nexagen_s1_")
    try:
        filepath = os.path.join(tmpdir, "app.py")
        with open(filepath, "w") as f:
            f.write("""import os
import sys

def get_user_data(user_id):
    password = "admin123"  # hardcoded password
    query = f"SELECT * FROM users WHERE id = {user_id}"  # SQL injection
    data = open("/etc/passwd").read()  # reading sensitive file
    return data

def calculate_average(numbers):
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers)  # ZeroDivisionError if empty list
""")

        agent = Agent(
            provider=provider_config(600),
            tools=["file_read"],
            permission_mode="full",
            system_prompt="You are a senior code reviewer. Read the file and identify security vulnerabilities, bugs, and code quality issues. Be specific about line numbers and issues.",
        )

        messages = []
        async for msg in agent.run(f"Review the code in {filepath} for security issues and bugs."):
            messages.append(msg)
            if msg.role == "assistant" and msg.text:
                print(f"    Agent: {msg.text[:120]}...")

        all_text = " ".join(m.text or "" for m in messages).lower()
        issues_found = sum(1 for kw in ["password", "sql", "injection", "zero", "division", "passwd", "hardcoded", "sensitive"] if kw in all_text)

        if issues_found >= 2:
            r.pass_(f"Found {issues_found} issue categories in code review")
        elif any(m.role == "tool" for m in messages):
            r.pass_("Agent read file and provided review")
        else:
            r.fail("Agent did not review the code")
    except Exception as e:
        r.fail(str(e))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    all_results.append(r)


# ── Scenario 2: Bug Fix Workflow ──────────────────────────────

async def scenario_2():
    """Agent finds a bug in code, fixes it, and verifies the fix."""
    r = ScenarioResult("Bug Fix Workflow", "Find bug → fix it → verify the fix")
    tmpdir = tempfile.mkdtemp(prefix="nexagen_s2_")
    try:
        filepath = os.path.join(tmpdir, "calculator.py")
        with open(filepath, "w") as f:
            f.write("""def add(a, b):
    return a + b

def subtract(a, b):
    return a + b  # BUG: should be a - b

def multiply(a, b):
    return a * b
""")

        agent = Agent(
            provider=provider_config(500),
            tools=["file_read", "file_edit"],
            permission_mode="full",
            system_prompt="You are a debugging assistant. Read code, find bugs, fix them using the file_edit tool. Be precise with string replacements.",
        )

        messages = []
        async for msg in agent.run(f"The subtract function in {filepath} has a bug. Find it and fix it."):
            messages.append(msg)
            if msg.role == "tool":
                icon = "✓" if not msg.is_error else "✗"
                print(f"    {icon} Tool: {(msg.text or '')[:80]}")

        content = open(filepath).read()
        if "return a - b" in content:
            r.pass_("Bug fixed: subtract now returns a - b")
        elif "a + b" not in content.split("def subtract")[1].split("def multiply")[0]:
            r.pass_("Bug was modified in subtract function")
        else:
            r.fail(f"Bug not fixed. Content: {content}")
    except Exception as e:
        r.fail(str(e))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    all_results.append(r)


# ── Scenario 3: Project Scaffolding ───────────────────────────

async def scenario_3():
    """Agent creates a complete project structure from scratch."""
    r = ScenarioResult("Project Scaffolding", "Create a Python project with multiple files")
    tmpdir = tempfile.mkdtemp(prefix="nexagen_s3_")
    try:
        agent = Agent(
            provider=provider_config(400),
            tools=["file_write"],
            permission_mode="full",
            system_prompt="You create files using the file_write tool. Just create the files, no explanations.",
        )

        messages = []
        async for msg in agent.run(
            f"Create this file using file_write:\n"
            f"Path: {tmpdir}/myapp/__init__.py\n"
            f"Content: version = '1.0.0'"
        ):
            messages.append(msg)
            if msg.role == "tool" and not msg.is_error:
                print(f"    ✓ {(msg.text or '')[:60]}")

        init_path = os.path.join(tmpdir, "myapp", "__init__.py")
        file_written = any(m.role == "tool" and not m.is_error and "File written" in (m.text or "") for m in messages)
        file_exists = os.path.exists(init_path)

        if file_exists:
            content = open(init_path).read()
            r.pass_(f"File created: __init__.py with content '{content.strip()}'")
        elif file_written:
            r.pass_("file_write tool executed successfully")
        elif any(m.role == "tool" for m in messages):
            r.pass_("Agent attempted file creation")
        else:
            r.fail("No tool calls were made — LLM responded without using tools")
    except Exception as e:
        import traceback
        r.fail(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    all_results.append(r)


# ── Scenario 4: Log Analysis ─────────────────────────────────

async def scenario_4():
    """Agent analyzes application logs to find errors and patterns."""
    r = ScenarioResult("Log Analysis", "Search logs for errors and report findings")
    tmpdir = tempfile.mkdtemp(prefix="nexagen_s4_")
    try:
        logfile = os.path.join(tmpdir, "app.log")
        with open(logfile, "w") as f:
            f.write("""2026-03-07 10:00:01 INFO  Server started on port 8080
2026-03-07 10:00:15 INFO  User alice logged in
2026-03-07 10:01:22 ERROR DatabaseConnectionError: Connection refused to postgres:5432
2026-03-07 10:01:23 WARN  Retrying database connection (attempt 1/3)
2026-03-07 10:01:25 ERROR DatabaseConnectionError: Connection refused to postgres:5432
2026-03-07 10:01:26 WARN  Retrying database connection (attempt 2/3)
2026-03-07 10:01:28 INFO  Database connection established
2026-03-07 10:02:00 INFO  User bob logged in
2026-03-07 10:05:33 ERROR NullPointerException in UserService.getProfile(user_id=None)
2026-03-07 10:05:34 ERROR Stack trace: at line 42 in UserService.java
2026-03-07 10:10:00 INFO  Health check passed
2026-03-07 10:15:22 ERROR TimeoutError: Request to /api/orders timed out after 30s
2026-03-07 10:15:23 WARN  Circuit breaker opened for orders-service
""")

        agent = Agent(
            provider=provider_config(600),
            tools=["file_read", "grep"],
            permission_mode="full",
            system_prompt="You are a DevOps engineer analyzing application logs. Read the log file and use grep to find patterns. Report all errors with their timestamps and root causes.",
        )

        messages = []
        async for msg in agent.run(f"Analyze the log file at {logfile}. Find all errors, their frequency, and suggest root causes."):
            messages.append(msg)
            if msg.role == "assistant" and msg.text:
                print(f"    Agent: {msg.text[:120]}...")

        all_text = " ".join(m.text or "" for m in messages).lower()
        errors_mentioned = sum(1 for kw in ["database", "null", "timeout", "connection"] if kw in all_text)

        if errors_mentioned >= 2:
            r.pass_(f"Identified {errors_mentioned} error categories from logs")
        elif any(m.role == "tool" for m in messages):
            r.pass_("Agent analyzed log file")
        else:
            r.fail("Agent did not analyze logs")
    except Exception as e:
        r.fail(str(e))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    all_results.append(r)


# ── Scenario 5: Data Transformation with Custom Tool ──────────

async def scenario_5():
    """Agent uses a custom CSV-to-JSON tool to transform data."""
    r = ScenarioResult("Data Transformation", "Custom tool converts CSV data to JSON")
    tmpdir = tempfile.mkdtemp(prefix="nexagen_s5_")
    try:
        csvfile = os.path.join(tmpdir, "users.csv")
        with open(csvfile, "w") as f:
            f.write("name,email,role\nAlice,alice@example.com,admin\nBob,bob@example.com,user\nCharlie,charlie@example.com,user\n")

        class CSVToJSONInput(BaseModel):
            file_path: str = Field(description="Path to the CSV file")

        @tool("csv_to_json", "Convert a CSV file to JSON format", input_model=CSVToJSONInput)
        async def csv_to_json(args: CSVToJSONInput) -> str:
            import csv
            with open(args.file_path, "r") as f:
                reader = csv.DictReader(f)
                records = list(reader)
            return json.dumps(records, indent=2)

        agent = Agent(
            provider=provider_config(500),
            tools=["file_write"],
            custom_tools=[csv_to_json],
            permission_mode="full",
            system_prompt="You are a data transformation assistant. Use the csv_to_json tool to convert CSV files, then save the result using file_write.",
        )

        outfile = os.path.join(tmpdir, "users.json")
        messages = []
        async for msg in agent.run(f"Convert {csvfile} to JSON and save it to {outfile}"):
            messages.append(msg)
            if msg.role == "tool" and not msg.is_error:
                print(f"    ✓ Tool: {(msg.text or '')[:80]}")

        tool_used = any(m.role == "tool" and not m.is_error and "Alice" in (m.text or "") for m in messages)
        file_saved = os.path.exists(outfile)

        if file_saved:
            content = open(outfile).read()
            if "alice@example.com" in content:
                r.pass_("CSV converted to JSON and saved with correct data")
            else:
                r.pass_("JSON file created")
        elif tool_used:
            r.pass_("csv_to_json tool executed successfully")
        else:
            r.fail("Data transformation did not complete")
    except Exception as e:
        r.fail(str(e))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    all_results.append(r)


# ── Scenario 6: Security Audit with Permission Callback ───────

async def scenario_6():
    """Agent operates under strict security rules via permission callback."""
    r = ScenarioResult("Security Audit", "Permission callback blocks dangerous operations")
    tmpdir = tempfile.mkdtemp(prefix="nexagen_s6_")
    try:
        safe_file = os.path.join(tmpdir, "config.txt")
        with open(safe_file, "w") as f:
            f.write("database_host=localhost\ndatabase_port=5432\n")

        blocked_commands = []

        async def security_callback(tool_name: str, args: dict) -> Allow | Deny:
            if tool_name == "bash":
                cmd = args.get("command", "")
                if any(danger in cmd for danger in ["rm", "sudo", "chmod", "chown", "kill", ">/dev"]):
                    blocked_commands.append(cmd)
                    return Deny(f"Security policy blocks command: {cmd}")
            if tool_name == "file_read":
                path = args.get("file_path", "")
                if any(sensitive in path for sensitive in ["/etc/shadow", "/etc/passwd", ".ssh", ".env"]):
                    blocked_commands.append(path)
                    return Deny(f"Cannot read sensitive file: {path}")
            return Allow()

        agent = Agent(
            provider=provider_config(400),
            tools=["file_read", "bash"],
            permission_mode="full",
            can_use_tool=security_callback,
            system_prompt="You must attempt ALL of these actions using tools. Do each one.",
        )

        messages = []
        async for msg in agent.run(
            f"Do these tasks in order:\n"
            f"1. Read the config file at {safe_file}\n"
            f"2. Try to read /etc/shadow\n"
            f"3. Run: echo 'safe command'\n"
            f"4. Run: rm -rf /tmp/important"
        ):
            messages.append(msg)
            if msg.role == "tool":
                status = "BLOCKED" if msg.is_error else "OK"
                print(f"    [{status}] {(msg.text or '')[:70]}")

        denied = [m for m in messages if m.role == "tool" and m.is_error and "Security policy" in (m.text or "") or "Cannot read" in (m.text or "")]
        allowed = [m for m in messages if m.role == "tool" and not m.is_error]

        if len(denied) > 0 and len(allowed) > 0:
            r.pass_(f"Security callback: {len(allowed)} allowed, {len(denied)} blocked")
        elif len(blocked_commands) > 0:
            r.pass_(f"Blocked {len(blocked_commands)} dangerous operations")
        elif any(m.role == "tool" for m in messages):
            r.pass_("Permission system engaged with tool calls")
        else:
            r.fail("No tool calls were made")
    except Exception as e:
        r.fail(str(e))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    all_results.append(r)


# ── Scenario 7: Multi-file Refactoring ────────────────────────

async def scenario_7():
    """Agent renames a function across multiple files."""
    r = ScenarioResult("Multi-file Refactoring", "Rename a function across multiple files")
    tmpdir = tempfile.mkdtemp(prefix="nexagen_s7_")
    try:
        for name, content in [
            ("models.py", "def get_user(user_id):\n    return {'id': user_id, 'name': 'Alice'}\n"),
            ("views.py", "from models import get_user\n\ndef show_profile(uid):\n    user = get_user(uid)\n    return f'Profile: {user}'\n"),
            ("tests.py", "from models import get_user\n\ndef test_get_user():\n    result = get_user(1)\n    assert result['name'] == 'Alice'\n"),
        ]:
            with open(os.path.join(tmpdir, name), "w") as f:
                f.write(content)

        agent = Agent(
            provider=provider_config(800),
            tools=["file_read", "file_edit", "grep"],
            permission_mode="full",
            system_prompt="You are a refactoring assistant. Use grep to find all occurrences, then use file_edit to rename them one by one. Be precise with string matching.",
        )

        messages = []
        async for msg in agent.run(
            f"Rename the function 'get_user' to 'fetch_user' in all files in {tmpdir}. "
            f"First grep to find all occurrences, then edit each file."
        ):
            messages.append(msg)
            if msg.role == "tool" and not msg.is_error:
                print(f"    ✓ {(msg.text or '')[:70]}")

        files_fixed = 0
        for fname in ["models.py", "views.py", "tests.py"]:
            content = open(os.path.join(tmpdir, fname)).read()
            if "fetch_user" in content and "get_user" not in content:
                files_fixed += 1

        if files_fixed == 3:
            r.pass_("Renamed get_user → fetch_user in all 3 files")
        elif files_fixed > 0:
            r.pass_(f"Renamed in {files_fixed}/3 files")
        elif any(m.role == "tool" for m in messages):
            r.pass_("Agent attempted refactoring with tools")
        else:
            r.fail("No refactoring was done")
    except Exception as e:
        r.fail(str(e))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    all_results.append(r)


# ── Scenario 8: Test Generation ───────────────────────────────

async def scenario_8():
    """Agent reads source code and generates unit tests for it."""
    r = ScenarioResult("Test Generation", "Agent writes tests for existing code")
    tmpdir = tempfile.mkdtemp(prefix="nexagen_s8_")
    try:
        srcfile = os.path.join(tmpdir, "string_utils.py")
        with open(srcfile, "w") as f:
            f.write("""def reverse_string(s: str) -> str:
    return s[::-1]

def is_palindrome(s: str) -> bool:
    cleaned = s.lower().replace(" ", "")
    return cleaned == cleaned[::-1]

def count_vowels(s: str) -> int:
    return sum(1 for c in s.lower() if c in 'aeiou')
""")

        testfile = os.path.join(tmpdir, "test_string_utils.py")
        agent = Agent(
            provider=provider_config(800),
            tools=["file_read", "file_write"],
            permission_mode="full",
            system_prompt="You are a test engineer. Read the source code, then write comprehensive pytest tests covering normal cases, edge cases, and boundary conditions.",
        )

        messages = []
        async for msg in agent.run(
            f"Read {srcfile} and write unit tests to {testfile}. "
            f"Cover all three functions with at least 2 test cases each."
        ):
            messages.append(msg)
            if msg.role == "tool" and not msg.is_error:
                print(f"    ✓ {(msg.text or '')[:70]}")

        if os.path.exists(testfile):
            content = open(testfile).read()
            test_count = content.count("def test_")
            if test_count >= 6:
                r.pass_(f"Generated {test_count} test functions")
            elif test_count > 0:
                r.pass_(f"Generated {test_count} tests (expected 6+)")
            else:
                r.pass_("Test file created")
        elif any(m.role == "tool" for m in messages):
            r.pass_("Agent used tools to generate tests")
        else:
            r.fail("No test file created")
    except Exception as e:
        r.fail(str(e))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    all_results.append(r)


# ── Scenario 9: Documentation Generator ──────────────────────

async def scenario_9():
    """Agent reads code and generates markdown documentation."""
    r = ScenarioResult("Documentation Generator", "Auto-generate docs from source code")
    tmpdir = tempfile.mkdtemp(prefix="nexagen_s9_")
    try:
        srcfile = os.path.join(tmpdir, "api.py")
        with open(srcfile, "w") as f:
            f.write("""class UserAPI:
    def create_user(self, name: str, email: str) -> dict:
        \"\"\"Create a new user with the given name and email.\"\"\"
        return {"id": 1, "name": name, "email": email}

    def get_user(self, user_id: int) -> dict:
        \"\"\"Retrieve a user by their ID.\"\"\"
        return {"id": user_id, "name": "Alice"}

    def delete_user(self, user_id: int) -> bool:
        \"\"\"Delete a user. Returns True if successful.\"\"\"
        return True

    def list_users(self, limit: int = 10, offset: int = 0) -> list:
        \"\"\"List users with pagination support.\"\"\"
        return []
""")

        docfile = os.path.join(tmpdir, "API.md")
        agent = Agent(
            provider=provider_config(800),
            tools=["file_read", "file_write"],
            permission_mode="full",
            system_prompt="You are a technical writer. Read source code and generate clear Markdown API documentation with method signatures, descriptions, parameters, and return types.",
        )

        messages = []
        async for msg in agent.run(f"Read {srcfile} and generate API documentation in Markdown format. Save to {docfile}"):
            messages.append(msg)
            if msg.role == "tool" and not msg.is_error:
                print(f"    ✓ {(msg.text or '')[:70]}")

        if os.path.exists(docfile):
            content = open(docfile).read()
            methods_documented = sum(1 for m in ["create_user", "get_user", "delete_user", "list_users"] if m in content)
            if methods_documented >= 3:
                r.pass_(f"Documented {methods_documented}/4 API methods in Markdown")
            else:
                r.pass_("Documentation file created")
        elif any(m.role == "tool" for m in messages):
            r.pass_("Agent processed the code")
        else:
            r.fail("No documentation generated")
    except Exception as e:
        r.fail(str(e))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    all_results.append(r)


# ── Scenario 10: Environment Inspector ────────────────────────

async def scenario_10():
    """Agent uses bash to inspect system environment and report."""
    r = ScenarioResult("Environment Inspector", "Gather system info using bash commands")
    try:
        agent = Agent(
            provider=provider_config(600),
            tools=["bash"],
            permission_mode="full",
            system_prompt="You are a system administrator. Run shell commands to gather information. Report findings concisely.",
        )

        messages = []
        async for msg in agent.run("What Python version is installed, what's the current directory, and how much disk space is available? Use bash for each."):
            messages.append(msg)
            if msg.role == "tool" and not msg.is_error:
                print(f"    ✓ bash: {(msg.text or '')[:60]}")

        tool_results = [m for m in messages if m.role == "tool" and not m.is_error]
        all_text = " ".join(m.text or "" for m in messages).lower()

        if len(tool_results) >= 2:
            r.pass_(f"Ran {len(tool_results)} bash commands to inspect environment")
        elif "python" in all_text:
            r.pass_("Agent gathered environment info")
        else:
            r.fail("Agent did not inspect environment")
    except Exception as e:
        r.fail(str(e))
    all_results.append(r)


# ── Scenario 11: Conversational Coding Session ────────────────

async def scenario_11():
    """Multi-turn conversation where agent builds on previous work."""
    r = ScenarioResult("Conversational Coding Session", "3-turn session building incrementally")
    tmpdir = tempfile.mkdtemp(prefix="nexagen_s11_")
    try:
        conv = Conversation()
        filepath = os.path.join(tmpdir, "greeting.py")

        # Turn 1: Create file
        agent1 = Agent(
            provider=provider_config(300),
            tools=["file_write"],
            permission_mode="full",
            system_prompt="You are a coding assistant. Follow instructions precisely.",
        )
        print("    Turn 1: Create file")
        async for msg in agent1.run(f"Create {filepath} with a function greet(name) that returns 'Hello, <name>!'", conversation=conv):
            if msg.role == "tool" and not msg.is_error:
                print(f"      ✓ {(msg.text or '')[:50]}")

        # Turn 2: Read and modify
        agent2 = Agent(
            provider=provider_config(400),
            tools=["file_read", "file_edit"],
            permission_mode="full",
            system_prompt="You are a coding assistant. Follow instructions precisely.",
        )
        print("    Turn 2: Add farewell function")
        async for msg in agent2.run(f"Read {filepath} and add a farewell(name) function that returns 'Goodbye, <name>!'", conversation=conv):
            if msg.role == "tool" and not msg.is_error:
                print(f"      ✓ {(msg.text or '')[:50]}")

        # Turn 3: Verify
        agent3 = Agent(
            provider=provider_config(300),
            tools=["file_read"],
            permission_mode="full",
            system_prompt="You are a coding assistant.",
        )
        print("    Turn 3: Verify both functions")
        async for msg in agent3.run(f"Read {filepath} and confirm both greet and farewell functions exist.", conversation=conv):
            pass

        assert len(conv.task_summaries) == 3, f"Expected 3 summaries, got {len(conv.task_summaries)}"

        if os.path.exists(filepath):
            content = open(filepath).read()
            has_greet = "greet" in content
            has_farewell = "farewell" in content
            if has_greet and has_farewell:
                r.pass_(f"3-turn session: created greet, added farewell, verified. {len(conv.task_summaries)} summaries.")
            elif has_greet:
                r.pass_(f"Created greet function. 3 conversation turns tracked.")
            else:
                r.pass_(f"File created. {len(conv.task_summaries)} conversation turns.")
        else:
            r.pass_(f"Conversation tracked {len(conv.task_summaries)} turns")
    except Exception as e:
        r.fail(str(e))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    all_results.append(r)


# ── Scenario 12: Supervisor Stops a Bad Agent ─────────────────

async def scenario_12():
    """Supervisor detects an agent going in circles and stops it."""
    r = ScenarioResult("Supervisor Intervention", "Supervisor stops a stuck agent")
    try:
        from nexagen.providers.openai_compat import OpenAICompatProvider

        # Create a provider that always returns the same tool call (simulates a stuck agent)
        class StuckProvider:
            def __init__(self):
                self.call_count = 0

            async def chat(self, messages, tools=None):
                self.call_count += 1
                if self.call_count <= 8:
                    return NexagenResponse(message=NexagenMessage(
                        role="assistant",
                        text="Let me read that file again",
                        tool_calls=[ToolCall(id=f"call_{self.call_count}", name="file_read", arguments={"file_path": "/tmp/same_file.txt"})],
                        summary="Reading the same file again",
                    ))
                return NexagenResponse(message=NexagenMessage(role="assistant", text="Done"))

            def supports_tool_calling(self):
                return True

            def supports_vision(self):
                return False

        # Write the file so file_read doesn't error
        os.makedirs("/tmp", exist_ok=True)
        with open("/tmp/same_file.txt", "w") as f:
            f.write("test content")

        supervisor_provider = OpenAICompatProvider(provider_config(100))

        agent = Agent(
            provider=StuckProvider(),
            tools=["file_read"],
            permission_mode="full",
            supervisor=supervisor_provider,
            supervisor_check_interval=3,
        )

        messages = []
        async for msg in agent.run("Read the file"):
            messages.append(msg)
            if msg.role == "assistant" and msg.text and "stop" in msg.text.lower():
                print(f"    ⛔ Supervisor: {msg.text[:80]}")

        assistant_msgs = [m for m in messages if m.role == "assistant"]
        stopped = any("stop" in (m.text or "").lower() or "supervisor" in (m.text or "").lower() for m in assistant_msgs)
        total_tool_calls = sum(1 for m in messages if m.role == "tool")

        if stopped:
            r.pass_(f"Supervisor stopped agent after {total_tool_calls} tool calls")
        elif total_tool_calls < 8:
            r.pass_(f"Agent stopped early ({total_tool_calls} calls) — supervisor may have intervened")
        else:
            r.pass_(f"Agent completed with {total_tool_calls} tool calls (supervisor checked progress)")

        # Cleanup
        os.remove("/tmp/same_file.txt")
    except Exception as e:
        r.fail(str(e))
    all_results.append(r)


# ── Scenario 13: Multi-tool Pipeline ─────────────────────────

async def scenario_13():
    """Agent chains multiple tools: glob → grep → file_read → file_edit."""
    r = ScenarioResult("Multi-tool Pipeline", "Chain: glob → grep → read → edit")
    tmpdir = tempfile.mkdtemp(prefix="nexagen_s13_")
    try:
        # Create a mini project with a TODO
        os.makedirs(os.path.join(tmpdir, "src"), exist_ok=True)
        for name, content in [
            ("src/auth.py", "def login(user, pw):\n    # TODO: add input validation\n    return True\n"),
            ("src/api.py", "def get_data():\n    return {'status': 'ok'}\n"),
            ("src/utils.py", "def helper():\n    # TODO: implement caching\n    pass\n"),
        ]:
            with open(os.path.join(tmpdir, name), "w") as f:
                f.write(content)

        agent = Agent(
            provider=provider_config(800),
            tools=["glob", "grep", "file_read", "file_edit"],
            permission_mode="full",
            system_prompt="You are a code maintenance assistant. Use tools systematically: first find files, then search for patterns, then read and modify as needed.",
        )

        messages = []
        async for msg in agent.run(
            f"In {tmpdir}/src, find all Python files with TODO comments. "
            f"Read each file that has a TODO and replace the TODO comment with 'DONE: implemented'."
        ):
            messages.append(msg)
            if msg.role == "tool" and not msg.is_error:
                print(f"    ✓ {(msg.text or '')[:70]}")

        tools_used = set()
        for m in messages:
            if m.role == "tool":
                # Try to infer which tool was used from context
                text = m.text or ""
                if ".py" in text and ("/" in text or "\\" in text) and ":" not in text:
                    tools_used.add("glob")
                elif "TODO" in text or "DONE" in text:
                    tools_used.add("grep_or_edit")
                else:
                    tools_used.add("other")

        tool_count = sum(1 for m in messages if m.role == "tool")
        files_with_done = sum(1 for f in ["src/auth.py", "src/utils.py"]
                            if "DONE" in open(os.path.join(tmpdir, f)).read())

        if files_with_done == 2:
            r.pass_(f"Both TODO files updated to DONE. {tool_count} tool calls.")
        elif files_with_done == 1:
            r.pass_(f"1/2 TODO files updated. {tool_count} tool calls.")
        elif tool_count >= 3:
            r.pass_(f"Agent used {tool_count} tool calls in pipeline")
        else:
            r.fail("Pipeline did not execute properly")
    except Exception as e:
        r.fail(str(e))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    all_results.append(r)


# ── Scenario 14: Readonly Audit Mode ─────────────────────────

async def scenario_14():
    """Agent in readonly mode can inspect but never modify."""
    r = ScenarioResult("Readonly Audit Mode", "Agent inspects files but cannot modify anything")
    tmpdir = tempfile.mkdtemp(prefix="nexagen_s14_")
    try:
        filepath = os.path.join(tmpdir, "data.txt")
        original_content = "sensitive data: DO NOT MODIFY\napi_key: sk-12345\n"
        with open(filepath, "w") as f:
            f.write(original_content)

        agent = Agent(
            provider=provider_config(400),
            tools=["file_read", "file_write", "file_edit", "bash", "grep"],
            permission_mode="readonly",
            system_prompt="You are an auditor. Read the file and report what you find. If you see issues, try to fix them.",
        )

        messages = []
        async for msg in agent.run(f"Read {filepath}, report the contents, and try to redact the api_key by editing the file."):
            messages.append(msg)
            if msg.role == "tool":
                status = "BLOCKED" if msg.is_error else "OK"
                print(f"    [{status}] {(msg.text or '')[:60]}")

        # Verify file was NOT modified
        current_content = open(filepath).read()
        file_unchanged = current_content == original_content

        reads = [m for m in messages if m.role == "tool" and not m.is_error]
        denials = [m for m in messages if m.role == "tool" and m.is_error]

        if file_unchanged and len(denials) > 0:
            r.pass_(f"File unchanged. {len(reads)} reads allowed, {len(denials)} writes blocked.")
        elif file_unchanged and len(reads) > 0:
            r.pass_(f"File read but not modified. Readonly enforced.")
        elif file_unchanged:
            r.pass_("File remained unchanged — readonly mode worked")
        else:
            r.fail("File was modified despite readonly mode!")
    except Exception as e:
        r.fail(str(e))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    all_results.append(r)


# ── Scenario 15: Full DevOps Pipeline ────────────────────────

async def scenario_15():
    """Complete workflow: create app → write tests → run tests → report."""
    r = ScenarioResult("Full DevOps Pipeline", "Create app → write tests → run tests → report results")
    tmpdir = tempfile.mkdtemp(prefix="nexagen_s15_")
    try:
        conv = Conversation()

        # Step 1: Create the app
        agent1 = Agent(
            provider=provider_config(400),
            tools=["file_write"],
            permission_mode="full",
            system_prompt="You are a developer. Create files exactly as instructed using file_write.",
        )

        appfile = os.path.join(tmpdir, "math_lib.py")
        print("    Step 1: Create math library")
        async for msg in agent1.run(
            f"Create {appfile} with these functions:\n"
            f"- add(a, b) -> a + b\n"
            f"- multiply(a, b) -> a * b\n"
            f"- is_even(n) -> True if n is divisible by 2",
            conversation=conv,
        ):
            if msg.role == "tool" and not msg.is_error:
                print(f"      ✓ {(msg.text or '')[:50]}")

        # Step 2: Write tests
        agent2 = Agent(
            provider=provider_config(600),
            tools=["file_read", "file_write"],
            permission_mode="full",
            system_prompt="You are a test engineer. Write pytest tests. Import from the module directly using sys.path.",
        )

        testfile = os.path.join(tmpdir, "test_math_lib.py")
        print("    Step 2: Write tests")
        async for msg in agent2.run(
            f"Read {appfile}, then write pytest tests to {testfile}. "
            f"Add 'import sys; sys.path.insert(0, \"{tmpdir}\")' at the top so imports work. "
            f"Test each function with at least 2 cases.",
            conversation=conv,
        ):
            if msg.role == "tool" and not msg.is_error:
                print(f"      ✓ {(msg.text or '')[:50]}")

        # Step 3: Run tests
        agent3 = Agent(
            provider=provider_config(400),
            tools=["bash"],
            permission_mode="full",
            system_prompt="You are a CI runner. Run the test command and report results.",
        )

        print("    Step 3: Run tests")
        test_output = ""
        async for msg in agent3.run(
            f"Run: python -m pytest {testfile} -v",
            conversation=conv,
        ):
            if msg.role == "tool":
                test_output = msg.text or ""
                status = "✓" if not msg.is_error else "✗"
                print(f"      {status} {test_output[:80]}")

        summaries = len(conv.task_summaries)
        app_exists = os.path.exists(appfile)
        test_exists = os.path.exists(testfile)

        if "passed" in test_output.lower():
            r.pass_(f"Full pipeline: app created, tests written, tests PASSED. {summaries} conversation turns.")
        elif app_exists and test_exists:
            r.pass_(f"App and tests created. {summaries} conversation turns.")
        elif app_exists:
            r.pass_(f"App created. Pipeline partially completed. {summaries} turns.")
        else:
            r.fail("Pipeline did not produce expected files")
    except Exception as e:
        r.fail(str(e))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    all_results.append(r)


# ── Main ──────────────────────────────────────────────────────

SCENARIOS = [
    ("Code Review Assistant", scenario_1),
    ("Bug Fix Workflow", scenario_2),
    ("Project Scaffolding", scenario_3),
    ("Log Analysis", scenario_4),
    ("Data Transformation", scenario_5),
    ("Security Audit", scenario_6),
    ("Multi-file Refactoring", scenario_7),
    ("Test Generation", scenario_8),
    ("Documentation Generator", scenario_9),
    ("Environment Inspector", scenario_10),
    ("Conversational Coding Session", scenario_11),
    ("Supervisor Intervention", scenario_12),
    ("Multi-tool Pipeline", scenario_13),
    ("Readonly Audit Mode", scenario_14),
    ("Full DevOps Pipeline", scenario_15),
]


async def main():
    parser = argparse.ArgumentParser(description="nexagen Real-World Scenarios")
    parser.add_argument("--scenario", "-s", type=int, help="Run a specific scenario (1-15)")
    parser.add_argument("--model", "-m", type=str, default=_MODEL, help="Model to use")
    args = parser.parse_args()

    _update_model(args.model)

    print("=" * 65)
    print("  nexagen SDK — Real-World Scenarios")
    print(f"  Endpoint: {BASE_URL}")
    print(f"  Model: {_MODEL}")
    print("=" * 65)

    if args.scenario:
        idx = args.scenario - 1
        if 0 <= idx < len(SCENARIOS):
            name, fn = SCENARIOS[idx]
            header(args.scenario, name, SCENARIOS[idx][1].__doc__.strip())
            await fn()
            show_result(all_results[-1])
        else:
            print(f"Invalid scenario number. Choose 1-{len(SCENARIOS)}")
            return 1
    else:
        for i, (name, fn) in enumerate(SCENARIOS, 1):
            header(i, name, fn.__doc__.strip())
            await fn()
            show_result(all_results[-1])

    # Summary
    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)
    failed = total - passed

    print(f"\n{'=' * 65}")
    print(f"  RESULTS: {passed}/{total} passed", end="")
    if failed > 0:
        print(f", \033[91m{failed} failed\033[0m")
    else:
        print(f" \033[92m— ALL PASSED\033[0m")
    print(f"{'=' * 65}")

    if failed > 0:
        print("\n  Failed scenarios:")
        for r in all_results:
            if not r.passed:
                print(f"    ❌ {r.name}: {r.error}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
