import shlex
import sys
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest

from agent.sandbox import sandbox_available
from agent.tooling.catalog import create_registry
from agent.tooling.tool_registry import ToolRegistry

pytestmark = pytest.mark.anyio
sandboxed = pytest.mark.skipif(not sandbox_available(), reason="no OS sandbox")
PYTHON = shlex.quote(sys.executable)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".agents").mkdir()
    return tmp_path


@pytest.fixture
def outside() -> Iterator[Path]:
    # Temporary directories stay writable, so probe the home directory instead.
    path = Path.home() / f".agent-sandbox-probe-{uuid4().hex}"
    yield path
    path.unlink(missing_ok=True)


async def run(registry: ToolRegistry, command: str) -> str:
    output, _ = await registry.execute("run_command", {"command": command})
    return output


@sandboxed
async def test_shell_operators_run_inside_workspace(workspace: Path) -> None:
    output = await run(
        create_registry(workspace), "echo hi | tr a-z A-Z > out.txt && cat out.txt"
    )

    assert "stdout:\nHI" in output


@sandboxed
async def test_writes_outside_workspace_fail(workspace: Path, outside: Path) -> None:
    await run(create_registry(workspace), f"touch {shlex.quote(str(outside))}")

    assert not outside.exists()


@sandboxed
@pytest.mark.parametrize("path", [".git/hooks-pre-commit", ".agents/memory.sqlite3"])
async def test_protected_dirs_are_read_only(workspace: Path, path: str) -> None:
    await run(create_registry(workspace), f"touch {path}")

    assert not (workspace / path).exists()


@sandboxed
async def test_network_is_unavailable(workspace: Path) -> None:
    connect = "import socket; socket.create_connection(('1.1.1.1', 80), timeout=3)"
    output = await run(create_registry(workspace), f'{PYTHON} -c "{connect}"')

    assert "exit_code: 1" in output


@sandboxed
async def test_malicious_conftest_cannot_escape(workspace: Path, outside: Path) -> None:
    (workspace / "conftest.py").write_text(
        f"from pathlib import Path\nPath({str(outside)!r}).touch()\n", encoding="utf-8"
    )

    await run(create_registry(workspace), f"{PYTHON} -m pytest -q")

    assert not outside.exists()


async def test_commands_do_not_inherit_api_keys(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-should-not-leak")
    output, _ = await create_registry(workspace).execute(
        "run_command",
        {"command": "env"},
        approval_granted=True,
    )

    assert "sk-should-not-leak" not in output


@pytest.mark.parametrize("path", [".git/hooks/pre-commit", ".agents/notes.txt"])
async def test_file_tools_cannot_write_protected_dirs(
    workspace: Path, path: str
) -> None:
    output, is_error = await create_registry(workspace).execute(
        "write_file", {"path": path, "content": "x"}
    )

    assert "Path is read-only" in output
    assert is_error is True
