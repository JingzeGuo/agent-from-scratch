"""Bounded shell command execution inside the workspace sandbox."""

import os
import re
import shlex
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from ..sandbox import SHELL, sandbox_argv
from ..security import APPROVAL_REASON, command_requires_approval
from ..workspace import resolve_workspace_path

SECRET_ENV_NAME = re.compile(r"(^|_)(API_KEY|TOKEN|SECRET|PASSWORD)($|_)", re.I)
# A failure anywhere in a pipeline fails the command, so `pytest ... | tail` cannot
# report exit code 0 for failing tests. Probed in a subshell first because shells
# without pipefail (older dash) exit on an unknown `set` option.
PIPEFAIL_PREFIX = "(set -o pipefail) 2>/dev/null && set -o pipefail\n"
# Builds the argv for (command, resolved working directory, timeout seconds).
# Used to run commands somewhere other than the local OS sandbox, e.g. a container.
CommandLauncher = Callable[[str, Path, float], list[str]]


def run_command(
    command: str,
    *,
    workspace_root: Path,
    cwd: str | None = None,
    timeout_seconds: float = 10.0,
    max_output_chars: int = 8000,
    approval_granted: bool = False,
    launcher: CommandLauncher | None = None,
) -> str:
    """Run a bounded shell command inside the workspace and return its observation."""
    if command_requires_approval() and not approval_granted:
        raise PermissionError(APPROVAL_REASON)
    command_cwd = _resolve_command_cwd(workspace_root, cwd)
    command = PIPEFAIL_PREFIX + _resolve_workspace_local_executable(
        command, workspace_root
    )
    if launcher is not None:
        args = launcher(command, command_cwd, timeout_seconds)
    else:
        args = sandbox_argv(command, workspace_root) or [SHELL, "-c", command]

    start = time.monotonic()
    # A new session lets a timeout kill the shell's children along with it.
    with subprocess.Popen(
        args,
        cwd=command_cwd,
        env=_command_env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            timed_out = False
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            timed_out = True
    return _format_command_result(
        exit_code=None if timed_out else process.returncode,
        timed_out=timed_out,
        duration_seconds=time.monotonic() - start,
        stdout=stdout,
        stderr=stderr,
        max_output_chars=max_output_chars,
    )


def _command_env() -> dict[str, str]:
    """Inherit the environment minus credentials such as provider API keys."""
    return {
        key: value
        for key, value in os.environ.items()
        if not SECRET_ENV_NAME.search(key)
    }


def _resolve_command_cwd(workspace_root: Path, cwd: str | None) -> Path:
    root = workspace_root.expanduser().resolve()
    if cwd is None:
        return root
    resolved = resolve_workspace_path(root, cwd)
    if not resolved.exists():
        raise FileNotFoundError(f"Working directory not found: {cwd}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Working directory is not a directory: {cwd}")
    return resolved


def _resolve_workspace_local_executable(command: str, workspace_root: Path) -> str:
    """Let `.venv/...` commands work from any cwd inside the workspace."""
    if not command.startswith(".venv/"):
        return command
    root = workspace_root.expanduser().resolve()
    return f"{shlex.quote(root.as_posix())}/{command}"


def _format_command_result(
    *,
    exit_code: int | None,
    timed_out: bool,
    duration_seconds: float,
    stdout: str,
    stderr: str,
    max_output_chars: int,
) -> str:
    exit_code_text = "null" if exit_code is None else str(exit_code)
    return "\n".join(
        [
            f"exit_code: {exit_code_text}",
            f"timed_out: {str(timed_out).lower()}",
            f"duration_seconds: {duration_seconds:.3f}",
            "stdout:",
            _truncate_output(stdout, max_output_chars),
            "stderr:",
            _truncate_output(stderr, max_output_chars),
        ]
    )


def _truncate_output(text: str, max_chars: int) -> str:
    if text == "":
        return "[empty]"
    if len(text) <= max_chars:
        return text.rstrip("\n")

    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars
    omitted = len(text) - max_chars
    return (
        text[:head_chars].rstrip("\n")
        + f"\n[... truncated {omitted} chars ...]\n"
        + text[-tail_chars:].lstrip("\n").rstrip("\n")
    )
