"""Bounded, shell-free command execution inside the workspace."""

import subprocess
import time
from pathlib import Path

from ..security import classify_command
from ..workspace import resolve_workspace_path


def run_command(
    command: str,
    *,
    workspace_root: Path,
    cwd: str | None = None,
    timeout_seconds: float = 10.0,
    max_output_chars: int = 8000,
    approval_granted: bool = False,
) -> str:
    """Run a bounded command inside the workspace and return its observation."""
    args = _parse_safe_command(command, approval_granted)
    command_cwd = _resolve_command_cwd(workspace_root, cwd)
    args = _resolve_workspace_local_executable(args, workspace_root)

    start = time.monotonic()
    try:
        completed = subprocess.run(
            args,
            cwd=command_cwd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        duration = time.monotonic() - start
        return _format_command_result(
            exit_code=completed.returncode,
            timed_out=False,
            duration_seconds=duration,
            stdout=completed.stdout,
            stderr=completed.stderr,
            max_output_chars=max_output_chars,
        )
    except subprocess.TimeoutExpired as e:
        duration = time.monotonic() - start
        return _format_command_result(
            exit_code=None,
            timed_out=True,
            duration_seconds=duration,
            stdout=_normalize_subprocess_output(e.stdout),
            stderr=_normalize_subprocess_output(e.stderr),
            max_output_chars=max_output_chars,
        )


def _parse_safe_command(command: str, approval_granted: bool = False) -> list[str]:
    policy = classify_command(command)
    if policy.decision == "blocked":
        raise ValueError(policy.reason)
    if policy.decision == "requires_approval" and not approval_granted:
        raise ValueError(policy.reason)
    return policy.args


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


def _resolve_workspace_local_executable(
    args: list[str],
    workspace_root: Path,
) -> list[str]:
    executable = args[0]
    if not executable.startswith(".venv/"):
        return args
    resolved = workspace_root.expanduser().resolve() / executable
    if not resolved.exists():
        return args
    return [resolved.as_posix(), *args[1:]]


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


def _normalize_subprocess_output(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output
