import os
import re
import shlex
from dataclasses import dataclass
from typing import Literal

CommandPolicyDecision = Literal["allowed", "blocked", "requires_approval"]

SECRET_PATTERNS = [
    (
        re.compile(r"(?i)\b(api[_-]?key|token|secret|password)\s*=\s*([^\s]+)"),
        r"\1=[REDACTED]",
    ),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"), "Bearer [REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"), "[REDACTED_SECRET]"),
]

SHELL_OPERATORS = {"|", "||", "&", "&&", ";", ">", ">>", "<", "2>", "2>>"}
BLOCKED_COMMANDS = {
    "chmod",
    "chown",
    "dd",
    "halt",
    "mkfs",
    "mount",
    "mv",
    "reboot",
    "rm",
    "shutdown",
    "sudo",
    "umount",
}
APPROVAL_COMMANDS = {
    "curl",
    "git",
    "pip",
    "python",
    "python3",
    "uv",
    "wget",
}
SAFE_PYTHON_MODULES = {"py_compile", "pytest", "ruff", "mypy"}
SAFE_DIRECT_COMMANDS = {"mypy", "pytest", "ruff"}
SAFE_GIT_SUBCOMMANDS = {"diff", "status"}


@dataclass(frozen=True)
class CommandPolicyResult:
    decision: CommandPolicyDecision
    reason: str
    args: list[str]


def redact_text(text: str) -> str:
    """Redact common secret-like values from user-visible traces."""
    redacted = text
    for pattern, replacement in SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    for pattern in _configured_secret_patterns():
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def classify_command(command: str) -> CommandPolicyResult:
    """Classify a command before execution.

    This is a controller-level policy check, not an OS sandbox.
    """
    try:
        args = shlex.split(command)
    except ValueError as e:
        raise ValueError(f"Invalid command syntax: {e}") from e

    if not args:
        raise ValueError("Command cannot be empty.")
    if any(operator in args for operator in SHELL_OPERATORS):
        return CommandPolicyResult(
            decision="blocked",
            reason="Shell operators are not supported by run_command.",
            args=args,
        )
    if "$(" in command or "`" in command:
        return CommandPolicyResult(
            decision="blocked",
            reason="Shell command substitution is not supported by run_command.",
            args=args,
        )
    if args[0] in BLOCKED_COMMANDS:
        return CommandPolicyResult(
            decision="blocked",
            reason=f"Blocked dangerous command: {args[0]}",
            args=args,
        )
    if args[:3] == ["git", "reset", "--hard"]:
        return CommandPolicyResult(
            decision="blocked",
            reason="Blocked dangerous command: git reset --hard",
            args=args,
        )
    if args[:2] == ["git", "clean"]:
        return CommandPolicyResult(
            decision="blocked",
            reason="Blocked dangerous command: git clean",
            args=args,
        )
    if (
        _is_safe_python_module_command(args)
        or _is_safe_python_version_command(args)
        or _is_safe_direct_command(args)
    ):
        return CommandPolicyResult(
            decision="allowed",
            reason="Command matches the safe coding-agent command policy.",
            args=args,
        )
    if _requires_approval(args):
        return CommandPolicyResult(
            decision="requires_approval",
            reason="Command requires approval because it may have broad side effects.",
            args=args,
        )
    return CommandPolicyResult(
        decision="requires_approval",
        reason="Command is outside the automatic safe command policy.",
        args=args,
    )


def _configured_secret_patterns() -> list[re.Pattern[str]]:
    raw_patterns = os.getenv("AGENT_TRACE_REDACT_PATTERNS", "")
    patterns: list[re.Pattern[str]] = []
    for raw_pattern in raw_patterns.splitlines():
        pattern = raw_pattern.strip()
        if not pattern:
            continue
        try:
            patterns.append(re.compile(pattern))
        except re.error:
            continue
    return patterns


def _is_safe_python_module_command(args: list[str]) -> bool:
    if len(args) < 3:
        return False
    executable = args[0].split("/")[-1]
    if executable not in {"python", "python3"} and not executable.endswith("python"):
        return False
    return args[1] == "-m" and args[2] in SAFE_PYTHON_MODULES


def _is_safe_python_version_command(args: list[str]) -> bool:
    if len(args) != 2:
        return False
    executable = args[0].split("/")[-1]
    if executable not in {"python", "python3"} and not executable.endswith("python"):
        return False
    return args[1] in {"--version", "-V"}


def _is_safe_direct_command(args: list[str]) -> bool:
    if args[0] == "git":
        return len(args) >= 2 and args[1] in SAFE_GIT_SUBCOMMANDS
    return args[0] in SAFE_DIRECT_COMMANDS


def _requires_approval(args: list[str]) -> bool:
    if args[0] in APPROVAL_COMMANDS:
        return True
    executable = args[0].split("/")[-1]
    return executable in APPROVAL_COMMANDS
