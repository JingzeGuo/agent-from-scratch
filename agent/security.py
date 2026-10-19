import os
import re
import shlex
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

from .sandbox import sandbox_available

CommandPolicyDecision = Literal["allowed", "requires_approval"]

SECRET_PATTERNS = [
    (
        re.compile(r"(?i)\b(api[_-]?key|token|secret|password)\s*=\s*([^\s]+)"),
        r"\1=[REDACTED]",
    ),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"), "Bearer [REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"), "[REDACTED_SECRET]"),
]


@dataclass(frozen=True)
class ToolApprovalPolicy:
    decision: CommandPolicyDecision
    reason: str
    args: list[str] = field(default_factory=list)


def redact_text(text: str) -> str:
    """Redact common secret-like values from user-visible traces."""
    redacted = text
    for pattern, replacement in SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    for pattern in _configured_secret_patterns():
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def classify_command(command: str) -> ToolApprovalPolicy:
    """Classify a command before execution.

    Sandboxed commands run automatically; without a sandbox, every command
    requires approval.
    """
    try:
        args = shlex.split(command)
    except ValueError as e:
        raise ValueError(f"Invalid command syntax: {e}") from e

    if not args:
        raise ValueError("Command cannot be empty.")
    if sandbox_available():
        return ToolApprovalPolicy(
            decision="allowed",
            reason="Command runs inside the OS sandbox.",
            args=args,
        )
    return ToolApprovalPolicy(
        decision="requires_approval",
        reason="Command requires approval because run_command is not sandboxed.",
        args=args,
    )


def _configured_secret_patterns() -> tuple[re.Pattern[str], ...]:
    return _compile_secret_patterns(os.getenv("AGENT_TRACE_REDACT_PATTERNS", ""))


@lru_cache(maxsize=8)
def _compile_secret_patterns(raw_patterns: str) -> tuple[re.Pattern[str], ...]:
    patterns: list[re.Pattern[str]] = []
    for raw_pattern in raw_patterns.splitlines():
        pattern = raw_pattern.strip()
        if not pattern:
            continue
        try:
            patterns.append(re.compile(pattern))
        except re.error:
            continue
    return tuple(patterns)
