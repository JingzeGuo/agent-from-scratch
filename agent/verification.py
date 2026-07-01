from .schemas import AgentStep, ToolCall, ToolResult, VerificationEvidence

ENVIRONMENT_FAILURE_PATTERNS = (
    "no module named pytest",
    "no module named ruff",
    "no module named mypy",
    "pytest: command not found",
    "ruff: command not found",
    "mypy: command not found",
    "python: command not found",
    "python3: command not found",
    "no such file or directory: 'python'",
    "no such file or directory: 'python3'",
)


def extract_verification_evidence(
    steps: list[AgentStep],
) -> VerificationEvidence:
    latest = _latest_run_command_result(steps)
    if latest is None:
        return VerificationEvidence(status="not_run")

    tool_call, tool_result = latest
    command = _command_text(tool_call)
    output = tool_result.content

    if tool_result.is_error:
        return VerificationEvidence(
            status="error",
            command=command,
            output=output,
        )

    timed_out = _parse_timed_out(output)
    exit_code = _parse_exit_code(output)
    if timed_out is True or exit_code is None:
        return VerificationEvidence(
            status="error",
            command=command,
            exit_code=exit_code,
            output=output,
        )
    if exit_code == 0:
        return VerificationEvidence(
            status="passed",
            command=command,
            exit_code=exit_code,
            output=output,
        )
    if _is_environment_failure(output):
        return VerificationEvidence(
            status="error",
            command=command,
            exit_code=exit_code,
            output=output,
        )
    return VerificationEvidence(
        status="failed",
        command=command,
        exit_code=exit_code,
        output=output,
    )


def infer_task_success(verification: VerificationEvidence) -> bool | None:
    if verification.status == "failed":
        return False
    return None


def _latest_run_command_result(
    steps: list[AgentStep],
) -> tuple[ToolCall, ToolResult] | None:
    latest: tuple[ToolCall, ToolResult] | None = None
    for step in steps:
        for tool_call, tool_result in zip(step.tool_calls, step.tool_results):
            if tool_call.name == "run_command":
                latest = tool_call, tool_result
    return latest


def _command_text(tool_call: ToolCall) -> str | None:
    command = tool_call.input.get("command")
    if isinstance(command, str):
        return command
    return None


def _parse_exit_code(output: str) -> int | None:
    value = _field_value(output, "exit_code")
    if value is None or value == "null":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_timed_out(output: str) -> bool | None:
    value = _field_value(output, "timed_out")
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def _field_value(output: str, field: str) -> str | None:
    prefix = f"{field}:"
    for line in output.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip()
    return None


def _is_environment_failure(output: str) -> bool:
    normalized = output.lower()
    return any(pattern in normalized for pattern in ENVIRONMENT_FAILURE_PATTERNS)
