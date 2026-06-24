from agent.schemas import AgentStep, ToolCall, ToolResult
from agent.verification import extract_verification_evidence, infer_task_success


def make_run_command_step(
    *,
    command: str = "pytest",
    content: str,
    is_error: bool = False,
    step_number: int = 1,
) -> AgentStep:
    return AgentStep(
        step_number=step_number,
        stop_reason="tool_use",
        tool_calls=[
            ToolCall(
                name="run_command",
                input={"command": command},
                tool_use_id=f"toolu_{step_number}",
            )
        ],
        tool_results=[
            ToolResult(
                tool_use_id=f"toolu_{step_number}",
                content=content,
                is_error=is_error,
            )
        ],
    )


def command_result(exit_code: str, timed_out: str = "false") -> str:
    return "\n".join(
        [
            f"exit_code: {exit_code}",
            f"timed_out: {timed_out}",
            "duration_seconds: 0.010",
            "stdout:",
            "ok",
            "stderr:",
            "[empty]",
        ]
    )


def test_extract_verification_returns_not_run_without_command() -> None:
    evidence = extract_verification_evidence([])

    assert evidence.status == "not_run"
    assert evidence.command is None
    assert evidence.exit_code is None
    assert evidence.output is None
    assert infer_task_success(evidence) is None


def test_extract_verification_passed_command() -> None:
    output = command_result("0")

    evidence = extract_verification_evidence(
        [make_run_command_step(command="pytest tests/test_tools.py", content=output)]
    )

    assert evidence.status == "passed"
    assert evidence.command == "pytest tests/test_tools.py"
    assert evidence.exit_code == 0
    assert evidence.output == output
    assert infer_task_success(evidence) is None


def test_extract_verification_failed_command() -> None:
    output = command_result("1")

    evidence = extract_verification_evidence([make_run_command_step(content=output)])

    assert evidence.status == "failed"
    assert evidence.exit_code == 1
    assert evidence.output == output
    assert infer_task_success(evidence) is False


def test_extract_verification_timeout_is_error() -> None:
    output = command_result("null", timed_out="true")

    evidence = extract_verification_evidence([make_run_command_step(content=output)])

    assert evidence.status == "error"
    assert evidence.exit_code is None
    assert evidence.output == output
    assert infer_task_success(evidence) is None


def test_extract_verification_tool_error_is_error() -> None:
    evidence = extract_verification_evidence(
        [
            make_run_command_step(
                content="Tool 'run_command' raised ValueError: bad command",
                is_error=True,
            )
        ]
    )

    assert evidence.status == "error"
    assert evidence.exit_code is None
    assert "bad command" in (evidence.output or "")
    assert infer_task_success(evidence) is None


def test_extract_verification_uses_latest_run_command() -> None:
    failed_output = command_result("1")
    passed_output = command_result("0")

    evidence = extract_verification_evidence(
        [
            make_run_command_step(content=failed_output, step_number=1),
            make_run_command_step(content=passed_output, step_number=2),
        ]
    )

    assert evidence.status == "passed"
    assert evidence.exit_code == 0
    assert evidence.output == passed_output
    assert infer_task_success(evidence) is None
