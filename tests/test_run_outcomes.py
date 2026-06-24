from agent.schemas import AgentRun, RunOutcome, VerificationEvidence
from agent.verification import infer_task_success


def make_run(
    *,
    termination: RunOutcome,
    final_stop_reason: str | None,
    verification: VerificationEvidence,
) -> AgentRun:
    return AgentRun(
        objective="Test objective",
        steps=[],
        termination=termination,
        final_stop_reason=final_stop_reason,
        verification=verification,
        task_success=infer_task_success(verification),
    )


def test_run_outcome_is_separate_from_provider_stop_reason() -> None:
    completed = make_run(
        termination="completed",
        final_stop_reason="end_turn",
        verification=VerificationEvidence(status="not_run"),
    )
    protocol_error = make_run(
        termination="protocol_error",
        final_stop_reason="max_tokens",
        verification=VerificationEvidence(status="not_run"),
    )

    assert completed.termination == "completed"
    assert completed.final_stop_reason == "end_turn"
    assert protocol_error.termination == "protocol_error"
    assert protocol_error.final_stop_reason == "max_tokens"


def test_failed_verification_is_negative_task_evidence() -> None:
    run = make_run(
        termination="completed",
        final_stop_reason="end_turn",
        verification=VerificationEvidence(
            status="failed",
            command="pytest",
            exit_code=1,
            output="1 failed",
        ),
    )

    assert run.verification.status == "failed"
    assert run.task_success is False


def test_passed_verification_does_not_prove_task_success() -> None:
    run = make_run(
        termination="completed",
        final_stop_reason="end_turn",
        verification=VerificationEvidence(
            status="passed",
            command="python -m py_compile app.py",
            exit_code=0,
            output="syntax ok",
        ),
    )

    assert run.verification.status == "passed"
    assert run.task_success is None


def test_no_verification_does_not_prove_task_success() -> None:
    run = make_run(
        termination="completed",
        final_stop_reason="end_turn",
        verification=VerificationEvidence(status="not_run"),
    )

    assert run.verification.status == "not_run"
    assert run.task_success is None


def test_day_9_outcome_vocabulary_is_available() -> None:
    outcomes: list[RunOutcome] = [
        "completed",
        "max_steps",
        "interrupted",
        "blocked",
        "refused",
        "protocol_error",
    ]

    assert outcomes == [
        "completed",
        "max_steps",
        "interrupted",
        "blocked",
        "refused",
        "protocol_error",
    ]
