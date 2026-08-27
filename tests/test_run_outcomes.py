from agent.schemas import AgentRun, RunOutcome


def make_run(
    *,
    termination: RunOutcome,
    final_stop_reason: str | None,
) -> AgentRun:
    return AgentRun(
        objective="Test objective",
        steps=[],
        termination=termination,
        final_stop_reason=final_stop_reason,
    )


def test_run_outcome_is_separate_from_provider_stop_reason() -> None:
    completed = make_run(
        termination="completed",
        final_stop_reason="end_turn",
    )
    protocol_error = make_run(
        termination="protocol_error",
        final_stop_reason="max_tokens",
    )

    assert completed.termination == "completed"
    assert completed.final_stop_reason == "end_turn"
    assert protocol_error.termination == "protocol_error"
    assert protocol_error.final_stop_reason == "max_tokens"

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
