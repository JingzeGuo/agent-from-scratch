import asyncio
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import pytest

from agent.schemas import ConsolidatedState, ContextBuildResult
from agent.state.context import (
    CONSOLIDATED_STATE_HEADER,
    ContextBudgetExceeded,
    ContextBuilder,
    ContextConfig,
    Message,
    count_tokens,
)


def make_state(
    objective: str | None = None, status: str = "Work in progress"
) -> ConsolidatedState:
    return ConsolidatedState(
        current_objective=objective,
        current_status=status,
        findings=["Root cause confirmed"],
        decisions=[],
        files_changed=["app.py"],
        unresolved=["Verify fix"],
        verification=["Earlier tests passed"],
        important_context=[],
    )


class FakeConsolidator:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[
            tuple[ConsolidatedState | None, list[Message], str | None]
        ] = []
        self.error = error

    async def consolidate(
        self,
        previous_state: ConsolidatedState | None,
        history_prefix: list[Message],
        current_objective: str | None,
        *,
        validate_state: Callable[[ConsolidatedState], None] | None = None,
    ) -> ConsolidatedState:
        self.calls.append((previous_state, deepcopy(history_prefix), current_objective))
        if self.error:
            raise self.error
        state = make_state(current_objective, f"Fold {len(self.calls)}")
        if validate_state:
            validate_state(state)
        return state


def task(name: str, size: int = 200) -> list[Message]:
    return [
        {"role": "user", "content": name},
        {"role": "assistant", "content": "observation " * size},
    ]


def exchange(name: str, size: int = 100) -> list[Message]:
    return [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": name,
                    "name": "run_command",
                    "input": {"command": "pytest -q"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": name,
                    "content": "exit_code: 1\n"
                    + "output " * size
                    + "\nAssertionError: expected 2",
                    "is_error": True,
                }
            ],
        },
    ]


def build(
    builder: ContextBuilder, messages: list[Message], **kwargs: Any
) -> ContextBuildResult:
    return asyncio.run(builder.build_with_metadata(messages, **kwargs))


def test_short_session_is_raw_and_copied_without_consolidation() -> None:
    fake = FakeConsolidator()
    builder = ContextBuilder(fake)
    messages = task("Fix bug", 30) + exchange("test", 20)
    original = deepcopy(messages)
    result = build(builder, messages, objective="Fix bug", task_starts=[0])
    assert result.messages == original
    assert result.messages is not messages
    result.messages[-1]["content"][0]["content"] = "changed copy"
    assert messages == original
    assert fake.calls == []
    assert not result.summary_included
    assert not result.hard_collapsed


def test_multiple_completed_tasks_stay_raw_below_soft_threshold() -> None:
    fake = FakeConsolidator()
    builder = ContextBuilder(fake)
    messages: list[Message] = []
    for i in range(4):
        messages.extend(task(f"Task {i}", 40))
        result = build(builder, messages, task_starts=list(range(0, len(messages), 2)))
        assert result.messages == messages
    assert fake.calls == []


def test_huge_recent_tool_result_retains_metadata_head_and_tail_independently() -> None:
    fake = FakeConsolidator()
    config = ContextConfig(max_tool_result_tokens=500, retained_tool_result_tokens=100)
    builder = ContextBuilder(fake, config)
    messages = task("Run tests", 1) + exchange("normal", 50) + exchange("huge", 2000)
    original = deepcopy(messages)
    result = build(builder, messages)
    assert result.messages[:-1] == messages[:-1]
    block = result.messages[-1]["content"][0]
    assert block["tool_use_id"] == "huge"
    assert block["is_error"] is True
    assert "exit_code: 1" in block["content"]
    assert "AssertionError: expected 2" in block["content"]
    assert "middle omitted" in block["content"]
    assert "pytest -q" in str(result.messages[-2])
    assert result.snipped_tool_results == 1
    assert fake.calls == []
    assert messages == original


def test_soft_pressure_folds_only_oldest_completed_task_and_preserves_active() -> None:
    fake = FakeConsolidator()
    builder = ContextBuilder(
        fake, ContextConfig(usable_context_tokens=1000, state_reserve_tokens=100)
    )
    messages = task("Task1", 300) + task("Task2", 200) + task("Active", 150)
    result = build(builder, messages, task_starts=[0, 2, 4], objective="Active")
    assert len(fake.calls) == 1
    assert fake.calls[0] == (None, messages[:2], "Active")
    assert result.messages[1:] == messages[2:]
    assert result.final_context_tokens < 650
    assert result.folded_message_count == 2
    assert not result.hard_collapsed


def test_additional_completed_tasks_fold_only_when_needed() -> None:
    fake = FakeConsolidator()
    builder = ContextBuilder(
        fake, ContextConfig(usable_context_tokens=1000, state_reserve_tokens=100)
    )
    messages = (
        task("Task1", 300)
        + task("Task2", 300)
        + task("Task3", 200)
        + task("Active", 150)
    )
    result = build(builder, messages, task_starts=[0, 2, 4, 6])
    assert result.messages[1:] == messages[4:]
    assert [m for _, prefix, _ in fake.calls for m in prefix] == messages[:4]
    assert result.final_context_tokens < 650
    assert result.folded_message_count == 4


def test_second_pressure_event_recursively_replaces_one_state() -> None:
    fake = FakeConsolidator()
    builder = ContextBuilder(
        fake, ContextConfig(usable_context_tokens=1000, state_reserve_tokens=100)
    )
    messages = task("Task1", 300) + task("Task2", 200) + task("Task3", 150)
    first = build(builder, messages, task_starts=[0, 2, 4])
    first_state = builder.state.consolidated_state
    assert first.folded_message_count == 2
    # Rebuilding without new pressure neither reconsolidates nor reintroduces history.
    assert build(builder, messages, task_starts=[0, 2, 4]).messages == first.messages
    assert len(fake.calls) == 1
    messages.extend(task("Task4", 200))
    second = build(builder, messages, task_starts=[0, 2, 4, 6])
    assert len(fake.calls) == 2
    assert fake.calls[1] == (first_state, messages[2:4], "Task4")
    assert second.messages[1:] == messages[4:]
    assert (
        sum(CONSOLIDATED_STATE_HEADER in str(m["content"]) for m in second.messages)
        == 1
    )
    assert builder.state.consolidated_state != first_state
    assert len(messages) == 8


def test_long_active_task_folds_old_prefix_at_complete_tool_boundary() -> None:
    fake = FakeConsolidator()
    builder = ContextBuilder(
        fake, ContextConfig(usable_context_tokens=1200, recent_message_count=3)
    )
    messages = [{"role": "user", "content": "Long task"}]
    for i in range(8):
        messages.extend(exchange(str(i), 100))
    result = build(builder, messages, task_starts=[0], objective="Long task")
    end = result.folded_message_count
    assert 0 < end <= len(messages) - 3
    assert messages[end]["role"] == "assistant"
    assert result.messages[1:] == messages[end:]
    assert result.messages[-4:] == messages[-4:]
    assert result.final_context_tokens < 780
    assert not result.hard_collapsed


def test_completed_tasks_are_exhausted_before_active_prefix() -> None:
    fake = FakeConsolidator()
    builder = ContextBuilder(
        fake, ContextConfig(usable_context_tokens=1200, recent_message_count=2)
    )
    messages = task("Done", 250) + [{"role": "user", "content": "Active"}]
    for i in range(7):
        messages.extend(exchange(str(i), 100))
    result = build(builder, messages, task_starts=[0, 2])
    assert result.folded_message_count > 2
    assert fake.calls[0][1][:2] == messages[:2]
    assert result.messages[-2:] == messages[-2:]
    assert result.final_context_tokens < 780


def test_failure_hard_collapses_without_committing_or_mutating_history() -> None:
    fake = FakeConsolidator(RuntimeError("provider unavailable"))
    builder = ContextBuilder(fake, ContextConfig(usable_context_tokens=1000))
    messages = task("Old", 700) + task("Active", 100) + exchange("latest", 40)
    original = deepcopy(messages)
    result = build(builder, messages, task_starts=[0, 2])
    assert result.hard_collapsed
    assert result.consolidation_error == "Consolidation failed (RuntimeError)"
    assert result.final_context_tokens < 900
    assert result.messages[-2:] == messages[-2:]
    assert "Current objective: Active" in str(result.messages[0])
    assert builder.state.folded_message_count == 0
    assert builder.state.consolidated_state is None
    assert messages == original


def test_emergency_after_success_keeps_latest_complete_exchange() -> None:
    fake = FakeConsolidator()
    builder = ContextBuilder(
        fake, ContextConfig(usable_context_tokens=1000, recent_message_count=6)
    )
    messages = task("Old", 500) + [{"role": "user", "content": "Active"}]
    for i in range(4):
        messages.extend(exchange(str(i), 250))
    result = build(builder, messages, task_starts=[0, 2])
    assert fake.calls
    assert result.summary_included
    assert result.hard_collapsed
    assert result.final_context_tokens < 900
    assert result.messages[-2:] == messages[-2:]


def test_emergency_refuses_indivisible_over_budget_exchange() -> None:
    builder = ContextBuilder(config=ContextConfig(usable_context_tokens=1000))
    with pytest.raises(ContextBudgetExceeded):
        build(builder, [{"role": "user", "content": "Active"}, *exchange("huge", 1200)])


def test_token_pressure_includes_request_overhead() -> None:
    fake = FakeConsolidator()
    builder = ContextBuilder(fake, ContextConfig(usable_context_tokens=1000))
    messages = task("Old", 300) + task("Active", 100)
    assert build(builder, messages).messages == messages
    result = build(builder, messages, request_overhead_tokens=250)
    assert len(fake.calls) == 1
    assert result.final_context_tokens < 650


def test_token_count_is_not_character_count_and_handles_special_text() -> None:
    assert count_tokens("hello world") == 2
    assert count_tokens("<|endoftext|> 中文") > 0


def test_exact_soft_threshold_triggers_consolidation() -> None:
    fake = FakeConsolidator()
    messages = task("Old", 300) + task("Active", 100)
    tokens = ContextBuilder().measure_tokens(messages)
    builder = ContextBuilder(
        fake, ContextConfig(usable_context_tokens=tokens * 2, soft_threshold=0.5)
    )
    result = build(builder, messages)
    assert len(fake.calls) == 1
    assert result.final_context_tokens < tokens


def test_oversized_generated_state_is_rejected_before_commit() -> None:
    from unittest.mock import AsyncMock

    fake = AsyncMock()
    fake.consolidate.return_value = make_state("Active", "verbose " * 1000)
    builder = ContextBuilder(fake, ContextConfig(usable_context_tokens=1000))
    messages = task("Old", 2000) + task("Active", 100)
    result = build(builder, messages)
    assert result.hard_collapsed
    assert builder.state.consolidated_state is None
    assert result.messages[-2:] == messages[-2:]


def test_folding_preserves_all_results_of_multi_tool_request() -> None:
    fake = FakeConsolidator()
    builder = ContextBuilder(
        fake, ContextConfig(usable_context_tokens=1000, recent_message_count=2)
    )
    messages = task("Active", 700)
    a, b = exchange("a", 20), exchange("b", 20)
    messages.extend(
        [
            {"role": "assistant", "content": [*a[0]["content"], *b[0]["content"]]},
            a[1],
            b[1],
        ]
    )
    result = build(builder, messages, task_starts=[0])
    assert result.messages[-3:] == messages[-3:]
    assert result.folded_message_count == 2


def test_inspection_does_not_call_llm_even_under_pressure() -> None:
    fake = FakeConsolidator()
    builder = ContextBuilder(fake, ContextConfig(usable_context_tokens=1000))
    messages = task("Old", 700) + task("Active", 100)
    assert builder.inspect(messages).messages == messages
    assert fake.calls == []


@pytest.mark.parametrize("reserve, expected_calls", [(0, 2), (1024, 1)])
def test_state_reserve_avoids_repeated_folding_calls(
    reserve: int, expected_calls: int
) -> None:
    from unittest.mock import AsyncMock

    fake = AsyncMock()
    fake.consolidate.return_value = make_state("Active", "detail " * 1000)
    builder = ContextBuilder(fake, ContextConfig(state_reserve_tokens=reserve))
    messages = (
        task("Task1", 4000)
        + task("Task2", 7000)
        + task("Task3", 6000)
        + task("Active", 7000)
    )
    result = build(builder, messages)
    assert fake.consolidate.call_count == expected_calls
    assert result.folded_message_count == 4
    assert result.messages[1:] == messages[4:]
    assert result.final_context_tokens < 32000 * 0.65
    assert not result.hard_collapsed


def test_state_limit_counts_objective_and_framing_and_accepts_exact_limit() -> None:
    from unittest.mock import AsyncMock

    objective = "Active " * 100
    expected = make_state(objective)
    probe = ContextBuilder()
    tokens = probe.measure_tokens([probe._state_message(expected)])
    assert tokens > count_tokens(expected.model_dump_json())
    for limit, rejected in [(tokens, False), (tokens - 1, True)]:
        fake = AsyncMock()
        fake.consolidate.return_value = make_state("Wrong old objective")
        builder = ContextBuilder(
            fake,
            ContextConfig(
                usable_context_tokens=1000,
                max_consolidated_state_tokens=limit,
                state_reserve_tokens=100,
            ),
        )
        result = build(
            builder, task("Old", 700) + task(objective, 10), objective=objective
        )
        assert result.hard_collapsed is rejected
        assert (builder.state.consolidated_state is None) is rejected


def test_reset_clears_folded_state() -> None:
    builder = ContextBuilder(
        FakeConsolidator(), ContextConfig(usable_context_tokens=1000)
    )
    build(builder, task("Old", 700) + task("Active", 100))
    assert builder.state.consolidated_state is not None
    builder.reset()
    messages = task("New", 10)
    assert build(builder, messages).messages == messages
    assert builder.state.folded_message_count == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"soft_threshold": 0.95},
        {"usable_context_tokens": 0},
        {"recent_message_count": 0},
        {"retained_tool_result_tokens": 20000},
        {"max_consolidated_state_tokens": 0},
        {"state_reserve_tokens": -1},
        {"state_reserve_tokens": 2049},
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ContextConfig(**kwargs)
