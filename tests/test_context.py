from typing import Any, cast

from anthropic.types import MessageParam

from agent.context import OMITTED_TOOL_RESULT_TEMPLATE, ContextBuilder
from agent.schemas import AgentStep, PendingAction, ToolCall, ToolResult


def first_content_block(message: MessageParam) -> dict[str, Any]:
    content = message["content"]
    assert isinstance(content, list)
    block = content[0]
    assert isinstance(block, dict)
    return cast(dict[str, Any], block)


def test_context_builder_returns_message_copy() -> None:
    messages: list[MessageParam] = [
        {
            "role": "user",
            "content": "Fix the bug",
        }
    ]
    builder = ContextBuilder()

    context = builder.build(messages)

    assert context == messages
    assert context is not messages


def test_context_builder_snips_old_large_tool_result() -> None:
    large_output = "x" * 20
    messages: list[MessageParam] = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_read",
                    "name": "read_file",
                    "input": {"path": "module.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_read",
                    "content": large_output,
                    "is_error": False,
                }
            ],
        },
        {
            "role": "user",
            "content": "Continue",
        },
    ]
    builder = ContextBuilder(max_tool_result_chars=10, recent_message_count=1)

    context = builder.build(messages)
    tool_result = first_content_block(context[1])

    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "toolu_read"
    assert tool_result["is_error"] is False
    assert tool_result["content"] == OMITTED_TOOL_RESULT_TEMPLATE.format(
        char_count=20
    )


def test_context_builder_keeps_recent_large_tool_result() -> None:
    large_output = "x" * 20
    messages: list[MessageParam] = [
        {
            "role": "user",
            "content": "Earlier",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_recent",
                    "content": large_output,
                    "is_error": False,
                }
            ],
        },
    ]
    builder = ContextBuilder(max_tool_result_chars=10, recent_message_count=1)

    context = builder.build(messages)
    tool_result = first_content_block(context[1])

    assert tool_result["content"] == large_output


def test_context_builder_does_not_mutate_original_messages() -> None:
    large_output = "x" * 20
    messages: list[MessageParam] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_read",
                    "content": large_output,
                    "is_error": False,
                }
            ],
        },
        {
            "role": "user",
            "content": "Continue",
        },
    ]
    builder = ContextBuilder(max_tool_result_chars=10, recent_message_count=1)

    context = builder.build(messages)

    assert first_content_block(context[0])["content"] != large_output
    assert first_content_block(messages[0])["content"] == large_output


def test_context_builder_extracts_structured_checkpoint() -> None:
    steps = [
        AgentStep(
            step_number=1,
            stop_reason="tool_use",
            text=["I will inspect, edit, and verify the context builder."],
            tool_calls=[
                ToolCall(
                    name="read_file",
                    input={"path": "agent/context.py"},
                    tool_use_id="toolu_read",
                ),
                ToolCall(
                    name="edit_file",
                    input={"path": "agent/context.py"},
                    tool_use_id="toolu_edit",
                ),
                ToolCall(
                    name="run_command",
                    input={"command": ".venv/bin/python -m pytest"},
                    tool_use_id="toolu_test",
                ),
            ],
            tool_results=[
                ToolResult(tool_use_id="toolu_read", content="file contents"),
                ToolResult(tool_use_id="toolu_edit", content="diff"),
                ToolResult(
                    tool_use_id="toolu_test",
                    content="exit_code: 0\ntimed_out: false\nstdout: passed",
                ),
            ],
        ),
        AgentStep(
            step_number=2,
            stop_reason="tool_use",
            tool_calls=[
                ToolCall(
                    name="edit_file",
                    input={"path": "agent/context.py"},
                    tool_use_id="toolu_bad_edit",
                )
            ],
            tool_results=[
                ToolResult(
                    tool_use_id="toolu_bad_edit",
                    content="Tool 'edit_file' raised ValueError: Exact text was not found",
                    is_error=True,
                )
            ],
        ),
    ]
    builder = ContextBuilder()

    pending_action = PendingAction(
        session_id="session-one",
        step_number=3,
        tool_name="run_command",
        tool_use_id="toolu_pending",
        tool_input={"command": ".venv/bin/python -m pytest"},
        started_at="2026-06-25T00:00:00+00:00",
    )
    checkpoint = builder.build_checkpoint(
        steps,
        objective="Finish Day 11 context compaction",
        pending_action=pending_action,
    )

    assert checkpoint.goal == "Finish Day 11 context compaction"
    assert checkpoint.files_read == ["agent/context.py"]
    assert checkpoint.files_changed == ["agent/context.py"]
    assert len(checkpoint.edits) == 2
    assert checkpoint.edits[0].tool_name == "edit_file"
    assert checkpoint.edits[0].status == "applied"
    assert checkpoint.edits[1].status == "error"
    assert checkpoint.decisions == [
        "I will inspect, edit, and verify the context builder."
    ]
    assert len(checkpoint.commands_run) == 1
    assert checkpoint.commands_run[0].command == ".venv/bin/python -m pytest"
    assert checkpoint.commands_run[0].status == "passed"
    assert checkpoint.commands_run[0].exit_code == 0
    assert len(checkpoint.tool_errors) == 1
    assert checkpoint.tool_errors[0].step_number == 2
    assert checkpoint.tool_errors[0].tool_name == "edit_file"
    assert "Exact text was not found" in checkpoint.tool_errors[0].message
    assert checkpoint.latest_verification.status == "passed"
    assert checkpoint.latest_verification.command == ".venv/bin/python -m pytest"
    assert checkpoint.pending_action == pending_action


def test_context_builder_prepends_structured_checkpoint_message() -> None:
    messages: list[MessageParam] = [
        {
            "role": "user",
            "content": "Continue",
        }
    ]
    steps = [
        AgentStep(
            step_number=1,
            stop_reason="tool_use",
            tool_calls=[
                ToolCall(
                    name="write_file",
                    input={"path": "tests/test_context.py"},
                    tool_use_id="toolu_write",
                )
            ],
            tool_results=[ToolResult(tool_use_id="toolu_write", content="diff")],
        )
    ]
    builder = ContextBuilder()

    pending_action = PendingAction(
        session_id="session-one",
        step_number=2,
        tool_name="run_command",
        tool_use_id="toolu_pending",
        tool_input={"command": ".venv/bin/python -m pytest"},
        started_at="2026-06-25T00:00:00+00:00",
    )

    context = builder.build(
        messages,
        steps,
        objective="Add context checkpoint",
        pending_action=pending_action,
    )

    assert context[0]["role"] == "user"
    assert isinstance(context[0]["content"], str)
    assert "[Structured context checkpoint]" in context[0]["content"]
    assert "Goal:" in context[0]["content"]
    assert "- Add context checkpoint" in context[0]["content"]
    assert "Files changed:" in context[0]["content"]
    assert "- tests/test_context.py" in context[0]["content"]
    assert "Edits:" in context[0]["content"]
    assert "- step 1 write_file tests/test_context.py: applied" in context[0]["content"]
    assert "Pending action:" in context[0]["content"]
    assert "- step 2 run_command (toolu_pending)" in context[0]["content"]
    assert context[1:] == messages


def test_context_builder_does_not_collapse_under_budget() -> None:
    messages: list[MessageParam] = [
        {
            "role": "user",
            "content": "Earlier context",
        },
        {
            "role": "user",
            "content": "Continue",
        },
    ]
    steps = [
        AgentStep(
            step_number=1,
            stop_reason="tool_use",
            tool_calls=[
                ToolCall(
                    name="read_file",
                    input={"path": "agent/context.py"},
                    tool_use_id="toolu_read",
                )
            ],
            tool_results=[ToolResult(tool_use_id="toolu_read", content="content")],
        )
    ]
    builder = ContextBuilder(max_context_chars=10_000)

    context = builder.build(messages, steps)

    assert context[1:] == messages


def test_context_builder_hard_collapses_over_budget() -> None:
    messages: list[MessageParam] = [
        {
            "role": "user",
            "content": f"old-{index}-" + ("x" * 30),
        }
        for index in range(6)
    ]
    steps = [
        AgentStep(
            step_number=1,
            stop_reason="tool_use",
            tool_calls=[
                ToolCall(
                    name="edit_file",
                    input={"path": "agent/context.py"},
                    tool_use_id="toolu_edit",
                )
            ],
            tool_results=[ToolResult(tool_use_id="toolu_edit", content="diff")],
        )
    ]
    builder = ContextBuilder(
        max_context_chars=120,
        collapse_recent_message_count=2,
    )

    context = builder.build(messages, steps)

    assert len(context) == 3
    assert "[Structured context checkpoint]" in context[0]["content"]
    assert context[1:] == messages[-2:]


def test_context_builder_hard_collapse_keeps_recent_complete_turn() -> None:
    tool_use_message: MessageParam = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_read",
                "name": "read_file",
                "input": {"path": "agent/context.py"},
            }
        ],
    }
    tool_result_message: MessageParam = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_read",
                "content": "x" * 30,
                "is_error": False,
            }
        ],
    }
    final_message: MessageParam = {
        "role": "assistant",
        "content": "Read the file.",
    }
    messages: list[MessageParam] = [
        {
            "role": "user",
            "content": "Old turn",
        },
        {
            "role": "assistant",
            "content": "Old answer",
        },
        {
            "role": "user",
            "content": "Recent turn",
        },
        tool_use_message,
        tool_result_message,
        final_message,
    ]
    steps = [
        AgentStep(
            step_number=1,
            stop_reason="tool_use",
            tool_calls=[
                ToolCall(
                    name="read_file",
                    input={"path": "agent/context.py"},
                    tool_use_id="toolu_read",
                )
            ],
            tool_results=[ToolResult(tool_use_id="toolu_read", content="content")],
        )
    ]
    builder = ContextBuilder(
        max_context_chars=120,
        collapse_recent_turn_count=1,
    )

    context = builder.build(messages, steps)

    assert context[1:] == [
        {
            "role": "user",
            "content": "Recent turn",
        },
        tool_use_message,
        tool_result_message,
        final_message,
    ]


def test_context_builder_records_reduction_for_synthetic_long_trajectory() -> None:
    messages: list[MessageParam] = []
    steps: list[AgentStep] = []
    for index in range(8):
        path = f"agent/module_{index}.py"
        tool_use_id = f"toolu_read_{index}"
        messages.extend(
            [
                {
                    "role": "user",
                    "content": f"Inspect {path}",
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tool_use_id,
                            "name": "read_file",
                            "input": {"path": path},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": "x" * 500,
                            "is_error": False,
                        }
                    ],
                },
            ]
        )
        steps.append(
            AgentStep(
                step_number=index + 1,
                stop_reason="tool_use",
                tool_calls=[
                    ToolCall(
                        name="read_file",
                        input={"path": path},
                        tool_use_id=tool_use_id,
                    )
                ],
                tool_results=[
                    ToolResult(
                        tool_use_id=tool_use_id,
                        content="x" * 500,
                    )
                ],
            )
        )

    steps.append(
        AgentStep(
            step_number=9,
            stop_reason="tool_use",
            text=["I will update the context builder and run focused tests."],
            tool_calls=[
                ToolCall(
                    name="edit_file",
                    input={"path": "agent/context.py"},
                    tool_use_id="toolu_edit",
                ),
                ToolCall(
                    name="run_command",
                    input={"command": ".venv/bin/python -m pytest tests/test_context.py"},
                    tool_use_id="toolu_test",
                ),
            ],
            tool_results=[
                ToolResult(tool_use_id="toolu_edit", content="diff"),
                ToolResult(
                    tool_use_id="toolu_test",
                    content="exit_code: 0\ntimed_out: false\nstdout: 9 passed",
                ),
            ],
        )
    )
    pending_action = PendingAction(
        session_id="session-one",
        step_number=10,
        tool_name="run_command",
        tool_use_id="toolu_pending",
        tool_input={"command": ".venv/bin/python -m pytest"},
        started_at="2026-06-25T00:00:00+00:00",
    )
    builder = ContextBuilder(
        max_tool_result_chars=100,
        recent_message_count=3,
        max_context_chars=1_200,
        collapse_recent_turn_count=2,
    )

    result = builder.build_with_metadata(
        messages,
        steps,
        objective="Complete Day 11 context compaction",
        pending_action=pending_action,
    )

    checkpoint_content = result.messages[0]["content"]
    assert isinstance(checkpoint_content, str)
    assert result.original_message_count == len(messages)
    assert result.final_message_count < result.original_message_count
    assert result.original_context_chars > result.final_context_chars
    assert result.snipped_tool_results > 0
    assert result.hard_collapsed is True
    assert result.checkpoint_included is True
    assert "- agent/context.py" in checkpoint_content
    assert "Latest verification:" in checkpoint_content
    assert "- passed: .venv/bin/python -m pytest tests/test_context.py" in checkpoint_content
    assert "Pending action:" in checkpoint_content
    assert "- step 10 run_command (toolu_pending)" in checkpoint_content
