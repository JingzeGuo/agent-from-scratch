from typing import Any, cast

from anthropic.types import MessageParam

from agent.context import OMITTED_TOOL_RESULT_TEMPLATE, ContextBuilder


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
