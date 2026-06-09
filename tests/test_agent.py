import asyncio
from collections.abc import Sequence
from typing import Any, cast

import pytest
from anthropic import AsyncAnthropic
from anthropic.types import (
    ContentBlock,
    Message,
    StopReason,
    TextBlock,
    ToolUseBlock,
    Usage,
)

from agent.agent import Agent
from agent.schemas import CalculatorInput
from agent.tool import Tool
from agent.tool_registry import ToolRegistry
from agent.tools import calculator


class FakeMessages:
    def __init__(self, responses: list[Message]) -> None:
        self.responses = responses
        self.call_count = 0

    def stream(self, **kwargs: Any) -> "FakeStreamManager":
        response = self.responses[self.call_count]
        self.call_count += 1
        return FakeStreamManager(response)


class FakeStreamManager:
    def __init__(self, response: Message) -> None:
        self.response = response
        self.text_stream = self._stream_text()

    async def __aenter__(self) -> "FakeStreamManager":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        return None

    async def _stream_text(self) -> Any:
        for block in self.response.content:
            if block.type == "text":
                midpoint = len(block.text) // 2
                for chunk in (block.text[:midpoint], block.text[midpoint:]):
                    if chunk:
                        yield chunk

    async def get_final_message(self) -> Message:
        return self.response


class FakeClient:
    def __init__(self, responses: list[Message]) -> None:
        self.messages = FakeMessages(responses)


def make_message(
    content: Sequence[ContentBlock],
    stop_reason: StopReason,
) -> Message:
    return Message(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-haiku-4-5",
        content=list(content),
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def create_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="calculator",
            description="Calculate an expression.",
            input_schema=CalculatorInput,
            fn=calculator,
        )
    )
    return registry


def create_agent(
    responses: list[Message],
    max_steps: int = 10,
) -> tuple[Agent, FakeMessages]:
    fake_client = FakeClient(responses)
    agent = Agent(
        client=cast(AsyncAnthropic, fake_client),
        registry=create_registry(),
        max_steps=max_steps,
    )
    return agent, fake_client.messages


def test_single_tool_call_completes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool_response = make_message(
        content=[
            ToolUseBlock(
                id="toolu_test",
                name="calculator",
                input={"expression": "1 + 1"},
                type="tool_use",
            )
        ],
        stop_reason="tool_use",
    )
    final_response = make_message(
        content=[TextBlock(text="The answer is 2.", type="text")],
        stop_reason="end_turn",
    )
    agent, messages = create_agent([tool_response, final_response])

    agent_run = asyncio.run(agent.run("Calculate 1 + 1"))

    assert messages.call_count == 2
    assert agent_run.objective == "Calculate 1 + 1"
    assert agent_run.termination == "completed"
    assert len(agent_run.steps) == 2
    assert len(agent.steps) == 2
    assert agent.steps[0].tool_calls[0].name == "calculator"
    assert agent.steps[0].tool_results[0].content == "2"
    assert agent.steps[0].tool_results[0].is_error is False
    assert agent.steps[1].text == ["The answer is 2."]
    assert capsys.readouterr().out == "The answer is 2.\n"


def test_agent_stops_at_max_steps(
    capsys: pytest.CaptureFixture[str],
) -> None:
    responses = [
        make_message(
            content=[
                ToolUseBlock(
                    id=f"toolu_{step}",
                    name="calculator",
                    input={"expression": "1 + 1"},
                    type="tool_use",
                )
            ],
            stop_reason="tool_use",
        )
        for step in range(2)
    ]
    agent, messages = create_agent(responses, max_steps=2)

    agent_run = asyncio.run(agent.run("Keep calculating"))

    assert messages.call_count == 2
    assert agent_run.termination == "max_steps"
    assert len(agent_run.steps) == 2
    assert len(agent.steps) == 2
    assert capsys.readouterr().out == (
        "Agent reached the 2-step limit. Task stopped.\n"
    )


def test_agent_handles_unexpected_stop_reason(
    capsys: pytest.CaptureFixture[str],
) -> None:
    response = make_message(
        content=[TextBlock(text="Partial response", type="text")],
        stop_reason="max_tokens",
    )
    agent, messages = create_agent([response])

    agent_run = asyncio.run(agent.run("Write a long response"))

    assert messages.call_count == 1
    assert agent_run.termination == "unexpected_stop"
    assert len(agent_run.steps) == 1
    assert len(agent.steps) == 1
    assert agent.steps[0].stop_reason == "max_tokens"
    assert capsys.readouterr().out == (
        "Partial response\n"
        "Unexpected stop reason: max_tokens\n"
    )


def test_agent_run_contains_only_current_task_steps() -> None:
    responses = [
        make_message(
            content=[TextBlock(text="First answer.", type="text")],
            stop_reason="end_turn",
        ),
        make_message(
            content=[TextBlock(text="Second answer.", type="text")],
            stop_reason="end_turn",
        ),
    ]
    agent, _ = create_agent(responses)

    first_run = asyncio.run(agent.run("First task"))
    second_run = asyncio.run(agent.run("Second task"))

    assert first_run.objective == "First task"
    assert len(first_run.steps) == 1
    assert second_run.objective == "Second task"
    assert len(second_run.steps) == 1
    assert len(agent.steps) == 2
