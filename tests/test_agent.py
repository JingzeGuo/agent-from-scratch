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
from agent.schemas import CalculatorInput, ReadFileInput, SearchWebInput
from agent.tool import Tool
from agent.tool_registry import ToolRegistry
from agent.tools import calculator


class FakeMessages:
    def __init__(self, responses: list[Message]) -> None:
        self.responses = responses
        self.call_count = 0
        self.requests: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> "FakeStreamManager":
        self.requests.append(
            {
                **kwargs,
                "messages": list(kwargs["messages"]),
            }
        )
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
    registry: ToolRegistry | None = None,
) -> tuple[Agent, FakeMessages]:
    fake_client = FakeClient(responses)
    agent = Agent(
        client=cast(AsyncAnthropic, fake_client),
        registry=registry or create_registry(),
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


def test_agent_recovers_from_invalid_tool_arguments() -> None:
    invalid_tool_response = make_message(
        content=[
            ToolUseBlock(
                id="toolu_invalid",
                name="calculator",
                input={"number": "1 + 1"},
                type="tool_use",
            )
        ],
        stop_reason="tool_use",
    )
    corrected_tool_response = make_message(
        content=[
            ToolUseBlock(
                id="toolu_corrected",
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
    agent, messages = create_agent(
        [invalid_tool_response, corrected_tool_response, final_response]
    )

    agent_run = asyncio.run(agent.run("Calculate 1 + 1"))

    first_step, second_step, final_step = agent_run.steps
    first_result = first_step.tool_results[0]
    assert first_result.is_error is True
    assert "field 'expression': Field required" in first_result.content
    assert second_step.tool_calls[0].input == {"expression": "1 + 1"}
    assert second_step.tool_results[0].content == "2"
    assert second_step.tool_results[0].is_error is False
    assert final_step.text == ["The answer is 2."]
    assert agent_run.termination == "completed"

    second_request_messages = messages.requests[1]["messages"]
    error_observation = second_request_messages[-1]["content"][0]
    assert error_observation["tool_use_id"] == "toolu_invalid"
    assert error_observation["is_error"] is True


def test_agent_recovers_from_missing_file_with_different_action() -> None:
    read_attempts = 0

    def missing_file(path: str) -> str:
        nonlocal read_attempts
        read_attempts += 1
        raise FileNotFoundError(f"File not found: {path}")

    def search_web(query: str, max_results: int = 5) -> str:
        return f"Found {max_results} results for: {query}"

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="read_file",
            description="Read a local file.",
            input_schema=ReadFileInput,
            fn=missing_file,
        )
    )
    registry.register(
        Tool(
            name="search_web",
            description="Search the web.",
            input_schema=SearchWebInput,
            fn=search_web,
        )
    )

    missing_file_response = make_message(
        content=[
            ToolUseBlock(
                id="toolu_missing",
                name="read_file",
                input={"path": "missing.txt"},
                type="tool_use",
            )
        ],
        stop_reason="tool_use",
    )
    fallback_response = make_message(
        content=[
            ToolUseBlock(
                id="toolu_search",
                name="search_web",
                input={"query": "requested information", "max_results": 3},
                type="tool_use",
            )
        ],
        stop_reason="tool_use",
    )
    final_response = make_message(
        content=[TextBlock(text="I found an alternative source.", type="text")],
        stop_reason="end_turn",
    )
    agent, messages = create_agent(
        [missing_file_response, fallback_response, final_response],
        registry=registry,
    )

    agent_run = asyncio.run(agent.run("Find the requested information"))

    first_step, second_step, _ = agent_run.steps
    first_result = first_step.tool_results[0]
    assert read_attempts == 1
    assert first_result.is_error is True
    assert "FileNotFoundError" in first_result.content
    assert second_step.tool_calls[0].name == "search_web"
    assert second_step.tool_results[0].is_error is False
    assert agent_run.termination == "completed"

    second_request_messages = messages.requests[1]["messages"]
    error_observation = second_request_messages[-1]["content"][0]
    assert error_observation["tool_use_id"] == "toolu_missing"
    assert error_observation["is_error"] is True
