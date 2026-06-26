import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from anthropic import AsyncAnthropic
from anthropic.types import (
    ContentBlock,
    Message,
    MessageParam,
    StopReason,
    TextBlock,
    ToolUseBlock,
    Usage,
)

from agent.agent import Agent
from agent.context import ContextBuilder
from agent.prompts import build_system_prompt
from agent.schemas import (
    AgentRun,
    AgentStep,
    CalculatorInput,
    ReadFileInput,
    SearchWebInput,
    ToolCall,
    ToolResult,
    VerificationEvidence,
)
from agent.session import SessionStore
from agent.setup import create_registry as create_workspace_registry
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


class FakeContextBuilder(ContextBuilder):
    def __init__(self, context: list[MessageParam]) -> None:
        self.context = context
        self.calls: list[list[MessageParam]] = []

    def build(self, messages: list[MessageParam]) -> list[MessageParam]:
        self.calls.append(list(messages))
        return self.context


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
    assert agent_run.final_stop_reason == "end_turn"
    assert agent_run.verification.status == "not_run"
    assert agent_run.task_success is None
    assert len(agent_run.steps) == 2
    assert len(agent.steps) == 2
    assert agent.steps[0].tool_calls[0].name == "calculator"
    assert agent.steps[0].tool_results[0].content == "2"
    assert agent.steps[0].tool_results[0].is_error is False
    assert agent.steps[1].text == ["The answer is 2."]
    assert capsys.readouterr().out == "Running calculator\nThe answer is 2.\n"


def test_agent_sends_coding_system_prompt() -> None:
    response = make_message(
        content=[TextBlock(text="Done.", type="text")],
        stop_reason="end_turn",
    )
    agent, messages = create_agent([response])

    asyncio.run(agent.run("Say done"))

    system_prompt = messages.requests[0]["system"]
    assert system_prompt == agent.system_prompt
    assert "You are a coding agent operating inside a local workspace." in system_prompt
    assert "`calculator`: Optional helper for math." in system_prompt
    assert "Inspect before editing" in system_prompt
    assert "Edit, then verify" in system_prompt


def test_agent_uses_context_builder_for_model_messages() -> None:
    response = make_message(
        content=[TextBlock(text="Done.", type="text")],
        stop_reason="end_turn",
    )
    agent, messages = create_agent([response])
    built_context: list[MessageParam] = [
        {
            "role": "user",
            "content": "Built context",
        }
    ]
    context_builder = FakeContextBuilder(built_context)
    agent.context_builder = context_builder

    asyncio.run(agent.run("Original task"))

    assert context_builder.calls == [
        [
            {
                "role": "user",
                "content": "Original task",
            }
        ]
    ]
    assert messages.requests[0]["messages"] == built_context


def test_build_system_prompt_uses_workspace_and_registered_tools(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path
    registry = create_registry()

    prompt = build_system_prompt(
        workspace_root=workspace_root,
        registry=registry,
    )

    assert workspace_root.as_posix() in prompt
    assert "`calculator`: Optional helper for math." in prompt
    assert "- `read_file`:" not in prompt


def test_agent_completes_read_search_edit_test_trajectory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = tmp_path / "module.py"
    target.write_text("def answer() -> int:\n    return 1\n", encoding="utf-8")

    registry = create_workspace_registry(tmp_path)
    responses = [
        make_message(
            content=[
                ToolUseBlock(
                    id="toolu_read",
                    name="read_file",
                    input={"path": "module.py"},
                    type="tool_use",
                )
            ],
            stop_reason="tool_use",
        ),
        make_message(
            content=[
                ToolUseBlock(
                    id="toolu_search",
                    name="search_text",
                    input={"pattern": "return 1", "file_pattern": "module.py"},
                    type="tool_use",
                )
            ],
            stop_reason="tool_use",
        ),
        make_message(
            content=[
                ToolUseBlock(
                    id="toolu_edit",
                    name="edit_file",
                    input={
                        "path": "module.py",
                        "old_text": "def answer() -> int:\n    return 1\n",
                        "new_text": "def answer() -> int:\n    return 2\n",
                    },
                    type="tool_use",
                )
            ],
            stop_reason="tool_use",
        ),
        make_message(
            content=[
                ToolUseBlock(
                    id="toolu_test",
                    name="run_command",
                    input={
                        "command": f"{sys.executable} -m py_compile module.py",
                    },
                    type="tool_use",
                )
            ],
            stop_reason="tool_use",
        ),
        make_message(
            content=[TextBlock(text="Verified the focused command.", type="text")],
            stop_reason="end_turn",
        ),
    ]
    agent, _ = create_agent(responses, registry=registry)

    agent_run = asyncio.run(agent.run("Fix module.py"))

    assert [step.tool_calls[0].name for step in agent_run.steps[:-1]] == [
        "read_file",
        "search_text",
        "edit_file",
        "run_command",
    ]
    assert target.read_text(encoding="utf-8") == "def answer() -> int:\n    return 2\n"
    assert "exit_code: 0" in agent_run.steps[3].tool_results[0].content
    assert agent_run.verification.status == "passed"
    assert agent_run.verification.command == f"{sys.executable} -m py_compile module.py"
    assert agent_run.verification.exit_code == 0
    assert agent_run.task_success is None
    assert agent_run.steps[-1].text == ["Verified the focused command."]
    assert capsys.readouterr().out == (
        "Reading module.py\n"
        "Searching workspace text\n"
        "Editing module.py\n"
        "Running command\n"
        "Verified the focused command.\n"
    )


def test_tool_success_does_not_prove_task_success() -> None:
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
    incorrect_final_response = make_message(
        content=[TextBlock(text="The answer is 3.", type="text")],
        stop_reason="end_turn",
    )
    agent, _ = create_agent([tool_response, incorrect_final_response])

    agent_run = asyncio.run(agent.run("Calculate 1 + 1"))

    assert agent_run.termination == "completed"
    assert agent_run.steps[0].tool_results[0].content == "2"
    assert agent_run.steps[-1].text == ["The answer is 3."]
    assert agent_run.verification.status == "not_run"
    assert agent_run.task_success is None


def test_failed_command_followed_by_repair_and_passing_command(
    tmp_path: Path,
) -> None:
    target = tmp_path / "module.py"
    target.write_text("def answer()\n    return 1\n", encoding="utf-8")

    command = f"{sys.executable} -m py_compile module.py"
    registry = create_workspace_registry(tmp_path)
    responses = [
        make_message(
            content=[
                ToolUseBlock(
                    id="toolu_read",
                    name="read_file",
                    input={"path": "module.py"},
                    type="tool_use",
                )
            ],
            stop_reason="tool_use",
        ),
        make_message(
            content=[
                ToolUseBlock(
                    id="toolu_failed_test",
                    name="run_command",
                    input={"command": command},
                    type="tool_use",
                )
            ],
            stop_reason="tool_use",
        ),
        make_message(
            content=[
                ToolUseBlock(
                    id="toolu_edit",
                    name="edit_file",
                    input={
                        "path": "module.py",
                        "old_text": "def answer()\n    return 1\n",
                        "new_text": "def answer() -> int:\n    return 1\n",
                    },
                    type="tool_use",
                )
            ],
            stop_reason="tool_use",
        ),
        make_message(
            content=[
                ToolUseBlock(
                    id="toolu_passing_test",
                    name="run_command",
                    input={"command": command},
                    type="tool_use",
                )
            ],
            stop_reason="tool_use",
        ),
        make_message(
            content=[TextBlock(text="The syntax check now passes.", type="text")],
            stop_reason="end_turn",
        ),
    ]
    agent, _ = create_agent(responses, registry=registry)

    agent_run = asyncio.run(agent.run("Repair module.py"))

    assert agent_run.steps[1].tool_results[0].is_error is False
    assert "exit_code: 1" in agent_run.steps[1].tool_results[0].content
    assert "exit_code: 0" in agent_run.steps[3].tool_results[0].content
    assert agent_run.verification.status == "passed"
    assert agent_run.verification.command == command
    assert agent_run.verification.exit_code == 0
    assert agent_run.task_success is None


def test_agent_recovers_from_failed_edit(
    tmp_path: Path,
) -> None:
    target = tmp_path / "module.py"
    target.write_text("def answer() -> int:\n    return 1\n", encoding="utf-8")

    registry = create_workspace_registry(tmp_path)
    responses = [
        make_message(
            content=[
                ToolUseBlock(
                    id="toolu_read",
                    name="read_file",
                    input={"path": "module.py"},
                    type="tool_use",
                )
            ],
            stop_reason="tool_use",
        ),
        make_message(
            content=[
                ToolUseBlock(
                    id="toolu_bad_edit",
                    name="edit_file",
                    input={
                        "path": "module.py",
                        "old_text": "return 0",
                        "new_text": "return 2",
                    },
                    type="tool_use",
                )
            ],
            stop_reason="tool_use",
        ),
        make_message(
            content=[
                ToolUseBlock(
                    id="toolu_reread",
                    name="read_file",
                    input={"path": "module.py"},
                    type="tool_use",
                )
            ],
            stop_reason="tool_use",
        ),
        make_message(
            content=[
                ToolUseBlock(
                    id="toolu_good_edit",
                    name="edit_file",
                    input={
                        "path": "module.py",
                        "old_text": "def answer() -> int:\n    return 1\n",
                        "new_text": "def answer() -> int:\n    return 2\n",
                    },
                    type="tool_use",
                )
            ],
            stop_reason="tool_use",
        ),
        make_message(
            content=[TextBlock(text="Recovered and applied the edit.", type="text")],
            stop_reason="end_turn",
        ),
    ]
    agent, messages = create_agent(responses, registry=registry)

    agent_run = asyncio.run(agent.run("Fix module.py"))

    failed_result = agent_run.steps[1].tool_results[0]
    assert failed_result.is_error is True
    assert "Exact text was not found" in failed_result.content
    assert agent_run.steps[2].tool_calls[0].name == "read_file"
    assert agent_run.steps[3].tool_results[0].is_error is False
    assert target.read_text(encoding="utf-8") == "def answer() -> int:\n    return 2\n"

    recovery_observation = messages.requests[2]["messages"][-1]["content"][0]
    assert recovery_observation["tool_use_id"] == "toolu_bad_edit"
    assert recovery_observation["is_error"] is True


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
    assert agent_run.final_stop_reason == "tool_use"
    assert agent_run.verification.status == "not_run"
    assert agent_run.task_success is None
    assert len(agent_run.steps) == 2
    assert len(agent.steps) == 2
    assert capsys.readouterr().out == (
        "Running calculator\n"
        "Running calculator\n"
        "Agent reached the 2-step limit. Task stopped.\n"
    )


def test_agent_handles_protocol_error_stop_reason(
    capsys: pytest.CaptureFixture[str],
) -> None:
    response = make_message(
        content=[TextBlock(text="Partial response", type="text")],
        stop_reason="max_tokens",
    )
    agent, messages = create_agent([response])

    agent_run = asyncio.run(agent.run("Write a long response"))

    assert messages.call_count == 1
    assert agent_run.termination == "protocol_error"
    assert agent_run.final_stop_reason == "max_tokens"
    assert agent_run.verification.status == "not_run"
    assert agent_run.task_success is None
    assert len(agent_run.steps) == 1
    assert len(agent.steps) == 1
    assert agent.steps[0].stop_reason == "max_tokens"
    assert capsys.readouterr().out == (
        "Partial response\n"
        "Protocol error stop reason: max_tokens\n"
    )


def test_completed_run_can_contain_failed_verification() -> None:
    agent_run = AgentRun(
        objective="Fix the bug",
        steps=[],
        termination="completed",
        final_stop_reason="end_turn",
        verification=VerificationEvidence(
            status="failed",
            command="pytest",
            exit_code=1,
            output="1 failed",
        ),
        task_success=False,
    )

    assert agent_run.termination == "completed"
    assert agent_run.final_stop_reason == "end_turn"
    assert agent_run.verification.status == "failed"
    assert agent_run.task_success is False


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
    assert agent.completed_runs == [first_run, second_run]


def test_agent_creates_snapshot_from_current_state(tmp_path: Path) -> None:
    target = tmp_path / "module.py"
    target.write_text("def answer() -> int:\n    return 1\n", encoding="utf-8")
    registry = create_workspace_registry(tmp_path)
    agent, _ = create_agent([], registry=registry)

    read_output, read_is_error = registry.execute("read_file", {"path": "module.py"})
    edit_output, edit_is_error = registry.execute(
        "edit_file",
        {
            "path": "module.py",
            "old_text": "def answer() -> int:\n    return 1\n",
            "new_text": "def answer() -> int:\n    return 2\n",
        },
    )
    step = AgentStep(
        step_number=1,
        stop_reason="tool_use",
        tool_calls=[
            ToolCall(
                name="edit_file",
                input={"path": "module.py"},
                tool_use_id="toolu_edit",
            )
        ],
        tool_results=[
            ToolResult(
                tool_use_id="toolu_edit",
                content=edit_output,
            )
        ],
    )
    run = AgentRun(
        objective="Fix module.py",
        steps=[step],
        termination="completed",
        final_stop_reason="end_turn",
        verification=VerificationEvidence(status="not_run"),
    )
    agent.messages = cast(
        list[MessageParam],
        [{"role": "user", "content": "Fix module.py"}],
    )
    agent.steps = [step]
    agent.completed_runs = [run]
    agent.token_tracker.add(Usage(input_tokens=12, output_tokens=8))

    snapshot = agent.create_snapshot("demo-session")

    assert read_is_error is False
    assert "return 1" in read_output
    assert edit_is_error is False
    assert snapshot.session_id == "demo-session"
    assert snapshot.workspace_root == tmp_path.as_posix()
    assert snapshot.provider == "anthropic"
    assert snapshot.model == "claude-haiku-4-5"
    assert snapshot.max_steps == 10
    assert snapshot.messages == [{"role": "user", "content": "Fix module.py"}]
    assert snapshot.steps == [step]
    assert snapshot.completed_runs == [run]
    assert snapshot.read_files == ["module.py"]
    assert snapshot.changed_files == ["module.py"]
    assert snapshot.original_file_contents == {
        "module.py": "def answer() -> int:\n    return 1\n"
    }
    assert snapshot.input_tokens == 12
    assert snapshot.output_tokens == 8
    assert snapshot.estimated_cost > 0


def test_agent_restores_snapshot_into_current_state(tmp_path: Path) -> None:
    target = tmp_path / "module.py"
    target.write_text("def answer() -> int:\n    return 1\n", encoding="utf-8")
    registry = create_workspace_registry(tmp_path)
    agent, _ = create_agent([], registry=registry)
    registry.execute("read_file", {"path": "module.py"})
    registry.execute(
        "edit_file",
        {
            "path": "module.py",
            "old_text": "def answer() -> int:\n    return 1\n",
            "new_text": "def answer() -> int:\n    return 2\n",
        },
    )
    agent.messages = cast(
        list[MessageParam],
        [{"role": "user", "content": "Fix module.py"}],
    )
    agent.token_tracker.add(Usage(input_tokens=12, output_tokens=8))
    snapshot = agent.create_snapshot("demo-session")

    restored_registry = create_workspace_registry(tmp_path)
    restored_agent, _ = create_agent([], registry=restored_registry)
    restored_agent.restore_snapshot(snapshot)

    assert restored_agent.provider == snapshot.provider
    assert restored_agent.model == snapshot.model
    assert restored_agent.max_steps == snapshot.max_steps
    assert restored_agent.messages == snapshot.messages
    assert restored_agent.steps == snapshot.steps
    assert restored_agent.completed_runs == snapshot.completed_runs
    assert restored_agent.registry.read_files == {target.resolve()}
    assert restored_agent.registry.changed_files == {target.resolve()}
    assert restored_agent.registry.original_file_contents == {
        target.resolve(): "def answer() -> int:\n    return 1\n"
    }
    assert restored_agent.token_tracker.input_tokens == 12
    assert restored_agent.token_tracker.output_tokens == 8
    assert restored_agent.token_tracker.estimated_cost == snapshot.estimated_cost


def test_agent_rejects_snapshot_from_different_workspace(tmp_path: Path) -> None:
    first_workspace = tmp_path / "first"
    second_workspace = tmp_path / "second"
    first_workspace.mkdir()
    second_workspace.mkdir()
    first_agent, _ = create_agent(
        [],
        registry=create_workspace_registry(first_workspace),
    )
    second_agent, _ = create_agent(
        [],
        registry=create_workspace_registry(second_workspace),
    )
    snapshot = first_agent.create_snapshot("demo-session")

    with pytest.raises(ValueError, match="does not match"):
        second_agent.restore_snapshot(snapshot)


def test_agent_records_pending_action_and_tool_events(tmp_path: Path) -> None:
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
    agent, _ = create_agent([tool_response, final_response])
    session_store = SessionStore(tmp_path / "sessions")
    agent.configure_session_recording(session_store, "session-one")

    asyncio.run(agent.run("Calculate 1 + 1"))

    pending_action = session_store.read_pending_action("session-one")
    events = session_store.read_events("session-one")
    assert pending_action is not None
    assert pending_action.step_number == 1
    assert pending_action.tool_name == "calculator"
    assert pending_action.tool_use_id == "toolu_test"
    assert pending_action.tool_input == {"expression": "1 + 1"}
    assert [event.event_type for event in events] == [
        "run_started",
        "tool_started",
        "tool_finished",
    ]
    assert events[0].objective == "Calculate 1 + 1"
    assert events[1].tool_name == "calculator"
    assert events[2].is_error is False


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

    def missing_file(path: str, offset: int = 1, limit: int = 200) -> str:
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
