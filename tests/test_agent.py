import asyncio
import shlex
import sys
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from threading import Event
from typing import Any, cast

import pytest
from pydantic import BaseModel

from agent.agent import Agent
from agent.context import ContextBuilder
from agent.prompts import build_system_prompt
from agent.schemas import (
    AgentRun,
    AgentStep,
    PendingAction,
    ProviderCapabilities,
    ProviderResponse,
    ReadFileInput,
    RunCommandInput,
    SearchWebInput,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from agent.security import ToolApprovalPolicy
from agent.session import SessionStore
from agent.setup import create_registry as create_workspace_registry
from agent.tool import Tool
from agent.tool_registry import ToolRegistry
from agent.tools import run_command

MessageParam = dict[str, Any]


class TextBlock(BaseModel):
    text: str
    type: str = "text"


class ToolUseBlock(BaseModel):
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


ContentBlock = TextBlock | ToolUseBlock
StopReason = str


class SampleToolInput(BaseModel):
    value: str


def sample_tool(value: str) -> str:
    return f"processed: {value}"


class FakeContextBuilder(ContextBuilder):
    def __init__(self, context: list[MessageParam]) -> None:
        self.context = context
        self.calls: list[list[MessageParam]] = []
        self.step_calls: list[list[AgentStep]] = []
        self.objective_calls: list[str | None] = []
        self.pending_action_calls: list[PendingAction | None] = []

    def build(
        self,
        messages: list[MessageParam],
        steps: list[AgentStep] | None = None,
        objective: str | None = None,
        pending_action: PendingAction | None = None,
    ) -> list[MessageParam]:
        self.calls.append(list(messages))
        self.step_calls.append(list(steps or []))
        self.objective_calls.append(objective)
        self.pending_action_calls.append(pending_action)
        return self.context


class FakeProviderAdapter:
    def __init__(
        self,
        responses: list[ProviderResponse] | None = None,
        capabilities: ProviderCapabilities | None = None,
    ) -> None:
        self.provider = "deepseek"
        self.model = "deepseek-v4-flash"
        self.capabilities = capabilities or ProviderCapabilities()
        self.responses = responses or []
        self.requests: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        return len(self.requests)

    async def stream_response(
        self,
        *,
        system: str,
        tools: list[ToolDefinition],
        messages: list[dict[str, Any]],
        on_text_delta: Callable[[str], None] | None = None,
    ) -> ProviderResponse:
        self.requests.append(
            {
                "system": system,
                "tools": list(tools),
                "messages": list(messages),
            }
        )
        response = self.responses.pop(0)
        for text in response.text:
            if on_text_delta is not None:
                on_text_delta(text)
        return response

    def tool_result_message(self, tool_results: list[ToolResult]) -> dict[str, Any]:
        return {
            "role": "user",
            "content": [
                {
                    "type": result.type,
                    "tool_use_id": result.tool_use_id,
                    "content": result.content,
                    "is_error": result.is_error,
                }
                for result in tool_results
            ],
        }


def make_message(
    content: Sequence[ContentBlock],
    stop_reason: StopReason,
) -> ProviderResponse:
    text = [block.text for block in content if isinstance(block, TextBlock)]
    tool_calls = [
        ToolCall(name=block.name, input=block.input, tool_use_id=block.id)
        for block in content
        if isinstance(block, ToolUseBlock)
    ]
    return ProviderResponse(
        message={
            "role": "assistant",
            "content": [block.model_dump() for block in content],
        },
        stop_reason=stop_reason,
        text=text,
        tool_calls=tool_calls,
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


def create_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="read_file",
            description="Return a sample value.",
            input_schema=SampleToolInput,
            fn=sample_tool,
        )
    )
    return registry


def create_command_registry(workspace_root: Path) -> ToolRegistry:
    registry = ToolRegistry(workspace_root)
    registry.register(
        Tool(
            name="run_command",
            description="Run a bounded command.",
            input_schema=RunCommandInput,
            fn=partial(run_command, workspace_root=workspace_root),
        )
    )
    return registry


def create_agent(
    responses: list[ProviderResponse],
    max_steps: int = 10,
    registry: ToolRegistry | None = None,
) -> tuple[Agent, FakeProviderAdapter]:
    provider = FakeProviderAdapter(responses)
    agent = Agent(
        provider_adapter=provider,
        registry=registry or create_registry(),
        max_steps=max_steps,
    )
    return agent, provider


def test_single_tool_call_completes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool_response = make_message(
        content=[
            ToolUseBlock(
                id="toolu_test",
                name="read_file",
                input={"value": "sample"},
                type="tool_use",
            )
        ],
        stop_reason="tool_use",
    )
    final_response = make_message(
        content=[TextBlock(text="Done.", type="text")],
        stop_reason="end_turn",
    )
    agent, messages = create_agent([tool_response, final_response])

    agent_run = asyncio.run(agent.run("Process a sample value"))

    assert messages.call_count == 2
    assert agent_run.objective == "Process a sample value"
    assert agent_run.termination == "completed"
    assert agent_run.final_stop_reason == "end_turn"
    assert len(agent_run.steps) == 2
    assert len(agent.steps) == 2
    assert agent.steps[0].tool_calls[0].name == "read_file"
    assert agent.steps[0].tool_results[0].content == "processed: sample"
    assert agent.steps[0].tool_results[0].is_error is False
    assert agent.steps[1].text == ["Done."]
    assert capsys.readouterr().out == "Running read_file\nDone.\n"


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
    assert "`read_file`: Read a bounded range of lines" in system_prompt
    assert "Inspect before editing" in system_prompt
    assert "Edit, then verify" in system_prompt
    assert "must run focused verification" in system_prompt
    assert "Do not run the same failing command again" in system_prompt
    assert "Do not create temporary verification scripts" in system_prompt
    assert "When a focused verification command passes" in system_prompt


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
    assert context_builder.step_calls == [[]]
    assert context_builder.objective_calls == ["Original task"]
    assert context_builder.pending_action_calls == [None]
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
    assert "`read_file`: Read a bounded range of lines" in prompt
    assert "must run focused verification" in prompt
    assert "`run_command` does not support shell operators" in prompt


def test_build_system_prompt_prefers_observed_venv_python(tmp_path: Path) -> None:
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")

    prompt = build_system_prompt(
        workspace_root=tmp_path,
        registry=create_registry(),
    )

    assert "This workspace has `.venv/bin/python`" in prompt
    assert "`.venv/bin/python -m pytest tests/test_target.py`" in prompt
    assert "Do not spend steps probing interpreter locations" in prompt
    assert "do not spend steps using glob searches" in prompt


def test_build_system_prompt_uses_python3_fallback_without_venv(
    tmp_path: Path,
) -> None:
    prompt = build_system_prompt(
        workspace_root=tmp_path,
        registry=create_registry(),
    )

    assert "`python3 -m pytest tests/test_target.py`" in prompt


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
    assert agent_run.steps[-1].text == ["Verified the focused command."]
    assert capsys.readouterr().out == (
        "Reading module.py\n"
        "Searching workspace text\n"
        "Editing module.py\n"
        "Running command\n"
        "Verified the focused command.\n"
    )


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


def test_agent_runs_approved_command_requiring_approval(tmp_path: Path) -> None:
    command = f"{shlex.quote(sys.executable)} -c \"print('approved')\""
    registry = create_command_registry(tmp_path)
    responses = [
        make_message(
            content=[
                ToolUseBlock(
                    id="toolu_command",
                    name="run_command",
                    input={"command": command},
                    type="tool_use",
                )
            ],
            stop_reason="tool_use",
        ),
        make_message(
            content=[TextBlock(text="Command ran.", type="text")],
            stop_reason="end_turn",
        ),
    ]
    agent, _ = create_agent(responses, registry=registry)
    agent.configure_approval_callback(lambda tool_call, policy: True)
    session_store = SessionStore(tmp_path / "sessions")
    agent.configure_session_recording(session_store, "session-one")

    agent_run = asyncio.run(agent.run("Run an approved command"))

    tool_result = agent_run.steps[0].tool_results[0]
    events = session_store.read_events("session-one")
    assert tool_result.is_error is False
    assert "stdout:\napproved" in tool_result.content
    assert [
        event.event_type
        for event in events
        if event.event_type.startswith("tool_approval")
    ] == ["tool_approval_requested", "tool_approval_granted"]
    assert "tool_started" in [event.event_type for event in events]


def test_agent_reuses_approval_for_same_command(tmp_path: Path) -> None:
    command = f"{shlex.quote(sys.executable)} -c \"print('approved')\""
    registry = create_command_registry(tmp_path)
    responses = [
        make_message(
            content=[
                ToolUseBlock(
                    id="toolu_command_one",
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
                    id="toolu_command_two",
                    name="run_command",
                    input={"command": command},
                    type="tool_use",
                )
            ],
            stop_reason="tool_use",
        ),
        make_message(
            content=[TextBlock(text="Commands ran.", type="text")],
            stop_reason="end_turn",
        ),
    ]
    agent, _ = create_agent(responses, registry=registry)
    approval_requests = 0

    def approve(tool_call: ToolCall, policy: ToolApprovalPolicy) -> bool:
        nonlocal approval_requests
        approval_requests += 1
        return True

    agent.configure_approval_callback(approve)

    agent_run = asyncio.run(agent.run("Run the same approved command twice"))

    assert approval_requests == 1
    assert all(
        not result.is_error for step in agent_run.steps for result in step.tool_results
    )


def test_agent_requests_approval_again_for_different_cwd(tmp_path: Path) -> None:
    command = f"{shlex.quote(sys.executable)} -c \"print('approved')\""
    (tmp_path / "nested").mkdir()
    registry = create_command_registry(tmp_path)
    responses = [
        make_message(
            content=[
                ToolUseBlock(
                    id="toolu_command_root",
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
                    id="toolu_command_nested",
                    name="run_command",
                    input={"command": command, "cwd": "nested"},
                    type="tool_use",
                )
            ],
            stop_reason="tool_use",
        ),
        make_message(
            content=[TextBlock(text="Commands ran.", type="text")],
            stop_reason="end_turn",
        ),
    ]
    agent, _ = create_agent(responses, registry=registry)
    approval_requests = 0

    def approve(tool_call: ToolCall, policy: ToolApprovalPolicy) -> bool:
        nonlocal approval_requests
        approval_requests += 1
        return True

    agent.configure_approval_callback(approve)

    asyncio.run(agent.run("Run approved commands in two directories"))

    assert approval_requests == 2


def test_agent_denies_command_requiring_approval_by_default(tmp_path: Path) -> None:
    command = f"{shlex.quote(sys.executable)} -c \"print('denied')\""
    registry = create_command_registry(tmp_path)
    responses = [
        make_message(
            content=[
                ToolUseBlock(
                    id="toolu_command",
                    name="run_command",
                    input={"command": command},
                    type="tool_use",
                )
            ],
            stop_reason="tool_use",
        ),
        make_message(
            content=[TextBlock(text="Command was not approved.", type="text")],
            stop_reason="end_turn",
        ),
    ]
    agent, _ = create_agent(responses, registry=registry)
    session_store = SessionStore(tmp_path / "sessions")
    agent.configure_session_recording(session_store, "session-one")

    agent_run = asyncio.run(agent.run("Run an unapproved command"))

    tool_result = agent_run.steps[0].tool_results[0]
    events = session_store.read_events("session-one")
    assert tool_result.is_error is True
    assert "approval denied" in tool_result.content
    assert [
        event.event_type
        for event in events
        if event.event_type.startswith("tool_approval")
    ] == ["tool_approval_requested", "tool_approval_denied"]
    assert "tool_started" not in [event.event_type for event in events]


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
                    name="read_file",
                    input={"value": "sample"},
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
    assert len(agent_run.steps) == 2
    assert len(agent.steps) == 2
    assert capsys.readouterr().out == (
        "Running read_file\n"
        "Running read_file\n"
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
    assert len(agent_run.steps) == 1
    assert len(agent.steps) == 1
    assert agent.steps[0].stop_reason == "max_tokens"
    assert capsys.readouterr().out == (
        "Partial response\nProtocol error stop reason: max_tokens\n"
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
    assert agent.completed_runs == [first_run, second_run]


def test_agent_rejects_provider_without_tool_support() -> None:
    adapter = FakeProviderAdapter(
        capabilities=ProviderCapabilities(supports_tools=False)
    )

    with pytest.raises(ValueError, match="does not support tools"):
        Agent(
            provider_adapter=adapter,
            registry=create_registry(),
        )


def test_agent_rejects_provider_without_streaming_support() -> None:
    adapter = FakeProviderAdapter(
        capabilities=ProviderCapabilities(supports_streaming=False)
    )

    with pytest.raises(ValueError, match="does not support streaming"):
        Agent(
            provider_adapter=adapter,
            registry=ToolRegistry(),
        )


def test_agent_executes_multiple_tool_calls_serially() -> None:
    events: list[str] = []

    def first_probe(value: str) -> str:
        events.append("first:start")
        events.append("first:end")
        return f"first result: {value}"

    def second_probe(value: str) -> str:
        events.append("second:start")
        events.append("second:end")
        return f"second result: {value}"

    first_call = ToolCall(
        name="first_probe",
        input={"value": "sample"},
        tool_use_id="call_first",
    )
    second_call = ToolCall(
        name="second_probe",
        input={"value": "second sample"},
        tool_use_id="call_second",
    )
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="first_probe",
            description="Record the first probe execution.",
            input_schema=SampleToolInput,
            fn=first_probe,
        )
    )
    registry.register(
        Tool(
            name="second_probe",
            description="Record the second probe execution.",
            input_schema=SampleToolInput,
            fn=second_probe,
        )
    )
    adapter = FakeProviderAdapter(
        responses=[
            ProviderResponse(
                message={
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": first_call.tool_use_id,
                            "name": first_call.name,
                            "input": first_call.input,
                        },
                        {
                            "type": "tool_use",
                            "id": second_call.tool_use_id,
                            "name": second_call.name,
                            "input": second_call.input,
                        },
                    ],
                },
                stop_reason="tool_use",
                tool_calls=[first_call, second_call],
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            ),
            ProviderResponse(
                message={
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Done."}],
                },
                stop_reason="end_turn",
                text=["Done."],
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            ),
        ]
    )
    agent = Agent(provider_adapter=adapter, registry=registry)

    agent_run = asyncio.run(agent.run("Run two probes"))

    assert events == [
        "first:start",
        "first:end",
        "second:start",
        "second:end",
    ]
    assert agent_run.termination == "completed"
    assert [result.tool_use_id for result in agent_run.steps[0].tool_results] == [
        "call_first",
        "call_second",
    ]
    assert [result.content for result in agent_run.steps[0].tool_results] == [
        "first result: sample",
        "second result: second sample",
    ]


def test_agent_executes_async_tool_calls_serially() -> None:
    async def async_probe(value: str) -> str:
        await asyncio.sleep(0)
        return f"async result: {value}"

    tool_call = ToolCall(
        name="async_probe",
        input={"value": "sample"},
        tool_use_id="call_async",
    )
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="async_probe",
            description="Run an async probe.",
            input_schema=SampleToolInput,
            fn=async_probe,
        )
    )
    adapter = FakeProviderAdapter(
        responses=[
            ProviderResponse(
                message={
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tool_call.tool_use_id,
                            "name": tool_call.name,
                            "input": tool_call.input,
                        }
                    ],
                },
                stop_reason="tool_use",
                tool_calls=[tool_call],
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            ),
            ProviderResponse(
                message={
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Done."}],
                },
                stop_reason="end_turn",
                text=["Done."],
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            ),
        ]
    )
    agent = Agent(provider_adapter=adapter, registry=registry)

    agent_run = asyncio.run(agent.run("Run async probe"))

    assert agent_run.termination == "completed"
    assert agent_run.steps[0].tool_results[0].content == "async result: sample"
    assert agent_run.steps[0].tool_results[0].is_error is False


def test_agent_executes_read_only_tool_calls_concurrently_preserving_order(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    first_started = Event()
    second_started = Event()

    def probe_read(value: str) -> str:
        if value == "first":
            events.append("first:start")
            first_started.set()
            if not second_started.wait(timeout=1):
                raise RuntimeError("second call did not start concurrently")
            events.append("first:end")
            return "first result"

        events.append("second:start")
        second_started.set()
        if not first_started.wait(timeout=1):
            raise RuntimeError("first call did not start concurrently")
        events.append("second:end")
        return "second result"

    first_call = ToolCall(
        name="read_file",
        input={"value": "first"},
        tool_use_id="call_first",
    )
    second_call = ToolCall(
        name="read_file",
        input={"value": "second"},
        tool_use_id="call_second",
    )
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="read_file",
            description="Record overlapping read-only execution.",
            input_schema=SampleToolInput,
            fn=probe_read,
        )
    )
    adapter = FakeProviderAdapter(
        responses=[
            ProviderResponse(
                message={
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": first_call.tool_use_id,
                            "name": first_call.name,
                            "input": first_call.input,
                        },
                        {
                            "type": "tool_use",
                            "id": second_call.tool_use_id,
                            "name": second_call.name,
                            "input": second_call.input,
                        },
                    ],
                },
                stop_reason="tool_use",
                tool_calls=[first_call, second_call],
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            ),
            ProviderResponse(
                message={
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Done."}],
                },
                stop_reason="end_turn",
                text=["Done."],
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            ),
        ]
    )
    agent = Agent(provider_adapter=adapter, registry=registry)
    session_store = SessionStore(tmp_path / "sessions")
    agent.configure_session_recording(session_store, "session-one")

    agent_run = asyncio.run(agent.run("Run two read-only probes"))

    assert events.index("second:start") < events.index("first:end")
    assert events.index("first:start") < events.index("second:end")
    assert [result.tool_use_id for result in agent_run.steps[0].tool_results] == [
        "call_first",
        "call_second",
    ]
    assert [result.content for result in agent_run.steps[0].tool_results] == [
        "first result",
        "second result",
    ]
    schedule_events = [
        event
        for event in session_store.read_events("session-one")
        if event.event_type == "tool_schedule_decided"
    ]
    assert len(schedule_events) == 1
    assert schedule_events[0].message == (
        "parallel: all tool calls are read-only; tools: read_file, read_file"
    )


def test_agent_does_not_bypass_approval_for_parallel_tool_calls() -> None:
    executed: list[str] = []

    def approval_required_read(value: str) -> str:
        executed.append(value)
        return value

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="read_file",
            description="Run a read that requires approval.",
            input_schema=SampleToolInput,
            fn=approval_required_read,
            approval_policy=ToolApprovalPolicy(
                decision="requires_approval",
                reason="Test approval policy.",
            ),
        )
    )
    responses = [
        make_message(
            content=[
                ToolUseBlock(
                    id="call_first",
                    name="read_file",
                    input={"value": "first"},
                ),
                ToolUseBlock(
                    id="call_second",
                    name="read_file",
                    input={"value": "second"},
                ),
            ],
            stop_reason="tool_use",
        ),
        make_message(
            content=[TextBlock(text="Done.")],
            stop_reason="end_turn",
        ),
    ]
    agent, _ = create_agent(responses, registry=registry)
    approval_requests: list[str] = []

    def deny(tool_call: ToolCall, policy: ToolApprovalPolicy) -> bool:
        approval_requests.append(tool_call.tool_use_id)
        return False

    agent.configure_approval_callback(deny)

    agent_run = asyncio.run(agent.run("Run two protected reads"))

    assert approval_requests == ["call_first", "call_second"]
    assert executed == []
    assert all(result.is_error for result in agent_run.steps[0].tool_results)


def test_agent_preserves_result_order_when_parallel_tool_call_fails() -> None:
    def probe_read(value: str) -> str:
        if value == "bad":
            raise ValueError("bad value")
        return f"{value} result"

    first_call = ToolCall(
        name="read_file",
        input={"value": "good"},
        tool_use_id="call_good",
    )
    second_call = ToolCall(
        name="read_file",
        input={"value": "bad"},
        tool_use_id="call_bad",
    )
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="read_file",
            description="Return one result and one error.",
            input_schema=SampleToolInput,
            fn=probe_read,
        )
    )
    adapter = FakeProviderAdapter(
        responses=[
            ProviderResponse(
                message={
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": first_call.tool_use_id,
                            "name": first_call.name,
                            "input": first_call.input,
                        },
                        {
                            "type": "tool_use",
                            "id": second_call.tool_use_id,
                            "name": second_call.name,
                            "input": second_call.input,
                        },
                    ],
                },
                stop_reason="tool_use",
                tool_calls=[first_call, second_call],
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            ),
            ProviderResponse(
                message={
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Done."}],
                },
                stop_reason="end_turn",
                text=["Done."],
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            ),
        ]
    )
    agent = Agent(provider_adapter=adapter, registry=registry)

    agent_run = asyncio.run(agent.run("Run two read-only probes"))

    results = agent_run.steps[0].tool_results
    assert [result.tool_use_id for result in results] == ["call_good", "call_bad"]
    assert results[0].content == "good result"
    assert results[0].is_error is False
    assert "ValueError: bad value" in results[1].content
    assert results[1].is_error is True


def test_sub_agent_runs_with_isolated_read_only_context(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sub_agent_call = ToolCall(
        name="sub_agent",
        input={
            "task": "Find session resume code.",
            "profile": "read_only_explorer",
            "max_steps": 2,
        },
        tool_use_id="call_sub_agent",
    )
    adapter = FakeProviderAdapter(
        responses=[
            ProviderResponse(
                message={
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": sub_agent_call.tool_use_id,
                            "name": sub_agent_call.name,
                            "input": sub_agent_call.input,
                        }
                    ],
                },
                stop_reason="tool_use",
                tool_calls=[sub_agent_call],
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            ),
            ProviderResponse(
                message={
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Relevant files: agent/session.py.",
                        }
                    ],
                },
                stop_reason="end_turn",
                text=["Relevant files: agent/session.py."],
                usage=TokenUsage(input_tokens=7, output_tokens=3),
            ),
            ProviderResponse(
                message={
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Done."}],
                },
                stop_reason="end_turn",
                text=["Done."],
                usage=TokenUsage(input_tokens=12, output_tokens=4),
            ),
        ]
    )
    registry = create_workspace_registry(tmp_path)
    agent = Agent(provider_adapter=adapter, registry=registry)
    session_store = SessionStore(tmp_path / "sessions")
    agent.configure_session_recording(session_store, "session-one")
    agent.messages.append({"role": "user", "content": "Parent-only history."})

    agent_run = asyncio.run(agent.run("Delegate exploration"))

    assert agent_run.termination == "completed"
    sub_agent_result = agent_run.steps[0].tool_results[0]
    assert sub_agent_result.is_error is False
    assert "Sub-agent result:" in sub_agent_result.content
    assert "profile: read_only_explorer" in sub_agent_result.content
    assert "termination: completed" in sub_agent_result.content
    assert "Relevant files: agent/session.py." in sub_agent_result.content
    child_request = adapter.requests[1]
    child_tool_names = {tool.name for tool in child_request["tools"]}
    assert child_tool_names == {
        "read_file",
        "glob_files",
        "search_text",
    }
    assert "Do not modify files, execute commands" in child_request["system"]
    assert child_request["messages"] == [
        {"role": "user", "content": "Find session resume code."}
    ]
    assert "Parent-only history" not in str(child_request["messages"])
    assert agent.token_tracker.input_tokens == 29
    assert agent.token_tracker.output_tokens == 12
    sub_agent_events = [
        event
        for event in session_store.read_events("session-one")
        if event.event_type in {"sub_agent_started", "sub_agent_finished"}
    ]
    assert [event.event_type for event in sub_agent_events] == [
        "sub_agent_started",
        "sub_agent_finished",
    ]
    assert sub_agent_events[0].run_id == agent_run.run_id
    assert sub_agent_events[0].tool_use_id == "call_sub_agent"
    assert sub_agent_events[0].objective == "Find session resume code."
    assert sub_agent_events[0].step_count == 2
    assert sub_agent_events[0].message == "profile: read_only_explorer"
    assert sub_agent_events[1].run_id == agent_run.run_id
    assert sub_agent_events[1].tool_use_id == "call_sub_agent"
    assert sub_agent_events[1].child_run_id is not None
    assert sub_agent_events[1].termination == "completed"
    assert sub_agent_events[1].step_count == 1
    assert sub_agent_events[1].input_tokens == 7
    assert sub_agent_events[1].output_tokens == 3
    assert sub_agent_events[1].text_preview == "Relevant files: agent/session.py."
    assert capsys.readouterr().out == "Running sub_agent\nDone.\n"


def test_sub_agent_enforces_child_step_budget(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("content\n", encoding="utf-8")
    sub_agent_call = ToolCall(
        name="sub_agent",
        input={"task": "Read notes.", "max_steps": 1},
        tool_use_id="call_sub_agent",
    )
    child_read_call = ToolCall(
        name="read_file",
        input={"path": "notes.txt"},
        tool_use_id="call_child_read",
    )
    adapter = FakeProviderAdapter(
        responses=[
            ProviderResponse(
                message={
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": sub_agent_call.tool_use_id,
                            "name": sub_agent_call.name,
                            "input": sub_agent_call.input,
                        }
                    ],
                },
                stop_reason="tool_use",
                tool_calls=[sub_agent_call],
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            ),
            ProviderResponse(
                message={
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": child_read_call.tool_use_id,
                            "name": child_read_call.name,
                            "input": child_read_call.input,
                        }
                    ],
                },
                stop_reason="tool_use",
                tool_calls=[child_read_call],
                usage=TokenUsage(input_tokens=7, output_tokens=3),
            ),
            ProviderResponse(
                message={
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Parent done."}],
                },
                stop_reason="end_turn",
                text=["Parent done."],
                usage=TokenUsage(input_tokens=11, output_tokens=4),
            ),
        ]
    )
    agent = Agent(
        provider_adapter=adapter,
        registry=create_workspace_registry(tmp_path),
    )

    agent_run = asyncio.run(agent.run("Delegate exploration"))

    sub_agent_result = agent_run.steps[0].tool_results[0]
    assert sub_agent_result.is_error is False
    assert "termination: max_steps" in sub_agent_result.content
    assert "steps: 1" in sub_agent_result.content
    assert "final_answer:\n[No final text returned by child agent.]" in (
        sub_agent_result.content
    )


@pytest.mark.anyio
async def test_agent_creates_snapshot_from_current_state(tmp_path: Path) -> None:
    target = tmp_path / "module.py"
    target.write_text("def answer() -> int:\n    return 1\n", encoding="utf-8")
    registry = create_workspace_registry(tmp_path)
    agent, _ = create_agent([], registry=registry)

    read_output, read_is_error = await registry.execute(
        "read_file", {"path": "module.py"}
    )
    edit_output, edit_is_error = await registry.execute(
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
    )
    agent.messages = cast(
        list[dict[str, Any]],
        [{"role": "user", "content": "Fix module.py"}],
    )
    agent.steps = [step]
    agent.completed_runs = [run]
    agent.token_tracker.add(TokenUsage(input_tokens=12, output_tokens=8))

    snapshot = agent.create_snapshot("demo-session")

    assert read_is_error is False
    assert "return 1" in read_output
    assert edit_is_error is False
    assert snapshot.session_id == "demo-session"
    assert snapshot.workspace_root == tmp_path.as_posix()
    assert snapshot.provider == "deepseek"
    assert snapshot.model == "deepseek-v4-flash"
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


@pytest.mark.anyio
async def test_agent_restores_snapshot_into_current_state(tmp_path: Path) -> None:
    target = tmp_path / "module.py"
    target.write_text("def answer() -> int:\n    return 1\n", encoding="utf-8")
    registry = create_workspace_registry(tmp_path)
    agent, _ = create_agent([], registry=registry)
    await registry.execute("read_file", {"path": "module.py"})
    await registry.execute(
        "edit_file",
        {
            "path": "module.py",
            "old_text": "def answer() -> int:\n    return 1\n",
            "new_text": "def answer() -> int:\n    return 2\n",
        },
    )
    agent.messages = cast(
        list[dict[str, Any]],
        [{"role": "user", "content": "Fix module.py"}],
    )
    agent.token_tracker.add(TokenUsage(input_tokens=12, output_tokens=8))
    snapshot = agent.create_snapshot("demo-session").model_copy(
        update={"provider": "legacy", "model": "legacy-model"}
    )

    restored_registry = create_workspace_registry(tmp_path)
    restored_agent, _ = create_agent([], registry=restored_registry)
    restored_agent.restore_snapshot(snapshot)

    assert restored_agent.provider == "deepseek"
    assert restored_agent.model == "deepseek-v4-flash"
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
                name="read_file",
                input={"value": "sample"},
                type="tool_use",
            )
        ],
        stop_reason="tool_use",
    )
    final_response = make_message(
        content=[TextBlock(text="Done.", type="text")],
        stop_reason="end_turn",
    )
    agent, _ = create_agent([tool_response, final_response])
    session_store = SessionStore(tmp_path / "sessions")
    agent.configure_session_recording(session_store, "session-one")

    agent_run = asyncio.run(agent.run("Process a sample value"))

    pending_action = session_store.read_pending_action("session-one")
    events = session_store.read_events("session-one")
    run_id = agent_run.run_id
    assert run_id is not None
    assert pending_action is not None
    assert pending_action.step_number == 1
    assert pending_action.tool_name == "read_file"
    assert pending_action.tool_use_id == "toolu_test"
    assert [event.event_type for event in events] == [
        "run_started",
        "model_request_started",
        "model_response_finished",
        "tool_started",
        "tool_finished",
        "step_finished",
        "model_request_started",
        "model_response_finished",
        "step_finished",
        "run_finished",
    ]
    assert {event.run_id for event in events} == {run_id}
    assert events[0].objective == "Process a sample value"
    assert events[0].provider == "deepseek"
    assert events[0].model == "deepseek-v4-flash"
    assert events[2].step_number == 1
    assert events[2].stop_reason == "tool_use"
    assert events[2].input_tokens == 10
    assert events[2].output_tokens == 5
    assert events[2].tool_call_count == 1
    assert events[2].latency_ms is not None
    assert events[3].tool_name == "read_file"
    assert events[4].is_error is False
    assert events[4].output_preview == "processed: sample"
    assert events[4].output_chars == 17
    assert events[5].stop_reason == "tool_use"
    assert events[-1].termination == "completed"
    assert events[-1].final_stop_reason == "end_turn"
    assert events[-1].step_count == 2
    assert events[-1].input_tokens == 20
    assert events[-1].output_tokens == 10
    assert events[-1].estimated_cost is not None


def test_agent_redacts_secret_like_tool_output_in_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_TRACE_REDACT_PATTERNS", r"CUSTOM-\d+")

    def secret_tool(value: str) -> str:
        return f"{value} api_key=sk-secret123456 token=plain-secret CUSTOM-12345"

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="secret_tool",
            description="Return a secret-like value.",
            input_schema=SampleToolInput,
            fn=secret_tool,
        )
    )
    tool_response = make_message(
        content=[
            ToolUseBlock(
                id="toolu_secret",
                name="secret_tool",
                input={"value": "result"},
                type="tool_use",
            )
        ],
        stop_reason="tool_use",
    )
    final_response = make_message(
        content=[TextBlock(text="Done.", type="text")],
        stop_reason="end_turn",
    )
    agent, _ = create_agent([tool_response, final_response], registry=registry)
    session_store = SessionStore(tmp_path / "sessions")
    agent.configure_session_recording(session_store, "session-one")

    asyncio.run(agent.run("Run secret tool"))

    tool_finished = [
        event
        for event in session_store.read_events("session-one")
        if event.event_type == "tool_finished"
    ][0]
    assert tool_finished.output_chars == len(
        "result api_key=sk-secret123456 token=plain-secret CUSTOM-12345"
    )
    assert tool_finished.output_preview == (
        "result api_key=[REDACTED] token=[REDACTED] [REDACTED]"
    )
    assert "sk-secret123456" not in tool_finished.output_preview
    assert "plain-secret" not in tool_finished.output_preview
    assert "CUSTOM-12345" not in tool_finished.output_preview


def test_agent_records_tool_failure_evidence_in_trace(tmp_path: Path) -> None:
    invalid_tool_response = make_message(
        content=[
            ToolUseBlock(
                id="toolu_invalid",
                name="read_file",
                input={"number": "sample"},
                type="tool_use",
            )
        ],
        stop_reason="tool_use",
    )
    final_response = make_message(
        content=[TextBlock(text="Done.", type="text")],
        stop_reason="end_turn",
    )
    agent, _ = create_agent([invalid_tool_response, final_response])
    session_store = SessionStore(tmp_path / "sessions")
    agent.configure_session_recording(session_store, "session-one")

    asyncio.run(agent.run("Process a sample value"))

    tool_finished = [
        event
        for event in session_store.read_events("session-one")
        if event.event_type == "tool_finished"
    ][0]
    assert tool_finished.is_error is True
    assert tool_finished.output_chars is not None
    assert tool_finished.output_chars > 0
    assert tool_finished.output_preview is not None
    assert "Validation error for tool 'read_file'" in tool_finished.output_preview


def test_agent_recovers_from_invalid_tool_arguments() -> None:
    invalid_tool_response = make_message(
        content=[
            ToolUseBlock(
                id="toolu_invalid",
                name="read_file",
                input={"number": "sample"},
                type="tool_use",
            )
        ],
        stop_reason="tool_use",
    )
    corrected_tool_response = make_message(
        content=[
            ToolUseBlock(
                id="toolu_corrected",
                name="read_file",
                input={"value": "sample"},
                type="tool_use",
            )
        ],
        stop_reason="tool_use",
    )
    final_response = make_message(
        content=[TextBlock(text="Done.", type="text")],
        stop_reason="end_turn",
    )
    agent, messages = create_agent(
        [invalid_tool_response, corrected_tool_response, final_response]
    )

    agent_run = asyncio.run(agent.run("Process a sample value"))

    first_step, second_step, final_step = agent_run.steps
    first_result = first_step.tool_results[0]
    assert first_result.is_error is True
    assert "field 'value': Field required" in first_result.content
    assert second_step.tool_calls[0].input == {"value": "sample"}
    assert second_step.tool_results[0].content == "processed: sample"
    assert second_step.tool_results[0].is_error is False
    assert final_step.text == ["Done."]
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
