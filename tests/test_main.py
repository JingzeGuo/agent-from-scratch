import asyncio
from pathlib import Path
from typing import cast

import pytest
from anthropic import AsyncAnthropic

from agent.agent import Agent
from agent.schemas import (
    AgentRun,
    CalculatorInput,
    PendingAction,
    SessionSnapshot,
    VerificationEvidence,
)
from agent.session import SessionStore
from agent.tool import Tool
from agent.tool_registry import ToolRegistry
from agent.tools import calculator
from main import (
    CliSessionState,
    checkpoint_session,
    default_sessions_dir,
    generate_session_id,
    handle_command,
    parse_cli_args,
    parse_one_shot_task,
    report_interrupted_action,
    run_cli,
)


class FakeRunAgent:
    def __init__(self) -> None:
        self.tasks: list[str] = []

    async def run(self, user_task: str) -> AgentRun:
        self.tasks.append(user_task)
        print("done")
        return AgentRun(
            objective=user_task,
            steps=[],
            termination="completed",
            final_stop_reason="end_turn",
            verification=VerificationEvidence(status="not_run"),
            task_success=None,
        )


class FakeCheckpointAgent(FakeRunAgent):
    def __init__(self) -> None:
        super().__init__()
        self.snapshots: list[tuple[str, str | None]] = []

    def create_snapshot(
        self,
        session_id: str,
        session_name: str | None = None,
    ) -> SessionSnapshot:
        self.snapshots.append((session_id, session_name))
        return SessionSnapshot(
            session_id=session_id,
            session_name=session_name,
            workspace_root="/workspace/project",
            provider="anthropic",
            model="claude-haiku-4-5",
            max_steps=10,
        )


def create_agent() -> Agent:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="calculator",
            description="Calculate an expression.",
            input_schema=CalculatorInput,
            fn=calculator,
        )
    )
    return Agent(
        client=AsyncAnthropic(api_key="test-key"),
        registry=registry,
    )


def test_help_lists_available_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    should_exit = handle_command("/help")

    assert should_exit is False
    assert capsys.readouterr().out == (
        "Available commands:\n"
        "  /help     Show available commands.\n"
        "  /model    Show or switch provider and model.\n"
        "  /diff     Show file changes from this session.\n"
        "  /rename   Rename the current session.\n"
        "  /sessions List saved sessions.\n"
        "  /exit     Exit the application.\n"
    )


def test_exit_requests_cli_exit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    should_exit = handle_command("/exit")

    assert should_exit is True
    assert capsys.readouterr().out == "Goodbye.\n"


def test_unknown_command_shows_help_hint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    should_exit = handle_command("/unknown")

    assert should_exit is False
    assert capsys.readouterr().out == (
        "Unknown command: /unknown\n"
        "Type /help to see available commands.\n"
    )


def test_model_command_shows_current_model(
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent = create_agent()

    should_exit = handle_command("/model", agent)

    assert should_exit is False
    assert capsys.readouterr().out == (
        "Current model: anthropic/claude-haiku-4-5\n"
    )


def test_parse_one_shot_task() -> None:
    assert parse_one_shot_task([]) is None
    assert parse_one_shot_task(["Fix", "the", "bug"]) == "Fix the bug"
    assert parse_one_shot_task(["  "]) == ""


def test_parse_cli_args_supports_resume_and_one_shot_task() -> None:
    args = parse_cli_args(["--resume", "day10", "Fix", "the", "bug"])

    assert args.resume_session_id == "day10"
    assert args.one_shot_task == "Fix the bug"


def test_parse_cli_args_supports_equals_resume_form() -> None:
    args = parse_cli_args(["--resume=day10"])

    assert args.resume_session_id == "day10"
    assert args.one_shot_task is None


def test_parse_cli_args_rejects_invalid_resume_usage() -> None:
    with pytest.raises(ValueError, match="Usage"):
        parse_cli_args(["--resume"])

    with pytest.raises(ValueError, match="only once"):
        parse_cli_args(["--resume", "one", "--resume", "two"])


def test_default_sessions_dir_is_workspace_local(tmp_path: Path) -> None:
    assert default_sessions_dir(tmp_path) == tmp_path / ".agents" / "sessions"


def test_generate_session_id_uses_safe_timestamp_format() -> None:
    session_id = generate_session_id()

    assert session_id.startswith("session-")
    assert "/" not in session_id
    assert " " not in session_id


def test_run_cli_executes_one_shot_task(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_agent = FakeRunAgent()

    def fail_input(prompt: str) -> str:
        raise AssertionError(f"Unexpected prompt: {prompt}")

    monkeypatch.setattr("builtins.input", fail_input)

    asyncio.run(run_cli(cast(Agent, fake_agent), "Fix the bug"))

    assert fake_agent.tasks == ["Fix the bug"]
    assert capsys.readouterr().out == "\nAssistant: done\n"


def test_run_cli_checkpoints_one_shot_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_agent = FakeCheckpointAgent()
    session_store = SessionStore(tmp_path / "sessions")
    session_state = CliSessionState(session_id="session-one", session_name="day10")

    def fail_input(prompt: str) -> str:
        raise AssertionError(f"Unexpected prompt: {prompt}")

    monkeypatch.setattr("builtins.input", fail_input)

    asyncio.run(
        run_cli(
            cast(Agent, fake_agent),
            "Fix the bug",
            session_store,
            session_state,
        )
    )

    loaded = session_store.load("session-one")
    events = session_store.read_events("session-one")
    assert fake_agent.tasks == ["Fix the bug"]
    assert fake_agent.snapshots == [("session-one", "day10")]
    assert loaded.session_id == "session-one"
    assert loaded.session_name == "day10"
    assert session_store.read_pending_action("session-one") is None
    assert [event.event_type for event in events] == ["checkpoint_saved"]
    assert capsys.readouterr().out == (
        "\nAssistant: done\n"
        "Checkpoint saved: session-one\n"
    )


def test_checkpoint_session_does_nothing_without_store_or_state() -> None:
    fake_agent = FakeCheckpointAgent()

    checkpoint_session(cast(Agent, fake_agent), None, None)

    assert fake_agent.snapshots == []


def test_checkpoint_session_clears_existing_pending_action(tmp_path: Path) -> None:
    fake_agent = FakeCheckpointAgent()
    session_store = SessionStore(tmp_path / "sessions")
    session_state = CliSessionState(session_id="session-one")
    session_store.write_pending_action(
        PendingAction(
            session_id="session-one",
            step_number=1,
            tool_name="calculator",
            tool_use_id="toolu_calc",
            tool_input={"expression": "1 + 1"},
            started_at="2026-06-25T00:00:00+00:00",
        )
    )

    checkpoint_session(cast(Agent, fake_agent), session_store, session_state)

    assert session_store.read_pending_action("session-one") is None
    assert [event.event_type for event in session_store.read_events("session-one")] == [
        "checkpoint_saved"
    ]


def test_report_interrupted_action_warns_and_clears_marker(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session_store = SessionStore(tmp_path / "sessions")
    session_store.write_pending_action(
        PendingAction(
            session_id="session-one",
            step_number=2,
            tool_name="edit_file",
            tool_use_id="toolu_edit",
            tool_input={"path": "agent.py"},
            started_at="2026-06-25T00:00:00+00:00",
        )
    )

    report_interrupted_action(session_store, "session-one")

    events = session_store.read_events("session-one")
    assert session_store.read_pending_action("session-one") is None
    assert len(events) == 1
    assert events[0].event_type == "interrupted_action_detected"
    assert events[0].tool_name == "edit_file"
    assert capsys.readouterr().out == (
        "Interrupted action detected: edit_file (toolu_edit)\n"
    )


def test_model_command_switches_provider(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    agent = create_agent()

    should_exit = handle_command("/model deepseek", agent)

    assert should_exit is False
    assert agent.provider == "deepseek"
    assert agent.model == "deepseek-v4-flash"
    assert capsys.readouterr().out == (
        "Switched model: deepseek/deepseek-v4-flash\n"
    )


def test_diff_command_shows_session_diff(
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent = create_agent()

    should_exit = handle_command("/diff", agent)

    assert should_exit is False
    assert capsys.readouterr().out == "[No files changed]\n"


def test_diff_command_requires_agent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    should_exit = handle_command("/diff")

    assert should_exit is False
    assert capsys.readouterr().out == "Diff command is unavailable.\n"


def test_rename_command_updates_current_session_name(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent = create_agent()
    session_store = SessionStore(tmp_path / "sessions")
    session_state = CliSessionState(session_id="session-one")

    should_exit = handle_command("/rename day10", agent, session_store, session_state)

    assert should_exit is False
    assert session_state.session_name == "day10"
    assert session_store.load("session-one").session_name == "day10"
    events = session_store.read_events("session-one")
    assert list((tmp_path / "sessions").glob("*.json")) == [
        tmp_path / "sessions" / "session-one.json"
    ]
    assert [event.event_type for event in events] == ["session_renamed"]
    assert events[0].session_name == "day10"
    assert capsys.readouterr().out == "Renamed session: day10\n"


def test_rename_command_requires_session_name(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent = create_agent()
    session_store = SessionStore(tmp_path / "sessions")
    session_state = CliSessionState(session_id="session-one")

    should_exit = handle_command("/rename", agent, session_store, session_state)

    assert should_exit is False
    assert capsys.readouterr().out == "Usage: /rename <session-name>\n"


def test_rename_command_requires_session_context(
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent = create_agent()

    should_exit = handle_command("/rename day10", agent)

    assert should_exit is False
    assert capsys.readouterr().out == "Rename command is unavailable.\n"


def test_sessions_command_lists_saved_sessions(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent = create_agent()
    session_store = SessionStore(tmp_path / "sessions")
    session_store.save(agent.create_snapshot("z-session", "later"))
    session_store.save(agent.create_snapshot("a-session"))

    should_exit = handle_command("/sessions", session_store=session_store)

    assert should_exit is False
    assert capsys.readouterr().out == (
        "Saved sessions:\n"
        "  a-session  [unnamed]\n"
        "  z-session  later\n"
    )


def test_sessions_command_handles_empty_store(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session_store = SessionStore(tmp_path / "sessions")

    should_exit = handle_command("/sessions", session_store=session_store)

    assert should_exit is False
    assert capsys.readouterr().out == "[No saved sessions]\n"


def test_sessions_command_requires_store(
    capsys: pytest.CaptureFixture[str],
) -> None:
    should_exit = handle_command("/sessions")

    assert should_exit is False
    assert capsys.readouterr().out == "Sessions command is unavailable.\n"
