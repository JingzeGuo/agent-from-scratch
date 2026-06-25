from pathlib import Path

import pytest

from agent.schemas import (
    AgentRun,
    AgentStep,
    SessionSnapshot,
    ToolCall,
    ToolResult,
    VerificationEvidence,
)
from agent.session import SessionStore


def make_snapshot(
    session_id: str = "day10-demo",
    session_name: str | None = None,
) -> SessionSnapshot:
    step = AgentStep(
        step_number=1,
        stop_reason="tool_use",
        tool_calls=[
            ToolCall(
                name="read_file",
                input={"path": "agent/agent.py"},
                tool_use_id="toolu_read",
            )
        ],
        tool_results=[
            ToolResult(
                tool_use_id="toolu_read",
                content="1: class Agent:",
            )
        ],
    )
    run = AgentRun(
        objective="Inspect the agent loop",
        steps=[step],
        termination="completed",
        final_stop_reason="end_turn",
        verification=VerificationEvidence(status="not_run"),
    )
    return SessionSnapshot(
        session_id=session_id,
        session_name=session_name,
        workspace_root="/workspace/project",
        provider="anthropic",
        model="claude-haiku-4-5",
        max_steps=10,
        messages=[
            {
                "role": "user",
                "content": "Inspect the agent loop",
            }
        ],
        steps=[step],
        completed_runs=[run],
        read_files=["agent/agent.py"],
        changed_files=["agent/agent.py"],
        original_file_contents={"agent/agent.py": "class Agent:\n    pass\n"},
        input_tokens=12,
        output_tokens=8,
        estimated_cost=0.000052,
    )


def test_session_store_round_trips_snapshot(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    snapshot = make_snapshot(session_name="day10")

    path = store.save(snapshot)
    loaded = store.load(snapshot.session_id)

    assert path == tmp_path / "sessions" / "day10-demo.json"
    assert loaded == snapshot


def test_session_store_finds_snapshot_by_id_or_name(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    snapshot = make_snapshot("session-one", "day10")
    store.save(snapshot)

    assert store.find("session-one") == snapshot
    assert store.find("day10") == snapshot


def test_session_store_reports_ambiguous_session_name(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    store.save(make_snapshot("session-one", "day10"))
    store.save(make_snapshot("session-two", "day10"))

    with pytest.raises(ValueError, match="ambiguous"):
        store.find("day10")


def test_session_store_lists_snapshots_sorted_by_id(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    second = make_snapshot("z-session", "later")
    first = make_snapshot("a-session")
    store.save(second)
    store.save(first)

    assert store.list_snapshots() == [first, second]


def test_session_store_creates_session_directory(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "missing" / "sessions"
    store = SessionStore(sessions_dir)

    store.save(make_snapshot())

    assert sessions_dir.is_dir()


@pytest.mark.parametrize(
    "session_id",
    [
        "",
        "../secret",
        "nested/session",
        ".hidden",
        "has space",
    ],
)
def test_session_store_rejects_unsafe_session_ids(
    tmp_path: Path,
    session_id: str,
) -> None:
    store = SessionStore(tmp_path / "sessions")

    with pytest.raises(ValueError, match="Session id must"):
        store.save(make_snapshot(session_id))

    with pytest.raises(ValueError, match="Session id must"):
        store.load(session_id)


def test_session_store_reports_missing_session(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")

    with pytest.raises(FileNotFoundError, match="Session not found"):
        store.load("missing-session")

    with pytest.raises(FileNotFoundError, match="Session not found"):
        store.find("missing-session")


def test_session_store_rejects_unsafe_session_names(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")

    with pytest.raises(ValueError, match="Session name must"):
        store.save(make_snapshot(session_name="has space"))

    with pytest.raises(ValueError, match="Session name must"):
        store.find("has space")
