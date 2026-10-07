import asyncio
import json
from pathlib import Path
from threading import Event
from typing import Any

import pytest

from agent.agent import Agent
from agent.provider import DeepSeekConfig
from agent.schemas import MemoryRecord, TaskMemorySnapshot
from agent.state.memory import MemoryStore, TaskMemoryContext
from agent.state.memory_formation import LIFECYCLE_PROMPT, LLMMemoryFormation
from agent.state.session import SessionStore
from agent.tooling.setup import create_registry
from main import main as cli_main
from main import run_cli
from tests.test_agent import (
    FakeContextBuilder,
    FakeProviderAdapter,
    TextBlock,
    ToolUseBlock,
    make_message,
)
from tests.test_memory import candidate, count, memory, provider, snapshot


def create_memory_agent(
    tmp_path: Path, responses: int = 2
) -> tuple[Agent, FakeProviderAdapter, MemoryStore]:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    llm = FakeProviderAdapter(
        [make_message([TextBlock(text="Done")], "end_turn") for _ in range(responses)]
    )
    agent = Agent(
        llm, create_registry(tmp_path), memory_store=store, stream_output=False
    )
    return agent, llm, store


async def no_formation(snapshot: TaskMemorySnapshot) -> None:
    pass


def test_each_task_loads_core_and_relevant_memory_without_touching_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, llm, store = create_memory_agent(tmp_path, responses=3)
    core = memory(
        "Prefer small targeted patches.",
        scope="user",
        access="core",
        provenance="explicit_user",
    )
    retrieved = memory("Parser token access needs an empty-list guard.")
    store.add(core)
    store.add(retrieved)
    assert agent.memory_formation is not None
    monkeypatch.setattr(agent.memory_formation, "form", no_formation)
    searches: list[str] = []
    search = store.search

    def record_search(query: str, **kwargs: Any) -> list[MemoryRecord]:
        searches.append(query)
        return search(query, **kwargs)

    monkeypatch.setattr(store, "search", record_search)
    # Two model steps in Task 1 still perform just one task-start retrieval.
    llm.responses[0] = make_message(
        [ToolUseBlock(id="read", name="glob_files", input={"pattern": "*.py"})],
        "tool_use",
    )

    async def exercise() -> None:
        await agent.run("Repair the parser")
        new_core = memory("Generated artifacts must not be edited.", access="core")
        store.add(new_core)
        await agent.run("Describe deployment")
        await agent.wait_for_memory_jobs()
        assert new_core.content in llm.requests[-1]["system"]

    try:
        asyncio.run(exercise())
        assert searches == ["Repair the parser", "Describe deployment"]
        for request in llm.requests:
            assert core.content in request["system"]
            assert (
                "Current user instructions override stored preferences"
                in request["system"]
            )
        assert retrieved.content in llm.requests[0]["system"]
        assert retrieved.content not in llm.requests[-1]["system"]
        raw = json.dumps(agent.messages)
        assert core.content not in raw and retrieved.content not in raw
        assert "Persistent Memory" not in raw
        assert agent.context_builder.state.consolidated_state is None
        assert agent.task_starts == [0, 4]
        assert len(agent.messages) == 6
        session = agent.create_snapshot("session")
        assert "Persistent Memory" not in session.model_dump_json()
    finally:
        store.close()


def test_memory_tokens_count_in_working_context_overhead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, _, store = create_memory_agent(tmp_path, responses=1)
    assert agent.memory_formation is not None
    monkeypatch.setattr(agent.memory_formation, "form", no_formation)
    builder = FakeContextBuilder([{"role": "user", "content": "Request"}])
    agent.context_builder = builder
    store.add(memory("Prefer targeted patches.", access="core"))
    baseline = agent._context_overhead_tokens()

    async def exercise() -> None:
        await agent.run("Repair parser")
        await agent.wait_for_memory_jobs()

    try:
        asyncio.run(exercise())
        memory_overhead = agent._context_overhead_tokens()
        assert memory_overhead > baseline
        runtime = [
            {
                "role": "user",
                "content": agent._step_budget_message(
                    step=1, remaining_steps=agent.max_steps
                ),
            }
        ]
        assert builder.overhead_calls == [
            memory_overhead + builder.measure_tokens(runtime)
        ]
        assert agent.task_memory_context.render() in agent._request_system_prompt()
    finally:
        store.close()


def test_next_task_runs_while_formation_pending_and_snapshot_is_detached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, llm, store = create_memory_agent(tmp_path)
    injected = memory(access="core")
    store.add(injected)
    captured: list[TaskMemorySnapshot] = []

    async def exercise() -> None:
        gate = asyncio.Event()
        started = asyncio.Event()

        async def blocked_formation(task_snapshot: TaskMemorySnapshot) -> None:
            started.set()
            await gate.wait()
            # Deliberately read snapshot only after live state has changed.
            captured.append(task_snapshot)

        assert agent.memory_formation is not None
        monkeypatch.setattr(agent.memory_formation, "form", blocked_formation)
        first = await asyncio.wait_for(agent.run("First task"), timeout=1)
        await asyncio.wait_for(started.wait(), timeout=1)
        assert not gate.is_set()
        assert len(agent._memory_jobs) == 1
        first.steps[0].text.append("Later mutation")
        first.objective = "Changed after return"
        store.update(injected.model_copy(update={"content": "Updated stored content"}))

        second = await asyncio.wait_for(agent.run("Second task"), timeout=1)
        assert second.termination == "completed" and not gate.is_set()
        assert len(agent._memory_jobs) == 2
        assert agent.task_starts == [0, 2]
        assert llm.requests[1]["messages"][0]["content"] == "First task"
        assert "Updated stored content" in llm.requests[1]["system"]
        agent.messages.clear()
        agent.completed_runs.clear()
        agent.task_memory_context = TaskMemoryContext()
        gate.set()
        await agent.wait_for_memory_jobs()
        assert not agent._memory_jobs
        by_id = {item.task_id: item for item in captured}
        assert first.run_id is not None and second.run_id is not None
        assert by_id[first.run_id].objective == "First task"
        assert by_id[second.run_id].objective == "Second task"
        assert "Later mutation" not in by_id[first.run_id].steps_json
        assert (
            json.loads(by_id[first.run_id].injected_memories_json)[0]["content"]
            == injected.content
        )

    try:
        asyncio.run(exercise())
    finally:
        store.close()


def test_formation_receives_only_current_task_steps_and_changed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "module.py"
    target.write_text("old\n", encoding="utf-8")
    agent, llm, store = create_memory_agent(tmp_path)
    # The file was already changed earlier this session; set differences would miss it.
    agent.registry.changed_files.add(target)
    agent.registry.changed_files.add(tmp_path / "previous.py")
    agent.registry.read_files.add(target)
    llm.responses = [
        make_message(
            [
                ToolUseBlock(
                    id="edit",
                    name="edit_file",
                    input={"path": "module.py", "old_text": "old", "new_text": "new"},
                )
            ],
            "tool_use",
        ),
        make_message([TextBlock(text="Verified edit")], "end_turn"),
    ]
    snapshots: list[TaskMemorySnapshot] = []

    async def capture(task_snapshot: TaskMemorySnapshot) -> None:
        snapshots.append(task_snapshot)

    assert agent.memory_formation is not None
    monkeypatch.setattr(agent.memory_formation, "form", capture)

    async def exercise() -> None:
        run = await agent.run("Edit module")
        # Deep mutation of the returned tool input/result cannot change formation.
        run.steps[0].tool_calls[0].input["path"] = "unrelated.py"
        run.steps[0].tool_results[0].content = "Changed later"
        await agent.wait_for_memory_jobs()

    try:
        asyncio.run(exercise())
        assert snapshots[0].files_changed == ("module.py",)
        assert "unrelated.py" not in snapshots[0].steps_json
        assert "Changed later" not in snapshots[0].steps_json
        assert any(
            "step:1 edit_file module.py" in ref for ref in snapshots[0].evidence_refs
        )
    finally:
        store.close()


def test_background_failure_is_logged_and_drain_does_not_break_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    agent, _, store = create_memory_agent(tmp_path)

    async def fail(task_snapshot: TaskMemorySnapshot) -> None:
        raise ValueError("Malformed memory JSON")

    assert agent.memory_formation is not None
    monkeypatch.setattr(agent.memory_formation, "form", fail)

    async def exercise() -> None:
        first = await agent.run("First")
        await agent.wait_for_memory_jobs()
        assert first.termination == "completed"
        assert await agent.run("Second")
        await agent.wait_for_memory_jobs()

    try:
        asyncio.run(exercise())
        assert "Memory formation failed for memory:run-" in caplog.text
        assert "Malformed memory JSON" in caplog.text
        assert not agent._memory_jobs
    finally:
        store.close()


def test_new_session_loads_memory_independently_of_session_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, _, store = create_memory_agent(tmp_path, responses=1)
    record = memory("Always verify generated schema boundaries.", access="core")
    store.add(record)
    sessions = SessionStore(tmp_path / "sessions")
    sessions.save(agent.create_snapshot("session-one"))
    store.close()
    new_agent, llm, reopened = create_memory_agent(tmp_path, responses=1)
    new_agent.restore_snapshot(sessions.load("session-one"))
    assert new_agent.task_memory_context.memories == ()
    assert new_agent.memory_formation is not None
    monkeypatch.setattr(new_agent.memory_formation, "form", no_formation)

    async def exercise() -> None:
        await new_agent.run("New task")
        await new_agent.wait_for_memory_jobs()

    try:
        asyncio.run(exercise())
        assert record.content in llm.requests[0]["system"]
    finally:
        reopened.close()


def test_background_lifecycle_jobs_see_prior_committed_add(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    llm = provider(
        {"candidates": [candidate()]}, {"action": "ADD"}, {"candidates": [candidate()]}
    )
    formation = LLMMemoryFormation(llm, store)

    async def exercise() -> None:
        started = asyncio.Event()
        gate = asyncio.Event()
        original = llm.stream_response

        async def delayed_response(**kwargs: Any) -> Any:
            response = await original(**kwargs)
            if kwargs["system"] == LIFECYCLE_PROMPT:
                started.set()
                await gate.wait()
            return response

        monkeypatch.setattr(llm, "stream_response", delayed_response)
        first = asyncio.create_task(formation.form(snapshot()))
        await asyncio.wait_for(started.wait(), timeout=1)
        second = asyncio.create_task(formation.form(snapshot()))
        await asyncio.sleep(0)
        assert llm.call_count == 3  # Two extractions; only the first lifecycle call.
        assert not first.done() and not second.done()
        gate.set()
        await asyncio.gather(first, second)

    try:
        asyncio.run(exercise())
        assert count(store) == 1
    finally:
        store.close()


def test_cli_enables_persistence_and_drains_formation_at_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preference = "The user prefers narrow, targeted patches."
    extracted = candidate(
        content=preference,
        route="CORE",
        scope="user",
        provenance="explicit_user",
        files=[],
        evidence_refs=["task:run-current:objective"],
        new_information="The user explicitly requested this persistent preference.",
    )
    llm = provider("Done", {"candidates": [extracted]}, {"action": "ADD"})
    inputs = iter(["Always prefer narrow targeted patches", "/exit"])
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AGENT_STATE_DIR", raising=False)
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    monkeypatch.setattr(
        "main.load_deepseek_config",
        lambda **kwargs: DeepSeekConfig(
            model=llm.model, api_key="test-key", base_url="https://example.invalid"
        ),
    )
    monkeypatch.setattr("main.DeepSeekProvider", lambda **kwargs: llm)
    monkeypatch.setattr(Agent, "_new_run_id", lambda self: "run-current")

    asyncio.run(cli_main([]))

    assert llm.call_count == 3
    store = MemoryStore(tmp_path / ".agents" / "memory.sqlite3")
    try:
        assert store.get_core()[0].content == preference
        assert store.get_core()[0].source_task_id == "run-current"
    finally:
        store.close()


def test_cli_background_job_progresses_during_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, _, store = create_memory_agent(tmp_path, responses=1)
    finished = Event()
    reads = 0

    def read_input(prompt: str) -> str:
        nonlocal reads
        reads += 1
        if reads == 1:
            return "Trivial task"
        assert finished.wait(timeout=2), "Input blocked background formation"
        return "/exit"

    async def formation(task_snapshot: TaskMemorySnapshot) -> None:
        await asyncio.sleep(0)
        finished.set()

    assert agent.memory_formation is not None
    monkeypatch.setattr(agent.memory_formation, "form", formation)
    monkeypatch.setattr("builtins.input", read_input)

    async def exercise() -> None:
        await run_cli(agent)
        await agent.wait_for_memory_jobs()

    try:
        asyncio.run(exercise())
        assert finished.is_set()
    finally:
        store.close()
