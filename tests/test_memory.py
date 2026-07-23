import asyncio
import math
from pathlib import Path
from typing import Any

import pytest

from agent.memory import (
    LocalBM25Retriever,
    MemoryCandidate,
    MemoryContext,
    MemoryRecord,
    MemorySearchResult,
    MemoryStore,
    MemorySummaryResult,
    MemorySystem,
    OkapiBM25,
    _tokenize,
)
from agent.schemas import AgentRun, TokenUsage, VerificationEvidence


def make_record(
    record_id: str,
    title: str,
    content: str,
    *,
    scope: str = "project",
    kind: str = "topic",
    tags: list[str] | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        id=record_id,
        scope=scope,  # type: ignore[arg-type]
        kind=kind,  # type: ignore[arg-type]
        title=title,
        content=content,
        tags=[] if tags is None else tags,
        created_at="2026-06-28T00:00:00+00:00",
        updated_at="2026-06-28T00:00:00+00:00",
    )


def test_memory_store_saves_redacted_record_in_json_index(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory", "project")
    record = make_record(
        "project-session-one",
        "Session with secret",
        "Use api_key=sk-secret123456 only in local env.",
        kind="session",
    )

    saved = store.save_record(record)

    index = store.index_path.read_text(encoding="utf-8")
    assert "sk-secret123456" not in index
    assert "api_key=[REDACTED]" in index
    assert {path.name for path in store.root.iterdir()} == {"index.json"}
    assert store.get_record("project-session-one") == saved


def test_memory_store_saves_profile_kinds_in_json_index(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory", "global")
    record = make_record(
        "global-preference-one",
        "Chinese explanations",
        "The user prefers Chinese explanations for learning discussions.",
        scope="global",
        kind="preference",
    )

    saved = store.save_record(record)

    records = store.list_records()
    assert records == [saved]
    assert {path.name for path in store.root.iterdir()} == {"index.json"}


def test_memory_store_rejects_record_from_another_scope(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory", "project")
    record = make_record(
        "global-preference-one",
        "Chinese explanations",
        "The user prefers Chinese explanations.",
        scope="global",
        kind="preference",
    )

    with pytest.raises(ValueError, match="Cannot save global memory"):
        store.save_record(record)


def test_memory_store_handles_corrupted_index(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory", "project")
    store.initialize()
    store.index_path.write_text("{not json", encoding="utf-8")

    assert store.list_records() == []

    with pytest.raises(ValueError, match="was not overwritten"):
        store.save_record(
            make_record(
                "project-session-two",
                "Another session",
                "This record must not overwrite a corrupted index.",
                kind="session",
            )
        )
    assert store.index_path.read_text(encoding="utf-8") == "{not json"


def test_okapi_bm25_matches_lucene_formula() -> None:
    retriever = OkapiBM25(
        [
            ["cat", "cat"],
            ["cat", "dog"],
        ]
    )

    scores = retriever.scores(["cat"])

    inverse_document_frequency = math.log(1 + (2 - 2 + 0.5) / (2 + 0.5))
    assert scores == pytest.approx(
        [
            inverse_document_frequency * ((2 * 2.2) / (2 + 1.2)),
            inverse_document_frequency,
        ]
    )


def test_memory_tokenizer_handles_chinese_and_code_identifiers() -> None:
    tokens = _tokenize("中文讲解 ContextBuilder context_builder")

    assert {"中", "文", "讲", "解", "中文", "文讲", "讲解"} <= set(tokens)
    assert {"context", "builder", "contextbuilder", "context_builder"} <= set(
        tokens
    )


def test_local_bm25_retriever_excludes_unrelated_memory() -> None:
    relevant = make_record(
        "context-memory",
        "Context compaction boundary",
        "Never collapse across tool call and tool result boundaries.",
        tags=["context", "tools"],
    )
    unrelated = make_record(
        "docs-memory",
        "README update",
        "Document installation commands and package metadata.",
    )
    retriever = LocalBM25Retriever()

    context = retriever.search(
        "context compaction tool boundary",
        [unrelated, relevant],
    )

    assert [result.record.id for result in context.results] == ["context-memory"]
    assert context.results[0].lexical_score > 0


class FakeSummarizer:
    async def summarize_run(
        self,
        **kwargs: Any,
    ) -> MemorySummaryResult:
        return MemorySummaryResult(
            candidates=[
                MemoryCandidate(
                    scope="global",
                    kind="cross_project_lesson",
                    title="Reject specific paths",
                    content="The lesson mentions agent/context.py and should not be global.",
                    tags=["coding"],
                    confidence="high",
                    evidence="The run touched agent/context.py.",
                ),
                MemoryCandidate(
                    scope="project",
                    kind="session",
                    title="Context task session",
                    content="The run updated context behavior and verified it.",
                    tags=["context"],
                    confidence="medium",
                    evidence="The run completed.",
                ),
            ],
            usage=TokenUsage(input_tokens=3, output_tokens=2),
        )


class EnvironmentFailureSummarizer:
    async def summarize_run(
        self,
        **kwargs: Any,
    ) -> MemorySummaryResult:
        return MemorySummaryResult(
            candidates=[
                MemoryCandidate(
                    scope="project",
                    kind="project_fact",
                    title="Taskledger tests failing",
                    content="The taskledger pytest suite is failing with exit code 1.",
                    tags=["taskledger", "tests", "failing"],
                    confidence="high",
                    evidence="python3 -m pytest exited with code 1.",
                )
            ]
        )


class UpdatingSummarizer:
    def __init__(self) -> None:
        self.calls = 0

    async def summarize_run(self, **kwargs: Any) -> MemorySummaryResult:
        self.calls += 1
        content = (
            "The project uses Python 3.12."
            if self.calls == 1
            else "The project requires Python 3.13."
        )
        title = (
            "Detected Python version"
            if self.calls == 1
            else "Required Python version"
        )
        return MemorySummaryResult(
            candidates=[
                MemoryCandidate(
                    scope="project",
                    kind="project_fact",
                    key="project.python.required_version",
                    title=title,
                    content=content,
                    tags=["python"],
                    confidence="high",
                    evidence="The project metadata declares the version.",
                )
            ]
        )


class UnsupportedProjectFactSummarizer:
    async def summarize_run(self, **kwargs: Any) -> MemorySummaryResult:
        return MemorySummaryResult(
            candidates=[
                MemoryCandidate(
                    scope="project",
                    kind="project_fact",
                    title="Unverified project fact",
                    content="This claim has no supporting run evidence.",
                    confidence="low",
                )
            ]
        )


class DummyProvider:
    pass


def test_memory_system_filters_project_specific_global_candidates(
    tmp_path: Path,
) -> None:
    memory_system = MemorySystem(
        project_store=MemoryStore(tmp_path / "project-memory", "project"),
        global_store=MemoryStore(tmp_path / "global-memory", "global"),
        summarizer=FakeSummarizer(),  # type: ignore[arg-type]
    )
    agent_run = AgentRun(
        run_id="run-one",
        objective="Update context",
        steps=[],
        termination="completed",
        final_stop_reason="end_turn",
        verification=VerificationEvidence(status="not_run"),
    )

    result = asyncio.run(
        memory_system.remember_run(
            provider_adapter=DummyProvider(),  # type: ignore[arg-type]
            session_id="session-one",
            agent_run=agent_run,
            workspace_root=tmp_path,
        )
    )

    assert len(result.saved_records) == 1
    assert result.saved_records[0].scope == "project"
    assert result.skipped_candidates == 1
    assert result.usage == TokenUsage(input_tokens=3, output_tokens=2)


def test_memory_system_skips_exact_duplicate_candidates(tmp_path: Path) -> None:
    project_store = MemoryStore(tmp_path / "project-memory", "project")
    memory_system = MemorySystem(
        project_store=project_store,
        global_store=MemoryStore(tmp_path / "global-memory", "global"),
        summarizer=FakeSummarizer(),  # type: ignore[arg-type]
    )
    agent_run = AgentRun(
        run_id="run-one",
        objective="Update context",
        steps=[],
        termination="completed",
        final_stop_reason="end_turn",
        verification=VerificationEvidence(status="not_run"),
    )

    first = asyncio.run(
        memory_system.remember_run(
            provider_adapter=DummyProvider(),  # type: ignore[arg-type]
            session_id="session-one",
            agent_run=agent_run,
            workspace_root=tmp_path,
        )
    )
    second = asyncio.run(
        memory_system.remember_run(
            provider_adapter=DummyProvider(),  # type: ignore[arg-type]
            session_id="session-one",
            agent_run=agent_run,
            workspace_root=tmp_path,
        )
    )

    assert len(first.saved_records) == 1
    assert second.saved_records == []
    assert second.skipped_candidates == 2
    assert len(project_store.list_records()) == 1


def test_memory_system_updates_stable_fact_without_duplicate_index_entries(
    tmp_path: Path,
) -> None:
    project_store = MemoryStore(tmp_path / "project-memory", "project")
    memory_system = MemorySystem(
        project_store=project_store,
        global_store=MemoryStore(tmp_path / "global-memory", "global"),
        summarizer=UpdatingSummarizer(),  # type: ignore[arg-type]
    )
    agent_run = AgentRun(
        run_id="run-one",
        objective="Inspect Python metadata",
        steps=[],
        termination="completed",
        final_stop_reason="end_turn",
        verification=VerificationEvidence(status="not_run"),
    )

    first = asyncio.run(
        memory_system.remember_run(
            provider_adapter=DummyProvider(),  # type: ignore[arg-type]
            session_id="session-one",
            agent_run=agent_run,
            workspace_root=tmp_path,
        )
    )
    second = asyncio.run(
        memory_system.remember_run(
            provider_adapter=DummyProvider(),  # type: ignore[arg-type]
            session_id="session-one",
            agent_run=agent_run,
            workspace_root=tmp_path,
        )
    )

    records = project_store.list_records()
    assert len(records) == 1
    assert records[0].id == first.saved_records[0].id == second.saved_records[0].id
    assert records[0].content == "The project requires Python 3.13."
    assert records[0].key == "project.python.required_version"
    assert records[0].created_at == first.saved_records[0].created_at


def test_memory_system_rejects_low_confidence_fact_without_evidence(
    tmp_path: Path,
) -> None:
    memory_system = MemorySystem(
        project_store=MemoryStore(tmp_path / "project-memory", "project"),
        global_store=MemoryStore(tmp_path / "global-memory", "global"),
        summarizer=UnsupportedProjectFactSummarizer(),  # type: ignore[arg-type]
    )
    agent_run = AgentRun(
        run_id="run-one",
        objective="Inspect project",
        steps=[],
        termination="completed",
        final_stop_reason="end_turn",
        verification=VerificationEvidence(status="not_run"),
    )

    result = asyncio.run(
        memory_system.remember_run(
            provider_adapter=DummyProvider(),  # type: ignore[arg-type]
            session_id="session-one",
            agent_run=agent_run,
            workspace_root=tmp_path,
        )
    )

    assert result.saved_records == []
    assert result.skipped_candidates == 1


def test_memory_system_reports_persistence_error_without_overwriting_index(
    tmp_path: Path,
) -> None:
    project_store = MemoryStore(tmp_path / "project-memory", "project")
    project_store.initialize()
    project_store.index_path.write_text("{not json", encoding="utf-8")
    memory_system = MemorySystem(
        project_store=project_store,
        global_store=MemoryStore(tmp_path / "global-memory", "global"),
        summarizer=FakeSummarizer(),  # type: ignore[arg-type]
    )
    agent_run = AgentRun(
        run_id="run-one",
        objective="Update context",
        steps=[],
        termination="completed",
        final_stop_reason="end_turn",
        verification=VerificationEvidence(status="not_run"),
    )

    result = asyncio.run(
        memory_system.remember_run(
            provider_adapter=DummyProvider(),  # type: ignore[arg-type]
            session_id="session-one",
            agent_run=agent_run,
            workspace_root=tmp_path,
        )
    )

    assert result.saved_records == []
    assert result.error is not None
    assert "was not overwritten" in result.error
    assert project_store.index_path.read_text(encoding="utf-8") == "{not json"


def test_memory_context_packs_multiple_results_within_character_budget() -> None:
    results = [
        MemorySearchResult(
            record=make_record(
                f"memory-{index}",
                f"Memory {index}",
                "Long memory content. " * 100,
            ),
            score=1.0 - (index / 10),
            lexical_score=1.0,
            boost_score=0.5,
        )
        for index in range(2)
    ]

    prompt = MemoryContext(results=results, max_context_chars=600).format_for_prompt()

    assert len(prompt) <= 600
    assert "memory-0" in prompt
    assert "memory-1" in prompt


def test_memory_system_skips_failure_claim_from_environment_error(
    tmp_path: Path,
) -> None:
    memory_system = MemorySystem(
        project_store=MemoryStore(tmp_path / "project-memory", "project"),
        global_store=MemoryStore(tmp_path / "global-memory", "global"),
        summarizer=EnvironmentFailureSummarizer(),  # type: ignore[arg-type]
    )
    agent_run = AgentRun(
        run_id="run-one",
        objective="Run taskledger tests",
        steps=[],
        termination="max_steps",
        final_stop_reason="tool_use",
        verification=VerificationEvidence(
            status="error",
            command="python3 -m pytest docs/live_memory_eval/taskledger/tests -q",
            exit_code=1,
            output="/path/to/python: No module named pytest",
        ),
    )

    result = asyncio.run(
        memory_system.remember_run(
            provider_adapter=DummyProvider(),  # type: ignore[arg-type]
            session_id="session-one",
            agent_run=agent_run,
            workspace_root=tmp_path,
        )
    )

    assert result.saved_records == []
    assert result.skipped_candidates == 1
