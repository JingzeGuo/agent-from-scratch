import asyncio
from pathlib import Path
from typing import Any

from agent.memory import (
    LocalHybridRetriever,
    MemoryCandidate,
    MemoryRecord,
    MemoryStore,
    MemorySummaryResult,
    MemorySystem,
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


def test_memory_store_saves_markdown_and_index_with_redaction(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory", "project")
    record = make_record(
        "project-session-one",
        "Session with secret",
        "Use api_key=sk-secret123456 only in local env.",
        kind="session",
    )

    saved = store.save_record(record)

    assert saved.path is not None
    markdown = (tmp_path / "memory" / saved.path).read_text(encoding="utf-8")
    assert "sk-secret123456" not in markdown
    assert "api_key=[REDACTED]" in markdown
    assert store.get_record("project-session-one") == saved


def test_memory_store_appends_profile_records(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory", "global")
    record = make_record(
        "global-preference-one",
        "Chinese explanations",
        "The user prefers Chinese explanations for learning discussions.",
        scope="global",
        kind="preference",
    )

    saved = store.save_record(record)

    assert saved.path == "profile.md"
    profile = (tmp_path / "memory" / "profile.md").read_text(encoding="utf-8")
    assert "Chinese explanations" in profile
    assert "global-preference-one" in profile


def test_memory_store_handles_corrupted_index(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory", "project")
    store.initialize()
    store.index_path.write_text("{not json", encoding="utf-8")

    assert store.list_records() == []


def test_local_hybrid_retriever_prefers_relevant_memory() -> None:
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
    retriever = LocalHybridRetriever()

    context = retriever.search(
        "context compaction tool boundary",
        [unrelated, relevant],
    )

    assert [result.record.id for result in context.results] == [
        "context-memory",
        "docs-memory",
    ]
    assert context.results[0].lexical_score > 0
    assert context.results[0].vector_score > 0


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
