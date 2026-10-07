import asyncio
import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from agent.schemas import MemoryRecord, TaskMemorySnapshot, TokenUsage
from agent.state.context import count_tokens
from agent.state.memory import (
    MemoryConfig,
    MemoryStore,
    load_task_memory,
    render_memories,
)
from agent.state.memory_formation import LLMMemoryFormation
from tests.test_agent import FakeProviderAdapter, TextBlock, make_message


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    memory_store = MemoryStore(tmp_path / "memory.sqlite3")
    yield memory_store
    memory_store.close()


def memory(
    content: str = "Parser token lists may be empty.", **changes: object
) -> MemoryRecord:
    return MemoryRecord.model_validate(
        {
            "id": f"memory-{uuid4().hex}",
            "kind": "fact",
            "access": "retrieval",
            "scope": "repository",
            "provenance": "repo_observation",
            "content": content,
            "files": ["src/parser.py"],
            "source_task_id": "earlier-run",
            "evidence_refs": ["task:earlier-run:step:1 read_file src/parser.py"],
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            **changes,
        }
    )


def snapshot(injected: list[MemoryRecord] | None = None) -> TaskMemorySnapshot:
    return TaskMemorySnapshot(
        task_id="run-current",
        objective="Inspect src/parser.py and repair empty token handling",
        termination="completed",
        steps_json="[]",
        files_changed=(),
        evidence_refs=("task:run-current:step:1 read_file src/parser.py",),
        injected_memories_json=json.dumps([m.model_dump() for m in injected or []]),
    )


def candidate(**changes: object) -> dict[str, object]:
    return {
        "route": "RETRIEVAL",
        "kind": "fact",
        "scope": "repository",
        "provenance": "repo_observation",
        "content": "The parser accepts an empty token list after validation.",
        "files": ["src/parser.py"],
        "evidence_refs": list(snapshot().evidence_refs),
        "new_information": "The current parser code confirms empty token support.",
        **changes,
    }


def provider(*outputs: object) -> FakeProviderAdapter:
    return FakeProviderAdapter(
        [
            make_message([TextBlock(text=json.dumps(output))], "end_turn")
            for output in outputs
        ]
    )


def count(store: MemoryStore) -> int:
    return int(store.connection.execute("SELECT count(*) FROM memories").fetchone()[0])


def test_sqlite_round_trip_and_scopes(tmp_path: Path) -> None:
    path = tmp_path / "state" / "memory.sqlite3"
    records = [
        memory(),
        memory(
            "Prefer small patches.",
            scope="user",
            access="core",
            provenance="explicit_user",
        ),
        memory(
            "Situation: empty tokens. Action: guard access. Outcome: passed. Lesson: validate first.",
            kind="experience",
            provenance="task_experience",
        ),
    ]
    store = MemoryStore(path)
    for record in records:
        store.add(record)
    store.close()
    reopened = MemoryStore(path)
    try:
        assert [reopened.get(m.id) for m in records] == records
        assert reopened.get_core() == [records[1]]
        assert reopened.get("missing") is None
    finally:
        reopened.close()


def test_lexical_path_relevance_deduplication_and_active_filter(
    store: MemoryStore,
) -> None:
    relevant = memory()
    store.add(relevant)
    store.add(memory(relevant.content + "!"))
    store.add(memory("The renderer uses offscreen surfaces.", files=["src/render.py"]))
    store.add(memory("Parser tokenization is deprecated.", status="deprecated"))
    store.add(memory("Old parser token handling.", status="superseded"))
    store.add(memory("The parser uses Python.", access="core"))
    found = store.search("Repair src/parser.py empty tokens")
    assert len(found) == 1
    assert found[0].content.startswith(relevant.content)
    assert store.search("unrelated observability dashboard") == []
    assert store.search('" OR * ( ) --') == []


def test_file_overlap_can_retrieve_without_objective_overlap(
    store: MemoryStore,
) -> None:
    record = memory("Historical lesson: validate the input before indexing.")
    store.add(record)
    assert store.search("Repair exception", files=["src/parser.py"]) == [record]
    assert store.search("Repair exception", files=["src/render.py"]) == []


def test_core_and_retrieval_budget_includes_section_metadata(
    store: MemoryStore,
) -> None:
    core = memory("Use pytest for verification.", access="core", files=[])
    retrieved = memory()
    store.add(memory("Oversized " * 1000, access="core"))
    store.add(core)
    store.add(memory("Parser " * 1000))
    store.add(retrieved)
    core_limit = count_tokens(render_memories("Core memories", [core]))
    retrieval_limit = count_tokens(
        render_memories("Relevant retrieved memories", [retrieved])
    )
    config = MemoryConfig(core_tokens=core_limit, retrieval_tokens=retrieval_limit)
    context = load_task_memory(store, "parser", config)
    assert context.core_memories == (core,)
    assert context.retrieved_memories == (retrieved,)
    assert (
        count_tokens(render_memories("Core memories", context.core_memories))
        <= core_limit
    )
    assert (
        count_tokens(
            render_memories("Relevant retrieved memories", context.retrieved_memories)
        )
        <= retrieval_limit
    )
    empty = load_task_memory(
        store, "parser", MemoryConfig(core_tokens=0, retrieval_tokens=0)
    )
    assert empty.memories == ()
    assert empty.render() == ""


def test_retrieval_obeys_top_k(store: MemoryStore) -> None:
    for description in (
        "syntax trees",
        "unicode normalization",
        "comments",
        "indentation",
        "strings",
        "operators",
    ):
        store.add(memory(f"Parser supports {description}.", files=[]))
    context = load_task_memory(store, "parser", MemoryConfig(top_k=3))
    assert len(context.retrieved_memories) == 3


def test_trivial_task_can_form_zero_memories(store: MemoryStore) -> None:
    llm = provider({"candidates": []})
    asyncio.run(LLMMemoryFormation(llm, store).form(snapshot()))
    assert count(store) == 0
    assert llm.call_count == 1
    assert llm.requests[0]["tools"] == []


def test_add_and_usage_accounting(store: MemoryStore) -> None:
    llm = provider({"candidates": [candidate()]}, {"action": "ADD"})
    usages: list[TokenUsage] = []
    asyncio.run(LLMMemoryFormation(llm, store, on_usage=usages.append).form(snapshot()))
    assert count(store) == 1
    record = store.search("parser")[0]
    assert record.source_task_id == "run-current"
    assert record.evidence_refs == list(snapshot().evidence_refs)
    assert record.status == "active"
    assert record.kind == "fact"
    assert len(usages) == 2


def test_merge_updates_same_record_and_retains_evidence(store: MemoryStore) -> None:
    original = memory()
    store.add(original)
    merged_content = original.content + " Validation accepts empty lists."
    llm = provider(
        {"candidates": [candidate()]},
        {"action": "MERGE", "target_id": original.id, "content": merged_content},
    )
    asyncio.run(LLMMemoryFormation(llm, store).form(snapshot()))
    assert count(store) == 1
    merged = store.get(original.id)
    assert merged is not None
    assert merged.content == merged_content
    assert merged.created_at == original.created_at
    assert merged.updated_at != original.updated_at
    assert merged.evidence_refs == original.evidence_refs + list(
        snapshot().evidence_refs
    )
    assert store.search("validation")[0].id == original.id  # FTS updated too.


def test_supersede_retains_old_record_and_filters_retrieval(store: MemoryStore) -> None:
    original = memory("The parser requires nonempty tokens.", access="core")
    store.add(original)
    llm = provider(
        {"candidates": [candidate(route="CORE")]},
        {"action": "SUPERSEDE", "target_id": original.id},
    )
    asyncio.run(LLMMemoryFormation(llm, store).form(snapshot()))
    assert count(store) == 2
    old = store.get(original.id)
    assert old is not None and old.status == "superseded"
    active = store.get_core()
    assert len(active) == 1 and active[0].supersedes == original.id
    assert store.search("parser", access=None) == active


@pytest.mark.parametrize("action", ["MERGE", "SUPERSEDE"])
@pytest.mark.parametrize(
    "access,route", [("core", "RETRIEVAL"), ("retrieval", "CORE")]
)
def test_memory_updates_inherit_access_despite_candidate_route(
    store: MemoryStore, action: str, access: str, route: str
) -> None:
    original = memory(access=access)
    store.add(original)
    llm = provider(
        {"candidates": [candidate(route=route)]},
        {
            "action": action,
            "target_id": original.id,
            "content": "The parser accepts empty token lists after validation.",
        },
    )

    asyncio.run(LLMMemoryFormation(llm, store).form(snapshot()))

    active = store.search("parser", access=None)
    assert len(active) == 1
    assert active[0].access == original.access


@pytest.mark.parametrize("action", ["MERGE", "SUPERSEDE"])
def test_inherited_core_access_still_obeys_budget(
    store: MemoryStore, action: str
) -> None:
    original = memory(access="core")
    store.add(original)
    budget = count_tokens(render_memories("Core memories", [original]))
    content = "Parser validation handles empty token lists. " * 100
    llm = provider(
        {"candidates": [candidate(route="CORE", content=content)]},
        {"action": action, "target_id": original.id, "content": content},
    )

    asyncio.run(
        LLMMemoryFormation(llm, store, config=MemoryConfig(core_tokens=budget)).form(
            snapshot()
        )
    )

    assert store.get_core() == []
    assert store.search("parser")[0].content == content


def test_lifecycle_noop_performs_no_write(store: MemoryStore) -> None:
    original = memory()
    store.add(original)
    llm = provider({"candidates": [candidate()]}, {"action": "NOOP"})
    before = store.connection.total_changes
    asyncio.run(LLMMemoryFormation(llm, store).form(snapshot()))
    assert store.connection.total_changes == before
    assert store.get(original.id) == original


@pytest.mark.parametrize("exact", [True, False])
def test_injected_memory_echo_never_creates_duplicate(
    store: MemoryStore, exact: bool
) -> None:
    original = memory()
    store.add(original)
    repeated = candidate(
        content=original.content if exact else "The parser can receive zero tokens.",
        new_information="Successful reuse" if exact else "",
    )
    llm = provider({"candidates": [repeated]})
    before = store.connection.total_changes
    asyncio.run(LLMMemoryFormation(llm, store).form(snapshot([original])))
    assert llm.call_count == 1
    assert store.connection.total_changes == before
    payload = json.loads(llm.requests[0]["messages"][0]["content"])
    assert payload["injected_memories"][0]["id"] == original.id


def test_semantic_echo_can_be_rejected_by_lifecycle(store: MemoryStore) -> None:
    original = memory()
    store.add(original)
    llm = provider({"candidates": [candidate()]}, {"action": "NOOP"})
    asyncio.run(LLMMemoryFormation(llm, store).form(snapshot([original])))
    payload = json.loads(llm.requests[1]["messages"][0]["content"])
    assert payload["similar_memories"][0]["id"] == original.id
    assert count(store) == 1


@pytest.mark.parametrize(
    "refs", [[], ["memory:M5"], ["task:other:step:5 run_command pytest"]]
)
def test_candidate_requires_current_task_evidence(
    store: MemoryStore, refs: list[str]
) -> None:
    llm = provider({"candidates": [candidate(evidence_refs=refs)]})
    asyncio.run(LLMMemoryFormation(llm, store).form(snapshot()))
    assert count(store) == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "experience", "provenance": "task_experience"},
        {"scope": "user", "provenance": "task_experience"},
    ],
)
def test_ineligible_core_candidates_go_to_retrieval(
    store: MemoryStore, changes: dict[str, str]
) -> None:
    llm = provider(
        {"candidates": [candidate(route="CORE", **changes)]}, {"action": "ADD"}
    )
    asyncio.run(LLMMemoryFormation(llm, store).form(snapshot()))
    assert store.get_core() == []
    assert len(store.search("parser")) == 1


def test_core_admission_is_bounded(store: MemoryStore) -> None:
    llm = provider({"candidates": [candidate(route="CORE")]}, {"action": "ADD"})
    asyncio.run(
        LLMMemoryFormation(llm, store, config=MemoryConfig(core_tokens=1)).form(
            snapshot()
        )
    )
    assert store.get_core() == []
    assert len(store.search("parser")) == 1


def test_experience_history_cannot_be_superseded(store: MemoryStore) -> None:
    old = memory(kind="experience", provenance="task_experience")
    store.add(old)
    llm = provider(
        {"candidates": [candidate(kind="experience", provenance="task_experience")]},
        {"action": "SUPERSEDE", "target_id": old.id},
    )
    with pytest.raises(ValueError, match="active fact"):
        asyncio.run(LLMMemoryFormation(llm, store).form(snapshot()))
    assert count(store) == 1 and store.get(old.id) == old


def test_replacement_is_atomic_on_insert_failure(store: MemoryStore) -> None:
    old = memory()
    store.add(old)
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint"):
        store.mark_superseded(old.id, old)
    assert store.get(old.id) == old


def test_snapshot_and_structured_output_are_validated(store: MemoryStore) -> None:
    captured = snapshot()
    with pytest.raises(ValidationError):
        captured.objective = "Mutated"
    llm = provider({"candidates": [candidate()] * 4})
    with pytest.raises(ValidationError):
        asyncio.run(LLMMemoryFormation(llm, store).form(captured))
    assert count(store) == 0


def test_unrelated_lifecycle_target_is_rejected(store: MemoryStore) -> None:
    original = memory()
    unrelated = memory("Deployment regions are fixed.", files=[])
    store.add(original)
    store.add(unrelated)
    llm = provider(
        {"candidates": [candidate()]},
        {"action": "MERGE", "target_id": unrelated.id, "content": "Wrong target"},
    )
    with pytest.raises(ValueError, match="active similar"):
        asyncio.run(LLMMemoryFormation(llm, store).form(snapshot()))
    assert store.get(unrelated.id) == unrelated
