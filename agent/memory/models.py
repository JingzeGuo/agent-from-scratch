"""Persistent memory records and the detached task snapshot used to form them."""

from typing import Literal

from pydantic import BaseModel, Field

from ..schemas import RunOutcome

MemoryKind = Literal["fact", "experience"]
MemoryAccess = Literal["core", "retrieval"]
MemoryScope = Literal["user", "repository"]
MemoryProvenance = Literal["explicit_user", "repo_observation", "task_experience"]


class MemoryRecord(BaseModel):
    """One distilled memory in a repository's persistent store."""

    id: str
    kind: MemoryKind
    access: MemoryAccess
    scope: MemoryScope
    provenance: MemoryProvenance
    content: str
    files: list[str] = Field(default_factory=list)
    source_task_id: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    status: Literal["active", "superseded", "deprecated"] = "active"
    supersedes: str | None = None
    created_at: str
    updated_at: str


class TaskMemorySnapshot(BaseModel):
    """Detached formation input; JSON strings keep nested trajectories immutable.

    Command results (including verification) live in steps_json. Injected memory
    records include their IDs and original content for echo detection.
    """

    model_config = {"frozen": True}

    task_id: str
    objective: str
    termination: RunOutcome
    steps_json: str
    files_changed: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    injected_memories_json: str
