from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from .provider import ProviderAdapter
from .schemas import AgentRun, AgentStep, TokenUsage, ToolResult
from .security import redact_text

MemoryScope = Literal["project", "global"]
MemoryKind = Literal[
    "profile",
    "preference",
    "cross_project_lesson",
    "project_fact",
    "session",
    "topic",
    "reflection",
]
MemoryConfidence = Literal["low", "medium", "high"]

PROFILE_KINDS: set[MemoryKind] = {
    "profile",
    "preference",
    "cross_project_lesson",
}
PROJECT_KINDS: set[MemoryKind] = {
    "profile",
    "project_fact",
    "session",
    "topic",
    "reflection",
}
GLOBAL_KINDS: set[MemoryKind] = {
    "profile",
    "preference",
    "cross_project_lesson",
}

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
PATH_LIKE_PATTERN = re.compile(
    r"(`[^`]*[/\\][^`]*`|[/\\][A-Za-z0-9_.-]+|[A-Za-z0-9_.-]+\.(py|md|toml|json|yaml|yml|txt))"
)
COMMAND_LIKE_PATTERN = re.compile(
    r"\b(pytest|ruff|mypy|python\s+-m|git\s+|uv\s+|pip\s+|npm\s+|pnpm\s+|cargo\s+)\b"
)
FAILURE_CLAIM_PATTERN = re.compile(
    r"\b(test|tests|test suite|verification|pytest|ruff|mypy)\b[^.\n]{0,80}"
    r"\b(fail|failed|failing|exit code 1)\b"
    r"|\b(fail|failed|failing)\b[^.\n]{0,80}"
    r"\b(test|tests|verification|pytest|ruff|mypy)\b",
    re.IGNORECASE,
)
ENVIRONMENT_VERIFICATION_PATTERN = re.compile(
    r"(approval denied|requires approval|no module named (pytest|ruff|mypy)|"
    r"command not found|no such file or directory)",
    re.IGNORECASE,
)
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}


class MemoryRecord(BaseModel):
    """One durable memory entry stored in a project or global memory store."""

    id: str
    scope: MemoryScope
    kind: MemoryKind
    title: str
    content: str
    tags: list[str] = Field(default_factory=list)
    source: str | None = None
    confidence: MemoryConfidence | None = None
    evidence: str | None = None
    created_at: str
    updated_at: str
    path: str | None = None


class MemoryCandidate(BaseModel):
    """Model-produced candidate that the controller may persist as memory."""

    scope: MemoryScope
    kind: MemoryKind
    title: str
    content: str
    tags: list[str] = Field(default_factory=list)
    confidence: MemoryConfidence = "medium"
    evidence: str | None = None


class MemorySearchResult(BaseModel):
    record: MemoryRecord
    score: float
    lexical_score: float
    vector_score: float
    boost_score: float


class MemoryContext(BaseModel):
    """Retrieved memory prepared for prompt injection."""

    results: list[MemorySearchResult] = Field(default_factory=list)
    max_context_chars: int = Field(default=4_000, ge=200)

    def is_empty(self) -> bool:
        return not self.results

    def format_for_prompt(self) -> str:
        lines = [
            "[Retrieved memory]",
            "Use these notes as supporting context. System rules and project rules take precedence.",
        ]
        for result in self.results:
            record = result.record
            tags = ", ".join(record.tags) if record.tags else "none"
            lines.extend(
                [
                    "",
                    f"- id: {record.id}",
                    f"  scope: {record.scope}",
                    f"  kind: {record.kind}",
                    f"  title: {record.title}",
                    f"  tags: {tags}",
                    f"  score: {result.score:.3f}",
                    "  content:",
                    _indent(record.content.strip() or "[empty]", "    "),
                ]
            )
        return _truncate_text("\n".join(lines), self.max_context_chars)


class MemorySummaryResult(BaseModel):
    candidates: list[MemoryCandidate] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=lambda: TokenUsage(input_tokens=0, output_tokens=0))


class MemoryWriteResult(BaseModel):
    saved_records: list[MemoryRecord] = Field(default_factory=list)
    skipped_candidates: int = 0
    error: str | None = None
    usage: TokenUsage = Field(default_factory=lambda: TokenUsage(input_tokens=0, output_tokens=0))


class MemoryStatus(BaseModel):
    enabled: bool
    project_root: str
    global_root: str
    project_records: int
    global_records: int


class MemoryIndex(BaseModel):
    records: list[MemoryRecord] = Field(default_factory=list)


class MemoryStore:
    """Filesystem-backed memory store with Markdown records and a JSON index."""

    def __init__(self, root: Path, scope: MemoryScope) -> None:
        self.root = root.expanduser()
        self.scope = scope

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in ("sessions", "topics", "reflections"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        if not self.profile_path.exists():
            self.profile_path.write_text("# Memory Profile\n", encoding="utf-8")
        if not self.index_path.exists():
            self._write_index(MemoryIndex())

    @property
    def index_path(self) -> Path:
        return self.root / "index.json"

    @property
    def profile_path(self) -> Path:
        return self.root / "profile.md"

    def list_records(self) -> list[MemoryRecord]:
        records = self._read_index().records
        if self.profile_path.exists() and not any(
            record.kind == "profile" and record.path == "profile.md"
            for record in records
        ):
            content = self.profile_path.read_text(encoding="utf-8")
            if content.strip() and content.strip() != "# Memory Profile":
                records.append(
                    MemoryRecord(
                        id=f"{self.scope}-profile",
                        scope=self.scope,
                        kind="profile",
                        title=f"{self.scope.title()} memory profile",
                        content=redact_text(content),
                        tags=["profile"],
                        source="profile.md",
                        created_at=_utc_timestamp(),
                        updated_at=_utc_timestamp(),
                        path="profile.md",
                    )
                )
        return records

    def get_record(self, record_id: str) -> MemoryRecord | None:
        for record in self.list_records():
            if record.id == record_id:
                return record
        return None

    def save_record(self, record: MemoryRecord) -> MemoryRecord:
        self.initialize()
        safe_record = self._redact_record(record)
        path = self._write_record_markdown(safe_record)
        safe_record = safe_record.model_copy(
            update={"path": path.relative_to(self.root).as_posix()}
        )
        index = self._read_index()
        records = [entry for entry in index.records if entry.id != safe_record.id]
        records.append(safe_record)
        self._write_index(MemoryIndex(records=records))
        return safe_record

    def _read_index(self) -> MemoryIndex:
        if not self.index_path.exists():
            return MemoryIndex()
        try:
            return MemoryIndex.model_validate_json(
                self.index_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, ValidationError):
            return MemoryIndex()

    def _write_index(self, index: MemoryIndex) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary_path = self.index_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            index.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self.index_path)

    def _write_record_markdown(self, record: MemoryRecord) -> Path:
        if record.kind in PROFILE_KINDS:
            self._append_profile_record(record)
            return self.profile_path

        directory = self._directory_for_kind(record.kind)
        directory.mkdir(parents=True, exist_ok=True)
        slug = _slugify(record.title)
        path = directory / f"{record.created_at[:10]}-{slug}-{record.id[-8:]}.md"
        path.write_text(self._format_record_markdown(record), encoding="utf-8")
        return path

    def _append_profile_record(self, record: MemoryRecord) -> None:
        if not self.profile_path.exists():
            self.profile_path.parent.mkdir(parents=True, exist_ok=True)
            self.profile_path.write_text("# Memory Profile\n", encoding="utf-8")
        with self.profile_path.open("a", encoding="utf-8") as file:
            file.write("\n")
            file.write(f"## {record.title}\n\n")
            file.write(f"- id: `{record.id}`\n")
            file.write(f"- kind: `{record.kind}`\n")
            if record.confidence is not None:
                file.write(f"- confidence: `{record.confidence}`\n")
            if record.evidence:
                file.write(f"- evidence: {record.evidence}\n")
            if record.tags:
                file.write(f"- tags: {', '.join(record.tags)}\n")
            file.write("\n")
            file.write(record.content.strip() + "\n")

    def _directory_for_kind(self, kind: MemoryKind) -> Path:
        if kind == "session":
            return self.root / "sessions"
        if kind == "reflection":
            return self.root / "reflections"
        return self.root / "topics"

    def _format_record_markdown(self, record: MemoryRecord) -> str:
        frontmatter = {
            "id": record.id,
            "scope": record.scope,
            "kind": record.kind,
            "tags": record.tags,
            "source": record.source,
            "confidence": record.confidence,
            "evidence": record.evidence,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }
        return (
            "---\n"
            + json.dumps(frontmatter, indent=2, ensure_ascii=False)
            + "\n---\n\n"
            + f"# {record.title}\n\n"
            + record.content.strip()
            + "\n"
        )

    def _redact_record(self, record: MemoryRecord) -> MemoryRecord:
        return record.model_copy(
            update={
                "title": redact_text(record.title),
                "content": redact_text(record.content),
                "tags": [redact_text(tag) for tag in record.tags],
                "source": None if record.source is None else redact_text(record.source),
                "evidence": None
                if record.evidence is None
                else redact_text(record.evidence),
            }
        )


class LocalHybridRetriever:
    """BM25-like lexical retrieval plus local TF-IDF cosine scoring."""

    def search(
        self,
        query: str,
        records: list[MemoryRecord],
        *,
        max_results: int = 5,
        max_context_chars: int = 4_000,
    ) -> MemoryContext:
        query_tokens = _tokenize(query)
        if not query_tokens or not records:
            return MemoryContext(max_context_chars=max_context_chars)

        documents = [_document_tokens(record) for record in records]
        document_frequencies = self._document_frequencies(documents)
        bm25_scores = self._bm25_scores(query_tokens, documents, document_frequencies)
        vector_scores = self._vector_scores(query_tokens, documents, document_frequencies)
        max_bm25 = max(bm25_scores) if bm25_scores else 0.0

        results: list[MemorySearchResult] = []
        for index, record in enumerate(records):
            normalized_bm25 = bm25_scores[index] / max_bm25 if max_bm25 > 0 else 0.0
            vector_score = vector_scores[index]
            boost_score = self._boost_score(record)
            score = (0.6 * normalized_bm25) + (0.3 * vector_score) + (0.1 * boost_score)
            if score <= 0:
                continue
            results.append(
                MemorySearchResult(
                    record=record,
                    score=score,
                    lexical_score=normalized_bm25,
                    vector_score=vector_score,
                    boost_score=boost_score,
                )
            )

        return MemoryContext(
            results=sorted(results, key=lambda result: result.score, reverse=True)[
                :max_results
            ],
            max_context_chars=max_context_chars,
        )

    def _document_frequencies(
        self,
        documents: list[list[str]],
    ) -> dict[str, int]:
        frequencies: dict[str, int] = {}
        for tokens in documents:
            for token in set(tokens):
                frequencies[token] = frequencies.get(token, 0) + 1
        return frequencies

    def _bm25_scores(
        self,
        query_tokens: list[str],
        documents: list[list[str]],
        document_frequencies: dict[str, int],
    ) -> list[float]:
        document_count = len(documents)
        average_length = (
            sum(len(document) for document in documents) / document_count
            if document_count
            else 0.0
        )
        k1 = 1.5
        b = 0.75
        scores: list[float] = []
        for document in documents:
            term_counts = Counter(document)
            document_length = len(document) or 1
            score = 0.0
            for token in set(query_tokens):
                frequency = term_counts[token]
                if frequency == 0:
                    continue
                document_frequency = document_frequencies.get(token, 0)
                idf = math.log(
                    1
                    + (document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                denominator = frequency + k1 * (
                    1 - b + b * (document_length / (average_length or 1))
                )
                score += idf * ((frequency * (k1 + 1)) / denominator)
            scores.append(score)
        return scores

    def _vector_scores(
        self,
        query_tokens: list[str],
        documents: list[list[str]],
        document_frequencies: dict[str, int],
    ) -> list[float]:
        document_count = len(documents)
        query_vector = self._tfidf_vector(
            Counter(query_tokens),
            document_frequencies,
            document_count,
        )
        query_norm = _vector_norm(query_vector)
        if query_norm == 0:
            return [0.0 for _ in documents]

        scores: list[float] = []
        for document in documents:
            document_vector = self._tfidf_vector(
                Counter(document),
                document_frequencies,
                document_count,
            )
            document_norm = _vector_norm(document_vector)
            if document_norm == 0:
                scores.append(0.0)
                continue
            dot_product = sum(
                value * document_vector.get(token, 0.0)
                for token, value in query_vector.items()
            )
            scores.append(dot_product / (query_norm * document_norm))
        return scores

    def _tfidf_vector(
        self,
        term_counts: Counter[str],
        document_frequencies: dict[str, int],
        document_count: int,
    ) -> dict[str, float]:
        total_terms = sum(term_counts.values()) or 1
        vector: dict[str, float] = {}
        for token, count in term_counts.items():
            tf = count / total_terms
            idf = math.log(1 + document_count / (1 + document_frequencies.get(token, 0)))
            vector[token] = tf * idf
        return vector

    def _boost_score(self, record: MemoryRecord) -> float:
        kind_boosts: dict[MemoryKind, float] = {
            "profile": 1.0,
            "preference": 0.9,
            "cross_project_lesson": 0.8,
            "topic": 0.7,
            "project_fact": 0.6,
            "reflection": 0.5,
            "session": 0.3,
        }
        recency = _recency_score(record.created_at)
        return min(1.0, (0.8 * kind_boosts.get(record.kind, 0.0)) + (0.2 * recency))


class MemorySummarizer:
    """Use the configured model to turn a completed run into memory candidates."""

    async def summarize_run(
        self,
        *,
        provider_adapter: ProviderAdapter,
        session_id: str,
        run_id: str,
        agent_run: AgentRun,
        workspace_root: Path | None,
    ) -> MemorySummaryResult:
        evidence = self._format_run_evidence(
            session_id=session_id,
            run_id=run_id,
            agent_run=agent_run,
            workspace_root=workspace_root,
        )
        response = await provider_adapter.stream_response(
            system=self._system_prompt(),
            tools=[],
            messages=[{"role": "user", "content": evidence}],
            on_text_delta=None,
        )
        text = "\n".join(response.text).strip()
        candidates = self._parse_candidates(text)
        return MemorySummaryResult(candidates=candidates, usage=response.usage)

    def _system_prompt(self) -> str:
        return """You summarize completed coding-agent runs into durable memory candidates.

Return JSON only, with this shape:
{
  "candidates": [
    {
      "scope": "project" | "global",
      "kind": "session" | "reflection" | "topic" | "project_fact" | "preference" | "cross_project_lesson",
      "title": "short title",
      "content": "concise durable memory",
      "tags": ["short", "tags"],
      "confidence": "low" | "medium" | "high",
      "evidence": "specific evidence from the run or user wording"
    }
  ]
}

Use project scope for repository facts, session notes, file-specific decisions, and concrete debugging history.
Use global scope only for stable user preferences or cross-project lessons that remain useful in another repository.
If verification failed because the interpreter, pytest, ruff, mypy, command approval, or another environment/tooling prerequisite was unavailable, do not claim the repository tests or code failed. Store that only as a session/tooling note if it is useful.
Do not include raw tool output, full diffs, secrets, API keys, or long logs."""

    def _format_run_evidence(
        self,
        *,
        session_id: str,
        run_id: str,
        agent_run: AgentRun,
        workspace_root: Path | None,
    ) -> str:
        facts = _extract_run_facts(agent_run.steps)
        lines = [
            "Summarize this completed coding-agent run into memory candidates.",
            "",
            f"session_id: {session_id}",
            f"run_id: {run_id}",
            f"objective: {redact_text(agent_run.objective)}",
            f"termination: {agent_run.termination}",
            f"task_success: {agent_run.task_success}",
            f"workspace_root: {workspace_root.as_posix() if workspace_root else '[none]'}",
            f"verification: {agent_run.verification.status}",
        ]
        if agent_run.verification.command is not None:
            lines.append(f"verification_command: {agent_run.verification.command}")
        if agent_run.verification.output is not None:
            lines.append(
                f"verification_output: {_first_diagnostic_line(agent_run.verification.output)}"
            )
        lines.extend(_format_list("files_read", facts["files_read"]))
        lines.extend(_format_list("files_changed", facts["files_changed"]))
        lines.extend(_format_list("commands_run", facts["commands_run"]))
        lines.extend(_format_list("tool_errors", facts["tool_errors"]))
        lines.extend(_format_list("model_decisions", facts["decisions"]))
        return redact_text("\n".join(lines))

    def _parse_candidates(self, text: str) -> list[MemoryCandidate]:
        if not text:
            return []
        data = json.loads(_extract_json_text(text))
        raw_candidates = data.get("candidates") if isinstance(data, dict) else None
        if not isinstance(raw_candidates, list):
            return []

        candidates: list[MemoryCandidate] = []
        for raw_candidate in raw_candidates:
            try:
                candidates.append(MemoryCandidate.model_validate(raw_candidate))
            except ValidationError:
                continue
        return candidates


class MemorySystem:
    """Coordinates retrieval, prompt context, summarization, and persistence."""

    def __init__(
        self,
        *,
        project_store: MemoryStore,
        global_store: MemoryStore,
        enabled: bool = True,
        max_results: int = 5,
        max_context_chars: int = 4_000,
        retriever: LocalHybridRetriever | None = None,
        summarizer: MemorySummarizer | None = None,
    ) -> None:
        self.project_store = project_store
        self.global_store = global_store
        self.enabled = enabled
        self.max_results = max_results
        self.max_context_chars = max_context_chars
        self.retriever = retriever or LocalHybridRetriever()
        self.summarizer = summarizer or MemorySummarizer()

    def initialize(self) -> None:
        if not self.enabled:
            return
        self.project_store.initialize()
        self.global_store.initialize()

    def status(self) -> MemoryStatus:
        return MemoryStatus(
            enabled=self.enabled,
            project_root=self.project_store.root.as_posix(),
            global_root=self.global_store.root.as_posix(),
            project_records=len(self.project_store.list_records()),
            global_records=len(self.global_store.list_records()),
        )

    def search(self, query: str) -> MemoryContext:
        if not self.enabled:
            return MemoryContext(max_context_chars=self.max_context_chars)
        records = self.project_store.list_records() + self.global_store.list_records()
        return self.retriever.search(
            query,
            records,
            max_results=self.max_results,
            max_context_chars=self.max_context_chars,
        )

    def get_record(self, record_id: str) -> MemoryRecord | None:
        return self.project_store.get_record(record_id) or self.global_store.get_record(
            record_id
        )

    async def remember_run(
        self,
        *,
        provider_adapter: ProviderAdapter,
        session_id: str,
        agent_run: AgentRun,
        workspace_root: Path | None,
    ) -> MemoryWriteResult:
        if not self.enabled:
            return MemoryWriteResult()
        try:
            summary = await self.summarizer.summarize_run(
                provider_adapter=provider_adapter,
                session_id=session_id,
                run_id=agent_run.run_id or "",
                agent_run=agent_run,
                workspace_root=workspace_root,
            )
        except Exception as error:
            return MemoryWriteResult(error=str(error))

        saved_records: list[MemoryRecord] = []
        skipped_candidates = 0
        for candidate in summary.candidates:
            if not self._candidate_allowed(candidate, workspace_root, agent_run):
                skipped_candidates += 1
                continue
            record = self._record_from_candidate(candidate, agent_run)
            store = self.global_store if candidate.scope == "global" else self.project_store
            saved_records.append(store.save_record(record))
        return MemoryWriteResult(
            saved_records=saved_records,
            skipped_candidates=skipped_candidates,
            usage=summary.usage,
        )

    def _candidate_allowed(
        self,
        candidate: MemoryCandidate,
        workspace_root: Path | None,
        agent_run: AgentRun,
    ) -> bool:
        if candidate.scope == "project":
            return candidate.kind in PROJECT_KINDS and not _unsupported_failure_claim(
                candidate,
                agent_run,
            )

        if candidate.kind not in GLOBAL_KINDS:
            return False
        if candidate.confidence == "low":
            return False
        if not candidate.evidence or not candidate.evidence.strip():
            return False

        global_text = "\n".join(
            [
                candidate.title,
                candidate.content,
                candidate.evidence or "",
                " ".join(candidate.tags),
            ]
        )
        if workspace_root is not None and workspace_root.as_posix() in global_text:
            return False
        if PATH_LIKE_PATTERN.search(global_text) is not None:
            return False
        if COMMAND_LIKE_PATTERN.search(global_text) is not None:
            return False
        return True

    def _record_from_candidate(
        self,
        candidate: MemoryCandidate,
        agent_run: AgentRun,
    ) -> MemoryRecord:
        now = _utc_timestamp()
        return MemoryRecord(
            id=_memory_id(candidate.scope, candidate.kind, candidate.title),
            scope=candidate.scope,
            kind=candidate.kind,
            title=candidate.title.strip(),
            content=candidate.content.strip(),
            tags=[tag.strip() for tag in candidate.tags if tag.strip()],
            source=agent_run.run_id,
            confidence=candidate.confidence,
            evidence=None
            if candidate.evidence is None
            else candidate.evidence.strip(),
            created_at=now,
            updated_at=now,
        )


def _extract_run_facts(steps: list[AgentStep]) -> dict[str, list[str]]:
    files_read: set[str] = set()
    files_changed: set[str] = set()
    commands_run: list[str] = []
    tool_errors: list[str] = []
    decisions: list[str] = []

    for step in steps:
        decisions.extend(text for text in step.text if text.strip())
        for tool_call, tool_result in zip(step.tool_calls, step.tool_results):
            path = tool_call.input.get("path")
            if tool_call.name == "read_file" and isinstance(path, str):
                files_read.add(path)
            if tool_call.name in {"edit_file", "write_file"} and isinstance(path, str):
                files_changed.add(path)
            if tool_call.name == "run_command":
                command = tool_call.input.get("command")
                if isinstance(command, str):
                    commands_run.append(_command_summary(command, tool_result))
            if tool_result.is_error:
                tool_errors.append(
                    f"step {step.step_number} {tool_call.name}: {_first_line(tool_result.content)}"
                )

    return {
        "files_read": sorted(files_read),
        "files_changed": sorted(files_changed),
        "commands_run": commands_run,
        "tool_errors": tool_errors,
        "decisions": [_truncate_text(decision, 500) for decision in decisions],
    }


def _command_summary(command: str, tool_result: ToolResult) -> str:
    first_line = _first_diagnostic_line(tool_result.content)
    status = "error" if tool_result.is_error else "ok"
    return f"{status}: {command} ({first_line})"


def _first_line(text: str) -> str:
    return text.splitlines()[0] if text.splitlines() else ""


def _first_diagnostic_line(text: str) -> str:
    metadata_prefixes = ("exit_code:", "timed_out:", "duration_seconds:")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped in {"stdout:", "stderr:", "[empty]"}:
            continue
        if stripped.startswith(metadata_prefixes):
            continue
        return stripped
    return _first_line(text)


def _unsupported_failure_claim(
    candidate: MemoryCandidate,
    agent_run: AgentRun,
) -> bool:
    if agent_run.verification.status != "error":
        return False
    verification_text = "\n".join(
        value
        for value in (
            agent_run.verification.command,
            agent_run.verification.output,
        )
        if value
    )
    if ENVIRONMENT_VERIFICATION_PATTERN.search(verification_text) is None:
        return False
    candidate_text = "\n".join(
        [
            candidate.title,
            candidate.content,
            candidate.evidence or "",
            " ".join(candidate.tags),
        ]
    )
    return FAILURE_CLAIM_PATTERN.search(candidate_text) is not None


def _format_list(heading: str, values: list[str]) -> list[str]:
    lines = [f"{heading}:"]
    if values:
        lines.extend(f"- {redact_text(value)}" for value in values)
    else:
        lines.append("- none")
    return lines


def _extract_json_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise json.JSONDecodeError("No JSON object found", stripped, 0)
    return stripped[start : end + 1]


def _document_tokens(record: MemoryRecord) -> list[str]:
    title_tokens = _tokenize(record.title) * 3
    tag_tokens = _tokenize(" ".join(record.tags)) * 2
    content_tokens = _tokenize(record.content)
    kind_tokens = _tokenize(f"{record.scope} {record.kind}")
    return title_tokens + tag_tokens + content_tokens + kind_tokens


def _tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_PATTERN.findall(text)
        if token.lower() not in STOP_WORDS
    ]


def _vector_norm(vector: dict[str, float]) -> float:
    return math.sqrt(sum(value * value for value in vector.values()))


def _recency_score(created_at: str) -> float:
    try:
        created = datetime.fromisoformat(created_at)
    except ValueError:
        return 0.0
    now = datetime.now(UTC)
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    age_days = max(0.0, (now - created).total_seconds() / 86_400)
    return 1 / (1 + age_days / 30)


def _memory_id(scope: MemoryScope, kind: MemoryKind, title: str) -> str:
    return f"{scope}-{kind}-{_slugify(title)}-{uuid4().hex[:8]}"


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text.lower()).strip("-")
    return slug[:48] or "memory"


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + f"\n[truncated after {max_chars} chars]"
