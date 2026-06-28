import re
from datetime import UTC, datetime
from pathlib import Path

from .schemas import PendingAction, SessionEvent, SessionSnapshot
from .security import redact_text

SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


class SessionStore:
    """Persist and load session snapshots as JSON files."""

    def __init__(self, sessions_dir: Path) -> None:
        self.sessions_dir = sessions_dir

    def save(self, snapshot: SessionSnapshot) -> Path:
        session_id = self._validate_session_id(snapshot.session_id)
        if snapshot.session_name is not None:
            self._validate_session_name(snapshot.session_name)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        path = self._snapshot_path(session_id)
        temporary_path = path.with_suffix(".json.tmp")
        temporary_path.write_text(
            snapshot.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
        return path

    def load(self, session_id: str) -> SessionSnapshot:
        safe_session_id = self._validate_session_id(session_id)
        path = self._snapshot_path(safe_session_id)
        if not path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")
        return SessionSnapshot.model_validate_json(path.read_text(encoding="utf-8"))

    def find(self, id_or_name: str) -> SessionSnapshot:
        lookup = self._validate_session_name(id_or_name)
        direct_path = self._snapshot_path(lookup)
        if direct_path.exists():
            return self.load(lookup)

        matches = [
            snapshot
            for snapshot in self.list_snapshots()
            if snapshot.session_name == lookup
        ]
        if not matches:
            raise FileNotFoundError(f"Session not found: {id_or_name}")
        if len(matches) > 1:
            raise ValueError(f"Session name is ambiguous: {id_or_name}")
        return matches[0]

    def list_snapshots(self) -> list[SessionSnapshot]:
        if not self.sessions_dir.exists():
            return []

        snapshots: list[SessionSnapshot] = []
        for path in self.sessions_dir.glob("*.json"):
            session_id = path.stem
            if SESSION_ID_PATTERN.fullmatch(session_id) is not None:
                snapshots.append(self.load(session_id))
        return sorted(snapshots, key=lambda snapshot: snapshot.session_id)

    def write_pending_action(self, pending_action: PendingAction) -> Path:
        session_id = self._validate_session_id(pending_action.session_id)
        pending_dir = self._pending_dir()
        pending_dir.mkdir(parents=True, exist_ok=True)

        path = self._pending_action_path(session_id)
        temporary_path = path.with_suffix(".json.tmp")
        temporary_path.write_text(
            pending_action.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
        return path

    def read_pending_action(self, session_id: str) -> PendingAction | None:
        safe_session_id = self._validate_session_id(session_id)
        path = self._pending_action_path(safe_session_id)
        if not path.exists():
            return None
        return PendingAction.model_validate_json(path.read_text(encoding="utf-8"))

    def clear_pending_action(self, session_id: str) -> None:
        safe_session_id = self._validate_session_id(session_id)
        self._pending_action_path(safe_session_id).unlink(missing_ok=True)

    def append_event(self, event: SessionEvent) -> Path:
        session_id = self._validate_session_id(event.session_id)
        events_dir = self._events_dir()
        events_dir.mkdir(parents=True, exist_ok=True)

        path = self._event_log_path(session_id)
        safe_event = self._redact_event(event)
        with path.open("a", encoding="utf-8") as file:
            file.write(safe_event.model_dump_json() + "\n")
        return path

    def read_events(self, session_id: str) -> list[SessionEvent]:
        safe_session_id = self._validate_session_id(session_id)
        path = self._event_log_path(safe_session_id)
        if not path.exists():
            return []
        return [
            SessionEvent.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def _pending_dir(self) -> Path:
        return self.sessions_dir / "pending"

    def _events_dir(self) -> Path:
        return self.sessions_dir / "events"

    def _snapshot_path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.json"

    def _pending_action_path(self, session_id: str) -> Path:
        return self._pending_dir() / f"{session_id}.json"

    def _event_log_path(self, session_id: str) -> Path:
        return self._events_dir() / f"{session_id}.jsonl"

    def _validate_session_id(self, session_id: str) -> str:
        if SESSION_ID_PATTERN.fullmatch(session_id) is None:
            raise ValueError(
                "Session id must start with a letter or number and contain only "
                "letters, numbers, dots, underscores, and hyphens."
            )
        return session_id

    def _validate_session_name(self, session_name: str) -> str:
        if SESSION_ID_PATTERN.fullmatch(session_name) is None:
            raise ValueError(
                "Session name must start with a letter or number and contain only "
                "letters, numbers, dots, underscores, and hyphens."
            )
        return session_name

    def _redact_event(self, event: SessionEvent) -> SessionEvent:
        updates: dict[str, object] = {}
        for field_name in ("text_preview", "output_preview", "message", "objective"):
            value = getattr(event, field_name)
            if isinstance(value, str):
                updates[field_name] = redact_text(value)
        if event.native_metadata is not None:
            updates["native_metadata"] = self._redact_value(event.native_metadata)
        if not updates:
            return event
        return event.model_copy(update=updates)

    def _redact_value(self, value: object) -> object:
        if isinstance(value, str):
            return redact_text(value)
        if isinstance(value, list):
            return [self._redact_value(item) for item in value]
        if isinstance(value, dict):
            return {key: self._redact_value(item) for key, item in value.items()}
        return value
