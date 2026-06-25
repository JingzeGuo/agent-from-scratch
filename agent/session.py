import re
from pathlib import Path

from .schemas import SessionSnapshot

SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


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

    def _snapshot_path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.json"

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
