import shlex
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..schemas import EditFileInput, ReadFileInput, ToolDefinition, WriteFileInput
from ..workspace import resolve_workspace_path
from .tool import Tool
from .tools import _build_unified_diff

ISOLATED_PATH_FIELDS = {
    "read_file": "path",
    "edit_file": "path",
    "write_file": "path",
    "get_diff": "path",
    "glob_files": "pattern",
    "search_text": "file_pattern",
}


class ToolRegistry:
    def __init__(
        self,
        workspace_root: Path,
        *,
        blocked_path_parts: frozenset[str] = frozenset(),
        blocked_command_names: frozenset[str] = frozenset(),
    ) -> None:
        self.tools: dict[str, Tool] = {}
        self.workspace_root = workspace_root
        self.blocked_path_parts = blocked_path_parts
        self.blocked_command_names = blocked_command_names
        self.read_files: set[Path] = set()
        self.changed_files: set[Path] = set()
        self.original_file_contents: dict[Path, str | None] = {}

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def to_tool_definitions(self) -> list[ToolDefinition]:
        return [tool.to_definition() for tool in self.tools.values()]

    def relative_path(self, path: Path) -> str:
        """Return a tracked absolute path as a workspace-relative POSIX path."""
        return path.resolve().relative_to(self.workspace_root.resolve()).as_posix()

    async def execute(
        self,
        name: str,
        raw_input: dict[str, Any],
        *,
        approval_granted: bool = False,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> tuple[str, bool]:
        tool = self.tools.get(name)
        if tool is None:
            return f"Unknown tool: '{name}'. Available: {list(self.tools)}", True

        parsed = tool.parse(raw_input)
        if isinstance(parsed, str):
            return parsed, True

        try:
            error = self._validate_execution_allowed(name, parsed)
            if error is not None:
                return error, True
            original_snapshot = self._snapshot_before_mutation(parsed)
        except Exception as e:
            return f"Tool '{name}' raised {type(e).__name__}: {e}", True

        output, is_error = await tool.run(
            parsed,
            self._tool_extra_kwargs(name, approval_granted, extra_kwargs),
        )
        if not is_error:
            self._record_successful_file_action(parsed, original_snapshot)
        return output, is_error

    def _validate_execution_allowed(
        self,
        tool_name: str,
        parsed: BaseModel,
    ) -> str | None:
        isolation_error = self._validate_isolation_policy(tool_name, parsed)
        if isolation_error is not None:
            return isolation_error
        if isinstance(parsed, EditFileInput):
            path = resolve_workspace_path(self.workspace_root, parsed.path)
            if path not in self.read_files:
                return (
                    "Tool 'edit_file' raised ValueError: "
                    f"File must be read before editing: {parsed.path}"
                )
        if isinstance(parsed, WriteFileInput) and parsed.overwrite:
            path = resolve_workspace_path(self.workspace_root, parsed.path)
            if path.exists() and path not in self.read_files:
                return (
                    "Tool 'write_file' raised ValueError: "
                    f"File must be read before overwriting: {parsed.path}"
                )
        return None

    def _validate_isolation_policy(
        self,
        tool_name: str,
        parsed: BaseModel,
    ) -> str | None:
        path_field = ISOLATED_PATH_FIELDS.get(tool_name)
        if path_field is not None:
            raw_path = getattr(parsed, path_field, None)
            if isinstance(raw_path, str):
                blocked_part = self._blocked_path_part(raw_path)
                if blocked_part is not None:
                    return (
                        f"Tool '{tool_name}' blocked by workspace isolation: "
                        f"path component '{blocked_part}' is unavailable."
                    )

        if tool_name != "run_command":
            return None
        raw_command = getattr(parsed, "command", None)
        if not isinstance(raw_command, str):
            return None
        try:
            args = shlex.split(raw_command)
        except ValueError:
            return None
        if not args:
            return None
        command_name = Path(args[0]).name
        if command_name in self.blocked_command_names:
            return (
                "Tool 'run_command' blocked by workspace isolation: "
                f"command '{command_name}' is unavailable."
            )
        for argument in args[1:]:
            blocked_part = self._blocked_path_part(argument)
            if blocked_part is not None:
                return (
                    "Tool 'run_command' blocked by workspace isolation: "
                    f"path component '{blocked_part}' is unavailable."
                )
        return None

    def _blocked_path_part(self, raw_path: str) -> str | None:
        if not self.blocked_path_parts:
            return None
        raw_parts = set(Path(raw_path).parts)
        for blocked_part in sorted(self.blocked_path_parts):
            if blocked_part in raw_parts:
                return blocked_part
        if any(char in raw_path for char in "*?[]"):
            return None
        try:
            resolved = resolve_workspace_path(self.workspace_root, raw_path)
            relative_parts = set(resolved.relative_to(self.workspace_root.resolve()).parts)
        except (OSError, ValueError):
            return None
        for blocked_part in sorted(self.blocked_path_parts):
            if blocked_part in relative_parts:
                return blocked_part
        return None

    def _tool_extra_kwargs(
        self,
        name: str,
        approval_granted: bool,
        extra_kwargs: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        merged_kwargs = dict(extra_kwargs or {})
        if name == "run_command" and approval_granted:
            merged_kwargs["approval_granted"] = True
        if not merged_kwargs:
            return None
        return merged_kwargs

    def get_diff(self, path: str | None = None) -> str:
        """Return unified diffs for files changed during this registry session."""
        paths = self._changed_paths_for_diff(path)
        if not paths:
            return "[No files changed]"
        return "\n\n".join(self._diff_for_changed_path(changed_path) for changed_path in paths)

    def _snapshot_before_mutation(
        self,
        parsed: BaseModel,
    ) -> tuple[Path, str | None] | None:
        if not isinstance(parsed, EditFileInput | WriteFileInput):
            return None
        path = resolve_workspace_path(self.workspace_root, parsed.path)
        if path in self.original_file_contents:
            return None
        if not path.exists():
            return path, None
        if isinstance(parsed, WriteFileInput) and not parsed.overwrite:
            return None
        return path, path.read_text(encoding="utf-8")

    def _record_successful_file_action(
        self,
        parsed: BaseModel,
        original_snapshot: tuple[Path, str | None] | None,
    ) -> None:
        if isinstance(parsed, ReadFileInput):
            self.read_files.add(resolve_workspace_path(self.workspace_root, parsed.path))
        elif isinstance(parsed, EditFileInput | WriteFileInput):
            if original_snapshot is not None:
                original_path, original_content = original_snapshot
                self.original_file_contents[original_path] = original_content
            self.changed_files.add(
                resolve_workspace_path(self.workspace_root, parsed.path)
            )

    def _changed_paths_for_diff(self, path: str | None) -> list[Path]:
        if path is None:
            return sorted(self.changed_files)
        resolved = resolve_workspace_path(self.workspace_root, path)
        if resolved not in self.changed_files:
            raise ValueError(f"File has not changed in this session: {path}")
        return [resolved]

    def _diff_for_changed_path(self, path: Path) -> str:
        original = self.original_file_contents.get(path)
        before = "" if original is None else original
        after = ""
        if path.exists():
            after = path.read_text(encoding="utf-8")
        return _build_unified_diff(
            path=path,
            before=before,
            after=after,
            workspace_root=self.workspace_root,
        )
