from pathlib import Path
from typing import Any

from .schemas import ToolDefinition
from .security import classify_command
from .tool import Tool
from .tools import _build_unified_diff
from .workspace import resolve_workspace_path


class ToolRegistry:
    def __init__(self, workspace_root: Path | None = None) -> None:
        self.tools: dict[str, Tool] = {}
        self.workspace_root = workspace_root
        self.read_files: set[Path] = set()
        self.changed_files: set[Path] = set()
        self.original_file_contents: dict[Path, str | None] = {}

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def to_tool_definitions(self) -> list[ToolDefinition]:
        return [tool.to_definition() for tool in self.tools.values()]

    def execute(
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

        error = self._validate_execution_allowed(name, raw_input, approval_granted)
        if error is not None:
            return error, True

        original_snapshot = self._snapshot_before_mutation(name, raw_input)
        output, is_error = tool.execute(
            raw_input,
            self._tool_extra_kwargs(name, approval_granted, extra_kwargs),
        )
        if not is_error:
            self._record_successful_file_action(name, raw_input, original_snapshot)
        return output, is_error

    async def execute_async(
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

        error = self._validate_execution_allowed(name, raw_input, approval_granted)
        if error is not None:
            return error, True

        original_snapshot = self._snapshot_before_mutation(name, raw_input)
        output, is_error = await tool.execute_async(
            raw_input,
            self._tool_extra_kwargs(name, approval_granted, extra_kwargs),
        )
        if not is_error:
            self._record_successful_file_action(name, raw_input, original_snapshot)
        return output, is_error

    def _validate_execution_allowed(
        self,
        name: str,
        raw_input: dict[str, Any],
        approval_granted: bool,
    ) -> str | None:
        if name == "edit_file":
            error = self._validate_edit_allowed(raw_input)
            if error is not None:
                return error
        if name == "write_file":
            error = self._validate_write_allowed(raw_input)
            if error is not None:
                return error
        if name == "run_command":
            error = self._validate_command_allowed(raw_input, approval_granted)
            if error is not None:
                return error
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
        if self.workspace_root is None:
            return "[No files changed]"
        paths = self._changed_paths_for_diff(path)
        if not paths:
            return "[No files changed]"
        return "\n\n".join(self._diff_for_changed_path(changed_path) for changed_path in paths)

    def _validate_edit_allowed(self, raw_input: dict[str, Any]) -> str | None:
        if self.workspace_root is None:
            return None
        raw_path = raw_input.get("path")
        if not isinstance(raw_path, str):
            return None
        try:
            path = resolve_workspace_path(self.workspace_root, raw_path)
        except Exception as e:
            return f"Tool 'edit_file' raised {type(e).__name__}: {e}"
        if path not in self.read_files:
            return (
                "Tool 'edit_file' raised ValueError: "
                f"File must be read before editing: {raw_path}"
            )
        return None

    def _validate_write_allowed(self, raw_input: dict[str, Any]) -> str | None:
        if self.workspace_root is None:
            return None
        raw_path = raw_input.get("path")
        if not isinstance(raw_path, str):
            return None
        try:
            path = resolve_workspace_path(self.workspace_root, raw_path)
        except Exception as e:
            return f"Tool 'write_file' raised {type(e).__name__}: {e}"
        if (
            path.exists()
            and self._raw_bool_is_true(raw_input.get("overwrite", False))
            and path not in self.read_files
        ):
            return (
                "Tool 'write_file' raised ValueError: "
                f"File must be read before overwriting: {raw_path}"
            )
        return None

    def _raw_bool_is_true(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value == 1
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return False

    def _validate_command_allowed(
        self,
        raw_input: dict[str, Any],
        approval_granted: bool,
    ) -> str | None:
        raw_command = raw_input.get("command")
        if not isinstance(raw_command, str):
            return None
        try:
            policy = classify_command(raw_command)
        except ValueError as e:
            return f"Tool 'run_command' raised ValueError: {e}"
        if policy.decision == "allowed":
            return None
        if approval_granted and policy.decision == "requires_approval":
            return None
        if policy.decision == "blocked":
            return f"Tool 'run_command' raised ValueError: {policy.reason}"
        return (
            "Tool 'run_command' requires approval: "
            f"{policy.reason} Command: {raw_command}"
        )

    def _snapshot_before_mutation(
        self,
        name: str,
        raw_input: dict[str, Any],
    ) -> tuple[Path, str | None] | None:
        if self.workspace_root is None or name not in {"edit_file", "write_file"}:
            return None
        raw_path = raw_input.get("path")
        if not isinstance(raw_path, str):
            return None
        path = resolve_workspace_path(self.workspace_root, raw_path)
        if path in self.original_file_contents:
            return None
        if not path.exists():
            return path, None
        if name == "write_file" and not self._raw_bool_is_true(
            raw_input.get("overwrite", False)
        ):
            return None
        return path, path.read_text(encoding="utf-8")

    def _record_successful_file_action(
        self,
        name: str,
        raw_input: dict[str, Any],
        original_snapshot: tuple[Path, str | None] | None,
    ) -> None:
        if self.workspace_root is None or name not in {
            "read_file",
            "edit_file",
            "write_file",
        }:
            return
        raw_path = raw_input.get("path")
        if not isinstance(raw_path, str):
            return
        path = resolve_workspace_path(self.workspace_root, raw_path)
        if name == "read_file":
            self.read_files.add(path)
        if name in {"edit_file", "write_file"}:
            if original_snapshot is not None:
                original_path, original_content = original_snapshot
                self.original_file_contents[original_path] = original_content
            self.changed_files.add(path)

    def _changed_paths_for_diff(self, path: str | None) -> list[Path]:
        if self.workspace_root is None:
            return []
        if path is None:
            return sorted(self.changed_files)
        resolved = resolve_workspace_path(self.workspace_root, path)
        if resolved not in self.changed_files:
            raise ValueError(f"File has not changed in this session: {path}")
        return [resolved]

    def _diff_for_changed_path(self, path: Path) -> str:
        if self.workspace_root is None:
            return "[No files changed]"
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
