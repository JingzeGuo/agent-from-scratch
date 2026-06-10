from pathlib import Path

import pytest

from agent.workspace import resolve_workspace_path


def test_resolves_workspace_relative_path(tmp_path: Path) -> None:
    workspace_root = tmp_path / "project"
    workspace_root.mkdir()

    resolved = resolve_workspace_path(
        workspace_root,
        "./tests/../agent/tool.py",
    )

    assert resolved == workspace_root / "agent" / "tool.py"


def test_allows_absolute_path_inside_workspace(tmp_path: Path) -> None:
    workspace_root = tmp_path / "project"
    workspace_root.mkdir()
    target = workspace_root / "docs" / "notes.md"

    resolved = resolve_workspace_path(workspace_root, str(target))

    assert resolved == target


def test_rejects_parent_path_escape(tmp_path: Path) -> None:
    workspace_root = tmp_path / "project"
    workspace_root.mkdir()

    with pytest.raises(ValueError, match="outside the workspace"):
        resolve_workspace_path(workspace_root, "../secret.txt")


def test_rejects_symlink_escape(tmp_path: Path) -> None:
    workspace_root = tmp_path / "project"
    workspace_root.mkdir()
    outside_file = tmp_path / "secret.txt"
    outside_file.write_text("secret", encoding="utf-8")
    (workspace_root / "link").symlink_to(outside_file)

    with pytest.raises(ValueError, match="outside the workspace"):
        resolve_workspace_path(workspace_root, "link")
