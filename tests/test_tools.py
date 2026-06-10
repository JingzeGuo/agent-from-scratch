from pathlib import Path

from agent.setup import create_registry


def test_read_file_reads_file_inside_workspace(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("workspace content", encoding="utf-8")
    registry = create_registry(tmp_path)

    output, is_error = registry.execute("read_file", {"path": "notes.txt"})

    assert output == "workspace content"
    assert is_error is False


def test_read_file_rejects_file_outside_workspace(tmp_path: Path) -> None:
    workspace_root = tmp_path / "project"
    workspace_root.mkdir()
    outside_file = tmp_path / "secret.txt"
    outside_file.write_text("secret", encoding="utf-8")
    registry = create_registry(workspace_root)

    output, is_error = registry.execute(
        "read_file",
        {"path": "../secret.txt"},
    )

    assert "Path is outside the workspace" in output
    assert is_error is True
