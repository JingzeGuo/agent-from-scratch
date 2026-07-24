import asyncio
import shlex
import sys
from pathlib import Path

from agent.setup import create_read_only_registry, create_registry


def test_glob_files_matches_files_by_pattern(tmp_path: Path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_agent.py").write_text("", encoding="utf-8")
    (tests_dir / "test_tools.py").write_text("", encoding="utf-8")
    (tests_dir / "helper.py").write_text("", encoding="utf-8")
    registry = create_registry(tmp_path)

    output, is_error = registry.execute(
        "glob_files",
        {"pattern": "tests/test_*.py"},
    )

    assert output == "tests/test_agent.py\ntests/test_tools.py"
    assert is_error is False


def test_glob_files_returns_relative_paths(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "tools.py").write_text("", encoding="utf-8")
    registry = create_registry(tmp_path)

    output, is_error = registry.execute(
        "glob_files",
        {"pattern": "**/*.py"},
    )

    assert output == "agent/tools.py"
    assert str(tmp_path) not in output
    assert is_error is False


def test_glob_files_skips_noisy_directories(tmp_path: Path) -> None:
    source_dir = tmp_path / "agent"
    source_dir.mkdir()
    (source_dir / "tools.py").write_text("", encoding="utf-8")
    venv_dir = tmp_path / ".venv"
    venv_dir.mkdir()
    (venv_dir / "ignored.py").write_text("", encoding="utf-8")
    registry = create_registry(tmp_path)

    output, is_error = registry.execute(
        "glob_files",
        {"pattern": "**/*.py"},
    )

    assert output == "agent/tools.py"
    assert ".venv" not in output
    assert is_error is False


def test_glob_files_allows_explicit_noisy_directory_pattern(tmp_path: Path) -> None:
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").write_text("", encoding="utf-8")
    registry = create_registry(tmp_path)

    output, is_error = registry.execute(
        "glob_files",
        {"pattern": ".venv/bin/python*"},
    )

    assert output == ".venv/bin/python"
    assert is_error is False


def test_glob_files_allows_explicit_workspace_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external_python = tmp_path / "external-python"
    external_python.write_text("", encoding="utf-8")
    venv_bin = workspace / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(external_python)
    registry = create_registry(workspace)

    output, is_error = registry.execute(
        "glob_files",
        {"pattern": ".venv/bin/python*"},
    )

    assert output == ".venv/bin/python"
    assert is_error is False


def test_glob_files_truncates_results(tmp_path: Path) -> None:
    for index in range(3):
        (tmp_path / f"file_{index}.py").write_text("", encoding="utf-8")
    registry = create_registry(tmp_path)

    output, is_error = registry.execute(
        "glob_files",
        {"pattern": "*.py", "max_results": 2},
    )

    assert output == "file_0.py\nfile_1.py\n[truncated after 2 files]"
    assert is_error is False


def test_glob_files_rejects_parent_path_escape(tmp_path: Path) -> None:
    workspace_root = tmp_path / "project"
    workspace_root.mkdir()
    (tmp_path / "secret.py").write_text("", encoding="utf-8")
    registry = create_registry(workspace_root)

    output, is_error = registry.execute(
        "glob_files",
        {"pattern": "../*.py"},
    )

    assert "Glob pattern must be workspace-relative" in output
    assert is_error is True


def test_search_text_finds_matches_with_path_and_line_number(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "workspace.py").write_text(
        "from pathlib import Path\n\n"
        "def resolve_workspace_path(path: str) -> Path:\n"
        "    return Path(path)\n",
        encoding="utf-8",
    )
    registry = create_registry(tmp_path)

    output, is_error = registry.execute(
        "search_text",
        {"pattern": r"def resolve_workspace_path", "file_pattern": "**/*.py"},
    )

    assert output == "agent/workspace.py:3: def resolve_workspace_path(path: str) -> Path:"
    assert is_error is False


def test_search_text_limits_files_by_pattern(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (agent_dir / "tools.py").write_text("def target() -> None:\n    pass\n")
    (tests_dir / "test_tools.py").write_text("def target() -> None:\n    pass\n")
    registry = create_registry(tmp_path)

    output, is_error = registry.execute(
        "search_text",
        {
            "pattern": r"def target",
            "file_pattern": "tests/*.py",
        },
    )

    assert output == "tests/test_tools.py:1: def target() -> None:"
    assert is_error is False


def test_search_text_skips_noisy_directories(tmp_path: Path) -> None:
    source_dir = tmp_path / "agent"
    source_dir.mkdir()
    (source_dir / "tools.py").write_text("needle\n", encoding="utf-8")
    venv_dir = tmp_path / ".venv"
    venv_dir.mkdir()
    (venv_dir / "ignored.py").write_text("needle\n", encoding="utf-8")
    registry = create_registry(tmp_path)

    output, is_error = registry.execute(
        "search_text",
        {"pattern": "needle", "file_pattern": "**/*.py"},
    )

    assert output == "agent/tools.py:1: needle"
    assert ".venv" not in output
    assert is_error is False


def test_search_text_truncates_matches(tmp_path: Path) -> None:
    for index in range(3):
        (tmp_path / f"file_{index}.py").write_text("needle\n", encoding="utf-8")
    registry = create_registry(tmp_path)

    output, is_error = registry.execute(
        "search_text",
        {"pattern": "needle", "file_pattern": "*.py", "max_matches": 2},
    )

    assert output == (
        "file_0.py:1: needle\n"
        "file_1.py:1: needle\n"
        "[truncated after 2 matches]"
    )
    assert is_error is False


def test_search_text_reports_invalid_regex(tmp_path: Path) -> None:
    registry = create_registry(tmp_path)

    output, is_error = registry.execute(
        "search_text",
        {"pattern": "["},
    )

    assert "Invalid regular expression" in output
    assert is_error is True


def test_search_text_rejects_parent_path_escape(tmp_path: Path) -> None:
    workspace_root = tmp_path / "project"
    workspace_root.mkdir()
    registry = create_registry(workspace_root)

    output, is_error = registry.execute(
        "search_text",
        {"pattern": "secret", "file_pattern": "../*.py"},
    )

    assert "File pattern must be workspace-relative" in output
    assert is_error is True


def test_read_file_reads_file_inside_workspace(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("workspace content\nsecond line\n", encoding="utf-8")
    registry = create_registry(tmp_path)

    output, is_error = registry.execute("read_file", {"path": "notes.txt"})

    assert output == "1: workspace content\n2: second line"
    assert is_error is False


def test_read_file_reads_line_range_with_line_numbers(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
    registry = create_registry(tmp_path)

    output, is_error = registry.execute(
        "read_file",
        {"path": "notes.txt", "offset": 2, "limit": 2},
    )

    assert output == "2: two\n3: three"
    assert is_error is False


def test_read_file_reports_offset_past_end(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("one\ntwo\n", encoding="utf-8")
    registry = create_registry(tmp_path)

    output, is_error = registry.execute(
        "read_file",
        {"path": "notes.txt", "offset": 5, "limit": 2},
    )

    assert output == "[No lines found from line 5. File has 2 lines.]"
    assert is_error is False


def test_read_file_rejects_invalid_line_range_arguments(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("content", encoding="utf-8")
    registry = create_registry(tmp_path)

    output, is_error = registry.execute(
        "read_file",
        {"path": "notes.txt", "offset": 0, "limit": 2},
    )

    assert "field 'offset': Input should be greater than or equal to 1" in output
    assert is_error is True


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


def test_read_file_tracks_read_file(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("content\n", encoding="utf-8")
    registry = create_registry(tmp_path)

    output, is_error = registry.execute("read_file", {"path": "notes.txt"})

    assert output == "1: content"
    assert target.resolve() in registry.read_files
    assert is_error is False


def test_edit_file_replaces_unique_match_and_returns_diff(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("one\ntwo\nthree\n", encoding="utf-8")
    registry = create_registry(tmp_path)
    registry.execute("read_file", {"path": "notes.txt"})

    output, is_error = registry.execute(
        "edit_file",
        {
            "path": "notes.txt",
            "old_text": "two",
            "new_text": "TWO",
        },
    )

    assert target.read_text(encoding="utf-8") == "one\nTWO\nthree\n"
    assert output == (
        "--- a/notes.txt\n"
        "+++ b/notes.txt\n"
        "@@ -1,3 +1,3 @@\n"
        " one\n"
        "-two\n"
        "+TWO\n"
        " three"
    )
    assert is_error is False


def test_edit_file_rejects_missing_match(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("one\ntwo\nthree\n", encoding="utf-8")
    registry = create_registry(tmp_path)
    registry.execute("read_file", {"path": "notes.txt"})

    output, is_error = registry.execute(
        "edit_file",
        {
            "path": "notes.txt",
            "old_text": "four",
            "new_text": "FOUR",
        },
    )

    assert target.read_text(encoding="utf-8") == "one\ntwo\nthree\n"
    assert "Exact text was not found" in output
    assert is_error is True


def test_edit_file_rejects_duplicate_match(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("repeat\nmiddle\nrepeat\n", encoding="utf-8")
    registry = create_registry(tmp_path)
    registry.execute("read_file", {"path": "notes.txt"})

    output, is_error = registry.execute(
        "edit_file",
        {
            "path": "notes.txt",
            "old_text": "repeat",
            "new_text": "changed",
        },
    )

    assert target.read_text(encoding="utf-8") == "repeat\nmiddle\nrepeat\n"
    assert "Exact text matched 2 times" in output
    assert is_error is True


def test_edit_file_requires_read_before_edit(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("one\ntwo\nthree\n", encoding="utf-8")
    registry = create_registry(tmp_path)

    output, is_error = registry.execute(
        "edit_file",
        {
            "path": "notes.txt",
            "old_text": "two",
            "new_text": "TWO",
        },
    )

    assert target.read_text(encoding="utf-8") == "one\ntwo\nthree\n"
    assert "File must be read before editing" in output
    assert is_error is True


def test_edit_file_tracks_changed_file(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("one\ntwo\nthree\n", encoding="utf-8")
    registry = create_registry(tmp_path)
    registry.execute("read_file", {"path": "notes.txt"})

    output, is_error = registry.execute(
        "edit_file",
        {
            "path": "notes.txt",
            "old_text": "two",
            "new_text": "TWO",
        },
    )

    assert target.resolve() in registry.changed_files
    assert "--- a/notes.txt" in output
    assert is_error is False


def test_edit_file_rejects_file_outside_workspace(tmp_path: Path) -> None:
    workspace_root = tmp_path / "project"
    workspace_root.mkdir()
    outside_file = tmp_path / "secret.txt"
    outside_file.write_text("secret", encoding="utf-8")
    registry = create_registry(workspace_root)

    output, is_error = registry.execute(
        "edit_file",
        {
            "path": "../secret.txt",
            "old_text": "secret",
            "new_text": "changed",
        },
    )

    assert outside_file.read_text(encoding="utf-8") == "secret"
    assert "Path is outside the workspace" in output
    assert is_error is True


def test_edit_file_invalid_utf8_snapshot_returns_tool_error(tmp_path: Path) -> None:
    target = tmp_path / "binary.txt"
    original_bytes = b"\xff\xfe\xfa"
    target.write_bytes(original_bytes)
    registry = create_registry(tmp_path)
    registry.read_files.add(target.resolve())

    output, is_error = registry.execute(
        "edit_file",
        {
            "path": "binary.txt",
            "old_text": "missing",
            "new_text": "replacement",
        },
    )

    assert target.read_bytes() == original_bytes
    assert target.resolve() not in registry.changed_files
    assert target.resolve() not in registry.original_file_contents
    assert "Tool 'edit_file' raised UnicodeDecodeError" in output
    assert is_error is True


def test_write_file_creates_new_file_and_returns_diff(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    registry = create_registry(tmp_path)

    output, is_error = registry.execute(
        "write_file",
        {
            "path": "notes.txt",
            "content": "one\ntwo\n",
        },
    )

    assert target.read_text(encoding="utf-8") == "one\ntwo\n"
    assert output == (
        "--- a/notes.txt\n"
        "+++ b/notes.txt\n"
        "@@ -0,0 +1,2 @@\n"
        "+one\n"
        "+two"
    )
    assert is_error is False


def test_write_file_creates_parent_directories(tmp_path: Path) -> None:
    target = tmp_path / "docs" / "notes.txt"
    registry = create_registry(tmp_path)

    output, is_error = registry.execute(
        "write_file",
        {
            "path": "docs/notes.txt",
            "content": "content\n",
        },
    )

    assert target.read_text(encoding="utf-8") == "content\n"
    assert "--- a/docs/notes.txt" in output
    assert is_error is False


def test_write_file_rejects_existing_file_without_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("original\n", encoding="utf-8")
    registry = create_registry(tmp_path)

    output, is_error = registry.execute(
        "write_file",
        {
            "path": "notes.txt",
            "content": "replacement\n",
        },
    )

    assert target.read_text(encoding="utf-8") == "original\n"
    assert "File already exists" in output
    assert is_error is True


def test_write_file_requires_read_before_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("original\n", encoding="utf-8")
    registry = create_registry(tmp_path)

    output, is_error = registry.execute(
        "write_file",
        {
            "path": "notes.txt",
            "content": "replacement\n",
            "overwrite": True,
        },
    )

    assert target.read_text(encoding="utf-8") == "original\n"
    assert "File must be read before overwriting" in output
    assert is_error is True


def test_write_file_overwrites_after_read(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("original\n", encoding="utf-8")
    registry = create_registry(tmp_path)
    registry.execute("read_file", {"path": "notes.txt"})

    output, is_error = registry.execute(
        "write_file",
        {
            "path": "notes.txt",
            "content": "replacement\n",
            "overwrite": True,
        },
    )

    assert target.read_text(encoding="utf-8") == "replacement\n"
    assert output == (
        "--- a/notes.txt\n"
        "+++ b/notes.txt\n"
        "@@ -1 +1 @@\n"
        "-original\n"
        "+replacement"
    )
    assert is_error is False


def test_write_file_overwrite_invalid_utf8_snapshot_returns_tool_error(
    tmp_path: Path,
) -> None:
    target = tmp_path / "binary.txt"
    original_bytes = b"\xff\xfe\xfa"
    target.write_bytes(original_bytes)
    registry = create_registry(tmp_path)
    registry.read_files.add(target.resolve())

    output, is_error = registry.execute(
        "write_file",
        {
            "path": "binary.txt",
            "content": "replacement\n",
            "overwrite": True,
        },
    )

    assert target.read_bytes() == original_bytes
    assert target.resolve() not in registry.changed_files
    assert target.resolve() not in registry.original_file_contents
    assert "Tool 'write_file' raised UnicodeDecodeError" in output
    assert is_error is True


def test_execute_async_invalid_utf8_snapshot_returns_tool_error(
    tmp_path: Path,
) -> None:
    target = tmp_path / "binary.txt"
    original_bytes = b"\xff\xfe\xfa"
    target.write_bytes(original_bytes)
    registry = create_registry(tmp_path)
    registry.read_files.add(target.resolve())

    output, is_error = asyncio.run(
        registry.execute_async(
            "edit_file",
            {
                "path": "binary.txt",
                "old_text": "missing",
                "new_text": "replacement",
            },
        )
    )

    assert target.read_bytes() == original_bytes
    assert target.resolve() not in registry.changed_files
    assert target.resolve() not in registry.original_file_contents
    assert "Tool 'edit_file' raised UnicodeDecodeError" in output
    assert is_error is True


def test_write_file_tracks_changed_file(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    registry = create_registry(tmp_path)

    output, is_error = registry.execute(
        "write_file",
        {
            "path": "notes.txt",
            "content": "content\n",
        },
    )

    assert target.resolve() in registry.changed_files
    assert "+++ b/notes.txt" in output
    assert is_error is False


def test_write_file_rejects_file_outside_workspace(tmp_path: Path) -> None:
    workspace_root = tmp_path / "project"
    workspace_root.mkdir()
    outside_file = tmp_path / "secret.txt"
    registry = create_registry(workspace_root)

    output, is_error = registry.execute(
        "write_file",
        {
            "path": "../secret.txt",
            "content": "secret\n",
        },
    )

    assert not outside_file.exists()
    assert "Path is outside the workspace" in output
    assert is_error is True


def test_sub_agent_is_registered_with_read_only_profile(tmp_path: Path) -> None:
    registry = create_registry(tmp_path)

    definitions = {tool.name: tool for tool in registry.to_tool_definitions()}

    assert "sub_agent" in definitions
    assert definitions["calculator"].kind == "pure"
    assert definitions["read_file"].kind == "read_only"
    assert definitions["edit_file"].kind == "write"
    assert definitions["run_command"].kind == "command"
    assert definitions["sub_agent"].kind == "delegated"
    assert definitions["fetch_url"].kind == "network"
    assert definitions["sub_agent"].input_schema["properties"]["profile"] == {
        "const": "read_only_explorer",
        "default": "read_only_explorer",
        "description": "The capability profile for the child agent.",
        "title": "Profile",
        "type": "string",
    }


def test_sub_agent_requires_parent_agent_when_executed_directly(
    tmp_path: Path,
) -> None:
    registry = create_registry(tmp_path)

    output, is_error = registry.execute(
        "sub_agent",
        {
            "task": "Explore session resume behavior.",
            "profile": "read_only_explorer",
            "max_steps": 3,
        },
    )

    assert "Tool 'sub_agent' raised ValueError" in output
    assert "not initialized with a parent agent" in output
    assert is_error is True


def test_sub_agent_rejects_unsupported_profile(tmp_path: Path) -> None:
    registry = create_registry(tmp_path)

    output, is_error = registry.execute(
        "sub_agent",
        {
            "task": "Explore the repository.",
            "profile": "coding_worker",
        },
    )

    assert "Validation error for tool 'sub_agent'" in output
    assert "field 'profile'" in output
    assert is_error is True


def test_sub_agent_rejects_excessive_step_budget(tmp_path: Path) -> None:
    registry = create_registry(tmp_path)

    output, is_error = registry.execute(
        "sub_agent",
        {
            "task": "Explore the repository.",
            "max_steps": 9,
        },
    )

    assert "field 'max_steps': Input should be less than or equal to 8" in output
    assert is_error is True


def test_read_only_registry_excludes_mutating_and_recursive_tools(
    tmp_path: Path,
) -> None:
    registry = create_read_only_registry(tmp_path)

    assert set(registry.tools) == {
        "calculator",
        "read_file",
        "glob_files",
        "search_text",
        "get_diff",
    }
    assert "edit_file" not in registry.tools
    assert "write_file" not in registry.tools
    assert "run_command" not in registry.tools
    assert "sub_agent" not in registry.tools


def test_get_diff_returns_no_changes_message(tmp_path: Path) -> None:
    registry = create_registry(tmp_path)

    output, is_error = registry.execute("get_diff", {})

    assert output == "[No files changed]"
    assert is_error is False


def test_get_diff_returns_changed_file_diff(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("one\ntwo\nthree\n", encoding="utf-8")
    registry = create_registry(tmp_path)
    registry.execute("read_file", {"path": "notes.txt"})
    registry.execute(
        "edit_file",
        {
            "path": "notes.txt",
            "old_text": "two",
            "new_text": "TWO",
        },
    )

    output, is_error = registry.execute("get_diff", {})

    assert output == (
        "--- a/notes.txt\n"
        "+++ b/notes.txt\n"
        "@@ -1,3 +1,3 @@\n"
        " one\n"
        "-two\n"
        "+TWO\n"
        " three"
    )
    assert is_error is False


def test_get_diff_filters_by_path(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("one\n", encoding="utf-8")
    second.write_text("two\n", encoding="utf-8")
    registry = create_registry(tmp_path)
    registry.execute("read_file", {"path": "first.txt"})
    registry.execute("read_file", {"path": "second.txt"})
    registry.execute(
        "edit_file",
        {"path": "first.txt", "old_text": "one", "new_text": "ONE"},
    )
    registry.execute(
        "edit_file",
        {"path": "second.txt", "old_text": "two", "new_text": "TWO"},
    )

    output, is_error = registry.execute("get_diff", {"path": "second.txt"})

    assert "--- a/second.txt" in output
    assert "+TWO" in output
    assert "first.txt" not in output
    assert is_error is False


def test_get_diff_tracks_new_file_from_write_file(tmp_path: Path) -> None:
    registry = create_registry(tmp_path)
    registry.execute(
        "write_file",
        {
            "path": "notes.txt",
            "content": "one\ntwo\n",
        },
    )

    output, is_error = registry.execute("get_diff", {})

    assert output == (
        "--- a/notes.txt\n"
        "+++ b/notes.txt\n"
        "@@ -0,0 +1,2 @@\n"
        "+one\n"
        "+two"
    )
    assert is_error is False


def test_get_diff_preserves_original_content_across_multiple_edits(
    tmp_path: Path,
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("one\ntwo\nthree\n", encoding="utf-8")
    registry = create_registry(tmp_path)
    registry.execute("read_file", {"path": "notes.txt"})
    registry.execute(
        "edit_file",
        {
            "path": "notes.txt",
            "old_text": "two",
            "new_text": "TWO",
        },
    )
    registry.execute(
        "edit_file",
        {
            "path": "notes.txt",
            "old_text": "three",
            "new_text": "THREE",
        },
    )

    output, is_error = registry.execute("get_diff", {})

    assert output == (
        "--- a/notes.txt\n"
        "+++ b/notes.txt\n"
        "@@ -1,3 +1,3 @@\n"
        " one\n"
        "-two\n"
        "-three\n"
        "+TWO\n"
        "+THREE"
    )
    assert is_error is False


def test_get_diff_rejects_unchanged_path(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("content\n", encoding="utf-8")
    registry = create_registry(tmp_path)

    output, is_error = registry.execute("get_diff", {"path": "notes.txt"})

    assert "File has not changed in this session" in output
    assert is_error is True


def test_run_command_returns_success_result(tmp_path: Path) -> None:
    target = tmp_path / "module.py"
    target.write_text("print('ok')\n", encoding="utf-8")
    registry = create_registry(tmp_path)

    output, is_error = registry.execute(
        "run_command",
        {"command": f"{shlex.quote(sys.executable)} -m py_compile module.py"},
    )

    assert "exit_code: 0" in output
    assert "timed_out: false" in output
    assert "stdout:\n[empty]" in output
    assert "stderr:\n[empty]" in output
    assert is_error is False


def test_run_command_returns_failure_exit_code(tmp_path: Path) -> None:
    target = tmp_path / "module.py"
    target.write_text("def broken(:\n", encoding="utf-8")
    registry = create_registry(tmp_path)

    output, is_error = registry.execute(
        "run_command",
        {"command": f"{shlex.quote(sys.executable)} -m py_compile module.py"},
    )

    assert "exit_code: 1" in output
    assert "timed_out: false" in output
    assert "SyntaxError" in output
    assert is_error is False


def test_run_command_times_out(tmp_path: Path) -> None:
    test_file = tmp_path / "test_slow.py"
    test_file.write_text(
        "import time\n\n"
        "def test_slow():\n"
        "    time.sleep(5)\n",
        encoding="utf-8",
    )
    registry = create_registry(tmp_path)

    output, is_error = registry.execute(
        "run_command",
        {
            "command": f"{shlex.quote(sys.executable)} -m pytest test_slow.py",
            "timeout_seconds": 0.1,
        },
    )

    assert "exit_code: null" in output
    assert "timed_out: true" in output
    assert is_error is False


def test_run_command_truncates_long_output(tmp_path: Path) -> None:
    test_file = tmp_path / "test_long_output.py"
    test_file.write_text(
        "def test_long_output():\n"
        "    assert False, 'x' * 250\n",
        encoding="utf-8",
    )
    registry = create_registry(tmp_path)

    output, is_error = registry.execute(
        "run_command",
        {
            "command": f"{shlex.quote(sys.executable)} -m pytest test_long_output.py",
            "max_output_chars": 200,
        },
    )

    assert "[... truncated" in output
    assert "exit_code: 1" in output
    assert is_error is False


def test_run_command_rejects_dangerous_command(tmp_path: Path) -> None:
    registry = create_registry(tmp_path)

    output, is_error = registry.execute(
        "run_command",
        {"command": "rm -rf ."},
    )

    assert "Blocked dangerous command: rm" in output
    assert is_error is True


def test_run_command_requires_approval_for_arbitrary_python(tmp_path: Path) -> None:
    registry = create_registry(tmp_path)

    output, is_error = registry.execute(
        "run_command",
        {"command": f"{shlex.quote(sys.executable)} -c \"print('needs approval')\""},
    )

    assert "requires approval" in output
    assert "broad side effects" in output
    assert is_error is True


def test_run_command_allows_python_version_probe(tmp_path: Path) -> None:
    registry = create_registry(tmp_path)

    output, is_error = registry.execute(
        "run_command",
        {"command": f"{shlex.quote(sys.executable)} --version"},
    )

    assert "exit_code: 0" in output
    assert "Python" in output
    assert is_error is False


def test_run_command_resolves_workspace_venv_from_subdirectory(
    tmp_path: Path,
) -> None:
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    fake_python = venv_bin / "python"
    fake_python.write_text("#!/bin/sh\nprintf 'fake python\\n'\n", encoding="utf-8")
    fake_python.chmod(0o755)
    (tmp_path / "package").mkdir()
    registry = create_registry(tmp_path)

    output, is_error = registry.execute(
        "run_command",
        {"command": ".venv/bin/python --version", "cwd": "package"},
    )

    assert "exit_code: 0" in output
    assert "stdout:\nfake python" in output
    assert is_error is False


def test_run_command_executes_after_approval(tmp_path: Path) -> None:
    registry = create_registry(tmp_path)

    output, is_error = registry.execute(
        "run_command",
        {"command": f"{shlex.quote(sys.executable)} -c \"print('approved')\""},
        approval_granted=True,
    )

    assert "exit_code: 0" in output
    assert "stdout:\napproved" in output
    assert is_error is False


def test_run_command_uses_workspace_relative_cwd(tmp_path: Path) -> None:
    subdir = tmp_path / "package"
    subdir.mkdir()
    (subdir / "module.py").write_text("value = 1\n", encoding="utf-8")
    registry = create_registry(tmp_path)

    output, is_error = registry.execute(
        "run_command",
        {
            "command": f"{shlex.quote(sys.executable)} -m py_compile module.py",
            "cwd": "package",
        },
    )

    assert "exit_code: 0" in output
    assert is_error is False


def test_run_command_rejects_cwd_outside_workspace(tmp_path: Path) -> None:
    workspace_root = tmp_path / "project"
    workspace_root.mkdir()
    (tmp_path / "outside").mkdir()
    (workspace_root / "module.py").write_text("value = 1\n", encoding="utf-8")
    registry = create_registry(workspace_root)

    output, is_error = registry.execute(
        "run_command",
        {
            "command": f"{shlex.quote(sys.executable)} -m py_compile module.py",
            "cwd": "../outside",
        },
    )

    assert "Path is outside the workspace" in output
    assert is_error is True
