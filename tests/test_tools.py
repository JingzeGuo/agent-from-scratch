from pathlib import Path

from agent.setup import create_registry


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
