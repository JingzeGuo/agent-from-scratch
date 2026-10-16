"""Workspace file tools: glob, search, read, edit, write, and unified diffs.

These are plain Python implementations; `Tool` adds schemas and validation.
"""

import difflib
import re
from collections.abc import Iterator
from pathlib import Path

from ..workspace import resolve_workspace_path

# Searching and whole-file rewrites skip large files so a 50MB log cannot
# blow up the context window; reads and edits are bounded by line ranges instead.
_MAX_FILE_BYTES = 100_000  # ~25k tokens worst case
_MAX_DIFF_CHARS = 20_000

_SKIPPED_DIRS = {".git", ".venv", "node_modules"}  # skipped at any depth
_SKIPPED_ROOT_DIRS = {"build", "dist"}  # build output at the workspace root only


def _skipped_dirs(path: Path) -> set[str]:
    """Return the noisy directories containing this workspace-relative file."""
    dirs = path.parts[:-1]
    skipped = {part for part in dirs if part in _SKIPPED_DIRS}
    if dirs and dirs[0] in _SKIPPED_ROOT_DIRS:
        skipped.add(dirs[0])
    return skipped


def _validate_workspace_pattern(pattern: str, *, kind: str) -> None:
    if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        raise ValueError(f"{kind} must be workspace-relative and cannot contain '..'")


def _iter_workspace_files(
    root: Path,
    pattern: str,
    *,
    include_explicit_skipped_dirs: bool = False,
    allow_workspace_symlinks: bool = False,
) -> Iterator[tuple[Path, Path]]:
    for candidate in sorted(root.glob(pattern)):
        if not candidate.is_file():
            continue

        resolved = candidate.resolve()
        if not resolved.is_relative_to(root) and not (
            allow_workspace_symlinks and candidate.is_symlink()
        ):
            continue

        relative_path = candidate.relative_to(root)
        skipped = _skipped_dirs(relative_path)
        if skipped and not (
            include_explicit_skipped_dirs and skipped & set(Path(pattern).parts)
        ):
            continue

        yield candidate, relative_path


def glob_files(
    pattern: str,
    *,
    workspace_root: Path,
    max_results: int = 50,
) -> str:
    """Find files in the workspace that match a glob pattern."""
    _validate_workspace_pattern(pattern, kind="Glob pattern")

    root = workspace_root.expanduser().resolve()
    matches: list[str] = []

    for _, relative_path in _iter_workspace_files(
        root,
        pattern,
        include_explicit_skipped_dirs=True,
        allow_workspace_symlinks=True,
    ):
        if len(matches) == max_results:
            return "\n".join(matches) + f"\n[truncated after {max_results} files]"
        matches.append(relative_path.as_posix())

    if not matches:
        return f"[No files matched pattern: {pattern}]"

    return "\n".join(matches)


def search_text(
    pattern: str,
    *,
    workspace_root: Path,
    file_pattern: str = "**/*",
    max_matches: int = 50,
) -> str:
    """Search workspace file contents with a regular expression."""
    _validate_workspace_pattern(file_pattern, kind="File pattern")

    try:
        regex = re.compile(pattern)
    except re.error as e:
        raise ValueError(f"Invalid regular expression: {e}") from e

    root = workspace_root.expanduser().resolve()
    matches: list[str] = []
    skipped_large = 0

    for candidate, relative_path in _iter_workspace_files(root, file_pattern):
        if candidate.stat().st_size > _MAX_FILE_BYTES:
            skipped_large += 1
            continue

        for line_number, line in enumerate(
            candidate.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            if regex.search(line):
                if len(matches) == max_matches:
                    return "\n".join(matches) + f"\n[truncated after {max_matches} matches]"
                matches.append(f"{relative_path.as_posix()}:{line_number}: {line}")

    note = (
        f"\n[Skipped {skipped_large} files over {_MAX_FILE_BYTES} bytes; "
        "inspect them with read_file offset/limit.]"
        if skipped_large
        else ""
    )
    if not matches:
        return f"[No matches found for pattern: {pattern}]{note}"
    return "\n".join(matches) + note


def read_file(
    path: str,
    *,
    workspace_root: Path,
    offset: int = 1,
    limit: int = 200,
) -> str:
    """Read a range of lines (at most 500) from a local text file."""
    p = resolve_workspace_path(workspace_root, path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not p.is_file():
        raise IsADirectoryError(f"Path is not a file: {path}")
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    start_index = offset - 1
    selected_lines = lines[start_index : start_index + limit]

    if not selected_lines:
        return f"[No lines found from line {offset}. File has {len(lines)} lines.]"

    return "\n".join(
        f"{line_number}: {line}"
        for line_number, line in enumerate(selected_lines, start=offset)
    )


def edit_file(
    path: str,
    old_text: str,
    new_text: str,
    *,
    workspace_root: Path,
) -> str:
    """Replace one exact text match in a workspace file and return a diff."""
    p = resolve_workspace_path(workspace_root, path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not p.is_file():
        raise IsADirectoryError(f"Path is not a file: {path}")

    # Match on LF text, then write back with the file's own line endings.
    raw = p.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in raw else None
    original = raw.replace("\r\n", "\n")
    match_count = original.count(old_text)
    if match_count == 0:
        raise ValueError("Exact text was not found in the file.")
    if match_count > 1:
        raise ValueError(
            f"Exact text matched {match_count} times. Provide a more specific old_text."
        )

    updated = original.replace(old_text, new_text, 1)
    p.write_text(updated, encoding="utf-8", newline=newline)
    return build_unified_diff(
        path=p,
        before=original,
        after=updated,
        workspace_root=workspace_root,
    )


def write_file(
    path: str,
    content: str,
    *,
    workspace_root: Path,
    overwrite: bool = False,
) -> str:
    """Create a file or intentionally overwrite an existing workspace file."""
    p = resolve_workspace_path(workspace_root, path)
    if p.exists() and not p.is_file():
        raise IsADirectoryError(f"Path is not a file: {path}")
    if p.exists() and not overwrite:
        raise FileExistsError(
            f"File already exists: {path}. Set overwrite=true to replace it."
        )

    original = ""
    if p.exists():
        size = p.stat().st_size
        if size > _MAX_FILE_BYTES:
            raise ValueError(
                f"File too large: {size} bytes (limit: {_MAX_FILE_BYTES}). "
                "Use edit_file for a targeted change."
            )
        original = p.read_text(encoding="utf-8")

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return build_unified_diff(
        path=p,
        before=original,
        after=content,
        workspace_root=workspace_root,
    )


def build_unified_diff(
    *,
    path: Path,
    before: str,
    after: str,
    workspace_root: Path,
) -> str:
    root = workspace_root.expanduser().resolve()
    relative_path = path.relative_to(root).as_posix()
    diff = "\n".join(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"a/{relative_path}",
            tofile=f"b/{relative_path}",
            lineterm="",
        )
    )
    if len(diff) > _MAX_DIFF_CHARS:
        return diff[:_MAX_DIFF_CHARS] + f"\n[truncated after {_MAX_DIFF_CHARS} chars]"
    return diff
