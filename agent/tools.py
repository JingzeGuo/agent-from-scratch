"""Tool implementations.

Each function here is a pure Python implementation that does NOT know
anything about LLMs, Pydantic, or schemas. They just take typed inputs
and return strings (or raise exceptions on failure).

The Tool wrapper class in `agent/tool.py` will adapt these into LLM-callable
tools with schema + validation.
"""

from __future__ import annotations

import ast
import difflib
import json
import operator
import os
import re
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx

from .workspace import resolve_workspace_path

_SKIPPED_DIRS = {".git", ".venv", "node_modules", "build", "dist"}

# ==========================================
# 1. calculator
# ==========================================
# Why not use `eval()` directly?
#   eval() can run ARBITRARY Python code — including `__import__('os').system('rm -rf /')`.
#   We use Python's `ast` module to parse the expression and only allow
#   a whitelist of math operators. This is the safe pattern.

Number = int | float

_ALLOWED_BINOPS: dict[
    type[ast.operator],
    Callable[[Number, Number], Number],
] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}
_ALLOWED_UNARYOPS: dict[
    type[ast.unaryop],
    Callable[[Number], Number],
] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float | int:
    """Recursively evaluate an AST node, allowing only math operations."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](
            _safe_eval(node.left), _safe_eval(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Disallowed expression: {ast.dump(node)}")


def calculator(expression: str) -> str:
    """Evaluate a math expression safely.

    Supports +, -, *, /, %, **, //, and unary +/-.
    Does NOT support function calls, variables, or imports.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except SyntaxError as e:
        raise ValueError(f"Invalid math expression syntax: {expression!r}") from e


# ==========================================
# 2. glob_files
# ==========================================


def _is_skipped_path(path: Path) -> bool:
    return any(part in _SKIPPED_DIRS for part in path.parts)


def _validate_workspace_pattern(pattern: str, *, kind: str) -> None:
    if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        raise ValueError(f"{kind} must be workspace-relative and cannot contain '..'")


def _iter_workspace_files(root: Path, pattern: str) -> Iterator[tuple[Path, Path]]:
    for candidate in sorted(root.glob(pattern)):
        if not candidate.is_file():
            continue

        resolved = candidate.resolve()
        if not resolved.is_relative_to(root):
            continue

        relative_path = candidate.relative_to(root)
        if _is_skipped_path(relative_path):
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

    for _, relative_path in _iter_workspace_files(root, pattern):
        matches.append(relative_path.as_posix())
        if len(matches) == max_results:
            break

    if not matches:
        return f"[No files matched pattern: {pattern}]"

    truncated = _has_more_glob_matches(root, pattern, max_results)
    output = "\n".join(matches)
    if truncated:
        output += f"\n[truncated after {max_results} files]"
    return output


def _has_more_glob_matches(root: Path, pattern: str, max_results: int) -> bool:
    match_count = 0
    for _ in _iter_workspace_files(root, pattern):
        match_count += 1
        if match_count > max_results:
            return True
    return False


# ==========================================
# 3. search_text
# ==========================================


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
    truncated = False

    for candidate, relative_path in _iter_workspace_files(root, file_pattern):
        if candidate.stat().st_size > _MAX_FILE_BYTES:
            continue

        for line_number, line in enumerate(
            candidate.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            if regex.search(line):
                matches.append(f"{relative_path.as_posix()}:{line_number}: {line}")
                if len(matches) == max_matches:
                    truncated = _has_more_text_matches(
                        root,
                        file_pattern,
                        regex,
                        max_matches,
                    )
                    output = "\n".join(matches)
                    if truncated:
                        output += f"\n[truncated after {max_matches} matches]"
                    return output

    if not matches:
        return f"[No matches found for pattern: {pattern}]"

    return "\n".join(matches)


def _has_more_text_matches(
    root: Path,
    file_pattern: str,
    regex: re.Pattern[str],
    max_matches: int,
) -> bool:
    match_count = 0
    for candidate, _ in _iter_workspace_files(root, file_pattern):
        if candidate.stat().st_size > _MAX_FILE_BYTES:
            continue

        for line in candidate.read_text(encoding="utf-8", errors="replace").splitlines():
            if regex.search(line):
                match_count += 1
                if match_count > max_matches:
                    return True
    return False


# ==========================================
# 4. read_file
# ==========================================
# We deliberately limit file size — don't blow up LLM's context window
# by reading a 50MB log file.

_MAX_FILE_BYTES = 100_000  # ~25k tokens worst case
_MAX_DIFF_CHARS = 20_000


def read_file(
    path: str,
    *,
    workspace_root: Path,
    offset: int = 1,
    limit: int = 200,
) -> str:
    """Read a range of lines from a local text file.

    Limited to 100KB and 500 lines to avoid blowing up the context window.
    """
    p = resolve_workspace_path(workspace_root, path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not p.is_file():
        raise IsADirectoryError(f"Path is not a file: {path}")
    size = p.stat().st_size
    if size > _MAX_FILE_BYTES:
        raise ValueError(
            f"File too large: {size} bytes (limit: {_MAX_FILE_BYTES}). "
            f"Consider reading specific lines instead."
        )
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    start_index = offset - 1
    selected_lines = lines[start_index : start_index + limit]

    if not selected_lines:
        return f"[No lines found from line {offset}. File has {len(lines)} lines.]"

    return "\n".join(
        f"{line_number}: {line}"
        for line_number, line in enumerate(selected_lines, start=offset)
    )


# ==========================================
# 5. edit_file
# ==========================================


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
    size = p.stat().st_size
    if size > _MAX_FILE_BYTES:
        raise ValueError(
            f"File too large: {size} bytes (limit: {_MAX_FILE_BYTES}). "
            "Use a smaller, targeted file."
        )

    original = p.read_text(encoding="utf-8")
    match_count = original.count(old_text)
    if match_count == 0:
        raise ValueError("Exact text was not found in the file.")
    if match_count > 1:
        raise ValueError(
            f"Exact text matched {match_count} times. Provide a more specific old_text."
        )

    updated = original.replace(old_text, new_text, 1)
    p.write_text(updated, encoding="utf-8")
    return _build_unified_diff(
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
    return _build_unified_diff(
        path=p,
        before=original,
        after=content,
        workspace_root=workspace_root,
    )


def _build_unified_diff(
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


# ==========================================
# 6. fetch_url
# ==========================================
# Returns plain text. We don't parse HTML here — that's a separate concern.

_FETCH_TIMEOUT_SECONDS = 10.0
_MAX_FETCH_CHARS = 20_000  # truncate long pages


def fetch_url(url: str) -> str:
    """Fetch the raw content of a URL.

    Truncates response to 20,000 characters to avoid context overflow.
    """
    with httpx.Client(timeout=_FETCH_TIMEOUT_SECONDS, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
    text = response.text
    if len(text) > _MAX_FETCH_CHARS:
        text = (
            text[:_MAX_FETCH_CHARS]
            + f"\n\n[... truncated, total {len(response.text)} chars ...]"
        )
    return text


# ==========================================
# 6. search_web (Tavily)
# ==========================================
# Tavily Search API: https://tavily.com
# - Free tier: 1000 queries/month, no credit card required
# - Designed for LLM/agent use cases (snippets are pre-optimized)
# - Uses POST + api_key in body (different from Brave's header-based auth)


def search_web(query: str, max_results: int = 5) -> str:
    """Search the web using Tavily Search API.

    Returns a JSON string with title/url/snippet for each result.
    Requires TAVILY_API_KEY in environment.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Get a free key at https://tavily.com"
        )

    with httpx.Client(timeout=10.0) as client:
        response = client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",  # basic | advanced (advanced costs more credits)
            },
        )
        response.raise_for_status()

    data = response.json()
    results = data.get("results", [])[:max_results]
    simplified = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),  # Tavily uses "content", not "description"
        }
        for r in results
    ]
    return json.dumps(simplified, ensure_ascii=False, indent=2)
