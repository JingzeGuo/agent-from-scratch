"""Tool implementations.

Each function here is a pure Python implementation that does NOT know
anything about LLMs, Pydantic, or schemas. They just take typed inputs
and return strings (or raise exceptions on failure).

The Tool wrapper class in `agent/tool.py` will adapt these into LLM-callable
tools with schema + validation.
"""

from __future__ import annotations

import ast
import operator
from pathlib import Path

import httpx

# ==========================================
# 1. calculator
# ==========================================
# Why not use `eval()` directly?
#   eval() can run ARBITRARY Python code — including `__import__('os').system('rm -rf /')`.
#   We use Python's `ast` module to parse the expression and only allow
#   a whitelist of math operators. This is the safe pattern.

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}
_ALLOWED_UNARYOPS = {
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
# 2. read_file
# ==========================================
# We deliberately limit file size — don't blow up LLM's context window
# by reading a 50MB log file.

_MAX_FILE_BYTES = 100_000  # ~25k tokens worst case


def read_file(path: str) -> str:
    """Read the contents of a local text file.

    Limited to 100KB to avoid blowing up the context window.
    """
    p = Path(path).expanduser().resolve()
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
    return p.read_text(encoding="utf-8", errors="replace")


# ==========================================
# 3. fetch_url
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
# 4. search_web (Tavily)
# ==========================================
# Tavily Search API: https://tavily.com
# - Free tier: 1000 queries/month, no credit card required
# - Designed for LLM/agent use cases (snippets are pre-optimized)
# - Uses POST + api_key in body (different from Brave's header-based auth)

import os
import json


def search_web(query: str, max_results: int = 5) -> str:
    """Search the web using Tavily Search API.

    Returns a JSON string with title/url/snippet for each result.
    Requires TAVILY_API_KEY in environment.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. " "Get a free key at https://tavily.com"
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
