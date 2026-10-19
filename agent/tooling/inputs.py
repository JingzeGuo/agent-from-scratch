"""Validated input schemas for the built-in tools."""

from typing import Literal

from pydantic import BaseModel, Field


class GlobFilesInput(BaseModel):
    """Input schema for the file globbing tool."""

    pattern: str = Field(
        description="A workspace-relative glob pattern, e.g., 'tests/test_*.py' or '**/*.py'."
    )
    max_results: int = Field(
        default=50,
        ge=1,
        le=200,
        description="The maximum number of matching files to return. Defaults to 50.",
    )


class SearchTextInput(BaseModel):
    """Input schema for the content search tool."""

    pattern: str = Field(
        description="A Python regular expression to search for in workspace files."
    )
    file_pattern: str = Field(
        default="**/*",
        description="A workspace-relative glob pattern limiting which files to search. Defaults to all files.",
    )
    max_matches: int = Field(
        default=50,
        ge=1,
        le=200,
        description="The maximum number of matching lines to return. Defaults to 50.",
    )


class ReadFileInput(BaseModel):
    """Input schema for the file reading tool."""

    path: str = Field(
        description="A workspace-relative path or an absolute path inside the workspace, e.g., 'agent/agent.py'."
    )
    offset: int = Field(
        default=1,
        ge=1,
        description="The 1-based line number to start reading from. Defaults to 1.",
    )
    limit: int = Field(
        default=200,
        ge=1,
        le=500,
        description="The maximum number of lines to read. Defaults to 200.",
    )


class EditFileInput(BaseModel):
    """Input schema for the exact file editing tool."""

    path: str = Field(
        description="A workspace-relative path or an absolute path inside the workspace, e.g., 'agent/agent.py'."
    )
    old_text: str = Field(
        min_length=1,
        description="The exact text to replace. It must appear exactly once in the file.",
    )
    new_text: str = Field(
        description="The replacement text.",
    )


class WriteFileInput(BaseModel):
    """Input schema for creating or intentionally overwriting a file."""

    path: str = Field(
        description="A workspace-relative path or an absolute path inside the workspace, e.g., 'tests/test_new.py'."
    )
    content: str = Field(
        description="The complete file content to write.",
    )
    overwrite: bool = Field(
        default=False,
        description="Whether to overwrite an existing file. Defaults to false.",
    )


class GetDiffInput(BaseModel):
    """Input schema for retrieving session file changes."""

    path: str | None = Field(
        default=None,
        description="Optional workspace-relative path to a changed file. If omitted, returns all session changes.",
    )


class RunCommandInput(BaseModel):
    """Input schema for running a bounded workspace command."""

    command: str = Field(
        min_length=1,
        description="Shell command to run, e.g., '.venv/bin/python -m pytest tests/test_tools.py'.",
    )
    cwd: str | None = Field(
        default=None,
        description="Optional workspace-relative working directory. Defaults to the workspace root.",
    )
    timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        le=120,
        description="Maximum command runtime in seconds. Defaults to 10.",
    )
    max_output_chars: int = Field(
        default=8000,
        ge=200,
        le=20000,
        description="Maximum characters to keep separately from stdout and stderr.",
    )


class SubAgentInput(BaseModel):
    """Input schema for delegating a bounded read-only exploration task."""

    task: str = Field(
        min_length=1,
        description=(
            "A self-contained, bounded read-only exploration task. State the "
            "specific question or goal, include relevant known context such as "
            "file paths, symbols, symptoms, and constraints, and specify the "
            "expected findings with supporting file paths. The child can inspect "
            "the shared workspace but does not inherit the parent conversation, "
            "so include any necessary information from earlier messages."
        ),
    )
    profile: Literal["read_only_explorer"] = Field(
        default="read_only_explorer",
        description="The capability profile for the child agent.",
    )
    max_steps: int | None = Field(
        default=None,
        ge=1,
        le=8,
        description="Maximum child-agent steps. Uses the profile default when omitted.",
    )


class SearchWebInput(BaseModel):
    """Input schema for the web search tool."""

    query: str = Field(
        description="The search keywords or natural language query used to retrieve information from the web, e.g., 'Python dataclass decorator usage'."
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="The maximum number of search results to return. Defaults to 5.",
    )


class FetchUrlInput(BaseModel):
    """Input schema for the URL fetching tool."""

    url: str = Field(
        description="The complete URL of the webpage or API endpoint to fetch, including the protocol prefix, e.g., 'https://example.com/api/data'."
    )
