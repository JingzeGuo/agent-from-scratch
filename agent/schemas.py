from typing import Any, Literal

from anthropic.types import ToolResultBlockParam
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    """Input schema for the calculator tool."""

    expression: str = Field(
        description="The mathematical expression string to be evaluated, e.g., '2 * (3 + 4)' or '1024 / 8'."
    )


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


class ToolCall(BaseModel):
    """Represents a tool call request initiated by the Agent."""

    name: str
    input: dict[str, Any]
    tool_use_id: str


class ToolResult(BaseModel):
    """Represents the execution result returned by a tool to the Agent."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False

    def to_anthropic_block(self) -> ToolResultBlockParam:
        return {
            "type": self.type,
            "tool_use_id": self.tool_use_id,
            "content": self.content,
            "is_error": self.is_error,
        }


class AgentStep(BaseModel):
    step_number: int = Field(ge=1)
    stop_reason: str | None
    text: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)


class VerificationEvidence(BaseModel):
    status: Literal["not_run", "passed", "failed", "error"]
    command: str | None = None
    exit_code: int | None = None
    output: str | None = None


class AgentRun(BaseModel):
    objective: str
    steps: list[AgentStep]
    termination: Literal["completed", "max_steps", "unexpected_stop"]
    final_stop_reason: str | None
    verification: VerificationEvidence
    task_success: bool | None = None
