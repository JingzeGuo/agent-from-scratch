from typing import Any, Literal

from pydantic import BaseModel, Field

RunOutcome = Literal[
    "completed",
    "max_steps",
    "interrupted",
    "blocked",
    "refused",
    "protocol_error",
]


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
        description="Command to run without shell operators, e.g., '.venv/bin/python -m pytest tests/test_tools.py'.",
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
        description="The delegated exploration task for the child agent.",
    )
    profile: Literal["read_only_explorer"] = Field(
        default="read_only_explorer",
        description="The capability profile for the child agent.",
    )
    max_steps: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum child-agent steps. Defaults to 3.",
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


class ToolDefinition(BaseModel):
    """Provider-neutral description of one callable tool."""

    name: str
    description: str
    input_schema: dict[str, Any]


class ToolResult(BaseModel):
    """Represents the execution result returned by a tool to the Agent."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


class TokenUsage(BaseModel):
    """Provider-neutral token usage for one model response."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class ProviderResponse(BaseModel):
    """Provider-neutral model response consumed by the agent controller."""

    message: dict[str, Any]
    stop_reason: str | None
    text: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: TokenUsage
    native_metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderCapabilities(BaseModel):
    """Provider features that affect whether an agent run is valid."""

    supports_tools: bool = True
    supports_streaming: bool = True
    supports_parallel_tool_calls: bool = True


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
    run_id: str | None = None
    objective: str
    steps: list[AgentStep]
    termination: RunOutcome
    final_stop_reason: str | None
    verification: VerificationEvidence
    task_success: bool | None = None


class CommandSummary(BaseModel):
    command: str
    status: Literal["passed", "failed", "error", "unknown"]
    exit_code: int | None = None


class EditSummary(BaseModel):
    step_number: int
    tool_name: Literal["edit_file", "write_file"]
    path: str
    status: Literal["applied", "error"]


class ToolErrorSummary(BaseModel):
    step_number: int
    tool_name: str
    message: str


class PendingAction(BaseModel):
    """Tool action that started before the latest durable checkpoint."""

    session_id: str
    step_number: int = Field(ge=1)
    tool_name: str
    tool_use_id: str
    tool_input: dict[str, Any]
    started_at: str


class ContextCheckpoint(BaseModel):
    """Structured facts retained when older raw context is compacted."""

    goal: str | None = None
    files_read: list[str] = Field(default_factory=list)
    files_changed: list[str] = Field(default_factory=list)
    edits: list[EditSummary] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    commands_run: list[CommandSummary] = Field(default_factory=list)
    tool_errors: list[ToolErrorSummary] = Field(default_factory=list)
    pending_action: PendingAction | None = None
    latest_verification: VerificationEvidence = Field(
        default_factory=lambda: VerificationEvidence(status="not_run")
    )


class ContextBuildResult(BaseModel):
    """Working context plus deterministic compaction measurements."""

    messages: list[dict[str, Any]]
    original_message_count: int = Field(ge=0)
    final_message_count: int = Field(ge=0)
    original_context_chars: int = Field(ge=0)
    final_context_chars: int = Field(ge=0)
    snipped_tool_results: int = Field(ge=0)
    hard_collapsed: bool
    checkpoint_included: bool


SessionEventType = Literal[
    "session_started",
    "session_resumed",
    "run_started",
    "model_request_started",
    "model_response_finished",
    "step_finished",
    "run_finished",
    "compaction_reported",
    "tool_schedule_decided",
    "tool_started",
    "tool_finished",
    "sub_agent_started",
    "sub_agent_finished",
    "checkpoint_saved",
    "session_renamed",
    "interrupted_action_detected",
]


class SessionEvent(BaseModel):
    """Append-only session lifecycle event used for observability."""

    event_type: SessionEventType
    session_id: str
    created_at: str
    run_id: str | None = None
    session_name: str | None = None
    objective: str | None = None
    step_number: int | None = Field(default=None, ge=1)
    provider: str | None = None
    model: str | None = None
    stop_reason: str | None = None
    termination: RunOutcome | None = None
    final_stop_reason: str | None = None
    task_success: bool | None = None
    verification_status: str | None = None
    step_count: int | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost: float | None = Field(default=None, ge=0.0)
    latency_ms: float | None = Field(default=None, ge=0.0)
    tool_call_count: int | None = Field(default=None, ge=0)
    original_message_count: int | None = Field(default=None, ge=0)
    final_message_count: int | None = Field(default=None, ge=0)
    original_context_chars: int | None = Field(default=None, ge=0)
    final_context_chars: int | None = Field(default=None, ge=0)
    snipped_tool_results: int | None = Field(default=None, ge=0)
    hard_collapsed: bool | None = None
    checkpoint_included: bool | None = None
    tool_name: str | None = None
    tool_use_id: str | None = None
    child_run_id: str | None = None
    is_error: bool | None = None
    text_preview: str | None = None
    output_preview: str | None = None
    output_chars: int | None = Field(default=None, ge=0)
    error_type: str | None = None
    native_metadata: dict[str, Any] | None = None
    message: str | None = None


class SessionSnapshot(BaseModel):
    """Serializable state required to resume a coding-agent session."""

    session_id: str
    session_name: str | None = None
    workspace_root: str
    provider: str
    model: str
    max_steps: int = Field(ge=1)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    steps: list[AgentStep] = Field(default_factory=list)
    completed_runs: list[AgentRun] = Field(default_factory=list)
    read_files: list[str] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)
    original_file_contents: dict[str, str | None] = Field(default_factory=dict)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    estimated_cost: float = Field(default=0.0, ge=0.0)
