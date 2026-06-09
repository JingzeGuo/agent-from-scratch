from typing import Any, Literal

from anthropic.types import ToolResultBlockParam
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    """Input schema for the calculator tool."""

    expression: str = Field(
        description="The mathematical expression string to be evaluated, e.g., '2 * (3 + 4)' or '1024 / 8'."
    )


class ReadFileInput(BaseModel):
    """Input schema for the file reading tool."""

    path: str = Field(
        description="The absolute or relative path to the local target file, e.g., './data/config.json' or '/var/log/app.log'."
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


class AgentRun(BaseModel):
    objective: str
    steps: list[AgentStep] = Field(default_factory=list)
    termination: Literal["completed", "max_steps", "unexpected_stop"]
