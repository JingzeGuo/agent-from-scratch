from dataclasses import dataclass
from typing import Any

import pytest

from agent.setup import create_registry


@dataclass(frozen=True)
class ToolSelectionCase:
    task: str
    expected_tool: str
    expected_arguments: dict[str, Any]


TOOL_SELECTION_CASES = [
    ToolSelectionCase(
        task="Calculate (144 / 12) + 7",
        expected_tool="calculator",
        expected_arguments={"expression": "(144 / 12) + 7"},
    ),
    ToolSelectionCase(
        task="Read the local file README.md",
        expected_tool="read_file",
        expected_arguments={"path": "README.md"},
    ),
    ToolSelectionCase(
        task="Read https://docs.python.org/3/library/asyncio-task.html",
        expected_tool="fetch_url",
        expected_arguments={
            "url": "https://docs.python.org/3/library/asyncio-task.html"
        },
    ),
    ToolSelectionCase(
        task="Find information about Python structured concurrency",
        expected_tool="search_web",
        expected_arguments={
            "query": "Python structured concurrency",
            "max_results": 5,
        },
    ),
    ToolSelectionCase(
        task="Find the official TaskGroup documentation and summarize it",
        expected_tool="search_web",
        expected_arguments={
            "query": "Python asyncio TaskGroup official documentation",
            "max_results": 3,
        },
    ),
]


@pytest.mark.parametrize(
    "case",
    TOOL_SELECTION_CASES,
    ids=[
        "calculation",
        "local-file",
        "known-url",
        "web-search",
        "multi-tool-first-action",
    ],
)
def test_expected_tool_arguments_match_registered_contract(
    case: ToolSelectionCase,
) -> None:
    registry = create_registry()

    assert case.expected_tool in registry.tools, case.task

    tool = registry.tools[case.expected_tool]
    parsed_input = tool.input_schema(**case.expected_arguments)

    assert parsed_input.model_dump() == case.expected_arguments


def test_correct_tool_with_wrong_argument_name_is_an_argument_error() -> None:
    registry = create_registry()

    output, is_error = registry.execute(
        "calculator",
        {"number": "23 * 9"},
    )

    assert is_error is True
    assert "field 'expression': Field required" in output
