import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import ValidationError

from agent.provider import DeepSeekProvider, load_deepseek_config
from agent.schemas import ToolDefinition
from agent.setup import create_registry

VAGUE_DESCRIPTIONS = {
    "calculator": "Process an input.",
    "read_file": "Get some content.",
    "search_text": "Find some content.",
    "fetch_url": "Get some content.",
    "search_web": "Find something.",
}


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


@dataclass(frozen=True)
class EvaluationResult:
    case: ToolSelectionCase
    selected_tool: str
    arguments: dict[str, Any]
    selection_correct: bool
    schema_valid: bool
    arguments_exact: bool


def build_tool_schemas(vague: bool) -> list[ToolDefinition]:
    schemas = create_registry(Path.cwd()).to_tool_definitions()
    if not vague:
        return schemas

    return [
        ToolDefinition(
            name=schema.name,
            description=VAGUE_DESCRIPTIONS[schema.name],
            input_schema=schema.input_schema,
        )
        for schema in schemas
    ]


def validate_arguments(tool_name: str, arguments: dict[str, Any]) -> bool:
    tool = create_registry(Path.cwd()).tools[tool_name]
    try:
        tool.input_schema(**arguments)
    except ValidationError:
        return False
    return True


async def evaluate_case(
    provider: DeepSeekProvider,
    case: ToolSelectionCase,
    tools: list[ToolDefinition],
) -> EvaluationResult:
    response = await provider.stream_response(
        system="Select the single best tool for the user's request.",
        tools=tools,
        messages=[{"role": "user", "content": case.task}],
    )
    tool_call = response.tool_calls[0]
    arguments = tool_call.input
    selection_correct = tool_call.name == case.expected_tool
    schema_valid = validate_arguments(tool_call.name, arguments)

    return EvaluationResult(
        case=case,
        selected_tool=tool_call.name,
        arguments=arguments,
        selection_correct=selection_correct,
        schema_valid=schema_valid,
        arguments_exact=selection_correct
        and arguments == case.expected_arguments,
    )


async def evaluate_variant(
    provider: DeepSeekProvider,
    label: str,
    vague: bool,
) -> list[EvaluationResult]:
    tools = build_tool_schemas(vague=vague)
    results = [
        await evaluate_case(provider, case, tools) for case in TOOL_SELECTION_CASES
    ]

    print(f"\n{label}")
    print("=" * len(label))
    for result in results:
        print(f"Task: {result.case.task}")
        print(f"Expected: {result.case.expected_tool}")
        print(f"Selected: {result.selected_tool} {result.arguments}")
        print(
            "Scores: "
            f"selection={result.selection_correct}, "
            f"schema_valid={result.schema_valid}, "
            f"expected_arguments={result.arguments_exact}"
        )
        print()

    total = len(results)
    selection_score = sum(result.selection_correct for result in results)
    schema_score = sum(result.schema_valid for result in results)
    argument_score = sum(result.arguments_exact for result in results)
    print(
        "Summary: "
        f"selection={selection_score}/{total}, "
        f"schema_valid={schema_score}/{total}, "
        f"expected_arguments={argument_score}/{total}"
    )
    return results


async def main() -> None:
    load_dotenv()
    config = load_deepseek_config()
    provider = DeepSeekProvider(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
    )

    await evaluate_variant(provider, "Clear descriptions", vague=False)
    await evaluate_variant(provider, "Vague descriptions", vague=True)


if __name__ == "__main__":
    asyncio.run(main())
