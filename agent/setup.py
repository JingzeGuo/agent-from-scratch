from functools import partial
from pathlib import Path

from .schemas import (
    CalculatorInput,
    FetchUrlInput,
    GlobFilesInput,
    ReadFileInput,
    SearchTextInput,
    SearchWebInput,
)
from .tool import Tool
from .tool_registry import ToolRegistry
from .tools import (
    calculator,
    fetch_url,
    glob_files,
    read_file,
    search_text,
    search_web,
)


def create_registry(workspace_root: Path) -> ToolRegistry:
    registry = ToolRegistry()
    tools = [
        Tool(
            name="calculator",
            description="Safely evaluate a mathematical expression.",
            input_schema=CalculatorInput,
            fn=calculator,
        ),
        Tool(
            name="read_file",
            description="Read the contents of a local text file.",
            input_schema=ReadFileInput,
            fn=partial(read_file, workspace_root=workspace_root),
        ),
        Tool(
            name="glob_files",
            description="Find workspace files that match a glob pattern.",
            input_schema=GlobFilesInput,
            fn=partial(glob_files, workspace_root=workspace_root),
        ),
        Tool(
            name="search_text",
            description="Search workspace file contents with a regular expression.",
            input_schema=SearchTextInput,
            fn=partial(search_text, workspace_root=workspace_root),
        ),
        Tool(
            name="fetch_url",
            description="Fetch the content of a URL.",
            input_schema=FetchUrlInput,
            fn=fetch_url,
        ),
        Tool(
            name="search_web",
            description="Search the web for relevant information.",
            input_schema=SearchWebInput,
            fn=search_web,
        ),
    ]
    for tool in tools:
        registry.register(tool)

    return registry
