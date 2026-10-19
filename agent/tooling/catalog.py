"""Built-in tool catalog and the registry factory for agent profiles."""

from functools import partial
from pathlib import Path

from .command import run_command
from .files import edit_file, glob_files, read_file, search_text, write_file
from .inputs import (
    EditFileInput,
    FetchUrlInput,
    GetDiffInput,
    GlobFilesInput,
    ReadFileInput,
    RunCommandInput,
    SearchTextInput,
    SearchWebInput,
    SubAgentInput,
    WriteFileInput,
)
from .sub_agent import sub_agent
from .tool import Tool
from .tool_registry import ToolRegistry
from .web import fetch_url, search_web


def create_registry(
    workspace_root: Path,
    *,
    allowed_tools: frozenset[str] | None = None,
    blocked_path_parts: frozenset[str] = frozenset(),
    blocked_command_names: frozenset[str] = frozenset(),
) -> ToolRegistry:
    registry = ToolRegistry(
        workspace_root,
        blocked_path_parts=blocked_path_parts,
        blocked_command_names=blocked_command_names,
    )
    tool_catalog = _create_tool_catalog(workspace_root, registry)
    if allowed_tools is not None:
        unknown_tools = allowed_tools.difference(tool_catalog)
        if unknown_tools:
            raise ValueError(f"Unknown allowed tools: {sorted(unknown_tools)}")

    selected_tools = [
        tool
        for name, tool in tool_catalog.items()
        if allowed_tools is None or name in allowed_tools
    ]

    for tool in selected_tools:
        registry.register(tool)

    return registry


def _create_tool_catalog(
    workspace_root: Path,
    registry: ToolRegistry,
) -> dict[str, Tool]:
    tools = [
        Tool(
            name="read_file",
            description="Read a bounded range of lines from a workspace text file.",
            input_schema=ReadFileInput,
            fn=partial(read_file, workspace_root=workspace_root),
        ),
        Tool(
            name="glob_files",
            description="Find workspace files that match a workspace-relative glob pattern.",
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
            name="edit_file",
            description="Replace one exact, unique text match in a workspace file and return a unified diff.",
            input_schema=EditFileInput,
            fn=partial(edit_file, workspace_root=workspace_root),
        ),
        Tool(
            name="write_file",
            description="Create a new file or intentionally overwrite a whole file and return a unified diff.",
            input_schema=WriteFileInput,
            fn=partial(write_file, workspace_root=workspace_root),
        ),
        Tool(
            name="get_diff",
            description="Return unified diffs for files changed during this session.",
            input_schema=GetDiffInput,
            fn=registry.get_diff,
        ),
        Tool(
            name="run_command",
            description=(
                "Run a bounded /bin/sh command inside the workspace sandbox. "
                "Shell operators are supported. Writes outside the workspace and "
                "to .git or .agents fail, and network access is unavailable. "
                "Return exit code, stdout, stderr, duration, and timeout status."
            ),
            input_schema=RunCommandInput,
            fn=partial(run_command, workspace_root=workspace_root),
        ),
        Tool(
            name="sub_agent",
            description="Delegate a bounded read-only repository exploration task to an isolated child agent.",
            input_schema=SubAgentInput,
            fn=sub_agent,
        ),
        Tool(
            name="fetch_url",
            description="Fetch the content of a known URL with bounded output.",
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
    return {tool.name: tool for tool in tools}
