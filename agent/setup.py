from dataclasses import dataclass
from functools import partial
from pathlib import Path

from .schemas import (
    CalculatorInput,
    EditFileInput,
    FetchUrlInput,
    GetDiffInput,
    GlobFilesInput,
    ReadFileInput,
    RunCommandInput,
    SearchTextInput,
    SearchWebInput,
    SubAgentInput,
    ToolKind,
    WriteFileInput,
)
from .tool import Tool
from .tool_registry import ToolRegistry
from .tools import (
    calculator,
    edit_file,
    fetch_url,
    glob_files,
    read_file,
    run_command,
    search_text,
    search_web,
    sub_agent,
    write_file,
)

READ_ONLY_FORBIDDEN_TOOL_KINDS: frozenset[ToolKind] = frozenset(
    {"write", "command", "network", "delegated"}
)


@dataclass(frozen=True)
class AgentProfile:
    allowed_tools: frozenset[str]
    default_max_steps: int
    max_steps_cap: int
    system_prompt_suffix: str = ""
    forbidden_tool_kinds: frozenset[ToolKind] = frozenset()


AGENT_PROFILES: dict[str, AgentProfile] = {
    "read_only_explorer": AgentProfile(
        allowed_tools=frozenset(
            {
                "calculator",
                "read_file",
                "glob_files",
                "search_text",
            }
        ),
        default_max_steps=6,
        max_steps_cap=8,
        system_prompt_suffix=(
            "Explore the repository and return concise supporting evidence "
            "with relevant file paths. Do not modify files, execute commands, "
            "access the network, or delegate work."
        ),
        forbidden_tool_kinds=READ_ONLY_FORBIDDEN_TOOL_KINDS,
    )
}


def create_registry(
    workspace_root: Path,
    *,
    allowed_tools: frozenset[str] | None = None,
    forbidden_tool_kinds: frozenset[ToolKind] = frozenset(),
) -> ToolRegistry:
    registry = ToolRegistry(workspace_root)
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
    forbidden_tools = [
        tool.name
        for tool in selected_tools
        if tool.kind in forbidden_tool_kinds
    ]
    if forbidden_tools:
        raise ValueError(
            f"Profile contains forbidden tool kinds: {sorted(forbidden_tools)}"
        )

    for tool in selected_tools:
        registry.register(tool)

    return registry


def _create_tool_catalog(
    workspace_root: Path,
    registry: ToolRegistry,
) -> dict[str, Tool]:
    tools = [
        Tool(
            name="calculator",
            description="Safely evaluate a mathematical expression.",
            input_schema=CalculatorInput,
            fn=calculator,
            kind="pure",
        ),
        Tool(
            name="read_file",
            description="Read the contents of a local text file.",
            input_schema=ReadFileInput,
            fn=partial(read_file, workspace_root=workspace_root),
            kind="read_only",
        ),
        Tool(
            name="glob_files",
            description="Find workspace files that match a glob pattern.",
            input_schema=GlobFilesInput,
            fn=partial(glob_files, workspace_root=workspace_root),
            kind="read_only",
        ),
        Tool(
            name="search_text",
            description="Search workspace file contents with a regular expression.",
            input_schema=SearchTextInput,
            fn=partial(search_text, workspace_root=workspace_root),
            kind="read_only",
        ),
        Tool(
            name="edit_file",
            description="Replace one exact text match in a workspace file and return a unified diff.",
            input_schema=EditFileInput,
            fn=partial(edit_file, workspace_root=workspace_root),
            kind="write",
        ),
        Tool(
            name="write_file",
            description="Create a new file or intentionally overwrite a file and return a unified diff.",
            input_schema=WriteFileInput,
            fn=partial(write_file, workspace_root=workspace_root),
            kind="write",
        ),
        Tool(
            name="get_diff",
            description="Return unified diffs for files changed during this session.",
            input_schema=GetDiffInput,
            fn=registry.get_diff,
            kind="read_only",
        ),
        Tool(
            name="run_command",
            description="Run a bounded command inside the workspace and return exit code, output, duration, and timeout status.",
            input_schema=RunCommandInput,
            fn=partial(run_command, workspace_root=workspace_root),
            kind="command",
        ),
        Tool(
            name="sub_agent",
            description="Delegate a bounded read-only repository exploration task to an isolated child agent.",
            input_schema=SubAgentInput,
            fn=sub_agent,
            kind="delegated",
        ),
        Tool(
            name="fetch_url",
            description="Fetch the content of a URL.",
            input_schema=FetchUrlInput,
            fn=fetch_url,
            kind="network",
        ),
        Tool(
            name="search_web",
            description="Search the web for relevant information.",
            input_schema=SearchWebInput,
            fn=search_web,
            kind="network",
        ),
    ]
    return {tool.name: tool for tool in tools}
