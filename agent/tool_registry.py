from typing import Any

from anthropic.types import ToolParam

from .tool import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def to_anthropic_schemas(self) -> list[ToolParam]:
        return [tool.to_anthropic_schema() for tool in self.tools.values()]

    def execute(
        self,
        name: str,
        raw_input: dict[str, Any],
    ) -> tuple[str, bool]:
        tool = self.tools.get(name)
        if tool is None:
            return f"Unknown tool: '{name}'. Available: {list(self.tools)}", True
        return tool.execute(raw_input)
