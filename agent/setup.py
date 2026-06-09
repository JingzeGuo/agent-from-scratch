from .schemas import CalculatorInput, FetchUrlInput, ReadFileInput, SearchWebInput
from .tool import Tool
from .tool_registry import ToolRegistry
from .tools import calculator, fetch_url, read_file, search_web


def create_registry() -> ToolRegistry:
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
            fn=read_file,
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
