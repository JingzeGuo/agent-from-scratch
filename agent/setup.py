from .schemas import CalculatorInput, FetchUrlInput, ReadFileInput, SearchWebInput
from .tool import Tool
from .tool_registry import ToolRegistry
from .tools import calculator, fetch_url, read_file, search_web


def create_registry() -> ToolRegistry:
    registry = ToolRegistry()

    calculator_tool = Tool(
        name="calculator",
        description="Safely evaluate a mathematical expression.",
        input_schema=CalculatorInput,
        fn=calculator,
    )
    read_file_tool = Tool(
        name="read_file",
        description="Read the contents of a local text file.",
        input_schema=ReadFileInput,
        fn=read_file,
    )
    fetch_url_tool = Tool(
        name="fetch_url",
        description="Fetch the content of a URL.",
        input_schema=FetchUrlInput,
        fn=fetch_url,
    )
    search_web_tool = Tool(
        name="search_web",
        description="Search the web for relevant information.",
        input_schema=SearchWebInput,
        fn=search_web,
    )

    registry.register(calculator_tool)
    registry.register(read_file_tool)
    registry.register(fetch_url_tool)
    registry.register(search_web_tool)

    return registry
