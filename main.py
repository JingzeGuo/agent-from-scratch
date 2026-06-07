from anthropic import Anthropic
from dotenv import load_dotenv

from agent.schemas import CalculatorInput, FetchUrlInput, ReadFileInput, SearchWebInput
from agent.tool import Tool
from agent.tool_registry import ToolRegistry
from agent.tools import calculator, fetch_url, read_file, search_web


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


def main() -> None:
    load_dotenv()

    client = Anthropic()
    registry = create_registry()
    messages = []
    max_steps = 10

    while True:
        user_task = input("\nYou: ").strip()
        if not user_task:
            print("Task cannot be empty.")
            continue
        if user_task.lower() in {"exit", "quit"}:
            print("Goodbye.")
            return

        messages.append(
            {
                "role": "user",
                "content": user_task,
            }
        )

        step = 0
        task_completed = False

        while step < max_steps:
            step += 1
            print(f"\n--- Step {step} ---")

            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=1024,
                tools=registry.to_anthropic_schemas(),
                messages=messages,
            )

            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                }
            )
            for block in response.content:
                if block.type == "text":
                    print(block.text)
            if response.stop_reason == "end_turn":
                task_completed = True
                break

            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    print(f"Tool: {block.name}")
                    print(f"Input: {block.input}")

                    output, is_error = registry.execute(
                        block.name,
                        block.input,
                    )
                    status = "failed" if is_error else "succeeded"
                    print(f"Status: {status}")
                    print(f"Output: {output}")

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": output,
                            "is_error": is_error,
                        }
                    )

            if not tool_results:
                print(f"Unexpected stop reason: {response.stop_reason}")
                task_completed = True
                break

            messages.append(
                {
                    "role": "user",
                    "content": tool_results,
                }
            )

        if not task_completed:
            print(f"Agent reached the {max_steps}-step limit. Task stopped.")


if __name__ == "__main__":
    main()
