import asyncio

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from agent.agent import Agent
from agent.setup import create_registry

COMMANDS = {
    "/help": "Show available commands.",
    "/exit": "Exit the application.",
}


def handle_command(command: str) -> bool:
    if command == "/help":
        print("Available commands:")
        for name, description in COMMANDS.items():
            print(f"  {name:<6} {description}")
        return False
    if command == "/exit":
        print("Goodbye.")
        return True

    print(f"Unknown command: {command}")
    print("Type /help to see available commands.")
    return False


async def main() -> None:
    load_dotenv()

    agent = Agent(
        client=AsyncAnthropic(),
        registry=create_registry(),
    )

    while True:
        user_task = input("\nYou: ").strip()
        if not user_task:
            print("Task cannot be empty.")
            continue
        if user_task.startswith("/"):
            if handle_command(user_task):
                return
            continue

        print("\nAssistant: ", end="", flush=True)
        await agent.run(user_task)


if __name__ == "__main__":
    asyncio.run(main())
