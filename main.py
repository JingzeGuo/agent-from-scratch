import asyncio
from pathlib import Path

from dotenv import load_dotenv

from agent.agent import Agent
from agent.provider import create_client, load_provider_config
from agent.setup import create_registry

COMMANDS = {
    "/help": "Show available commands.",
    "/model": "Show or switch provider and model.",
    "/diff": "Show file changes from this session.",
    "/exit": "Exit the application.",
}


def handle_command(command: str, agent: Agent | None = None) -> bool:
    if command == "/help":
        print("Available commands:")
        for name, description in COMMANDS.items():
            print(f"  {name:<7} {description}")
        return False
    if command == "/model":
        if agent is None:
            print("Model command is unavailable.")
        else:
            print(f"Current model: {agent.provider}/{agent.model}")
        return False
    if command.startswith("/model "):
        if agent is None:
            print("Model command is unavailable.")
            return False

        parts = command.split()
        if len(parts) > 3:
            print("Usage: /model <anthropic|deepseek> [model]")
            return False

        provider = parts[1]
        model = parts[2] if len(parts) == 3 else None
        try:
            config = load_provider_config(provider=provider, model=model)
            agent.switch_provider(
                client=create_client(config),
                provider=config.provider,
                model=config.model,
            )
        except ValueError as error:
            print(f"Cannot switch model: {error}")
            return False

        print(f"Switched model: {agent.provider}/{agent.model}")
        return False
    if command == "/diff" or command.startswith("/diff "):
        if agent is None:
            print("Diff command is unavailable.")
            return False

        parts = command.split(maxsplit=1)
        path = parts[1] if len(parts) == 2 else None
        try:
            print(agent.registry.get_diff(path))
        except ValueError as error:
            print(f"Cannot show diff: {error}")
        return False
    if command == "/exit":
        print("Goodbye.")
        return True

    print(f"Unknown command: {command}")
    print("Type /help to see available commands.")
    return False


async def main() -> None:
    load_dotenv()
    config = load_provider_config()

    workspace_root = Path.cwd().resolve()
    registry = create_registry(workspace_root)
    agent = Agent(
        client=create_client(config),
        registry=registry,
        model=config.model,
        provider=config.provider,
    )
    print(f"Provider: {agent.provider} | Model: {agent.model}")

    while True:
        user_task = input("\nYou: ").strip()
        if not user_task:
            print("Task cannot be empty.")
            continue
        if user_task.startswith("/"):
            if handle_command(user_task, agent):
                return
            continue

        print("\nAssistant: ", end="", flush=True)
        await agent.run(user_task)


if __name__ == "__main__":
    asyncio.run(main())
