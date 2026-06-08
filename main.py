import asyncio

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from agent.agent import Agent
from agent.setup import create_registry


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
        if user_task.lower() in {"exit", "quit"}:
            print("Goodbye.")
            return
        with agent.token_tracker:
            await agent.run(user_task)


if __name__ == "__main__":
    asyncio.run(main())
