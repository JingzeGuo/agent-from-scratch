from anthropic import Anthropic
from dotenv import load_dotenv

from agent.agent import Agent
from agent.setup import create_registry


def main() -> None:
    load_dotenv()

    agent = Agent(
        client=Anthropic(),
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
        agent.run(user_task)


if __name__ == "__main__":
    main()
