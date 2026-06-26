import asyncio
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

from agent.agent import Agent
from agent.provider import create_client, load_provider_config
from agent.schemas import SessionEvent
from agent.session import SessionStore, utc_timestamp
from agent.setup import create_registry

COMMANDS = {
    "/help": "Show available commands.",
    "/model": "Show or switch provider and model.",
    "/diff": "Show file changes from this session.",
    "/compact": "Show compacted context metrics.",
    "/rename": "Rename the current session.",
    "/sessions": "List saved sessions.",
    "/exit": "Exit the application.",
}


class CliArgs(BaseModel):
    resume_session_id: str | None
    one_shot_task: str | None


class CliSessionState(BaseModel):
    session_id: str
    session_name: str | None = None


def parse_one_shot_task(argv: Sequence[str]) -> str | None:
    if not argv:
        return None
    return " ".join(argv).strip()


def parse_cli_args(argv: Sequence[str]) -> CliArgs:
    remaining_args: list[str] = []
    resume_session_id: str | None = None
    index = 0

    while index < len(argv):
        arg = argv[index]
        if arg == "--resume":
            if resume_session_id is not None:
                raise ValueError("Use --resume only once.")
            if index + 1 >= len(argv):
                raise ValueError("Usage: --resume <session-id-or-name>")
            resume_session_id = argv[index + 1]
            index += 2
            continue
        if arg.startswith("--resume="):
            if resume_session_id is not None:
                raise ValueError("Use --resume only once.")
            resume_session_id = arg.removeprefix("--resume=")
            if not resume_session_id:
                raise ValueError("Usage: --resume <session-id-or-name>")
            index += 1
            continue

        remaining_args.append(arg)
        index += 1

    return CliArgs(
        resume_session_id=resume_session_id,
        one_shot_task=parse_one_shot_task(remaining_args),
    )


def default_sessions_dir(workspace_root: Path) -> Path:
    return workspace_root / ".agents" / "sessions"


def generate_session_id() -> str:
    return datetime.now().strftime("session-%Y%m%d-%H%M%S-%f")


def checkpoint_session(
    agent: Agent,
    session_store: SessionStore | None,
    session_state: CliSessionState | None,
) -> None:
    if session_store is None or session_state is None:
        return
    session_store.save(
        agent.create_snapshot(
            session_id=session_state.session_id,
            session_name=session_state.session_name,
        )
    )
    session_store.clear_pending_action(session_state.session_id)
    session_store.append_event(
        SessionEvent(
            event_type="checkpoint_saved",
            session_id=session_state.session_id,
            session_name=session_state.session_name,
            created_at=utc_timestamp(),
        )
    )
    print(f"Checkpoint saved: {session_state.session_id}")


def report_interrupted_action(
    session_store: SessionStore,
    session_id: str,
) -> None:
    pending_action = session_store.read_pending_action(session_id)
    if pending_action is None:
        return

    message = (
        "Interrupted action detected: "
        f"{pending_action.tool_name} ({pending_action.tool_use_id})"
    )
    session_store.append_event(
        SessionEvent(
            event_type="interrupted_action_detected",
            session_id=session_id,
            created_at=utc_timestamp(),
            step_number=pending_action.step_number,
            tool_name=pending_action.tool_name,
            tool_use_id=pending_action.tool_use_id,
            message=message,
        )
    )
    session_store.clear_pending_action(session_id)
    print(message)


def handle_command(
    command: str,
    agent: Agent | None = None,
    session_store: SessionStore | None = None,
    session_state: CliSessionState | None = None,
) -> bool:
    if command == "/help":
        print("Available commands:")
        width = max(len(name) for name in COMMANDS)
        for name, description in COMMANDS.items():
            print(f"  {name:<{width}} {description}")
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
    if command == "/compact":
        if agent is None:
            print("Compact command is unavailable.")
            return False

        result = agent.build_context_result()
        print("Context compaction:")
        print(f"  original messages: {result.original_message_count}")
        print(f"  final messages: {result.final_message_count}")
        print(f"  original chars: {result.original_context_chars}")
        print(f"  final chars: {result.final_context_chars}")
        print(f"  snipped tool results: {result.snipped_tool_results}")
        print(f"  checkpoint included: {result.checkpoint_included}")
        print(f"  hard collapsed: {result.hard_collapsed}")
        return False
    if command == "/rename" or command.startswith("/rename "):
        if agent is None or session_store is None or session_state is None:
            print("Rename command is unavailable.")
            return False

        parts = command.split()
        if len(parts) != 2:
            print("Usage: /rename <session-name>")
            return False

        session_name = parts[1]
        previous_name = session_state.session_name
        session_state.session_name = session_name
        try:
            session_store.save(
                agent.create_snapshot(
                    session_id=session_state.session_id,
                    session_name=session_state.session_name,
                )
            )
            session_store.append_event(
                SessionEvent(
                    event_type="session_renamed",
                    session_id=session_state.session_id,
                    session_name=session_state.session_name,
                    created_at=utc_timestamp(),
                )
            )
        except ValueError as error:
            session_state.session_name = previous_name
            print(f"Cannot rename session: {error}")
            return False

        print(f"Renamed session: {session_name}")
        return False
    if command == "/sessions":
        if session_store is None:
            print("Sessions command is unavailable.")
            return False

        snapshots = session_store.list_snapshots()
        if not snapshots:
            print("[No saved sessions]")
            return False

        print("Saved sessions:")
        for snapshot in snapshots:
            session_name = snapshot.session_name or "[unnamed]"
            print(f"  {snapshot.session_id}  {session_name}")
        return False
    if command == "/exit":
        print("Goodbye.")
        return True

    print(f"Unknown command: {command}")
    print("Type /help to see available commands.")
    return False


async def run_cli(
    agent: Agent,
    one_shot_task: str | None = None,
    session_store: SessionStore | None = None,
    session_state: CliSessionState | None = None,
) -> None:
    if one_shot_task is not None:
        if not one_shot_task:
            print("Task cannot be empty.")
            return
        print("\nAssistant: ", end="", flush=True)
        await agent.run(one_shot_task)
        checkpoint_session(agent, session_store, session_state)
        return

    while True:
        user_task = input("\nYou: ").strip()
        if not user_task:
            print("Task cannot be empty.")
            continue
        if user_task.startswith("/"):
            if handle_command(user_task, agent, session_store, session_state):
                return
            continue

        print("\nAssistant: ", end="", flush=True)
        await agent.run(user_task)
        checkpoint_session(agent, session_store, session_state)


async def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    raw_args = sys.argv[1:] if argv is None else argv
    try:
        cli_args = parse_cli_args(raw_args)
    except ValueError as error:
        print(error)
        return

    workspace_root = Path.cwd().resolve()
    session_store = SessionStore(default_sessions_dir(workspace_root))
    config = load_provider_config()
    registry = create_registry(workspace_root)
    agent = Agent(
        client=create_client(config),
        registry=registry,
        model=config.model,
        provider=config.provider,
    )
    session_state = CliSessionState(session_id=generate_session_id())
    if cli_args.resume_session_id is not None:
        snapshot = session_store.find(cli_args.resume_session_id)
        resumed_config = load_provider_config(
            provider=snapshot.provider,
            model=snapshot.model,
        )
        agent.switch_provider(
            client=create_client(resumed_config),
            provider=resumed_config.provider,
            model=resumed_config.model,
        )
        agent.restore_snapshot(snapshot)
        session_state = CliSessionState(
            session_id=snapshot.session_id,
            session_name=snapshot.session_name,
        )
        report_interrupted_action(session_store, session_state.session_id)
        session_store.append_event(
            SessionEvent(
                event_type="session_resumed",
                session_id=session_state.session_id,
                session_name=session_state.session_name,
                created_at=utc_timestamp(),
            )
        )
        print(f"Resumed session: {snapshot.session_id}")
    else:
        session_store.append_event(
            SessionEvent(
                event_type="session_started",
                session_id=session_state.session_id,
                created_at=utc_timestamp(),
            )
        )
    agent.configure_session_recording(session_store, session_state.session_id)
    print(f"Provider: {agent.provider} | Model: {agent.model}")
    await run_cli(agent, cli_args.one_shot_task, session_store, session_state)


if __name__ == "__main__":
    asyncio.run(main())
