import argparse
import asyncio
import os
import sys
import traceback
from collections.abc import Sequence
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import NoReturn

from dotenv import load_dotenv
from pydantic import BaseModel

from ..agent import Agent
from ..memory.store import MemoryStore
from ..provider import (
    DeepSeekProvider,
    ProviderRequestError,
    format_provider_request_error,
    load_deepseek_config,
)
from ..schemas import SessionEvent
from ..state.context import ContextBudgetExceeded
from ..state.session import SessionStore, utc_timestamp
from ..tooling.catalog import create_registry
from .commands import (
    CliSessionState,
    checkpoint_session,
    format_commands,
    handle_command,
    prompt_tool_approval,
    report_interrupted_action,
)

PACKAGE_NAME = "agent-from-scratch"
FALLBACK_VERSION = "0.1.0"


class CliArgs(BaseModel):
    resume_session_id: str | None
    api_key: str | None
    eval_args: list[str] | None
    show_help: bool = False
    show_version: bool = False


class CliInputResult(BaseModel):
    task: str | None = None
    should_exit: bool = False


class _ArgumentParser(argparse.ArgumentParser):
    """Raise instead of exiting so main() reports every usage error the same way."""

    def error(self, message: str) -> NoReturn:
        raise ValueError(message)


ARG_PARSER = _ArgumentParser(
    prog="agent",
    usage="agent [options]\n       agent eval [eval-options] [cases...]",
    add_help=False,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=f"interactive commands:\n{format_commands()}",
)
ARG_PARSER.add_argument(
    "-h", "--help", action="store_true", dest="show_help", help="Show this help message."
)
ARG_PARSER.add_argument(
    "--version", action="store_true", dest="show_version", help="Show the installed version."
)
ARG_PARSER.add_argument(
    "--resume",
    dest="resume_session_id",
    metavar="<session-id-or-name>",
    help="Resume a saved session.",
)
ARG_PARSER.add_argument("--api-key", metavar="<key>", help="Provide the provider API key.")
ARG_PARSER.add_argument("command", nargs="?", choices=["eval"], help=argparse.SUPPRESS)
ARG_PARSER.add_argument("eval_args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)


def parse_cli_args(argv: Sequence[str]) -> CliArgs:
    args = ARG_PARSER.parse_args(argv)
    return CliArgs(
        resume_session_id=args.resume_session_id,
        api_key=args.api_key,
        eval_args=args.eval_args if args.command == "eval" else None,
        show_help=args.show_help,
        show_version=args.show_version,
    )


def package_version() -> str:
    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return FALLBACK_VERSION


async def run_eval_command(
    eval_args: Sequence[str],
    *,
    api_key: str | None = None,
) -> int:
    from ..evaluation.coding_tasks import run_eval_cli

    return await run_eval_cli(eval_args, api_key=api_key)


def print_configuration_error(error: ValueError) -> None:
    print(f"Configuration error: {error}")
    print("Set it in .env or export it in your shell.")


def default_sessions_dir(workspace_root: Path) -> Path:
    return default_agent_state_dir(workspace_root) / "sessions"


def default_agent_state_dir(workspace_root: Path) -> Path:
    configured = os.getenv("AGENT_STATE_DIR")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = workspace_root / path
        return path.resolve()
    return workspace_root / ".agents"


def ensure_agent_state_gitignore(workspace_root: Path) -> None:
    state_dir = default_agent_state_dir(workspace_root)
    workspace_state_dir = (workspace_root / ".agents").resolve()
    if state_dir.resolve() != workspace_state_dir:
        return
    state_dir.mkdir(parents=True, exist_ok=True)
    gitignore_path = state_dir / ".gitignore"
    if gitignore_path.exists():
        content = gitignore_path.read_text(encoding="utf-8")
        if "memory.sqlite3*" not in content.splitlines():
            gitignore_path.write_text(
                content.rstrip("\n") + "\nmemory.sqlite3*\n", encoding="utf-8"
            )
        return
    gitignore_path.write_text(
        "\n".join(
            [
                "# Agent runtime state.",
                "# Ignore generated sessions, memory, and evaluation outputs.",
                "sessions/",
                "memory.sqlite3*",
                "evals/",
                "pending/",
                "",
            ]
        ),
        encoding="utf-8",
    )


def generate_session_id() -> str:
    return datetime.now().strftime("session-%Y%m%d-%H%M%S-%f")


def read_user_task() -> CliInputResult:
    try:
        first_line = input("\nYou: ")
    except EOFError:
        return CliInputResult(should_exit=True)
    except KeyboardInterrupt:
        print("\nInput canceled.")
        return CliInputResult()

    if first_line.strip() != "/paste":
        return CliInputResult(task=first_line.strip())

    print(
        "Paste mode: type /send on its own line to submit "
        "or /cancel to discard."
    )
    lines: list[str] = []
    while True:
        try:
            line = input("... ")
        except EOFError:
            print("\nMultiline prompt canceled.")
            return CliInputResult(should_exit=True)
        except KeyboardInterrupt:
            print("\nMultiline prompt canceled.")
            return CliInputResult()

        if line == "/send":
            return CliInputResult(task="\n".join(lines))
        if line == "/cancel":
            print("Multiline prompt canceled.")
            return CliInputResult()
        lines.append(line)


async def run_cli(
    agent: Agent,
    session_store: SessionStore,
    session_state: CliSessionState,
) -> None:
    while True:
        user_input = await asyncio.to_thread(read_user_task)
        if user_input.should_exit:
            print("Goodbye.")
            return
        user_task = user_input.task
        if user_task is None:
            continue
        if not user_task.strip():
            print("Task cannot be empty.")
            continue
        if user_task.startswith("/"):
            if handle_command(
                user_task,
                agent,
                session_store,
                session_state,
            ):
                return
            continue

        print("\nAssistant: ", end="", flush=True)
        try:
            await agent.run(user_task)
        except ProviderRequestError as error:
            print_provider_request_error(error)
        except ContextBudgetExceeded as error:
            print(f"Context budget exceeded: {error}. Use /reset to clear the context.")
        checkpoint_session(agent, session_store, session_state)


def print_provider_request_error(error: ProviderRequestError) -> None:
    print(format_provider_request_error(error))
    if provider_debug_enabled():
        traceback.print_exception(error.__cause__ or error)


def provider_debug_enabled() -> bool:
    return os.getenv("AGENT_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


async def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    raw_args = sys.argv[1:] if argv is None else argv
    try:
        cli_args = parse_cli_args(raw_args)
    except ValueError as error:
        print(error)
        return
    if cli_args.show_help:
        ARG_PARSER.print_help()
        return
    if cli_args.show_version:
        print(f"{PACKAGE_NAME} {package_version()}")
        return
    if cli_args.eval_args is not None:
        if cli_args.resume_session_id is not None:
            print("Use --resume with interactive tasks, not eval.")
            return
        exit_code = await run_eval_command(cli_args.eval_args, api_key=cli_args.api_key)
        if exit_code:
            raise SystemExit(exit_code)
        return

    workspace_root = Path.cwd().resolve()
    session_store = SessionStore(default_sessions_dir(workspace_root))
    try:
        config = load_deepseek_config(api_key=cli_args.api_key)
    except ValueError as error:
        print_configuration_error(error)
        return
    ensure_agent_state_gitignore(workspace_root)
    registry = create_registry(workspace_root)
    memory_store = MemoryStore(default_agent_state_dir(workspace_root) / "memory.sqlite3")
    provider = DeepSeekProvider(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
    )
    agent = Agent(
        provider_adapter=provider,
        registry=registry,
        memory_store=memory_store,
    )
    agent.configure_approval_callback(prompt_tool_approval)
    session_state = CliSessionState(session_id=generate_session_id())
    if cli_args.resume_session_id is not None:
        snapshot = session_store.find(cli_args.resume_session_id)
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
    try:
        await run_cli(agent, session_store, session_state)
    finally:
        await agent.wait_for_memory_jobs()
        memory_store.close()
        await provider.aclose()


def cli() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    cli()
