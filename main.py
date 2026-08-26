import asyncio
import os
import sys
import traceback
from collections.abc import Sequence
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

from agent.agent import Agent
from agent.cli_commands import (
    COMMANDS,
    CliSessionState,
    checkpoint_session,
    handle_command,
    prompt_tool_approval,
    report_interrupted_action,
)
from agent.provider import (
    DeepSeekProvider,
    ProviderRequestError,
    format_provider_request_error,
    load_deepseek_config,
)
from agent.schemas import SessionEvent
from agent.session import SessionStore, utc_timestamp
from agent.setup import create_registry

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


def parse_cli_args(argv: Sequence[str]) -> CliArgs:
    resume_session_id: str | None = None
    api_key: str | None = None
    eval_args: list[str] | None = None
    show_help = False
    show_version = False
    index = 0

    while index < len(argv):
        arg = argv[index]
        if arg == "eval":
            eval_args = list(argv[index + 1 :])
            break
        if arg in {"--help", "-h"}:
            show_help = True
            index += 1
            continue
        if arg == "--version":
            show_version = True
            index += 1
            continue
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
        if arg == "--api-key":
            if api_key is not None:
                raise ValueError("Use --api-key only once.")
            if index + 1 >= len(argv):
                raise ValueError("Usage: --api-key <key>")
            api_key = argv[index + 1]
            index += 2
            continue
        if arg.startswith("--api-key="):
            if api_key is not None:
                raise ValueError("Use --api-key only once.")
            api_key = arg.removeprefix("--api-key=")
            if not api_key:
                raise ValueError("Usage: --api-key <key>")
            index += 1
            continue

        raise ValueError(f"Unexpected argument: {arg}")

    return CliArgs(
        resume_session_id=resume_session_id,
        api_key=api_key,
        eval_args=eval_args,
        show_help=show_help,
        show_version=show_version,
    )


def package_version() -> str:
    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return FALLBACK_VERSION


def print_cli_help() -> None:
    print("Usage:")
    print("  agent [options]")
    print("  agent eval [eval-options] [cases...]")
    print("")
    print("Options:")
    print("  -h, --help                       Show this help message.")
    print("  --version                        Show the installed version.")
    print("  --resume <session-id-or-name>    Resume a saved session.")
    print("  --api-key <key>                  Provide the provider API key.")
    print("")
    print("Interactive commands:")
    width = max(len(name) for name in COMMANDS)
    for name, description in COMMANDS.items():
        print(f"  {name:<{width}} {description}")


async def run_eval_command(
    eval_args: Sequence[str],
    *,
    api_key: str | None = None,
) -> int:
    from scripts.evaluate_coding_tasks import run_eval_cli

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
        return
    gitignore_path.write_text(
        "\n".join(
            [
                "# Agent runtime state.",
                "# Ignore generated sessions and evaluation outputs.",
                "sessions/",
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
    session_store: SessionStore | None = None,
    session_state: CliSessionState | None = None,
) -> None:
    while True:
        user_input = read_user_task()
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
            continue
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
        print_cli_help()
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
    agent = Agent(
        provider_adapter=DeepSeekProvider(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
        ),
        registry=registry,
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
    await run_cli(agent, session_store, session_state)


def cli() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    cli()
