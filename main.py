import asyncio
import os
import sys
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
    deny_tool_approval,
    format_memory_record,
    handle_command,
    handle_command_async,
    prompt_tool_approval,
    report_interrupted_action,
)
from agent.mcp import McpError, McpToolManager, load_mcp_tools_from_env
from agent.memory import MemoryStore, MemorySystem
from agent.provider import create_provider_adapter, load_provider_config
from agent.schemas import SessionEvent
from agent.session import SessionStore, utc_timestamp
from agent.setup import create_registry

PACKAGE_NAME = "agent-from-scratch"
FALLBACK_VERSION = "0.1.0"

__all__ = [
    "CliSessionState",
    "checkpoint_session",
    "default_global_memory_dir",
    "default_project_memory_dir",
    "default_sessions_dir",
    "format_memory_record",
    "generate_session_id",
    "handle_command",
    "handle_command_async",
    "main",
    "parse_cli_args",
    "parse_one_shot_task",
    "prompt_tool_approval",
    "report_interrupted_action",
    "run_eval_command",
    "run_cli",
]


class CliArgs(BaseModel):
    resume_session_id: str | None
    api_key: str | None
    eval_args: list[str] | None
    one_shot_task: str | None
    show_help: bool = False
    show_version: bool = False


def parse_one_shot_task(argv: Sequence[str]) -> str | None:
    if not argv:
        return None
    return " ".join(argv).strip()


def parse_cli_args(argv: Sequence[str]) -> CliArgs:
    remaining_args: list[str] = []
    resume_session_id: str | None = None
    api_key: str | None = None
    eval_args: list[str] | None = None
    show_help = False
    show_version = False
    index = 0

    while index < len(argv):
        arg = argv[index]
        if arg == "eval" and not remaining_args:
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

        remaining_args.append(arg)
        index += 1

    return CliArgs(
        resume_session_id=resume_session_id,
        api_key=api_key,
        eval_args=eval_args,
        one_shot_task=parse_one_shot_task(remaining_args),
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
    print("  agent [options] [task]")
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
    try:
        from scripts.evaluate_coding_tasks import run_eval_cli
    except ModuleNotFoundError as error:
        if error.name != "scripts":
            raise
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from scripts.evaluate_coding_tasks import run_eval_cli

    return await run_eval_cli(eval_args, api_key=api_key)


def print_configuration_error(error: ValueError) -> None:
    print(f"Configuration error: {error}")
    print("Set it in .env or export it in your shell.")


def print_mcp_error(error: ValueError | McpError) -> None:
    print(f"MCP configuration error: {error}")


def default_sessions_dir(workspace_root: Path) -> Path:
    return workspace_root / ".agents" / "sessions"


def default_project_memory_dir(workspace_root: Path) -> Path:
    return workspace_root / ".agents" / "memory"


def default_global_memory_dir() -> Path:
    configured = os.getenv("AGENT_MEMORY_GLOBAL_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path("~/.agent-from-scratch/memory").expanduser()


def memory_enabled_from_env() -> bool:
    value = os.getenv("AGENT_MEMORY_ENABLED", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def memory_int_from_env(name: str, default: int, minimum: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(minimum, parsed)


def create_memory_system(workspace_root: Path) -> MemorySystem:
    memory_system = MemorySystem(
        project_store=MemoryStore(default_project_memory_dir(workspace_root), "project"),
        global_store=MemoryStore(default_global_memory_dir(), "global"),
        enabled=memory_enabled_from_env(),
        max_results=memory_int_from_env("AGENT_MEMORY_MAX_RESULTS", 5, 1),
        max_context_chars=memory_int_from_env(
            "AGENT_MEMORY_MAX_CONTEXT_CHARS",
            4_000,
            200,
        ),
    )
    memory_system.initialize()
    return memory_system


def generate_session_id() -> str:
    return datetime.now().strftime("session-%Y%m%d-%H%M%S-%f")


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
            if await handle_command_async(
                user_task,
                agent,
                session_store,
                session_state,
            ):
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
    if cli_args.show_help:
        print_cli_help()
        return
    if cli_args.show_version:
        print(f"{PACKAGE_NAME} {package_version()}")
        return
    if cli_args.eval_args is not None:
        if cli_args.resume_session_id is not None:
            print("Use --resume with interactive or one-shot tasks, not eval.")
            return
        exit_code = await run_eval_command(cli_args.eval_args, api_key=cli_args.api_key)
        if exit_code:
            raise SystemExit(exit_code)
        return

    workspace_root = Path.cwd().resolve()
    session_store = SessionStore(default_sessions_dir(workspace_root))
    mcp_manager = McpToolManager()
    try:
        config = load_provider_config(api_key=cli_args.api_key)
    except ValueError as error:
        print_configuration_error(error)
        return
    try:
        registry = create_registry(workspace_root)
        try:
            mcp_manager = await load_mcp_tools_from_env(registry, workspace_root)
        except (ValueError, McpError) as error:
            print_mcp_error(error)
            return
        agent = Agent(
            provider_adapter=create_provider_adapter(config),
            registry=registry,
            model=config.model,
            provider=config.provider,
        )
        agent.configure_memory(create_memory_system(workspace_root))
        if cli_args.one_shot_task is None:
            agent.configure_approval_callback(prompt_tool_approval)
        else:
            agent.configure_approval_callback(deny_tool_approval)
        session_state = CliSessionState(session_id=generate_session_id())
        if cli_args.resume_session_id is not None:
            snapshot = session_store.find(cli_args.resume_session_id)
            try:
                resumed_config = load_provider_config(
                    provider=snapshot.provider,
                    model=snapshot.model,
                    api_key=cli_args.api_key,
                )
            except ValueError as error:
                print_configuration_error(error)
                return
            agent.switch_provider(create_provider_adapter(resumed_config))
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
    finally:
        await mcp_manager.close()


def cli() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    cli()
