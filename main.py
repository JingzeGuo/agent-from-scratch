import asyncio
import os
import traceback
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from pydantic import BaseModel

from agent.agent import Agent
from agent.cli_commands import (
    CliSessionState,
    checkpoint_session,
    handle_command,
    handle_command_async,
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
from scripts.evaluate_coding_tasks import run_evaluation

PACKAGE_NAME = "agent-from-scratch"
FALLBACK_VERSION = "0.1.0"
app = typer.Typer(
    name="agent",
    help="Run a local terminal coding agent.",
    add_completion=False,
    no_args_is_help=False,
    invoke_without_command=True,
)

__all__ = [
    "CliSessionState",
    "app",
    "checkpoint_session",
    "default_agent_state_dir",
    "default_sessions_dir",
    "ensure_agent_state_gitignore",
    "entrypoint",
    "eval_command",
    "generate_session_id",
    "handle_command",
    "handle_command_async",
    "prompt_tool_approval",
    "repl_loop",
    "report_interrupted_action",
    "root_command",
    "start_interactive_session",
]


class CliInputResult(BaseModel):
    task: str | None = None
    should_exit: bool = False


def package_version() -> str:
    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return FALLBACK_VERSION


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


async def repl_loop(
    agent: Agent,
    session_store: SessionStore | None = None,
    session_state: CliSessionState | None = None,
) -> None:
    """Route terminal input to local slash commands or the Agent execution loop."""
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
            if await handle_command_async(
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


async def start_interactive_session(
    *,
    resume_session_id: str | None = None,
    api_key: str | None = None,
) -> None:
    """Configure one interactive session, then enter its terminal REPL."""
    load_dotenv()
    workspace_root = Path.cwd().resolve()
    session_store = SessionStore(default_sessions_dir(workspace_root))
    try:
        config = load_deepseek_config(api_key=api_key)
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
    if resume_session_id is not None:
        snapshot = session_store.find(resume_session_id)
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
    await repl_loop(agent, session_store, session_state)


def version_callback(value: bool) -> bool:
    if value:
        typer.echo(f"{PACKAGE_NAME} {package_version()}")
        raise typer.Exit()
    return value


@app.callback()
def root_command(
    context: typer.Context,
    resume_session_id: Annotated[
        str | None,
        typer.Option(
            "--resume",
            metavar="SESSION",
            help="Resume a saved interactive session by ID or name.",
        ),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", help="Override DEEPSEEK_API_KEY."),
    ] = None,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Show the installed version and exit.",
        ),
    ] = False,
) -> None:
    """Configure the shell CLI and start a REPL when no subcommand is given."""
    del version
    context.obj = {"api_key": api_key}
    if context.invoked_subcommand is not None:
        if resume_session_id is not None:
            raise typer.BadParameter(
                "--resume is only valid when launching the interactive agent."
            )
        return
    asyncio.run(
        start_interactive_session(
            resume_session_id=resume_session_id,
            api_key=api_key,
        )
    )


@app.command("eval")
def eval_command(
    context: typer.Context,
    cases: Annotated[
        list[str] | None,
        typer.Argument(help="Local evaluation case names."),
    ] = None,
    list_cases: Annotated[
        bool,
        typer.Option("--list", help="List local evaluation cases and exit."),
    ] = False,
    real_model: Annotated[
        bool,
        typer.Option("--real-model", help="Use the configured live model."),
    ] = False,
    max_steps: Annotated[
        int | None,
        typer.Option("--max-steps", min=1, help="Override the agent step limit."),
    ] = None,
    keep_workspaces: Annotated[
        bool,
        typer.Option("--keep-workspaces", help="Keep temporary task workspaces."),
    ] = False,
    swe_bench: Annotated[
        Path | None,
        typer.Option(
            "--swe-bench",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Generate patches for instances in a JSON or JSONL file.",
        ),
    ] = None,
    swe_bench_limit: Annotated[
        int | None,
        typer.Option("--swe-bench-limit", min=1, help="Limit selected instances."),
    ] = None,
    instance_ids: Annotated[
        list[str] | None,
        typer.Option(
            "--instance-id",
            help="Select an instance ID; may be repeated.",
        ),
    ] = None,
    predictions_path: Annotated[
        Path,
        typer.Option(
            "--swe-bench-predictions",
            help="Write standard prediction JSONL to this path.",
        ),
    ] = Path(".agents/evals/swe-bench-predictions.jsonl"),
) -> None:
    """Run deterministic, live-model, or patch-generation evaluation."""
    root_options = context.find_root().obj or {}
    api_key = root_options.get("api_key")
    try:
        exit_code = asyncio.run(
            run_evaluation(
                selected_names=cases,
                list_cases=list_cases,
                real_model=real_model,
                max_steps=max_steps,
                keep_workspaces=keep_workspaces,
                swe_bench=swe_bench,
                swe_bench_limit=swe_bench_limit,
                instance_ids=instance_ids,
                predictions_path=predictions_path,
                api_key=api_key,
            )
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    if exit_code:
        raise typer.Exit(exit_code)


def entrypoint() -> None:
    """Run the shell-level Typer application."""
    app()


if __name__ == "__main__":
    entrypoint()
