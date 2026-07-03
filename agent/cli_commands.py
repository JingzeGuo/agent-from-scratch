import json

from pydantic import BaseModel

from .agent import Agent
from .memory import MemoryRecord
from .provider import create_provider_adapter, load_provider_config
from .schemas import SessionEvent, ToolCall
from .security import ToolApprovalPolicy
from .session import SessionStore, utc_timestamp
from .workspace import resolve_workspace_path

COMMANDS = {
    "/help": "Show available commands.",
    "/model": "Show or switch provider and model.",
    "/tokens": "Show token usage and estimated cost.",
    "/status": "Show current session and agent state.",
    "/reset": "Clear the current conversation context.",
    "/save": "Save the current session checkpoint.",
    "/diff": "Show file changes from this session.",
    "/compact": "Show compacted context metrics.",
    "/memory": "Manage memory status, search, show, and reflection.",
    "/trace": "Show or export structured trace events.",
    "/rename": "Rename the current session.",
    "/sessions": "List saved sessions.",
    "/exit": "Exit the application.",
}


class CliSessionState(BaseModel):
    session_id: str
    session_name: str | None = None


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


def prompt_tool_approval(
    tool_call: ToolCall,
    policy: ToolApprovalPolicy,
) -> bool:
    raw_command = tool_call.input.get("command")
    print("\nApproval required:")
    print(f"  Tool: {tool_call.name}")
    print(f"  Reason: {policy.reason}")
    if isinstance(raw_command, str):
        print(f"  Command: {raw_command}")
    else:
        print(f"  Input: {_format_tool_approval_input(tool_call.input)}")
    answer = input("Approve tool? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def deny_tool_approval(
    tool_call: ToolCall,
    policy: ToolApprovalPolicy,
) -> bool:
    return False


def _format_tool_approval_input(tool_input: dict[str, object]) -> str:
    try:
        text = json.dumps(tool_input, ensure_ascii=True, sort_keys=True)
    except TypeError:
        text = str(tool_input)
    if len(text) <= 500:
        return text
    return text[:500] + "... [truncated]"


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
            print("Usage: /model <anthropic|deepseek|openai> [model]")
            return False

        provider = parts[1]
        model = parts[2] if len(parts) == 3 else None
        try:
            config = load_provider_config(provider=provider, model=model)
            agent.switch_provider(create_provider_adapter(config))
        except ValueError as error:
            print(f"Cannot switch model: {error}")
            return False

        print(f"Switched model: {agent.provider}/{agent.model}")
        return False
    if command == "/tokens":
        if agent is None:
            print("Tokens command is unavailable.")
            return False

        input_tokens = agent.token_tracker.input_tokens
        output_tokens = agent.token_tracker.output_tokens
        total_tokens = input_tokens + output_tokens
        print(f"Input tokens: {input_tokens}")
        print(f"Output tokens: {output_tokens}")
        print(f"Total tokens: {total_tokens}")
        print(f"Estimated cost: ${agent.token_tracker.estimated_cost:.6f}")
        return False
    if command == "/status":
        if agent is None:
            print("Status command is unavailable.")
            return False

        session_id = "[none]" if session_state is None else session_state.session_id
        session_name = (
            "[none]"
            if session_state is None or session_state.session_name is None
            else session_state.session_name
        )
        workspace = (
            "[none]"
            if agent.registry.workspace_root is None
            else agent.registry.workspace_root.as_posix()
        )
        agent_state = (
            "[unavailable]"
            if session_store is None
            else session_store.sessions_dir.parent.as_posix()
        )
        pending_action = "[unavailable]"
        if session_store is not None and session_state is not None:
            pending = session_store.read_pending_action(session_state.session_id)
            pending_action = (
                "none"
                if pending is None
                else f"{pending.tool_name} ({pending.tool_use_id})"
            )

        print("Status:")
        print(f"  Session: {session_id}")
        print(f"  Name: {session_name}")
        print(f"  Workspace files: {workspace}")
        print(f"  Agent state: {agent_state}")
        print(f"  Provider: {agent.provider}")
        print(f"  Model: {agent.model}")
        print(f"  Max steps: {agent.max_steps}")
        print(f"  Messages: {len(agent.messages)}")
        print(f"  Completed runs: {len(agent.completed_runs)}")
        print(f"  Files read: {len(agent.registry.read_files)}")
        print(f"  Files changed: {len(agent.registry.changed_files)}")
        print(f"  Pending action: {pending_action}")
        print(f"  Input tokens: {agent.token_tracker.input_tokens}")
        print(f"  Output tokens: {agent.token_tracker.output_tokens}")
        print(f"  Estimated cost: ${agent.token_tracker.estimated_cost:.6f}")
        return False
    if command == "/reset":
        if agent is None:
            print("Reset command is unavailable.")
            return False

        agent.messages.clear()
        agent.steps.clear()
        print("Conversation context reset.")
        return False
    if command == "/save":
        if agent is None or session_store is None or session_state is None:
            print("Save command is unavailable.")
            return False

        checkpoint_session(agent, session_store, session_state)
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
        if session_store is not None and session_state is not None:
            session_store.append_event(
                SessionEvent(
                    event_type="compaction_reported",
                    session_id=session_state.session_id,
                    created_at=utc_timestamp(),
                    original_message_count=result.original_message_count,
                    final_message_count=result.final_message_count,
                    original_context_chars=result.original_context_chars,
                    final_context_chars=result.final_context_chars,
                    snipped_tool_results=result.snipped_tool_results,
                    checkpoint_included=result.checkpoint_included,
                    hard_collapsed=result.hard_collapsed,
                )
            )
        print("Context compaction:")
        print(f"  original messages: {result.original_message_count}")
        print(f"  final messages: {result.final_message_count}")
        print(f"  original chars: {result.original_context_chars}")
        print(f"  final chars: {result.final_context_chars}")
        print(f"  snipped tool results: {result.snipped_tool_results}")
        print(f"  checkpoint included: {result.checkpoint_included}")
        print(f"  hard collapsed: {result.hard_collapsed}")
        return False
    if command == "/memory" or command.startswith("/memory "):
        return handle_memory_command(command, agent)
    if command == "/trace" or command.startswith("/trace "):
        if session_store is None or session_state is None:
            print("Trace command is unavailable.")
            return False

        events = session_store.read_events(session_state.session_id)
        if not events:
            print("[No trace events]")
            return False

        parts = command.split(maxsplit=1)
        if len(parts) == 2:
            if agent is None or agent.registry.workspace_root is None:
                print("Trace export is unavailable.")
                return False
            try:
                export_path = resolve_workspace_path(
                    agent.registry.workspace_root,
                    parts[1],
                )
            except ValueError as error:
                print(f"Cannot export trace: {error}")
                return False
            export_path.parent.mkdir(parents=True, exist_ok=True)
            export_path.write_text(
                "\n".join(event.model_dump_json() for event in events) + "\n",
                encoding="utf-8",
            )
            print(f"Trace exported: {export_path}")
            return False

        for event in events:
            print(event.model_dump_json())
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


def handle_memory_command(
    command: str,
    agent: Agent | None = None,
) -> bool:
    memory_system = None if agent is None else agent.memory_system
    if memory_system is None:
        print("Memory is unavailable.")
        return False

    parts = command.split(maxsplit=2)
    action = parts[1] if len(parts) >= 2 else "status"
    if action == "status":
        status = memory_system.status()
        print("Memory:")
        print(f"  Enabled: {status.enabled}")
        print(f"  Project root: {status.project_root}")
        print(f"  Global root: {status.global_root}")
        print(f"  Project records: {status.project_records}")
        print(f"  Global records: {status.global_records}")
        return False

    if action == "search":
        if len(parts) != 3 or not parts[2].strip():
            print("Usage: /memory search <query>")
            return False
        context = memory_system.search(parts[2])
        if context.is_empty():
            print("[No memory matches]")
            return False
        for result in context.results:
            result_record = result.record
            print(
                f"{result_record.id}  {result_record.scope}/{result_record.kind}  "
                f"{result.score:.3f}  {result_record.title}"
            )
        return False

    if action == "show":
        if len(parts) != 3 or not parts[2].strip():
            print("Usage: /memory show <id>")
            return False
        record = memory_system.get_record(parts[2].strip())
        if record is None:
            print(f"Memory not found: {parts[2].strip()}")
            return False
        print(format_memory_record(record))
        return False

    if action == "reflect":
        print("Memory reflection is available in the interactive CLI.")
        return False

    print("Usage: /memory [status|search|show|reflect]")
    return False


def format_memory_record(record: MemoryRecord) -> str:
    tags = ", ".join(record.tags) if record.tags else "none"
    lines = [
        f"ID: {record.id}",
        f"Scope: {record.scope}",
        f"Kind: {record.kind}",
        f"Title: {record.title}",
        f"Tags: {tags}",
    ]
    if record.confidence is not None:
        lines.append(f"Confidence: {record.confidence}")
    if record.evidence:
        lines.append(f"Evidence: {record.evidence}")
    lines.extend(["", record.content])
    return "\n".join(lines)


async def handle_command_async(
    command: str,
    agent: Agent | None = None,
    session_store: SessionStore | None = None,
    session_state: CliSessionState | None = None,
) -> bool:
    if command == "/memory reflect":
        if agent is None or agent.memory_system is None:
            print("Memory is unavailable.")
            return False
        result = await agent.remember_last_run()
        if result is None:
            print("No completed run to reflect.")
            return False
        if result.error is not None:
            print(f"Memory reflection failed: {result.error}")
            return False
        print(
            "Memory reflection saved "
            f"{len(result.saved_records)} records; skipped {result.skipped_candidates}."
        )
        return False
    return handle_command(command, agent, session_store, session_state)
