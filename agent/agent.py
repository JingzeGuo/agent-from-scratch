from pathlib import Path
from typing import Any, cast

from .context import ContextBuilder
from .prompts import build_system_prompt
from .provider import ProviderAdapter
from .schemas import (
    AgentRun,
    AgentStep,
    ContextBuildResult,
    PendingAction,
    RunOutcome,
    SessionEvent,
    SessionSnapshot,
    ToolCall,
    ToolResult,
)
from .session import SessionStore, utc_timestamp
from .token_tracker import TokenTracker
from .tool_registry import ToolRegistry
from .verification import extract_verification_evidence, infer_task_success
from .workspace import resolve_workspace_path


def format_tool_activity(tool_call: ToolCall) -> str:
    tool_input = tool_call.input
    if tool_call.name == "read_file" and isinstance(tool_input.get("path"), str):
        return f"Reading {tool_input['path']}"
    if tool_call.name == "edit_file" and isinstance(tool_input.get("path"), str):
        return f"Editing {tool_input['path']}"
    if tool_call.name == "write_file" and isinstance(tool_input.get("path"), str):
        return f"Writing {tool_input['path']}"
    if tool_call.name == "search_text":
        return "Searching workspace text"
    if tool_call.name == "glob_files":
        return "Finding workspace files"
    if tool_call.name == "run_command":
        return "Running command"
    if tool_call.name == "get_diff":
        return "Checking session diff"
    return f"Running {tool_call.name}"


class Agent:
    def __init__(
        self,
        provider_adapter: ProviderAdapter,
        registry: ToolRegistry,
        model: str | None = None,
        provider: str | None = None,
        max_steps: int = 10,
    ) -> None:
        self.provider_adapter = provider_adapter
        self.registry = registry
        self.model = provider_adapter.model if model is None else model
        self.provider = provider_adapter.provider if provider is None else provider
        self.max_steps = max_steps
        self.messages: list[dict[str, Any]] = []
        self.steps: list[AgentStep] = []
        self.completed_runs: list[AgentRun] = []
        self.session_store: SessionStore | None = None
        self.session_id: str | None = None
        self.context_builder = ContextBuilder()
        self.token_tracker = TokenTracker(model=self.model)
        self.system_prompt = build_system_prompt(
            workspace_root=registry.workspace_root,
            registry=registry,
        )
        self._validate_provider_capabilities(provider_adapter)

    def switch_provider(
        self,
        provider_adapter: ProviderAdapter,
    ) -> None:
        if not self._provider_switch_is_safe():
            raise ValueError("Cannot switch provider during an incomplete tool exchange.")
        self._validate_provider_capabilities(provider_adapter)
        self.token_tracker.switch_model(provider_adapter.model)
        self.provider_adapter = provider_adapter
        self.provider = provider_adapter.provider
        self.model = provider_adapter.model

    def configure_session_recording(
        self,
        session_store: SessionStore | None,
        session_id: str | None,
    ) -> None:
        self.session_store = session_store
        self.session_id = session_id

    def build_context_result(self, objective: str | None = None) -> ContextBuildResult:
        return self.context_builder.build_with_metadata(
            cast(Any, self.messages),
            steps=self.steps,
            objective=objective,
            pending_action=self._current_pending_action(),
        )

    def create_snapshot(
        self,
        session_id: str,
        session_name: str | None = None,
    ) -> SessionSnapshot:
        workspace_root = self.registry.workspace_root
        return SessionSnapshot(
            session_id=session_id,
            session_name=session_name,
            workspace_root="" if workspace_root is None else workspace_root.as_posix(),
            provider=self.provider,
            model=self.model,
            max_steps=self.max_steps,
            messages=cast(list[dict[str, Any]], self.messages),
            steps=self.steps,
            completed_runs=self.completed_runs,
            read_files=self._snapshot_paths(self.registry.read_files),
            changed_files=self._snapshot_paths(self.registry.changed_files),
            original_file_contents=self._snapshot_original_file_contents(),
            input_tokens=self.token_tracker.input_tokens,
            output_tokens=self.token_tracker.output_tokens,
            estimated_cost=self.token_tracker.estimated_cost,
        )

    def restore_snapshot(self, snapshot: SessionSnapshot) -> None:
        self._validate_snapshot_workspace(snapshot)
        self.provider = snapshot.provider
        self.model = snapshot.model
        self.max_steps = snapshot.max_steps
        self.messages = list(snapshot.messages)
        self.steps = list(snapshot.steps)
        self.completed_runs = list(snapshot.completed_runs)
        self.registry.read_files = {
            self._restore_snapshot_path(path) for path in snapshot.read_files
        }
        self.registry.changed_files = {
            self._restore_snapshot_path(path) for path in snapshot.changed_files
        }
        self.registry.original_file_contents = {
            self._restore_snapshot_path(path): content
            for path, content in snapshot.original_file_contents.items()
        }
        self.token_tracker = TokenTracker(model=snapshot.model)
        self.token_tracker.input_tokens = snapshot.input_tokens
        self.token_tracker.output_tokens = snapshot.output_tokens
        self.token_tracker._estimated_cost = snapshot.estimated_cost
        self.system_prompt = build_system_prompt(
            workspace_root=self.registry.workspace_root,
            registry=self.registry,
        )

    async def run(self, user_task: str) -> AgentRun:
        run_steps: list[AgentStep] = []
        self._record_run_started(user_task)
        self.messages.append(
            {
                "role": "user",
                "content": user_task,
            }
        )

        for step in range(1, self.max_steps + 1):
            text_blocks: list[str] = []
            tool_calls: list[ToolCall] = []
            tool_results: list[ToolResult] = []

            streamed_text = False

            def print_text_delta(text: str) -> None:
                nonlocal streamed_text
                print(text, end="", flush=True)
                streamed_text = True

            model_messages = cast(
                list[dict[str, Any]],
                self.context_builder.build(
                    cast(Any, self.messages),
                    self.steps,
                    objective=user_task,
                    pending_action=self._current_pending_action(),
                ),
            )
            response = await self.provider_adapter.stream_response(
                system=self.system_prompt,
                tools=self.registry.to_tool_definitions(),
                messages=model_messages,
                on_text_delta=print_text_delta,
            )

            if streamed_text:
                print()
            self.token_tracker.add(response.usage)

            self.messages.append(response.message)
            text_blocks.extend(response.text)

            if self._has_unsupported_parallel_tool_calls(response.tool_calls):
                tool_calls.extend(response.tool_calls)
                agent_step = AgentStep(
                    step_number=step,
                    stop_reason=response.stop_reason,
                    text=text_blocks,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                )
                run_steps.append(agent_step)
                self.steps.append(agent_step)
                print(
                    "Protocol error: provider returned parallel tool calls "
                    "but does not support them."
                )
                return self._finish_run(
                    objective=user_task,
                    steps=run_steps,
                    termination="protocol_error",
                    final_stop_reason=response.stop_reason,
                )

            if response.stop_reason != "end_turn":
                for tool_call in response.tool_calls:
                    tool_calls.append(tool_call)

                    self._record_tool_started(
                        step_number=step,
                        tool_call=tool_call,
                    )
                    print(format_tool_activity(tool_call))
                    output, is_error = self.registry.execute(
                        tool_call.name,
                        tool_call.input,
                    )
                    self._record_tool_finished(
                        step_number=step,
                        tool_call=tool_call,
                        is_error=is_error,
                    )
                    tool_results.append(
                        ToolResult(
                            tool_use_id=tool_call.tool_use_id,
                            content=output,
                            is_error=is_error,
                        )
                    )

            agent_step = AgentStep(
                step_number=step,
                stop_reason=response.stop_reason,
                text=text_blocks,
                tool_calls=tool_calls,
                tool_results=tool_results,
            )
            run_steps.append(agent_step)
            self.steps.append(agent_step)

            if response.stop_reason == "end_turn":
                return self._finish_run(
                    objective=user_task,
                    steps=run_steps,
                    termination="completed",
                    final_stop_reason=response.stop_reason,
                )

            if not tool_results:
                print(f"Protocol error stop reason: {response.stop_reason}")
                return self._finish_run(
                    objective=user_task,
                    steps=run_steps,
                    termination="protocol_error",
                    final_stop_reason=response.stop_reason,
                )

            self.messages.append(self.provider_adapter.tool_result_message(tool_results))
        print(f"Agent reached the {self.max_steps}-step limit. Task stopped.")
        return self._finish_run(
            objective=user_task,
            steps=run_steps,
            termination="max_steps",
            final_stop_reason=response.stop_reason,
        )

    def _finish_run(
        self,
        objective: str,
        steps: list[AgentStep],
        termination: RunOutcome,
        final_stop_reason: str | None,
    ) -> AgentRun:
        verification = extract_verification_evidence(steps)
        agent_run = AgentRun(
            objective=objective,
            steps=steps,
            termination=termination,
            final_stop_reason=final_stop_reason,
            verification=verification,
            task_success=infer_task_success(verification),
        )
        self.completed_runs.append(agent_run)
        return agent_run

    def _record_run_started(self, objective: str) -> None:
        self._append_session_event(
            SessionEvent(
                event_type="run_started",
                session_id=self.session_id or "",
                created_at=utc_timestamp(),
                objective=objective,
            )
        )

    def _record_tool_started(
        self,
        step_number: int,
        tool_call: ToolCall,
    ) -> None:
        if self.session_store is None or self.session_id is None:
            return
        pending_action = PendingAction(
            session_id=self.session_id,
            step_number=step_number,
            tool_name=tool_call.name,
            tool_use_id=tool_call.tool_use_id,
            tool_input=tool_call.input,
            started_at=utc_timestamp(),
        )
        self.session_store.write_pending_action(pending_action)
        self._append_session_event(
            SessionEvent(
                event_type="tool_started",
                session_id=self.session_id,
                created_at=pending_action.started_at,
                step_number=step_number,
                tool_name=tool_call.name,
                tool_use_id=tool_call.tool_use_id,
            )
        )

    def _record_tool_finished(
        self,
        step_number: int,
        tool_call: ToolCall,
        is_error: bool,
    ) -> None:
        self._append_session_event(
            SessionEvent(
                event_type="tool_finished",
                session_id=self.session_id or "",
                created_at=utc_timestamp(),
                step_number=step_number,
                tool_name=tool_call.name,
                tool_use_id=tool_call.tool_use_id,
                is_error=is_error,
            )
        )

    def _append_session_event(self, event: SessionEvent) -> None:
        if self.session_store is None or self.session_id is None:
            return
        self.session_store.append_event(
            event.model_copy(update={"session_id": self.session_id})
        )

    def _current_pending_action(self) -> PendingAction | None:
        if self.session_store is None or self.session_id is None:
            return None
        return self.session_store.read_pending_action(self.session_id)

    def _provider_switch_is_safe(self) -> bool:
        if self._current_pending_action() is not None:
            return False
        if not self.steps:
            return True
        return self.steps[-1].stop_reason == "end_turn"

    def _validate_provider_capabilities(
        self,
        provider_adapter: ProviderAdapter,
    ) -> None:
        if not provider_adapter.capabilities.supports_streaming:
            raise ValueError(
                "Provider does not support streaming: "
                f"{provider_adapter.provider}/{provider_adapter.model}"
            )
        if self.registry.tools and not provider_adapter.capabilities.supports_tools:
            raise ValueError(
                "Provider does not support tools, but tools are registered: "
                f"{provider_adapter.provider}/{provider_adapter.model}"
            )

    def _has_unsupported_parallel_tool_calls(
        self,
        tool_calls: list[ToolCall],
    ) -> bool:
        return (
            len(tool_calls) > 1
            and not self.provider_adapter.capabilities.supports_parallel_tool_calls
        )

    def _snapshot_paths(self, paths: set[Path]) -> list[str]:
        workspace_root = self.registry.workspace_root
        if workspace_root is None:
            return sorted(path.as_posix() for path in paths)

        root = workspace_root.resolve()
        return sorted(path.resolve().relative_to(root).as_posix() for path in paths)

    def _snapshot_original_file_contents(self) -> dict[str, str | None]:
        workspace_root = self.registry.workspace_root
        if workspace_root is None:
            return {
                path.as_posix(): content
                for path, content in self.registry.original_file_contents.items()
            }

        root = workspace_root.resolve()
        return {
            path.resolve().relative_to(root).as_posix(): content
            for path, content in self.registry.original_file_contents.items()
        }

    def _restore_snapshot_path(self, path: str) -> Path:
        workspace_root = self.registry.workspace_root
        if workspace_root is None:
            return Path(path)
        return resolve_workspace_path(workspace_root, path)

    def _validate_snapshot_workspace(self, snapshot: SessionSnapshot) -> None:
        workspace_root = self.registry.workspace_root
        if workspace_root is None:
            return
        if not snapshot.workspace_root:
            raise ValueError("Snapshot workspace root is missing.")
        snapshot_root = Path(snapshot.workspace_root).expanduser().resolve()
        if snapshot_root != workspace_root.expanduser().resolve():
            raise ValueError(
                "Snapshot workspace root does not match the current agent workspace."
            )
