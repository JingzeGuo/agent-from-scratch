import asyncio
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any, cast
from uuid import uuid4

from .context import ContextBuilder
from .memory import MemoryContext, MemorySystem, MemoryWriteResult
from .prompts import build_system_prompt
from .provider import ProviderAdapter
from .schemas import (
    AgentRun,
    AgentStep,
    ContextBuildResult,
    PendingAction,
    RunOutcome,
    SessionEvent,
    SessionEventType,
    SessionSnapshot,
    SubAgentInput,
    ToolCall,
    ToolResult,
)
from .security import CommandPolicyResult, classify_command, redact_text
from .session import SessionStore, utc_timestamp
from .token_tracker import TokenTracker
from .tool_registry import ToolRegistry
from .verification import extract_verification_evidence, infer_task_success
from .workspace import resolve_workspace_path

TRACE_PREVIEW_CHARS = 500
SUB_AGENT_RESULT_CHARS = 4_000
PARALLEL_READ_ONLY_TOOLS = {
    "calculator",
    "read_file",
    "glob_files",
    "search_text",
    "get_diff",
}
ApprovalCallback = Callable[[ToolCall, CommandPolicyResult], bool]


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
        stream_output: bool = True,
        approval_callback: ApprovalCallback | None = None,
    ) -> None:
        self.provider_adapter = provider_adapter
        self.registry = registry
        self.model = provider_adapter.model if model is None else model
        self.provider = provider_adapter.provider if provider is None else provider
        self.max_steps = max_steps
        self.stream_output = stream_output
        self.approval_callback = approval_callback
        self.messages: list[dict[str, Any]] = []
        self.steps: list[AgentStep] = []
        self.completed_runs: list[AgentRun] = []
        self.session_store: SessionStore | None = None
        self.session_id: str | None = None
        self.memory_system: MemorySystem | None = None
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
            raise ValueError(
                "Cannot switch provider during an incomplete tool exchange."
            )
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

    def configure_approval_callback(
        self,
        approval_callback: ApprovalCallback | None,
    ) -> None:
        self.approval_callback = approval_callback

    def configure_memory(
        self,
        memory_system: MemorySystem | None,
    ) -> None:
        self.memory_system = memory_system

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
        run_id = self._new_run_id()
        self._record_run_started(run_id, user_task)
        self.messages.append(
            {
                "role": "user",
                "content": user_task,
            }
        )
        memory_context = self._retrieve_memory_context(user_task)
        self._record_memory_retrieved(run_id, memory_context)

        for step in range(1, self.max_steps + 1):
            text_blocks: list[str] = []
            tool_calls: list[ToolCall] = []
            tool_results: list[ToolResult] = []

            streamed_text = False

            def print_text_delta(text: str) -> None:
                nonlocal streamed_text
                print(text, end="", flush=True)
                streamed_text = True

            context_kwargs: dict[str, Any] = {}
            if memory_context is not None:
                context_kwargs["memory_context"] = memory_context
            model_messages = cast(
                list[dict[str, Any]],
                self.context_builder.build(
                    cast(Any, self.messages),
                    self.steps,
                    objective=user_task,
                    pending_action=self._current_pending_action(),
                    **context_kwargs,
                ),
            )
            model_request_started = perf_counter()
            self._record_model_request_started(
                run_id=run_id,
                step_number=step,
            )
            response = await self.provider_adapter.stream_response(
                system=self.system_prompt,
                tools=self.registry.to_tool_definitions(),
                messages=model_messages,
                on_text_delta=print_text_delta if self.stream_output else None,
            )

            if streamed_text:
                print()
            model_latency_ms = (perf_counter() - model_request_started) * 1000
            estimated_cost_before = self.token_tracker.estimated_cost
            self.token_tracker.add(response.usage)
            estimated_cost_delta = (
                self.token_tracker.estimated_cost - estimated_cost_before
            )
            self._record_model_response_finished(
                run_id=run_id,
                step_number=step,
                stop_reason=response.stop_reason,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                estimated_cost_delta=estimated_cost_delta,
                latency_ms=model_latency_ms,
                tool_call_count=len(response.tool_calls),
                text=response.text,
                native_metadata=response.native_metadata,
            )

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
                self._record_step_finished(
                    run_id=run_id,
                    agent_step=agent_step,
                )
                print(
                    "Protocol error: provider returned parallel tool calls "
                    "but does not support them."
                )
                return await self._finish_run_and_remember(
                    run_id=run_id,
                    objective=user_task,
                    steps=run_steps,
                    termination="protocol_error",
                    final_stop_reason=response.stop_reason,
                )

            if response.stop_reason != "end_turn":
                tool_calls.extend(response.tool_calls)
                tool_results.extend(
                    await self._execute_tool_calls(
                        run_id=run_id,
                        step_number=step,
                        tool_calls=response.tool_calls,
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
            self._record_step_finished(
                run_id=run_id,
                agent_step=agent_step,
            )

            if response.stop_reason == "end_turn":
                return await self._finish_run_and_remember(
                    run_id=run_id,
                    objective=user_task,
                    steps=run_steps,
                    termination="completed",
                    final_stop_reason=response.stop_reason,
                )

            if not tool_results:
                print(f"Protocol error stop reason: {response.stop_reason}")
                return await self._finish_run_and_remember(
                    run_id=run_id,
                    objective=user_task,
                    steps=run_steps,
                    termination="protocol_error",
                    final_stop_reason=response.stop_reason,
                )

            self.messages.append(
                self.provider_adapter.tool_result_message(tool_results)
            )
        print(f"Agent reached the {self.max_steps}-step limit. Task stopped.")
        return await self._finish_run_and_remember(
            run_id=run_id,
            objective=user_task,
            steps=run_steps,
            termination="max_steps",
            final_stop_reason=response.stop_reason,
        )

    async def remember_last_run(self) -> MemoryWriteResult | None:
        if not self.completed_runs:
            return None
        return await self._remember_run(self.completed_runs[-1])

    def _retrieve_memory_context(self, user_task: str) -> MemoryContext | None:
        if self.memory_system is None:
            return None
        return self.memory_system.search(user_task)

    def _record_memory_retrieved(
        self,
        run_id: str,
        memory_context: MemoryContext | None,
    ) -> None:
        if memory_context is None:
            return
        titles = [result.record.title for result in memory_context.results]
        self._append_session_event(
            SessionEvent(
                event_type="memory_retrieved",
                session_id=self.session_id or "",
                created_at=utc_timestamp(),
                run_id=run_id,
                message=f"retrieved {len(titles)} memory records",
                text_preview=self._preview_text("; ".join(titles)),
            )
        )

    async def _finish_run_and_remember(
        self,
        run_id: str,
        objective: str,
        steps: list[AgentStep],
        termination: RunOutcome,
        final_stop_reason: str | None,
    ) -> AgentRun:
        agent_run = self._finish_run(
            run_id=run_id,
            objective=objective,
            steps=steps,
            termination=termination,
            final_stop_reason=final_stop_reason,
        )
        await self._remember_run(agent_run)
        return agent_run

    async def _remember_run(self, agent_run: AgentRun) -> MemoryWriteResult | None:
        if self.memory_system is None:
            return None
        result = await self.memory_system.remember_run(
            provider_adapter=self.provider_adapter,
            session_id=self.session_id or "",
            agent_run=agent_run,
            workspace_root=self.registry.workspace_root,
        )
        self.token_tracker.add(result.usage)
        return result

    def _finish_run(
        self,
        run_id: str,
        objective: str,
        steps: list[AgentStep],
        termination: RunOutcome,
        final_stop_reason: str | None,
    ) -> AgentRun:
        verification = extract_verification_evidence(steps)
        agent_run = AgentRun(
            run_id=run_id,
            objective=objective,
            steps=steps,
            termination=termination,
            final_stop_reason=final_stop_reason,
            verification=verification,
            task_success=infer_task_success(verification),
        )
        self.completed_runs.append(agent_run)
        self._record_run_finished(
            run_id=run_id,
            objective=objective,
            agent_run=agent_run,
        )
        return agent_run

    def _new_run_id(self) -> str:
        return f"run-{uuid4().hex}"

    def _record_run_started(self, run_id: str, objective: str) -> None:
        self._append_session_event(
            SessionEvent(
                event_type="run_started",
                session_id=self.session_id or "",
                created_at=utc_timestamp(),
                run_id=run_id,
                objective=objective,
                provider=self.provider,
                model=self.model,
            )
        )

    def _record_model_request_started(
        self,
        run_id: str,
        step_number: int,
    ) -> None:
        self._append_session_event(
            SessionEvent(
                event_type="model_request_started",
                session_id=self.session_id or "",
                created_at=utc_timestamp(),
                run_id=run_id,
                step_number=step_number,
                provider=self.provider,
                model=self.model,
            )
        )

    def _record_model_response_finished(
        self,
        run_id: str,
        step_number: int,
        stop_reason: str | None,
        input_tokens: int,
        output_tokens: int,
        estimated_cost_delta: float,
        latency_ms: float,
        tool_call_count: int,
        text: list[str],
        native_metadata: dict[str, Any],
    ) -> None:
        self._append_session_event(
            SessionEvent(
                event_type="model_response_finished",
                session_id=self.session_id or "",
                created_at=utc_timestamp(),
                run_id=run_id,
                step_number=step_number,
                provider=self.provider,
                model=self.model,
                stop_reason=stop_reason,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost=estimated_cost_delta,
                latency_ms=latency_ms,
                tool_call_count=tool_call_count,
                text_preview=self._preview_text("\n".join(text)),
                native_metadata=native_metadata,
            )
        )

    def _record_tool_started(
        self,
        run_id: str,
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
                run_id=run_id,
                step_number=step_number,
                tool_name=tool_call.name,
                tool_use_id=tool_call.tool_use_id,
            )
        )

    async def _execute_tool_calls(
        self,
        run_id: str,
        step_number: int,
        tool_calls: list[ToolCall],
    ) -> list[ToolResult]:
        if self._can_execute_tool_calls_concurrently(tool_calls):
            self._record_tool_schedule_decided(
                run_id=run_id,
                step_number=step_number,
                tool_calls=tool_calls,
                mode="parallel",
                reason="all tool calls are read-only",
            )
            return await self._execute_tool_calls_concurrently(
                run_id=run_id,
                step_number=step_number,
                tool_calls=tool_calls,
            )

        if len(tool_calls) > 1:
            self._record_tool_schedule_decided(
                run_id=run_id,
                step_number=step_number,
                tool_calls=tool_calls,
                mode="serial",
                reason="one or more tool calls may mutate state or depend on ordering",
            )
        return await self._execute_tool_calls_serially(
            run_id=run_id,
            step_number=step_number,
            tool_calls=tool_calls,
        )

    async def _execute_tool_calls_serially(
        self,
        run_id: str,
        step_number: int,
        tool_calls: list[ToolCall],
    ) -> list[ToolResult]:
        tool_results: list[ToolResult] = []
        for tool_call in tool_calls:
            tool_results.append(
                await self._execute_one_tool_call(
                    run_id=run_id,
                    step_number=step_number,
                    tool_call=tool_call,
                )
            )
        return tool_results

    async def _execute_tool_calls_concurrently(
        self,
        run_id: str,
        step_number: int,
        tool_calls: list[ToolCall],
    ) -> list[ToolResult]:
        for tool_call in tool_calls:
            self._record_tool_started(
                run_id=run_id,
                step_number=step_number,
                tool_call=tool_call,
            )
            if self.stream_output:
                print(format_tool_activity(tool_call))

        tasks = [
            asyncio.to_thread(self._run_tool_call, tool_call)
            for tool_call in tool_calls
        ]
        completed = await asyncio.gather(*tasks)

        tool_results: list[ToolResult] = []
        for tool_call, output, is_error, latency_ms in completed:
            self._record_tool_finished(
                run_id=run_id,
                step_number=step_number,
                tool_call=tool_call,
                is_error=is_error,
                output=output,
                latency_ms=latency_ms,
            )
            tool_results.append(
                ToolResult(
                    tool_use_id=tool_call.tool_use_id,
                    content=output,
                    is_error=is_error,
                )
            )
        return tool_results

    async def _execute_one_tool_call(
        self,
        run_id: str,
        step_number: int,
        tool_call: ToolCall,
    ) -> ToolResult:
        approval = self._required_approval(tool_call)
        if approval is not None:
            approved = self._request_tool_approval(
                run_id=run_id,
                step_number=step_number,
                tool_call=tool_call,
                policy=approval,
            )
            if not approved:
                output = self._format_approval_denied(tool_call, approval)
                self._record_tool_finished(
                    run_id=run_id,
                    step_number=step_number,
                    tool_call=tool_call,
                    is_error=True,
                    output=output,
                    latency_ms=0.0,
                )
                return ToolResult(
                    tool_use_id=tool_call.tool_use_id,
                    content=output,
                    is_error=True,
                )

        self._record_tool_started(
            run_id=run_id,
            step_number=step_number,
            tool_call=tool_call,
        )
        if self.stream_output:
            print(format_tool_activity(tool_call))
        if tool_call.name == "sub_agent":
            _, output, is_error, latency_ms = await self._run_tool_call_async(
                tool_call,
                extra_kwargs={
                    "parent_agent": self,
                    "run_id": run_id,
                    "step_number": step_number,
                    "tool_call": tool_call,
                },
            )
        else:
            _, output, is_error, latency_ms = await self._run_tool_call_async(
                tool_call,
                approval_granted=approval is not None,
            )
        self._record_tool_finished(
            run_id=run_id,
            step_number=step_number,
            tool_call=tool_call,
            is_error=is_error,
            output=output,
            latency_ms=latency_ms,
        )
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            content=output,
            is_error=is_error,
        )

    def _format_sub_agent_result(
        self,
        profile: str,
        child_run: AgentRun,
        child_agent: "Agent",
    ) -> str:
        final_answer = self._sub_agent_final_answer(child_run)
        result = "\n".join(
            [
                "Sub-agent result:",
                f"child_run_id: {child_run.run_id}",
                f"profile: {profile}",
                f"termination: {child_run.termination}",
                f"steps: {len(child_run.steps)}",
                f"final_stop_reason: {child_run.final_stop_reason}",
                f"input_tokens: {child_agent.token_tracker.input_tokens}",
                f"output_tokens: {child_agent.token_tracker.output_tokens}",
                "final_answer:",
                final_answer,
            ]
        )
        return self._bounded_sub_agent_result(result)

    def _sub_agent_final_answer(self, child_run: AgentRun) -> str:
        for step in reversed(child_run.steps):
            if step.text:
                return "\n".join(step.text)
        return "[No final text returned by child agent.]"

    def _bounded_sub_agent_result(self, result: str) -> str:
        if len(result) <= SUB_AGENT_RESULT_CHARS:
            return result
        return (
            result[:SUB_AGENT_RESULT_CHARS].rstrip()
            + f"\n[truncated after {SUB_AGENT_RESULT_CHARS} chars]"
        )

    def _record_sub_agent_started(
        self,
        run_id: str,
        step_number: int,
        tool_call: ToolCall,
        parsed_input: SubAgentInput,
    ) -> None:
        self._append_session_event(
            SessionEvent(
                event_type="sub_agent_started",
                session_id=self.session_id or "",
                created_at=utc_timestamp(),
                run_id=run_id,
                step_number=step_number,
                tool_name=tool_call.name,
                tool_use_id=tool_call.tool_use_id,
                objective=parsed_input.task,
                step_count=parsed_input.max_steps,
                message=f"profile: {parsed_input.profile}",
            )
        )

    def _record_sub_agent_finished(
        self,
        run_id: str,
        step_number: int,
        tool_call: ToolCall,
        child_run: AgentRun,
        child_agent: "Agent",
    ) -> None:
        self._append_session_event(
            SessionEvent(
                event_type="sub_agent_finished",
                session_id=self.session_id or "",
                created_at=utc_timestamp(),
                run_id=run_id,
                step_number=step_number,
                tool_name=tool_call.name,
                tool_use_id=tool_call.tool_use_id,
                child_run_id=child_run.run_id,
                termination=child_run.termination,
                final_stop_reason=child_run.final_stop_reason,
                step_count=len(child_run.steps),
                input_tokens=child_agent.token_tracker.input_tokens,
                output_tokens=child_agent.token_tracker.output_tokens,
                text_preview=self._preview_text(
                    self._sub_agent_final_answer(child_run)
                ),
            )
        )

    def _run_tool_call(
        self,
        tool_call: ToolCall,
        *,
        approval_granted: bool = False,
    ) -> tuple[ToolCall, str, bool, float]:
        tool_started = perf_counter()
        output, is_error = self.registry.execute(
            tool_call.name,
            tool_call.input,
            approval_granted=approval_granted,
        )
        tool_latency_ms = (perf_counter() - tool_started) * 1000
        return tool_call, output, is_error, tool_latency_ms

    async def _run_tool_call_async(
        self,
        tool_call: ToolCall,
        *,
        approval_granted: bool = False,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> tuple[ToolCall, str, bool, float]:
        tool_started = perf_counter()
        output, is_error = await self.registry.execute_async(
            tool_call.name,
            tool_call.input,
            approval_granted=approval_granted,
            extra_kwargs=extra_kwargs,
        )
        tool_latency_ms = (perf_counter() - tool_started) * 1000
        return tool_call, output, is_error, tool_latency_ms

    def _required_approval(
        self,
        tool_call: ToolCall,
    ) -> CommandPolicyResult | None:
        if tool_call.name != "run_command":
            return None
        raw_command = tool_call.input.get("command")
        if not isinstance(raw_command, str):
            return None
        try:
            policy = classify_command(raw_command)
        except ValueError:
            return None
        if policy.decision != "requires_approval":
            return None
        return policy

    def _request_tool_approval(
        self,
        *,
        run_id: str,
        step_number: int,
        tool_call: ToolCall,
        policy: CommandPolicyResult,
    ) -> bool:
        self._record_tool_approval_event(
            event_type="tool_approval_requested",
            run_id=run_id,
            step_number=step_number,
            tool_call=tool_call,
            policy=policy,
        )
        approved = False
        if self.approval_callback is not None:
            approved = self.approval_callback(tool_call, policy)
        self._record_tool_approval_event(
            event_type="tool_approval_granted" if approved else "tool_approval_denied",
            run_id=run_id,
            step_number=step_number,
            tool_call=tool_call,
            policy=policy,
        )
        return approved

    def _record_tool_approval_event(
        self,
        *,
        event_type: SessionEventType,
        run_id: str,
        step_number: int,
        tool_call: ToolCall,
        policy: CommandPolicyResult,
    ) -> None:
        raw_command = tool_call.input.get("command")
        command = raw_command if isinstance(raw_command, str) else "[unknown]"
        self._append_session_event(
            SessionEvent(
                event_type=event_type,
                session_id=self.session_id or "",
                created_at=utc_timestamp(),
                run_id=run_id,
                step_number=step_number,
                tool_name=tool_call.name,
                tool_use_id=tool_call.tool_use_id,
                message=f"{policy.reason} Command: {command}",
            )
        )

    def _format_approval_denied(
        self,
        tool_call: ToolCall,
        policy: CommandPolicyResult,
    ) -> str:
        raw_command = tool_call.input.get("command")
        command = raw_command if isinstance(raw_command, str) else "[unknown]"
        return (
            f"Tool '{tool_call.name}' approval denied: "
            f"{policy.reason} Command: {command}"
        )

    def _can_execute_tool_calls_concurrently(
        self,
        tool_calls: list[ToolCall],
    ) -> bool:
        return len(tool_calls) > 1 and all(
            tool_call.name in PARALLEL_READ_ONLY_TOOLS for tool_call in tool_calls
        )

    def _record_tool_schedule_decided(
        self,
        run_id: str,
        step_number: int,
        tool_calls: list[ToolCall],
        mode: str,
        reason: str,
    ) -> None:
        tool_names = ", ".join(tool_call.name for tool_call in tool_calls)
        self._append_session_event(
            SessionEvent(
                event_type="tool_schedule_decided",
                session_id=self.session_id or "",
                created_at=utc_timestamp(),
                run_id=run_id,
                step_number=step_number,
                tool_call_count=len(tool_calls),
                message=f"{mode}: {reason}; tools: {tool_names}",
            )
        )

    def _record_tool_finished(
        self,
        run_id: str,
        step_number: int,
        tool_call: ToolCall,
        is_error: bool,
        output: str,
        latency_ms: float,
    ) -> None:
        self._append_session_event(
            SessionEvent(
                event_type="tool_finished",
                session_id=self.session_id or "",
                created_at=utc_timestamp(),
                run_id=run_id,
                step_number=step_number,
                tool_name=tool_call.name,
                tool_use_id=tool_call.tool_use_id,
                is_error=is_error,
                latency_ms=latency_ms,
                output_preview=self._preview_text(output),
                output_chars=len(output),
                error_type="tool_error" if is_error else None,
            )
        )

    def _record_step_finished(
        self,
        run_id: str,
        agent_step: AgentStep,
    ) -> None:
        self._append_session_event(
            SessionEvent(
                event_type="step_finished",
                session_id=self.session_id or "",
                created_at=utc_timestamp(),
                run_id=run_id,
                step_number=agent_step.step_number,
                stop_reason=agent_step.stop_reason,
                tool_call_count=len(agent_step.tool_calls),
                text_preview=self._preview_text("\n".join(agent_step.text)),
            )
        )

    def _record_run_finished(
        self,
        run_id: str,
        objective: str,
        agent_run: AgentRun,
    ) -> None:
        self._append_session_event(
            SessionEvent(
                event_type="run_finished",
                session_id=self.session_id or "",
                created_at=utc_timestamp(),
                run_id=run_id,
                objective=objective,
                termination=agent_run.termination,
                final_stop_reason=agent_run.final_stop_reason,
                task_success=agent_run.task_success,
                verification_status=agent_run.verification.status,
                step_count=len(agent_run.steps),
                input_tokens=self.token_tracker.input_tokens,
                output_tokens=self.token_tracker.output_tokens,
                estimated_cost=self.token_tracker.estimated_cost,
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

    def _preview_text(self, text: str) -> str:
        redacted = redact_text(text)
        if len(redacted) <= TRACE_PREVIEW_CHARS:
            return redacted
        return redacted[:TRACE_PREVIEW_CHARS] + "... [truncated]"

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
