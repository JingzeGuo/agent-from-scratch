"""Delegation to an isolated, read-only child agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ..schemas import AgentRun, TokenUsage

if TYPE_CHECKING:
    from ..agent import Agent


@dataclass(frozen=True)
class AgentProfile:
    allowed_tools: frozenset[str]
    default_max_steps: int
    system_prompt_suffix: str = ""


AGENT_PROFILES: dict[str, AgentProfile] = {
    "read_only_explorer": AgentProfile(
        allowed_tools=frozenset(
            {
                "read_file",
                "glob_files",
                "search_text",
            }
        ),
        default_max_steps=6,
        system_prompt_suffix=(
            "Explore the repository and return concise supporting evidence "
            "with relevant file paths. Do not modify files, execute commands, "
            "access the network, or delegate work."
        ),
    )
}

_SUB_AGENT_RESULT_CHARS = 4_000


async def sub_agent(
    task: str,
    profile: Literal["read_only_explorer"] = "read_only_explorer",
    max_steps: int | None = None,
    *,
    parent_agent: Agent,
) -> str:
    """Run a bounded read-only child agent and return a compact summary."""
    from ..agent import Agent
    from .catalog import create_registry

    profile_config = AGENT_PROFILES[profile]
    effective_max_steps = (
        profile_config.default_max_steps if max_steps is None else max_steps
    )
    child_agent = Agent(
        provider_adapter=parent_agent.provider_adapter,
        registry=create_registry(
            parent_agent.registry.workspace_root,
            allowed_tools=profile_config.allowed_tools,
            blocked_path_parts=parent_agent.registry.blocked_path_parts,
            blocked_command_names=parent_agent.registry.blocked_command_names,
        ),
        max_steps=effective_max_steps,
        stream_output=False,
        system_prompt_suffix=profile_config.system_prompt_suffix,
        activity_prefix=parent_agent.activity_prefix,
        activity_label=f"sub:{profile}",
    )
    child_agent.configure_session_recording(
        parent_agent.session_store,
        parent_agent.session_id,
        record_pending_actions=False,
    )
    if child_agent.activity_prefix:
        child_agent.print_activity(f"Starting with max_steps={effective_max_steps}.")
    child_run = await child_agent.run(task)
    if child_agent.activity_prefix:
        child_agent.print_activity(
            f"Finished termination={child_run.termination} "
            f"steps={len(child_run.steps)}."
        )
    parent_agent.token_tracker.add(
        TokenUsage(
            input_tokens=child_agent.token_tracker.input_tokens,
            output_tokens=child_agent.token_tracker.output_tokens,
        )
    )
    return _format_sub_agent_result(profile, child_run, child_agent)


def _format_sub_agent_result(
    profile: str,
    child_run: AgentRun,
    child_agent: Agent,
) -> str:
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
            child_run.final_answer() or "[No final text returned by child agent.]",
        ]
    )
    if len(result) <= _SUB_AGENT_RESULT_CHARS:
        return result
    return (
        result[:_SUB_AGENT_RESULT_CHARS].rstrip()
        + f"\n[truncated after {_SUB_AGENT_RESULT_CHARS} chars]"
    )
