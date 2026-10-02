"""Token-pressure policy over full, append-only session history."""

import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import tiktoken

from .consolidation import Consolidator
from .schemas import ConsolidatedState, ContextBuildResult, WorkingContextState

Message = dict[str, Any]
CONSOLIDATED_STATE_HEADER = "[Consolidated state]"


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """BPE estimate, not a provider-exact count; callers may inject a tokenizer."""
    return len(_encoding().encode(text, disallowed_special=()))


@dataclass(frozen=True)
class ContextConfig:
    # Input budget after reserving model output. System/tools count against it.
    usable_context_tokens: int = 32_000
    soft_threshold: float = 0.65
    emergency_threshold: float = 0.90
    max_tool_result_tokens: int = 16_000
    retained_tool_result_tokens: int = 2_000
    recent_message_count: int = 8
    collapse_recent_message_count: int = 12
    collapse_recent_turn_count: int = 2

    def __post_init__(self) -> None:
        if not 0 < self.soft_threshold < self.emergency_threshold < 1:
            raise ValueError("Require 0 < soft < emergency < 1")
        if self.usable_context_tokens <= 0:
            raise ValueError("usable_context_tokens must be positive")
        if not 0 < self.retained_tool_result_tokens < self.max_tool_result_tokens:
            raise ValueError(
                "Tool result retention must be positive and below its limit"
            )
        if (
            min(
                self.recent_message_count,
                self.collapse_recent_message_count,
                self.collapse_recent_turn_count,
            )
            < 1
        ):
            raise ValueError("Recent history counts must be positive")


class ContextBudgetExceeded(ValueError):
    """Even the latest indivisible message/tool exchange cannot fit safely."""


class ContextBuilder:
    def __init__(
        self,
        consolidator: Consolidator | None = None,
        config: ContextConfig | None = None,
        token_counter: Callable[[str], int] = count_tokens,
    ) -> None:
        self.consolidator = consolidator
        self.config = config or ContextConfig()
        self.token_counter = token_counter
        self.state = WorkingContextState()

    def reset(self) -> None:
        self.state = WorkingContextState()

    def measure_tokens(self, messages: list[Message]) -> int:
        # Include roles, block types, tool IDs/arguments and message framing.
        return sum(
            4 + self.token_counter(json.dumps(m, ensure_ascii=False)) for m in messages
        )

    async def build(
        self,
        messages: list[Message],
        *,
        objective: str | None = None,
        task_starts: list[int] | None = None,
        request_overhead_tokens: int = 0,
    ) -> list[Message]:
        return (
            await self.build_with_metadata(
                messages,
                objective=objective,
                task_starts=task_starts,
                request_overhead_tokens=request_overhead_tokens,
            )
        ).messages

    async def build_with_metadata(
        self,
        messages: list[Message],
        *,
        objective: str | None = None,
        task_starts: list[int] | None = None,
        request_overhead_tokens: int = 0,
    ) -> ContextBuildResult:
        raw, snipped = self._protected_history(messages)
        starts = self._task_starts(raw, task_starts)
        objective = self._objective(raw, starts, objective)
        context = self._assemble(raw, objective)
        soft = self.config.usable_context_tokens * self.config.soft_threshold
        emergency = self.config.usable_context_tokens * self.config.emergency_threshold
        error: str | None = None
        hard_collapsed = False

        while self.measure_tokens(context) + request_overhead_tokens >= soft:
            candidates = self._fold_boundaries(raw, starts)
            if not candidates:
                break
            # Skip only prefixes whose raw suffix alone cannot fit. Stop at the
            # first viable boundary, then remeasure the actual LLM-produced state.
            end = next(
                (
                    i
                    for i in candidates
                    if self.measure_tokens(raw[i:]) + request_overhead_tokens < soft
                ),
                candidates[-1],
            )
            try:
                if self.consolidator is None:
                    raise ValueError("No consolidator configured")
                state = await self.consolidator.consolidate(
                    self.state.consolidated_state.model_copy(deep=True)
                    if self.state.consolidated_state
                    else None,
                    deepcopy(raw[self.state.folded_message_count : end]),
                    objective,
                )
                state = ConsolidatedState.model_validate(state.model_dump())
                state.current_objective = objective
                if (
                    self.measure_tokens([self._state_message(state)])
                    + request_overhead_tokens
                    >= emergency
                ):
                    raise ValueError(
                        "Consolidated state exceeds emergency token budget"
                    )
                candidate = [self._state_message(state), *raw[end:]]
                if self.measure_tokens(candidate) >= self.measure_tokens(context):
                    raise ValueError("Consolidation did not reduce context tokens")
                self.state = WorkingContextState(
                    consolidated_state=state,
                    folded_message_count=end,
                )
                context = candidate
            except Exception as exc:
                # Do not swallow cancellation/interrupts (BaseException).
                error = f"Consolidation failed ({type(exc).__name__})"
                break

        if error or self.measure_tokens(context) + request_overhead_tokens >= emergency:
            context = self._collapse_context(
                raw, starts, objective, request_overhead_tokens
            )
            hard_collapsed = True
        return self._result(
            messages, context, snipped, request_overhead_tokens, hard_collapsed, error
        )

    def inspect(
        self,
        messages: list[Message],
        *,
        objective: str | None = None,
        task_starts: list[int] | None = None,
        request_overhead_tokens: int = 0,
    ) -> ContextBuildResult:
        """Inspect the working view without invoking an LLM or folding history."""
        raw, snipped = self._protected_history(messages)
        objective = self._objective(raw, self._task_starts(raw, task_starts), objective)
        return self._result(
            messages,
            self._assemble(raw, objective),
            snipped,
            request_overhead_tokens,
            False,
            None,
        )

    def _assemble(self, raw: list[Message], objective: str | None) -> list[Message]:
        if self.state.folded_message_count > len(raw):
            raise ValueError(
                "Working context offset exceeds raw history; reset context first"
            )
        state = self.state.consolidated_state
        prefix = (
            []
            if state is None
            else [
                self._state_message(
                    state.model_copy(update={"current_objective": objective})
                )
            ]
        )
        return [*prefix, *raw[self.state.folded_message_count :]]

    @staticmethod
    def _state_message(state: ConsolidatedState) -> Message:
        return {
            "role": "user",
            "content": f"{CONSOLIDATED_STATE_HEADER}\n{state.model_dump_json()}",
        }

    @staticmethod
    def _task_starts(raw: list[Message], starts: list[int] | None) -> list[int]:
        if starts is None:
            starts = [
                i
                for i, m in enumerate(raw)
                if m.get("role") == "user" and isinstance(m.get("content"), str)
            ]
        return sorted({i for i in starts if 0 <= i < len(raw)})

    @staticmethod
    def _objective(
        raw: list[Message], starts: list[int], objective: str | None
    ) -> str | None:
        if objective is None and starts:
            content = raw[starts[-1]].get("content")
            if isinstance(content, str):
                return content
        return objective

    @staticmethod
    def _safe_boundaries(raw: list[Message]) -> list[int]:
        """Never split an assistant's tool requests from any of their results."""
        pending: set[str] = set()
        boundaries = [0]
        for i, message in enumerate(raw):
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_use":
                        pending.add(block["id"])
                    elif block.get("type") == "tool_result":
                        pending.discard(block["tool_use_id"])
            if not pending:
                boundaries.append(i + 1)
        return boundaries

    def _fold_boundaries(self, raw: list[Message], starts: list[int]) -> list[int]:
        safe = self._safe_boundaries(raw)
        offset = self.state.folded_message_count
        active_start = starts[-1] if starts else 0
        # A new task start is the exclusive end of the preceding task.
        completed = [i for i in starts if offset < i <= active_start and i in safe]
        recent_limit = len(raw) - self.config.recent_message_count
        active = [i for i in safe if max(offset, active_start) < i <= recent_limit]
        return [*completed, *active]

    def _collapse_context(
        self,
        raw: list[Message],
        starts: list[int],
        objective: str | None,
        overhead: int,
    ) -> list[Message]:
        # Loss is explicit. Raw storage and the last validated state stay intact,
        # so a later request can retry consolidation.
        warning = {
            "role": "user",
            "content": "[Emergency context fallback: consolidation unavailable or insufficient; "
            "older raw history may be omitted.]\nCurrent objective: "
            + (objective or "unknown"),
        }
        state = self.state.consolidated_state
        prefix = (
            [warning]
            if state is None
            else [
                self._state_message(
                    state.model_copy(update={"current_objective": objective})
                ),
                warning,
            ]
        )
        desired = (
            starts[-self.config.collapse_recent_turn_count]
            if len(starts) >= self.config.collapse_recent_turn_count
            else starts[0]
            if starts
            else max(0, len(raw) - self.config.collapse_recent_message_count)
        )
        safe = [
            i
            for i in self._safe_boundaries(raw)
            if self.state.folded_message_count <= i < len(raw)
        ]
        before = [i for i in safe if i <= desired]
        start = before[-1] if before else self.state.folded_message_count
        limit = self.config.usable_context_tokens * self.config.emergency_threshold
        for i in [start, *(j for j in safe if j > start)]:
            candidate = [*prefix, *raw[i:]]
            if self.measure_tokens(candidate) + overhead < limit:
                return candidate
        raise ContextBudgetExceeded("Latest context cannot fit emergency token budget")

    def _protected_history(self, messages: list[Message]) -> tuple[list[Message], int]:
        raw = deepcopy(messages)
        snipped = 0
        for message in raw:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                output = block.get("content")
                if not isinstance(output, str):
                    continue
                tokens = self.token_counter(output)
                if tokens <= self.config.max_tool_result_tokens:
                    continue
                half = self.config.retained_tool_result_tokens // 2
                head = self._clip(output, half)
                tail = self._clip(output, half, tail=True)
                block["content"] = (
                    f"{head}\n[Oversized tool result truncated: "
                    f"approximately {tokens} tokens; middle omitted]\n{tail}"
                )
                snipped += 1
        return raw, snipped

    def _clip(self, text: str, tokens: int, *, tail: bool = False) -> str:
        low, high = 0, len(text)
        while low < high:
            mid = (low + high + 1) // 2
            part = text[-mid:] if tail else text[:mid]
            if self.token_counter(part) <= tokens:
                low = mid
            else:
                high = mid - 1
        return (text[-low:] if tail else text[:low]) if low else ""

    def _result(
        self,
        original: list[Message],
        context: list[Message],
        snipped: int,
        overhead: int,
        hard_collapsed: bool,
        error: str | None,
    ) -> ContextBuildResult:
        return ContextBuildResult(
            messages=context,
            original_message_count=len(original),
            final_message_count=len(context),
            original_context_chars=sum(
                len(str(m.get("content", ""))) for m in original
            ),
            final_context_chars=sum(len(str(m.get("content", ""))) for m in context),
            original_context_tokens=self.measure_tokens(original) + overhead,
            final_context_tokens=self.measure_tokens(context) + overhead,
            snipped_tool_results=snipped,
            hard_collapsed=hard_collapsed,
            # Retained field name for existing CLI/session trace consumers.
            summary_included=self.state.consolidated_state is not None,
            folded_message_count=self.state.folded_message_count,
            consolidation_error=error,
        )
