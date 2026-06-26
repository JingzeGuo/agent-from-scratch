from copy import deepcopy
from typing import Literal, cast

from anthropic.types import MessageParam

from .schemas import (
    AgentStep,
    CommandSummary,
    ContextBuildResult,
    ContextCheckpoint,
    EditSummary,
    PendingAction,
    ToolErrorSummary,
    ToolResult,
)
from .verification import extract_verification_evidence

OMITTED_TOOL_RESULT_TEMPLATE = "[Older tool result omitted: {char_count} chars]"
CONTEXT_CHECKPOINT_HEADER = "[Structured context checkpoint]"


class ContextBuilder:
    """Build the working context sent to the model."""

    def __init__(
        self,
        max_tool_result_chars: int = 8_000,
        recent_message_count: int = 8,
        max_context_chars: int = 40_000,
        collapse_recent_message_count: int = 12,
        collapse_recent_turn_count: int = 2,
    ) -> None:
        self.max_tool_result_chars = max_tool_result_chars
        self.recent_message_count = recent_message_count
        self.max_context_chars = max_context_chars
        self.collapse_recent_message_count = collapse_recent_message_count
        self.collapse_recent_turn_count = collapse_recent_turn_count

    def build(
        self,
        messages: list[MessageParam],
        steps: list[AgentStep] | None = None,
        objective: str | None = None,
        pending_action: PendingAction | None = None,
    ) -> list[MessageParam]:
        return cast(
            list[MessageParam],
            self.build_with_metadata(
                messages,
                steps=steps,
                objective=objective,
                pending_action=pending_action,
            ).messages,
        )

    def build_with_metadata(
        self,
        messages: list[MessageParam],
        steps: list[AgentStep] | None = None,
        objective: str | None = None,
        pending_action: PendingAction | None = None,
    ) -> ContextBuildResult:
        original_context_chars = self._context_chars(messages)
        context = cast(list[MessageParam], deepcopy(messages))
        older_message_count = max(0, len(context) - self.recent_message_count)

        snipped_tool_results = 0
        for message in context[:older_message_count]:
            snipped_tool_results += self._snip_large_tool_results(message)

        checkpoint_message: MessageParam | None = None
        if steps:
            checkpoint_message = self._checkpoint_message(
                self.build_checkpoint(
                    steps,
                    objective=objective,
                    pending_action=pending_action,
                )
            )
            context.insert(0, checkpoint_message)

        hard_collapsed = False
        if checkpoint_message is not None and self._context_chars(context) > self.max_context_chars:
            context = self._collapse_context(
                checkpoint_message=checkpoint_message,
                messages=context[1:],
            )
            hard_collapsed = True

        return ContextBuildResult(
            messages=cast(list[dict[str, object]], context),
            original_message_count=len(messages),
            final_message_count=len(context),
            original_context_chars=original_context_chars,
            final_context_chars=self._context_chars(context),
            snipped_tool_results=snipped_tool_results,
            hard_collapsed=hard_collapsed,
            checkpoint_included=checkpoint_message is not None,
        )

    def build_checkpoint(
        self,
        steps: list[AgentStep],
        objective: str | None = None,
        pending_action: PendingAction | None = None,
    ) -> ContextCheckpoint:
        files_read: set[str] = set()
        files_changed: set[str] = set()
        edits: list[EditSummary] = []
        decisions: list[str] = []
        commands_run: list[CommandSummary] = []
        tool_errors: list[ToolErrorSummary] = []

        for step in steps:
            decisions.extend(self._decision_texts(step.text))
            for tool_call, tool_result in zip(step.tool_calls, step.tool_results):
                path = tool_call.input.get("path")
                if tool_call.name == "read_file" and isinstance(path, str):
                    files_read.add(path)
                if tool_call.name in {"edit_file", "write_file"} and isinstance(
                    path, str
                ):
                    files_changed.add(path)
                    edit_tool_name: Literal["edit_file", "write_file"] = cast(
                        Literal["edit_file", "write_file"],
                        tool_call.name,
                    )
                    edits.append(
                        EditSummary(
                            step_number=step.step_number,
                            tool_name=edit_tool_name,
                            path=path,
                            status="error" if tool_result.is_error else "applied",
                        )
                    )
                if tool_call.name == "run_command":
                    command = tool_call.input.get("command")
                    if isinstance(command, str):
                        commands_run.append(
                            CommandSummary(
                                command=command,
                                status=self._command_status(tool_result),
                                exit_code=self._extract_exit_code(tool_result.content),
                            )
                        )
                if tool_result.is_error:
                    tool_errors.append(
                        ToolErrorSummary(
                            step_number=step.step_number,
                            tool_name=tool_call.name,
                            message=self._first_line(tool_result.content),
                        )
                    )

        return ContextCheckpoint(
            goal=objective,
            files_read=sorted(files_read),
            files_changed=sorted(files_changed),
            edits=edits,
            decisions=decisions,
            commands_run=commands_run,
            tool_errors=tool_errors,
            pending_action=pending_action,
            latest_verification=extract_verification_evidence(steps),
        )

    def _snip_large_tool_results(self, message: MessageParam) -> int:
        content = message.get("content")
        if not isinstance(content, list):
            return 0

        snipped_count = 0
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_result":
                continue

            tool_result_content = block.get("content")
            if not isinstance(tool_result_content, str):
                continue
            if len(tool_result_content) <= self.max_tool_result_chars:
                continue

            block["content"] = OMITTED_TOOL_RESULT_TEMPLATE.format(
                char_count=len(tool_result_content)
            )
            snipped_count += 1
        return snipped_count

    def _collapse_context(
        self,
        checkpoint_message: MessageParam,
        messages: list[MessageParam],
    ) -> list[MessageParam]:
        recent_start = self._recent_complete_turn_start(messages)
        recent_start = self._expand_to_tool_boundary(messages, recent_start)
        return [checkpoint_message, *messages[recent_start:]]

    def _recent_complete_turn_start(self, messages: list[MessageParam]) -> int:
        turn_starts = [
            index
            for index, message in enumerate(messages)
            if self._message_is_user_text_turn(message)
        ]
        if not turn_starts:
            return max(0, len(messages) - self.collapse_recent_message_count)
        if len(turn_starts) <= self.collapse_recent_turn_count:
            return turn_starts[0]
        return turn_starts[-self.collapse_recent_turn_count]

    def _message_is_user_text_turn(self, message: MessageParam) -> bool:
        return message.get("role") == "user" and isinstance(message.get("content"), str)

    def _expand_to_tool_boundary(
        self,
        messages: list[MessageParam],
        recent_start: int,
    ) -> int:
        if recent_start <= 0 or recent_start >= len(messages):
            return recent_start
        if not self._message_has_tool_result(messages[recent_start]):
            return recent_start
        if self._message_has_tool_use(messages[recent_start - 1]):
            return recent_start - 1
        return recent_start

    def _message_has_tool_result(self, message: MessageParam) -> bool:
        content = message.get("content")
        if not isinstance(content, list):
            return False
        return any(
            isinstance(block, dict) and block.get("type") == "tool_result"
            for block in content
        )

    def _message_has_tool_use(self, message: MessageParam) -> bool:
        content = message.get("content")
        if not isinstance(content, list):
            return False
        return any(
            isinstance(block, dict) and block.get("type") == "tool_use"
            for block in content
        )

    def _context_chars(self, messages: list[MessageParam]) -> int:
        return sum(len(str(message.get("content", ""))) for message in messages)

    def _checkpoint_message(self, checkpoint: ContextCheckpoint) -> MessageParam:
        return {
            "role": "user",
            "content": self._format_checkpoint(checkpoint),
        }

    def _format_checkpoint(self, checkpoint: ContextCheckpoint) -> str:
        lines = [CONTEXT_CHECKPOINT_HEADER]
        lines.append("Goal:")
        lines.append(f"- {checkpoint.goal}" if checkpoint.goal else "- none")
        lines.extend(self._format_list("Files read", checkpoint.files_read))
        lines.extend(self._format_list("Files changed", checkpoint.files_changed))

        lines.append("Edits:")
        if checkpoint.edits:
            for edit in checkpoint.edits:
                lines.append(
                    f"- step {edit.step_number} {edit.tool_name} "
                    f"{edit.path}: {edit.status}"
                )
        else:
            lines.append("- none")

        lines.append("Decisions:")
        if checkpoint.decisions:
            lines.extend(f"- {decision}" for decision in checkpoint.decisions)
        else:
            lines.append("- none")

        lines.append("Commands run:")
        if checkpoint.commands_run:
            for command in checkpoint.commands_run:
                exit_code = (
                    "" if command.exit_code is None else f" exit_code={command.exit_code}"
                )
                lines.append(f"- {command.status}:{exit_code} {command.command}")
        else:
            lines.append("- none")

        lines.append("Tool errors:")
        if checkpoint.tool_errors:
            for error in checkpoint.tool_errors:
                lines.append(
                    f"- step {error.step_number} {error.tool_name}: {error.message}"
                )
        else:
            lines.append("- none")

        lines.append("Pending action:")
        if checkpoint.pending_action is None:
            lines.append("- none")
        else:
            pending = checkpoint.pending_action
            lines.append(
                f"- step {pending.step_number} {pending.tool_name} "
                f"({pending.tool_use_id})"
            )

        verification = checkpoint.latest_verification
        lines.append("Latest verification:")
        if verification.command is None:
            lines.append(f"- {verification.status}")
        else:
            lines.append(f"- {verification.status}: {verification.command}")
        return "\n".join(lines)

    def _format_list(self, heading: str, values: list[str]) -> list[str]:
        lines = [f"{heading}:"]
        if values:
            lines.extend(f"- {value}" for value in values)
        else:
            lines.append("- none")
        return lines

    def _command_status(
        self,
        tool_result: ToolResult,
    ) -> Literal["passed", "failed", "error", "unknown"]:
        if tool_result.is_error:
            return "error"
        exit_code = self._extract_exit_code(tool_result.content)
        if exit_code == 0:
            return "passed"
        if exit_code is None:
            return "unknown"
        return "failed"

    def _extract_exit_code(self, output: str) -> int | None:
        prefix = "exit_code:"
        for line in output.splitlines():
            if not line.startswith(prefix):
                continue
            value = line.removeprefix(prefix).strip()
            try:
                return int(value)
            except ValueError:
                return None
        return None

    def _first_line(self, value: str) -> str:
        first_line = value.splitlines()[0] if value.splitlines() else value
        if len(first_line) <= 160:
            return first_line
        return first_line[:157] + "..."

    def _decision_texts(self, texts: list[str]) -> list[str]:
        decisions: list[str] = []
        for text in texts:
            first_line = self._first_line(text.strip())
            if first_line:
                decisions.append(first_line)
        return decisions
