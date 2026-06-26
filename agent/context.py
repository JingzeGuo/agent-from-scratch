from copy import deepcopy
from typing import cast

from anthropic.types import MessageParam

OMITTED_TOOL_RESULT_TEMPLATE = "[Older tool result omitted: {char_count} chars]"


class ContextBuilder:
    """Build the working context sent to the model."""

    def __init__(
        self,
        max_tool_result_chars: int = 8_000,
        recent_message_count: int = 8,
    ) -> None:
        self.max_tool_result_chars = max_tool_result_chars
        self.recent_message_count = recent_message_count

    def build(self, messages: list[MessageParam]) -> list[MessageParam]:
        context = cast(list[MessageParam], deepcopy(messages))
        older_message_count = max(0, len(context) - self.recent_message_count)

        for message in context[:older_message_count]:
            self._snip_large_tool_results(message)

        return context

    def _snip_large_tool_results(self, message: MessageParam) -> None:
        content = message.get("content")
        if not isinstance(content, list):
            return

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
