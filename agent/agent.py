from anthropic import AsyncAnthropic
from anthropic.types import MessageParam

from .schemas import AgentStep, ToolCall, ToolResult
from .token_tracker import TokenTracker
from .tool_registry import ToolRegistry


class Agent:
    def __init__(
        self,
        client: AsyncAnthropic,
        registry: ToolRegistry,
        model: str = "claude-haiku-4-5",
        max_steps: int = 10,
    ) -> None:
        self.client = client
        self.registry = registry
        self.model = model
        self.max_steps = max_steps
        self.messages: list[MessageParam] = []
        self.steps: list[AgentStep] = []
        self.token_tracker = TokenTracker()

    async def run(self, user_task: str) -> None:
        self.messages.append(
            {
                "role": "user",
                "content": user_task,
            }
        )

        step = 0
        task_completed = False

        while step < self.max_steps:
            step += 1
            text_blocks: list[str] = []
            tool_calls: list[ToolCall] = []
            tool_results: list[ToolResult] = []
            print(f"\n--- Step {step} ---")

            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                tools=self.registry.to_anthropic_schemas(),
                messages=self.messages,
            )
            self.token_tracker.add(response.usage)

            self.messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                }
            )
            for block in response.content:
                if block.type == "text":
                    text_blocks.append(block.text)
                    print(block.text)
            if response.stop_reason == "end_turn":
                self.steps.append(
                    AgentStep(
                        step_number=step,
                        stop_reason=response.stop_reason,
                        text=text_blocks,
                        tool_calls=tool_calls,
                        tool_results=tool_results,
                    )
                )
                task_completed = True
                break

            for block in response.content:
                if block.type == "tool_use":
                    tool_call = ToolCall(
                        name=block.name,
                        input=block.input,
                        tool_use_id=block.id,
                    )
                    tool_calls.append(tool_call)
                    print(f"Tool: {block.name}")
                    print(f"Input: {block.input}")

                    output, is_error = self.registry.execute(
                        tool_call.name,
                        tool_call.input,
                    )
                    status = "failed" if is_error else "succeeded"
                    print(f"Status: {status}")
                    print(f"Output: {output}")
                    tool_result = ToolResult(
                        tool_use_id=tool_call.tool_use_id,
                        content=output,
                        is_error=is_error,
                    )
                    tool_results.append(tool_result)

            if not tool_results:
                self.steps.append(
                    AgentStep(
                        step_number=step,
                        stop_reason=response.stop_reason,
                        text=text_blocks,
                        tool_calls=tool_calls,
                        tool_results=tool_results,
                    )
                )
                print(f"Unexpected stop reason: {response.stop_reason}")
                task_completed = True
                break

            self.messages.append(
                {
                    "role": "user",
                    "content": [result.to_anthropic_block() for result in tool_results],
                }
            )
            self.steps.append(
                AgentStep(
                    step_number=step,
                    stop_reason=response.stop_reason,
                    text=text_blocks,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                )
            )

        if not task_completed:
            print(f"Agent reached the {self.max_steps}-step limit. Task stopped.")
