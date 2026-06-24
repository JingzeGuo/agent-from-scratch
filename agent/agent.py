from anthropic import AsyncAnthropic
from anthropic.types import MessageParam

from .prompts import build_system_prompt
from .schemas import (
    AgentRun,
    AgentStep,
    ToolCall,
    ToolResult,
)
from .token_tracker import TokenTracker
from .tool_registry import ToolRegistry
from .verification import extract_verification_evidence, infer_task_success


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
        client: AsyncAnthropic,
        registry: ToolRegistry,
        model: str = "claude-haiku-4-5",
        provider: str = "anthropic",
        max_steps: int = 10,
    ) -> None:
        self.client = client
        self.registry = registry
        self.model = model
        self.provider = provider
        self.max_steps = max_steps
        self.messages: list[MessageParam] = []
        self.steps: list[AgentStep] = []
        self.token_tracker = TokenTracker(model=model)
        self.system_prompt = build_system_prompt(
            workspace_root=registry.workspace_root,
            registry=registry,
        )

    def switch_provider(
        self,
        client: AsyncAnthropic,
        provider: str,
        model: str,
    ) -> None:
        self.token_tracker.switch_model(model)
        self.client = client
        self.provider = provider
        self.model = model

    async def run(self, user_task: str) -> AgentRun:
        run_steps: list[AgentStep] = []
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
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=1024,
                system=self.system_prompt,
                tools=self.registry.to_anthropic_schemas(),
                messages=self.messages,
            ) as stream:
                async for text in stream.text_stream:
                    print(text, end="", flush=True)
                    streamed_text = True
                response = await stream.get_final_message()

            if streamed_text:
                print()
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

            if response.stop_reason != "end_turn":
                for block in response.content:
                    if block.type == "tool_use":
                        tool_call = ToolCall(
                            name=block.name,
                            input=block.input,
                            tool_use_id=block.id,
                        )
                        tool_calls.append(tool_call)

                        print(format_tool_activity(tool_call))
                        output, is_error = self.registry.execute(
                            tool_call.name,
                            tool_call.input,
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
                verification = extract_verification_evidence(run_steps)
                return AgentRun(
                    objective=user_task,
                    steps=run_steps,
                    termination="completed",
                    final_stop_reason=response.stop_reason,
                    verification=verification,
                    task_success=infer_task_success(verification),
                )

            if not tool_results:
                verification = extract_verification_evidence(run_steps)
                print(f"Protocol error stop reason: {response.stop_reason}")
                return AgentRun(
                    objective=user_task,
                    steps=run_steps,
                    termination="protocol_error",
                    final_stop_reason=response.stop_reason,
                    verification=verification,
                    task_success=infer_task_success(verification),
                )

            self.messages.append(
                {
                    "role": "user",
                    "content": [result.to_anthropic_block() for result in tool_results],
                }
            )
        print(f"Agent reached the {self.max_steps}-step limit. Task stopped.")
        verification = extract_verification_evidence(run_steps)
        return AgentRun(
            objective=user_task,
            steps=run_steps,
            termination="max_steps",
            final_stop_reason=response.stop_reason,
            verification=verification,
            task_success=infer_task_success(verification),
        )
