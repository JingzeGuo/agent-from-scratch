from anthropic import Anthropic

from .tool_registry import ToolRegistry


class Agent:
    def __init__(
        self,
        client: Anthropic,
        registry: ToolRegistry,
        model: str = "claude-haiku-4-5",
        max_steps: int = 10,
    ) -> None:
        self.client = client
        self.registry = registry
        self.model = model
        self.max_steps = max_steps
        self.messages = []

    def run(self, user_task: str) -> None:
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
            print(f"\n--- Step {step} ---")

            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                tools=self.registry.to_anthropic_schemas(),
                messages=self.messages,
            )

            self.messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                }
            )
            for block in response.content:
                if block.type == "text":
                    print(block.text)
            if response.stop_reason == "end_turn":
                task_completed = True
                break

            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    print(f"Tool: {block.name}")
                    print(f"Input: {block.input}")

                    output, is_error = self.registry.execute(
                        block.name,
                        block.input,
                    )
                    status = "failed" if is_error else "succeeded"
                    print(f"Status: {status}")
                    print(f"Output: {output}")

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": output,
                            "is_error": is_error,
                        }
                    )

            if not tool_results:
                print(f"Unexpected stop reason: {response.stop_reason}")
                task_completed = True
                break

            self.messages.append(
                {
                    "role": "user",
                    "content": tool_results,
                }
            )

        if not task_completed:
            print(f"Agent reached the {self.max_steps}-step limit. Task stopped.")
