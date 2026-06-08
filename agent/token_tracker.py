from types import TracebackType

from anthropic.types import Usage


class TokenTracker:
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0

    def __enter__(self) -> "TokenTracker":
        self.input_tokens = 0
        self.output_tokens = 0
        return self

    def add(self, usage: Usage) -> None:
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        total_tokens = self.input_tokens + self.output_tokens
        print(
            f"Input tokens: {self.input_tokens}\n"
            f"Output tokens: {self.output_tokens}\n"
            f"Total tokens: {total_tokens}"
        )
