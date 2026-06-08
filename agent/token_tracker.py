from dataclasses import dataclass
from types import TracebackType

from anthropic.types import Usage


@dataclass(frozen=True)
class ModelPricing:
    input_per_million: float
    output_per_million: float


MODEL_PRICING = {
    "claude-haiku-4-5": ModelPricing(
        input_per_million=1.0,
        output_per_million=5.0,
    ),
    "claude-haiku-4-5-20251001": ModelPricing(
        input_per_million=1.0,
        output_per_million=5.0,
    ),
}


class TokenTracker:
    def __init__(self, model: str = "claude-haiku-4-5") -> None:
        if model not in MODEL_PRICING:
            raise ValueError(f"No pricing configured for model: {model}")

        self.pricing = MODEL_PRICING[model]
        self.input_tokens = 0
        self.output_tokens = 0

    def __enter__(self) -> "TokenTracker":
        self.input_tokens = 0
        self.output_tokens = 0
        return self

    def add(self, usage: Usage) -> None:
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens

    @property
    def estimated_cost(self) -> float:
        input_cost = self.input_tokens * self.pricing.input_per_million
        output_cost = self.output_tokens * self.pricing.output_per_million
        return (input_cost + output_cost) / 1_000_000

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
            f"Total tokens: {total_tokens}\n"
            f"Estimated cost: ${self.estimated_cost:.6f}"
        )
