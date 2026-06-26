from dataclasses import dataclass
from types import TracebackType

from .schemas import TokenUsage


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
    "deepseek-v4-flash": ModelPricing(
        input_per_million=0.14,
        output_per_million=0.28,
    ),
    "deepseek-v4-pro": ModelPricing(
        input_per_million=0.435,
        output_per_million=0.87,
    ),
    "deepseek-chat": ModelPricing(
        input_per_million=0.14,
        output_per_million=0.28,
    ),
    "deepseek-reasoner": ModelPricing(
        input_per_million=0.14,
        output_per_million=0.28,
    ),
    "gpt-4o-mini": ModelPricing(
        input_per_million=0.15,
        output_per_million=0.60,
    ),
}


class TokenTracker:
    def __init__(self, model: str = "claude-haiku-4-5") -> None:
        if model not in MODEL_PRICING:
            raise ValueError(f"No pricing configured for model: {model}")

        self.pricing = MODEL_PRICING[model]
        self.input_tokens = 0
        self.output_tokens = 0
        self._estimated_cost = 0.0

    def __enter__(self) -> "TokenTracker":
        self.input_tokens = 0
        self.output_tokens = 0
        self._estimated_cost = 0.0
        return self

    def add(self, usage: TokenUsage) -> None:
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        input_cost = usage.input_tokens * self.pricing.input_per_million
        output_cost = usage.output_tokens * self.pricing.output_per_million
        self._estimated_cost += (input_cost + output_cost) / 1_000_000

    def switch_model(self, model: str) -> None:
        if model not in MODEL_PRICING:
            raise ValueError(f"No pricing configured for model: {model}")
        self.pricing = MODEL_PRICING[model]

    @property
    def estimated_cost(self) -> float:
        return self._estimated_cost

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
