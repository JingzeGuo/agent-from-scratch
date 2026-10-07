from dataclasses import dataclass

from ..schemas import TokenUsage


@dataclass(frozen=True)
class ModelPricing:
    input_per_million: float
    output_per_million: float


MODEL_PRICING = {
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
}


class TokenTracker:
    def __init__(self, model: str = "deepseek-v4-flash") -> None:
        if model not in MODEL_PRICING:
            raise ValueError(f"No pricing configured for model: {model}")

        self.pricing = MODEL_PRICING[model]
        self.input_tokens = 0
        self.output_tokens = 0
        self._estimated_cost = 0.0

    def add(self, usage: TokenUsage) -> None:
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        input_cost = usage.input_tokens * self.pricing.input_per_million
        output_cost = usage.output_tokens * self.pricing.output_per_million
        self._estimated_cost += (input_cost + output_cost) / 1_000_000

    @property
    def estimated_cost(self) -> float:
        return self._estimated_cost
