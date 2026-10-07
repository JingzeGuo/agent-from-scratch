import pytest

from agent.schemas import TokenUsage
from agent.state.token_tracker import TokenTracker


def test_token_tracker_accumulates_usage() -> None:
    tracker = TokenTracker()
    tracker.add(TokenUsage(input_tokens=100, output_tokens=20))
    tracker.add(TokenUsage(input_tokens=50, output_tokens=10))

    assert tracker.input_tokens == 150
    assert tracker.output_tokens == 30
    assert tracker.estimated_cost == pytest.approx(0.0000294)


def test_token_tracker_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="No pricing configured"):
        TokenTracker(model="unknown-model")
