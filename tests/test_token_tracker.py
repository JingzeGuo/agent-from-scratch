import pytest
from anthropic.types import Usage

from agent.token_tracker import TokenTracker


def test_token_tracker_accumulates_usage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with TokenTracker() as tracker:
        tracker.add(Usage(input_tokens=100, output_tokens=20))
        tracker.add(Usage(input_tokens=50, output_tokens=10))

    assert tracker.input_tokens == 150
    assert tracker.output_tokens == 30
    assert tracker.estimated_cost == pytest.approx(0.0003)
    assert capsys.readouterr().out == (
        "Input tokens: 150\n"
        "Output tokens: 30\n"
        "Total tokens: 180\n"
        "Estimated cost: $0.000300\n"
    )


def test_token_tracker_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="No pricing configured"):
        TokenTracker(model="unknown-model")


def test_token_tracker_preserves_cost_across_model_switch() -> None:
    tracker = TokenTracker()
    tracker.add(Usage(input_tokens=100, output_tokens=20))

    tracker.switch_model("deepseek-v4-flash")
    tracker.add(Usage(input_tokens=100, output_tokens=20))

    assert tracker.input_tokens == 200
    assert tracker.output_tokens == 40
    assert tracker.estimated_cost == pytest.approx(0.0002196)
