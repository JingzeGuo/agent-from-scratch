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
    assert capsys.readouterr().out == (
        "Input tokens: 150\nOutput tokens: 30\nTotal tokens: 180\n"
    )
