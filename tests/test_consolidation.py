import asyncio
import json
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from agent.consolidation import LLMConsolidator
from agent.context import ContextBuilder
from agent.schemas import ProviderResponse, TokenUsage, WorkingContextState
from tests.test_context import build, make_state, task


def response(text: str) -> ProviderResponse:
    return ProviderResponse(
        message={"role": "assistant", "content": text},
        text=[text],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


def test_consolidator_passes_state_prefix_objective_and_tracks_usage() -> None:
    provider = AsyncMock()
    provider.stream_response.return_value = response(
        make_state("wrong old task").model_dump_json()
    )
    usage: list[TokenUsage] = []
    consolidator = LLMConsolidator(provider, on_usage=usage.append)
    previous = make_state("Old")
    original = deepcopy(previous)
    prefix = [{"role": "user", "content": "Earlier task"}]
    state = asyncio.run(consolidator.consolidate(previous, prefix, "Active"))
    request = provider.stream_response.call_args.kwargs
    payload = json.loads(request["messages"][0]["content"])
    assert payload["previous_state"] == previous.model_dump()
    assert payload["history_prefix"] == prefix
    assert payload["current_objective"] == "Active"
    assert request["tools"] == []
    assert state.current_objective == "Active"
    assert previous == original
    assert len(usage) == 1


def test_invalid_structured_output_is_retried_once_and_validated() -> None:
    provider = AsyncMock()
    provider.stream_response.side_effect = [
        response('{"findings": "wrong type"}'),
        response(make_state().model_dump_json()),
    ]
    state = asyncio.run(LLMConsolidator(provider).consolidate(None, [], "Active"))
    assert state.current_objective == "Active"
    assert provider.stream_response.call_count == 2


def test_invalid_output_after_repair_fails() -> None:
    provider = AsyncMock()
    provider.stream_response.return_value = response("{}")
    with pytest.raises(ValueError, match="after repair"):
        asyncio.run(LLMConsolidator(provider).consolidate(None, [], "Active"))
    assert provider.stream_response.call_count == 2


def test_cancelled_consolidation_propagates() -> None:
    provider = AsyncMock()
    provider.stream_response.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(LLMConsolidator(provider).consolidate(None, [], "Active"))


@pytest.mark.parametrize("repair_succeeds", [True, False])
def test_oversized_state_retries_once_before_commit_or_fallback(
    repair_succeeds: bool,
) -> None:
    provider = AsyncMock()
    oversized = make_state("Wrong old objective", "detail " * 3000)
    corrected = make_state("Wrong old objective") if repair_succeeds else oversized
    provider.stream_response.side_effect = [
        response(oversized.model_dump_json()),
        response(corrected.model_dump_json()),
    ]
    usage: list[TokenUsage] = []
    builder = ContextBuilder(LLMConsolidator(provider, on_usage=usage.append))
    previous = WorkingContextState(
        consolidated_state=make_state("Old"), folded_message_count=2
    )
    builder.state = previous.model_copy(deep=True)
    messages = task("Old", 200) + task("Aged", 22000) + task("Active", 500)
    original = deepcopy(messages)
    result = build(builder, messages, objective="Active")
    assert provider.stream_response.call_count == 2
    assert len(usage) == 2
    assert (
        "Compress the state further"
        in provider.stream_response.call_args.kwargs["messages"][-1]["content"]
    )
    assert messages == original
    if repair_succeeds:
        assert not result.hard_collapsed
        assert builder.state.folded_message_count == 4
        assert builder.state.consolidated_state is not None
        assert builder.state.consolidated_state.current_objective == "Active"
        assert result.messages[1:] == messages[4:]
    else:
        assert result.hard_collapsed
        assert builder.state == previous
