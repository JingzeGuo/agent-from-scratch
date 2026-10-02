import asyncio
import json
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from agent.consolidation import LLMConsolidator
from agent.schemas import ProviderResponse, TokenUsage
from tests.test_context import make_state


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
