import asyncio
import json
from typing import Any, cast

import httpx
import pytest

from agent.provider import (
    DeepSeekProvider,
    ProviderRequestError,
    format_provider_request_error,
    load_deepseek_config,
)
from agent.schemas import ProviderCapabilities, ToolDefinition, ToolResult


class FakeHttpClient:
    def __init__(self, stream_chunks: list[dict[str, Any]]) -> None:
        self.stream_chunks = stream_chunks
        self.requests: list[dict[str, Any]] = []

    def stream(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> "FakeStreamManager":
        self.requests.append(
            {"method": method, "url": url, "headers": headers, "json": json}
        )
        return FakeStreamManager(self.stream_chunks, url)


class FakeStreamManager:
    def __init__(self, stream_chunks: list[dict[str, Any]], url: str) -> None:
        self.response = FakeStreamResponse(stream_chunks, url)

    async def __aenter__(self) -> "FakeStreamResponse":
        return self.response

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        return None


class FakeStreamResponse:
    def __init__(self, stream_chunks: list[dict[str, Any]], url: str) -> None:
        self.stream_chunks = stream_chunks
        self.request = httpx.Request("POST", url)

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self) -> Any:
        for chunk in self.stream_chunks:
            yield f"data: {json.dumps(chunk)}"
        yield "data: [DONE]"


class FailingHttpClient:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def stream(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> "FailingStreamManager":
        del method, url, headers, json
        return FailingStreamManager(self.error)


class FailingStreamManager:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def __aenter__(self) -> FakeStreamResponse:
        raise self.error

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        return None


class BadJsonHttpClient:
    def stream(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> "BadJsonStreamManager":
        del method, headers, json
        return BadJsonStreamManager(url)


class BadJsonStreamManager:
    def __init__(self, url: str) -> None:
        self.response = BadJsonStreamResponse(url)

    async def __aenter__(self) -> "BadJsonStreamResponse":
        return self.response

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        return None


class BadJsonStreamResponse:
    def __init__(self, url: str) -> None:
        self.request = httpx.Request("POST", url)

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self) -> Any:
        yield "data: {not-json"


def make_provider(
    http_client: object,
    *,
    capabilities: ProviderCapabilities | None = None,
) -> DeepSeekProvider:
    return DeepSeekProvider(
        model="deepseek-v4-flash",
        api_key="deepseek-key",
        base_url="https://api.deepseek.com/",
        http_client=cast(httpx.AsyncClient, http_client),
        capabilities=capabilities,
    )


def final_text_chunks(text: str = "Done.") -> list[dict[str, Any]]:
    return [
        {
            "id": "deepseek_test",
            "model": "deepseek-v4-flash",
            "choices": [
                {
                    "delta": {"role": "assistant", "content": text},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "deepseek_test",
            "model": "deepseek-v4-flash",
            "choices": [{"delta": {}, "finish_reason": "stop"}],
        },
        {
            "id": "deepseek_test",
            "model": "deepseek-v4-flash",
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        },
    ]


def test_loads_deepseek_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)

    config = load_deepseek_config()

    assert config.model == "deepseek-v4-flash"
    assert config.api_key == "deepseek-key"
    assert config.base_url == "https://api.deepseek.com"


def test_deepseek_config_uses_explicit_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "env-model")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.test")

    config = load_deepseek_config(model="cli-model", api_key="cli-key")

    assert config.model == "cli-model"
    assert config.api_key == "cli-key"
    assert config.base_url == "https://example.test"


def test_deepseek_config_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY is not set"):
        load_deepseek_config()


def test_deepseek_provider_normalizes_streamed_tool_response() -> None:
    fake_client = FakeHttpClient(
        [
            {
                "id": "deepseek_test",
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "delta": {"role": "assistant", "content": "I will "},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "deepseek_test",
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "delta": {
                            "content": "calculate.",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_calc",
                                    "type": "function",
                                    "function": {
                                        "name": "calculator",
                                        "arguments": '{"expression": ',
                                    },
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "deepseek_test",
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '"1 + 1"}'},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
            {
                "id": "deepseek_test",
                "model": "deepseek-v4-flash",
                "choices": [],
                "usage": {"prompt_tokens": 12, "completion_tokens": 6},
            },
        ]
    )
    provider = make_provider(fake_client)
    streamed: list[str] = []

    response = asyncio.run(
        provider.stream_response(
            system="system prompt",
            tools=[
                ToolDefinition(
                    name="calculator",
                    description="Calculate.",
                    input_schema={"type": "object"},
                )
            ],
            messages=[{"role": "user", "content": "Calculate 1 + 1"}],
            on_text_delta=streamed.append,
        )
    )

    assert streamed == ["I will ", "calculate."]
    assert response.stop_reason == "tool_use"
    assert response.text == ["I will calculate."]
    assert response.tool_calls[0].model_dump() == {
        "name": "calculator",
        "input": {"expression": "1 + 1"},
        "tool_use_id": "call_calc",
    }
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 6
    request = fake_client.requests[0]
    assert request["url"] == "https://api.deepseek.com/chat/completions"
    assert request["json"]["stream"] is True
    assert request["json"]["stream_options"] == {"include_usage": True}
    assert "parallel_tool_calls" not in request["json"]


def test_deepseek_provider_maps_stop_to_end_turn() -> None:
    provider = make_provider(FakeHttpClient(final_text_chunks()))

    response = asyncio.run(
        provider.stream_response(
            system="system prompt",
            tools=[],
            messages=[{"role": "user", "content": "Say done"}],
        )
    )

    assert response.stop_reason == "end_turn"
    assert response.text == ["Done."]


def test_deepseek_provider_builds_tool_result_message() -> None:
    provider = make_provider(FakeHttpClient([]))

    message = provider.tool_result_message(
        [ToolResult(tool_use_id="call_calc", content="2", is_error=False)]
    )

    assert message == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "call_calc",
                "content": "2",
                "is_error": False,
            }
        ],
    }


def test_deepseek_provider_converts_tool_result_history() -> None:
    fake_client = FakeHttpClient(final_text_chunks("The answer is 2."))
    provider = make_provider(fake_client)

    asyncio.run(
        provider.stream_response(
            system="system prompt",
            tools=[],
            messages=[
                {"role": "user", "content": "Calculate 1 + 1"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_calc",
                            "name": "calculator",
                            "input": {"expression": "1 + 1"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_calc",
                            "content": "2",
                            "is_error": False,
                        }
                    ],
                },
            ],
        )
    )

    request_messages = fake_client.requests[0]["json"]["messages"]
    assert request_messages[-2]["role"] == "assistant"
    assert request_messages[-2]["tool_calls"][0]["id"] == "call_calc"
    assert request_messages[-1] == {
        "role": "tool",
        "tool_call_id": "call_calc",
        "content": "2",
    }


def test_deepseek_provider_rejects_unsupported_capabilities() -> None:
    provider = make_provider(
        FakeHttpClient([]),
        capabilities=ProviderCapabilities(supports_tools=False),
    )

    with pytest.raises(ValueError, match="does not support tools"):
        asyncio.run(
            provider.stream_response(
                system="system prompt",
                tools=[
                    ToolDefinition(
                        name="calculator",
                        description="Calculate.",
                        input_schema={"type": "object"},
                    )
                ],
                messages=[{"role": "user", "content": "Calculate"}],
            )
        )


def test_deepseek_provider_wraps_network_errors() -> None:
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    provider = make_provider(
        FailingHttpClient(httpx.ConnectError("DNS failure", request=request))
    )

    with pytest.raises(ProviderRequestError) as error:
        asyncio.run(
            provider.stream_response(
                system="system prompt",
                tools=[],
                messages=[{"role": "user", "content": "Hello"}],
            )
        )

    assert error.value.kind == "network"
    assert error.value.provider == "deepseek"
    assert "network connection failed" in format_provider_request_error(error.value)


def test_deepseek_provider_wraps_rate_limit_errors() -> None:
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    response = httpx.Response(429, request=request, text='{"error":"rate limit"}')
    provider = make_provider(
        FailingHttpClient(
            httpx.HTTPStatusError(
                "rate limited",
                request=request,
                response=response,
            )
        )
    )

    with pytest.raises(ProviderRequestError) as error:
        asyncio.run(
            provider.stream_response(
                system="system prompt",
                tools=[],
                messages=[{"role": "user", "content": "Hello"}],
            )
        )

    assert error.value.kind == "rate_limit"
    assert "rate limit" in format_provider_request_error(error.value)


def test_deepseek_provider_wraps_bad_stream_json() -> None:
    provider = make_provider(BadJsonHttpClient())

    with pytest.raises(ProviderRequestError) as error:
        asyncio.run(
            provider.stream_response(
                system="system prompt",
                tools=[],
                messages=[{"role": "user", "content": "Hello"}],
            )
        )

    assert error.value.kind == "response_format"
    assert "unexpected response format" in format_provider_request_error(error.value)
