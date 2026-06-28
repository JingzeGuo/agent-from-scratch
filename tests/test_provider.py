import asyncio
import json
from collections.abc import Sequence
from typing import Any, cast

import httpx
import pytest
from anthropic import AsyncAnthropic
from anthropic.types import (
    ContentBlock,
    Message,
    TextBlock,
    ToolUseBlock,
    Usage,
)

from agent.provider import (
    AnthropicProviderAdapter,
    OpenAICompatibleProviderAdapter,
    load_provider_config,
)
from agent.schemas import ProviderCapabilities, ToolDefinition, ToolResult


class FakeMessages:
    def __init__(self, response: Message) -> None:
        self.response = response
        self.request: dict[str, Any] | None = None

    def stream(self, **kwargs: Any) -> "FakeStreamManager":
        self.request = kwargs
        return FakeStreamManager(self.response)


class FakeStreamManager:
    def __init__(self, response: Message) -> None:
        self.response = response
        self.text_stream = self._stream_text()

    async def __aenter__(self) -> "FakeStreamManager":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        return None

    async def _stream_text(self) -> Any:
        for block in self.response.content:
            if block.type == "text":
                yield block.text

    async def get_final_message(self) -> Message:
        return self.response


class FakeClient:
    def __init__(self, response: Message) -> None:
        self.messages = FakeMessages(response)


class FakeOpenAIHttpClient:
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
    ) -> "FakeOpenAIStreamManager":
        self.requests.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "json": json,
            }
        )
        return FakeOpenAIStreamManager(self.stream_chunks, url)


class FakeOpenAIStreamManager:
    def __init__(self, stream_chunks: list[dict[str, Any]], url: str) -> None:
        self.response = FakeOpenAIStreamResponse(stream_chunks, url)

    async def __aenter__(self) -> "FakeOpenAIStreamResponse":
        return self.response

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        return None


class FakeOpenAIStreamResponse:
    def __init__(self, stream_chunks: list[dict[str, Any]], url: str) -> None:
        self.stream_chunks = stream_chunks
        self.request = httpx.Request("POST", url)

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self) -> Any:
        for chunk in self.stream_chunks:
            yield f"data: {json_dumps(chunk)}"
        yield "data: [DONE]"


def json_dumps(value: dict[str, Any]) -> str:
    return json.dumps(value)


def make_message(
    content: Sequence[ContentBlock],
    stop_reason: str,
) -> Message:
    return Message(
        id="msg_provider_test",
        type="message",
        role="assistant",
        model="claude-haiku-4-5",
        content=list(content),
        stop_reason=cast(Any, stop_reason),
        stop_sequence=None,
        usage=Usage(input_tokens=12, output_tokens=6),
    )


def test_loads_anthropic_provider_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.delenv("AGENT_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    config = load_provider_config()

    assert config.provider == "anthropic"
    assert config.model == "claude-haiku-4-5"
    assert config.api_key == "anthropic-key"
    assert config.base_url is None


def test_loads_deepseek_provider_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)

    config = load_provider_config()

    assert config.provider == "deepseek"
    assert config.model == "deepseek-v4-flash"
    assert config.api_key == "deepseek-key"
    assert config.base_url == "https://api.deepseek.com/anthropic"


def test_loads_openai_provider_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    config = load_provider_config()

    assert config.provider == "openai"
    assert config.model == "gpt-4o-mini"
    assert config.api_key == "openai-key"
    assert config.base_url == "https://api.openai.com/v1"


def test_provider_config_uses_cli_api_key_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")

    config = load_provider_config(api_key="cli-key")

    assert config.api_key == "cli-key"


def test_provider_config_requires_matching_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY is not set"):
        load_provider_config(provider="deepseek")


def test_anthropic_adapter_normalizes_tool_response() -> None:
    message = make_message(
        content=[
            TextBlock(text="I will calculate it.", type="text"),
            ToolUseBlock(
                id="toolu_calc",
                name="calculator",
                input={"expression": "1 + 1"},
                type="tool_use",
            ),
        ],
        stop_reason="tool_use",
    )
    fake_client = FakeClient(message)
    adapter = AnthropicProviderAdapter(
        provider="anthropic",
        model="claude-haiku-4-5",
        client=cast(AsyncAnthropic, fake_client),
    )
    streamed: list[str] = []

    response = asyncio.run(
        adapter.stream_response(
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

    assert streamed == ["I will calculate it."]
    assert response.stop_reason == "tool_use"
    assert response.text == ["I will calculate it."]
    assert response.tool_calls[0].tool_use_id == "toolu_calc"
    assert response.tool_calls[0].name == "calculator"
    assert response.tool_calls[0].input == {"expression": "1 + 1"}
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 6
    assert fake_client.messages.request is not None
    assert fake_client.messages.request["tools"][0]["name"] == "calculator"


def test_anthropic_adapter_builds_tool_result_message() -> None:
    adapter = AnthropicProviderAdapter(
        provider="anthropic",
        model="claude-haiku-4-5",
        client=cast(AsyncAnthropic, object()),
    )

    message = adapter.tool_result_message(
        [
            ToolResult(
                tool_use_id="toolu_calc",
                content="2",
                is_error=False,
            )
        ]
    )

    assert message == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_calc",
                "content": "2",
                "is_error": False,
            }
        ],
    }


def test_anthropic_adapter_rejects_tools_when_capability_is_disabled() -> None:
    message = make_message(
        content=[TextBlock(text="Done.", type="text")],
        stop_reason="end_turn",
    )
    adapter = AnthropicProviderAdapter(
        provider="anthropic",
        model="claude-haiku-4-5",
        client=cast(AsyncAnthropic, FakeClient(message)),
        capabilities=ProviderCapabilities(supports_tools=False),
    )

    with pytest.raises(ValueError, match="does not support tools"):
        asyncio.run(
            adapter.stream_response(
                system="system prompt",
                tools=[
                    ToolDefinition(
                        name="calculator",
                        description="Calculate.",
                        input_schema={"type": "object"},
                    )
                ],
                messages=[{"role": "user", "content": "Calculate 1 + 1"}],
            )
        )


def test_openai_adapter_normalizes_tool_response() -> None:
    fake_client = FakeOpenAIHttpClient(
        [
            {
                "id": "chatcmpl_test",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "delta": {"role": "assistant", "content": "I will "},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl_test",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "delta": {"content": "calculate it."},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl_test",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "delta": {
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
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl_test",
                "model": "gpt-4o-mini",
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
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl_test",
                "model": "gpt-4o-mini",
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
            },
            {
                "id": "chatcmpl_test",
                "model": "gpt-4o-mini",
                "choices": [],
                "usage": {"prompt_tokens": 12, "completion_tokens": 6},
            },
        ]
    )
    adapter = OpenAICompatibleProviderAdapter(
        provider="openai",
        model="gpt-4o-mini",
        api_key="openai-key",
        base_url="https://api.openai.com/v1",
        http_client=cast(httpx.AsyncClient, fake_client),
    )
    streamed: list[str] = []

    response = asyncio.run(
        adapter.stream_response(
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

    assert streamed == ["I will ", "calculate it."]
    assert response.stop_reason == "tool_use"
    assert response.text == ["I will calculate it."]
    assert response.tool_calls[0].tool_use_id == "call_calc"
    assert response.tool_calls[0].name == "calculator"
    assert response.tool_calls[0].input == {"expression": "1 + 1"}
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 6
    assert response.message == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "I will calculate it."},
            {
                "type": "tool_use",
                "id": "call_calc",
                "name": "calculator",
                "input": {"expression": "1 + 1"},
            },
        ],
    }
    assert fake_client.requests[0]["url"] == (
        "https://api.openai.com/v1/chat/completions"
    )
    assert fake_client.requests[0]["json"]["stream"] is True
    assert fake_client.requests[0]["json"]["stream_options"] == {"include_usage": True}
    assert fake_client.requests[0]["json"]["parallel_tool_calls"] is True
    assert fake_client.requests[0]["json"]["tools"][0] == {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate.",
            "parameters": {"type": "object"},
        },
    }


def test_openai_adapter_maps_stop_finish_reason_to_end_turn() -> None:
    fake_client = FakeOpenAIHttpClient(
        [
            {
                "id": "chatcmpl_test",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "delta": {"role": "assistant", "content": "Do"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl_test",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "delta": {"content": "ne."},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl_test",
                "model": "gpt-4o-mini",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
            },
            {
                "id": "chatcmpl_test",
                "model": "gpt-4o-mini",
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
        ]
    )
    adapter = OpenAICompatibleProviderAdapter(
        provider="openai",
        model="gpt-4o-mini",
        api_key="openai-key",
        base_url="https://api.openai.com/v1/",
        http_client=cast(httpx.AsyncClient, fake_client),
    )

    response = asyncio.run(
        adapter.stream_response(
            system="system prompt",
            tools=[],
            messages=[{"role": "user", "content": "Say done"}],
        )
    )

    assert response.stop_reason == "end_turn"
    assert response.text == ["Done."]
    assert fake_client.requests[0]["url"] == (
        "https://api.openai.com/v1/chat/completions"
    )
    assert "tools" not in fake_client.requests[0]["json"]


def test_openai_adapter_sends_parallel_tool_call_capability() -> None:
    fake_client = FakeOpenAIHttpClient(
        [
            {
                "id": "chatcmpl_test",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "delta": {"role": "assistant", "content": "Done."},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl_test",
                "model": "gpt-4o-mini",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
            },
            {
                "id": "chatcmpl_test",
                "model": "gpt-4o-mini",
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
        ]
    )
    adapter = OpenAICompatibleProviderAdapter(
        provider="openai",
        model="gpt-4o-mini",
        api_key="openai-key",
        base_url="https://api.openai.com/v1",
        http_client=cast(httpx.AsyncClient, fake_client),
        capabilities=ProviderCapabilities(supports_parallel_tool_calls=False),
    )

    asyncio.run(
        adapter.stream_response(
            system="system prompt",
            tools=[
                ToolDefinition(
                    name="calculator",
                    description="Calculate.",
                    input_schema={"type": "object"},
                )
            ],
            messages=[{"role": "user", "content": "Calculate 1 + 1"}],
        )
    )

    assert fake_client.requests[0]["json"]["parallel_tool_calls"] is False


def test_openai_adapter_rejects_tools_when_capability_is_disabled() -> None:
    adapter = OpenAICompatibleProviderAdapter(
        provider="openai",
        model="gpt-4o-mini",
        api_key="openai-key",
        base_url="https://api.openai.com/v1",
        http_client=cast(httpx.AsyncClient, FakeOpenAIHttpClient([])),
        capabilities=ProviderCapabilities(supports_tools=False),
    )

    with pytest.raises(ValueError, match="does not support tools"):
        asyncio.run(
            adapter.stream_response(
                system="system prompt",
                tools=[
                    ToolDefinition(
                        name="calculator",
                        description="Calculate.",
                        input_schema={"type": "object"},
                    )
                ],
                messages=[{"role": "user", "content": "Calculate 1 + 1"}],
            )
        )


def test_openai_adapter_converts_tool_result_history() -> None:
    fake_client = FakeOpenAIHttpClient(
        [
            {
                "id": "chatcmpl_test",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "content": "The answer is 2.",
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl_test",
                "model": "gpt-4o-mini",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
            },
            {
                "id": "chatcmpl_test",
                "model": "gpt-4o-mini",
                "choices": [],
                "usage": {"prompt_tokens": 20, "completion_tokens": 5},
            },
        ]
    )
    adapter = OpenAICompatibleProviderAdapter(
        provider="openai",
        model="gpt-4o-mini",
        api_key="openai-key",
        base_url="https://api.openai.com/v1",
        http_client=cast(httpx.AsyncClient, fake_client),
    )

    asyncio.run(
        adapter.stream_response(
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
    assert request_messages[-2] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_calc",
                "type": "function",
                "function": {
                    "name": "calculator",
                    "arguments": '{"expression": "1 + 1"}',
                },
            }
        ],
    }
    assert request_messages[-1] == {
        "role": "tool",
        "tool_call_id": "call_calc",
        "content": "2",
    }
