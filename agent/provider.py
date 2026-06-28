import json
import os
from collections.abc import Callable
from typing import Any, Literal, Protocol, cast

import httpx
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolParam
from pydantic import BaseModel

from .schemas import (
    ProviderCapabilities,
    ProviderResponse,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolResult,
)

ProviderName = Literal["anthropic", "deepseek", "openai"]

DEFAULT_MODELS: dict[ProviderName, str] = {
    "anthropic": "claude-haiku-4-5",
    "deepseek": "deepseek-v4-flash",
    "openai": "gpt-4o-mini",
}

DEFAULT_BASE_URLS: dict[ProviderName, str | None] = {
    "anthropic": None,
    "deepseek": "https://api.deepseek.com",
    "openai": "https://api.openai.com/v1",
}


class ProviderConfig(BaseModel):
    provider: ProviderName
    model: str
    api_key: str
    base_url: str | None = None


class ProviderAdapter(Protocol):
    """Provider contract consumed by the agent controller."""

    provider: str
    model: str
    capabilities: ProviderCapabilities

    async def stream_response(
        self,
        *,
        system: str,
        tools: list[ToolDefinition],
        messages: list[dict[str, Any]],
        on_text_delta: Callable[[str], None] | None = None,
    ) -> ProviderResponse:
        """Return one normalized model response."""
        ...

    def tool_result_message(self, tool_results: list[ToolResult]) -> dict[str, Any]:
        """Build a provider-specific message carrying tool observations."""
        ...


class AnthropicProviderAdapter:
    """Adapter for Anthropic Messages API."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        client: AsyncAnthropic,
        capabilities: ProviderCapabilities | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.client = client
        self.capabilities = capabilities or ProviderCapabilities()

    async def stream_response(
        self,
        *,
        system: str,
        tools: list[ToolDefinition],
        messages: list[dict[str, Any]],
        on_text_delta: Callable[[str], None] | None = None,
    ) -> ProviderResponse:
        self._validate_request_capabilities(tools)
        text_blocks: list[str] = []
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=1024,
            system=system,
            tools=self._tool_params(tools),
            messages=cast(list[MessageParam], messages),
        ) as stream:
            async for text in stream.text_stream:
                if on_text_delta is not None:
                    on_text_delta(text)
            response = await stream.get_final_message()

        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_blocks.append(block.text)
            if block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        name=block.name,
                        input=block.input,
                        tool_use_id=block.id,
                    )
                )

        return ProviderResponse(
            message={
                "role": response.role,
                "content": response.content,
            },
            stop_reason=response.stop_reason,
            text=text_blocks,
            tool_calls=tool_calls,
            usage=TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            native_metadata={
                "id": response.id,
                "model": response.model,
                "provider": self.provider,
            },
        )

    def tool_result_message(self, tool_results: list[ToolResult]) -> dict[str, Any]:
        return {
            "role": "user",
            "content": [
                {
                    "type": result.type,
                    "tool_use_id": result.tool_use_id,
                    "content": result.content,
                    "is_error": result.is_error,
                }
                for result in tool_results
            ],
        }

    def _tool_params(self, tools: list[ToolDefinition]) -> list[ToolParam]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in tools
        ]

    def _validate_request_capabilities(self, tools: list[ToolDefinition]) -> None:
        if not self.capabilities.supports_streaming:
            raise ValueError(
                "Provider does not support streaming: "
                f"{self.provider}/{self.model}"
            )
        if tools and not self.capabilities.supports_tools:
            raise ValueError(
                "Provider does not support tools: "
                f"{self.provider}/{self.model}"
            )


class OpenAICompatibleProviderAdapter:
    """Adapter for OpenAI-compatible chat completions providers."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key: str,
        base_url: str,
        http_client: httpx.AsyncClient | None = None,
        capabilities: ProviderCapabilities | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.http_client = http_client
        self.capabilities = capabilities or ProviderCapabilities()

    async def stream_response(
        self,
        *,
        system: str,
        tools: list[ToolDefinition],
        messages: list[dict[str, Any]],
        on_text_delta: Callable[[str], None] | None = None,
    ) -> ProviderResponse:
        self._validate_request_capabilities(tools)
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [
                {"role": "system", "content": system},
                *self._openai_messages(messages),
            ],
        }
        if tools:
            payload["tools"] = self._openai_tools(tools)
            payload["tool_choice"] = "auto"
            if self.provider == "openai":
                payload["parallel_tool_calls"] = (
                    self.capabilities.supports_parallel_tool_calls
                )

        response_data = await self._stream_chat_completions(payload, on_text_delta)
        choice = response_data["choices"][0]
        message = choice["message"]
        finish_reason = choice.get("finish_reason")
        text = message.get("content") or ""

        tool_calls = self._tool_calls_from_message(message)
        return ProviderResponse(
            message=self._provider_neutral_assistant_message(text, tool_calls),
            stop_reason=self._normalize_finish_reason(finish_reason),
            text=[text] if text else [],
            tool_calls=tool_calls,
            usage=TokenUsage(
                input_tokens=response_data.get("usage", {}).get("prompt_tokens", 0),
                output_tokens=response_data.get("usage", {}).get(
                    "completion_tokens",
                    0,
                ),
            ),
            native_metadata={
                "id": response_data.get("id"),
                "model": response_data.get("model"),
                "provider": self.provider,
                "finish_reason": finish_reason,
            },
        )

    def tool_result_message(self, tool_results: list[ToolResult]) -> dict[str, Any]:
        return {
            "role": "user",
            "content": [
                {
                    "type": result.type,
                    "tool_use_id": result.tool_use_id,
                    "content": result.content,
                    "is_error": result.is_error,
                }
                for result in tool_results
            ],
        }

    async def _stream_chat_completions(
        self,
        payload: dict[str, Any],
        on_text_delta: Callable[[str], None] | None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.http_client is not None:
            async with self.http_client.stream(
                "POST",
                url,
                headers=headers,
                json=payload,
            ) as response:
                return await self._collect_stream(response, on_text_delta)

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                url,
                headers=headers,
                json=payload,
            ) as response:
                return await self._collect_stream(response, on_text_delta)

    async def _collect_stream(
        self,
        response: httpx.Response,
        on_text_delta: Callable[[str], None] | None,
    ) -> dict[str, Any]:
        response.raise_for_status()

        response_id: str | None = None
        response_model: str | None = None
        finish_reason: str | None = None
        usage: dict[str, Any] = {}
        text_parts: list[str] = []
        tool_calls_by_index: dict[int, dict[str, Any]] = {}

        async for line in response.aiter_lines():
            data = self._parse_stream_line(line)
            if data is None:
                continue
            if data == "[DONE]":
                break
            if not isinstance(data, dict):
                continue

            chunk = data
            response_id = response_id or chunk.get("id")
            response_model = response_model or chunk.get("model")
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]

            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta") or {}

            text_delta = delta.get("content")
            if isinstance(text_delta, str) and text_delta:
                text_parts.append(text_delta)
                if on_text_delta is not None:
                    on_text_delta(text_delta)

            self._accumulate_tool_call_deltas(
                tool_calls_by_index,
                delta.get("tool_calls") or [],
            )

        return {
            "id": response_id,
            "model": response_model,
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {
                        "role": "assistant",
                        "content": "".join(text_parts),
                        "tool_calls": [
                            tool_calls_by_index[index]
                            for index in sorted(tool_calls_by_index)
                        ],
                    },
                }
            ],
            "usage": usage,
        }

    def _parse_stream_line(self, line: str) -> dict[str, Any] | str | None:
        stripped = line.strip()
        if not stripped or not stripped.startswith("data:"):
            return None
        data = stripped.removeprefix("data:").strip()
        if data == "[DONE]":
            return data
        parsed = json.loads(data)
        if not isinstance(parsed, dict):
            raise ValueError("OpenAI-compatible stream chunk must be a JSON object.")
        return parsed

    def _accumulate_tool_call_deltas(
        self,
        tool_calls_by_index: dict[int, dict[str, Any]],
        tool_call_deltas: list[Any],
    ) -> None:
        for raw_tool_call in tool_call_deltas:
            if not isinstance(raw_tool_call, dict):
                continue
            index = raw_tool_call.get("index")
            if not isinstance(index, int):
                index = len(tool_calls_by_index)
            tool_call = tool_calls_by_index.setdefault(
                index,
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            if isinstance(raw_tool_call.get("id"), str):
                tool_call["id"] = raw_tool_call["id"]
            if isinstance(raw_tool_call.get("type"), str):
                tool_call["type"] = raw_tool_call["type"]

            function_delta = raw_tool_call.get("function") or {}
            if not isinstance(function_delta, dict):
                continue
            function = tool_call["function"]
            if isinstance(function_delta.get("name"), str):
                function["name"] = function_delta["name"]
            if isinstance(function_delta.get("arguments"), str):
                function["arguments"] += function_delta["arguments"]

    def _openai_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in tools
        ]

    def _validate_request_capabilities(self, tools: list[ToolDefinition]) -> None:
        if not self.capabilities.supports_streaming:
            raise ValueError(
                "Provider does not support streaming: "
                f"{self.provider}/{self.model}"
            )
        if tools and not self.capabilities.supports_tools:
            raise ValueError(
                "Provider does not support tools: "
                f"{self.provider}/{self.model}"
            )

    def _openai_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role == "user" and isinstance(content, list):
                converted.extend(self._openai_tool_result_messages(content))
                continue
            if role == "assistant" and isinstance(content, list):
                converted.append(self._openai_assistant_message(content))
                continue
            converted.append({"role": role, "content": content})
        return converted

    def _openai_tool_result_messages(
        self,
        content: list[Any],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for block in content:
            block_type = self._block_value(block, "type")
            if block_type != "tool_result":
                continue
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": self._block_value(block, "tool_use_id"),
                    "content": str(self._block_value(block, "content") or ""),
                }
            )
        return messages

    def _openai_assistant_message(self, content: list[Any]) -> dict[str, Any]:
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in content:
            block_type = self._block_value(block, "type")
            if block_type == "text":
                text = self._block_value(block, "text")
                if isinstance(text, str):
                    text_parts.append(text)
            if block_type == "tool_use":
                tool_input = self._block_value(block, "input")
                tool_calls.append(
                    {
                        "id": self._block_value(block, "id"),
                        "type": "function",
                        "function": {
                            "name": self._block_value(block, "name"),
                            "arguments": json.dumps(tool_input or {}),
                        },
                    }
                )

        message: dict[str, Any] = {
            "role": "assistant",
            "content": "\n".join(text_parts) if text_parts else None,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message

    def _tool_calls_from_message(self, message: dict[str, Any]) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []
        for raw_tool_call in message.get("tool_calls") or []:
            function = raw_tool_call.get("function") or {}
            tool_calls.append(
                ToolCall(
                    name=function.get("name", ""),
                    input=self._parse_tool_arguments(function.get("arguments")),
                    tool_use_id=raw_tool_call.get("id", ""),
                )
            )
        return tool_calls

    def _provider_neutral_assistant_message(
        self,
        text: str,
        tool_calls: list[ToolCall],
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        if text:
            content.append({"type": "text", "text": text})
        for tool_call in tool_calls:
            content.append(
                {
                    "type": "tool_use",
                    "id": tool_call.tool_use_id,
                    "name": tool_call.name,
                    "input": tool_call.input,
                }
            )
        return {"role": "assistant", "content": content}

    def _parse_tool_arguments(self, arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if not isinstance(arguments, str):
            return {}
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        if not isinstance(parsed, dict):
            return {}
        return parsed

    def _normalize_finish_reason(self, finish_reason: str | None) -> str | None:
        if finish_reason == "stop":
            return "end_turn"
        if finish_reason == "tool_calls":
            return "tool_use"
        return finish_reason

    def _block_value(self, block: Any, key: str) -> Any:
        if isinstance(block, dict):
            return block.get(key)
        return getattr(block, key, None)


def load_provider_config(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> ProviderConfig:
    provider_name = provider or os.getenv("AGENT_PROVIDER", "anthropic")
    if provider_name not in DEFAULT_MODELS:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Available: {list(DEFAULT_MODELS)}"
        )

    typed_provider: ProviderName = provider_name
    prefix = typed_provider.upper()
    configured_api_key = api_key or os.getenv(f"{prefix}_API_KEY", "")
    if not configured_api_key:
        raise ValueError(f"{prefix}_API_KEY is not set")

    configured_model = (
        model
        or os.getenv(f"{prefix}_MODEL")
        or DEFAULT_MODELS[typed_provider]
    )
    base_url = (
        os.getenv(f"{prefix}_BASE_URL")
        or DEFAULT_BASE_URLS[typed_provider]
    )
    return ProviderConfig(
        provider=typed_provider,
        model=configured_model,
        api_key=configured_api_key,
        base_url=base_url,
    )


def create_client(config: ProviderConfig) -> AsyncAnthropic:
    return AsyncAnthropic(
        api_key=config.api_key,
        base_url=config.base_url,
    )


def create_provider_adapter(config: ProviderConfig) -> ProviderAdapter:
    if config.provider in {"deepseek", "openai"}:
        if config.base_url is None:
            raise ValueError(f"{config.provider.upper()}_BASE_URL is not configured")
        return OpenAICompatibleProviderAdapter(
            provider=config.provider,
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
        )
    return AnthropicProviderAdapter(
        provider=config.provider,
        model=config.model,
        client=create_client(config),
    )
