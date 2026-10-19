import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel

from .schemas import (
    ProviderResponse,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolResult,
)

ProviderErrorKind = Literal[
    "authentication",
    "rate_limit",
    "timeout",
    "network",
    "provider_status",
    "response_format",
    "request",
]

DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekConfig(BaseModel):
    model: str
    api_key: str
    base_url: str


class ProviderRequestError(RuntimeError):
    """User-facing provider request failure with a classified cause."""

    def __init__(
        self,
        *,
        kind: ProviderErrorKind,
        provider: str,
        model: str,
        detail: str,
    ) -> None:
        self.kind = kind
        self.provider = provider
        self.model = model
        self.detail = detail
        super().__init__(detail)


class ProviderAdapter(Protocol):
    """Provider contract consumed by the agent controller."""

    provider: str
    model: str

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


async def ask_json[T: BaseModel](
    provider: ProviderAdapter,
    *,
    system: str,
    messages: list[dict[str, Any]],
    schema: type[T],
    on_usage: Callable[[TokenUsage], None] | None = None,
) -> T:
    """Make one tool-free request whose reply must be JSON matching schema.

    Raises ValueError (including pydantic's ValidationError) on a bad reply.
    """
    response = await provider.stream_response(system=system, tools=[], messages=messages)
    if on_usage:
        on_usage(response.usage)
    if response.tool_calls or response.stop_reason != "end_turn":
        raise ValueError("Model did not return a complete JSON answer")
    text = "\n".join(response.text).strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    return schema.model_validate_json(text)


def format_provider_request_error(error: ProviderRequestError) -> str:
    """Return a concise, actionable CLI message for provider failures."""
    suggestions = {
        "authentication": (
            "Please check your API key and provider permissions, then retry."
        ),
        "rate_limit": "Please wait and retry, or reduce request volume.",
        "timeout": "Please check your network, base URL, or proxy, then retry.",
        "network": (
            "Please check your base URL, DNS/network access, or proxy, then retry."
        ),
        "provider_status": "Please retry later.",
        "response_format": (
            "Please retry; the provider returned an unexpected response shape."
        ),
        "request": (
            "Please check your provider configuration, base URL, and model name."
        ),
    }
    return (
        f"Provider request failed: {error.detail}\n"
        f"Provider: {error.provider}/{error.model}\n"
        f"{suggestions[error.kind]}\n"
        "Set AGENT_DEBUG=1 to print the underlying traceback."
    )


@dataclass
class _StreamedCompletion:
    """One chat completion assembled from streamed chunks."""

    id: str | None = None
    model: str | None = None
    finish_reason: str | None = None
    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)


class DeepSeekProvider:
    """Thin client for DeepSeek's Chat Completions API."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.provider = "deepseek"
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.http_client = http_client

    async def aclose(self) -> None:
        """Close the HTTP client shared by every request from this provider."""
        if self.http_client is not None:
            await self.http_client.aclose()

    async def stream_response(
        self,
        *,
        system: str,
        tools: list[ToolDefinition],
        messages: list[dict[str, Any]],
        on_text_delta: Callable[[str], None] | None = None,
    ) -> ProviderResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [
                {"role": "system", "content": system},
                *self._deepseek_messages(messages),
            ],
        }
        if tools:
            payload["tools"] = self._deepseek_tools(tools)
            payload["tool_choice"] = "auto"

        try:
            completion = await self._stream_chat_completions(payload, on_text_delta)
        except Exception as e:
            raise classify_provider_exception(
                e,
                provider=self.provider,
                model=self.model,
            ) from e
        tool_calls = [
            ToolCall(
                name=raw_tool_call["function"]["name"],
                input=self._parse_tool_arguments(raw_tool_call["function"]["arguments"]),
                tool_use_id=raw_tool_call["id"],
            )
            for raw_tool_call in completion.tool_calls
        ]
        return ProviderResponse(
            message=self._provider_neutral_assistant_message(completion.text, tool_calls),
            stop_reason=self._normalize_finish_reason(completion.finish_reason),
            text=[completion.text] if completion.text else [],
            tool_calls=tool_calls,
            usage=TokenUsage(
                input_tokens=completion.usage.get("prompt_tokens", 0),
                output_tokens=completion.usage.get("completion_tokens", 0),
            ),
            native_metadata={
                "id": completion.id,
                "model": completion.model,
                "provider": self.provider,
                "finish_reason": completion.finish_reason,
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
    ) -> _StreamedCompletion:
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(timeout=60.0)
        async with self.http_client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
        ) as response:
            return await self._collect_stream(response, on_text_delta)

    async def _collect_stream(
        self,
        response: httpx.Response,
        on_text_delta: Callable[[str], None] | None,
    ) -> _StreamedCompletion:
        response.raise_for_status()

        completion = _StreamedCompletion()
        text_parts: list[str] = []
        tool_calls_by_index: dict[int, dict[str, Any]] = {}

        async for line in response.aiter_lines():
            stripped = line.strip()
            if not stripped.startswith("data:"):
                continue
            data = stripped.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if not isinstance(chunk, dict):
                raise ValueError("DeepSeek stream chunk must be a JSON object.")

            completion.id = completion.id or chunk.get("id")
            completion.model = completion.model or chunk.get("model")
            if isinstance(chunk.get("usage"), dict):
                completion.usage = chunk["usage"]

            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            completion.finish_reason = choice.get("finish_reason") or completion.finish_reason
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

        completion.text = "".join(text_parts)
        completion.tool_calls = [
            tool_calls_by_index[index] for index in sorted(tool_calls_by_index)
        ]
        return completion

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

    def _deepseek_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
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

    def _deepseek_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role == "user" and isinstance(content, list):
                converted.extend(self._deepseek_tool_result_messages(content))
                continue
            if role == "assistant" and isinstance(content, list):
                converted.append(self._deepseek_assistant_message(content))
                continue
            converted.append({"role": role, "content": content})
        return converted

    def _deepseek_tool_result_messages(
        self,
        content: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "role": "tool",
                "tool_call_id": block.get("tool_use_id"),
                "content": str(block.get("content") or ""),
            }
            for block in content
            if block.get("type") == "tool_result"
        ]

    def _deepseek_assistant_message(
        self,
        content: list[dict[str, Any]],
    ) -> dict[str, Any]:
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in content:
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
            if block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id"),
                        "type": "function",
                        "function": {
                            "name": block.get("name"),
                            "arguments": json.dumps(block.get("input") or {}),
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

    def _parse_tool_arguments(self, arguments: str) -> dict[str, Any]:
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _normalize_finish_reason(self, finish_reason: str | None) -> str | None:
        if finish_reason == "stop":
            return "end_turn"
        if finish_reason == "tool_calls":
            return "tool_use"
        return finish_reason


def load_deepseek_config(
    model: str | None = None,
    api_key: str | None = None,
) -> DeepSeekConfig:
    configured_api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
    if not configured_api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set")

    configured_model = (
        model
        or os.getenv("DEEPSEEK_MODEL")
        or DEFAULT_DEEPSEEK_MODEL
    )
    configured_base_url = (
        os.getenv("DEEPSEEK_BASE_URL")
        or DEFAULT_DEEPSEEK_BASE_URL
    )
    return DeepSeekConfig(
        model=configured_model,
        api_key=configured_api_key,
        base_url=configured_base_url,
    )


def classify_provider_exception(
    error: Exception,
    *,
    provider: str,
    model: str,
) -> ProviderRequestError:
    if isinstance(error, ProviderRequestError):
        return error
    if isinstance(error, httpx.TimeoutException):
        return ProviderRequestError(
            kind="timeout",
            provider=provider,
            model=model,
            detail="network timeout while contacting the provider",
        )
    if isinstance(error, httpx.NetworkError):
        return ProviderRequestError(
            kind="network",
            provider=provider,
            model=model,
            detail="network connection failed while contacting the provider",
        )
    if isinstance(error, httpx.HTTPStatusError):
        return _provider_http_status_error(
            error.response.status_code,
            _bounded_error_detail(error.response.text),
            provider=provider,
            model=model,
        )
    if isinstance(
        error,
        (
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
        ),
    ):
        return ProviderRequestError(
            kind="response_format",
            provider=provider,
            model=model,
            detail="provider returned an unexpected response format",
        )
    return ProviderRequestError(
        kind="request",
        provider=provider,
        model=model,
        detail=_bounded_error_detail(str(error) or error.__class__.__name__),
    )


def _provider_http_status_error(
    status_code: int,
    detail: str,
    *,
    provider: str,
    model: str,
) -> ProviderRequestError:
    if status_code in {401, 403}:
        return ProviderRequestError(
            kind="authentication",
            provider=provider,
            model=model,
            detail=f"authentication failed with HTTP {status_code}",
        )
    if status_code == 429:
        return ProviderRequestError(
            kind="rate_limit",
            provider=provider,
            model=model,
            detail="provider rate limit was reached",
        )
    if status_code >= 500:
        return ProviderRequestError(
            kind="provider_status",
            provider=provider,
            model=model,
            detail=f"provider returned HTTP {status_code}: {detail}",
        )
    return ProviderRequestError(
        kind="request",
        provider=provider,
        model=model,
        detail=f"provider returned HTTP {status_code}: {detail}",
    )


def _bounded_error_detail(detail: str, max_chars: int = 300) -> str:
    normalized = " ".join(detail.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rstrip() + "... [truncated]"
