import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from anthropic.types import ToolParam
from pydantic import BaseModel, ValidationError

from .retry import is_transient_error, retry
from .schemas import ToolDefinition, ToolKind


@dataclass
class Tool:
    name: str
    description: str
    input_schema: type[BaseModel]
    fn: Callable[..., Any]
    kind: ToolKind = "read_only"

    def to_definition(self) -> ToolDefinition:
        """Build a provider-neutral tool definition."""
        json_schema = self.input_schema.model_json_schema()
        json_schema.pop("title", None)
        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=json_schema,
            kind=self.kind,
        )

    def to_anthropic_schema(self) -> ToolParam:
        """Build the tool definition expected by the Anthropic Messages API."""
        definition = self.to_definition()
        return {
            "name": definition.name,
            "description": definition.description,
            "input_schema": definition.input_schema,
        }

    @retry(max_attempts=3, backoff=2)
    def _run(
        self,
        parsed_input: BaseModel,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = parsed_input.model_dump()
        if extra_kwargs is not None:
            kwargs.update(extra_kwargs)
        return self.fn(**kwargs)

    async def _run_async(
        self,
        parsed_input: BaseModel,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = parsed_input.model_dump()
        if extra_kwargs is not None:
            kwargs.update(extra_kwargs)

        wait_time = 1.0
        for attempt in range(1, 4):
            try:
                result = self.fn(**kwargs)
                if inspect.isawaitable(result):
                    return await result
                return result
            except Exception as e:
                if attempt == 3 or not is_transient_error(e):
                    raise
                await asyncio.sleep(wait_time)
                wait_time *= 2

        raise RuntimeError("Retry loop ended unexpectedly")

    def execute(
        self,
        raw_input: dict[str, Any],
        extra_kwargs: dict[str, Any] | None = None,
    ) -> tuple[str, bool]:
        if self._is_async_callable():
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self.execute_async(raw_input, extra_kwargs))
            return f"Tool '{self.name}' requires async execution.", True

        parsed_or_error = self._parse(raw_input)
        if isinstance(parsed_or_error, str):
            return parsed_or_error, True
        try:
            result = self._run(parsed_or_error, extra_kwargs)
            return str(result), False
        except Exception as e:
            return f"Tool '{self.name}' raised {type(e).__name__}: {e}", True

    async def execute_async(
        self,
        raw_input: dict[str, Any],
        extra_kwargs: dict[str, Any] | None = None,
    ) -> tuple[str, bool]:
        parsed_or_error = self._parse(raw_input)
        if isinstance(parsed_or_error, str):
            return parsed_or_error, True
        try:
            result = await self._run_async(parsed_or_error, extra_kwargs)
            return str(result), False
        except Exception as e:
            return f"Tool '{self.name}' raised {type(e).__name__}: {e}", True

    def _parse(self, raw_input: dict[str, Any]) -> BaseModel | str:
        try:
            return self.input_schema(**raw_input)
        except ValidationError as e:
            error_lines = [f"Validation error for tool '{self.name}':"]
            for err in e.errors():
                field = ".".join(str(p) for p in err["loc"])
                error_lines.append(f"  - field '{field}': {err['msg']}")
            return "\n".join(error_lines)

    def _is_async_callable(self) -> bool:
        if inspect.iscoroutinefunction(self.fn):
            return True
        wrapped = getattr(self.fn, "func", None)
        return inspect.iscoroutinefunction(wrapped)
