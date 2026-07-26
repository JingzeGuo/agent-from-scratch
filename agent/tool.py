import asyncio
import inspect
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from .retry import retry, retry_async
from .schemas import ToolDefinition
from .security import ToolApprovalPolicy


@dataclass
class Tool:
    name: str
    description: str
    input_schema: type[BaseModel]
    fn: Callable[..., Any]
    definition_input_schema: dict[str, Any] | None = None
    approval_policy: ToolApprovalPolicy | None = None

    def to_definition(self) -> ToolDefinition:
        """Build a provider-neutral tool definition."""
        if self.definition_input_schema is None:
            json_schema = self.input_schema.model_json_schema()
            json_schema.pop("title", None)
        else:
            json_schema = deepcopy(self.definition_input_schema)
        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=json_schema,
        )

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

    @retry_async(max_attempts=3, backoff=2)
    async def _run_async(
        self,
        parsed_input: BaseModel,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = parsed_input.model_dump()
        if extra_kwargs is not None:
            kwargs.update(extra_kwargs)

        result = self.fn(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

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
