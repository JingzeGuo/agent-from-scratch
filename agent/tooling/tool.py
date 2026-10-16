import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from ..schemas import ToolDefinition
from .retry import retry


@dataclass
class Tool:
    name: str
    description: str
    input_schema: type[BaseModel]
    fn: Callable[..., Any]

    def to_definition(self) -> ToolDefinition:
        """Build a provider-neutral tool definition."""
        json_schema = self.input_schema.model_json_schema()
        json_schema.pop("title", None)
        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=json_schema,
        )

    async def execute(
        self,
        raw_input: dict[str, Any],
        extra_kwargs: dict[str, Any] | None = None,
    ) -> tuple[str, bool]:
        parsed_or_error = self.parse(raw_input)
        if isinstance(parsed_or_error, str):
            return parsed_or_error, True
        return await self.run(parsed_or_error, extra_kwargs)

    def parse(self, raw_input: dict[str, Any]) -> BaseModel | str:
        """Validate raw model input, returning a readable error on failure."""
        try:
            return self.input_schema(**raw_input)
        except ValidationError as e:
            error_lines = [f"Validation error for tool '{self.name}':"]
            for err in e.errors():
                field = ".".join(str(p) for p in err["loc"])
                error_lines.append(f"  - field '{field}': {err['msg']}")
            return "\n".join(error_lines)

    async def run(
        self,
        parsed_input: BaseModel,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> tuple[str, bool]:
        """Run already-validated input, turning exceptions into observations."""
        try:
            return str(await self._invoke(parsed_input, extra_kwargs)), False
        except Exception as e:
            return f"Tool '{self.name}' raised {type(e).__name__}: {e}", True

    @retry(max_attempts=3, backoff=2)
    async def _invoke(
        self,
        parsed_input: BaseModel,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = parsed_input.model_dump()
        if extra_kwargs is not None:
            kwargs.update(extra_kwargs)
        # iscoroutinefunction also recognizes functools.partial of async functions.
        if inspect.iscoroutinefunction(self.fn):
            return await self.fn(**kwargs)
        return await asyncio.to_thread(self.fn, **kwargs)
