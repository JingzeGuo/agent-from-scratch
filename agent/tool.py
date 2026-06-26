from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from anthropic.types import ToolParam
from pydantic import BaseModel, ValidationError

from .retry import retry
from .schemas import ToolDefinition


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

    def to_anthropic_schema(self) -> ToolParam:
        """Build the tool definition expected by the Anthropic Messages API."""
        definition = self.to_definition()
        return {
            "name": definition.name,
            "description": definition.description,
            "input_schema": definition.input_schema,
        }

    @retry(max_attempts=3, backoff=2)
    def _run(self, parsed_input: BaseModel) -> Any:
        return self.fn(**parsed_input.model_dump())

    def execute(self, raw_input: dict[str, Any]) -> tuple[str, bool]:
        try:
            parsed = self.input_schema(**raw_input)
        except ValidationError as e:
            error_lines = [f"Validation error for tool '{self.name}':"]
            for err in e.errors():
                field = ".".join(str(p) for p in err["loc"])
                error_lines.append(f"  - field '{field}': {err['msg']}")
            return "\n".join(error_lines), True
        try:
            result = self._run(parsed)
            return str(result), False
        except Exception as e:
            return f"Tool '{self.name}' raised {type(e).__name__}: {e}", True
