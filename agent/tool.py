from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from .retry import retry


@dataclass
class Tool:
    name: str
    description: str
    input_schema: type[BaseModel]
    fn: Callable

    def to_anthropic_schema(self) -> dict[str, Any]:
        """Build the tool definition expected by the Anthropic Messages API."""
        json_schema = self.input_schema.model_json_schema()
        # Anthropic doesn't need Pydantic's "title" field; strip it for cleanliness.
        json_schema.pop("title", None)
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": json_schema,
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
