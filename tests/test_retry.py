import httpx
import pytest

from agent.retry import is_transient_error, retry
from agent.schemas import CalculatorInput
from agent.tool import Tool


def test_retry_succeeds_on_third_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0
    sleep_calls: list[float] = []

    monkeypatch.setattr("agent.retry.time.sleep", sleep_calls.append)

    @retry(max_attempts=3, backoff=2)
    def flaky_operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TimeoutError("temporary failure")
        return "success"

    assert flaky_operation() == "success"
    assert attempts == 3
    assert sleep_calls == [1.0, 2.0]


def test_retry_raises_after_max_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    monkeypatch.setattr("agent.retry.time.sleep", lambda _: None)

    @retry(max_attempts=3)
    def failing_operation() -> None:
        nonlocal attempts
        attempts += 1
        raise TimeoutError("service unavailable")

    with pytest.raises(TimeoutError, match="service unavailable"):
        failing_operation()

    assert attempts == 3


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("missing.txt"),
        ValueError("invalid input"),
        RuntimeError("missing configuration"),
    ],
)
def test_retry_does_not_repeat_permanent_errors(error: Exception) -> None:
    attempts = 0

    @retry(max_attempts=3)
    def failing_operation() -> None:
        nonlocal attempts
        attempts += 1
        raise error

    with pytest.raises(type(error), match=str(error)):
        failing_operation()

    assert attempts == 1


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (404, False),
        (408, True),
        (429, True),
        (503, True),
    ],
)
def test_http_status_error_classification(
    status_code: int,
    expected: bool,
) -> None:
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(status_code, request=request)
    error = httpx.HTTPStatusError(
        f"HTTP {status_code}",
        request=request,
        response=response,
    )

    assert is_transient_error(error) is expected


def test_validation_error_does_not_run_tool() -> None:
    attempts = 0

    def calculator(expression: str) -> str:
        nonlocal attempts
        attempts += 1
        return expression

    tool = Tool(
        name="calculator",
        description="Calculate an expression.",
        input_schema=CalculatorInput,
        fn=calculator,
    )

    output, is_error = tool.execute({})

    assert is_error is True
    assert "field 'expression': Field required" in output
    assert attempts == 0
