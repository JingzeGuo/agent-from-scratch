import httpx
import pytest
from pydantic import BaseModel

from agent.retry import is_transient_error, retry_async
from agent.tool import Tool


class SampleToolInput(BaseModel):
    value: str


@pytest.mark.anyio
async def test_retry_async_succeeds_on_third_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    sleep_calls: list[float] = []

    async def fake_sleep(wait_time: float) -> None:
        sleep_calls.append(wait_time)

    monkeypatch.setattr("agent.retry.asyncio.sleep", fake_sleep)

    @retry_async(max_attempts=3, backoff=2)
    async def flaky_operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TimeoutError("temporary failure")
        return "success"

    assert await flaky_operation() == "success"
    assert attempts == 3
    assert sleep_calls == [1.0, 2.0]


@pytest.mark.anyio
async def test_retry_async_raises_after_max_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("agent.retry.asyncio.sleep", no_sleep)

    @retry_async(max_attempts=3)
    async def failing_operation() -> None:
        nonlocal attempts
        attempts += 1
        raise TimeoutError("service unavailable")

    with pytest.raises(TimeoutError, match="service unavailable"):
        await failing_operation()

    assert attempts == 3


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("missing.txt"),
        ValueError("invalid input"),
        RuntimeError("missing configuration"),
    ],
)
@pytest.mark.anyio
async def test_retry_async_does_not_repeat_permanent_errors(
    error: Exception,
) -> None:
    attempts = 0

    @retry_async(max_attempts=3)
    async def failing_operation() -> None:
        nonlocal attempts
        attempts += 1
        raise error

    with pytest.raises(type(error), match=str(error)):
        await failing_operation()

    assert attempts == 1


@pytest.mark.anyio
async def test_async_tool_uses_retry_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    sleep_calls: list[float] = []

    async def fake_sleep(wait_time: float) -> None:
        sleep_calls.append(wait_time)

    monkeypatch.setattr("agent.retry.asyncio.sleep", fake_sleep)

    async def sample_tool(value: str) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("temporary failure")
        return value

    tool = Tool(
        name="read_file",
        description="Process a value.",
        input_schema=SampleToolInput,
        fn=sample_tool,
    )

    output, is_error = await tool.execute_async({"value": "sample"})

    assert output == "sample"
    assert is_error is False
    assert attempts == 2
    assert sleep_calls == [1.0]


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


@pytest.mark.anyio
async def test_validation_error_does_not_run_tool() -> None:
    attempts = 0

    def sample_tool(value: str) -> str:
        nonlocal attempts
        attempts += 1
        return value

    tool = Tool(
        name="read_file",
        description="Process a value.",
        input_schema=SampleToolInput,
        fn=sample_tool,
    )

    output, is_error = await tool.execute_async({})

    assert is_error is True
    assert "field 'value': Field required" in output
    assert attempts == 0
