import pytest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def no_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent.security.sandbox_available", lambda: False)
    monkeypatch.setattr("agent.tooling.command.sandbox_argv", lambda *args: None)
