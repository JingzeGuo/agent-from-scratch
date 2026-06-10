import pytest
from anthropic import AsyncAnthropic

from agent.agent import Agent
from agent.schemas import CalculatorInput
from agent.tool import Tool
from agent.tool_registry import ToolRegistry
from agent.tools import calculator
from main import handle_command


def create_agent() -> Agent:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="calculator",
            description="Calculate an expression.",
            input_schema=CalculatorInput,
            fn=calculator,
        )
    )
    return Agent(
        client=AsyncAnthropic(api_key="test-key"),
        registry=registry,
    )


def test_help_lists_available_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    should_exit = handle_command("/help")

    assert should_exit is False
    assert capsys.readouterr().out == (
        "Available commands:\n"
        "  /help   Show available commands.\n"
        "  /model  Show or switch provider and model.\n"
        "  /exit   Exit the application.\n"
    )


def test_exit_requests_cli_exit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    should_exit = handle_command("/exit")

    assert should_exit is True
    assert capsys.readouterr().out == "Goodbye.\n"


def test_unknown_command_shows_help_hint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    should_exit = handle_command("/unknown")

    assert should_exit is False
    assert capsys.readouterr().out == (
        "Unknown command: /unknown\n"
        "Type /help to see available commands.\n"
    )


def test_model_command_shows_current_model(
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent = create_agent()

    should_exit = handle_command("/model", agent)

    assert should_exit is False
    assert capsys.readouterr().out == (
        "Current model: anthropic/claude-haiku-4-5\n"
    )


def test_model_command_switches_provider(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    agent = create_agent()

    should_exit = handle_command("/model deepseek", agent)

    assert should_exit is False
    assert agent.provider == "deepseek"
    assert agent.model == "deepseek-v4-flash"
    assert capsys.readouterr().out == (
        "Switched model: deepseek/deepseek-v4-flash\n"
    )
