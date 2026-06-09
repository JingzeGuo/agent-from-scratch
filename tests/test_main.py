import pytest

from main import handle_command


def test_help_lists_available_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    should_exit = handle_command("/help")

    assert should_exit is False
    assert capsys.readouterr().out == (
        "Available commands:\n"
        "  /help  Show available commands.\n"
        "  /exit  Exit the application.\n"
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
