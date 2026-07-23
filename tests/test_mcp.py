import asyncio
import sys
from pathlib import Path

import pytest

from agent.mcp import (
    McpClient,
    McpError,
    McpServerConfig,
    load_mcp_server_configs,
    load_mcp_tools,
    parse_mcp_server_configs,
)
from agent.tool_registry import ToolRegistry


def test_parse_mcp_server_configs_supports_mcp_servers_format() -> None:
    configs = parse_mcp_server_configs(
        {
            "mcpServers": {
                "demo": {
                    "command": "python",
                    "args": ["server.py"],
                    "env": {"TOKEN": "test-token"},
                    "cwd": "tools",
                    "trust": "trusted",
                    "approval": "never",
                    "allowedTools": ["echo"],
                    "blockedTools": ["delete"],
                    "readOnlyTools": ["echo"],
                    "allowExternalCwd": True,
                }
            }
        }
    )

    assert configs == [
        McpServerConfig(
            name="demo",
            command="python",
            args=["server.py"],
            env={"TOKEN": "test-token"},
            cwd="tools",
            trust="trusted",
            approval="never",
            allowedTools=["echo"],
            blockedTools=["delete"],
            readOnlyTools=["echo"],
            allowExternalCwd=True,
        )
    ]


def test_load_mcp_server_configs_reads_default_workspace_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AGENT_MCP_CONFIG", raising=False)
    config_dir = tmp_path / ".agents"
    config_dir.mkdir()
    (config_dir / "mcp.json").write_text(
        '{"mcpServers":{"demo":{"command":"python","args":["server.py"]}}}',
        encoding="utf-8",
    )

    configs = load_mcp_server_configs(tmp_path)

    assert configs == [
        McpServerConfig(name="demo", command="python", args=["server.py"])
    ]


def test_load_mcp_tools_registers_stdio_server_tools(tmp_path: Path) -> None:
    server_path = tmp_path / "fake_mcp_server.py"
    server_path.write_text(_fake_mcp_server_source(), encoding="utf-8")
    registry = ToolRegistry(tmp_path)

    async def run_case() -> tuple[str, bool]:
        manager = await load_mcp_tools(
            registry,
            tmp_path,
            [
                McpServerConfig(
                    name="demo",
                    command=sys.executable,
                    args=[str(server_path)],
                    readOnlyTools=["echo"],
                )
            ],
        )
        try:
            assert manager.tool_count == 1
            definitions = {
                definition.name: definition
                for definition in registry.to_tool_definitions()
            }
            definition = definitions["mcp_demo__echo"]
            assert definition.kind == "mcp"
            assert definition.description == (
                "Echo text. (MCP server `demo`, tool `echo`.)"
            )
            assert definition.input_schema == {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to echo.",
                    }
                },
                "required": ["text"],
            }
            return await registry.execute_async(
                "mcp_demo__echo",
                {"text": "hello"},
            )
        finally:
            await manager.close()

    output, is_error = asyncio.run(run_case())

    assert output == "echo: hello"
    assert is_error is False


@pytest.mark.parametrize(
    ("arguments", "expected_error"),
    [
        ({}, "'text' is a required property"),
        ({"text": 123}, "123 is not of type 'string'"),
    ],
)
def test_mcp_tool_validates_arguments_before_calling_server(
    tmp_path: Path,
    arguments: dict[str, object],
    expected_error: str,
) -> None:
    server_path = tmp_path / "fake_mcp_server.py"
    server_path.write_text(_fake_mcp_server_source(), encoding="utf-8")
    registry = ToolRegistry(tmp_path)

    async def run_case() -> tuple[str, bool]:
        manager = await load_mcp_tools(
            registry,
            tmp_path,
            [
                McpServerConfig(
                    name="demo",
                    command=sys.executable,
                    args=[str(server_path)],
                    readOnlyTools=["echo"],
                )
            ],
        )
        try:
            return await registry.execute_async("mcp_demo__echo", arguments)
        finally:
            await manager.close()

    output, is_error = asyncio.run(run_case())

    assert is_error is True
    assert "McpToolInputValidationError" in output
    assert expected_error in output


def test_load_mcp_tools_skips_blocked_tools_before_allowed_tools(
    tmp_path: Path,
) -> None:
    server_path = tmp_path / "fake_mcp_server.py"
    server_path.write_text(
        _fake_mcp_server_source_for_tools(["read", "write", "delete"]),
        encoding="utf-8",
    )
    registry = ToolRegistry(tmp_path)

    async def run_case() -> list[str]:
        manager = await load_mcp_tools(
            registry,
            tmp_path,
            [
                McpServerConfig(
                    name="demo",
                    command=sys.executable,
                    args=[str(server_path)],
                    allowedTools=["read", "delete"],
                    blockedTools=["delete"],
                )
            ],
        )
        try:
            assert manager.tool_count == 1
            return sorted(definition.name for definition in registry.to_tool_definitions())
        finally:
            await manager.close()

    assert asyncio.run(run_case()) == ["mcp_demo__read"]


def test_load_mcp_tools_registers_all_unblocked_tools_when_allowed_tools_omitted(
    tmp_path: Path,
) -> None:
    server_path = tmp_path / "fake_mcp_server.py"
    server_path.write_text(
        _fake_mcp_server_source_for_tools(["read", "write", "delete"]),
        encoding="utf-8",
    )
    registry = ToolRegistry(tmp_path)

    async def run_case() -> list[str]:
        manager = await load_mcp_tools(
            registry,
            tmp_path,
            [
                McpServerConfig(
                    name="demo",
                    command=sys.executable,
                    args=[str(server_path)],
                    blockedTools=["delete"],
                )
            ],
        )
        try:
            assert manager.tool_count == 2
            return sorted(definition.name for definition in registry.to_tool_definitions())
        finally:
            await manager.close()

    assert asyncio.run(run_case()) == ["mcp_demo__read", "mcp_demo__write"]


def test_mcp_relative_cwd_is_confined_to_workspace(tmp_path: Path) -> None:
    client = McpClient(
        McpServerConfig(name="demo", command="python", cwd="tools"),
        tmp_path,
    )

    assert client._resolve_cwd() == (tmp_path / "tools").resolve()

    escaping_client = McpClient(
        McpServerConfig(name="demo", command="python", cwd="../outside"),
        tmp_path,
    )
    with pytest.raises(McpError, match="outside the workspace"):
        escaping_client._resolve_cwd()


def test_mcp_absolute_cwd_requires_explicit_opt_in(tmp_path: Path) -> None:
    external_cwd = tmp_path.parent / "external-tools"

    blocked_client = McpClient(
        McpServerConfig(name="demo", command="python", cwd=str(external_cwd)),
        tmp_path,
    )
    with pytest.raises(McpError, match="absolute cwd"):
        blocked_client._resolve_cwd()

    allowed_client = McpClient(
        McpServerConfig(
            name="demo",
            command="python",
            cwd=str(external_cwd),
            allowExternalCwd=True,
        ),
        tmp_path,
    )
    assert allowed_client._resolve_cwd() == external_cwd.resolve()


def _fake_mcp_server_source() -> str:
    return """\
import json
import sys


def send(message):
    sys.stdout.write(json.dumps(message) + "\\n")
    sys.stdout.flush()


for line in sys.stdin:
    message = json.loads(line)
    if "id" not in message:
        continue
    request_id = message["id"]
    method = message.get("method")
    if method == "initialize":
        send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fake", "version": "0.1.0"},
                },
            }
        )
    elif method == "tools/list":
        send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "Echo text.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "text": {
                                        "type": "string",
                                        "description": "Text to echo.",
                                    }
                                },
                                "required": ["text"],
                            },
                        }
                    ]
                },
            }
        )
    elif method == "tools/call":
        arguments = message["params"]["arguments"]
        send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {"type": "text", "text": "echo: " + arguments["text"]}
                    ]
                },
            }
        )
    else:
        send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": "unknown method"},
            }
        )
"""


def _fake_mcp_server_source_for_tools(tool_names: list[str]) -> str:
    return f"""\
import json
import sys

TOOLS = {tool_names!r}


def send(message):
    sys.stdout.write(json.dumps(message) + "\\n")
    sys.stdout.flush()


for line in sys.stdin:
    message = json.loads(line)
    if "id" not in message:
        continue
    request_id = message["id"]
    method = message.get("method")
    if method == "initialize":
        send(
            {{
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {{
                    "protocolVersion": "2024-11-05",
                    "capabilities": {{"tools": {{}}}},
                    "serverInfo": {{"name": "fake", "version": "0.1.0"}},
                }},
            }}
        )
    elif method == "tools/list":
        send(
            {{
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {{
                    "tools": [
                        {{
                            "name": tool_name,
                            "description": f"{{tool_name}} tool.",
                            "inputSchema": {{"type": "object", "properties": {{}}}},
                        }}
                        for tool_name in TOOLS
                    ]
                }},
            }}
        )
    elif method == "tools/call":
        send(
            {{
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {{
                    "content": [
                        {{"type": "text", "text": message["params"]["name"]}}
                    ]
                }},
            }}
        )
    else:
        send(
            {{
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {{"code": -32601, "message": "unknown method"}},
            }}
        )
"""
