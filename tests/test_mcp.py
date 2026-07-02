import asyncio
import sys
from pathlib import Path

import pytest

from agent.mcp import (
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
