from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
from asyncio.subprocess import Process
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from .tool import Tool
from .tool_registry import ToolRegistry

MCP_CLIENT_NAME = "agent-from-scratch"
MCP_PROTOCOL_VERSION = "2024-11-05"
DEFAULT_MCP_CONFIG_PATH = Path(".agents") / "mcp.json"
JSONRPC_VERSION = "2.0"
MAX_TOOL_NAME_LENGTH = 64
MCP_INPUT_MODEL = cast(
    type[BaseModel],
    create_model("McpToolInput", __config__=ConfigDict(extra="allow")),
)


class McpError(RuntimeError):
    """Raised when an MCP server transport or protocol operation fails."""


class McpToolError(RuntimeError):
    """Raised when an MCP tool reports an error result."""


class McpServerConfig(BaseModel):
    """Configuration for one stdio MCP server."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1)
    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    max_output_chars: int = Field(default=12_000, ge=1_000, le=50_000)


class McpToolInfo(BaseModel):
    """Tool metadata returned by an MCP server."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    name: str = Field(min_length=1)
    description: str | None = None
    input_schema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
        alias="inputSchema",
    )


class McpToolManager:
    """Owns long-lived MCP clients registered into a tool registry."""

    def __init__(self) -> None:
        self.clients: list[McpClient] = []
        self.tool_count = 0

    async def close(self) -> None:
        for client in self.clients:
            await client.close()


class McpClient:
    """Minimal stdio MCP client for initialization, tool listing, and tool calls."""

    def __init__(self, config: McpServerConfig, workspace_root: Path) -> None:
        self.config = config
        self.workspace_root = workspace_root
        self.process: Process | None = None
        self._next_id = 1
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_lines: list[str] = []

    async def connect(self) -> None:
        cwd = self._resolve_cwd()
        env = os.environ.copy()
        env.update(self.config.env)
        try:
            self.process = await asyncio.create_subprocess_exec(
                self.config.command,
                *self.config.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
                env=env,
            )
        except OSError as e:
            raise McpError(
                f"Failed to start MCP server '{self.config.name}': {e}"
            ) from e
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        await self._initialize()

    async def list_tools(self) -> list[McpToolInfo]:
        tools: list[McpToolInfo] = []
        params: dict[str, Any] = {}
        while True:
            result = await self._request("tools/list", params)
            raw_tools = result.get("tools")
            if not isinstance(raw_tools, list):
                raise McpError(
                    f"MCP server '{self.config.name}' returned invalid tools/list result."
                )
            for raw_tool in raw_tools:
                try:
                    tools.append(McpToolInfo.model_validate(raw_tool))
                except ValidationError as e:
                    raise McpError(
                        f"MCP server '{self.config.name}' returned invalid tool metadata: {e}"
                    ) from e
            next_cursor = result.get("nextCursor")
            if not isinstance(next_cursor, str) or not next_cursor:
                return tools
            params = {"cursor": next_cursor}

    async def call_tool(self, remote_tool_name: str, arguments: dict[str, Any]) -> str:
        result = await self._request(
            "tools/call",
            {"name": remote_tool_name, "arguments": arguments},
        )
        output = _format_tool_call_result(result, self.config.max_output_chars)
        if result.get("isError") is True:
            raise McpToolError(output)
        return output

    async def close(self) -> None:
        process = self.process
        if process is None:
            return
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except TimeoutError:
                process.kill()
                await process.wait()
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stderr_task

    def _resolve_cwd(self) -> Path:
        if self.config.cwd is None:
            return self.workspace_root
        path = Path(self.config.cwd).expanduser()
        if not path.is_absolute():
            path = self.workspace_root / path
        return path.resolve()

    async def _initialize(self) -> None:
        await self._request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": MCP_CLIENT_NAME,
                    "version": "0.1.0",
                },
            },
        )
        await self._send(
            {
                "jsonrpc": JSONRPC_VERSION,
                "method": "notifications/initialized",
            }
        )

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = self._next_request_id()
        await self._send(
            {
                "jsonrpc": JSONRPC_VERSION,
                "id": request_id,
                "method": method,
                "params": params,
            }
        )
        while True:
            message = await self._read_message()
            if message.get("id") == request_id:
                error = message.get("error")
                if isinstance(error, dict):
                    raise McpError(
                        f"MCP server '{self.config.name}' returned error for {method}: "
                        f"{_format_jsonrpc_error(error)}"
                    )
                result = message.get("result", {})
                if not isinstance(result, dict):
                    raise McpError(
                        f"MCP server '{self.config.name}' returned non-object result for {method}."
                    )
                return result
            await self._handle_unmatched_message(message)

    async def _send(self, message: dict[str, Any]) -> None:
        process = self._require_process()
        if process.stdin is None:
            raise McpError(f"MCP server '{self.config.name}' has no stdin.")
        payload = json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n"
        process.stdin.write(payload)
        await process.stdin.drain()

    async def _read_message(self) -> dict[str, Any]:
        process = self._require_process()
        if process.stdout is None:
            raise McpError(f"MCP server '{self.config.name}' has no stdout.")
        try:
            line = await asyncio.wait_for(
                process.stdout.readline(),
                timeout=self.config.timeout_seconds,
            )
        except TimeoutError as e:
            raise McpError(
                f"Timed out waiting for MCP server '{self.config.name}'."
                f"{self._stderr_summary()}"
            ) from e
        if not line:
            raise McpError(
                f"MCP server '{self.config.name}' closed stdout."
                f"{self._stderr_summary()}"
            )
        try:
            message = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise McpError(
                f"MCP server '{self.config.name}' returned invalid JSON."
            ) from e
        if not isinstance(message, dict):
            raise McpError(
                f"MCP server '{self.config.name}' returned a non-object message."
            )
        return message

    async def _handle_unmatched_message(self, message: dict[str, Any]) -> None:
        if "id" in message and isinstance(message.get("method"), str):
            await self._send(
                {
                    "jsonrpc": JSONRPC_VERSION,
                    "id": message["id"],
                    "error": {
                        "code": -32601,
                        "message": "Client method not supported.",
                    },
                }
            )

    async def _drain_stderr(self) -> None:
        process = self._require_process()
        if process.stderr is None:
            return
        while True:
            line = await process.stderr.readline()
            if not line:
                return
            text = line.decode("utf-8", errors="replace").rstrip()
            self._stderr_lines.append(text)
            if len(self._stderr_lines) > 20:
                self._stderr_lines.pop(0)

    def _next_request_id(self) -> int:
        request_id = self._next_id
        self._next_id += 1
        return request_id

    def _require_process(self) -> Process:
        if self.process is None:
            raise McpError(f"MCP server '{self.config.name}' is not connected.")
        return self.process

    def _stderr_summary(self) -> str:
        if not self._stderr_lines:
            return ""
        stderr = "\n".join(self._stderr_lines)[-1_000:]
        return f" Recent stderr:\n{stderr}"


def load_mcp_server_configs(workspace_root: Path) -> list[McpServerConfig]:
    config_path = resolve_mcp_config_path(workspace_root)
    if config_path is None:
        return []
    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ValueError(f"Could not read MCP config {config_path}: {e}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid MCP config JSON in {config_path}: {e}") from e
    return parse_mcp_server_configs(raw_config)


def resolve_mcp_config_path(workspace_root: Path) -> Path | None:
    configured = os.getenv("AGENT_MCP_CONFIG")
    if configured is not None:
        if not configured.strip():
            return None
        configured_path = Path(configured).expanduser()
        if not configured_path.is_absolute():
            configured_path = workspace_root / configured_path
        return configured_path
    default_path = workspace_root / DEFAULT_MCP_CONFIG_PATH
    if default_path.exists():
        return default_path
    return None


def parse_mcp_server_configs(raw_config: Any) -> list[McpServerConfig]:
    if not isinstance(raw_config, dict):
        raise ValueError("MCP config must be a JSON object.")

    raw_servers = raw_config.get("mcpServers", raw_config.get("servers"))
    if raw_servers is None:
        raise ValueError("MCP config must contain 'mcpServers' or 'servers'.")

    server_items = _iter_raw_server_items(raw_servers)
    configs: list[McpServerConfig] = []
    for name, raw_server in server_items:
        if not isinstance(raw_server, dict):
            raise ValueError(f"MCP server '{name}' must be a JSON object.")
        server_data = dict(raw_server)
        server_data["name"] = name
        try:
            configs.append(McpServerConfig.model_validate(server_data))
        except ValidationError as e:
            raise ValueError(f"Invalid MCP server '{name}': {e}") from e
    return configs


async def load_mcp_tools_from_env(
    registry: ToolRegistry,
    workspace_root: Path,
) -> McpToolManager:
    return await load_mcp_tools(
        registry,
        workspace_root,
        load_mcp_server_configs(workspace_root),
    )


async def load_mcp_tools(
    registry: ToolRegistry,
    workspace_root: Path,
    configs: list[McpServerConfig],
) -> McpToolManager:
    manager = McpToolManager()
    try:
        for config in configs:
            client = McpClient(config, workspace_root)
            await client.connect()
            manager.clients.append(client)
            tools = await client.list_tools()
            for tool_info in tools:
                register_mcp_tool(registry, client, config.name, tool_info)
                manager.tool_count += 1
    except Exception:
        await manager.close()
        raise
    return manager


def register_mcp_tool(
    registry: ToolRegistry,
    client: McpClient,
    server_name: str,
    tool_info: McpToolInfo,
) -> None:
    local_name = _unique_tool_name(
        _build_local_tool_name(server_name, tool_info.name),
        registry,
    )
    description = (
        f"MCP tool `{tool_info.name}` from server `{server_name}`."
        if not tool_info.description
        else f"{tool_info.description} (MCP server `{server_name}`, tool `{tool_info.name}`.)"
    )

    async def call_mcp_tool(**arguments: Any) -> str:
        return await client.call_tool(tool_info.name, arguments)

    registry.register(
        Tool(
            name=local_name,
            description=description,
            input_schema=MCP_INPUT_MODEL,
            fn=call_mcp_tool,
            kind="mcp",
            definition_input_schema=_object_schema(tool_info.input_schema),
        )
    )


def _iter_raw_server_items(raw_servers: Any) -> list[tuple[str, Any]]:
    if isinstance(raw_servers, dict):
        return [(str(name), raw_server) for name, raw_server in raw_servers.items()]
    if isinstance(raw_servers, list):
        items: list[tuple[str, Any]] = []
        for index, raw_server in enumerate(raw_servers, start=1):
            if not isinstance(raw_server, dict):
                raise ValueError(f"MCP server entry {index} must be a JSON object.")
            raw_name = raw_server.get("name")
            if not isinstance(raw_name, str) or not raw_name:
                raise ValueError(f"MCP server entry {index} must include a name.")
            items.append((raw_name, raw_server))
        return items
    raise ValueError("'mcpServers' or 'servers' must be an object or list.")


def _build_local_tool_name(server_name: str, remote_tool_name: str) -> str:
    base_name = f"mcp_{_safe_name_part(server_name)}__{_safe_name_part(remote_tool_name)}"
    if len(base_name) <= MAX_TOOL_NAME_LENGTH:
        return base_name
    digest = hashlib.sha1(base_name.encode("utf-8")).hexdigest()[:8]
    keep = MAX_TOOL_NAME_LENGTH - len(digest) - 1
    return f"{base_name[:keep]}_{digest}"


def _safe_name_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    if not cleaned:
        return "tool"
    return cleaned


def _unique_tool_name(base_name: str, registry: ToolRegistry) -> str:
    if base_name not in registry.tools:
        return base_name
    for index in range(2, 100):
        suffix = f"_{index}"
        candidate = f"{base_name[: MAX_TOOL_NAME_LENGTH - len(suffix)]}{suffix}"
        if candidate not in registry.tools:
            return candidate
    raise ValueError(f"Could not create unique MCP tool name for {base_name}.")


def _object_schema(schema: dict[str, Any]) -> dict[str, Any]:
    if not schema:
        return {"type": "object", "properties": {}}
    normalized = dict(schema)
    if normalized.get("type") != "object":
        normalized["type"] = "object"
    normalized.setdefault("properties", {})
    return normalized


def _format_tool_call_result(result: dict[str, Any], max_chars: int) -> str:
    content = result.get("content")
    if not isinstance(content, list):
        return "[MCP tool returned no content]"
    rendered_parts = [_format_content_item(item) for item in content]
    output = "\n".join(part for part in rendered_parts if part)
    if not output:
        output = "[MCP tool returned empty content]"
    return _truncate(output, max_chars)


def _format_content_item(item: Any) -> str:
    if not isinstance(item, dict):
        return str(item)
    item_type = item.get("type")
    if item_type == "text":
        text = item.get("text")
        return text if isinstance(text, str) else ""
    if item_type == "image":
        mime_type = item.get("mimeType")
        detail = f": {mime_type}" if isinstance(mime_type, str) else ""
        return f"[MCP image content{detail}]"
    if item_type == "resource":
        resource = item.get("resource")
        if isinstance(resource, dict):
            uri = resource.get("uri")
            if isinstance(uri, str):
                return f"[MCP resource: {uri}]"
        return "[MCP resource content]"
    return f"[MCP {item_type or 'unknown'} content]"


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n[truncated after {max_chars} chars]"


def _format_jsonrpc_error(error: dict[str, Any]) -> str:
    code = error.get("code")
    message = error.get("message")
    if isinstance(message, str):
        return f"{code}: {message}"
    return str(error)
