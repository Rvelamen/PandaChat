import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
import uuid
import anyio
from httpx import HTTPStatusError
from mcp import ClientSession
from mcp.client.stdio import (
    StdioServerParameters,
    get_default_environment,
)

from anyio import Event, create_task_group
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client, MCP_SESSION_ID
from mcp.client.websocket import websocket_client
from mcp.types import JSONRPCMessage, ServerCapabilities
from typing import AsyncGenerator, Callable, Dict, Optional, List, Any
import json
import os

from database import get_db_async
from models.mcp_server import MCPServer


class MCPConnection:
    def __init__(self, server_name: str, config: dict):
        self.server_name = server_name
        self.config = config
        self.session: Optional[ClientSession] = None
        self.capabilities: Optional[ServerCapabilities] = None
        self.tools: Dict[str, dict] = {}
        self._transport_client = None
        self._session_context: Optional[Callable[[], Optional[str]]] = None
        self._shutdown_event = Event()
        self._initialized_event = Event()
        self.session_id: Optional[str] = None
        self.error_info = ""

    def _config_value(self, key: str, default: Any = None) -> Any:
        """
        Helper to retrieve configuration values.
        Prefers top-level keys, falls back to nested `config` payloads.
        """
        if key in self.config and self.config[key] is not None:
            return self.config[key]
        nested = self.config.get("config") or {}
        return nested.get(key, default)

    def bind_session_tracker(
        self, session_getter: Optional[Callable[[], Optional[str]]]
    ) -> None:
        """Store the callback provided by transports that expose session IDs."""
        self._session_context = session_getter

    def refresh_session_state(self) -> None:
        """Pull the latest session metadata from the transport if available."""
        if self._session_context:
            self.session_id = self._session_context()

    def transport_context_factory(self):
        if self.config["transport"] == "stdio":
            server_params = StdioServerParameters(
                command=self.config["command"],
                args=self.config["args"] or [],
                env={**get_default_environment(), **(self.config["env"] or {})},
            )
            # Create stdio client config with redirected stderr
            return stdio_client(server=server_params)
        elif self.config["transport"] in ["streamable_http", "streamable-http", "http"]:
            headers = (self._config_value("headers", {}) or {}).copy()
            if self.session_id:
                headers[MCP_SESSION_ID] = self.session_id

            kwargs = {
                "url": self._config_value("url"),
                "headers": headers,
                "terminate_on_close": self._config_value("terminate_on_close", True),
            }

            timeout_seconds = self._config_value("http_timeout_seconds")
            if timeout_seconds:
                kwargs["timeout"] = timedelta(seconds=timeout_seconds)

            sse_read_timeout_seconds = self._config_value("read_timeout_seconds")
            if sse_read_timeout_seconds:
                kwargs["sse_read_timeout"] = timedelta(seconds=sse_read_timeout_seconds)

            return streamablehttp_client(
                **kwargs,
            )
        elif self.config["transport"] == "sse":
            kwargs = {
                "url": self._config_value("url"),
                "headers": self._config_value("headers", {}),
            }

            if self._config_value("http_timeout_seconds"):
                kwargs["timeout"] = self._config_value("http_timeout_seconds")

            if self._config_value("read_timeout_seconds"):
                kwargs["sse_read_timeout"] = self._config_value("read_timeout_seconds")

            return sse_client(**kwargs)
        elif self.config["transport"] == "websocket":
            return websocket_client(url=self._config_value("url"))
        else:
            raise ValueError(f"Unsupported transport: {self.config['transport']}")

    async def initialize_session(self) -> None:
        """
        Initializes the server connection and session.
        Must be called within an async context.
        """

        await self.session.initialize()
        await self.refresh_tools()

        # Now the session is ready for use
        self._initialized_event.set()

    async def wait_for_initialized(self) -> None:
        """
        Wait until the session is fully initialized.
        """
        await self._initialized_event.wait()

    def request_shutdown(self) -> None:
        """
        Request the server to shut down. Signals the server lifecycle task to exit.
        """
        self._shutdown_event.set()

    async def wait_for_shutdown_request(self) -> None:
        """
        Wait until the shutdown event is set.
        """
        await self._shutdown_event.wait()

    async def create_session(self, read_stream, write_stream):
        """创建会话"""
        self.session = ClientSession(read_stream, write_stream)
        return self.session

    async def refresh_tools(self):
        """刷新可用工具列表"""
        if not self.session:
            raise RuntimeError("Not connected to server")
        tools = await self.session.list_tools()
        self.tools = {tool.name: tool.model_dump() for tool in tools.tools}

    def get_tool(self, tool_name: str) -> Optional[dict]:
        """获取指定工具的信息"""
        return self.tools.get(tool_name)

    def list_tools(self) -> List[str]:
        """列出所有可用的工具名称"""
        return list(self.tools.values())

    async def call_tool(self, tool_name: str, **kwargs) -> Any:
        """调用指定工具"""
        if not self.session:
            raise RuntimeError("Not connected to server")

        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")

        # 发送请求并等待结果
        result = await self.session.call_tool(name=tool_name, arguments=kwargs)
        return result

    async def disconnect(self):
        self.session.request_shutdown()

    @property
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self.session is not None


async def _server_lifecycle_task(server_conn: MCPConnection) -> None:
    """
    Manage the lifecycle of a single server connection.
    Runs inside the MCPConnectionManager's shared TaskGroup.
    """
    server_name = server_conn.server_name
    try:
        transport_context = server_conn.transport_context_factory()

        async with transport_context as (read_stream, write_stream, *extras):
            session_tracker = extras[0] if extras else None
            server_conn.bind_session_tracker(session_tracker)

            # Build a session
            await server_conn.create_session(read_stream, write_stream)

            async with server_conn.session:
                # Initialize the session
                await server_conn.initialize_session()
                server_conn.refresh_session_state()

                # Wait until we're asked to shut down
                await server_conn.wait_for_shutdown_request()
    except Exception as exc:
        server_conn._initialized_event.set()
        server_conn.request_shutdown()
        server_conn.error_info = exc
        
        import traceback
        traceback.format_exc()
        if "httpx.HTTPStatusError: Client error '404 Not Found'" in traceback.format_exc():
            server_conn.error_info = "服务器不存在: 404 Not Found"
        else:
            server_conn.error_info = str(exc)


class MCPConnectionManager:
    def __init__(self, config_path: str = "./mcp.json"):
        self.servers: Dict[str, MCPConnection] = {}
        self.config_path = config_path
        self.servers_config = {}

    async def __aenter__(self):
        # We create a task group to manage all server lifecycle tasks
        tg = create_task_group()
        # Enter the task group context
        await tg.__aenter__()
        self._tg_active = True
        self._tg = tg
        await self.load_config()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure clean shutdown of all connections before exiting."""
        try:
            # First request all servers to shutdown
            print("MCPConnectionManager: shutting down all server tasks...")
            await self.disconnect_all()

            # TODO: saqadri (FA1) - Is this needed?
            # Add a small delay to allow for clean shutdown
            await anyio.sleep(0.5)

            # Then close the task group if it's active
            if self._tg_active:
                await self._tg.__aexit__(exc_type, exc_val, exc_tb)
                self._tg_active = False
                self._tg = None
        except AttributeError:  # Handle missing `_exceptions`
            pass
        except Exception as e:
            print(f"MCPConnectionManager: Error during shutdown: {e}")

    async def load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            self.servers_config = json.load(f).get("mcp", {}).get("servers", {})
            
        async with get_db_async() as db:
            servers = db.query(MCPServer).filter(MCPServer.is_active == True).all()
            self.servers_config.update({server.name: server.to_dict() for server in servers})

    async def get_server(self, server_name: str) -> MCPConnection:
        server_conn = self.servers.get(server_name)
        if server_conn and server_conn.is_healthy():
            return server_conn

        # If server exists but isn't healthy, remove it so we can create a new one
        if server_conn:
            print(f"{server_name}: Server exists but is unhealthy, recreating...")
            self.servers.pop(server_name)
            server_conn.request_shutdown()

        server_conn = MCPConnection(
            server_name=server_name, config=self.servers_config[server_name]
        )
        self._tg.start_soon(_server_lifecycle_task, server_conn)
        await server_conn.wait_for_initialized()
        return server_conn

    async def close_all(self):
        for server in self.servers.values():
            await server.disconnect()
