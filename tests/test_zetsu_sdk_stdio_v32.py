from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from mcp import Client, types

import research_workspace.zetsu_sdk_stdio as sdk
from research_workspace.ast_context import tool_definition
from research_workspace.zetsu_sdk_stdio import ZetsuBackend, ZetsuSdkBridgeError, build_server


def test_backend_refuses_non_loopback() -> None:
    with pytest.raises(ZetsuSdkBridgeError, match="zetsu_backend_loopback_only"):
        ZetsuBackend("http://example.com/mcp", "secret")


def test_backend_uses_sdk_client(monkeypatch) -> None:
    class Fake:
        async def list_tools(self):
            return types.ListToolsResult.model_validate({"tools": []})

    @asynccontextmanager
    async def factory(_endpoint: str, _token: str, _timeout: float) -> AsyncIterator[Any]:
        yield Fake()

    monkeypatch.setattr(sdk, "_sdk_client", factory)
    assert asyncio.run(ZetsuBackend("http://127.0.0.1:8765/mcp", "s").list_tools()) == []


def test_inprocess_sdk_schema_and_containment(tmp_path: Path) -> None:
    (tmp_path / "sample.py").write_text("def target():\n    return 1\n", encoding="utf-8")

    class Backend:
        def list_tools(self):
            return [{
                "name": "agent_task",
                "description": "agent",
                "inputSchema": {"type": "object", "properties": {}},
            }]

        def call_tool(self, name, arguments):
            raise AssertionError((name, arguments))

    async def scenario():
        async with Client(build_server(Backend(), tmp_path)) as client:
            tools = await client.list_tools()
            ast_tool = next(tool for tool in tools.tools if tool.name == "ast_context")
            assert ast_tool.input_schema == tool_definition()["inputSchema"]
            good = await client.call_tool(
                "ast_context", {"path": "sample.py", "pattern": "target"}
            )
            assert good.is_error is False
            escaped = await client.call_tool(
                "ast_context", {"path": "../outside.py", "pattern": "x"}
            )
            assert escaped.is_error is True

    asyncio.run(scenario())
