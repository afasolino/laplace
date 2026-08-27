from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from mcp import Client, types

import research_workspace.zetsu_sdk_stdio as sdk_module
from research_workspace.ast_context import tool_definition
from research_workspace.zetsu_sdk_stdio import ZetsuBackend, build_server


def test_ast_context_tool_name_is_stable() -> None:
    assert tool_definition()["name"] == "ast_context"


def test_zetsu_backend_is_stateless_and_uses_official_sdk_client(
    monkeypatch,
) -> None:
    calls: list[tuple[str, str, float]] = []

    class FakeSdkClient:
        async def list_tools(self) -> types.ListToolsResult:
            return types.ListToolsResult.model_validate({"tools": []})

    @asynccontextmanager
    async def fake_sdk_client(
        endpoint: str,
        token: str,
        timeout: float,
    ) -> AsyncIterator[Any]:
        calls.append((endpoint, token, timeout))
        yield FakeSdkClient()

    monkeypatch.setattr(sdk_module, "_sdk_client", fake_sdk_client)
    backend = ZetsuBackend("http://127.0.0.1:8765/mcp", "secret")
    assert asyncio.run(backend.list_tools()) == []
    assert calls == [("http://127.0.0.1:8765/mcp", "secret", 30.0)]


def test_official_sdk_can_connect_in_process(tmp_path: Path) -> None:
    class FakeBackend:
        def list_tools(self):
            return [
                {
                    "name": "search",
                    "description": "search",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                }
            ]

        def call_tool(self, name, arguments):
            assert name == "search"
            return {
                "content": [{"type": "text", "text": "ok"}],
                "structuredContent": {"query": arguments["query"]},
                "isError": False,
            }

    async def scenario() -> None:
        server = build_server(FakeBackend(), tmp_path)
        async with Client(server) as client:
            tools = await client.list_tools()
            assert [tool.name for tool in tools.tools] == ["search"]
            result = await client.call_tool("search", {"query": "x"})
            assert result.is_error is False
            assert result.structured_content == {"query": "x"}

    asyncio.run(scenario())
