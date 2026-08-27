from pathlib import Path

from research_workspace.ast_context import tool_definition
from research_workspace.zetsu_sdk_stdio import ZetsuBackend


def test_ast_context_tool_name_is_stable() -> None:
    assert tool_definition()["name"] == "ast_context"


def test_zetsu_backend_is_stateless_and_uses_backend_result_shape(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    calls: list[tuple[str, str]] = []

    def fake_rpc(endpoint, token, payload, *, timeout, protocol_version=None):  # type: ignore[no-untyped-def]
        calls.append((endpoint, str(payload["method"])))
        return 200, {"jsonrpc": "2.0", "id": payload["id"], "result": {"tools": []}}

    monkeypatch.setattr("research_workspace.zetsu_sdk_stdio._rpc", fake_rpc)
    backend = ZetsuBackend("http://127.0.0.1:8765/mcp", "secret")
    assert backend.list_tools() == []
    assert calls == [("http://127.0.0.1:8765/mcp", "tools/list")]


def test_official_sdk_can_connect_in_process(tmp_path: Path) -> None:
    import asyncio

    from mcp import Client

    from research_workspace.zetsu_sdk_stdio import build_server

    class FakeBackend:
        def list_tools(self):  # type: ignore[no-untyped-def]
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

        def call_tool(self, name, arguments):  # type: ignore[no-untyped-def]
            assert name == "search"
            return {
                "content": [{"type": "text", "text": "ok"}],
                "structuredContent": {"query": arguments["query"]},
                "isError": False,
            }

    async def scenario() -> None:
        server = build_server(FakeBackend(), tmp_path)  # type: ignore[arg-type]
        async with Client(server) as client:
            tools = await client.list_tools()
            assert [tool.name for tool in tools.tools] == ["search"]
            result = await client.call_tool("search", {"query": "x"})
            assert result.is_error is False
            assert result.structured_content == {"query": "x"}

    asyncio.run(scenario())
