"""Official-MCP-SDK stdio bridge from Codex/Hermes to authenticated Zetsu.

The SDK owns MCP framing, initialization, legacy compatibility and stdio
transport.  Laplace retains only authorization-aware tool semantics.  Existing
Zetsu HTTP remains the authenticated backend and source of truth.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeAlias, cast

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from .ast_context import AstContextError, render_ast_context, tool_definition
from .zetsu_cli import _result, _rpc
from .zetsu_config import DEFAULT_ENDPOINT, DEFAULT_TOKEN_ENV
from .zetsu_mcp import MCP_LATEST_PROTOCOL_VERSION, ZETSU_INSTRUCTIONS, ZETSU_SCHEMA_VERSION
from .zetsu_runtime import default_state_root, load_local_plus_token

JsonObject: TypeAlias = dict[str, object]


class ZetsuSdkBridgeError(RuntimeError):
    pass


def _token(state_root: Path, token_env_var: str) -> str:
    value = os.environ.get(token_env_var)
    if value:
        return value
    try:
        return load_local_plus_token(state_root)
    except Exception as exc:  # narrow category differs across runtime revisions
        raise ZetsuSdkBridgeError("zetsu_local_token_unavailable") from exc


class ZetsuBackend:
    """Stateless client for the already-authenticated loopback Zetsu endpoint."""

    def __init__(self, endpoint: str, token: str, *, timeout: float = 30.0) -> None:
        self.endpoint = endpoint
        self.token = token
        self.timeout = min(max(float(timeout), 1.0), 120.0)

    def request(self, method: str, params: Mapping[str, object] | None = None) -> JsonObject:
        request_id = f"sdk-{uuid.uuid4().hex}"
        _, payload = _rpc(
            self.endpoint,
            self.token,
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": dict(params or {}),
            },
            timeout=self.timeout,
            protocol_version=MCP_LATEST_PROTOCOL_VERSION,
        )
        return _result(payload)

    def list_tools(self) -> list[JsonObject]:
        result = self.request("tools/list")
        raw = result.get("tools")
        if not isinstance(raw, list):
            raise ZetsuSdkBridgeError("zetsu_tools_list_invalid")
        tools: list[JsonObject] = []
        for item in raw:
            if isinstance(item, dict) and all(isinstance(key, str) for key in item):
                tools.append(cast(JsonObject, item))
        return tools

    def call_tool(self, name: str, arguments: Mapping[str, object]) -> JsonObject:
        return self.request("tools/call", {"name": name, "arguments": dict(arguments)})


def _mcp_call_result(value: Mapping[str, object]) -> types.CallToolResult:
    portable = {
        key: value[key]
        for key in ("content", "structuredContent", "isError", "_meta")
        if key in value
    }
    return types.CallToolResult.model_validate(portable)


def build_server(backend: ZetsuBackend, repository_root: Path) -> Server[Any]:
    """Build an SDK-owned MCP server while preserving Zetsu's exact tool schemas."""

    repository = repository_root.expanduser().resolve()

    async def list_tools(
        _ctx: Any, _params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        definitions = backend.list_tools()
        names = {str(item.get("name")) for item in definitions}
        # Structural context is exposed only to principals already authorized for
        # repository-agent use by the backend capability filter.
        if "agent_task" in names and "ast_context" not in names:
            definitions.append(tool_definition())
        return types.ListToolsResult(
            tools=[types.Tool.model_validate(item) for item in definitions]
        )

    async def call_tool(
        _ctx: Any, params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        arguments = params.arguments or {}
        try:
            if params.name == "ast_context":
                backend_names = {str(item.get("name")) for item in backend.list_tools()}
                if "agent_task" not in backend_names:
                    raise ZetsuSdkBridgeError("ast_context_not_authorized")
                path = arguments.get("path")
                pattern = arguments.get("pattern")
                ignore_case = arguments.get("ignore_case", False)
                max_chars = arguments.get("max_chars", 12_000)
                if not isinstance(path, str) or not isinstance(pattern, str):
                    raise AstContextError("ast_context_invalid_arguments")
                if not isinstance(ignore_case, bool) or not isinstance(max_chars, int):
                    raise AstContextError("ast_context_invalid_arguments")
                value = render_ast_context(
                    repository,
                    path,
                    pattern,
                    ignore_case=ignore_case,
                    max_chars=max_chars,
                )
                return types.CallToolResult.model_validate(
                    {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(value, ensure_ascii=False, sort_keys=True),
                            }
                        ],
                        "structuredContent": value,
                        "isError": False,
                    }
                )
            return _mcp_call_result(backend.call_tool(params.name, arguments))
        except (AstContextError, ZetsuSdkBridgeError, ValueError) as exc:
            error = {"error": str(exc)}
            return types.CallToolResult.model_validate(
                {
                    "content": [{"type": "text", "text": json.dumps(error)}],
                    "structuredContent": error,
                    "isError": True,
                }
            )

    return Server(
        "laplace-zetsu",
        version=ZETSU_SCHEMA_VERSION,
        instructions=ZETSU_INSTRUCTIONS,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


async def _serve(server: Server[Any]) -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="laplace-zetsu-mcp")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--state-root", type=Path, default=default_state_root())
    parser.add_argument("--endpoint", default=os.environ.get("LAPLACE_ZETSU_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--token-env-var", default=DEFAULT_TOKEN_ENV)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        backend = ZetsuBackend(
            args.endpoint,
            _token(args.state_root.expanduser().resolve(), args.token_env_var),
            timeout=args.timeout,
        )
        # Fail before entering stdio if the protected backend is not reachable or
        # the credential is invalid; Codex then reports a clean MCP startup failure.
        backend.list_tools()
        server = build_server(backend, args.repo)
        asyncio.run(_serve(server))
        return 0
    except (ZetsuSdkBridgeError, OSError, RuntimeError, ValueError) as exc:
        print(f"laplace-zetsu-mcp: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
