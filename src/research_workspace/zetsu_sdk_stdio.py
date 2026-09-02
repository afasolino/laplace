"""Official-MCP-SDK stdio bridge from Codex/Hermes to authenticated Zetsu."""
from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import sys
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, TypeAlias, cast
from urllib.parse import urlsplit

import httpx2
from mcp import Client, types
from mcp.client.streamable_http import streamable_http_client
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from .ast_context import AstContextError, render_ast_context, tool_definition
from .zetsu_config import DEFAULT_ENDPOINT, DEFAULT_TOKEN_ENV
from .zetsu_mcp import ZETSU_INSTRUCTIONS, ZETSU_SCHEMA_VERSION
from .zetsu_runtime import default_state_root, load_local_plus_token

JsonObject: TypeAlias = dict[str, object]
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class ZetsuSdkBridgeError(RuntimeError):
    pass


def _token(state_root: Path, token_env_var: str) -> str:
    value = os.environ.get(token_env_var)
    if value:
        return value
    try:
        return load_local_plus_token(state_root)
    except Exception as exc:
        raise ZetsuSdkBridgeError("zetsu_local_token_unavailable") from exc


def _loopback_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in _LOOPBACK_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or bool(parsed.query)
        or bool(parsed.fragment)
    ):
        raise ZetsuSdkBridgeError("zetsu_backend_loopback_only")
    return endpoint


@asynccontextmanager
async def _sdk_client(endpoint: str, token: str, timeout: float) -> AsyncIterator[Client]:
    async with httpx2.AsyncClient(
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
        follow_redirects=False,
    ) as http_client:
        transport = streamable_http_client(endpoint, http_client=http_client)
        async with Client(transport, read_timeout_seconds=timeout) as client:
            yield client


async def _resolve(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


class ZetsuBackend:
    def __init__(self, endpoint: str, token: str, *, timeout: float = 30.0) -> None:
        self.endpoint = _loopback_endpoint(endpoint)
        self.token = token
        self.timeout = min(max(float(timeout), 1.0), 120.0)

    def _tool_timeout(
        self,
        name: str,
        arguments: Mapping[str, object],
    ) -> float:
        if name != "agent_task":
            return self.timeout
        raw_wait = arguments.get("wait_timeout_seconds", 1_800)
        if isinstance(raw_wait, bool) or not isinstance(raw_wait, (int, float)):
            return self.timeout
        wait = min(max(float(raw_wait), 1.0), 3_600.0)
        return min(max(self.timeout, wait + 30.0), 3_630.0)

    async def list_tools(self) -> list[JsonObject]:
        try:
            async with _sdk_client(self.endpoint, self.token, self.timeout) as client:
                result = await client.list_tools()
        except Exception as exc:
            raise ZetsuSdkBridgeError(
                f"zetsu_backend_unavailable:{type(exc).__name__}"
            ) from exc
        tools: list[JsonObject] = []
        for item in result.tools:
            raw = item.model_dump(mode="json", by_alias=True, exclude_none=True)
            if isinstance(raw, dict) and all(isinstance(key, str) for key in raw):
                tools.append(cast(JsonObject, raw))
        return tools

    async def call_tool(self, name: str, arguments: Mapping[str, object]) -> JsonObject:
        try:
            timeout = self._tool_timeout(name, arguments)
            async with _sdk_client(self.endpoint, self.token, timeout) as client:
                result = await client.call_tool(name, dict(arguments))
        except Exception as exc:
            raise ZetsuSdkBridgeError(
                f"zetsu_backend_unavailable:{type(exc).__name__}"
            ) from exc
        raw = result.model_dump(mode="json", by_alias=True, exclude_none=True)
        if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
            raise ZetsuSdkBridgeError("zetsu_tool_result_invalid")
        return cast(JsonObject, raw)


def _mcp_call_result(value: Mapping[str, object]) -> types.CallToolResult:
    portable = {
        key: value[key]
        for key in ("content", "structuredContent", "isError", "_meta")
        if key in value
    }
    return types.CallToolResult.model_validate(portable)


def build_server(backend: Any, repository_root: Path) -> Server[Any]:
    repository = repository_root.expanduser().resolve()

    async def list_tools(
        _ctx: Any, _params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        definitions = cast(list[JsonObject], await _resolve(backend.list_tools()))
        names = {str(item.get("name")) for item in definitions}
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
                definitions = cast(list[JsonObject], await _resolve(backend.list_tools()))
                if "agent_task" not in {str(item.get("name")) for item in definitions}:
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
                        "content": [{
                            "type": "text",
                            "text": json.dumps(value, ensure_ascii=False, sort_keys=True),
                        }],
                        "structuredContent": value,
                        "isError": False,
                    }
                )
            result = cast(
                Mapping[str, object],
                await _resolve(backend.call_tool(params.name, arguments)),
            )
            return _mcp_call_result(result)
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


async def _run_bridge(backend: ZetsuBackend, repository_root: Path) -> None:
    await backend.list_tools()
    await _serve(build_server(backend, repository_root))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="laplace-zetsu-mcp")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--state-root", type=Path, default=default_state_root())
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("LAPLACE_ZETSU_ENDPOINT", DEFAULT_ENDPOINT),
    )
    parser.add_argument("--token-env-var", default=DEFAULT_TOKEN_ENV)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        backend = ZetsuBackend(
            _loopback_endpoint(args.endpoint),
            _token(args.state_root.expanduser().resolve(), args.token_env_var),
            timeout=args.timeout,
        )
        asyncio.run(_run_bridge(backend, args.repo))
        return 0
    except (ZetsuSdkBridgeError, OSError, RuntimeError, ValueError) as exc:
        print(f"laplace-zetsu-mcp: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
