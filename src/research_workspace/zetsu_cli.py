"""Idempotent Codex configuration and live diagnostics for Zetsu."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Mapping, Sequence, cast
from urllib.parse import urlsplit, urlunsplit

from .zetsu_config import DEFAULT_ENDPOINT, DEFAULT_TOKEN_ENV, ZetsuConfigError
from .zetsu_config import configure as configure_zetsu
from .zetsu_config import remove as remove_zetsu
from .zetsu_config import status as zetsu_status

LEGACY_PROTOCOL_VERSION = "2025-11-25"


def _default_endpoint() -> str:
    configured = os.environ.get("LAPLACE_ZETSU_ENDPOINT")
    if configured:
        return configured
    external = os.environ.get("LAPLACE_EXTERNAL_URL")
    return external.rstrip("/") + "/mcp" if external else DEFAULT_ENDPOINT


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="laplace zetsu")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("configure", "status", "test", "remove"):
        item = sub.add_parser(name)
        item.add_argument("--repo", type=Path, default=Path.cwd())
        item.add_argument("--json", action="store_true")
    configure = sub.choices["configure"]
    configure.add_argument("--endpoint", default=_default_endpoint())
    configure.add_argument("--token-env-var", default=DEFAULT_TOKEN_ENV)
    for name in ("status", "test"):
        item = sub.choices[name]
        item.add_argument("--offline", action="store_true")
        item.add_argument("--timeout", type=float, default=10.0)
    return parser


def _emit(payload: Mapping[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        print(f"{key}: {value}")


def _decode_response(body: bytes, content_type: str) -> dict[str, object]:
    text = body.decode("utf-8", errors="strict")
    if "text/event-stream" in content_type:
        candidates = [line[5:].strip() for line in text.splitlines() if line.startswith("data:")]
        if not candidates:
            raise ZetsuConfigError("mcp_sse_response_missing_data")
        text = candidates[-1]
    raw: object = json.loads(text)
    if not isinstance(raw, dict):
        raise ZetsuConfigError("mcp_response_not_object")
    return cast(dict[str, object], raw)


def _rpc(
    endpoint: str,
    token: str,
    payload: Mapping[str, object],
    *,
    timeout: float,
    protocol_version: str | None = None,
) -> tuple[int, dict[str, object] | None]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "User-Agent": "laplace-zetsu-cli/1.1",
    }
    if protocol_version is not None:
        headers["MCP-Protocol-Version"] = protocol_version
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        method="POST",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
            body = response.read(1_000_001)
            if len(body) > 1_000_000:
                raise ZetsuConfigError("mcp_response_too_large")
            if not body:
                return response.status, None
            return response.status, _decode_response(body, response.headers.get("Content-Type", ""))
    except urllib.error.HTTPError as exc:
        detail = exc.read(4_096).decode("utf-8", errors="replace")
        raise ZetsuConfigError(f"mcp_http_error:{exc.code}:{detail[:200]}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ZetsuConfigError(f"mcp_connection_failed:{type(exc).__name__}") from exc


def _result(payload: dict[str, object] | None) -> dict[str, object]:
    if payload is None:
        raise ZetsuConfigError("mcp_empty_response")
    error = payload.get("error")
    if error is not None:
        raise ZetsuConfigError(f"mcp_rpc_error:{json.dumps(error, sort_keys=True)[:400]}")
    value = payload.get("result")
    if not isinstance(value, dict):
        raise ZetsuConfigError("mcp_result_missing")
    return cast(dict[str, object], value)


def _online_probe(
    endpoint: str,
    token_env_var: str,
    timeout: float,
    *,
    retrieval: bool,
) -> dict[str, object]:
    token = os.environ.get(token_env_var)
    if not token:
        raise ZetsuConfigError(f"missing_token_env:{token_env_var}")
    _, initialized = _rpc(
        endpoint,
        token,
        {
            "jsonrpc": "2.0",
            "id": "zetsu-initialize",
            "method": "initialize",
            "params": {
                "protocolVersion": LEGACY_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "laplace-zetsu-test", "version": "1.1"},
            },
        },
        timeout=timeout,
    )
    initialized_result = _result(initialized)
    negotiated = str(initialized_result.get("protocolVersion", LEGACY_PROTOCOL_VERSION))
    _rpc(
        endpoint,
        token,
        {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
        timeout=timeout,
        protocol_version=negotiated,
    )
    _, listed = _rpc(
        endpoint,
        token,
        {"jsonrpc": "2.0", "id": "zetsu-tools", "method": "tools/list", "params": {}},
        timeout=timeout,
        protocol_version=negotiated,
    )
    listed_result = _result(listed)
    raw_tools = listed_result.get("tools")
    tools = [
        str(item["name"])
        for item in raw_tools
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    ] if isinstance(raw_tools, list) else []
    retrieval_ok: bool | None = None
    if retrieval:
        if "search" not in tools:
            raise ZetsuConfigError("mcp_search_tool_unavailable")
        _, called = _rpc(
            endpoint,
            token,
            {
                "jsonrpc": "2.0",
                "id": "zetsu-search",
                "method": "tools/call",
                "params": {
                    "name": "search",
                    "arguments": {
                        "query": "zetsu connectivity probe",
                        "max_results": 1,
                        "max_chars": 512,
                    },
                },
            },
            timeout=timeout,
            protocol_version=negotiated,
        )
        called_result = _result(called)
        retrieval_ok = called_result.get("isError") is False
        if not retrieval_ok:
            raise ZetsuConfigError("mcp_retrieval_check_failed")
    return {
        "reachable": True,
        "protocol_version": negotiated,
        "server_info": initialized_result.get("serverInfo"),
        "available_tools": tools,
        "retrieval_check": retrieval_ok,
    }


def _status_url(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    return urlunsplit((parsed.scheme, parsed.netloc, "/api/v1/zetsu/status", "", ""))


def _readiness_url(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    return urlunsplit((parsed.scheme, parsed.netloc, "/api/v1/readiness", "", ""))


def _laplace_status(endpoint: str, token_env_var: str, timeout: float) -> dict[str, object]:
    token = os.environ.get(token_env_var)
    if not token:
        raise ZetsuConfigError(f"missing_token_env:{token_env_var}")
    request = urllib.request.Request(
        _status_url(endpoint),
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
            raw: object = json.loads(response.read(1_000_000))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise ZetsuConfigError(f"laplace_status_failed:{type(exc).__name__}") from exc
    if not isinstance(raw, dict):
        raise ZetsuConfigError("laplace_status_invalid")
    return cast(dict[str, object], raw)


def _laplace_readiness(endpoint: str, timeout: float) -> dict[str, object]:
    request = urllib.request.Request(
        _readiness_url(endpoint), headers={"Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
            body = response.read(1_000_000)
    except urllib.error.HTTPError as exc:
        if exc.code != 503:
            raise ZetsuConfigError(f"laplace_readiness_failed:http_{exc.code}") from exc
        body = exc.read(1_000_000)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ZetsuConfigError(f"laplace_readiness_failed:{type(exc).__name__}") from exc
    try:
        raw: object = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ZetsuConfigError("laplace_readiness_invalid") from exc
    if not isinstance(raw, dict):
        raise ZetsuConfigError("laplace_readiness_invalid")
    reasons = raw.get("reasons")
    reason_values = [str(item) for item in reasons] if isinstance(reasons, list) else []
    qwen_failures = [
        item
        for item in reason_values
        if item.endswith(":quality") or item.endswith(":standard")
    ]
    codev_failures = [item for item in reason_values if item.endswith(":economy")]
    model_endpoints_required = (
        raw.get("model_endpoints_required") is True and raw.get("fixture_mode") is not True
    )
    return {
        **cast(dict[str, object], raw),
        "qwen_ready": model_endpoints_required and not qwen_failures,
        "qwen_failures": qwen_failures,
        "codev_ready": model_endpoints_required and not codev_failures,
        "codev_failures": codev_failures,
    }


def _codex_recognition(repository: Path) -> dict[str, object]:
    try:
        completed = subprocess.run(
            ["codex", "mcp", "get", "zetsu", "--json"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"recognized": False, "detail": type(exc).__name__}
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        return {"recognized": False, "detail": detail[-1][:300] if detail else "not_found"}
    try:
        value: object = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"recognized": False, "detail": "invalid_codex_json"}
    return {"recognized": isinstance(value, dict), "configuration": value}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    repository = args.repo.resolve()
    try:
        if args.command == "configure":
            value = configure_zetsu(
                repository,
                endpoint=args.endpoint,
                token_env_var=args.token_env_var,
            )
            payload: dict[str, object] = {
                "ok": value.configured and value.skill_installed and value.compatible,
                "action": "configure",
                **value.as_dict(),
                "codex": _codex_recognition(repository),
            }
        elif args.command == "remove":
            value = remove_zetsu(repository)
            payload = {
                "ok": not value.configured and not value.skill_installed,
                "action": "remove",
                **value.as_dict(),
            }
        else:
            value = zetsu_status(repository)
            local_ok = value.configured and value.skill_installed and value.compatible
            online: dict[str, object] | None = None
            laplace: dict[str, object] | None = None
            readiness: dict[str, object] | None = None
            detail = "offline"
            online_ok = True
            if not args.offline and local_ok and value.endpoint and value.token_env_var:
                try:
                    online = _online_probe(
                        value.endpoint,
                        value.token_env_var,
                        args.timeout,
                        retrieval=args.command == "test",
                    )
                    laplace = _laplace_status(
                        value.endpoint,
                        value.token_env_var,
                        args.timeout,
                    )
                    readiness = _laplace_readiness(value.endpoint, args.timeout)
                    detail = "ok"
                except ZetsuConfigError as exc:
                    online_ok = False
                    detail = str(exc)
            elif not local_ok:
                online_ok = False
                detail = "configuration_incomplete_or_incompatible"
            payload = {
                "ok": local_ok and online_ok,
                "action": args.command,
                "local_configuration": local_ok,
                "online": online,
                "laplace": laplace,
                "readiness": readiness,
                "detail": detail,
                **value.as_dict(),
                "codex": _codex_recognition(repository),
            }
        _emit(payload, as_json=args.json)
        return 0 if bool(payload["ok"]) else 2
    except (ZetsuConfigError, ValueError) as exc:
        payload = {"ok": False, "error": str(exc)}
        _emit(payload, as_json=getattr(args, "json", False))
        return 2


if __name__ == "__main__":
    sys.exit(main())
