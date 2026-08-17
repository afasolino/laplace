from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from research_workspace.zetsu_mcp import (
    MCP_LATEST_PROTOCOL_VERSION,
    ZetsuError,
    ZetsuMcpDispatcher,
    ZetsuService,
)
from research_workspace.service_tiers import ModelLane
from research_workspace.user_capabilities import Capability


class _Service:
    def available_tools(self, user_id: str) -> tuple[dict[str, object], ...]:
        assert user_id == "owner"
        return (
            {
                "name": "search",
                "description": "fixture",
                "inputSchema": {"type": "object"},
            },
        )

    def status(self, user_id: str) -> dict[str, object]:
        assert user_id == "owner"
        return {"status": "READY", "available_tools": ["search"]}

    def call(
        self, user_id: str, name: str, arguments: dict[str, object]
    ) -> dict[str, object]:
        assert (user_id, name) == ("owner", "search")
        assert arguments["max_chars"] == 512
        return {"grounded": False, "evidence": [], "max_chars": 512}


def test_legacy_initialize_discovery_and_tool_call() -> None:
    dispatcher = ZetsuMcpDispatcher(cast(object, _Service()))  # type: ignore[arg-type]
    initialized = dispatcher.dispatch(
        "owner",
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-11-25"},
        },
    )
    assert initialized.status_code == 200
    assert initialized.payload is not None
    assert initialized.payload["result"]["protocolVersion"] == "2025-11-25"  # type: ignore[index]

    listed = dispatcher.dispatch(
        "owner", {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    )
    assert listed.payload is not None
    assert listed.payload["result"]["tools"][0]["name"] == "search"  # type: ignore[index]

    called = dispatcher.dispatch(
        "owner",
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "search",
                "arguments": {"query": "q", "max_chars": 512},
            },
        },
    )
    assert called.payload is not None
    assert called.payload["result"]["isError"] is False  # type: ignore[index]


def test_modern_discovery_and_cacheable_tool_list() -> None:
    dispatcher = ZetsuMcpDispatcher(cast(object, _Service()))  # type: ignore[arg-type]
    discovered = dispatcher.dispatch(
        "owner",
        {"jsonrpc": "2.0", "id": "d", "method": "server/discover", "params": {}},
        protocol_header=MCP_LATEST_PROTOCOL_VERSION,
    )
    assert discovered.payload is not None
    assert discovered.payload["result"]["resultType"] == "complete"  # type: ignore[index]
    listed = dispatcher.dispatch(
        "owner",
        {"jsonrpc": "2.0", "id": "l", "method": "tools/list", "params": {}},
        protocol_header=MCP_LATEST_PROTOCOL_VERSION,
    )
    assert listed.payload is not None
    assert listed.payload["result"]["cacheScope"] == "private"  # type: ignore[index]


class _RtlTiered:
    def __init__(self) -> None:
        self.called: dict[str, object] | None = None

    def effective_capabilities(self, _user_id: str) -> frozenset[Capability]:
        return frozenset({Capability.AGENT})

    def agent(self, **kwargs: object) -> dict[str, object]:
        self.called = kwargs
        return {
            "status": "SUCCESS",
            "session_id": kwargs["session_id"],
            "model_id": "laplace-codev",
            "effective_lane": "economy",
            "response": {"content": "patch"},
        }


def test_rtl_task_is_codev_routed_only_after_bounded_policy_check() -> None:
    tiered = _RtlTiered()
    service = ZetsuService(Path.cwd(), cast(Any, None), cast(Any, tiered))
    arguments = {
        "session_id": "rtl-session",
        "instruction": "Implement the specified counter.",
        "task_kind": "implementation",
        "rtl_scope": "bounded_module",
        "editable_sources": ["counter.sv"],
        "module_count": 1,
        "max_chars": 512,
    }
    result = service.call("owner", "rtl_task", arguments)
    assert result["model_id"] == "laplace-codev"
    assert tiered.called is not None
    assert tiered.called["lane"] is ModelLane.ECONOMY

    with pytest.raises(ZetsuError, match="invalid_rtl_scope"):
        service.call("owner", "rtl_task", {**arguments, "rtl_scope": "multi_file_subsystem"})

    with pytest.raises(ZetsuError, match="unexpected_tool_arguments"):
        service.call("owner", "rtl_task", {**arguments, "shell": "uname -a"})
