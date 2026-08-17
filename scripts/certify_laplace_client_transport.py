#!/usr/bin/env python3
"""Exercise real authenticated Laplace Client pairing and operation transport."""

from __future__ import annotations

import argparse
import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Sequence

from research_workspace.client_bridge import (
    DEFAULT_TOKEN_ENV,
    ClientConnection,
    LaplaceClientTransport,
    WorkspaceRegistry,
    pair_client,
    serve_once,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--token-env-var", default=DEFAULT_TOKEN_ENV)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="laplace-client-transport-", dir="/tmp") as raw:
        root = Path(raw) / "project"
        root.mkdir()
        (root / "input.txt").write_text("transport evidence\n", encoding="utf-8")
        registry = WorkspaceRegistry(Path(raw) / "state/workspaces.json")
        grant = registry.grant(root, writable=True, allowed_commands=("python3",))
        first = pair_client(
            registry,
            endpoint=arguments.endpoint,
            token_env_var=arguments.token_env_var,
            name="transport-certification",
        )
        second = pair_client(
            registry,
            endpoint=arguments.endpoint,
            token_env_var=arguments.token_env_var,
            name="transport-certification-reconnect",
        )
        device_id = str(first["device_id"])
        assert second["device_id"] == device_id
        connection = ClientConnection(
            endpoint=arguments.endpoint,
            token_env_var=arguments.token_env_var,
            device_id=device_id,
            name="transport-certification",
        )
        transport = LaplaceClientTransport(
            arguments.endpoint, arguments.token_env_var
        )

        read_operation = transport.request(
            "POST",
            f"/api/v1/client/devices/{device_id}/operations",
            {
                "workspace_id": grant.workspace_id,
                "action": "read",
                "arguments": {"path": "input.txt"},
            },
        )
        processed = serve_once(registry, connection)
        read_result = transport.request(
            "GET",
            f"/api/v1/client/operations/{read_operation['operation_id']}",
        )
        assert processed["status"] == "PROCESSED"
        assert read_result["state"] == "COMPLETE"
        assert read_result["result"] == {"content": "transport evidence\n"}

        run_operation = transport.request(
            "POST",
            f"/api/v1/client/devices/{device_id}/operations",
            {
                "workspace_id": grant.workspace_id,
                "action": "run",
                "arguments": {
                    "argv": ["python3", "-c", "import time; time.sleep(30)"],
                    "timeout": 60,
                },
            },
        )
        outcome: dict[str, object] = {}

        def worker() -> None:
            outcome.update(serve_once(registry, connection))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        operation_id = str(run_operation["operation_id"])
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            current = transport.request(
                "GET", f"/api/v1/client/operations/{operation_id}"
            )
            if current["state"] == "CLAIMED":
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("client_operation_not_claimed")
        transport.request(
            "POST", f"/api/v1/client/operations/{operation_id}/cancel", {}
        )
        thread.join(timeout=15)
        assert not thread.is_alive()
        cancelled = transport.request(
            "GET", f"/api/v1/client/operations/{operation_id}"
        )
        assert cancelled["state"] == "CANCELLED"
        transport.request("DELETE", f"/api/v1/client/devices/{device_id}")

        print(
            json.dumps(
                {
                    "status": "PASSED",
                    "pairing": True,
                    "reconnect_same_device": True,
                    "authenticated_operation": True,
                    "result_retrieval": True,
                    "cancellation": True,
                    "revoke": True,
                    "command_execution_location": "client",
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
