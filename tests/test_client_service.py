from __future__ import annotations

from pathlib import Path

import pytest

from research_workspace.client_service import ClientDeviceStore, ClientServiceError


def test_client_pair_operation_cancel_result_and_isolation(tmp_path: Path) -> None:
    store = ClientDeviceStore(tmp_path / "client/devices.sqlite3")
    device = store.pair(
        "user-a",
        name="workstation",
        capabilities={"tools": {"python": True}},
    )
    device_id = str(device["device_id"])

    operation = store.enqueue(
        "user-a",
        device_id,
        workspace_id="ws-" + "a" * 24,
        action="run",
        arguments={"argv": ["python", "-V"]},
    )
    operation_id = str(operation["operation_id"])
    claimed = store.claim("user-a", device_id)
    assert claimed is not None
    assert claimed["operation_id"] == operation_id
    assert claimed["state"] == "CLAIMED"

    cancelled = store.cancel("user-a", operation_id)
    assert cancelled["state"] == "CANCEL_REQUESTED"
    completed = store.complete(
        "user-a",
        device_id,
        operation_id,
        result={"cancelled": True},
        failed=False,
    )
    assert completed["state"] == "CANCELLED"

    with pytest.raises(ClientServiceError, match="client_device_not_found"):
        store.claim("user-b", device_id)
    with pytest.raises(ClientServiceError, match="client_operation_not_found"):
        store.get_operation("user-b", operation_id)


def test_client_repair_is_idempotent_and_revoke_cancels_pending(tmp_path: Path) -> None:
    store = ClientDeviceStore(tmp_path / "client.sqlite3")
    first = store.pair("owner", name="pc", capabilities={"version": 1})
    repaired = store.pair(
        "owner",
        name="pc-renamed",
        capabilities={"version": 2},
        device_id=str(first["device_id"]),
    )
    assert repaired["device_id"] == first["device_id"]
    assert repaired["name"] == "pc-renamed"

    operation = store.enqueue(
        "owner",
        str(first["device_id"]),
        workspace_id="ws-" + "b" * 24,
        action="read",
        arguments={"path": "a.txt"},
    )
    store.revoke("owner", str(first["device_id"]))
    assert store.get_operation("owner", str(operation["operation_id"]))["state"] == "CANCELLED"
    with pytest.raises(ClientServiceError, match="client_device_revoked"):
        store.heartbeat("owner", str(first["device_id"]), {})
