from __future__ import annotations

import json
import urllib.error

import pytest

from research_workspace.notifications import (
    LocalNotificationAdapter,
    NotificationConfig,
)


def test_notifications_are_disabled_and_side_effect_free_by_default() -> None:
    result = LocalNotificationAdapter().send(
        "RUN_COMPLETE", {"run_id": "run-1", "status": "COMPLETE"}
    )
    assert result == {
        "status": "DISABLED_NOOP",
        "event": "RUN_COMPLETE",
        "measured_result_affected": False,
    }


def test_notification_configuration_rejects_remote_or_credentialed_endpoints() -> None:
    with pytest.raises(ValueError, match="local"):
        NotificationConfig(enabled=True, endpoint="https://ntfy.example/topic")
    with pytest.raises(ValueError, match="credentials"):
        NotificationConfig(enabled=True, endpoint="http://user:pass@localhost/topic")


def test_notification_metadata_is_allowlisted_and_bounded() -> None:
    adapter = LocalNotificationAdapter()
    with pytest.raises(ValueError, match="forbidden"):
        adapter.send("RUN_COMPLETE", {"prompt": "secret"})
    result = adapter.send("RUN_COMPLETE", {"run_id": "x" * 500})
    assert result["status"] == "DISABLED_NOOP"


def test_delivery_failure_is_only_an_operator_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> object:
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    adapter = LocalNotificationAdapter(
        NotificationConfig(enabled=True, endpoint="http://127.0.0.1:8123/laplace")
    )
    result = adapter.send(
        "TERMINAL_FAILURE",
        {"run_id": "run-1", "failure_category": "verification_failure"},
    )
    assert result["status"] == "OPERATOR_WARNING"
    assert result["measured_result_affected"] is False
    assert "offline" not in json.dumps(result)

