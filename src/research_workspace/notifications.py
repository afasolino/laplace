"""Optional, bounded local notifications that cannot affect measured results."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Mapping, TypeAlias
from urllib.parse import urlsplit

JsonObject: TypeAlias = dict[str, object]

_ALLOWED_EVENTS = frozenset(
    {
        "RUN_COMPLETE",
        "TERMINAL_FAILURE",
        "APPROVAL_REQUIRED",
        "RESEARCH_REPORT_READY",
        "CERTIFICATION_BUNDLE_READY",
        "GPU_RELEASED",
    }
)
_ALLOWED_METADATA = frozenset(
    {
        "run_id",
        "job_id",
        "status",
        "failure_category",
        "bundle_name",
        "local_gui_path",
        "approval_id",
    }
)


@dataclass(frozen=True)
class NotificationConfig:
    """Local notification configuration; disabled by default."""

    enabled: bool = False
    endpoint: str | None = None
    timeout_seconds: float = 3.0

    def __post_init__(self) -> None:
        if not 0.1 <= self.timeout_seconds <= 10:
            raise ValueError("notification timeout must be between 0.1 and 10 seconds")
        if not self.enabled:
            return
        if self.endpoint is None:
            raise ValueError("enabled notifications require an endpoint")
        parsed = urlsplit(self.endpoint)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("notification endpoint must use HTTP")
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("notification endpoint must be local")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("notification endpoint must not embed credentials")


class LocalNotificationAdapter:
    """Send metadata-only ntfy-compatible requests, returning warnings on failure."""

    def __init__(self, config: NotificationConfig = NotificationConfig()) -> None:
        self.config = config

    @staticmethod
    def _bounded_metadata(metadata: Mapping[str, object]) -> JsonObject:
        unexpected = sorted(set(metadata).difference(_ALLOWED_METADATA))
        if unexpected:
            raise ValueError(
                "notification metadata contains forbidden keys: "
                + ", ".join(unexpected)
            )
        bounded: JsonObject = {}
        for key, value in metadata.items():
            if isinstance(value, bool | int | float):
                bounded[key] = value
            elif isinstance(value, str):
                bounded[key] = value[:300]
            elif value is None:
                bounded[key] = None
            else:
                raise ValueError(f"notification metadata {key} must be scalar")
        return bounded

    def send(self, event: str, metadata: Mapping[str, object]) -> JsonObject:
        if event not in _ALLOWED_EVENTS:
            raise ValueError("unsupported notification event")
        bounded = self._bounded_metadata(metadata)
        if not self.config.enabled:
            return {
                "status": "DISABLED_NOOP",
                "event": event,
                "measured_result_affected": False,
            }
        assert self.config.endpoint is not None
        body = json.dumps(
            {
                "topic": "laplace-local",
                "title": event.replace("_", " ").title(),
                "message": json.dumps(bounded, sort_keys=True, separators=(",", ":")),
                "tags": ["computer"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        request = urllib.request.Request(  # nosec B310 - config enforces loopback.
            self.config.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(  # nosec B310 - config enforces loopback.
                request,
                timeout=self.config.timeout_seconds,
            ) as response:
                status_code = response.status
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            return {
                "status": "OPERATOR_WARNING",
                "event": event,
                "warning_category": "notification_delivery_failure",
                "error_type": type(exc).__name__,
                "measured_result_affected": False,
            }
        if not 200 <= status_code < 300:
            return {
                "status": "OPERATOR_WARNING",
                "event": event,
                "warning_category": "notification_http_failure",
                "http_status": status_code,
                "measured_result_affected": False,
            }
        return {
            "status": "DELIVERED",
            "event": event,
            "http_status": status_code,
            "measured_result_affected": False,
        }

