"""Durable checkpoint storage for bounded Zetsu agent runs."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Mapping, cast

from .service_tiers import ServiceTierError

JsonObject = dict[str, object]
MAX_CHECKPOINT_BYTES = 2 * 1024 * 1024


class AgentCheckpointStore:
    """Atomic owner-independent files keyed by opaque session digest."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            os.chmod(self.root, 0o700)

    def path(self, session_id: str) -> Path:
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"

    def write(self, session_id: str, value: Mapping[str, object]) -> Path:
        target = self.path(session_id)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(dict(value), handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def read(self, session_id: str) -> JsonObject | None:
        target = self.path(session_id)
        if not target.is_file():
            return None
        try:
            size = target.stat().st_size
            if size > MAX_CHECKPOINT_BYTES:
                raise ServiceTierError("zetsu_agent_checkpoint_too_large")
            value: object = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ServiceTierError("zetsu_agent_checkpoint_invalid") from exc
        if not isinstance(value, dict):
            raise ServiceTierError("zetsu_agent_checkpoint_invalid")
        return cast(JsonObject, value)
