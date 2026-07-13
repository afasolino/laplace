"""Public fixture for constrained local JSON output."""

import json
from pathlib import Path
from typing import Any


def write_json_output(root: Path, relative_name: str, payload: dict[str, Any]) -> Path:
    """Write a JSON result under root and return the created path."""
    target = root / relative_name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return target
