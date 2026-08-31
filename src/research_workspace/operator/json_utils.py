"""Exact JSON encodings shared by Operator transport routes."""

from __future__ import annotations

import json


def canonical_json_bytes(value: object) -> bytes:
    """Encode a value for stable identity binding without changing its JSON form."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sorted_json_text(value: object) -> str:
    """Encode an SSE payload using the established public route representation."""

    return json.dumps(value, sort_keys=True)
