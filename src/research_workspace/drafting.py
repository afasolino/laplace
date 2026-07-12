from __future__ import annotations

import re
from typing import Any


def claim_source_map(claims: list[str], packet: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = packet.get("evidence", [])
    return [
        {"claim": claim, "sources": evidence if evidence else [], "supported": bool(evidence)}
        for claim in claims
    ]


def ground_text(text: str, packet: dict[str, Any]) -> str:
    if not packet.get("evidence"):
        return text + " [SOURCE REQUIRED]"
    return text


def preserve_numbers(original: str, revised: str) -> list[str]:
    numbers = set(re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:\s*[A-Za-z%]+)?", original))
    return sorted(
        numbers - set(re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:\s*[A-Za-z%]+)?", revised))
    )
