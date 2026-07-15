from __future__ import annotations

from dataclasses import dataclass


class RecordError(ValueError):
    pass


@dataclass(frozen=True)
class Record:
    name: str
    count: int


def parse_record(text: str, line_number: int = 1) -> Record:
    """Parse ``name=<text>,count=<integer>`` into an immutable record."""
    try:
        fields = dict(part.split("=", 1) for part in text.split(","))
        return Record(name=fields["name"], count=int(fields["count"]))
    except (KeyError, ValueError) as exc:
        raise RecordError(f"line {line_number}: invalid record") from exc
