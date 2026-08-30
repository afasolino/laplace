"""Safe research-record presentation helpers for Operator transport routes."""

from __future__ import annotations

import json
from pathlib import Path


def research_summaries(root: Path) -> list[dict[str, object]]:
    """List bounded public research-job summaries without server paths."""

    summaries: list[dict[str, object]] = []
    if not root.is_dir():
        return summaries
    for path in sorted(root.glob("*/job.json"), reverse=True)[:20]:
        try:
            value: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            summaries.append(
                {
                    key: value.get(key)
                    for key in (
                        "research_job_id", "question", "research_mode", "status",
                        "current_stage", "created_at",
                    )
                }
            )
    return summaries


def sanitize_research_payload(value: object) -> object:
    """Remove local paths while retaining cited evidence metadata."""

    if isinstance(value, list):
        return [sanitize_research_payload(item) for item in value]
    if not isinstance(value, dict):
        return value
    sanitized: dict[str, object] = {}
    for raw_key, item in value.items():
        key = str(raw_key)
        if key in {"local_snapshot_path", "report_path", "evidence_ledger_path"}:
            continue
        if key == "canonical_url" and isinstance(item, str) and item.startswith("file:"):
            sanitized[key] = "local-document://authorized-source"
            continue
        sanitized[key] = sanitize_research_payload(item)
    return sanitized
