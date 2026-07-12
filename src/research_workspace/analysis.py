from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LogFinding:
    kind: str
    line: int
    message: str
    evidence: str


def parse_log(text: str, profile: str = "generic") -> dict[str, Any]:
    lines = text.splitlines()
    patterns = [
        ("python_exception", re.compile(r"^(?:Traceback|\w*(?:Error|Exception):)")),
        ("build_error", re.compile(r"\b(?:error|fatal error):", re.I)),
        ("git_error", re.compile(r"^fatal:\s", re.I)),
        (
            "timing_or_utilization",
            re.compile(r"\b(?:slack|utilization|latency|throughput|power|accuracy)\b", re.I),
        ),
    ]
    findings: list[LogFinding] = []
    metrics: list[dict[str, str]] = []
    for index, line in enumerate(lines, 1):
        for kind, pattern in patterns:
            if pattern.search(line):
                findings.append(LogFinding(kind, index, line.strip(), line))
                break
        for match in re.finditer(
            r"([A-Za-z][A-Za-z _/-]{1,30})\s*[:=]\s*(-?\d+(?:\.\d+)?)\s*([A-Za-z%/]+)?", line
        ):
            metrics.append(
                {
                    "name": match.group(1).strip(),
                    "value": match.group(2),
                    "unit": match.group(3) or "",
                }
            )
    return {
        "profile": profile,
        "first_actionable_error": asdict(findings[0]) if findings else None,
        "findings": [asdict(x) for x in findings],
        "metrics": metrics,
    }


def compare_runs(good: dict[str, Any], failed: dict[str, Any]) -> dict[str, Any]:
    good_metrics = {m["name"]: m for m in good.get("metrics", [])}
    failed_metrics = {m["name"]: m for m in failed.get("metrics", [])}
    changes = []
    for name in sorted(set(good_metrics) | set(failed_metrics)):
        if good_metrics.get(name) != failed_metrics.get(name):
            changes.append(
                {
                    "name": name,
                    "known_good": good_metrics.get(name),
                    "failed": failed_metrics.get(name),
                }
            )
    return {
        "observed": {
            "metric_changes": changes,
            "failed_first_error": failed.get("first_actionable_error"),
        },
        "interpretation": "Metric differences are candidates, not proven causes.",
        "proposed_diagnostics": [
            "Inspect the first actionable error and reproduce with the smallest bounded change."
        ],
    }


def analyze_file(path: Path, profile: str = "generic") -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {"records": value}
    return parse_log(path.read_text(encoding="utf-8", errors="replace"), profile)
