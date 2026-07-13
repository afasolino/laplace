"""Public fixture for a safe-path, atomic-output CLI task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_json(root: Path, relative_output: str, payload: dict[str, object]) -> Path:
    """Write JSON below root and return the output path."""
    # Intentional seeded defect: traversal is possible and replacement is not atomic.
    target = root / relative_output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    write_json(args.root, args.output, {"status": "WRITTEN"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
