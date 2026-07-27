#!/usr/bin/env python3
"""Keep owned dual GPU routes alive for the real HTTP/GUI certification."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from research_workspace.model_servers import ModelServerController, ModelServerSafetyError


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=root)
    parser.add_argument("--certification-root", type=Path, required=True)
    parser.add_argument("--live-api-name", default="live_api_v2")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    root = arguments.repository_root.resolve()
    certification_root = arguments.certification_root.resolve()
    state = certification_root / "live_routes_v2"
    live_api_root = certification_root / arguments.live_api_name
    lifecycle_path = (
        certification_root
        / f"live_integration_lifecycle_{arguments.live_api_name}.json"
    )
    controller = ModelServerController(root, state)
    lifecycle: dict[str, object] = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "start": None,
        "api_run": None,
        "release": None,
    }
    returncode = 2
    try:
        lifecycle["start"] = controller.start(startup_timeout_seconds=600)
        environment = dict(os.environ)
        environment.setdefault(
            "PLAYWRIGHT_BROWSERS_PATH", "/tmp/laplace-playwright-browsers"
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(root / "scripts/run_tiered_live_api_certification.py"),
                "--repository-root",
                str(root),
                "--output-root",
                str(live_api_root),
            ],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=3_600,
        )
        lifecycle["api_run"] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        returncode = completed.returncode
    except (ModelServerSafetyError, OSError, subprocess.TimeoutExpired) as exc:
        lifecycle["failure"] = {
            "category": str(getattr(exc, "category", type(exc).__name__)),
            "evidence": getattr(exc, "evidence", {}),
        }
    finally:
        try:
            lifecycle["release"] = controller.release_owned(timeout_seconds=90)
        except ModelServerSafetyError as exc:
            lifecycle["release"] = {
                "status": "FAILED",
                "category": exc.category,
                "evidence": exc.evidence,
            }
            returncode = 2
        lifecycle["finished_at_utc"] = datetime.now(UTC).isoformat()
        lifecycle_path.write_text(
            json.dumps(lifecycle, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "status": "PASS" if returncode == 0 else "FAIL",
                "live_api_root": str(live_api_root),
                "lifecycle": str(lifecycle_path),
            }
        )
    )
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
