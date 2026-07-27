#!/usr/bin/env python3
"""Read-only GPU and Laplace profile ownership monitor."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

from research_workspace.serving_profile_runtime import (
    ServingProfileRuntime,
    ServingRuntimeError,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    parser.add_argument("--count", type=int, default=1)
    arguments = parser.parse_args(argv)
    runtime = ServingProfileRuntime(arguments.state_root)
    for index in range(arguments.count):
        try:
            result = runtime.status()
        except ServingRuntimeError as exc:
            result = {
                "status": "ERROR",
                "failure_category": exc.category,
                "evidence": exc.evidence,
            }
        print(json.dumps(result, sort_keys=True))
        if index + 1 < arguments.count:
            time.sleep(arguments.interval_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
