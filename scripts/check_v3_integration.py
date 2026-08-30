#!/usr/bin/env python3
"""Smoke-check the declared non-GPU v3 integration dependencies."""

from __future__ import annotations

import importlib


def main() -> int:
    for name in ("gradio", "grep_ast", "mcp", "httpx2"):
        importlib.import_module(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
