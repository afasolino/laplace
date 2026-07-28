#!/usr/bin/env python3
"""Compatibility executable for the identity-bound `laplace-migrate` CLI."""

from __future__ import annotations

from research_workspace.migration_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
