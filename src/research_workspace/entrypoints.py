"""Dependency-light console entry point dispatchers."""

from __future__ import annotations

from typing import Sequence


def laplace_main(argv: Sequence[str] | None = None) -> int:
    """Report package identity without importing optional runtime dependencies."""

    from .cli_registry import dispatch

    return dispatch(argv)
