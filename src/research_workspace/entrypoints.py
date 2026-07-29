"""Dependency-light console entry point dispatchers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


def laplace_main(argv: Sequence[str] | None = None) -> int:
    """Report package identity without importing optional runtime dependencies."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["--version"]:
        from .versioning import version_line

        print(version_line(Path.cwd()))
        return 0
    from .laplace_cli import main

    return main(arguments)
