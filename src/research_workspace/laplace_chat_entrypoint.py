"""Compatibility console entrypoint for ``laplace chat``, web and Codex.

The installer records the original console target in ORIGINAL_ENTRYPOINT.
All other invocations are forwarded unchanged.
"""
from __future__ import annotations

from collections.abc import Callable


def _original() -> Callable[[], int]:
    """Compatibility hook retained for installers and legacy callers."""

    from .cli_registry import dispatch

    return dispatch

def main() -> int:
    import sys

    if len(sys.argv) >= 2 and sys.argv[1] in {"chat", "web", "codex", "ast-context"}:
        from .cli_registry import dispatch

        return dispatch()
    return _original()()


if __name__ == "__main__":
    raise SystemExit(main())
