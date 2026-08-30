"""Canonical command registry for the ``laplace`` console script."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path


Command = Callable[[Sequence[str]], int]


def _chat(argv: Sequence[str]) -> int:
    from .chat_cli import main as chat_main

    return int(chat_main(list(argv)))


def _web(argv: Sequence[str]) -> int:
    from .laplace_web import main as web_main

    return int(web_main(list(argv)))


def _codex(argv: Sequence[str]) -> int:
    from .zetsu_codex import main as codex_main

    return int(codex_main(list(argv)))


def _ast_context(argv: Sequence[str]) -> int:
    from .ast_context import main as ast_context_main

    return int(ast_context_main(list(argv)))


def dispatch(argv: Sequence[str] | None = None) -> int:
    """Run the one authoritative command mapping with compatibility aliases."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    commands: dict[str, Command] = {
        "chat": _chat,
        "web": _web,
        "codex": _codex,
        "ast-context": _ast_context,
    }
    if arguments and arguments[0] in commands:
        return commands[arguments[0]](arguments[1:])
    if arguments == ["--version"]:
        from .versioning import version_line

        print(version_line(Path.cwd()))
        return 0
    if arguments and arguments[0] == "zetsu":
        from .zetsu_cli import main as zetsu_main

        return int(zetsu_main(arguments[1:]))
    from .laplace_cli import main as laplace_main

    return int(laplace_main(arguments))
