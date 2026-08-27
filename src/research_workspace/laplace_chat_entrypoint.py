"""Compatibility console entrypoint for ``laplace chat``, web and Codex.

The installer records the original console target in ORIGINAL_ENTRYPOINT.
All other invocations are forwarded unchanged.
"""
from __future__ import annotations

import importlib
import sys
from typing import Callable, cast

# Replaced deterministically by install.py.
ORIGINAL_ENTRYPOINT = 'research_workspace.entrypoints:laplace_main'


def _original() -> Callable[[], object]:
    if ORIGINAL_ENTRYPOINT == "__ORIGINAL_ENTRYPOINT__":
        raise RuntimeError("laplace_chat_entrypoint_not_installed")
    module_name, function_name = ORIGINAL_ENTRYPOINT.split(":", 1)
    module = importlib.import_module(module_name)
    function: object = getattr(module, function_name)
    if not callable(function):
        raise RuntimeError("original_laplace_entrypoint_not_callable")
    return cast(Callable[[], object], function)


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "chat":
        from .chat_cli import main as chat_main

        return int(chat_main(sys.argv[2:]))
    if len(sys.argv) >= 2 and sys.argv[1] == "web":
        from .laplace_web import main as web_main

        return int(web_main(sys.argv[2:]))
    if len(sys.argv) >= 2 and sys.argv[1] == "codex":
        from .zetsu_codex import main as codex_main

        return int(codex_main(sys.argv[2:]))
    if len(sys.argv) >= 2 and sys.argv[1] == "ast-context":
        from .ast_context import main as ast_context_main

        return int(ast_context_main(sys.argv[2:]))
    result = _original()()
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
