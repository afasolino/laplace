"""prompt_toolkit-backed input for ``laplace chat``.

The dependency supplies mature bracketed-paste, history and completion support;
Laplace keeps only the tiny policy layer that decides which keys submit a turn.
"""
from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import FuzzyCompleter, WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.shortcuts import CompleteStyle

from .chat_discovery import command_names

_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


def _history_path(state_root: Path, repo_id: str) -> Path:
    safe_repo = _SAFE_COMPONENT.sub("_", repo_id)[:128] or "repository"
    folder = state_root.resolve() / "chat_history_v3"
    folder.mkdir(parents=True, exist_ok=True)
    try:
        folder.chmod(0o700)
    except OSError:
        pass
    path = folder / f"{safe_repo}.history"
    if not path.exists():
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
        os.close(descriptor)
    else:
        try:
            path.chmod(0o600)
        except OSError:
            pass
    return path


def build_chat_input(state_root: Path, repo_id: str) -> Callable[[], str]:
    """Return one blocking input function.

    Enter submits. Alt+Enter and Ctrl+J insert a newline. Bracketed paste is
    handled by prompt_toolkit as one buffer operation, so pasted paragraphs are
    not split into multiple Laplace turns.
    """

    bindings = KeyBindings()

    @bindings.add("enter")
    def _submit(event: KeyPressEvent) -> None:
        event.current_buffer.validate_and_handle()

    @bindings.add("escape", "enter")
    def _newline_alt(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

    @bindings.add("c-j")
    def _newline_ctrl(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

    toolbar = f"repo={repo_id} · /help · /frontends · Alt+Enter newline"
    session: PromptSession[str] = PromptSession(
        multiline=True,
        history=FileHistory(str(_history_path(state_root, repo_id))),
        completer=FuzzyCompleter(
            WordCompleter(command_names(), ignore_case=True, sentence=True)
        ),
        auto_suggest=AutoSuggestFromHistory(),
        complete_style=CompleteStyle.MULTI_COLUMN,
        complete_while_typing=False,
        key_bindings=bindings,
        enable_history_search=True,
        enable_open_in_editor=False,
        bottom_toolbar=toolbar,
    )

    def read() -> str:
        return session.prompt("laplace> ")

    return read
