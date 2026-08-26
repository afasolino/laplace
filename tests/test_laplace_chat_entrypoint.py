from __future__ import annotations

import sys

import pytest

from research_workspace import laplace_chat_entrypoint as entrypoint


@pytest.mark.parametrize(
    "arguments",
    (
        ("zetsu",),
        ("zetsu", "start"),
        ("zetsu", "start", "--nocodev"),
        ("zetsu", "stop"),
        ("zetsu", "codex"),
        ("research", "status"),
    ),
)
def test_only_exact_chat_subcommand_is_intercepted(monkeypatch, arguments: tuple[str, ...]) -> None:
    calls: list[str] = []
    monkeypatch.setattr(sys, "argv", ["laplace", *arguments])
    monkeypatch.setattr(entrypoint, "_original", lambda: lambda: calls.append("original") or 7)
    assert entrypoint.main() == 7
    assert calls == ["original"]


def test_chat_subcommand_is_dispatched_without_rewriting_other_args(monkeypatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(sys, "argv", ["laplace", "chat", "--mode", "chat"])
    monkeypatch.setattr(entrypoint, "_original", lambda: lambda: calls.append("original") or 0)
    from research_workspace import chat_cli

    monkeypatch.setattr(chat_cli, "main", lambda argv: calls.append(argv) or 3)
    assert entrypoint.main() == 3
    assert calls == [["--mode", "chat"]]
