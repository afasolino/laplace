"""Deterministic, bounded labels for repository-agent objectives."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,31}")
_SKIP = frozenset(
    {
        "a",
        "an",
        "add",
        "analyse",
        "analyze",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "change",
        "check",
        "current",
        "describe",
        "do",
        "explain",
        "find",
        "fix",
        "focus",
        "for",
        "from",
        "how",
        "identify",
        "implement",
        "in",
        "inspect",
        "into",
        "investigate",
        "is",
        "it",
        "modify",
        "not",
        "of",
        "on",
        "or",
        "please",
        "report",
        "review",
        "show",
        "summarise",
        "summarize",
        "tell",
        "that",
        "the",
        "this",
        "to",
        "trace",
        "update",
        "where",
        "with",
        "works",
        "write",
        "you",
        "your",
    }
)
_MAX_LABEL_CHARS = 64
_MAX_WORDS = 4
_MIN_WORDS = 2


def _display_token(token: str) -> str:
    if token.isupper() and len(token) <= 8:
        return token
    if any(character in token for character in "._+"):
        return token
    return token[:1].upper() + token[1:].lower()


def derive_task_label(instruction: str) -> str:
    """Return a deterministic 2-4 word label without a model call."""

    tokens = _TOKEN_RE.findall(instruction.replace("\x00", " "))
    informative = [token for token in tokens if token.lower() not in _SKIP]
    selected = informative[:_MAX_WORDS]
    if not selected:
        selected = ["Repository", "Task"]
    elif len(selected) < _MIN_WORDS:
        selected.append("Task")
    rendered = [_display_token(token) for token in selected]
    while rendered and len(" ".join(rendered)) > _MAX_LABEL_CHARS:
        if len(rendered) > _MIN_WORDS:
            rendered.pop()
            continue
        overflow = len(" ".join(rendered)) - _MAX_LABEL_CHARS
        rendered[-1] = rendered[-1][: max(1, len(rendered[-1]) - overflow)]
    return " ".join(rendered)
