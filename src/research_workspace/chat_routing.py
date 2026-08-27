"""Deterministic per-message capability routing for Laplace v3.1.

Routing is deliberately local and model-free.  It classifies the current user
message only; previous repository-agent observations are never input features.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, TypeAlias

RouteOverride: TypeAlias = Literal["auto", "chat", "agent", "retrieval", "corpus", "runtime"]
ResolvedRoute: TypeAlias = Literal["chat", "agent", "retrieval", "corpus", "runtime"]


@dataclass(frozen=True)
class RouteDecision:
    route: ResolvedRoute
    reason: str


_RUNTIME_PATTERNS = (
    re.compile(r"\b(?:tokens?|tok)\s*(?:/|per\s+)s(?:ec(?:ond)?)?\b", re.IGNORECASE),
    re.compile(r"\b(?:throughput|decode\s+rate|serving\s+rate)\b", re.IGNORECASE),
    re.compile(r"\b(?:gpu|vram)\s+(?:memory|utili[sz]ation|usage)\b", re.IGNORECASE),
    re.compile(r"\bmtp(?:\s+depth)?\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+model\s+(?:are\s+you|is\s+laplace)\s+(?:using|running)\b", re.IGNORECASE),
)
_CORPUS_PATTERNS = (
    re.compile(r"\bwhat\s+is\s+(?:in\s+)?my\s+corpus\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+is\s+my\s+corpus\s+about\b", re.IGNORECASE),
    re.compile(r"\b(?:show|list|describe)\s+(?:me\s+)?my\s+(?:corpus|corpora)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(?:documents?|files?)\s+are\s+in\s+my\s+(?:corpus|corpora|uploads?)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+have\s+i\s+uploaded\b", re.IGNORECASE),
)
_RETRIEVAL_PATTERNS = (
    re.compile(r"\b(?:search|find|look\s+up)\b.{0,40}\bmy\s+(?:corpus|notes?|documents?|uploads?)\b", re.IGNORECASE),
    re.compile(r"\b(?:according\s+to|based\s+on)\s+my\s+(?:notes?|documents?|corpus)\b", re.IGNORECASE),
    re.compile(r"\bin\s+my\s+(?:notes?|documents?|corpus)\b", re.IGNORECASE),
)
_PATH_PATTERN = re.compile(
    r"(?:^|\s)(?:[A-Za-z0-9_.-]+/)+(?:[A-Za-z0-9_.-]+\.(?:py|md|toml|json|ya?ml|rs|go|c|cc|cpp|h|hpp|sv|v))\b",
    re.IGNORECASE,
)
_CODE_FILE_PATTERN = re.compile(r"\b[A-Za-z0-9_.-]+\.(?:py|toml|json|ya?ml|rs|go|c|cc|cpp|h|hpp|sv|v)\b", re.IGNORECASE)
_ENGINEERING_VERB = re.compile(r"\b(?:fix|edit|change|modify|implement|refactor|debug|test|inspect|review)\b", re.IGNORECASE)
_ENGINEERING_OBJECT = re.compile(r"\b(?:code|test|tests|file|function|class|module|repository|repo|worktree|diff|scheduler|authorization)\b", re.IGNORECASE)
_SYMBOL_QUERY = re.compile(r"\b(?:where\s+is|find|locate)\b.{0,80}\b(?:defined|definition|implemented|function|class|symbol)\b", re.IGNORECASE)


def route_message(text: str, override: RouteOverride = "auto") -> RouteDecision:
    """Resolve one message without using prior semantic state or a model call."""
    if override != "auto":
        return RouteDecision(route=override, reason="explicit_override")
    value = text.strip()
    if any(pattern.search(value) for pattern in _RUNTIME_PATTERNS):
        return RouteDecision(route="runtime", reason="runtime_telemetry_query")
    if any(pattern.search(value) for pattern in _CORPUS_PATTERNS):
        return RouteDecision(route="corpus", reason="corpus_introspection_query")
    if any(pattern.search(value) for pattern in _RETRIEVAL_PATTERNS):
        return RouteDecision(route="retrieval", reason="personal_corpus_retrieval_query")
    if _PATH_PATTERN.search(value) or _CODE_FILE_PATTERN.search(value):
        return RouteDecision(route="agent", reason="repository_path_or_source_reference")
    if _ENGINEERING_VERB.search(value) and _ENGINEERING_OBJECT.search(value):
        return RouteDecision(route="agent", reason="repository_engineering_operation")
    if _SYMBOL_QUERY.search(value):
        return RouteDecision(route="agent", reason="repository_symbol_query")
    return RouteDecision(route="chat", reason="general_conversation_fallback")
