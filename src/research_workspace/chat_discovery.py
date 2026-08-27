"""Deterministic CLI discovery for ``laplace chat``.

This module intentionally renders only capabilities already exposed by the
resident Operator.  It does not create another skill registry or model call.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

COMMAND_HELP: tuple[str, ...] = (
    "/help",
    "/skills",
    "/capabilities",
    "/verification",
    "/frontends",
    "/route auto|chat|agent|retrieval|corpus|runtime",
    "/mode agent|chat",
    "/access read|confirm|write",
    "/status",
    "/tasks",
    "/diff",
    "/tests",
    "/result <artifact> [offset]",
    "/cancel",
    "/watch",
    "/contract",
    "/context",
    "/history",
    "/compact",
    "/model [quality|standard|economy]",
    "/new",
    "/resume <session-id|last>",
    "/exit",
)


def command_names() -> tuple[str, ...]:
    """Return the slash commands used by completion and help."""

    return tuple(item.split(maxsplit=1)[0] for item in COMMAND_HELP)


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str) and item)


def render_frontends() -> str:
    """Render mature upstream-backed user surfaces without a model call."""

    return (
        "Upstream-backed frontends and context:\n"
        "- terminal: laplace chat  (prompt_toolkit)\n"
        "- web: laplace web  (Gradio; loopback only)\n"
        "- Codex: laplace codex install|status|launch  (official MCP SDK + codex mcp)\n"
        "- AST context: laplace ast-context PATTERN PATH  (grep-ast/tree-sitter)"
    )


def render_capabilities(payload: Mapping[str, object]) -> str:
    """Render the Operator capability packet without exposing implementation noise."""

    tier = payload.get("capability_tier")
    capabilities = sorted(_strings(payload.get("capabilities")))
    lanes = _strings(payload.get("model_lanes"))
    repositories = payload.get("authorized_repositories")
    repo_ids: list[str] = []
    if isinstance(repositories, Sequence) and not isinstance(repositories, (str, bytes)):
        for item in repositories:
            if isinstance(item, Mapping):
                repo_id = item.get("repo_id")
                if isinstance(repo_id, str) and repo_id:
                    repo_ids.append(repo_id)
    lines = [
        f"tier={tier if isinstance(tier, str) and tier else 'unknown'}",
        "capabilities=" + (", ".join(capabilities) if capabilities else "none"),
        "model_lanes=" + (", ".join(lanes) if lanes else "none"),
        "authorized_repositories=" + (", ".join(sorted(set(repo_ids))) if repo_ids else "none"),
    ]
    return "\n".join(lines)


def render_skills(payload: Mapping[str, object]) -> str:
    """Render available governed Laplace surfaces from effective capabilities."""

    available = set(_strings(payload.get("capabilities")))
    surfaces: tuple[tuple[str, str, frozenset[str]], ...] = (
        (
            "repository-agent",
            "inspect, edit and deterministically verify an authorized repository worktree",
            frozenset({"agent"}),
        ),
        (
            "chat",
            "use the configured local conversational model lane",
            frozenset({"chat"}),
        ),
        (
            "retrieval",
            "search owner-authorized personal corpus evidence",
            frozenset({"personal_corpus"}),
        ),
        (
            "research",
            "run the bounded research-plane workflow",
            frozenset({"research"}),
        ),
        (
            "operator",
            "inspect local Operator/control-plane state",
            frozenset({"operator"}),
        ),
        (
            "model-admin",
            "inspect or administer serving profiles when authorized",
            frozenset({"model_admin"}),
        ),
        (
            "repository-admin",
            "manage repository grants when authorized",
            frozenset({"repository_admin"}),
        ),
    )
    lines = ["Available governed capabilities:"]
    matched = 0
    for name, description, required in surfaces:
        if required <= available:
            lines.append(f"- {name}: {description}")
            matched += 1
    if matched == 0:
        lines.append("- none exposed by the current Operator credential")
    lines.append("Use /help for commands and /contract for the active Operator protocol.")
    return "\n".join(lines)
