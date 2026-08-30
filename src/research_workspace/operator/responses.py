"""Bounded public response shaping for Operator agent-turn routes."""

from __future__ import annotations

from collections.abc import Mapping

JsonObject = dict[str, object]


def agent_result_content(result: Mapping[str, object]) -> str:
    """Return bounded display content without exposing private artifacts."""

    content = result.get("content")
    if isinstance(content, str) and content.strip():
        return content[:128_000]
    summary = result.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary[:128_000]
    result_id = result.get("result_id")
    return f"Agent turn completed. Result reference: {result_id or 'unavailable'}"


def public_agent_result(result: Mapping[str, object]) -> JsonObject:
    """Expose bounded evidence references without server-local artifact paths."""

    public = {
        key: result.get(key)
        for key in (
            "status", "session_id", "repo_id", "model_id", "effective_lane", "content",
            "changed_paths", "verification", "validation_history", "unresolved_failures",
            "evidence_refs", "promotion", "truncated", "max_chars", "elapsed_seconds",
            "telemetry", "result_id", "result_artifacts", "delivery_status", "worktree_release",
            "retrieval", "manager_control", "agent_conversation_message_id",
            "agent_conversation_persistence",
        )
        if key in result
    }
    handoff = result.get("handoff")
    if isinstance(handoff, Mapping):
        public["handoff"] = {
            key: handoff.get(key)
            for key in ("patch_chars", "patch_sha256", "patch_inline", "patch")
            if key in handoff
        }
    return public
