"""Bounded repository and corpus tools used by Zetsu orchestration."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from .personal_corpus import PersonalCorpusStore
from .service_tiers import ServiceTierError
from .zetsu_context import compact_personal_results
from .zetsu_state import AgentExecutionState, AgentRunContext

_MAX_READ_CHARS = 64_000
_MAX_SEARCH_MATCHES = 80

EnsureActive = Callable[[AgentRunContext, AgentExecutionState], None]
RelativeTarget = Callable[[Path, object], Path]
MaterializationFailure = Callable[[AgentRunContext, str], None]


def search(
    ctx: AgentRunContext,
    state: AgentExecutionState,
    action: Mapping[str, object],
    *,
    ensure_active: EnsureActive,
    relative_target: RelativeTarget,
) -> str:
    """Return bounded text matches from the authorized worktree only."""

    worktree = ctx.worktree
    ensure_active(ctx, state)
    query = action.get("query")
    pattern = action.get("glob", "*")
    if not isinstance(query, str) or not query or len(query) > 4_000:
        raise ServiceTierError("zetsu_agent_search_invalid")
    if (
        not isinstance(pattern, str)
        or not pattern
        or len(pattern) > 200
        or ".." in Path(pattern).parts
    ):
        raise ServiceTierError("zetsu_agent_glob_invalid")
    matches: list[str] = []
    for path_index, path in enumerate(worktree.rglob(pattern)):
        if path_index % 16 == 0:
            ensure_active(ctx, state)
        if len(matches) >= _MAX_SEARCH_MATCHES:
            break
        relative = path.relative_to(worktree).as_posix()
        if ".git" in path.relative_to(worktree).parts:
            continue
        try:
            safe_path = relative_target(worktree, relative)
            if not safe_path.is_file() or safe_path.stat().st_size > 1_000_000:
                continue
            text = safe_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError, ServiceTierError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if query in line:
                matches.append(f"{relative}:{line_number}:{line[:500]}")
                if len(matches) >= _MAX_SEARCH_MATCHES:
                    break
    return "\n".join(matches) if matches else "NO_MATCHES"


def read(
    ctx: AgentRunContext | Path,
    action: Mapping[str, object],
    *,
    relative_target: RelativeTarget,
    materialization_failure: MaterializationFailure,
) -> str:
    """Read one bounded set of repository-relative text files."""

    worktree = ctx.worktree if isinstance(ctx, AgentRunContext) else ctx
    values = action.get("paths")
    if values is None:
        values = [action.get("path")]
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or not 1 <= len(values) <= 8
        or len(set(str(item) for item in values)) != len(values)
    ):
        raise ServiceTierError("zetsu_agent_read_paths_invalid")
    result: dict[str, str] = {}
    remaining = _MAX_READ_CHARS
    for value in values:
        target = relative_target(worktree, value)
        if not target.is_file() or target.stat().st_size > 2_000_000:
            if not target.exists() and isinstance(value, str) and isinstance(ctx, AgentRunContext):
                materialization_failure(ctx, value)
            raise ServiceTierError("zetsu_agent_read_target_unavailable")
        try:
            content = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ServiceTierError("zetsu_agent_read_target_not_text") from exc
        relative = target.relative_to(worktree).as_posix()
        result[relative] = content[:remaining]
        remaining -= len(result[relative])
        if remaining <= 0:
            break
    if len(result) == 1 and action.get("paths") is None:
        return next(iter(result.values()))
    return json.dumps(result, sort_keys=True, ensure_ascii=False)


def retrieve(
    ctx: AgentRunContext,
    state: AgentExecutionState,
    action: Mapping[str, object],
    *,
    corpus: PersonalCorpusStore | None,
    ensure_active: EnsureActive,
) -> str:
    """Retrieve compact, owner-scoped personal-corpus evidence."""

    if corpus is None:
        raise ServiceTierError("zetsu_agent_retrieval_unavailable")
    ensure_active(ctx, state)
    query = action.get("query")
    if not isinstance(query, str) or not query.strip() or len(query) > 4_000:
        raise ServiceTierError("zetsu_agent_retrieval_query_invalid")
    raw = corpus.search(ctx.user_id, query.strip(), limit=8)
    ensure_active(ctx, state)
    packet = compact_personal_results(query.strip(), raw, max_results=6, max_chars=8_000)
    evidence = packet.get("evidence")
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, Mapping):
                chunk_id = item.get("chunk_id")
                if isinstance(chunk_id, str) and chunk_id not in state.evidence_refs:
                    state.evidence_refs.append(chunk_id)
    return json.dumps(packet, sort_keys=True, ensure_ascii=False)
