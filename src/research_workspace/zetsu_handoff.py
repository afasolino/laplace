"""Exact, bounded handoff artifacts for Zetsu worktree results."""

from __future__ import annotations

import difflib
import hashlib
import os
import uuid
from pathlib import Path
from typing import Sequence

from .agent_infrastructure.git import run_git
from .repository_authorization import RepositoryAuthorizationError, validate_workspace_path
from .service_tiers import ServiceTierError
from .zetsu_checkpoint import AgentCheckpointStore

JsonObject = dict[str, object]
_MAX_NEW_FILE_CHARS = 256_000


class ZetsuHandoffStore:
    """Persist exact worktree patches separately from agent narration."""

    def __init__(self, checkpoints: AgentCheckpointStore) -> None:
        self._checkpoints = checkpoints

    def exact_patch(self, worktree: Path, changed_paths: Sequence[str]) -> str:
        if not changed_paths:
            return ""
        tracked = run_git(
            worktree,
            ["diff", "--no-ext-diff", "--binary", "HEAD", "--", *changed_paths],
        )
        if tracked.returncode != 0:
            raise ServiceTierError("zetsu_agent_handoff_patch_unavailable")
        patch = tracked.stdout
        for relative in changed_paths:
            listed = run_git(worktree, ["ls-files", "--error-unmatch", "--", relative])
            if listed.returncode == 0:
                continue
            target = self._relative_target(worktree, relative)
            if not target.is_file() or target.stat().st_size > _MAX_NEW_FILE_CHARS:
                raise ServiceTierError("zetsu_agent_handoff_new_file_unavailable")
            try:
                content = target.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise ServiceTierError("zetsu_agent_handoff_new_file_not_text") from exc
            patch += f"diff --git a/{relative} b/{relative}\nnew file mode 100644\n"
            patch += "".join(
                difflib.unified_diff(
                    (),
                    content.splitlines(keepends=True),
                    fromfile="/dev/null",
                    tofile=f"b/{relative}",
                )
            )
        return patch

    def create(
        self,
        worktree: Path,
        session_id: str,
        changed_paths: Sequence[str],
        *,
        max_chars: int,
    ) -> JsonObject:
        """Capture a complete patch artifact; inline narration remains disabled."""

        del max_chars
        patch = self.exact_patch(worktree, changed_paths)
        digest = hashlib.sha256(patch.encode("utf-8")).hexdigest()
        artifact = self._checkpoints.path(session_id).with_suffix(".patch")
        temporary = artifact.with_name(f".{artifact.name}.{uuid.uuid4().hex}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(patch)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, artifact)
        finally:
            temporary.unlink(missing_ok=True)
        return {
            "patch": None,
            "patch_inline": False,
            "patch_chars": len(patch),
            "patch_sha256": digest,
            "patch_path": str(artifact),
        }

    def evidence(self, session_id: str, *, max_chars: int) -> JsonObject:
        """Return a bounded persisted handoff only for an authorized coordinator caller."""

        artifact = self._checkpoints.path(session_id).with_suffix(".patch")
        if not artifact.is_file():
            raise ServiceTierError("zetsu_agent_handoff_patch_unavailable")
        try:
            size = artifact.stat().st_size
            content = artifact.read_text(encoding="utf-8") if size <= max_chars else None
        except (OSError, UnicodeDecodeError) as exc:
            raise ServiceTierError("zetsu_agent_handoff_patch_unavailable") from exc
        try:
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        except OSError as exc:
            raise ServiceTierError("zetsu_agent_handoff_patch_unavailable") from exc
        return {
            "patch": content,
            "patch_inline": content is not None,
            "patch_chars": size,
            "patch_sha256": digest,
            "patch_path": str(artifact),
        }

    @staticmethod
    def _relative_target(worktree: Path, value: object) -> Path:
        if not isinstance(value, str) or not value or len(value) > 500:
            raise ServiceTierError("zetsu_agent_path_invalid")
        normalized = value.replace("\\", "/")
        if normalized == ".git" or normalized.startswith(".git/"):
            raise ServiceTierError("zetsu_agent_git_metadata_forbidden")
        try:
            return validate_workspace_path(worktree, normalized)
        except RepositoryAuthorizationError as exc:
            raise ServiceTierError(f"zetsu_agent_{exc.category}", exc.evidence) from exc
