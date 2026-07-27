"""Plus agent sessions bound to dedicated, server-created Git worktrees."""

from __future__ import annotations

import re
import subprocess  # nosec B404
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Mapping, TypeAlias

from .repository_authorization import (
    RepositoryAuthorizationError,
    RepositoryAuthorizationStore,
    validate_workspace_path,
)

JsonObject: TypeAlias = dict[str, object]
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class AgentSandboxError(RuntimeError):
    """An agent session could not be confined to its authorized worktree."""

    def __init__(self, category: str, evidence: JsonObject | None = None) -> None:
        super().__init__(category)
        self.category = category
        self.evidence = evidence or {}


@dataclass(frozen=True)
class AgentToolPolicy:
    policy_id: str
    allowed_tools: tuple[str, ...]
    network_enabled: bool = False
    max_commands: int = 100
    max_wall_seconds: int = 1_800

    def __post_init__(self) -> None:
        if self.network_enabled:
            raise ValueError("local Plus agent policies cannot enable network")
        if not self.allowed_tools or self.max_commands < 1 or self.max_wall_seconds < 1:
            raise ValueError("invalid agent tool policy")


@dataclass(frozen=True)
class AgentSessionBinding:
    session_id: str
    user_id: str
    repo_id: str
    canonical_repository_root: str
    worktree_root: str
    base_revision: str
    grant_revision: int
    tool_policy: AgentToolPolicy
    environment: Mapping[str, str]
    created_at_utc: str

    def to_json(self) -> JsonObject:
        value = asdict(self)
        value["tool_policy"] = asdict(self.tool_policy)
        value["environment"] = dict(self.environment)
        return value


def _session_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("invalid session_id")
    return value


class AgentSandboxManager:
    """Creates and revalidates isolated worktrees without accepting client roots."""

    def __init__(
        self,
        sandbox_root: Path,
        authorizations: RepositoryAuthorizationStore,
        *,
        runner: CommandRunner = subprocess.run,
        environment_allowlist: frozenset[str] = frozenset(
            {"LANG", "LC_ALL", "PATH", "PYTHONUTF8", "TZ"}
        ),
    ) -> None:
        self.sandbox_root = sandbox_root.resolve()
        self.sandbox_root.mkdir(parents=True, exist_ok=True)
        self.authorizations = authorizations
        self._runner = runner
        self.environment_allowlist = environment_allowlist
        self._sessions: dict[str, AgentSessionBinding] = {}

    def create(
        self,
        *,
        user_id: str,
        repo_id: str,
        session_id: str,
        tool_policy: AgentToolPolicy,
        environment: Mapping[str, str] | None = None,
    ) -> AgentSessionBinding:
        identifier = _session_identifier(session_id)
        if identifier in self._sessions:
            raise AgentSandboxError("session_exists", {"session_id": identifier})
        grant = self.authorizations.require_grant(user_id, repo_id)
        target = self.sandbox_root / user_id / identifier
        if target.exists():
            raise AgentSandboxError("worktree_exists", {"path": str(target)})
        target.parent.mkdir(parents=True, exist_ok=True)
        supplied_environment = dict(environment or {})
        forbidden = sorted(set(supplied_environment) - self.environment_allowlist)
        if forbidden:
            raise AgentSandboxError("environment_not_allowed", {"variables": forbidden})
        result = self._runner(
            [
                "git",
                "-C",
                str(grant.repository.canonical_root),
                "worktree",
                "add",
                "--detach",
                str(target),
                grant.base_revision,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        if result.returncode != 0:
            raise AgentSandboxError(
                "worktree_creation_failed",
                {"returncode": result.returncode, "stderr": result.stderr[-2_000:]},
            )
        binding = AgentSessionBinding(
            session_id=identifier,
            user_id=user_id,
            repo_id=repo_id,
            canonical_repository_root=str(grant.repository.canonical_root),
            worktree_root=str(target.resolve(strict=True)),
            base_revision=grant.base_revision,
            grant_revision=grant.revision,
            tool_policy=tool_policy,
            environment=supplied_environment,
            created_at_utc=datetime.now(UTC).isoformat(),
        )
        self._sessions[identifier] = binding
        return binding

    def require_active(self, session_id: str, *, user_id: str) -> AgentSessionBinding:
        identifier = _session_identifier(session_id)
        binding = self._sessions.get(identifier)
        if binding is None or binding.user_id != user_id:
            raise AgentSandboxError("unknown_agent_session")
        try:
            current = self.authorizations.require_grant(binding.user_id, binding.repo_id)
        except RepositoryAuthorizationError as exc:
            raise AgentSandboxError("repository_authorization_revoked", exc.evidence) from exc
        if current.revision != binding.grant_revision:
            raise AgentSandboxError(
                "repository_authorization_changed",
                {
                    "session_revision": binding.grant_revision,
                    "current_revision": current.revision,
                },
            )
        root = Path(binding.worktree_root)
        if not root.is_dir():
            raise AgentSandboxError("worktree_unavailable")
        return binding

    def validate_path(
        self, session_id: str, *, user_id: str, relative_path: str
    ) -> Path:
        binding = self.require_active(session_id, user_id=user_id)
        try:
            return validate_workspace_path(Path(binding.worktree_root), relative_path)
        except RepositoryAuthorizationError as exc:
            raise AgentSandboxError(exc.category, exc.evidence) from exc

    def close_if_clean(self, session_id: str, *, user_id: str) -> JsonObject:
        binding = self.require_active(session_id, user_id=user_id)
        root = Path(binding.worktree_root)
        status = self._runner(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if status.returncode != 0 or status.stdout.strip():
            return {
                "status": "PRESERVED_DIRTY_WORKTREE",
                "session_id": session_id,
                "worktree_root": str(root),
            }
        remove = self._runner(
            [
                "git",
                "-C",
                binding.canonical_repository_root,
                "worktree",
                "remove",
                str(root),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        if remove.returncode != 0:
            raise AgentSandboxError(
                "clean_worktree_release_failed",
                {"returncode": remove.returncode, "stderr": remove.stderr[-2_000:]},
            )
        self._sessions.pop(session_id, None)
        return {"status": "RELEASED_CLEAN_WORKTREE", "session_id": session_id}

    def status(self, session_id: str, *, user_id: str) -> JsonObject:
        """Revalidate a binding and report its owned worktree state."""

        binding = self.require_active(session_id, user_id=user_id)
        root = Path(binding.worktree_root)
        status = self._runner(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if status.returncode != 0:
            raise AgentSandboxError(
                "worktree_status_failed",
                {"returncode": status.returncode, "stderr": status.stderr[-2_000:]},
            )
        return {
            "status": "DIRTY" if status.stdout.strip() else "ACTIVE_CLEAN",
            "session_id": session_id,
            "repo_id": binding.repo_id,
            "worktree_root": binding.worktree_root,
            "base_revision": binding.base_revision,
            "grant_revision": binding.grant_revision,
        }

    @staticmethod
    def fixed_environment(binding: AgentSessionBinding) -> dict[str, str]:
        base = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "TZ": "UTC"}
        base.update(binding.environment)
        base["LAPLACE_AGENT_SESSION"] = binding.session_id
        base["LAPLACE_REPOSITORY_ID"] = binding.repo_id
        return base
