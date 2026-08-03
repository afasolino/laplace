"""Server-owned repository registry, user grants, and path escape prevention."""

from __future__ import annotations

import os
import re
import sqlite3
import stat
import subprocess  # nosec B404
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Callable, Sequence, TypeAlias

JsonObject: TypeAlias = dict[str, object]
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class RepositoryAuthorizationError(RuntimeError):
    """A repository grant or path check failed closed."""

    def __init__(self, category: str, evidence: JsonObject | None = None) -> None:
        super().__init__(category)
        self.category = category
        self.evidence = evidence or {}


@dataclass(frozen=True)
class RegisteredRepository:
    repo_id: str
    canonical_root: Path
    device: int
    inode: int
    registered_at_utc: str


@dataclass(frozen=True)
class RepositoryGrant:
    user_id: str
    repository: RegisteredRepository
    active: bool
    revision: int
    base_revision: str


def _identifier(value: str, *, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError(f"invalid {label}")
    return value


def _sqlite_filesystem_id(value: int) -> int:
    """Map an unsigned 64-bit host file ID into SQLite's signed range."""

    if 0 <= value < 2**63:
        return value
    if 2**63 <= value < 2**64:
        return value - 2**64
    raise RepositoryAuthorizationError(
        "filesystem_identity_out_of_range",
        {"bit_length": value.bit_length()},
    )


def _git(
    root: Path,
    arguments: Sequence[str],
    *,
    runner: CommandRunner = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    return runner(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


class RepositoryAuthorizationStore:
    """Registry and grants whose roots can only be selected by an operator."""

    def __init__(self, path: Path, *, runner: CommandRunner = subprocess.run) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._runner = runner
        self._lock = threading.RLock()
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS repositories (
                    repo_id TEXT PRIMARY KEY,
                    canonical_root TEXT NOT NULL UNIQUE,
                    device INTEGER NOT NULL,
                    inode INTEGER NOT NULL,
                    registered_at_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS repository_grants (
                    user_id TEXT NOT NULL,
                    repo_id TEXT NOT NULL,
                    active INTEGER NOT NULL CHECK(active IN (0, 1)),
                    revision INTEGER NOT NULL,
                    base_revision TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    PRIMARY KEY(user_id, repo_id),
                    FOREIGN KEY(repo_id) REFERENCES repositories(repo_id)
                );
                """
            )
        os.chmod(self.path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 15000")
        return connection

    def register(self, repo_id: str, root: Path) -> RegisteredRepository:
        normalized = _identifier(repo_id, label="repo_id")
        canonical = root.resolve(strict=True)
        result = _git(canonical, ("rev-parse", "--show-toplevel"), runner=self._runner)
        if result.returncode != 0:
            raise RepositoryAuthorizationError(
                "not_a_git_repository",
                {"repo_id": normalized, "root": str(canonical)},
            )
        git_root = Path(result.stdout.strip()).resolve(strict=True)
        if git_root != canonical:
            raise RepositoryAuthorizationError(
                "repository_root_mismatch",
                {"requested": str(canonical), "git_root": str(git_root)},
            )
        details = canonical.stat()
        device = _sqlite_filesystem_id(details.st_dev)
        inode = _sqlite_filesystem_id(details.st_ino)
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO repositories (
                    repo_id, canonical_root, device, inode, registered_at_utc
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(repo_id) DO UPDATE SET
                    canonical_root=excluded.canonical_root,
                    device=excluded.device,
                    inode=excluded.inode,
                    registered_at_utc=excluded.registered_at_utc
                """,
                (normalized, str(canonical), device, inode, now),
            )
        return RegisteredRepository(
            normalized, canonical, device, inode, now
        )

    def grant(self, user_id: str, repo_id: str, *, base_revision: str = "HEAD") -> RepositoryGrant:
        user = _identifier(user_id, label="user_id")
        repository = self.repository(repo_id)
        revision_result = _git(
            repository.canonical_root,
            ("rev-parse", "--verify", f"{base_revision}^{{commit}}"),
            runner=self._runner,
        )
        if revision_result.returncode != 0:
            raise RepositoryAuthorizationError(
                "invalid_base_revision",
                {"repo_id": repo_id, "base_revision": base_revision},
            )
        commit = revision_result.stdout.strip()
        with self._lock, self._connect() as connection:
            current = connection.execute(
                """
                SELECT revision FROM repository_grants
                WHERE user_id = ? AND repo_id = ?
                """,
                (user, repository.repo_id),
            ).fetchone()
            revision = int(current["revision"]) + 1 if current is not None else 1
            connection.execute(
                """
                INSERT INTO repository_grants (
                    user_id, repo_id, active, revision, base_revision, updated_at_utc
                ) VALUES (?, ?, 1, ?, ?, ?)
                ON CONFLICT(user_id, repo_id) DO UPDATE SET
                    active=1,
                    revision=excluded.revision,
                    base_revision=excluded.base_revision,
                    updated_at_utc=excluded.updated_at_utc
                """,
                (user, repository.repo_id, revision, commit, datetime.now(UTC).isoformat()),
            )
        return RepositoryGrant(user, repository, True, revision, commit)

    def revoke(self, user_id: str, repo_id: str) -> RepositoryGrant:
        current = self.require_grant(user_id, repo_id)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE repository_grants
                SET active = 0, revision = ?, updated_at_utc = ?
                WHERE user_id = ? AND repo_id = ?
                """,
                (
                    current.revision + 1,
                    datetime.now(UTC).isoformat(),
                    current.user_id,
                    current.repository.repo_id,
                ),
            )
        return RepositoryGrant(
            current.user_id,
            current.repository,
            False,
            current.revision + 1,
            current.base_revision,
        )

    def repository(self, repo_id: str) -> RegisteredRepository:
        normalized = _identifier(repo_id, label="repo_id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM repositories WHERE repo_id = ?", (normalized,)
            ).fetchone()
        if row is None:
            raise RepositoryAuthorizationError("unknown_repository", {"repo_id": normalized})
        root = Path(str(row["canonical_root"]))
        try:
            details = root.resolve(strict=True).stat()
        except OSError as exc:
            raise RepositoryAuthorizationError(
                "registered_repository_unavailable",
                {"repo_id": normalized, "error": type(exc).__name__},
            ) from exc
        device = _sqlite_filesystem_id(details.st_dev)
        inode = _sqlite_filesystem_id(details.st_ino)
        if device != int(row["device"]) or inode != int(row["inode"]):
            raise RepositoryAuthorizationError(
                "registered_repository_identity_changed",
                {"repo_id": normalized},
            )
        return RegisteredRepository(
            repo_id=normalized,
            canonical_root=root.resolve(strict=True),
            device=int(row["device"]),
            inode=int(row["inode"]),
            registered_at_utc=str(row["registered_at_utc"]),
        )

    def require_grant(self, user_id: str, repo_id: str) -> RepositoryGrant:
        user = _identifier(user_id, label="user_id")
        repository = self.repository(repo_id)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT active, revision, base_revision FROM repository_grants
                WHERE user_id = ? AND repo_id = ?
                """,
                (user, repository.repo_id),
            ).fetchone()
        if row is None or not bool(row["active"]):
            raise RepositoryAuthorizationError(
                "repository_not_authorized",
                {"user_id": user, "repo_id": repository.repo_id},
            )
        return RepositoryGrant(
            user_id=user,
            repository=repository,
            active=True,
            revision=int(row["revision"]),
            base_revision=str(row["base_revision"]),
        )

    def assert_revision(self, grant: RepositoryGrant) -> RepositoryGrant:
        current = self.require_grant(grant.user_id, grant.repository.repo_id)
        if current.revision != grant.revision:
            raise RepositoryAuthorizationError(
                "repository_grant_changed",
                {
                    "repo_id": grant.repository.repo_id,
                    "session_revision": grant.revision,
                    "current_revision": current.revision,
                },
            )
        return current

    def authorized_for_user(self, user_id: str) -> list[dict[str, object]]:
        """Return only active logical grants; never infer authorization from a path."""

        user = _identifier(user_id, label="user_id")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT g.repo_id, g.base_revision, g.revision, g.updated_at_utc
                FROM repository_grants AS g
                JOIN repositories AS r ON r.repo_id = g.repo_id
                WHERE g.user_id = ? AND g.active = 1
                ORDER BY g.repo_id
                """,
                (user,),
            ).fetchall()
        return [
            {
                "repo_id": str(row["repo_id"]),
                "logical_name": str(row["repo_id"]),
                "description": "Operator-authorized local repository",
                "base_revision": str(row["base_revision"]),
                "grant_revision": int(row["revision"]),
                "updated_at_utc": str(row["updated_at_utc"]),
            }
            for row in rows
        ]

    def operator_inventory(self) -> list[dict[str, object]]:
        """Return the administrative repository inventory, including canonical roots."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT r.repo_id, r.canonical_root, r.registered_at_utc,
                       COUNT(CASE WHEN g.active = 1 THEN 1 END) AS active_grants
                FROM repositories AS r
                LEFT JOIN repository_grants AS g ON g.repo_id = r.repo_id
                GROUP BY r.repo_id, r.canonical_root, r.registered_at_utc
                ORDER BY r.repo_id
                """
            ).fetchall()
        return [
            {
                "repo_id": str(row["repo_id"]),
                "logical_name": str(row["repo_id"]),
                "canonical_root": str(row["canonical_root"]),
                "registered_at_utc": str(row["registered_at_utc"]),
                "active_grants": int(row["active_grants"]),
            }
            for row in rows
        ]


def _submodule_paths(root: Path) -> tuple[PurePosixPath, ...]:
    file = root / ".gitmodules"
    if not file.is_file():
        return ()
    paths: list[PurePosixPath] = []
    for line in file.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.fullmatch(r"\s*path\s*=\s*(.+?)\s*", line)
        if match:
            candidate = PurePosixPath(match.group(1))
            if not candidate.is_absolute() and ".." not in candidate.parts:
                paths.append(candidate)
    return tuple(paths)


def _mount_points() -> tuple[Path, ...]:
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return ()
    points: list[Path] = []
    for line in lines:
        fields = line.split()
        if len(fields) >= 5:
            points.append(Path(fields[4].replace("\\040", " ")))
    return tuple(points)


def validate_workspace_path(
    root: Path,
    relative_path: str,
    *,
    mount_points: Sequence[Path] | None = None,
) -> Path:
    """Reject traversal, links, mounts, nested Git roots, and submodule escapes."""

    if "\x00" in relative_path:
        raise RepositoryAuthorizationError("invalid_path")
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise RepositoryAuthorizationError("path_escape", {"path": relative_path})
    canonical_root = root.resolve(strict=True)
    candidate = canonical_root.joinpath(*relative.parts)
    root_device = canonical_root.stat().st_dev
    current = canonical_root
    for part in relative.parts:
        current = current / part
        try:
            details = os.lstat(current)
        except FileNotFoundError:
            break
        if stat.S_ISLNK(details.st_mode):
            raise RepositoryAuthorizationError("symlink_escape", {"path": str(current)})
        if details.st_dev != root_device:
            raise RepositoryAuthorizationError("filesystem_escape", {"path": str(current)})
        if stat.S_ISREG(details.st_mode) and details.st_nlink > 1:
            raise RepositoryAuthorizationError("hardlink_escape", {"path": str(current)})
        if (current / ".git").exists():
            raise RepositoryAuthorizationError("nested_repository_escape", {"path": str(current)})
    resolved = candidate.resolve(strict=False)
    if resolved != canonical_root and canonical_root not in resolved.parents:
        raise RepositoryAuthorizationError("path_escape", {"path": relative_path})
    pure_relative = PurePosixPath(*relative.parts)
    for submodule in _submodule_paths(canonical_root):
        if pure_relative == submodule or submodule in pure_relative.parents:
            raise RepositoryAuthorizationError(
                "submodule_escape", {"path": relative_path, "submodule": str(submodule)}
            )
    mounts = tuple(mount_points) if mount_points is not None else _mount_points()
    for mount in mounts:
        mount_resolved = mount.resolve(strict=False)
        if mount_resolved != canonical_root and canonical_root in mount_resolved.parents:
            if resolved == mount_resolved or mount_resolved in resolved.parents:
                raise RepositoryAuthorizationError(
                    "bind_mount_escape", {"path": relative_path, "mount": str(mount)}
                )
    return candidate
