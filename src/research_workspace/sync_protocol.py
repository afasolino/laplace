"""Versioned desktop/server Git synchronization protocol."""

from __future__ import annotations

import hashlib
import re
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SyncError(RuntimeError):
    """Synchronization failed closed with a user-safe category."""


class ChangeKind(StrEnum):
    MODIFIED = "modified"
    ADDED = "added"
    DELETED = "deleted"
    UNTRACKED = "untracked"


class SyncDirection(StrEnum):
    UPLOAD = "upload"
    DOWNLOAD = "download"


class SyncState(StrEnum):
    PLANNED = "PLANNED"
    TRANSFERRING = "TRANSFERRING"
    TRANSFERRED = "TRANSFERRED"
    CONFLICT = "CONFLICT"
    APPLIED = "APPLIED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class FileChangeV1(_Strict):
    schema_version: Literal[1] = 1
    logical_path: str = Field(min_length=1, max_length=500)
    kind: ChangeKind
    staged: bool
    included: bool

    @field_validator("kind", mode="before")
    @classmethod
    def kind_from_json(cls, value: object) -> object:
        return ChangeKind(value) if isinstance(value, str) else value

    @field_validator("logical_path")
    @classmethod
    def safe_path(cls, value: str) -> str:
        return clean_logical_path(value)


class RepositorySnapshotV1(_Strict):
    schema_version: Literal[1] = 1
    logical_repository_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    branch: str = Field(min_length=1, max_length=500)
    head: str = Field(pattern=r"^[a-f0-9]{40}$")
    dirty: bool
    remotes: tuple[str, ...]
    changes: tuple[FileChangeV1, ...]
    untracked_included: Literal[False] = False


class SyncPlanV1(_Strict):
    schema_version: Literal[1] = 1
    plan_id: str = Field(pattern=r"^sync-plan-[a-f0-9]{24}$")
    logical_repository_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    direction: SyncDirection
    base_head: str = Field(pattern=r"^[a-f0-9]{40}$")
    branch: str = Field(min_length=1, max_length=500)
    changed_paths: tuple[str, ...]
    patch_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    patch_size_bytes: int = Field(ge=0)
    requires_confirmation: Literal[True] = True
    force_push: Literal[False] = False
    untracked_included: Literal[False] = False

    @field_validator("direction", mode="before")
    @classmethod
    def direction_from_json(cls, value: object) -> object:
        return SyncDirection(value) if isinstance(value, str) else value

    @field_validator("changed_paths", mode="before")
    @classmethod
    def safe_paths(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        return tuple(clean_logical_path(str(item)) for item in value)


class SyncOperationV1(_Strict):
    schema_version: Literal[1] = 1
    operation_id: str = Field(pattern=r"^sync-[a-f0-9]{24}$")
    owner_id: str = Field(min_length=1, max_length=160)
    plan: SyncPlanV1
    state: SyncState
    transferred_bytes: int = Field(ge=0)
    created_at_utc: str
    updated_at_utc: str
    conflict_reason: str | None = Field(default=None, max_length=160)

    @field_validator("state", mode="before")
    @classmethod
    def state_from_json(cls, value: object) -> object:
        return SyncState(value) if isinstance(value, str) else value


class SyncReceiptV1(_Strict):
    schema_version: Literal[1] = 1
    operation_id: str = Field(pattern=r"^sync-[a-f0-9]{24}$")
    accepted_offset: int = Field(ge=0)
    complete: bool
    patch_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class TransportPolicyV1(_Strict):
    schema_version: Literal[1] = 1
    transport: Literal["fixture", "ssh", "https"]
    host: str | None = Field(default=None, max_length=253)
    host_key_verification: bool
    tls_verification: bool
    credential_storage: Literal[False] = False

    @model_validator(mode="after")
    def verification_required(self) -> TransportPolicyV1:
        if self.transport == "fixture":
            if self.host is not None:
                raise ValueError("fixture transport has no host")
            return self
        if self.host is None or not self.host or any(
            character in self.host for character in ("/", "\\", "@", ":")
        ):
            raise ValueError("transport host is invalid")
        if self.transport == "ssh" and not self.host_key_verification:
            raise ValueError("SSH host-key verification is required")
        if self.transport == "https" and not self.tls_verification:
            raise ValueError("HTTPS TLS verification is required")
        return self


@runtime_checkable
class SyncTransport(Protocol):
    def upload_chunk(
        self,
        *,
        owner_id: str,
        logical_repository_id: str,
        operation_id: str,
        base_head: str,
        offset: int,
        content: bytes,
        final: bool,
        expected_sha256: str,
    ) -> SyncReceiptV1: ...

    def download_patch(
        self,
        *,
        owner_id: str,
        logical_repository_id: str,
        operation_id: str,
    ) -> bytes: ...


def clean_logical_path(value: str) -> str:
    if "\x00" in value or "\\" in value or any(character.isspace() for character in value):
        raise ValueError("sync_path_invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(
        part in {"", ".", "..", ".git"} for part in path.parts
    ):
        raise ValueError("sync_path_invalid")
    return path.as_posix()


def validate_patch(
    patch: bytes,
    *,
    maximum_bytes: int,
    maximum_files: int,
) -> tuple[str, ...]:
    if not patch or len(patch) > maximum_bytes or b"\x00" in patch:
        raise SyncError("sync_patch_size_or_binary_rejected")
    try:
        text = patch.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SyncError("sync_patch_not_utf8") from exc
    forbidden = (
        "GIT binary patch",
        "new file mode 120000",
        "old mode 120000",
        "Subproject commit ",
        "rename from ",
        "rename to ",
        "copy from ",
        "copy to ",
    )
    if any(marker in text for marker in forbidden):
        raise SyncError("sync_patch_form_rejected")
    paths: list[str] = []
    for line in text.splitlines():
        if not line.startswith("diff --git a/"):
            continue
        match = re.fullmatch(r"diff --git a/([^ ]+) b/([^ ]+)", line)
        if match is None or match.group(1) != match.group(2):
            raise SyncError("sync_patch_header_invalid")
        try:
            paths.append(clean_logical_path(match.group(1)))
        except ValueError as exc:
            raise SyncError("sync_patch_path_rejected") from exc
    unique = tuple(sorted(set(paths)))
    if not unique or len(unique) > maximum_files:
        raise SyncError("sync_patch_file_count_rejected")
    return unique


def confirmation_for(plan_id: str) -> str:
    return f"confirm:{plan_id}"


def operation_id(plan: SyncPlanV1, owner_id: str) -> str:
    material = f"{owner_id}\n{plan.plan_id}\n{plan.patch_sha256}"
    return "sync-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
