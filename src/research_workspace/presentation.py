"""Shared frontend presentation contracts; rendering remains mode-specific."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .contracts import CitationV1


class _Presentation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class MarkdownPresentationV1(_Presentation):
    schema_version: Literal[1] = 1
    markdown: str = Field(max_length=4_000_000)
    raw_html_allowed: Literal[False] = False
    tables_copyable: bool = True


class CitationPresentationV1(_Presentation):
    schema_version: Literal[1] = 1
    citations: tuple[CitationV1, ...]
    file_page_section_chunk_required: Literal[True] = True


class ProgressPresentationV1(_Presentation):
    schema_version: Literal[1] = 1
    state: Literal[
        "VALIDATING",
        "QUEUED",
        "ADMITTED",
        "PREPARING_CONTEXT",
        "RETRIEVING",
        "GENERATING",
        "VALIDATING_OUTPUT",
        "ESCALATING",
        "COMPLETE",
        "CANCELLED",
        "TIMED_OUT",
        "FAILED",
    ]
    elapsed_seconds: float = Field(ge=0)
    queue_position: int | None = Field(default=None, ge=1)
    route_id: str | None = Field(default=None, max_length=80)
    model_display_name: str | None = Field(default=None, max_length=160)
    trace_id: str = Field(min_length=1, max_length=160)
    percent_complete: None = None
    private_reasoning: None = None


class ConversationControlsV1(_Presentation):
    schema_version: Literal[1] = 1
    can_stop: bool
    can_retry: bool
    can_edit: bool
    can_export: bool


class StorageExplanationV1(_Presentation):
    schema_version: Literal[1] = 1
    storage_class: Literal[
        "conversation",
        "draft",
        "attachment",
        "personal_corpus",
        "shared_corpus",
        "repository",
        "worktree",
        "artifact",
        "audit",
    ]
    owner: str
    logical_location: str
    indexed: bool
    retention: str
    deletion: str
    access: str


class NavigationItemV1(_Presentation):
    schema_version: Literal[1] = 1
    item_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,79}$")
    label: str = Field(min_length=1, max_length=80)
    required_capability: str | None = Field(default=None, max_length=80)
    mode: Literal["both", "desktop", "server"] = "both"


def capability_navigation(
    definitions: Sequence[NavigationItemV1],
    *,
    capabilities: Sequence[str],
    mode: Literal["desktop", "server"],
) -> tuple[NavigationItemV1, ...]:
    allowed = set(capabilities)
    return tuple(
        item
        for item in definitions
        if item.mode in {"both", mode}
        and (item.required_capability is None or item.required_capability in allowed)
    )


def storage_catalog(
    values: Mapping[str, Mapping[str, object]],
) -> tuple[StorageExplanationV1, ...]:
    """Validate one storage vocabulary for Help, Account and System surfaces."""

    result: list[StorageExplanationV1] = []
    for storage_class in sorted(values):
        record = dict(values[storage_class])
        record["storage_class"] = storage_class
        result.append(StorageExplanationV1.model_validate(record))
    return tuple(result)
