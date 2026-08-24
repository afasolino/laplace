"""Versioned, owner-authorized Zetsu MCP semantics.

HTTP transport and Laplace authentication are mounted by :mod:`operator_api` so
Zetsu cannot accidentally create a weaker parallel identity or public daemon.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, TypeAlias, cast

from .model_routing import (
    RoutingTaskMetadata,
    assess_rtl_worker_eligibility,
)
from .agent_sandbox import AgentSandboxError
from .laplace_core import LaplaceCore
from .personal_corpus import CorpusError, PersonalCorpusStore
from .service_tiers import ModelLane, ServiceTierError, TieredServingService
from .user_capabilities import Capability
from .versioning import version_record
from .zetsu_agent import ZetsuAgentCoordinator
from .zetsu_results import ZetsuResultError
from .zetsu_context import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_RESULTS,
    compact_expanded_results,
    compact_personal_results,
    normalize_budget,
)

JsonObject: TypeAlias = dict[str, object]

ZETSU_SCHEMA_VERSION = "1.5"
ZETSU_SKILL_VERSION = "1.5.0"
MCP_LATEST_PROTOCOL_VERSION = "2026-07-28"
MCP_LEGACY_PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26")
MCP_SUPPORTED_PROTOCOL_VERSIONS = (MCP_LATEST_PROTOCOL_VERSION, *MCP_LEGACY_PROTOCOL_VERSIONS)
ZETSU_INSTRUCTIONS = (
    "Local work stays in Codex. Zetsu supplies compact evidence, verified Qwen repository "
    "tasks, and policy-bounded CodeV RTL; expand exact evidence only when needed."
)


class ZetsuError(RuntimeError):
    """A Zetsu schema, policy, or authorization check failed."""

    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


def _object(value: object, *, label: str) -> JsonObject:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ZetsuError(f"invalid_{label}")
    return cast(JsonObject, value)


def _text(value: object, *, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ZetsuError(f"invalid_{label}")
    return value.strip()


def _integer(value: object, *, label: str, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ZetsuError(f"invalid_{label}")
    return value


_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _session_id(value: object) -> str:
    if not isinstance(value, str) or _SESSION_ID_RE.fullmatch(value) is None:
        raise ZetsuError("invalid_session_id")
    return value


def _verification_argv(value: object) -> list[str] | None:
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or not 1 <= len(value) <= 64
        or any(
            not isinstance(item, str) or not item or len(item) > 1_000 or "\x00" in item
            for item in value
        )
    ):
        raise ZetsuError("invalid_verification_argv")
    return list(value)


def _rough_tokens(value: object) -> int:
    """Tokenizer-independent accounting estimate; exact model usage is reported separately."""

    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return max(1, (len(encoded) + 3) // 4)


def _telemetry_requested(args: Mapping[str, object]) -> bool:
    value = args.get("telemetry", False)
    if not isinstance(value, bool):
        raise ZetsuError("invalid_telemetry")
    return value


def _boolean(args: Mapping[str, object], name: str, *, default: bool = False) -> bool:
    value = args.get(name, default)
    if not isinstance(value, bool):
        raise ZetsuError(f"invalid_{name}")
    return value


def _schema(properties: JsonObject, required: tuple[str, ...]) -> JsonObject:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def tool_definitions() -> tuple[JsonObject, ...]:
    budget = {"type": "integer", "minimum": 512, "maximum": 24_000}
    count = {"type": "integer", "minimum": 1, "maximum": 20}
    query = {"type": "string", "minLength": 1, "maxLength": 4_000}
    return (
        {
            "name": "search",
            "description": "Search compact owner evidence with provenance.",
            "inputSchema": _schema(
                {
                    "query": query,
                    "max_results": count,
                    "max_chars": budget,
                    "corpus_id": {"type": ["string", "null"], "maxLength": 128},
                    "telemetry": {"type": "boolean"},
                },
                ("query",),
            ),
            "annotations": {"readOnlyHint": True, "openWorldHint": False},
        },
        {
            "name": "get_evidence",
            "description": "Expand selected evidence chunk IDs.",
            "inputSchema": _schema(
                {
                    "chunk_ids": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 20,
                        "uniqueItems": True,
                        "items": {"type": "string", "pattern": "^chk_[a-f0-9]{32}$"},
                    },
                    "max_chars": budget,
                    "telemetry": {"type": "boolean"},
                },
                ("chunk_ids",),
            ),
            "annotations": {"readOnlyHint": True, "openWorldHint": False},
        },
        {
            "name": "project_context",
            "description": "Retrieve compact persistent project context.",
            "inputSchema": _schema(
                {
                    "query": query,
                    "max_results": count,
                    "max_chars": budget,
                    "telemetry": {"type": "boolean"},
                },
                ("query",),
            ),
            "annotations": {"readOnlyHint": True, "openWorldHint": False},
        },
        {
            "name": "experiment_context",
            "description": "Retrieve compact experiment context.",
            "inputSchema": _schema(
                {
                    "query": query,
                    "max_results": count,
                    "max_chars": budget,
                    "telemetry": {"type": "boolean"},
                },
                ("query",),
            ),
            "annotations": {"readOnlyHint": True, "openWorldHint": False},
        },
        {
            "name": "delegate",
            "description": "Run bounded Qwen reasoning without tools.",
            "inputSchema": _schema(
                {
                    "instruction": {"type": "string", "minLength": 1, "maxLength": 40_000},
                    "domain": {"type": "string", "maxLength": 64},
                    "lane": {"type": "string", "enum": ["quality", "standard"]},
                    "max_chars": budget,
                    "telemetry": {"type": "boolean"},
                },
                ("instruction",),
            ),
            "annotations": {"readOnlyHint": True, "openWorldHint": False},
        },
        {
            "name": "agent_task",
            "description": "Run an isolated verified Qwen repository task with optional promotion.",
            "inputSchema": _schema(
                {
                    "repo_id": {"type": "string", "minLength": 1, "maxLength": 128},
                    "session_id": {
                        "type": ["string", "null"],
                        "minLength": 1,
                        "maxLength": 128,
                        "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$",
                    },
                    "instruction": {"type": "string", "minLength": 1, "maxLength": 40_000},
                    "lane": {"type": "string", "enum": ["quality", "standard"]},
                    "max_steps": {"type": "integer", "minimum": 1, "maximum": 32},
                    "max_chars": budget,
                    "verification_argv": {
                        "type": ["array", "null"],
                        "minItems": 1,
                        "maxItems": 64,
                        "items": {"type": "string", "minLength": 1, "maxLength": 1_000},
                    },
                    "apply_to_repository": {"type": "boolean"},
                    "wait_timeout_seconds": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 3_600,
                    },
                    "telemetry": {"type": "boolean"},
                },
                ("repo_id", "instruction"),
            ),
            "annotations": {
                "readOnlyHint": False,
                "destructiveHint": False,
                "openWorldHint": False,
            },
        },
        {
            "name": "agent_task_status",
            "description": "Read the owner-authorized queue/execution state of an agent task.",
            "inputSchema": _schema(
                {
                    "session_id": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 128,
                        "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$",
                    }
                },
                ("session_id",),
            ),
            "annotations": {"readOnlyHint": True, "openWorldHint": False},
        },
        {
            "name": "cancel_agent_task",
            "description": "Cancel an owner-authorized agent task while it is queued.",
            "inputSchema": _schema(
                {
                    "session_id": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 128,
                        "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$",
                    }
                },
                ("session_id",),
            ),
            "annotations": {
                "readOnlyHint": False,
                "destructiveHint": False,
                "openWorldHint": False,
            },
        },
        {
            "name": "rtl_task",
            "description": "Run a policy-bounded CodeV RTL task.",
            "inputSchema": _schema(
                {
                    "session_id": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 128,
                        "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$",
                    },
                    "instruction": {"type": "string", "minLength": 1, "maxLength": 40_000},
                    "task_kind": {"type": "string", "enum": ["implementation", "repair"]},
                    "rtl_scope": {
                        "type": "string",
                        "enum": ["bounded_module"],
                    },
                    "editable_sources": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 1,
                        "items": {"type": "string", "maxLength": 500},
                    },
                    "module_count": {"type": "integer", "minimum": 1, "maximum": 1},
                    "max_chars": budget,
                },
                (
                    "session_id",
                    "instruction",
                    "task_kind",
                    "rtl_scope",
                    "editable_sources",
                    "module_count",
                ),
            ),
            "annotations": {
                "readOnlyHint": False,
                "destructiveHint": False,
                "openWorldHint": False,
            },
        },
        {
            "name": "verify",
            "description": "Return owned agent verification or exact handoff evidence.",
            "inputSchema": _schema(
                {
                    "session_id": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 128,
                        "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$",
                    },
                    "include_patch": {"type": "boolean"},
                    "max_chars": budget,
                },
                ("session_id",),
            ),
            "annotations": {"readOnlyHint": True, "openWorldHint": False},
        },
        {
            "name": "get_result",
            "description": "Read an exact durable task artifact in bounded byte pages.",
            "inputSchema": _schema(
                {
                    "result_id": {"type": "string", "pattern": "^res_[a-f0-9]{32}$"},
                    "session_id": {
                        "type": "string",
                        "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$",
                    },
                    "repo_id": {"type": "string", "minLength": 1, "maxLength": 128},
                    "artifact": {
                        "type": "string",
                        "pattern": "^[a-z][a-z0-9_.-]{0,127}$",
                    },
                    "offset": {"type": "integer", "minimum": 0},
                    "max_bytes": {"type": "integer", "minimum": 1, "maximum": 65536},
                },
                ("result_id", "session_id", "repo_id", "artifact"),
            ),
            "annotations": {"readOnlyHint": True, "openWorldHint": False},
        },
    )


_TOOL_CAPABILITIES: Mapping[str, Capability] = {
    "search": Capability.PERSONAL_CORPUS,
    "get_evidence": Capability.PERSONAL_CORPUS,
    "project_context": Capability.PERSONAL_CORPUS,
    "experiment_context": Capability.PERSONAL_CORPUS,
    "delegate": Capability.CHAT,
    "agent_task": Capability.AGENT,
    "agent_task_status": Capability.AGENT,
    "cancel_agent_task": Capability.AGENT,
    "rtl_task": Capability.AGENT,
    "verify": Capability.AGENT,
    "get_result": Capability.AGENT,
}

_TOOL_ARGUMENTS: Mapping[str, frozenset[str]] = {
    "search": frozenset({"query", "max_results", "max_chars", "corpus_id", "telemetry"}),
    "get_evidence": frozenset({"chunk_ids", "max_chars", "telemetry"}),
    "project_context": frozenset({"query", "max_results", "max_chars", "telemetry"}),
    "experiment_context": frozenset({"query", "max_results", "max_chars", "telemetry"}),
    "delegate": frozenset({"instruction", "domain", "lane", "max_chars", "telemetry"}),
    "agent_task": frozenset(
        {
            "repo_id",
            "session_id",
            "instruction",
            "lane",
            "max_steps",
            "max_chars",
            "verification_argv",
            "apply_to_repository",
            "wait_timeout_seconds",
            "telemetry",
        }
    ),
    "agent_task_status": frozenset({"session_id"}),
    "cancel_agent_task": frozenset({"session_id"}),
    "rtl_task": frozenset(
        {
            "session_id",
            "instruction",
            "task_kind",
            "rtl_scope",
            "editable_sources",
            "module_count",
            "max_chars",
        }
    ),
    "verify": frozenset({"session_id", "include_patch", "max_chars"}),
    "get_result": frozenset(
        {"result_id", "session_id", "repo_id", "artifact", "offset", "max_bytes"}
    ),
}


def _compact_verification(value: object) -> JsonObject | None:
    if not isinstance(value, Mapping):
        return None
    return {
        key: value.get(key)
        for key in (
            "argv",
            "command_id",
            "returncode",
            "passed",
            "aborted_category",
            "qualifies_for_mutation",
            "mutation_epoch",
            "worktree_mutated",
        )
    }


def _compact_agent_result(result: Mapping[str, object]) -> JsonObject:
    handoff = result.get("handoff")
    handoff_ref = (
        {key: handoff.get(key) for key in ("patch_chars", "patch_sha256", "patch_path")}
        if isinstance(handoff, Mapping)
        else None
    )
    content = result.get("content")
    short_result = str(content)[:512] if isinstance(content, str) else ""
    compact: JsonObject = {
        "status": result.get("status"),
        "session_id": result.get("session_id"),
        "repo_id": result.get("repo_id"),
        "model_id": result.get("model_id"),
        "effective_lane": result.get("effective_lane"),
        "result": short_result,
        "changed_paths": result.get("changed_paths"),
        "verification": _compact_verification(result.get("verification")),
        "unresolved_failures": result.get("unresolved_failures"),
        "evidence_refs": result.get("evidence_refs"),
        "checkpoint_path": result.get("checkpoint_path"),
        "handoff": handoff_ref,
        "promotion": result.get("promotion"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "result_id": result.get("result_id"),
        "delivery_status": result.get("delivery_status"),
        "result_artifacts": result.get("result_artifacts"),
        "worktree_release": result.get("worktree_release"),
    }
    if isinstance(result.get("telemetry"), Mapping):
        compact["telemetry"] = dict(cast(Mapping[str, object], result["telemetry"]))
    return compact


def _bounded_model_result(result: Mapping[str, object], *, max_chars: int) -> JsonObject:
    budget = normalize_budget(max_chars)
    envelope = result.get("response")
    response = envelope if isinstance(envelope, dict) else {}
    content_value = response.get("content", response.get("patch", ""))
    content = str(content_value) if content_value is not None else ""
    truncated = len(content) > budget
    if truncated:
        content = content[: budget - 1].rstrip() + "…"
    return {
        "status": result.get("status"),
        "request_id": result.get("request_id"),
        "trace_id": result.get("trace_id"),
        "session_id": result.get("session_id"),
        "model_id": result.get("model_id"),
        "effective_lane": result.get("effective_lane"),
        "content": content,
        "truncated": truncated,
        "max_chars": budget,
    }


class ZetsuService:
    """Authenticated semantic facade over existing owner-isolated Laplace services."""

    def __init__(
        self,
        repository_root: Path,
        corpus: PersonalCorpusStore,
        tiered: TieredServingService,
        *,
        core: LaplaceCore | None = None,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.corpus = corpus
        self.tiered = tiered
        self.core = core or LaplaceCore(self.repository_root, corpus, tiered)
        # Retrieval and the dedicated CodeV path do not require a Qwen-agent sandbox.
        # Construct the coordinator only for agent_task, while retaining one shared
        # per-session lock table even when concurrent requests arrive first.
        self._agent_coordinator: ZetsuAgentCoordinator | None = None
        self._agent_coordinator_lock = threading.Lock()

    def _agent_service(self) -> ZetsuAgentCoordinator:
        with self._agent_coordinator_lock:
            if self._agent_coordinator is None:
                self._agent_coordinator = self._core().agent_coordinator
            return self._agent_coordinator

    def _core(self) -> LaplaceCore:
        """Return the injected core, retaining compatibility with small fixtures."""

        core = getattr(self, "core", None)
        if isinstance(core, LaplaceCore):
            return core
        core = LaplaceCore(self.repository_root, self.corpus, self.tiered)
        self.core = core
        return core

    def available_tools(self, user_id: str) -> tuple[JsonObject, ...]:
        capabilities = self.tiered.effective_capabilities(user_id)
        return tuple(
            definition
            for definition in tool_definitions()
            if _TOOL_CAPABILITIES[str(definition["name"])] in capabilities
        )

    def status(self, user_id: str) -> JsonObject:
        tools = self.available_tools(user_id)
        quality = self.tiered.lane_policy.routes[ModelLane.QUALITY]
        standard = self.tiered.lane_policy.routes[ModelLane.STANDARD]
        economy = self.tiered.lane_policy.routes[ModelLane.ECONOMY]
        build = version_record(self.repository_root)
        return {
            "status": "READY",
            "zetsu_schema_version": ZETSU_SCHEMA_VERSION,
            "skill_version": ZETSU_SKILL_VERSION,
            "mcp_protocol_versions": list(MCP_SUPPORTED_PROTOCOL_VERSIONS),
            "laplace_server_revision": build["git_revision"],
            "available_tools": [str(item["name"]) for item in tools],
            "qwen": {
                "quality_model_id": quality.model_id,
                "standard_model_id": standard.model_id,
                "configured": quality.model_id == standard.model_id,
            },
            "codev": {
                "model_id": economy.model_id,
                "available": (
                    self.tiered.lane_policy.codev_enabled
                    and "codev" in economy.model_id.casefold()
                ),
                "state": (
                    "required"
                    if self.tiered.lane_policy.codev_enabled
                    else "intentionally_disabled"
                ),
                "policy": "bounded_policy_eligible_rtl_only",
            },
            "agent_scheduler": self._core().scheduler_status(user_id=user_id),
        }

    def call(self, user_id: str, name: str, arguments: Mapping[str, object]) -> JsonObject:
        available = {str(item["name"]) for item in self.available_tools(user_id)}
        if name not in available:
            raise ZetsuError("zetsu_tool_not_authorized")
        args = dict(arguments)
        if set(args) - _TOOL_ARGUMENTS[name]:
            raise ZetsuError("unexpected_tool_arguments")
        if name in {"search", "project_context", "experiment_context"}:
            query = _text(args.get("query"), label="query", maximum=4_000)
            max_results = _integer(
                args.get("max_results", DEFAULT_MAX_RESULTS),
                label="max_results",
                minimum=1,
                maximum=20,
            )
            max_chars = _integer(
                args.get("max_chars", DEFAULT_MAX_CHARS),
                label="max_chars",
                minimum=512,
                maximum=24_000,
            )
            corpus_id_value = args.get("corpus_id") if name == "search" else None
            if corpus_id_value is not None and not isinstance(corpus_id_value, str):
                raise ZetsuError("invalid_corpus_id")
            raw = self._core().retrieve(
                user_id,
                query,
                corpus_id=corpus_id_value,
                limit=max_results,
            )
            packet = compact_personal_results(
                query,
                raw,
                max_results=max_results,
                max_chars=max_chars,
            )
            packet["context_kind"] = name
            if _telemetry_requested(args):
                raw_results = raw.get("results") if isinstance(raw, Mapping) else None
                packet["_telemetry"] = {
                    "token_estimate_method": "utf8_json_bytes_div4",
                    "retrieved_result_tokens_approx": _rough_tokens(raw),
                    "selected_evidence_tokens_approx": _rough_tokens(
                        packet.get("evidence", packet)
                    ),
                    "payload_tokens_approx": _rough_tokens(packet),
                    "retrieved_result_count": (
                        len(raw_results) if isinstance(raw_results, list) else None
                    ),
                }
            return packet
        if name == "get_evidence":
            chunk_ids = args.get("chunk_ids")
            if (
                not isinstance(chunk_ids, list)
                or not 1 <= len(chunk_ids) <= 20
                or not all(
                    isinstance(item, str) and re.fullmatch(r"chk_[a-f0-9]{32}", item)
                    for item in chunk_ids
                )
                or len(set(chunk_ids)) != len(chunk_ids)
            ):
                raise ZetsuError("invalid_chunk_ids")
            max_chars = _integer(
                args.get("max_chars", 12_000),
                label="max_chars",
                minimum=512,
                maximum=24_000,
            )
            typed_chunk_ids = tuple(item for item in chunk_ids if isinstance(item, str))
            raw_evidence = self._core().evidence(user_id, typed_chunk_ids)
            packet = compact_expanded_results(raw_evidence, max_chars=max_chars)
            if _telemetry_requested(args):
                packet["_telemetry"] = {
                    "token_estimate_method": "utf8_json_bytes_div4",
                    "selected_evidence_tokens_approx": _rough_tokens(raw_evidence),
                    "payload_tokens_approx": _rough_tokens(packet),
                    "selected_evidence_count": (
                        len(raw_evidence) if isinstance(raw_evidence, list) else None
                    ),
                }
            return packet
        if name == "delegate":
            instruction = _text(args.get("instruction"), label="instruction", maximum=40_000)
            domain = _text(args.get("domain", "general"), label="domain", maximum=64)
            lane_value = args.get("lane", "quality")
            if lane_value not in {"quality", "standard"}:
                raise ZetsuError("invalid_delegate_lane")
            result = self._core().chat(
                user_id=user_id,
                lane=ModelLane(str(lane_value)),
                messages=(
                    {"role": "system", "content": ZETSU_INSTRUCTIONS},
                    {"role": "user", "content": instruction},
                ),
                domain=domain,
            )
            packet = _bounded_model_result(
                result,
                max_chars=_integer(
                    args.get("max_chars", DEFAULT_MAX_CHARS),
                    label="max_chars",
                    minimum=512,
                    maximum=24_000,
                ),
            )
            if _telemetry_requested(args):
                response = result.get("response")
                usage = response.get("usage") if isinstance(response, Mapping) else None
                packet["_telemetry"] = {
                    "model_reported_usage": dict(usage) if isinstance(usage, Mapping) else None,
                    "model_reported_usage_exact": isinstance(usage, Mapping),
                    "payload_tokens_approx": _rough_tokens(packet),
                    "token_estimate_method": "utf8_json_bytes_div4",
                }
            return packet
        if name == "agent_task_status":
            return self._agent_service().task_status(
                user_id=user_id,
                session_id=_session_id(args.get("session_id")),
            )
        if name == "cancel_agent_task":
            return self._agent_service().cancel_queued(
                user_id=user_id,
                session_id=_session_id(args.get("session_id")),
            )
        if name == "agent_task":
            # Validate all non-effectful arguments before an agent can mutate its worktree.
            telemetry_requested = _telemetry_requested(args)
            repo_id = _text(args.get("repo_id"), label="repo_id", maximum=128)
            instruction = _text(args.get("instruction"), label="instruction", maximum=40_000)
            session_value = args.get("session_id")
            if session_value is not None:
                session_value = _session_id(session_value)
            lane_value = args.get("lane", "quality")
            if lane_value not in {"quality", "standard"}:
                raise ZetsuError("invalid_agent_lane")
            verification_argv = _verification_argv(args.get("verification_argv"))
            result = self._agent_service().run(
                user_id=user_id,
                repo_id=repo_id,
                instruction=instruction,
                lane=ModelLane(str(lane_value)),
                session_id=session_value,
                max_steps=_integer(
                    args.get("max_steps", 12), label="max_steps", minimum=1, maximum=32
                ),
                max_chars=_integer(
                    args.get("max_chars", DEFAULT_MAX_CHARS),
                    label="max_chars",
                    minimum=512,
                    maximum=24_000,
                ),
                verification_argv=verification_argv,
                apply_to_repository=_boolean(args, "apply_to_repository"),
                wait_timeout_seconds=_integer(
                    args.get("wait_timeout_seconds", 1_800),
                    label="wait_timeout_seconds",
                    minimum=1,
                    maximum=3_600,
                ),
            )
            if not telemetry_requested:
                result.pop("telemetry", None)
            return _compact_agent_result(result)
        if name == "rtl_task":
            session_id = _session_id(args.get("session_id"))
            instruction = _text(args.get("instruction"), label="instruction", maximum=40_000)
            editable = args.get("editable_sources")
            if (
                not isinstance(editable, list)
                or len(editable) != 1
                or not all(isinstance(item, str) and item for item in editable)
            ):
                raise ZetsuError("invalid_editable_sources")
            task_kind = args.get("task_kind")
            rtl_scope = args.get("rtl_scope")
            if task_kind not in {"implementation", "repair"}:
                raise ZetsuError("invalid_task_kind")
            if rtl_scope != "bounded_module":
                raise ZetsuError("invalid_rtl_scope")
            metadata = RoutingTaskMetadata(
                task_id=session_id,
                experiment_arm="zetsu",
                domain="systemverilog",
                task_kind=task_kind,
                rtl_scope=rtl_scope,
                worker_eligible=True,
                editable_sources=tuple(editable),
                module_count=_integer(
                    args.get("module_count"), label="module_count", minimum=1, maximum=1
                ),
                synthesizable=True,
                explicit_ports=True,
                cycle_behavior_specified=True,
                deterministic_verification=True,
                unresolved_architecture=False,
            )
            eligibility = assess_rtl_worker_eligibility(metadata)
            if not eligibility.eligible:
                raise ZetsuError(f"rtl_task_not_eligible:{eligibility.reason}")
            task_kind_value: Literal["implementation", "repair"] = (
                "implementation" if task_kind == "implementation" else "repair"
            )
            module_count = _integer(
                args.get("module_count"), label="module_count", minimum=1, maximum=1
            )
            result = self._core().rtl_task(
                user_id=user_id,
                instruction=instruction,
                task_kind=task_kind_value,
                editable_sources=tuple(editable),
                module_count=module_count,
                session_id=session_id,
            )
            return _bounded_model_result(
                result,
                max_chars=_integer(
                    args.get("max_chars", DEFAULT_MAX_CHARS),
                    label="max_chars",
                    minimum=512,
                    maximum=24_000,
                ),
            )
        if name == "get_result":
            return self._agent_service().results.page(
                user_id=user_id,
                repo_id=_text(args.get("repo_id"), label="repo_id", maximum=128),
                session_id=_session_id(args.get("session_id")),
                result_id=_text(args.get("result_id"), label="result_id", maximum=36),
                artifact=_text(args.get("artifact"), label="artifact", maximum=128),
                offset=_integer(
                    args.get("offset", 0), label="offset", minimum=0, maximum=2**63 - 1
                ),
                max_bytes=_integer(
                    args.get("max_bytes", 24_000),
                    label="max_bytes",
                    minimum=1,
                    maximum=65_536,
                ),
            )
        session_id = _session_id(args.get("session_id"))
        status = self.tiered.agent_session_status(user_id=user_id, session_id=session_id)
        verification = status.get("last_result")
        worktree_status = status.get("worktree_status")
        if verification is None and isinstance(worktree_status, Mapping):
            verification = {
                "verification_summary": worktree_status.get("verification_summary"),
                "changed_paths": worktree_status.get("changed_paths"),
                "diff_hash": worktree_status.get("diff_hash"),
            }
        result = {
            "status": status.get("status"),
            "session_id": session_id,
            "repo_id": status.get("repo_id"),
            "worktree_status": worktree_status,
            "verification": verification,
        }
        if _boolean(args, "include_patch"):
            result["handoff"] = self._agent_service().handoff_evidence(
                session_id,
                max_chars=_integer(
                    args.get("max_chars", DEFAULT_MAX_CHARS),
                    label="max_chars",
                    minimum=512,
                    maximum=24_000,
                ),
            )
        return result


@dataclass(frozen=True)
class McpDispatchResult:
    status_code: int
    payload: JsonObject | None


class ZetsuMcpDispatcher:
    """Small stateless MCP dispatcher supporting current and legacy Codex clients."""

    def __init__(self, service: ZetsuService) -> None:
        self.service = service

    @staticmethod
    def _response(request_id: object, result: JsonObject) -> JsonObject:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error(request_id: object, code: int, message: str) -> JsonObject:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }

    def dispatch(
        self,
        user_id: str,
        request: Mapping[str, object],
        *,
        protocol_header: str | None = None,
    ) -> McpDispatchResult:
        request_id = request.get("id")
        if request.get("jsonrpc") != "2.0" or not isinstance(request.get("method"), str):
            return McpDispatchResult(400, self._error(request_id, -32600, "Invalid Request"))
        method = str(request["method"])
        if method.startswith("notifications/"):
            return McpDispatchResult(202, None)
        modern = protocol_header == MCP_LATEST_PROTOCOL_VERSION
        if protocol_header is not None and protocol_header not in MCP_SUPPORTED_PROTOCOL_VERSIONS:
            return McpDispatchResult(
                400,
                self._error(request_id, -32600, "Unsupported MCP protocol version"),
            )
        if method == "server/discover":
            status = self.service.status(user_id)
            return McpDispatchResult(
                200,
                self._response(
                    request_id,
                    {
                        "resultType": "complete",
                        "supportedVersions": list(MCP_SUPPORTED_PROTOCOL_VERSIONS),
                        "capabilities": {"tools": {}},
                        "instructions": ZETSU_INSTRUCTIONS,
                        "ttlMs": 300_000,
                        "cacheScope": "private",
                        "_meta": {
                            "io.modelcontextprotocol/serverInfo": {
                                "name": "Laplace Zetsu",
                                "version": ZETSU_SCHEMA_VERSION,
                            },
                            "org.laplace/zetsu": status,
                        },
                    },
                ),
            )
        if method == "initialize":
            params = request.get("params")
            requested = params.get("protocolVersion") if isinstance(params, dict) else None
            selected = (
                str(requested)
                if requested in MCP_LEGACY_PROTOCOL_VERSIONS
                else MCP_LEGACY_PROTOCOL_VERSIONS[0]
            )
            return McpDispatchResult(
                200,
                self._response(
                    request_id,
                    {
                        "protocolVersion": selected,
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {"name": "Laplace Zetsu", "version": ZETSU_SCHEMA_VERSION},
                        "instructions": ZETSU_INSTRUCTIONS,
                    },
                ),
            )
        if method == "ping":
            result: JsonObject = {"resultType": "complete"} if modern else {}
            return McpDispatchResult(200, self._response(request_id, result))
        if method == "tools/list":
            result = {"tools": list(self.service.available_tools(user_id))}
            if modern:
                result.update({"resultType": "complete", "ttlMs": 300_000, "cacheScope": "private"})
            return McpDispatchResult(200, self._response(request_id, result))
        if method == "tools/call":
            try:
                params = _object(request.get("params"), label="tool_call")
                name = _text(params.get("name"), label="tool_name", maximum=128)
                arguments = _object(params.get("arguments", {}), label="tool_arguments")
                value = self.service.call(user_id, name, arguments)
            except (
                ValueError,
                ZetsuError,
                ServiceTierError,
                AgentSandboxError,
                CorpusError,
                ZetsuResultError,
            ) as exc:
                category = getattr(exc, "category", str(exc))
                result = {
                    "content": [{"type": "text", "text": json.dumps({"error": category})}],
                    "structuredContent": {"error": category},
                    "isError": True,
                }
            else:
                result = {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(value, sort_keys=True, separators=(",", ":")),
                        }
                    ],
                    "structuredContent": value,
                    "isError": False,
                }
            if modern:
                result["resultType"] = "complete"
            return McpDispatchResult(200, self._response(request_id, result))
        return McpDispatchResult(404, self._error(request_id, -32601, "Method not found"))
