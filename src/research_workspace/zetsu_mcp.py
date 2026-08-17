"""Versioned, owner-authorized Zetsu MCP semantics.

HTTP transport and Laplace authentication are mounted by :mod:`operator_api` so
Zetsu cannot accidentally create a weaker parallel identity or public daemon.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, TypeAlias, cast

from .model_routing import (
    RoutingTaskMetadata,
    assess_rtl_worker_eligibility,
)
from .personal_corpus import PersonalCorpusStore
from .service_tiers import ModelLane, TieredServingService
from .user_capabilities import Capability
from .versioning import version_record
from .zetsu_context import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_RESULTS,
    compact_expanded_results,
    compact_personal_results,
    normalize_budget,
)

JsonObject: TypeAlias = dict[str, object]

ZETSU_SCHEMA_VERSION = "1.1"
ZETSU_SKILL_VERSION = "1.1.0"
MCP_LATEST_PROTOCOL_VERSION = "2026-07-28"
MCP_LEGACY_PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26")
MCP_SUPPORTED_PROTOCOL_VERSIONS = (MCP_LATEST_PROTOCOL_VERSION, *MCP_LEGACY_PROTOCOL_VERSIONS)
ZETSU_INSTRUCTIONS = (
    "Use Zetsu for compact owner-authorized evidence and persistent Laplace memory. "
    "Use Codex local repository and shell tools for the current checkout. Expand only "
    "selected evidence IDs. Delegate only bounded tasks; CodeV is restricted to policy-"
    "eligible RTL implementation or repair with deterministic verification."
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
            "description": "Compact ranked owner-authorized evidence with exact chunk provenance.",
            "inputSchema": _schema(
                {
                    "query": query,
                    "max_results": count,
                    "max_chars": budget,
                    "corpus_id": {"type": ["string", "null"], "maxLength": 128},
                },
                ("query",),
            ),
            "annotations": {"readOnlyHint": True, "openWorldHint": False},
        },
        {
            "name": "get_evidence",
            "description": "Expand only selected evidence chunk IDs within a hard response budget.",
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
                },
                ("chunk_ids",),
            ),
            "annotations": {"readOnlyHint": True, "openWorldHint": False},
        },
        {
            "name": "project_context",
            "description": "Compact persistent project context; expand selected chunks only.",
            "inputSchema": _schema(
                {"query": query, "max_results": count, "max_chars": budget},
                ("query",),
            ),
            "annotations": {"readOnlyHint": True, "openWorldHint": False},
        },
        {
            "name": "experiment_context",
            "description": "Compact prior experiment/result context with preserved provenance.",
            "inputSchema": _schema(
                {"query": query, "max_results": count, "max_chars": budget},
                ("query",),
            ),
            "annotations": {"readOnlyHint": True, "openWorldHint": False},
        },
        {
            "name": "delegate",
            "description": "Bounded Qwen supervisor reasoning with no repository or shell access.",
            "inputSchema": _schema(
                {
                    "instruction": {"type": "string", "minLength": 1, "maxLength": 40_000},
                    "domain": {"type": "string", "maxLength": 64},
                    "lane": {"type": "string", "enum": ["quality", "standard"]},
                    "max_chars": budget,
                },
                ("instruction",),
            ),
            "annotations": {"readOnlyHint": True, "openWorldHint": False},
        },
        {
            "name": "rtl_task",
            "description": "Policy-bounded CodeV RTL implementation/repair in an existing worktree.",
            "inputSchema": _schema(
                {
                    "session_id": {"type": "string", "minLength": 1, "maxLength": 128},
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
            "annotations": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False},
        },
        {
            "name": "verify",
            "description": "Return deterministic verification evidence for an owned agent session.",
            "inputSchema": _schema(
                {"session_id": {"type": "string", "minLength": 1, "maxLength": 128}},
                ("session_id",),
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
    "rtl_task": Capability.AGENT,
    "verify": Capability.AGENT,
}

_TOOL_ARGUMENTS: Mapping[str, frozenset[str]] = {
    "search": frozenset({"query", "max_results", "max_chars", "corpus_id"}),
    "get_evidence": frozenset({"chunk_ids", "max_chars"}),
    "project_context": frozenset({"query", "max_results", "max_chars"}),
    "experiment_context": frozenset({"query", "max_results", "max_chars"}),
    "delegate": frozenset({"instruction", "domain", "lane", "max_chars"}),
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
    "verify": frozenset({"session_id"}),
}


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
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.corpus = corpus
        self.tiered = tiered

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
                "available": "codev" in economy.model_id.casefold(),
                "policy": "bounded_policy_eligible_rtl_only",
            },
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
            raw = self.corpus.search(
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
            return packet
        if name == "get_evidence":
            chunk_ids = args.get("chunk_ids")
            if not isinstance(chunk_ids, list) or not all(isinstance(item, str) for item in chunk_ids):
                raise ZetsuError("invalid_chunk_ids")
            max_chars = _integer(
                args.get("max_chars", 12_000),
                label="max_chars",
                minimum=512,
                maximum=24_000,
            )
            raw = self.corpus.evidence(user_id, tuple(chunk_ids))
            return compact_expanded_results(raw, max_chars=max_chars)
        if name == "delegate":
            instruction = _text(args.get("instruction"), label="instruction", maximum=40_000)
            domain = _text(args.get("domain", "general"), label="domain", maximum=64)
            lane_value = args.get("lane", "quality")
            if lane_value not in {"quality", "standard"}:
                raise ZetsuError("invalid_delegate_lane")
            result = self.tiered.chat(
                user_id=user_id,
                lane=ModelLane(str(lane_value)),
                messages=(
                    {"role": "system", "content": ZETSU_INSTRUCTIONS},
                    {"role": "user", "content": instruction},
                ),
                domain=domain,
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
        if name == "rtl_task":
            session_id = _text(args.get("session_id"), label="session_id", maximum=128)
            instruction = _text(args.get("instruction"), label="instruction", maximum=40_000)
            editable = args.get("editable_sources")
            if not isinstance(editable, list) or len(editable) != 1 or not all(
                isinstance(item, str) and item for item in editable
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
                    args.get("module_count"), label="module_count", minimum=1, maximum=8
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
            result = self.tiered.agent(
                user_id=user_id,
                session_id=session_id,
                lane=ModelLane.ECONOMY,
                instruction=instruction,
                domain="systemverilog",
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
        session_id = _text(args.get("session_id"), label="session_id", maximum=128)
        status = self.tiered.agent_session_status(user_id=user_id, session_id=session_id)
        return {
            "status": status.get("status"),
            "session_id": session_id,
            "repo_id": status.get("repo_id"),
            "worktree_status": status.get("worktree_status"),
            "verification": status.get("last_result"),
        }


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
            params = _object(request.get("params"), label="tool_call")
            name = _text(params.get("name"), label="tool_name", maximum=128)
            arguments = _object(params.get("arguments", {}), label="tool_arguments")
            try:
                value = self.service.call(user_id, name, arguments)
            except (ValueError, ZetsuError) as exc:
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
