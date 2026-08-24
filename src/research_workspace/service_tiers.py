"""Orthogonal capability tiers, model lanes, validation, and request auditing."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess  # nosec B404
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Iterator, Protocol, TypeAlias
from urllib.parse import urlsplit

from .agent_sandbox import (
    AgentSandboxError,
    AgentSandboxManager,
    AgentSessionBinding,
    AgentToolPolicy,
)
from .domain_registry import DEFAULT_DOMAIN_REGISTRY, DomainRegistry
from .repository_authorization import (
    RepositoryAuthorizationError,
    validate_workspace_path,
)
from .user_capabilities import (
    Capability,
    CapabilityTier,
    UserCapabilityError,
    UserCapabilityStore,
)

JsonObject: TypeAlias = dict[str, object]


class ModelLane(StrEnum):
    QUALITY = "quality"
    STANDARD = "standard"
    ECONOMY = "economy"


class ServiceTierError(RuntimeError):
    """A tier, lane, or validation policy rejected a request."""

    def __init__(self, category: str, evidence: JsonObject | None = None) -> None:
        super().__init__(category)
        self.category = category
        self.evidence = evidence or {}


@dataclass(frozen=True)
class ModelRoute:
    lane: ModelLane
    model_id: str
    endpoint: str
    priority: int
    context_limit: int = 32_768
    output_limit: int = 2_048

    def __post_init__(self) -> None:
        if self.context_limit < 2_048 or self.output_limit < 1:
            raise ValueError("invalid model route limits")


@dataclass(frozen=True)
class LanePolicy:
    routes: Mapping[ModelLane, ModelRoute]
    quality_reserved_slots: int = 2
    standard_capacity: int = 6
    economy_capacity: int = 12

    def __post_init__(self) -> None:
        if set(self.routes) != set(ModelLane):
            raise ValueError("all model lanes require independent routes")
        if self.routes[ModelLane.QUALITY].model_id == self.routes[ModelLane.ECONOMY].model_id:
            raise ValueError("quality and economy routes must be independently configured")
        if self.quality_reserved_slots not in {1, 2}:
            raise ValueError("quality reserved slots must be 1 or 2")
        if self.standard_capacity not in {2, 4, 6, 8}:
            raise ValueError("invalid standard capacity")
        if self.economy_capacity not in {4, 8, 12}:
            raise ValueError("invalid economy capacity")


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    gate_id: str
    reason: str


@dataclass(frozen=True)
class AdmissionTicket:
    ticket_id: str
    lane: ModelLane
    queue_position_at_arrival: int
    queue_wait_seconds: float = 0.0


class PriorityAdmissionScheduler:
    """In-process reservation policy with observable, deterministic queue order."""

    def __init__(self, policy: LanePolicy) -> None:
        self.policy = policy
        self._condition = threading.Condition()
        self._active: dict[ModelLane, int] = {lane: 0 for lane in ModelLane}
        self._waiting: list[AdmissionTicket] = []

    @staticmethod
    def _priority(lane: ModelLane) -> int:
        return {
            ModelLane.QUALITY: 0,
            ModelLane.STANDARD: 1,
            ModelLane.ECONOMY: 2,
        }[lane]

    def _capacity_available(self, lane: ModelLane) -> bool:
        total = sum(self._active.values())
        if lane is ModelLane.QUALITY:
            return total < self.policy.economy_capacity
        lower_tier_limit = self.policy.economy_capacity - self.policy.quality_reserved_slots
        if total >= lower_tier_limit:
            return False
        if lane is ModelLane.STANDARD:
            return self._active[lane] < self.policy.standard_capacity
        return self._active[lane] < self.policy.economy_capacity

    def _is_next(self, ticket: AdmissionTicket) -> bool:
        eligible = sorted(
            self._waiting,
            key=lambda item: (self._priority(item.lane), self._waiting.index(item)),
        )
        return bool(eligible) and eligible[0] == ticket

    @contextmanager
    def admit(self, lane: ModelLane) -> Iterator[AdmissionTicket]:
        queued_at = time.monotonic()
        with self._condition:
            ticket = AdmissionTicket(
                ticket_id=f"queue-{uuid.uuid4().hex}",
                lane=lane,
                queue_position_at_arrival=len(self._waiting),
            )
            self._waiting.append(ticket)
            while not self._is_next(ticket) or not self._capacity_available(lane):
                self._condition.wait()
            self._waiting.remove(ticket)
            self._active[lane] += 1
            admitted = AdmissionTicket(
                ticket_id=ticket.ticket_id,
                lane=ticket.lane,
                queue_position_at_arrival=ticket.queue_position_at_arrival,
                queue_wait_seconds=max(0.0, time.monotonic() - queued_at),
            )
        try:
            yield admitted
        finally:
            with self._condition:
                self._active[lane] -= 1
                self._condition.notify_all()

    def snapshot(self) -> JsonObject:
        with self._condition:
            ordered = sorted(
                self._waiting,
                key=lambda item: (self._priority(item.lane), self._waiting.index(item)),
            )
            return {
                "active": {lane.value: self._active[lane] for lane in ModelLane},
                "waiting": [
                    {
                        "ticket_id": item.ticket_id,
                        "lane": item.lane.value,
                        "queue_position": index,
                    }
                    for index, item in enumerate(ordered)
                ],
                "quality_reserved_slots": self.policy.quality_reserved_slots,
                "standard_capacity": self.policy.standard_capacity,
                "economy_capacity": self.policy.economy_capacity,
            }


class ChatBackend(Protocol):
    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        route: ModelRoute,
        tools: Sequence[Mapping[str, object]],
        request_id: str,
    ) -> JsonObject: ...


class ResponseValidator(Protocol):
    def validate(self, response: JsonObject, *, domain: str) -> ValidationResult: ...


class AgentBackend(Protocol):
    def run(
        self,
        *,
        binding: AgentSessionBinding,
        instruction: str,
        route: ModelRoute,
        request_id: str,
    ) -> JsonObject: ...


class LocalOpenAIChatBackend:
    """Small localhost-only adapter for the frozen OpenAI-compatible servers."""

    def __init__(self, *, timeout_seconds: float = 300) -> None:
        self.timeout_seconds = timeout_seconds

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        route: ModelRoute,
        tools: Sequence[Mapping[str, object]],
        request_id: str,
    ) -> JsonObject:
        endpoint = urlsplit(route.endpoint)
        if endpoint.scheme != "http" or endpoint.hostname not in {"127.0.0.1", "localhost"}:
            raise ServiceTierError("non_local_model_endpoint", {"endpoint": route.endpoint})
        body: JsonObject = {
            "model": route.model_id,
            "messages": [dict(message) for message in messages],
            "temperature": 0,
            "max_tokens": route.output_limit,
            "priority": route.priority,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if tools:
            if len(tools) != 1:
                raise ServiceTierError("unsupported_tool_schema_count")
            tool = tools[0]
            function = tool.get("function")
            if (
                tool.get("type") != "function"
                or not isinstance(function, Mapping)
                or not isinstance(function.get("name"), str)
                or not isinstance(function.get("parameters"), Mapping)
            ):
                raise ServiceTierError("invalid_tool_schema")
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": function["name"],
                    "description": function.get("description", ""),
                    "schema": dict(function["parameters"]),
                    "strict": True,
                },
            }
        request = urllib.request.Request(  # nosec B310 - localhost checked above
            route.endpoint.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Request-Id": request_id,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(  # nosec B310 - localhost checked above
                request, timeout=self.timeout_seconds
            ) as response:
                raw: object = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            error_body = exc.read(4_096).decode("utf-8", errors="replace")
            raise ServiceTierError(
                "local_model_request_failed",
                {
                    "error_type": type(exc).__name__,
                    "http_status": exc.code,
                    "model_id": route.model_id,
                    "response_body": error_body,
                },
            ) from exc
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise ServiceTierError(
                "local_model_request_failed",
                {"error_type": type(exc).__name__, "model_id": route.model_id},
            ) from exc
        if not isinstance(raw, dict):
            raise ServiceTierError("invalid_model_response")
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ServiceTierError("invalid_model_response")
        first = choices[0]
        message = first.get("message")
        if not isinstance(message, dict):
            raise ServiceTierError("invalid_model_response")
        content = message.get("content")
        return {
            "content": content if isinstance(content, str) else "",
            "tool_calls": message.get("tool_calls", []),
            "finish_reason": first.get("finish_reason"),
            "usage": raw.get("usage", {}),
        }


class ValidatedPatchAgentBackend:
    """Validate a structured edit and apply its server-rendered Git patch."""

    _TOOL: Mapping[str, object] = {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": (
                "Replace one exact text occurrence in one file inside the bound "
                "repository worktree. The server renders and validates the Git patch."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Repository-relative file path.",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Exact non-empty text currently present once.",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Exact replacement text.",
                    },
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    }

    def __init__(self, chat: ChatBackend) -> None:
        self.chat = chat

    @staticmethod
    def _extract_edit(response: JsonObject) -> tuple[str, str, str]:
        parsed: object | None = None
        tool_calls = response.get("tool_calls")
        if isinstance(tool_calls, list) and len(tool_calls) == 1:
            call = tool_calls[0]
            function = call.get("function") if isinstance(call, dict) else None
            if isinstance(function, dict) and function.get("name") == "apply_patch":
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    parsed = json.loads(arguments)
        if parsed is None:
            content = response.get("content")
            if isinstance(content, str):
                parsed = json.loads(content)
        if isinstance(parsed, dict):
            path = parsed.get("path")
            old_text = parsed.get("old_text")
            new_text = parsed.get("new_text")
            if all(isinstance(value, str) for value in (path, old_text, new_text)):
                return str(path), str(old_text), str(new_text)
        raise ServiceTierError("agent_edit_missing")

    @staticmethod
    def _render_patch(
        binding: AgentSessionBinding,
        path: str,
        old_text: str,
        new_text: str,
    ) -> str:
        encoded_edit_size = len((path + old_text + new_text).encode("utf-8"))
        if (
            not old_text
            or old_text == new_text
            or encoded_edit_size > 2_000_000
            or "\x00" in path + old_text + new_text
        ):
            raise ServiceTierError("agent_edit_invalid")
        worktree = Path(binding.worktree_root)
        target = validate_workspace_path(worktree, path)
        if path == ".git" or path.startswith(".git/"):
            raise ServiceTierError("agent_patch_git_metadata")
        if not target.is_file():
            raise ServiceTierError("agent_edit_target_unavailable", {"path": path})
        original_bytes = target.read_bytes()
        if len(original_bytes) > 2_000_000:
            raise ServiceTierError("agent_edit_target_too_large", {"path": path})
        try:
            original = original_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ServiceTierError(
                "agent_edit_target_not_utf8",
                {"path": path},
            ) from exc
        occurrences = original.count(old_text)
        if occurrences != 1:
            raise ServiceTierError(
                "agent_edit_match_not_unique",
                {"path": path, "match_count": occurrences},
            )
        target.write_text(
            original.replace(old_text, new_text, 1),
            encoding="utf-8",
            newline="",
        )
        try:
            rendered = subprocess.run(  # nosec B603 B607 - fixed Git query
                ["git", "-C", str(worktree), "diff", "--no-ext-diff", "--", path],
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
                env=AgentSandboxManager.fixed_environment(binding),
            )
        finally:
            target.write_bytes(original_bytes)
        if rendered.returncode != 0 or not rendered.stdout:
            raise ServiceTierError(
                "agent_diff_render_failed",
                {"returncode": rendered.returncode},
            )
        return rendered.stdout

    @staticmethod
    def _validate_patch(binding: AgentSessionBinding, patch: str) -> tuple[str, ...]:
        if len(patch.encode("utf-8")) > 2_000_000 or "\x00" in patch:
            raise ServiceTierError("agent_patch_too_large")
        forbidden_markers = (
            "GIT binary patch",
            "new file mode 120000",
            "old mode 120000",
            "rename from ",
            "rename to ",
            "copy from ",
            "copy to ",
        )
        if any(marker in patch for marker in forbidden_markers):
            raise ServiceTierError("agent_patch_forbidden_form")
        paths: list[str] = []
        for line in patch.splitlines():
            if not line.startswith("diff --git a/"):
                continue
            parts = line.split()
            if len(parts) != 4 or not parts[2].startswith("a/") or not parts[3].startswith("b/"):
                raise ServiceTierError("agent_patch_invalid_header")
            left = parts[2][2:]
            right = parts[3][2:]
            if left != right:
                raise ServiceTierError("agent_patch_cross_path")
            validate_workspace_path(Path(binding.worktree_root), right)
            if right == ".git" or right.startswith(".git/"):
                raise ServiceTierError("agent_patch_git_metadata")
            paths.append(right)
        if not paths:
            raise ServiceTierError("agent_patch_empty")
        return tuple(sorted(set(paths)))

    def run(
        self,
        *,
        binding: AgentSessionBinding,
        instruction: str,
        route: ModelRoute,
        request_id: str,
    ) -> JsonObject:
        if "apply_patch" not in binding.tool_policy.allowed_tools:
            raise ServiceTierError("agent_tool_not_allowed", {"tool": "apply_patch"})
        response = self.chat.complete(
            messages=(
                {
                    "role": "system",
                    "content": (
                        "You are in a server-bound Git worktree. Return exactly one JSON "
                        'object with the fields "path", "old_text", and "new_text". The '
                        "path must be repository-relative; old_text must be an exact, "
                        "non-empty substring that occurs once; new_text is its replacement. "
                        "Never emit commands, paths outside the repository, symlinks, binary "
                        "content, renames, or network actions."
                    ),
                },
                {"role": "user", "content": instruction},
            ),
            route=route,
            tools=(self._TOOL,),
            request_id=request_id,
        )
        try:
            path, old_text, new_text = self._extract_edit(response)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ServiceTierError("agent_edit_missing") from exc
        patch = self._render_patch(binding, path, old_text, new_text)
        paths = self._validate_patch(binding, patch)
        worktree = Path(binding.worktree_root)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".laplace-agent-", suffix=".patch", dir=worktree
        )
        temporary = Path(temporary_name)
        rendered_diff = ""
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(patch)
                handle.flush()
                os.fsync(handle.fileno())
            for arguments in (
                ("apply", "--check", "--recount", str(temporary)),
                ("apply", "--whitespace=error", "--recount", str(temporary)),
                ("diff", "--check"),
            ):
                result = subprocess.run(  # nosec B603 B607 - fixed git executable and verbs
                    ["git", "-C", str(worktree), *arguments],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=120,
                    env=AgentSandboxManager.fixed_environment(binding),
                )
                if result.returncode != 0:
                    raise ServiceTierError(
                        "agent_patch_validation_failed",
                        {
                            "git_operation": arguments[0],
                            "returncode": result.returncode,
                            "stderr": result.stderr[-2_000:],
                        },
                    )
            diff_result = subprocess.run(  # nosec B603 B607 - fixed Git query
                ["git", "-C", str(worktree), "diff", "--no-ext-diff", "--"],
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
                env=AgentSandboxManager.fixed_environment(binding),
            )
            if diff_result.returncode != 0:
                raise ServiceTierError(
                    "agent_diff_render_failed",
                    {"returncode": diff_result.returncode},
                )
            rendered_diff = diff_result.stdout
        finally:
            temporary.unlink(missing_ok=True)
        return {
            "content": f"Validated patch applied to {len(paths)} path(s).",
            "finish_reason": "stop",
            "verification_status": "PASSED",
            "modified_paths": list(paths),
            "diff": rendered_diff,
            "tests": [
                {"name": "Patch preflight", "status": "PASSED"},
                {"name": "Git whitespace validation", "status": "PASSED"},
            ],
        }


class TierAuditLog:
    """Append-only JSONL audit with no prompts, secrets, or response bodies."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, value: Mapping[str, object]) -> None:
        line = json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n"
        with self._lock:
            descriptor = os.open(
                self.path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o600,
            )
            try:
                os.write(descriptor, line.encode("utf-8"))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


class StrictResponseValidator:
    """Deterministic hard gates shared by fixture and live certification."""

    def validate(self, response: JsonObject, *, domain: str) -> ValidationResult:
        content = response.get("content")
        if not isinstance(content, str) or not content.strip():
            return ValidationResult(False, "non_empty_content", "missing response content")
        finish_reason = response.get("finish_reason")
        if finish_reason in {"length", "max_tokens"}:
            return ValidationResult(False, "no_silent_truncation", "output was truncated")
        if domain == "json":
            try:
                parsed: object = json.loads(content)
            except json.JSONDecodeError:
                return ValidationResult(False, "valid_json", "malformed JSON")
            if not isinstance(parsed, dict):
                return ValidationResult(False, "valid_json", "JSON root must be an object")
        citations = response.get("citations")
        if citations is not None and (
            not isinstance(citations, list)
            or any(not isinstance(item, dict) or not item.get("source_id") for item in citations)
        ):
            return ValidationResult(False, "citation_integrity", "citation lacks source_id")
        if response.get("verification_status") == "FAILED" and response.get("status") == "SUCCESS":
            return ValidationResult(
                False,
                "verification_truthfulness",
                "failed gate was represented as success",
            )
        return ValidationResult(True, "strict_response", "all deterministic gates passed")


class TieredServingService:
    """Enforce capability before routing and preserve lane independence."""

    def __init__(
        self,
        *,
        users: UserCapabilityStore,
        sandboxes: AgentSandboxManager,
        lane_policy: LanePolicy,
        chat_backend: ChatBackend,
        agent_backend: AgentBackend,
        audit_log: TierAuditLog,
        validator: ResponseValidator | None = None,
        scheduler: PriorityAdmissionScheduler | None = None,
        domain_registry: DomainRegistry = DEFAULT_DOMAIN_REGISTRY,
    ) -> None:
        self.users = users
        self.sandboxes = sandboxes
        self.lane_policy = lane_policy
        self.chat_backend = chat_backend
        self.agent_backend = agent_backend
        self.audit_log = audit_log
        self.validator = validator or StrictResponseValidator()
        self.scheduler = scheduler or PriorityAdmissionScheduler(lane_policy)
        self.domain_registry = domain_registry
        self._session_lock = threading.Lock()
        self._session_results: dict[str, JsonObject] = {}
        self._cancelled_sessions: dict[str, str] = {}

    def _audit_denial(
        self,
        *,
        user_id: str,
        mode: str,
        failure_category: str,
        session_id: str | None = None,
        repo_id: str | None = None,
        requested_lane: ModelLane | None = None,
    ) -> None:
        self.audit_log.append(
            {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "outcome": "DENIED",
                "failure_category": failure_category,
                "user_id": user_id,
                "session_id": session_id,
                "mode": mode,
                "repo_id": repo_id,
                "sandbox_id": session_id if mode == "agent" else None,
                "requested_quality_lane": (
                    requested_lane.value if requested_lane is not None else None
                ),
                "effective_quality_lane": None,
                "route": None,
                "tool_policy_id": "chat-only-no-tools-v1" if mode == "chat" else None,
                "network_policy_id": (
                    "local-model-only-v1" if mode == "chat" else "network-denied-v1"
                ),
                "queue_wait_seconds": 0.0,
                "trace_id": f"denied-{uuid.uuid4().hex}",
            }
        )

    def capability(self, user_id: str) -> CapabilityTier:
        return self.users.require(
            user_id,
            frozenset({CapabilityTier.BASIC, CapabilityTier.PLUS, CapabilityTier.OPERATOR}),
        ).tier

    def effective_capabilities(self, user_id: str) -> frozenset[Capability]:
        assignment = self.users.get(user_id)
        if not assignment.enabled:
            raise UserCapabilityError("user_disabled")
        return assignment.capabilities

    def _route(self, lane: ModelLane, *, domain: str) -> ModelRoute:
        if lane is ModelLane.ECONOMY and domain != "systemverilog":
            standard = self.lane_policy.routes[ModelLane.STANDARD]
            economy = self.lane_policy.routes[ModelLane.ECONOMY]
            return ModelRoute(
                lane=ModelLane.ECONOMY,
                model_id=standard.model_id,
                endpoint=standard.endpoint,
                priority=economy.priority,
                context_limit=economy.context_limit,
                output_limit=economy.output_limit,
            )
        return self.lane_policy.routes[lane]

    def chat(
        self,
        *,
        user_id: str,
        lane: ModelLane,
        messages: Sequence[Mapping[str, str]],
        domain: str = "general",
        session_id: str | None = None,
    ) -> JsonObject:
        effective_session_id = session_id or f"chat-session-{uuid.uuid4().hex}"
        self.domain_registry.require(domain, surface="chat")
        try:
            capability = self.users.require_capability(user_id, Capability.CHAT)
        except UserCapabilityError as exc:
            self._audit_denial(
                user_id=user_id,
                mode="chat",
                failure_category=str(exc),
                session_id=effective_session_id,
                requested_lane=lane,
            )
            raise
        if not messages or any(
            set(message) != {"role", "content"}
            or message.get("role") not in {"system", "user", "assistant"}
            or not isinstance(message.get("content"), str)
            for message in messages
        ):
            self._audit_denial(
                user_id=user_id,
                mode="chat",
                failure_category="invalid_chat_messages",
                session_id=effective_session_id,
                requested_lane=lane,
            )
            raise ServiceTierError("invalid_chat_messages")
        request_id = f"chat-{uuid.uuid4().hex}"
        trace_id = f"trace-{uuid.uuid4().hex}"
        try:
            route = self._route(lane, domain=domain)
        except ServiceTierError as exc:
            self._audit_denial(
                user_id=user_id,
                mode="chat",
                failure_category=str(getattr(exc, "category", str(exc))),
                session_id=effective_session_id,
                requested_lane=lane,
            )
            raise
        with self.scheduler.admit(lane) as ticket:
            # Intentionally pass an empty tuple, not dormant tool definitions.
            response = self.chat_backend.complete(
                messages=messages,
                route=route,
                tools=(),
                request_id=request_id,
            )
            result = self.validator.validate(response, domain=domain)
            escalation: JsonObject | None = None
            if not result.passed and lane is not ModelLane.QUALITY:
                quality = self.lane_policy.routes[ModelLane.QUALITY]
                response = self.chat_backend.complete(
                    messages=messages,
                    route=quality,
                    tools=(),
                    request_id=request_id,
                )
                second = self.validator.validate(response, domain=domain)
                escalation = {
                    "from_lane": lane.value,
                    "to_lane": ModelLane.QUALITY.value,
                    "trigger_gate": result.gate_id,
                    "passed": second.passed,
                }
                result = second
                route = quality
        request_hash = hashlib.sha256(
            json.dumps(list(messages), sort_keys=True).encode("utf-8")
        ).hexdigest()
        self.audit_log.append(
            {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "request_id": request_id,
                "request_sha256": request_hash,
                "user_id": user_id,
                "capability_tier": capability.tier.value,
                "session_id": effective_session_id,
                "mode": "chat",
                "repo_id": None,
                "sandbox_id": None,
                "requested_quality_lane": lane.value,
                "effective_quality_lane": route.lane.value,
                "route": route.model_id,
                "tool_policy_id": "chat-only-no-tools-v1",
                "network_policy_id": "local-model-only-v1",
                "queue_wait_seconds": ticket.queue_wait_seconds,
                "trace_id": trace_id,
                "requested_tier": lane.value,
                "effective_tier": route.lane.value,
                "priority": route.priority,
                "context_limit": route.context_limit,
                "output_limit": route.output_limit,
                "escalated": escalation is not None,
                "escalation_reason": (
                    escalation.get("trigger_gate") if escalation is not None else None
                ),
                "validator_results": [asdict(result)],
                "requested_lane": lane.value,
                "effective_model_id": route.model_id,
                "domain": domain,
                "tool_schema_count": 0,
                "queue_position_at_arrival": ticket.queue_position_at_arrival,
                "validation": asdict(result),
                "escalation": escalation,
            }
        )
        if not result.passed:
            raise ServiceTierError(
                "response_validation_failed",
                {"gate_id": result.gate_id, "reason": result.reason},
            )
        return {
            "status": "SUCCESS",
            "request_id": request_id,
            "session_id": effective_session_id,
            "trace_id": trace_id,
            "capability_tier": capability.tier.value,
            "requested_lane": lane.value,
            "effective_lane": route.lane.value,
            "model_id": route.model_id,
            "queue_position": ticket.queue_position_at_arrival,
            "queue_wait_seconds": ticket.queue_wait_seconds,
            "priority": route.priority,
            "context_limit": route.context_limit,
            "output_limit": route.output_limit,
            "escalation": escalation,
            "response": response,
        }

    def create_agent_session(
        self,
        *,
        user_id: str,
        repo_id: str,
        session_id: str,
        tool_policy: AgentToolPolicy,
        task_title: str = "New Agent task",
        instruction_digest: str = "",
        lane: str | None = None,
        sanitized_model_name: str | None = None,
        idempotency_key: str | None = None,
    ) -> JsonObject:
        try:
            capability = self.users.require_capability(user_id, Capability.AGENT)
            binding = self.sandboxes.create(
                user_id=user_id,
                repo_id=repo_id,
                session_id=session_id,
                tool_policy=tool_policy,
                task_title=task_title,
                instruction_digest=instruction_digest,
                lane=lane,
                sanitized_model_name=sanitized_model_name,
                idempotency_key=idempotency_key,
            )
        except (
            AgentSandboxError,
            RepositoryAuthorizationError,
            UserCapabilityError,
        ) as exc:
            self._audit_denial(
                user_id=user_id,
                mode="agent",
                failure_category=str(getattr(exc, "category", str(exc))),
                session_id=session_id,
                repo_id=repo_id,
            )
            raise
        with self._session_lock:
            self._cancelled_sessions.pop(session_id, None)
            self._session_results.pop(session_id, None)
        self.audit_log.append(
            {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "outcome": "BOUND",
                "action": "AGENT_SESSION_BOUND",
                "user_id": user_id,
                "capability_tier": capability.tier.value,
                "session_id": session_id,
                "mode": "agent",
                "repo_id": repo_id,
                "sandbox_id": session_id,
                "worktree_root": binding.worktree_root,
                "base_revision": binding.base_revision,
                "grant_revision": binding.grant_revision,
                "network_enabled": False,
                "tool_policy_id": tool_policy.policy_id,
                "network_policy_id": "network-denied-v1",
                "requested_quality_lane": None,
                "effective_quality_lane": None,
                "route": None,
                "queue_wait_seconds": 0.0,
                "trace_id": f"trace-{uuid.uuid4().hex}",
            }
        )
        return {"status": "BOUND", "binding": binding.to_json()}

    def agent(
        self,
        *,
        user_id: str,
        session_id: str,
        lane: ModelLane,
        instruction: str,
        domain: str,
    ) -> JsonObject:
        self.domain_registry.require(domain, surface="agent")
        capability = self.users.require_capability(user_id, Capability.AGENT)
        if not instruction.strip() or len(instruction) > 100_000:
            raise ServiceTierError("invalid_agent_instruction")
        with self._session_lock:
            if session_id in self._cancelled_sessions:
                raise ServiceTierError("agent_session_cancelled")
        try:
            binding = self.sandboxes.require_active(session_id, user_id=user_id)
        except AgentSandboxError as exc:
            self._audit_denial(
                user_id=user_id,
                mode="agent",
                failure_category=exc.category,
                session_id=session_id,
                requested_lane=lane,
            )
            raise
        route = self._route(lane, domain=domain)
        self.sandboxes.start_task(
            session_id,
            user_id=user_id,
            lane=lane.value,
            sanitized_model_name=route.model_id,
            instruction_digest=hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        )
        request_id = f"agent-{uuid.uuid4().hex}"
        trace_id = f"trace-{uuid.uuid4().hex}"
        with self.scheduler.admit(lane) as ticket:
            response = self.agent_backend.run(
                binding=binding,
                instruction=instruction,
                route=route,
                request_id=request_id,
            )
            validation = self.validator.validate(response, domain=domain)
            escalation: JsonObject | None = None
            if not validation.passed and lane is not ModelLane.QUALITY:
                quality = self.lane_policy.routes[ModelLane.QUALITY]
                response = self.agent_backend.run(
                    binding=binding,
                    instruction=instruction,
                    route=quality,
                    request_id=request_id,
                )
                second = self.validator.validate(response, domain=domain)
                escalation = {
                    "from_lane": lane.value,
                    "to_lane": ModelLane.QUALITY.value,
                    "trigger_gate": validation.gate_id,
                    "passed": second.passed,
                }
                validation = second
                route = quality
        self.audit_log.append(
            {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "request_id": request_id,
                "request_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
                "user_id": user_id,
                "capability_tier": capability.tier.value,
                "session_id": session_id,
                "mode": "agent",
                "repo_id": binding.repo_id,
                "sandbox_id": session_id,
                "worktree_root": binding.worktree_root,
                "requested_quality_lane": lane.value,
                "effective_quality_lane": route.lane.value,
                "route": route.model_id,
                "tool_policy_id": binding.tool_policy.policy_id,
                "network_policy_id": "network-denied-v1",
                "queue_wait_seconds": ticket.queue_wait_seconds,
                "trace_id": trace_id,
                "requested_tier": lane.value,
                "effective_tier": route.lane.value,
                "priority": route.priority,
                "context_limit": route.context_limit,
                "output_limit": route.output_limit,
                "escalated": escalation is not None,
                "escalation_reason": (
                    escalation.get("trigger_gate") if escalation is not None else None
                ),
                "validator_results": [asdict(validation)],
                "requested_lane": lane.value,
                "effective_model_id": route.model_id,
                "queue_position_at_arrival": ticket.queue_position_at_arrival,
                "validation": asdict(validation),
                "escalation": escalation,
            }
        )
        if not validation.passed:
            self.sandboxes.record_result(
                session_id,
                user_id=user_id,
                command_count=0,
                verification_summary=f"FAILED:{validation.gate_id}",
                failed=True,
            )
            raise ServiceTierError(
                "agent_validation_failed",
                {"gate_id": validation.gate_id, "reason": validation.reason},
            )
        result: JsonObject = {
            "status": "SUCCESS",
            "request_id": request_id,
            "trace_id": trace_id,
            "session_id": session_id,
            "repo_id": binding.repo_id,
            "effective_lane": route.lane.value,
            "model_id": route.model_id,
            "queue_position": ticket.queue_position_at_arrival,
            "queue_wait_seconds": ticket.queue_wait_seconds,
            "priority": route.priority,
            "context_limit": route.context_limit,
            "output_limit": route.output_limit,
            "escalation": escalation,
            "response": response,
        }
        with self._session_lock:
            self._session_results[session_id] = result
        command_count = 0
        raw_command_count = response.get("command_count")
        if isinstance(raw_command_count, int) and raw_command_count >= 0:
            command_count = raw_command_count
        self.sandboxes.record_result(
            session_id,
            user_id=user_id,
            command_count=command_count,
            verification_summary=(
                f"PASSED:{validation.gate_id}"
                if validation.passed
                else f"FAILED:{validation.gate_id}"
            ),
        )
        return result

    def agent_session_status(self, *, user_id: str, session_id: str) -> JsonObject:
        """Return only the authenticated user's binding and last result."""

        self.users.require_capability(user_id, Capability.AGENT)
        with self._session_lock:
            cancelled_repo = self._cancelled_sessions.get(session_id)
            result = self._session_results.get(session_id)
        if cancelled_repo is not None:
            return {
                "status": "CANCELLED",
                "session_id": session_id,
                "repo_id": cancelled_repo,
                "sandbox_id": session_id,
                "worktree_status": {"status": "RELEASED_OR_PRESERVED_AFTER_CANCEL"},
                "last_result": result,
            }
        binding_status = self.sandboxes.status(session_id, user_id=user_id)
        return {
            "status": binding_status["status"],
            "session_id": session_id,
            "repo_id": binding_status["repo_id"],
            "sandbox_id": session_id,
            "worktree_status": binding_status,
            "last_result": result,
        }

    def cancel_agent_session(self, *, user_id: str, session_id: str) -> JsonObject:
        """Cancel future work and release only a clean owned worktree."""

        self.users.require_capability(user_id, Capability.AGENT)
        binding = self.sandboxes.require_active(session_id, user_id=user_id)
        result = self.sandboxes.cancel(session_id, user_id=user_id)
        with self._session_lock:
            self._cancelled_sessions[session_id] = binding.repo_id
            last_result = self._session_results.get(session_id)
        return {
            **result,
            "cancelled": True,
            "last_result": last_result,
        }
