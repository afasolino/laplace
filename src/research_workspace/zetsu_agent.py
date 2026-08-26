"""Bounded Qwen repository-agent orchestration for Zetsu.

Request identity and execution state are explicit per run. Model summaries are
semantic aids only; exact resumable state is checkpointed independently.
"""

from __future__ import annotations

import difflib
import hashlib
import io
import json
import os
import shutil
import signal
import subprocess  # nosec B404 - argv is allowlisted and shell=False
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, Mapping, Sequence, cast

from .agent_sandbox import (
    AgentSandboxError,
    AgentSandboxManager,
    AgentSessionBinding,
    AgentToolPolicy,
)
from .bounded_aci import BoundedACIError, BoundedRepositoryACI
from .context_planner import DEFAULT_COMPACTION_RATIO, ContextPlanner, ContextPlannerError
from .personal_corpus import CorpusError, PersonalCorpusStore
from .repository_authorization import RepositoryAuthorizationError, validate_workspace_path
from .service_tiers import ModelLane, ServiceTierError, TieredServingService
from .verification_policy import validate_verification_argv
from .zetsu_context import compact_personal_results
from .result_store import ResultStore, ResultStoreError
from .zetsu_scheduler import (
    AgentAdmission,
    AgentSchedulerError,
    AgentTaskScheduler,
    capacity_policy,
)

JsonObject = dict[str, object]

_MAX_READ_CHARS = 64_000
_MAX_NEW_FILE_CHARS = 256_000
_MAX_SEARCH_MATCHES = 80
_MAX_OBSERVATION_CHARS = 80_000
_MAX_RECENT_OBSERVATIONS = 8
_MAX_CHANGED_PATHS = 256
_MAX_GIT_STATUS_BYTES = 4 * 1024 * 1024
_MAX_CHECKPOINT_BYTES = 2 * 1024 * 1024
_MAX_VERIFY_CAPTURE_BYTES = 32 * 1024
_CHECKPOINT_SCHEMA = 3
_MAX_OUTPUT_CAP_CONTINUATIONS = 4


def _rough_tokens(value: str) -> int:
    """Approximate tokens only when the serving abstraction exposes no tokenizer."""

    return max(1, (len(value.encode("utf-8")) + 3) // 4)


def _usage_tokens(value: Mapping[str, object]) -> tuple[int | None, int | None, int | None]:
    response = value.get("response")
    usage = response.get("usage") if isinstance(response, Mapping) else None
    if not isinstance(usage, Mapping):
        return None, None, None

    def read(*names: str) -> int | None:
        for name in names:
            raw = usage.get(name)
            if isinstance(raw, int) and raw >= 0:
                return raw
        return None

    return (
        read("prompt_tokens", "input_tokens"),
        read("completion_tokens", "output_tokens"),
        read("cached_tokens", "cached_input_tokens"),
    )


@dataclass
class AgentTelemetry:
    qwen_input_tokens: int = 0
    qwen_output_tokens: int = 0
    qwen_cached_tokens: int = 0
    qwen_usage_reported_calls: int = 0
    qwen_calls: int = 0
    agent_steps: int = 0
    tool_calls: int = 0
    verification_calls: int = 0
    compactions: int = 0
    compaction_input_tokens: int = 0
    compaction_output_tokens: int = 0
    approximate_active_context_tokens_before_last_compaction: int = 0
    approximate_active_context_tokens_after_last_compaction: int = 0
    last_model_reported_input_tokens: int | None = None

    def add_usage(self, value: Mapping[str, object], *, compaction: bool = False) -> None:
        prompt, completion, cached = _usage_tokens(value)
        self.qwen_calls += 1
        if prompt is not None or completion is not None or cached is not None:
            self.qwen_usage_reported_calls += 1
        if prompt is not None:
            self.qwen_input_tokens += prompt
            self.last_model_reported_input_tokens = prompt
        if completion is not None:
            self.qwen_output_tokens += completion
        if cached is not None:
            self.qwen_cached_tokens += cached
        if compaction:
            self.compaction_input_tokens += prompt or 0
            self.compaction_output_tokens += completion or 0

    @classmethod
    def from_mapping(cls, value: object) -> "AgentTelemetry":
        if not isinstance(value, Mapping):
            raise ServiceTierError("zetsu_agent_checkpoint_telemetry_invalid")
        telemetry = cls()
        integer_fields = set(telemetry.__dataclass_fields__) - {"last_model_reported_input_tokens"}
        allowed = integer_fields | {
            "last_model_reported_input_tokens",
            "qwen_token_usage_source",
            "qwen_token_usage_complete",
            "approximate_context_token_method",
        }
        if set(value) - allowed:
            raise ServiceTierError("zetsu_agent_checkpoint_telemetry_invalid")
        for name in integer_fields:
            raw = value.get(name)
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                raise ServiceTierError("zetsu_agent_checkpoint_telemetry_invalid")
            setattr(telemetry, name, raw)
        raw_last = value.get("last_model_reported_input_tokens")
        if raw_last is not None and (
            isinstance(raw_last, bool) or not isinstance(raw_last, int) or raw_last < 0
        ):
            raise ServiceTierError("zetsu_agent_checkpoint_telemetry_invalid")
        telemetry.last_model_reported_input_tokens = raw_last
        if value.get("qwen_token_usage_source") != "model_reported_per_request":
            raise ServiceTierError("zetsu_agent_checkpoint_telemetry_invalid")
        if value.get("approximate_context_token_method") != "utf8_json_bytes_div4":
            raise ServiceTierError("zetsu_agent_checkpoint_telemetry_invalid")
        complete = value.get("qwen_token_usage_complete")
        if not isinstance(complete, bool) or complete != (
            telemetry.qwen_usage_reported_calls == telemetry.qwen_calls
        ):
            raise ServiceTierError("zetsu_agent_checkpoint_telemetry_invalid")
        if (
            telemetry.qwen_usage_reported_calls > telemetry.qwen_calls
            or telemetry.verification_calls > telemetry.tool_calls
            or telemetry.compactions > telemetry.qwen_calls
            or telemetry.qwen_calls != telemetry.agent_steps + telemetry.compactions
        ):
            raise ServiceTierError("zetsu_agent_checkpoint_telemetry_invalid")
        return telemetry

    def as_dict(self) -> JsonObject:
        return {
            "qwen_input_tokens": self.qwen_input_tokens,
            "qwen_output_tokens": self.qwen_output_tokens,
            "qwen_cached_tokens": self.qwen_cached_tokens,
            "qwen_usage_reported_calls": self.qwen_usage_reported_calls,
            "qwen_calls": self.qwen_calls,
            "qwen_token_usage_source": "model_reported_per_request",
            "qwen_token_usage_complete": self.qwen_usage_reported_calls == self.qwen_calls,
            "approximate_context_token_method": "utf8_json_bytes_div4",
            "agent_steps": self.agent_steps,
            "tool_calls": self.tool_calls,
            "verification_calls": self.verification_calls,
            "compactions": self.compactions,
            "compaction_input_tokens": self.compaction_input_tokens,
            "compaction_output_tokens": self.compaction_output_tokens,
            "approximate_active_context_tokens_before_last_compaction": self.approximate_active_context_tokens_before_last_compaction,
            "approximate_active_context_tokens_after_last_compaction": self.approximate_active_context_tokens_after_last_compaction,
            "last_model_reported_input_tokens": self.last_model_reported_input_tokens,
        }


@dataclass
class AgentExecutionState:
    objective: str
    step: int = 0
    summary: str = ""
    recent_observations: list[str] = field(default_factory=list)
    changed_paths: list[str] = field(default_factory=list)
    worktree_head: str = ""
    worktree_status_sha256: str = ""
    lane: str = ""
    model_id: str = ""
    context_limit: int = 0
    required_verification_argv: list[str] | None = None
    validation_history: list[JsonObject] = field(default_factory=list)
    unresolved_failures: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    next_state: str = "choose_action"
    mutation_epoch: int = 0
    last_verified_epoch: int = -1
    command_count: int = 0
    consumed_wall_seconds: float = 0.0
    target_initial_head: str = ""
    target_initial_status_sha256: str = ""
    target_applied_status_sha256: str = ""
    applied_patch_sha256: str = ""
    output_cap_continuations: int = 0
    telemetry: AgentTelemetry = field(default_factory=AgentTelemetry)


@dataclass(frozen=True)
class AgentRunContext:
    user_id: str
    session_id: str
    repo_id: str
    lane: ModelLane
    binding: AgentSessionBinding
    worktree: Path
    max_steps: int
    max_chars: int
    compaction_ratio: float
    model_id: str
    context_limit: int
    required_verification_argv: tuple[str, ...] | None
    run_started: float
    remaining_wall_seconds: float
    apply_to_repository: bool = False


class AgentCheckpointStore:
    """Atomic owner-independent files keyed by opaque session digest."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            os.chmod(self.root, 0o700)

    def path(self, session_id: str) -> Path:
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"

    def write(self, session_id: str, value: Mapping[str, object]) -> Path:
        target = self.path(session_id)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(dict(value), handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def read(self, session_id: str) -> JsonObject | None:
        target = self.path(session_id)
        if not target.is_file():
            return None
        try:
            size = target.stat().st_size
            if size > _MAX_CHECKPOINT_BYTES:
                raise ServiceTierError("zetsu_agent_checkpoint_too_large")
            value: object = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ServiceTierError("zetsu_agent_checkpoint_invalid") from exc
        if not isinstance(value, dict):
            raise ServiceTierError("zetsu_agent_checkpoint_invalid")
        return cast(JsonObject, value)


class ZetsuAgentCoordinator:
    """Iterative Qwen agent confined to one owner-authorized worktree."""

    def __init__(
        self,
        tiered: TieredServingService,
        corpus: PersonalCorpusStore | None = None,
        *,
        checkpoint_store: AgentCheckpointStore | None = None,
        result_store: ResultStore | None = None,
        scheduler: AgentTaskScheduler | None = None,
    ) -> None:
        self.tiered = tiered
        self.corpus = corpus
        self.context_planner = ContextPlanner()
        self.checkpoints = checkpoint_store or AgentCheckpointStore(
            tiered.sandboxes.sandbox_root / "zetsu_agent_checkpoints"
        )
        self.results = result_store or ResultStore(
            tiered.sandboxes.sandbox_root / "zetsu_agent_results"
        )
        scheduler_capable = all(
            hasattr(tiered.sandboxes, name)
            for name in ("reconcile", "capacity_snapshot", "per_user_quota")
        )
        self.scheduler = scheduler or (
            AgentTaskScheduler(
                tiered.sandboxes.sandbox_root / "zetsu_agent_scheduler.sqlite3",
                tiered.sandboxes,
                capacity_policy(codev_enabled=tiered.lane_policy.codev_enabled),
            )
            if scheduler_capable
            else None
        )
        self._session_locks_guard = threading.Lock()
        self._session_locks: dict[str, threading.Lock] = {}
        self._target_locks_guard = threading.Lock()
        self._target_locks: dict[str, threading.Lock] = {}

    def _session_lock(self, session_id: str) -> threading.Lock:
        with self._session_locks_guard:
            return self._session_locks.setdefault(session_id, threading.Lock())

    def _target_lock(self, repository: Path) -> threading.Lock:
        key = str(repository.resolve())
        with self._target_locks_guard:
            return self._target_locks.setdefault(key, threading.Lock())

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are the bounded local Qwen repository agent subordinate to Codex. Return exactly "
            "one JSON object selecting one action. Allowed actions: "
            '{"action":"repo_map","query":"literal","token_budget":1000}; '
            '{"action":"find_symbol","name":"Symbol"}; '
            '{"action":"find_references","name":"Symbol"}; '
            '{"action":"search_text","query":"literal","glob":"*.py"}; '
            '{"action":"read_region","path":"relative/a.py","start_line":1,"end_line":40}; '
            '{"action":"inspect_diff","paths":["relative/a.py"]}; '
            '{"action":"edit_region","path":"relative/a.py","old_text":"exact once",'
            '"new_text":"replacement"}; '
            '{"action":"create_text_file","path":"relative/a.py","content":"text"}; '
            '{"action":"git_state"}; '
            '{"action":"search","query":"literal","glob":"*.py"}; '
            '{"action":"read","paths":["relative/a","relative/b"]}; '
            '{"action":"retrieve","query":"owner-authorized knowledge query"}; '
            '{"action":"edit","edits":[{"path":"relative/path","old_text":"exact once",'
            '"new_text":"replacement"}]}; '
            '{"action":"create","path":"relative/path","content":"text"}; '
            '{"action":"verify","argv":["pytest","tests/test_x.py","-q"]}; '
            '{"action":"finish","result":"concise verified result"}. '
            "Batch related reads and known edits. Inspect only enough to edit once, run the exact "
            "caller verifier, and stop immediately when it passes with no unresolved failure. "
            "Use small exact edits for large files; never emit an entire large file when bounded "
            "read/edit operations can complete it safely. "
            "Do not narrate. "
            "Use only the supplied worktree and compact retrieval interface. Never request shell, "
            "network, Git mutation, .git access, unrestricted corpus access, or paths outside the worktree. "
            "After any mutation, deterministic verification must pass after the latest mutation before finish."
        )

    @staticmethod
    def _json_content(result: Mapping[str, object]) -> JsonObject:
        response = result.get("response")
        content = response.get("content") if isinstance(response, Mapping) else None
        if not isinstance(content, str):
            raise ServiceTierError("zetsu_agent_action_missing")
        try:
            parsed: object = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ServiceTierError("zetsu_agent_action_not_json") from exc
        if not isinstance(parsed, dict) or not isinstance(parsed.get("action"), str):
            raise ServiceTierError("zetsu_agent_action_invalid")
        return cast(JsonObject, parsed)

    @staticmethod
    def _summary_content(result: Mapping[str, object]) -> str:
        response = result.get("response")
        content = response.get("content") if isinstance(response, Mapping) else None
        if not isinstance(content, str):
            raise ServiceTierError("zetsu_agent_compaction_missing")
        try:
            parsed: object = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ServiceTierError("zetsu_agent_compaction_not_json") from exc
        if not isinstance(parsed, dict):
            raise ServiceTierError("zetsu_agent_compaction_invalid")
        summary = parsed.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise ServiceTierError("zetsu_agent_compaction_invalid")
        return summary.strip()

    @staticmethod
    def _worktree(binding_root: str) -> Path:
        root = Path(binding_root).resolve()
        if not root.is_dir():
            raise ServiceTierError("zetsu_agent_worktree_unavailable")
        return root

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

    @staticmethod
    def _run_git(
        worktree: Path, argv: Sequence[str], *, timeout: float = 30.0
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(worktree), *argv],
            capture_output=True,
            text=True,
            check=False,
            timeout=max(1.0, timeout),
        )

    @classmethod
    def _worktree_state(cls, worktree: Path) -> tuple[str, str, list[str]]:
        head = cls._run_git(worktree, ["rev-parse", "--verify", "HEAD"])
        status = cls._run_git(worktree, ["status", "--porcelain=v1", "--untracked-files=all"])
        if head.returncode != 0 or status.returncode != 0:
            raise ServiceTierError("zetsu_agent_git_state_unavailable")
        status_bytes = status.stdout.encode("utf-8")
        if len(status_bytes) > _MAX_GIT_STATUS_BYTES:
            raise ServiceTierError("zetsu_agent_git_status_too_large")
        changed: list[str] = []
        for line in status.stdout.splitlines():
            if len(line) < 4:
                continue
            path = line[3:]
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            changed.append(path.strip('"'))
        unique_changed = sorted(dict.fromkeys(changed))
        return (
            head.stdout.strip(),
            hashlib.sha256(status_bytes).hexdigest(),
            unique_changed[:_MAX_CHANGED_PATHS],
        )

    def _raise_materialization_failure_if_needed(
        self, ctx: AgentRunContext, relative_path: str
    ) -> None:
        """Differentiate caller-only dirty files from an ordinary missing target."""

        canonical = Path(ctx.binding.canonical_repository_root).resolve(strict=True)
        try:
            target = validate_workspace_path(canonical, relative_path)
        except RepositoryAuthorizationError:
            return
        head = self._run_git(canonical, ["rev-parse", "--verify", "HEAD"])
        status = self._run_git(
            canonical, ["status", "--porcelain=v1", "--untracked-files=all"]
        )
        if head.returncode != 0 or status.returncode != 0:
            return
        current_head = head.stdout.strip()
        working_tree_clean = not bool(status.stdout.strip())
        if target.is_file() and (
            not working_tree_clean or current_head != ctx.binding.base_revision
        ):
            category = (
                "repository_state_not_materialized"
                if not working_tree_clean
                else "repository_revision_not_materialized"
            )
            raise ServiceTierError(
                category,
                {
                    "canonical_root": str(canonical),
                    "current_head": current_head,
                    "granted_revision": ctx.binding.base_revision,
                    "working_tree_clean": working_tree_clean,
                    "path": relative_path,
                },
            )

    def _exact_patch(self, worktree: Path, changed_paths: Sequence[str]) -> str:
        tracked = self._run_git(
            worktree,
            ["diff", "--no-ext-diff", "--binary", "HEAD", "--"],
        )
        if tracked.returncode != 0:
            raise ServiceTierError("zetsu_agent_handoff_patch_unavailable")
        patch = tracked.stdout
        for relative in changed_paths:
            listed = self._run_git(
                worktree,
                ["ls-files", "--error-unmatch", "--", relative],
            )
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

    def _handoff_patch(
        self,
        worktree: Path,
        session_id: str,
        changed_paths: Sequence[str],
        *,
        max_chars: int,
    ) -> JsonObject:
        """Capture exact verified code separately from disposable agent narration."""

        del max_chars
        patch = self._exact_patch(worktree, changed_paths)
        digest = hashlib.sha256(patch.encode("utf-8")).hexdigest()
        artifact = self.checkpoints.path(session_id).with_suffix(".patch")
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

    def handoff_evidence(self, session_id: str, *, max_chars: int) -> JsonObject:
        """Expand a persisted exact handoff only when an authorized caller asks."""

        artifact = self.checkpoints.path(session_id).with_suffix(".patch")
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

    def _apply_verified_handoff(
        self,
        ctx: AgentRunContext,
        state: AgentExecutionState,
        handoff: Mapping[str, object],
    ) -> JsonObject:
        """Promote one verified patch into its clean, revision-bound repository."""

        if not ctx.apply_to_repository:
            return {"requested": False, "applied": False}
        if not self._finish_allowed(state) or state.mutation_epoch < 1 or not state.changed_paths:
            raise ServiceTierError("zetsu_agent_apply_without_verified_mutation")
        patch_path_value = handoff.get("patch_path")
        patch_sha = handoff.get("patch_sha256")
        if not isinstance(patch_path_value, str) or not isinstance(patch_sha, str):
            raise ServiceTierError("zetsu_agent_handoff_patch_unavailable")
        try:
            patch_path = Path(patch_path_value).resolve(strict=True)
        except OSError as exc:
            raise ServiceTierError("zetsu_agent_handoff_patch_unavailable") from exc
        if patch_path != self.checkpoints.path(ctx.session_id).with_suffix(".patch"):
            raise ServiceTierError("zetsu_agent_handoff_patch_identity_mismatch")
        try:
            observed_patch_sha = hashlib.sha256(patch_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ServiceTierError("zetsu_agent_handoff_patch_unavailable") from exc
        if observed_patch_sha != patch_sha:
            raise ServiceTierError("zetsu_agent_handoff_patch_identity_mismatch")

        try:
            target = Path(ctx.binding.canonical_repository_root).resolve(strict=True)
        except OSError as exc:
            raise ServiceTierError("zetsu_agent_apply_target_unavailable") from exc
        if target == ctx.worktree:
            raise ServiceTierError("zetsu_agent_apply_target_not_isolated")
        for relative in state.changed_paths:
            self._relative_target(target, relative)
        with self._target_lock(target):
            head, status_sha, changed = self._worktree_state(target)
            if state.applied_patch_sha256:
                if (
                    state.applied_patch_sha256 == patch_sha
                    and head == state.target_initial_head
                    and status_sha == state.target_applied_status_sha256
                    and changed == state.changed_paths
                ):
                    return {
                        "requested": True,
                        "applied": True,
                        "already_applied": True,
                        "target_status_sha256": status_sha,
                    }
                raise ServiceTierError("zetsu_agent_apply_target_drift")
            if head == state.target_initial_head and changed:
                target_patch_sha = hashlib.sha256(
                    self._exact_patch(target, changed).encode("utf-8")
                ).hexdigest()
                if changed == state.changed_paths and target_patch_sha == patch_sha:
                    state.applied_patch_sha256 = patch_sha
                    state.target_applied_status_sha256 = status_sha
                    return {
                        "requested": True,
                        "applied": True,
                        "already_applied": True,
                        "recovered_after_checkpoint_gap": True,
                        "target_status_sha256": status_sha,
                    }
                raise ServiceTierError("zetsu_agent_apply_target_drift")
            if (
                head != state.target_initial_head
                or status_sha != state.target_initial_status_sha256
                or changed
            ):
                raise ServiceTierError("zetsu_agent_apply_target_drift")
            checked = self._run_git(
                target,
                ["apply", "--check", "--whitespace=error-all", str(patch_path)],
            )
            if checked.returncode != 0:
                raise ServiceTierError(
                    "zetsu_agent_apply_check_failed", {"stderr": checked.stderr[-2_000:]}
                )
            applied = self._run_git(
                target,
                ["apply", "--whitespace=error-all", str(patch_path)],
            )
            if applied.returncode != 0:
                raise ServiceTierError(
                    "zetsu_agent_apply_failed", {"stderr": applied.stderr[-2_000:]}
                )
            after_head, after_status, after_changed = self._worktree_state(target)
            diff_check = self._run_git(target, ["diff", "--check"])
            if (
                after_head != state.target_initial_head
                or after_changed != state.changed_paths
                or diff_check.returncode != 0
            ):
                rolled_back = self._run_git(target, ["apply", "--reverse", str(patch_path)])
                if rolled_back.returncode != 0:
                    raise ServiceTierError("zetsu_agent_apply_rollback_failed")
                raise ServiceTierError("zetsu_agent_apply_postcondition_failed")
            state.applied_patch_sha256 = patch_sha
            state.target_applied_status_sha256 = after_status
            return {
                "requested": True,
                "applied": True,
                "already_applied": False,
                "target_status_sha256": after_status,
            }

    @staticmethod
    def _check_text(value: str, *, maximum: int) -> None:
        if len(value) > maximum or "\x00" in value:
            raise ServiceTierError("zetsu_agent_text_invalid")

    def _search(
        self, ctx: AgentRunContext, state: AgentExecutionState, action: Mapping[str, object]
    ) -> str:
        worktree = ctx.worktree
        self._ensure_active(ctx, state)
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
                self._ensure_active(ctx, state)
            if len(matches) >= _MAX_SEARCH_MATCHES:
                break
            relative = path.relative_to(worktree).as_posix()
            if ".git" in path.relative_to(worktree).parts:
                continue
            try:
                safe_path = self._relative_target(worktree, relative)
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

    def _read(self, ctx: AgentRunContext | Path, action: Mapping[str, object]) -> str:
        # Preserve the path-based helper contract used by older deterministic
        # tests; materialization checks require the full run context.
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
            target = self._relative_target(worktree, value)
            if not target.is_file() or target.stat().st_size > 2_000_000:
                if (
                    not target.exists()
                    and isinstance(value, str)
                    and isinstance(ctx, AgentRunContext)
                ):
                    self._raise_materialization_failure_if_needed(ctx, value)
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

    def _retrieve(
        self, ctx: AgentRunContext, state: AgentExecutionState, action: Mapping[str, object]
    ) -> str:
        if self.corpus is None:
            raise ServiceTierError("zetsu_agent_retrieval_unavailable")
        self._ensure_active(ctx, state)
        query = action.get("query")
        if not isinstance(query, str) or not query.strip() or len(query) > 4_000:
            raise ServiceTierError("zetsu_agent_retrieval_query_invalid")
        raw = self.corpus.search(ctx.user_id, query.strip(), limit=8)
        self._ensure_active(ctx, state)
        packet = compact_personal_results(query.strip(), raw, max_results=6, max_chars=8_000)
        evidence = packet.get("evidence")
        if isinstance(evidence, list):
            for item in evidence:
                if isinstance(item, Mapping):
                    chunk_id = item.get("chunk_id")
                    if isinstance(chunk_id, str) and chunk_id not in state.evidence_refs:
                        state.evidence_refs.append(chunk_id)
        return json.dumps(packet, sort_keys=True, ensure_ascii=False)

    def _typed_aci(self, ctx: AgentRunContext, state: AgentExecutionState) -> BoundedRepositoryACI:
        """Construct the structured ACI while retaining coordinator ownership checks."""

        return BoundedRepositoryACI(
            ctx.worktree,
            owner_user_id=ctx.user_id,
            session_id=ctx.session_id,
            allow_mutation="apply_patch" in ctx.binding.tool_policy.allowed_tools,
            required_verification_argv=ctx.required_verification_argv,
            is_cancelled=lambda: self.tiered.agent_session_status(
                user_id=ctx.user_id, session_id=ctx.session_id
            ).get("status")
            == "CANCELLED",
        )

    @staticmethod
    def _require_edit_policy(policy: AgentToolPolicy) -> None:
        if "apply_patch" not in policy.allowed_tools:
            raise ServiceTierError("zetsu_agent_edit_not_allowed")

    def _edit(self, ctx: AgentRunContext, action: Mapping[str, object]) -> str:
        self._require_edit_policy(ctx.binding.tool_policy)
        edits = action.get("edits")
        if edits is None:
            edits = [action]
        if (
            not isinstance(edits, Sequence)
            or isinstance(edits, (str, bytes))
            or not 1 <= len(edits) <= 16
            or not all(isinstance(item, Mapping) for item in edits)
        ):
            raise ServiceTierError("zetsu_agent_edit_invalid")
        originals: dict[Path, str] = {}
        updated: dict[Path, str] = {}
        for item in edits:
            target = self._relative_target(ctx.worktree, item.get("path"))
            old = item.get("old_text")
            new = item.get("new_text")
            if not isinstance(old, str) or not old or not isinstance(new, str):
                raise ServiceTierError("zetsu_agent_edit_invalid")
            self._check_text(old, maximum=_MAX_NEW_FILE_CHARS)
            self._check_text(new, maximum=_MAX_NEW_FILE_CHARS)
            if not target.is_file():
                self._raise_materialization_failure_if_needed(
                    ctx, target.relative_to(ctx.worktree).as_posix()
                )
                raise ServiceTierError("zetsu_agent_edit_target_unavailable")
            if target not in originals:
                try:
                    originals[target] = target.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError) as exc:
                    raise ServiceTierError("zetsu_agent_edit_target_not_text") from exc
            current = updated.get(target, originals[target])
            if current.count(old) != 1:
                raise ServiceTierError("zetsu_agent_edit_anchor_not_unique")
            replacement = current.replace(old, new, 1)
            self._check_text(replacement, maximum=_MAX_NEW_FILE_CHARS)
            updated[target] = replacement
        try:
            for target, replacement in updated.items():
                target.write_text(replacement, encoding="utf-8")
            check = self._run_git(ctx.worktree, ["diff", "--check"])
            if check.returncode != 0:
                raise ServiceTierError(
                    "zetsu_agent_edit_diff_check_failed", {"stderr": check.stderr[-2_000:]}
                )
        except (OSError, ServiceTierError):
            for target, original in originals.items():
                target.write_text(original, encoding="utf-8")
            raise
        paths = ",".join(sorted(path.relative_to(ctx.worktree).as_posix() for path in updated))
        return f"EDITED:{paths}:replacements={len(edits)}"

    def _create(self, ctx: AgentRunContext, action: Mapping[str, object]) -> str:
        self._require_edit_policy(ctx.binding.tool_policy)
        target = self._relative_target(ctx.worktree, action.get("path"))
        content = action.get("content")
        if not isinstance(content, str):
            raise ServiceTierError("zetsu_agent_create_invalid")
        self._check_text(content, maximum=_MAX_NEW_FILE_CHARS)
        if target.exists() or not target.parent.is_dir():
            if not target.exists():
                self._raise_materialization_failure_if_needed(
                    ctx, target.relative_to(ctx.worktree).as_posix()
                )
            raise ServiceTierError("zetsu_agent_create_target_invalid")
        try:
            with target.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
        except OSError as exc:
            raise ServiceTierError("zetsu_agent_create_failed") from exc
        return f"CREATED:{target.relative_to(ctx.worktree).as_posix()}"

    @classmethod
    def _verify_argv(cls, worktree: Path, value: object) -> list[str]:
        return validate_verification_argv(worktree, value)

    @staticmethod
    def _verification_qualifies(
        argv: Sequence[str], required_verification_argv: Sequence[str] | None
    ) -> bool:
        if required_verification_argv is None or list(argv) != list(required_verification_argv):
            return False
        executable = Path(argv[0]).name
        lowered = {item.casefold() for item in argv[1:]}
        if executable == "pytest":
            return not bool(lowered & {"--collect-only", "--co", "--help", "-h", "--version"})
        return False

    @staticmethod
    def _mutation_marker(epoch: int) -> str:
        return f"latest_mutation_unverified:epoch={epoch}"

    @staticmethod
    def _verification_command_id(argv: Sequence[str]) -> str:
        encoded = json.dumps(list(argv), separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    @staticmethod
    def _remove_resolved_verification_failures(state: AgentExecutionState, command_id: str) -> None:
        marker = f":cmd={command_id}:"
        state.unresolved_failures = [
            item for item in state.unresolved_failures if marker not in item
        ]

    def _remaining_wall(self, ctx: AgentRunContext, state: AgentExecutionState) -> float:
        elapsed_this_run = time.monotonic() - ctx.run_started
        remaining = ctx.remaining_wall_seconds - elapsed_this_run
        if remaining <= 0:
            raise ServiceTierError("zetsu_agent_wall_budget_exhausted")
        return remaining

    def _ensure_active(self, ctx: AgentRunContext, state: AgentExecutionState) -> None:
        self._remaining_wall(ctx, state)
        status = self.tiered.agent_session_status(user_id=ctx.user_id, session_id=ctx.session_id)
        if status.get("status") == "CANCELLED":
            raise ServiceTierError("agent_session_cancelled")
        self.tiered.sandboxes.require_active(ctx.session_id, user_id=ctx.user_id)
        heartbeat = getattr(self.tiered.sandboxes, "heartbeat_task", None)
        if callable(heartbeat):
            heartbeat(ctx.session_id, user_id=ctx.user_id)

    def _consume_tool_budget(self, ctx: AgentRunContext, state: AgentExecutionState) -> None:
        if state.command_count >= ctx.binding.tool_policy.max_commands:
            raise ServiceTierError("zetsu_agent_command_budget_exhausted")
        state.command_count += 1
        state.telemetry.tool_calls += 1

    @staticmethod
    def _stop_process_tree(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except (OSError, ProcessLookupError):
            pass
        try:
            process.wait(timeout=2.0)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except (OSError, ProcessLookupError):
            pass
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass

    @staticmethod
    def _read_file_tail(file_handle: BinaryIO, *, limit: int = _MAX_VERIFY_CAPTURE_BYTES) -> str:
        # TemporaryFile is seekable. Read only a bounded suffix to avoid loading noisy
        # verification output into coordinator memory.
        file_handle.seek(0, os.SEEK_END)
        size = file_handle.tell()
        file_handle.seek(max(0, size - limit), os.SEEK_SET)
        raw = file_handle.read()
        if not isinstance(raw, bytes):
            return str(raw)[-limit:]
        return raw.decode("utf-8", errors="replace")

    def _verify(
        self, ctx: AgentRunContext, state: AgentExecutionState, action: Mapping[str, object]
    ) -> JsonObject:
        if "run_tests" not in ctx.binding.tool_policy.allowed_tools:
            raise ServiceTierError("zetsu_agent_verify_not_allowed")
        argv = self._verify_argv(ctx.worktree, action.get("argv"))
        env = AgentSandboxManager.fixed_environment(ctx.binding)
        executable = shutil.which(argv[0], path=env.get("PATH"))
        if executable is None:
            raise ServiceTierError("zetsu_agent_verify_executable_missing", {"executable": argv[0]})
        timeout = min(600.0, self._remaining_wall(ctx, state))
        deadline = time.monotonic() + timeout
        before_head, before_status, before_changed = self._worktree_state(ctx.worktree)
        returncode: int | None = None
        stdout = ""
        stderr = ""
        aborted_category: str | None = None
        with (
            tempfile.TemporaryFile(mode="w+b") as stdout_file,
            tempfile.TemporaryFile(mode="w+b") as stderr_file,
        ):
            process = subprocess.Popen(  # nosec B603
                [executable, *argv[1:]],
                cwd=ctx.worktree,
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=os.name == "posix",
            )
            try:
                while process.poll() is None:
                    self._ensure_active(ctx, state)
                    if time.monotonic() >= deadline:
                        raise ServiceTierError("zetsu_agent_verification_timeout")
                    time.sleep(0.10)
                returncode = process.returncode
            except ServiceTierError as exc:
                aborted_category = exc.category
                self._stop_process_tree(process)
                returncode = process.returncode
            finally:
                verification_number = len(state.validation_history) + 1
                stdout_artifact = f"verify-{verification_number:03d}.stdout"
                stderr_artifact = f"verify-{verification_number:03d}.stderr"
                self.results.stage_stream(ctx.session_id, stdout_artifact, stdout_file)
                self.results.stage_stream(ctx.session_id, stderr_artifact, stderr_file)
                stdout = self._read_file_tail(stdout_file)
                stderr = self._read_file_tail(stderr_file)

        after_head, after_status, after_changed = self._worktree_state(ctx.worktree)
        verification_mutated_worktree = (
            before_head != after_head
            or before_status != after_status
            or before_changed != after_changed
        )
        command_id = self._verification_command_id(argv)
        if verification_mutated_worktree:
            state.mutation_epoch += 1
            state.unresolved_failures = [
                item
                for item in state.unresolved_failures
                if not item.startswith("latest_mutation_unverified:epoch=")
            ]
            state.unresolved_failures.append(self._mutation_marker(state.mutation_epoch))
            state.unresolved_failures.append(
                f"verification_mutated_worktree:epoch={state.mutation_epoch}:cmd={command_id}"
            )
            if aborted_category is None:
                aborted_category = "zetsu_agent_verification_mutated_worktree"

        qualifies = (
            self._verification_qualifies(argv, ctx.required_verification_argv)
            and not verification_mutated_worktree
        )
        passed = aborted_category is None and returncode == 0 and not verification_mutated_worktree
        record: JsonObject = {
            "argv": argv,
            "command_id": command_id,
            "returncode": returncode,
            "passed": passed,
            "aborted_category": aborted_category,
            "qualifies_for_mutation": qualifies,
            "mutation_epoch": state.mutation_epoch,
            "worktree_mutated": verification_mutated_worktree,
            "changed_paths_before": before_changed,
            "changed_paths_after": after_changed,
            "stdout_tail": stdout[-8_000:],
            "stderr_tail": stderr[-8_000:],
            "stdout_artifact": stdout_artifact,
            "stderr_artifact": stderr_artifact,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        state.validation_history.append(record)
        state.validation_history = state.validation_history[-16:]
        state.telemetry.verification_calls += 1

        if passed:
            self._remove_resolved_verification_failures(state, command_id)
            if qualifies:
                state.last_verified_epoch = state.mutation_epoch
                marker = self._mutation_marker(state.mutation_epoch)
                state.unresolved_failures = [
                    item for item in state.unresolved_failures if item != marker
                ]
        else:
            failure = (
                f"verification_failed:epoch={state.mutation_epoch}:cmd={command_id}:"
                f"category={aborted_category or 'returncode'}:rc={returncode}"
            )
            self._remove_resolved_verification_failures(state, command_id)
            state.unresolved_failures.append(failure)

        if aborted_category is not None:
            raise ServiceTierError(aborted_category, {"verification": record})
        return record

    @staticmethod
    def _state_digest(state: AgentExecutionState) -> str:
        validation_summary: list[JsonObject] = []
        for record in state.validation_history[-8:]:
            validation_summary.append(
                {
                    key: record.get(key)
                    for key in (
                        "command_id",
                        "returncode",
                        "passed",
                        "aborted_category",
                        "qualifies_for_mutation",
                        "mutation_epoch",
                    )
                }
            )
        exact = {
            "objective": state.objective,
            "step": state.step,
            "changed_paths": state.changed_paths,
            "worktree_head": state.worktree_head,
            "worktree_status_sha256": state.worktree_status_sha256,
            "lane": state.lane,
            "model_id": state.model_id,
            "context_limit": state.context_limit,
            "required_verification_argv": state.required_verification_argv,
            "target_initial_head": state.target_initial_head,
            "applied_patch_sha256": state.applied_patch_sha256,
            "validation_history": validation_summary,
            "unresolved_failures": state.unresolved_failures,
            "evidence_refs": state.evidence_refs,
            "next_state": state.next_state,
            "mutation_epoch": state.mutation_epoch,
            "last_verified_epoch": state.last_verified_epoch,
            "command_count": state.command_count,
            "output_cap_continuations": state.output_cap_continuations,
        }
        return json.dumps(exact, sort_keys=True, ensure_ascii=False)

    @classmethod
    def _exact_state_object(cls, state: AgentExecutionState) -> JsonObject:
        try:
            value: object = json.loads(cls._state_digest(state))
        except json.JSONDecodeError as exc:
            raise ServiceTierError("zetsu_agent_exact_state_invalid") from exc
        if not isinstance(value, dict):
            raise ServiceTierError("zetsu_agent_exact_state_invalid")
        return cast(JsonObject, value)

    def _messages(
        self, state: AgentExecutionState, ctx: AgentRunContext | None = None
    ) -> tuple[dict[str, str], dict[str, str]]:
        owner = ctx.user_id if ctx is not None else "agent"
        session = ctx.session_id if ctx is not None else "session"
        policy: JsonObject = {
            "policy_id": ctx.binding.tool_policy.policy_id if ctx is not None else "internal",
            "allowed_tools": (
                list(ctx.binding.tool_policy.allowed_tools) if ctx is not None else []
            ),
            "network_enabled": False,
            "max_commands": (
                ctx.binding.tool_policy.max_commands if ctx is not None else 0
            ),
        }
        required = ctx.required_verification_argv if ctx is not None else state.required_verification_argv
        try:
            plan = self.context_planner.plan(
                owner_user_id=owner,
                session_id=session,
                objective=state.objective,
                exact_state=self._exact_state_object(state),
                policy=policy,
                required_verification_argv=required,
                system_prompt=self._system_prompt(),
                recent_trajectory=state.recent_observations,
                semantic_summary=state.summary,
                compaction_ratio=ctx.compaction_ratio if ctx is not None else DEFAULT_COMPACTION_RATIO,
            )
        except ContextPlannerError as exc:
            raise ServiceTierError(exc.category, exc.evidence) from exc
        return plan.messages

    @staticmethod
    def _approximate_active_tokens(messages: Sequence[Mapping[str, str]]) -> int:
        serialized = json.dumps(list(messages), sort_keys=True, ensure_ascii=False)
        return _rough_tokens(serialized)

    def _compact(self, ctx: AgentRunContext, state: AgentExecutionState, threshold: int) -> None:
        messages_before = self._messages(state, ctx)
        before = self._approximate_active_tokens(messages_before)
        exact_state = self._exact_state_object(state)
        self._ensure_active(ctx, state)
        result = self.tiered.chat(
            user_id=ctx.user_id,
            lane=ctx.lane,
            domain="json",
            session_id=ctx.session_id,
            messages=self.context_planner.compaction_messages(
                objective=state.objective,
                exact_state=exact_state,
                prior_summary=state.summary,
                recent_trajectory=state.recent_observations,
            ),
        )
        state.telemetry.add_usage(result, compaction=True)
        state.summary = self._summary_content(result)
        state.recent_observations = state.recent_observations[-3:]
        # The last reported prompt count described the pre-compaction/compaction prompt,
        # not the newly compacted active context. Do not reuse it as the next trigger.
        state.telemetry.last_model_reported_input_tokens = None
        after = self._approximate_active_tokens(self._messages(state, ctx))
        state.telemetry.compactions += 1
        state.telemetry.approximate_active_context_tokens_before_last_compaction = before
        state.telemetry.approximate_active_context_tokens_after_last_compaction = after
        if after >= threshold:
            raise ServiceTierError("zetsu_agent_compaction_insufficient")

    @staticmethod
    def _owner_hash(user_id: str) -> str:
        return hashlib.sha256(user_id.encode("utf-8")).hexdigest()

    def _checkpoint_value(self, ctx: AgentRunContext, state: AgentExecutionState) -> JsonObject:
        head, status_sha256, changed = self._worktree_state(ctx.worktree)
        state.worktree_head = head
        state.worktree_status_sha256 = status_sha256
        state.changed_paths = changed
        elapsed = time.monotonic() - ctx.run_started
        consumed = state.consumed_wall_seconds + elapsed
        return {
            "schema_version": _CHECKPOINT_SCHEMA,
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "session_id": ctx.session_id,
            "user_id_sha256": self._owner_hash(ctx.user_id),
            "repo_id": ctx.repo_id,
            "base_revision": ctx.binding.base_revision,
            "objective": state.objective,
            "step": state.step,
            "summary": state.summary,
            "recent_observations": state.recent_observations[-_MAX_RECENT_OBSERVATIONS:],
            "changed_paths": state.changed_paths,
            "worktree_head": state.worktree_head,
            "worktree_status_sha256": state.worktree_status_sha256,
            "lane": ctx.lane.value,
            "model_id": ctx.model_id,
            "context_limit": ctx.context_limit,
            "required_verification_argv": (
                list(ctx.required_verification_argv)
                if ctx.required_verification_argv is not None
                else None
            ),
            "validation_history": state.validation_history[-16:],
            "unresolved_failures": state.unresolved_failures[-16:],
            "evidence_refs": state.evidence_refs[-64:],
            "next_state": state.next_state,
            "mutation_epoch": state.mutation_epoch,
            "last_verified_epoch": state.last_verified_epoch,
            "command_count": state.command_count,
            "output_cap_continuations": state.output_cap_continuations,
            "consumed_wall_seconds": round(consumed, 6),
            "apply_to_repository": ctx.apply_to_repository,
            "target_initial_head": state.target_initial_head,
            "target_initial_status_sha256": state.target_initial_status_sha256,
            "target_applied_status_sha256": state.target_applied_status_sha256,
            "applied_patch_sha256": state.applied_patch_sha256,
            "step_limit": ctx.max_steps,
            "max_chars": ctx.max_chars,
            "compaction_ratio": ctx.compaction_ratio,
            "tool_policy": {
                "policy_id": ctx.binding.tool_policy.policy_id,
                "allowed_tools": list(ctx.binding.tool_policy.allowed_tools),
                "network_enabled": ctx.binding.tool_policy.network_enabled,
                "max_commands": ctx.binding.tool_policy.max_commands,
                "max_wall_seconds": ctx.binding.tool_policy.max_wall_seconds,
            },
            "telemetry": state.telemetry.as_dict(),
        }

    def _checkpoint(self, ctx: AgentRunContext, state: AgentExecutionState) -> Path:
        return self.checkpoints.write(ctx.session_id, self._checkpoint_value(ctx, state))

    def _restore(self, ctx: AgentRunContext, instruction: str) -> AgentExecutionState:
        raw = self.checkpoints.read(ctx.session_id)
        if raw is None:
            raise ServiceTierError("zetsu_agent_checkpoint_missing")
        if raw.get("schema_version") != _CHECKPOINT_SCHEMA:
            raise ServiceTierError("zetsu_agent_checkpoint_schema_unsupported")
        if raw.get("user_id_sha256") != self._owner_hash(ctx.user_id):
            raise ServiceTierError("zetsu_agent_checkpoint_owner_mismatch")
        if (
            raw.get("repo_id") != ctx.repo_id
            or raw.get("base_revision") != ctx.binding.base_revision
        ):
            raise ServiceTierError("zetsu_agent_checkpoint_repository_mismatch")
        if raw.get("objective") != instruction:
            raise ServiceTierError("zetsu_agent_resume_objective_mismatch")
        if (
            raw.get("step_limit") != ctx.max_steps
            or raw.get("max_chars") != ctx.max_chars
            or raw.get("compaction_ratio") != ctx.compaction_ratio
            or raw.get("apply_to_repository", False) is not ctx.apply_to_repository
        ):
            raise ServiceTierError("zetsu_agent_resume_budget_mismatch")
        expected_tool_policy: JsonObject = {
            "policy_id": ctx.binding.tool_policy.policy_id,
            "allowed_tools": list(ctx.binding.tool_policy.allowed_tools),
            "network_enabled": ctx.binding.tool_policy.network_enabled,
            "max_commands": ctx.binding.tool_policy.max_commands,
            "max_wall_seconds": ctx.binding.tool_policy.max_wall_seconds,
        }
        if raw.get("tool_policy") != expected_tool_policy:
            raise ServiceTierError("zetsu_agent_resume_tool_policy_mismatch")
        if (
            raw.get("lane") != ctx.lane.value
            or raw.get("model_id") != ctx.model_id
            or raw.get("context_limit") != ctx.context_limit
        ):
            raise ServiceTierError("zetsu_agent_resume_route_mismatch")
        expected_verifier = (
            list(ctx.required_verification_argv)
            if ctx.required_verification_argv is not None
            else None
        )
        if raw.get("required_verification_argv") != expected_verifier:
            raise ServiceTierError("zetsu_agent_resume_verifier_mismatch")

        def integer(name: str, *, minimum: int = 0) -> int:
            value = raw.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ServiceTierError("zetsu_agent_checkpoint_state_invalid")
            return value

        def number(name: str, *, minimum: float = 0.0) -> float:
            value = raw.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ServiceTierError("zetsu_agent_checkpoint_state_invalid")
            result = float(value)
            if result < minimum:
                raise ServiceTierError("zetsu_agent_checkpoint_state_invalid")
            return result

        def string(name: str, *, maximum: int, allow_empty: bool = True) -> str:
            value = raw.get(name)
            if not isinstance(value, str) or len(value) > maximum or "\x00" in value:
                raise ServiceTierError("zetsu_agent_checkpoint_state_invalid")
            if not allow_empty and not value:
                raise ServiceTierError("zetsu_agent_checkpoint_state_invalid")
            return value

        def optional_string(name: str, *, maximum: int) -> str:
            value = raw.get(name, "")
            if not isinstance(value, str) or len(value) > maximum or "\x00" in value:
                raise ServiceTierError("zetsu_agent_checkpoint_state_invalid")
            return value

        def strings(name: str, *, maximum_items: int, maximum_chars: int) -> list[str]:
            value = raw.get(name)
            if (
                not isinstance(value, list)
                or len(value) > maximum_items
                or any(
                    not isinstance(item, str) or len(item) > maximum_chars or "\x00" in item
                    for item in value
                )
            ):
                raise ServiceTierError("zetsu_agent_checkpoint_state_invalid")
            return list(value)

        validation = raw.get("validation_history")
        if (
            not isinstance(validation, list)
            or len(validation) > 16
            or any(not isinstance(item, dict) for item in validation)
        ):
            raise ServiceTierError("zetsu_agent_checkpoint_state_invalid")
        validation_history = [cast(JsonObject, dict(item)) for item in validation]

        summary = string("summary", maximum=128_000)
        recent_observations = strings(
            "recent_observations",
            maximum_items=_MAX_RECENT_OBSERVATIONS,
            maximum_chars=_MAX_OBSERVATION_CHARS,
        )
        changed_paths = strings(
            "changed_paths",
            maximum_items=_MAX_CHANGED_PATHS,
            maximum_chars=500,
        )
        unresolved_failures = strings("unresolved_failures", maximum_items=16, maximum_chars=2_000)
        evidence_refs = strings("evidence_refs", maximum_items=64, maximum_chars=2_000)
        worktree_head = string("worktree_head", maximum=128, allow_empty=False)
        worktree_status_sha256 = string("worktree_status_sha256", maximum=64, allow_empty=False)
        if len(worktree_status_sha256) != 64 or any(
            ch not in "0123456789abcdef" for ch in worktree_status_sha256
        ):
            raise ServiceTierError("zetsu_agent_checkpoint_state_invalid")
        next_state = string("next_state", maximum=128, allow_empty=False)
        step = integer("step")
        mutation_epoch = integer("mutation_epoch")
        last_verified_raw = raw.get("last_verified_epoch")
        if (
            isinstance(last_verified_raw, bool)
            or not isinstance(last_verified_raw, int)
            or last_verified_raw < -1
        ):
            raise ServiceTierError("zetsu_agent_checkpoint_state_invalid")
        command_count = integer("command_count")
        output_cap_continuations_raw = raw.get("output_cap_continuations", 0)
        if (
            isinstance(output_cap_continuations_raw, bool)
            or not isinstance(output_cap_continuations_raw, int)
            or not 0 <= output_cap_continuations_raw <= _MAX_OUTPUT_CAP_CONTINUATIONS
        ):
            raise ServiceTierError("zetsu_agent_checkpoint_state_invalid")
        consumed_wall_seconds = number("consumed_wall_seconds")
        target_initial_head = optional_string("target_initial_head", maximum=128)
        target_initial_status_sha256 = optional_string("target_initial_status_sha256", maximum=64)
        target_applied_status_sha256 = optional_string("target_applied_status_sha256", maximum=64)
        applied_patch_sha256 = optional_string("applied_patch_sha256", maximum=64)
        for digest in (
            target_initial_status_sha256,
            target_applied_status_sha256,
            applied_patch_sha256,
        ):
            if digest and (len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest)):
                raise ServiceTierError("zetsu_agent_checkpoint_state_invalid")
        if ctx.apply_to_repository:
            if (
                target_initial_head != ctx.binding.base_revision
                or not target_initial_status_sha256
                or bool(target_applied_status_sha256) is not bool(applied_patch_sha256)
            ):
                raise ServiceTierError("zetsu_agent_checkpoint_state_invalid")
        elif any(
            (
                target_initial_head,
                target_initial_status_sha256,
                target_applied_status_sha256,
                applied_patch_sha256,
            )
        ):
            raise ServiceTierError("zetsu_agent_checkpoint_state_invalid")

        state = AgentExecutionState(
            objective=instruction,
            step=step,
            summary=summary,
            recent_observations=recent_observations,
            changed_paths=changed_paths,
            worktree_head=worktree_head,
            worktree_status_sha256=worktree_status_sha256,
            lane=ctx.lane.value,
            model_id=ctx.model_id,
            context_limit=ctx.context_limit,
            required_verification_argv=(
                list(ctx.required_verification_argv)
                if ctx.required_verification_argv is not None
                else None
            ),
            validation_history=validation_history,
            unresolved_failures=unresolved_failures,
            evidence_refs=evidence_refs,
            next_state=next_state,
            mutation_epoch=mutation_epoch,
            last_verified_epoch=last_verified_raw,
            command_count=command_count,
            output_cap_continuations=output_cap_continuations_raw,
            consumed_wall_seconds=consumed_wall_seconds,
            target_initial_head=target_initial_head,
            target_initial_status_sha256=target_initial_status_sha256,
            target_applied_status_sha256=target_applied_status_sha256,
            applied_patch_sha256=applied_patch_sha256,
            telemetry=AgentTelemetry.from_mapping(raw.get("telemetry")),
        )
        if not (
            0 <= state.step <= ctx.max_steps
            and 0 <= state.command_count <= ctx.binding.tool_policy.max_commands
            and 0.0 <= state.consumed_wall_seconds <= ctx.binding.tool_policy.max_wall_seconds
            and -1 <= state.last_verified_epoch <= state.mutation_epoch
            and state.telemetry.tool_calls == state.command_count
        ):
            raise ServiceTierError("zetsu_agent_checkpoint_accounting_invalid")

        head, status_sha256, changed = self._worktree_state(ctx.worktree)
        if (
            head != state.worktree_head
            or status_sha256 != state.worktree_status_sha256
            or changed != state.changed_paths
        ):
            raise ServiceTierError(
                "zetsu_agent_checkpoint_worktree_drift",
                {
                    "checkpoint_head": state.worktree_head,
                    "current_head": head,
                    "checkpoint_status_sha256": state.worktree_status_sha256,
                    "current_status_sha256": status_sha256,
                    "checkpoint_changed_paths": state.changed_paths,
                    "current_changed_paths": changed,
                },
            )
        return state

    def _finish_allowed(self, state: AgentExecutionState) -> bool:
        if state.unresolved_failures:
            return False
        return state.mutation_epoch == 0 or state.last_verified_epoch == state.mutation_epoch

    @staticmethod
    def _record_recoverable_model_failure(
        state: AgentExecutionState, step: int, exc: ServiceTierError
    ) -> bool:
        if exc.category != "response_validation_failed":
            return False
        usage = exc.evidence.get("model_reported_usage")
        state.telemetry.add_usage(
            {"response": {"usage": dict(usage) if isinstance(usage, Mapping) else None}}
        )
        state.telemetry.agent_steps += 1
        gate = exc.evidence.get("gate_id")
        reason = exc.evidence.get("reason")
        state.recent_observations.append(
            f"STEP {step} MODEL_RESPONSE_REJECTED gate={gate} reason={reason}; "
            "return one valid JSON action"
        )
        state.recent_observations = state.recent_observations[-_MAX_RECENT_OBSERVATIONS:]
        state.next_state = "retry_valid_json_action"
        return True

    def _record_model_failure(
        self,
        ctx: AgentRunContext,
        state: AgentExecutionState,
        step: int,
        exc: ServiceTierError,
    ) -> bool:
        output_cap = exc.category == "model_output_limit_reached" or (
            exc.category == "response_validation_failed"
            and exc.evidence.get("gate_id") == "no_silent_truncation"
        )
        if not output_cap:
            return self._record_recoverable_model_failure(state, step, exc)
        usage = exc.evidence.get("model_reported_usage")
        state.telemetry.add_usage(
            {"response": {"usage": dict(usage) if isinstance(usage, Mapping) else None}}
        )
        state.telemetry.agent_steps += 1
        if state.output_cap_continuations >= _MAX_OUTPUT_CAP_CONTINUATIONS:
            return False
        state.output_cap_continuations += 1
        partial = exc.evidence.get("partial_content")
        if isinstance(partial, str):
            self.results.stage_stream(
                ctx.session_id,
                f"output-cap-{state.output_cap_continuations:03d}.partial",
                io.BytesIO(partial.encode("utf-8")),
            )
        state.recent_observations.append(
            (
                f"STEP {step} MODEL_OUTPUT_CAP continuation="
                f"{state.output_cap_continuations}/{_MAX_OUTPUT_CAP_CONTINUATIONS}; "
                "the partial action was not executed, so decompose the same remaining operation "
                "into a smaller bounded action without repeating confirmed mutations"
            )
        )
        state.recent_observations = state.recent_observations[-_MAX_RECENT_OBSERVATIONS:]
        state.next_state = "continue_after_output_cap"
        return True

    def _record_failure_best_effort(
        self,
        ctx: AgentRunContext,
        state: AgentExecutionState,
        category: str,
        *,
        terminal: bool = True,
    ) -> None:
        try:
            self.tiered.sandboxes.record_result(
                ctx.session_id,
                user_id=ctx.user_id,
                command_count=state.command_count,
                verification_summary=f"FAILED:{category}",
                failed=True,
                terminal=terminal,
            )
        except Exception:
            # Preserve the original failure/cancellation category.
            pass

    def _release_setup_session_best_effort(
        self, *, user_id: str, session_id: str, creating: bool
    ) -> None:
        """Release only a fresh, untouched setup session after preflight failure."""

        if not creating:
            return
        try:
            inspect = self.tiered.sandboxes.inspect(session_id, user_id=user_id)
            if (
                inspect.get("state") == "ACTIVE"
                and inspect.get("physical_state") == "PRESENT"
                and inspect.get("changed_paths") == []
            ):
                self.tiered.sandboxes.close_if_clean(session_id, user_id=user_id)
        except Exception:
            # The original deterministic failure is more useful than cleanup noise.
            pass

    def _finalize_terminal_failure_best_effort(
        self,
        ctx: AgentRunContext,
        state: AgentExecutionState,
        category: str,
        *,
        terminal: bool = True,
    ) -> None:
        """Persist diagnostics first and release only an explicitly terminal task."""

        try:
            head, status_sha256, changed = self._worktree_state(ctx.worktree)
            state.worktree_head = head
            state.worktree_status_sha256 = status_sha256
            state.changed_paths = changed
            checkpoint = self._checkpoint(ctx, state)
            handoff = self._handoff_patch(
                ctx.worktree,
                ctx.session_id,
                changed,
                max_chars=ctx.max_chars,
            )
            failure: JsonObject = {
                "status": "FAILED",
                "session_id": ctx.session_id,
                "repo_id": ctx.repo_id,
                "category": category,
                "changed_paths": changed,
                "validation_history": state.validation_history,
                "unresolved_failures": state.unresolved_failures,
                "checkpoint_path": str(checkpoint),
                "handoff": handoff,
                "telemetry": state.telemetry.as_dict(),
            }
            artifacts: dict[str, Path | bytes | str] = {
                "result.json": json.dumps(
                    failure, indent=2, sort_keys=True, ensure_ascii=False
                )
                + "\n",
                "checkpoint.json": checkpoint,
            }
            patch_path = handoff.get("patch_path")
            if isinstance(patch_path, str):
                artifacts["handoff.patch"] = Path(patch_path)
            artifacts.update(self.results.staging_artifacts(ctx.session_id))
            delivery = self.results.persist(
                user_id=ctx.user_id,
                repo_id=ctx.repo_id,
                session_id=ctx.session_id,
                status="FAILED",
                summary=f"FAILED:{category}",
                artifacts=artifacts,
            )
            self.tiered.sandboxes.record_result(
                ctx.session_id,
                user_id=ctx.user_id,
                command_count=state.command_count,
                verification_summary=f"FAILED:{category}",
                failed=True,
                terminal=terminal,
            )
            cleanup = getattr(self.tiered.sandboxes, "authorize_terminal_cleanup", None)
            if terminal and callable(cleanup):
                cleanup(
                    ctx.session_id,
                    user_id=ctx.user_id,
                    result_id=str(delivery["result_id"]),
                )
            self.results.clear_staging(ctx.session_id)
        except Exception:
            self._record_failure_best_effort(ctx, state, category, terminal=terminal)

    def run(
        self,
        *,
        user_id: str,
        repo_id: str,
        instruction: str,
        lane: ModelLane = ModelLane.QUALITY,
        session_id: str | None = None,
        max_steps: int = 12,
        max_chars: int = 8_000,
        compaction_ratio: float = 0.80,
        verification_argv: Sequence[str] | None = None,
        apply_to_repository: bool = False,
        wait_timeout_seconds: float = 1_800.0,
        persistent_session: bool = False,
        restart_objective: bool = False,
    ) -> JsonObject:
        self._validate_run_request(
            instruction=instruction,
            lane=lane,
            max_steps=max_steps,
            max_chars=max_chars,
            compaction_ratio=compaction_ratio,
            verification_argv=verification_argv,
            apply_to_repository=apply_to_repository,
        )
        if restart_objective and not persistent_session:
            raise ServiceTierError("repository_agent_turn_mode_invalid")
        if persistent_session and apply_to_repository:
            raise ServiceTierError("repository_agent_turn_canonical_apply_forbidden")
        effective_session = session_id or f"zetsu-{uuid.uuid4().hex}"
        has_session = getattr(self.tiered.sandboxes, "has_session", None)
        creating = session_id is None or (
            callable(has_session) and not has_session(effective_session, user_id=user_id)
        )
        lock = self._session_lock(effective_session)
        if not lock.acquire(blocking=False):
            raise ServiceTierError("zetsu_agent_session_busy")
        admission: AgentAdmission | None = None
        try:
            digest = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
            if self.scheduler is not None:
                try:
                    admission = self.scheduler.wait_for_admission(
                        user_id=user_id,
                        repo_id=repo_id,
                        session_id=effective_session,
                        instruction_digest=digest,
                        wait_timeout_seconds=wait_timeout_seconds,
                    )
                except AgentSchedulerError as exc:
                    raise ServiceTierError(exc.category, exc.evidence) from exc
            result = self._run_unlocked(
                user_id=user_id,
                repo_id=repo_id,
                instruction=instruction,
                lane=lane,
                session_id=effective_session,
                creating=creating,
                max_steps=max_steps,
                max_chars=max_chars,
                compaction_ratio=compaction_ratio,
                verification_argv=verification_argv,
                apply_to_repository=apply_to_repository,
                persistent_session=persistent_session,
                restart_objective=restart_objective,
            )
            if admission is not None and self.scheduler is not None:
                terminal = "SUCCEEDED" if result.get("status") == "SUCCESS" else "FAILED"
                result_id = result.get("result_id")
                self.scheduler.finish(
                    admission,
                    state=terminal,
                    result_id=str(result_id) if isinstance(result_id, str) else None,
                    failure_category=(
                        None if terminal == "SUCCEEDED" else str(result.get("status"))
                    ),
                )
                result["scheduler"] = {
                    "state": terminal,
                    "ticket_id": admission.ticket_id,
                    "queue_position_at_arrival": admission.queue_position_at_arrival,
                    "queue_wait_seconds": admission.queue_wait_seconds,
                }
            return result
        except (KeyboardInterrupt, SystemExit):
            if admission is not None and self.scheduler is not None:
                self.scheduler.finish(
                    admission,
                    state="RESUMABLE",
                    failure_category="process_interrupted",
                )
            raise
        except Exception as exc:
            if admission is not None and self.scheduler is not None:
                category = getattr(exc, "category", type(exc).__name__)
                scheduler_state = "FAILED"
                reconcile_exit = getattr(
                    self.tiered.sandboxes, "reconcile_executor_exit", None
                )
                if callable(reconcile_exit):
                    try:
                        reconcile_exit(effective_session, user_id=user_id)
                        lifecycle = self.tiered.sandboxes.status(
                            effective_session, user_id=user_id
                        )
                        if lifecycle.get("lifecycle_state") == "INTERRUPTED_RESUMABLE":
                            scheduler_state = "RESUMABLE"
                    except Exception:
                        pass
                self.scheduler.finish(
                    admission,
                    state=scheduler_state,
                    failure_category=str(category),
                )
            self._release_setup_session_best_effort(
                user_id=user_id,
                session_id=effective_session,
                creating=creating,
            )
            raise
        finally:
            lock.release()

    def run_turn(
        self,
        *,
        user_id: str,
        repo_id: str,
        instruction: str,
        lane: ModelLane,
        session_id: str,
        max_steps: int = 12,
        max_chars: int = 8_000,
        verification_argv: Sequence[str] | None = None,
        wait_timeout_seconds: float = 1_800.0,
    ) -> JsonObject:
        """Run one bounded turn while preserving the owned worktree for continuation."""

        return self.run(
            user_id=user_id,
            repo_id=repo_id,
            instruction=instruction,
            lane=lane,
            session_id=session_id,
            max_steps=max_steps,
            max_chars=max_chars,
            verification_argv=verification_argv,
            apply_to_repository=False,
            wait_timeout_seconds=wait_timeout_seconds,
            persistent_session=True,
            restart_objective=True,
        )

    def result_page(
        self,
        *,
        user_id: str,
        repo_id: str,
        session_id: str,
        result_id: str,
        artifact: str,
        offset: int,
        max_bytes: int,
    ) -> JsonObject:
        try:
            return self.results.page(
                user_id=user_id,
                repo_id=repo_id,
                session_id=session_id,
                result_id=result_id,
                artifact=artifact,
                offset=offset,
                max_bytes=max_bytes,
            )
        except ResultStoreError as exc:
            raise ServiceTierError(exc.category) from exc

    @staticmethod
    def _validate_run_request(
        *,
        instruction: str,
        lane: ModelLane,
        max_steps: int,
        max_chars: int,
        compaction_ratio: float,
        verification_argv: Sequence[str] | None,
        apply_to_repository: bool,
    ) -> None:
        if lane not in {ModelLane.QUALITY, ModelLane.STANDARD}:
            raise ServiceTierError("zetsu_agent_lane_invalid")
        if not instruction.strip() or len(instruction) > 40_000:
            raise ServiceTierError("zetsu_agent_instruction_invalid")
        if not 1 <= max_steps <= 32 or not 512 <= max_chars <= 24_000:
            raise ServiceTierError("zetsu_agent_budget_invalid")
        if not 0.75 <= compaction_ratio <= 0.85:
            raise ServiceTierError("zetsu_agent_compaction_ratio_invalid")
        if not isinstance(apply_to_repository, bool):
            raise ServiceTierError("zetsu_agent_apply_mode_invalid")
        if apply_to_repository and verification_argv is None:
            raise ServiceTierError("zetsu_agent_apply_requires_verifier")

    def scheduler_status(self, *, user_id: str) -> JsonObject:
        if self.scheduler is None:
            return {"status": "UNAVAILABLE"}
        return self.scheduler.snapshot(user_id=user_id)

    def task_status(self, *, user_id: str, session_id: str) -> JsonObject:
        if self.scheduler is None:
            raise ServiceTierError("agent_scheduler_unavailable")
        try:
            return self.scheduler.task_status(user_id=user_id, session_id=session_id)
        except AgentSchedulerError as exc:
            raise ServiceTierError(exc.category, exc.evidence) from exc

    def cancel_queued(self, *, user_id: str, session_id: str) -> JsonObject:
        if self.scheduler is None:
            raise ServiceTierError("agent_scheduler_unavailable")
        try:
            return self.scheduler.cancel(user_id=user_id, session_id=session_id)
        except AgentSchedulerError as exc:
            raise ServiceTierError(exc.category, exc.evidence) from exc

    def _run_unlocked(
        self,
        *,
        user_id: str,
        repo_id: str,
        instruction: str,
        lane: ModelLane = ModelLane.QUALITY,
        session_id: str | None = None,
        creating: bool | None = None,
        max_steps: int = 12,
        max_chars: int = 8_000,
        compaction_ratio: float = 0.80,
        verification_argv: Sequence[str] | None = None,
        apply_to_repository: bool = False,
        persistent_session: bool = False,
        restart_objective: bool = False,
    ) -> JsonObject:
        self._validate_run_request(
            instruction=instruction,
            lane=lane,
            max_steps=max_steps,
            max_chars=max_chars,
            compaction_ratio=compaction_ratio,
            verification_argv=verification_argv,
            apply_to_repository=apply_to_repository,
        )
        if restart_objective and not persistent_session:
            raise ServiceTierError("repository_agent_turn_mode_invalid")
        if persistent_session and apply_to_repository:
            raise ServiceTierError("repository_agent_turn_canonical_apply_forbidden")

        effective_session = session_id or f"zetsu-{uuid.uuid4().hex}"
        creating = session_id is None if creating is None else creating
        digest = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
        if creating:
            self.tiered.create_agent_session(
                user_id=user_id,
                repo_id=repo_id,
                session_id=effective_session,
                tool_policy=AgentToolPolicy(
                    policy_id="zetsu-qwen-agent-v2",
                    allowed_tools=("apply_patch", "run_tests"),
                    network_enabled=False,
                    max_commands=max_steps * 2,
                    max_wall_seconds=1_800,
                ),
                task_title="Zetsu Qwen delegated task",
                instruction_digest=digest,
                lane=lane.value,
            )

        binding = self.tiered.sandboxes.require_active(effective_session, user_id=user_id)
        if binding.repo_id != repo_id:
            raise ServiceTierError("zetsu_agent_repo_mismatch")
        route = self.tiered.lane_policy.routes[lane]
        worktree = self._worktree(binding.worktree_root)
        required_verification_argv = (
            tuple(self._verify_argv(worktree, verification_argv))
            if verification_argv is not None
            else None
        )
        if required_verification_argv is not None and not self._verification_qualifies(
            required_verification_argv, required_verification_argv
        ):
            raise ServiceTierError("zetsu_agent_required_verifier_not_qualifying")
        run_started = time.monotonic()
        prior_consumed = 0.0
        if not creating:
            raw = self.checkpoints.read(effective_session)
            consumed = raw.get("consumed_wall_seconds") if isinstance(raw, Mapping) else None
            if isinstance(consumed, (int, float)):
                prior_consumed = float(consumed)
                if not 0.0 <= prior_consumed <= binding.tool_policy.max_wall_seconds:
                    raise ServiceTierError("zetsu_agent_checkpoint_wall_budget_invalid")
        remaining_wall = binding.tool_policy.max_wall_seconds - prior_consumed
        if remaining_wall <= 0:
            raise ServiceTierError("zetsu_agent_wall_budget_exhausted")
        ctx = AgentRunContext(
            user_id=user_id,
            session_id=effective_session,
            repo_id=repo_id,
            lane=lane,
            binding=binding,
            worktree=worktree,
            max_steps=max_steps,
            max_chars=max_chars,
            compaction_ratio=compaction_ratio,
            model_id=route.model_id,
            context_limit=route.context_limit,
            required_verification_argv=required_verification_argv,
            run_started=run_started,
            remaining_wall_seconds=remaining_wall,
            apply_to_repository=apply_to_repository,
        )
        fresh_state = AgentExecutionState(
            objective=instruction,
            lane=lane.value,
            model_id=route.model_id,
            context_limit=route.context_limit,
            required_verification_argv=(
                list(required_verification_argv)
                if required_verification_argv is not None
                else None
            ),
        )
        if creating:
            state = fresh_state
        elif restart_objective:
            prior_raw = self.checkpoints.read(effective_session)
            if prior_raw is None:
                state = fresh_state
            else:
                prior_objective = prior_raw.get("objective")
                if not isinstance(prior_objective, str) or not prior_objective:
                    raise ServiceTierError("zetsu_agent_checkpoint_state_invalid")
                prior_verifier_raw = prior_raw.get("required_verification_argv")
                verifier_upgrade = (
                    prior_verifier_raw is None
                    and required_verification_argv is not None
                )
                restore_ctx = (
                    replace(ctx, required_verification_argv=None)
                    if verifier_upgrade
                    else ctx
                )
                prior = self._restore(restore_ctx, prior_objective)
                if verifier_upgrade and (
                    prior.mutation_epoch != 0
                    or prior.last_verified_epoch not in {-1, 0}
                    or prior.changed_paths
                ):
                    raise ServiceTierError("zetsu_agent_verifier_upgrade_unsafe")
                state = AgentExecutionState(
                    objective=instruction,
                    summary=prior.summary,
                    recent_observations=[
                        *prior.recent_observations[-3:],
                        "PERSISTENT TURN BOUNDARY: continue in the same owned worktree",
                    ],
                    changed_paths=list(prior.changed_paths),
                    worktree_head=prior.worktree_head,
                    worktree_status_sha256=prior.worktree_status_sha256,
                    lane=lane.value,
                    model_id=route.model_id,
                    context_limit=route.context_limit,
                    required_verification_argv=(
                        list(required_verification_argv)
                        if required_verification_argv is not None
                        else None
                    ),
                    validation_history=list(prior.validation_history[-8:]),
                    unresolved_failures=list(prior.unresolved_failures),
                    evidence_refs=list(prior.evidence_refs[-32:]),
                    mutation_epoch=prior.mutation_epoch,
                    last_verified_epoch=prior.last_verified_epoch,
                    command_count=prior.command_count,
                    consumed_wall_seconds=prior.consumed_wall_seconds,
                    telemetry=prior.telemetry,
                )
        else:
            state = self._restore(ctx, instruction)
        if creating or restart_objective:
            head, status_sha256, changed = self._worktree_state(worktree)
            state.worktree_head = head
            state.worktree_status_sha256 = status_sha256
            state.changed_paths = changed
            if apply_to_repository:
                target = Path(binding.canonical_repository_root).resolve(strict=True)
                target_head, target_status, target_changed = self._worktree_state(target)
                if target_head != binding.base_revision or target_changed:
                    raise ServiceTierError("zetsu_agent_apply_target_not_clean")
                state.target_initial_head = target_head
                state.target_initial_status_sha256 = target_status
        resume_verified_finish = (
            not creating and state.next_state == "finished" and self._finish_allowed(state)
        )
        if state.step >= max_steps and not resume_verified_finish:
            raise ServiceTierError("zetsu_agent_step_budget_exhausted")

        self.tiered.sandboxes.start_task(
            effective_session,
            user_id=user_id,
            lane=lane.value,
            sanitized_model_name=route.model_id,
            instruction_digest=digest,
            remaining_wall_seconds=remaining_wall,
        )
        threshold = int(route.context_limit * compaction_ratio)
        status = "SUCCESS" if resume_verified_finish else "INCOMPLETE"
        final_result = (
            "Resumed from a persisted caller-verified finish state."
            if resume_verified_finish
            else ""
        )
        failure_category: str | None = None
        try:
            pending_steps = () if resume_verified_finish else range(state.step + 1, max_steps + 1)
            for step in pending_steps:
                state.step = step
                self._ensure_active(ctx, state)
                messages = self._messages(state, ctx)
                approximate = self._approximate_active_tokens(messages)
                reported = state.telemetry.last_model_reported_input_tokens or 0
                if self.context_planner.should_compact(
                    approximate_tokens=max(approximate, reported),
                    context_limit=route.context_limit,
                    ratio=compaction_ratio,
                ):
                    self._compact(ctx, state, threshold)
                    self._checkpoint(ctx, state)
                    messages = self._messages(state, ctx)
                self._ensure_active(ctx, state)
                try:
                    result = self.tiered.chat(
                        user_id=user_id,
                        lane=lane,
                        domain="json",
                        session_id=effective_session,
                        messages=messages,
                    )
                except ServiceTierError as exc:
                    if self._record_model_failure(ctx, state, step, exc):
                        self._checkpoint(ctx, state)
                        continue
                    if (
                        exc.category == "model_output_limit_reached"
                        and state.output_cap_continuations
                        >= _MAX_OUTPUT_CAP_CONTINUATIONS
                    ):
                        raise ServiceTierError(
                            "output_cap_continuation_budget_exhausted",
                            {"continuations": state.output_cap_continuations},
                        ) from exc
                    raise
                state.telemetry.add_usage(result)
                state.telemetry.agent_steps += 1
                self._ensure_active(ctx, state)
                action = self._json_content(result)
                action_name = str(action["action"])

                if action_name == "finish":
                    value = action.get("result")
                    if not isinstance(value, str) or not value.strip():
                        raise ServiceTierError("zetsu_agent_finish_invalid")
                    if not self._finish_allowed(state):
                        observation = "FINISH_REJECTED: deterministic verification required after latest mutation"
                        state.recent_observations.append(
                            f"STEP {step} ACTION finish\n{observation}"
                        )
                        state.next_state = "verify_latest_mutation"
                        self._checkpoint(ctx, state)
                        continue
                    final_result = value.strip()
                    status = "SUCCESS"
                    state.summary = final_result
                    state.next_state = "finished"
                    self._checkpoint(ctx, state)
                    break

                self._consume_tool_budget(ctx, state)
                if action_name in {
                    "repo_map",
                    "find_symbol",
                    "find_references",
                    "search_text",
                    "read_region",
                    "inspect_diff",
                    "edit_region",
                    "create_text_file",
                    "git_state",
                }:
                    try:
                        aci = self._typed_aci(ctx, state)
                        if action_name == "repo_map":
                            focus = action.get("focus_paths", ())
                            if not isinstance(focus, Sequence) or isinstance(focus, (str, bytes)):
                                raise ServiceTierError("zetsu_agent_focus_paths_invalid")
                            result = aci.repo_map(
                                query=cast(str, action.get("query", "")),
                                focus_paths=tuple(cast(str, item) for item in focus),
                                token_budget=cast(int, action.get("token_budget", 1_000)),
                            )
                        elif action_name == "find_symbol":
                            result = aci.find_symbol(cast(str, action.get("name")))
                        elif action_name == "find_references":
                            result = aci.find_references(cast(str, action.get("name")))
                        elif action_name == "search_text":
                            result = aci.search_text(
                                query=cast(str, action.get("query")),
                                glob=cast(str, action.get("glob", "*")),
                            )
                        elif action_name == "read_region":
                            result = aci.read_region(
                                path=cast(str, action.get("path")),
                                start_line=cast(int, action.get("start_line")),
                                end_line=cast(int, action.get("end_line")),
                            )
                        elif action_name == "inspect_diff":
                            paths = action.get("paths", ())
                            if not isinstance(paths, Sequence) or isinstance(paths, (str, bytes)):
                                raise ServiceTierError("zetsu_agent_diff_paths_invalid")
                            result = aci.inspect_diff(
                                paths=tuple(cast(str, item) for item in paths)
                            )
                        elif action_name == "edit_region":
                            if ctx.required_verification_argv is None:
                                raise ServiceTierError("zetsu_agent_mutation_requires_verifier")
                            result = aci.edit_region(
                                path=cast(str, action.get("path")),
                                old_text=cast(str, action.get("old_text")),
                                new_text=cast(str, action.get("new_text")),
                            )
                            state.mutation_epoch += 1
                            state.unresolved_failures = [
                                item
                                for item in state.unresolved_failures
                                if not item.startswith("latest_mutation_unverified:")
                            ]
                            state.unresolved_failures.append(
                                self._mutation_marker(state.mutation_epoch)
                            )
                        elif action_name == "create_text_file":
                            if ctx.required_verification_argv is None:
                                raise ServiceTierError("zetsu_agent_mutation_requires_verifier")
                            result = aci.create_text_file(
                                path=cast(str, action.get("path")),
                                content=cast(str, action.get("content")),
                            )
                            state.mutation_epoch += 1
                            state.unresolved_failures = [
                                item
                                for item in state.unresolved_failures
                                if not item.startswith("latest_mutation_unverified:")
                            ]
                            state.unresolved_failures.append(
                                self._mutation_marker(state.mutation_epoch)
                            )
                        else:
                            result = aci.git_state()
                        observation = json.dumps(result, sort_keys=True, ensure_ascii=False)
                    except BoundedACIError as exc:
                        raise ServiceTierError(exc.category, exc.evidence) from exc
                elif action_name == "search":
                    observation = self._search(ctx, state, action)
                elif action_name == "read":
                    observation = self._read(ctx, action)
                elif action_name == "retrieve":
                    observation = self._retrieve(ctx, state, action)
                elif action_name == "edit":
                    if ctx.required_verification_argv is None:
                        raise ServiceTierError("zetsu_agent_mutation_requires_verifier")
                    observation = self._edit(ctx, action)
                    state.mutation_epoch += 1
                    state.unresolved_failures = [
                        item
                        for item in state.unresolved_failures
                        if not item.startswith("latest_mutation_unverified:epoch=")
                    ]
                    state.unresolved_failures.append(self._mutation_marker(state.mutation_epoch))
                elif action_name == "create":
                    if ctx.required_verification_argv is None:
                        raise ServiceTierError("zetsu_agent_mutation_requires_verifier")
                    observation = self._create(ctx, action)
                    state.mutation_epoch += 1
                    state.unresolved_failures = [
                        item
                        for item in state.unresolved_failures
                        if not item.startswith("latest_mutation_unverified:epoch=")
                    ]
                    state.unresolved_failures.append(self._mutation_marker(state.mutation_epoch))
                elif action_name == "verify":
                    verification = self._verify(ctx, state, action)
                    observation = json.dumps(verification, sort_keys=True)
                else:
                    raise ServiceTierError("zetsu_agent_action_unknown", {"action": action_name})

                head, worktree_status_sha256, changed = self._worktree_state(worktree)
                state.worktree_head = head
                state.worktree_status_sha256 = worktree_status_sha256
                state.changed_paths = changed
                state.recent_observations.append(
                    f"STEP {step} ACTION {action_name}\n{observation[-_MAX_OBSERVATION_CHARS:]}"
                )
                state.recent_observations = state.recent_observations[-_MAX_RECENT_OBSERVATIONS:]
                state.next_state = "choose_action"
                self._checkpoint(ctx, state)
                if (
                    action_name == "verify"
                    and state.mutation_epoch > 0
                    and self._finish_allowed(state)
                ):
                    final_result = "Caller-selected deterministic verification passed after the latest mutation."
                    status = "SUCCESS"
                    state.next_state = "finished"
                    self._checkpoint(ctx, state)
                    break
            if status != "SUCCESS":
                failure_category = (
                    "output_cap_continuation_budget_exhausted"
                    if state.output_cap_continuations
                    else "max_steps_exhausted"
                )
        except (KeyboardInterrupt, SystemExit):
            try:
                self._checkpoint(ctx, state)
                interrupted = getattr(self.tiered.sandboxes, "record_interrupted", None)
                if callable(interrupted):
                    interrupted(
                        ctx.session_id,
                        user_id=ctx.user_id,
                        reason="process_interrupted",
                    )
            except Exception:
                pass
            raise
        except (AgentSandboxError, CorpusError) as exc:
            failure_category = getattr(exc, "category", type(exc).__name__)
            try:
                self._checkpoint(ctx, state)
            except Exception:
                pass
            self._finalize_terminal_failure_best_effort(
                ctx,
                state,
                str(failure_category),
                terminal=not persistent_session,
            )
            evidence = getattr(exc, "evidence", None)
            raise ServiceTierError(
                str(failure_category), evidence if isinstance(evidence, dict) else None
            ) from exc
        except ResultStoreError as exc:
            failure_category = exc.category
            self._finalize_terminal_failure_best_effort(
                ctx, state, exc.category, terminal=not persistent_session
            )
            raise ServiceTierError(exc.category) from exc
        except (ServiceTierError, subprocess.TimeoutExpired) as exc:
            failure_category = getattr(exc, "category", "zetsu_agent_timeout")
            try:
                self._checkpoint(ctx, state)
            except Exception:
                pass
            self._finalize_terminal_failure_best_effort(
                ctx,
                state,
                str(failure_category),
                terminal=not persistent_session,
            )
            raise
        finally:
            elapsed = time.monotonic() - run_started
            state.consumed_wall_seconds += elapsed
            if persistent_session:
                try:
                    self._checkpoint(replace(ctx, run_started=time.monotonic()), state)
                except Exception:
                    # Preserve the original deterministic outcome; a later turn
                    # will fail closed if the checkpoint is unavailable or corrupt.
                    pass

        head, worktree_status_sha256, changed = self._worktree_state(worktree)
        state.worktree_head = head
        state.worktree_status_sha256 = worktree_status_sha256
        state.changed_paths = changed
        if not final_result:
            final_result = "Maximum bounded agent steps reached before verified finish."
        authoritative_content = final_result
        truncated = len(authoritative_content) > max_chars
        if truncated:
            final_result = authoritative_content[: max_chars - 1].rstrip() + "…"
        handoff = self._handoff_patch(
            worktree,
            effective_session,
            changed,
            max_chars=max_chars,
        )
        try:
            promotion = self._apply_verified_handoff(ctx, state, handoff)
            self._checkpoint(replace(ctx, run_started=time.monotonic()), state)
        except ServiceTierError as exc:
            self._record_failure_best_effort(
                ctx, state, exc.category, terminal=not persistent_session
            )
            raise
        resumable = persistent_session and status == "INCOMPLETE"
        verification_summary = (
            f"PASSED:verified_epoch={state.last_verified_epoch}"
            if status == "SUCCESS"
            else (
                f"INCOMPLETE:{failure_category or 'incomplete'}"
                if resumable
                else f"FAILED:{failure_category or 'incomplete'}"
            )
        )
        authoritative: JsonObject = {
            "status": status,
            "session_id": effective_session,
            "repo_id": repo_id,
            "model_id": route.model_id,
            "effective_lane": lane.value,
            "content": authoritative_content,
            "changed_paths": changed,
            "verification": state.validation_history[-1] if state.validation_history else None,
            "validation_history": state.validation_history,
            "unresolved_failures": state.unresolved_failures,
            "evidence_refs": state.evidence_refs,
            "checkpoint_path": str(self.checkpoints.path(effective_session)),
            "handoff": handoff,
            "promotion": promotion,
            "truncated": truncated,
            "max_chars": max_chars,
            "elapsed_seconds": round(time.monotonic() - run_started, 3),
            "telemetry": state.telemetry.as_dict(),
        }
        artifacts: dict[str, Path | bytes | str] = {
            "result.json": json.dumps(
                authoritative, indent=2, sort_keys=True, ensure_ascii=False
            )
            + "\n",
            "checkpoint.json": self.checkpoints.path(effective_session),
        }
        patch_path = handoff.get("patch_path")
        if isinstance(patch_path, str):
            artifacts["handoff.patch"] = Path(patch_path)
        artifacts.update(self.results.staging_artifacts(effective_session))
        try:
            delivery = self.results.persist(
                user_id=user_id,
                repo_id=repo_id,
                session_id=effective_session,
                status=status,
                summary=final_result,
                artifacts=artifacts,
            )
        except ResultStoreError as exc:
            self._record_failure_best_effort(
                ctx, state, exc.category, terminal=not persistent_session
            )
            raise ServiceTierError(exc.category) from exc
        result_id = str(delivery["result_id"])
        self.tiered.sandboxes.record_result(
            effective_session,
            user_id=user_id,
            command_count=state.command_count,
            verification_summary=verification_summary,
            failed=status != "SUCCESS" and not resumable,
            terminal=not persistent_session,
            resumable=resumable,
            result_id=result_id,
        )
        cleanup = getattr(self.tiered.sandboxes, "authorize_terminal_cleanup", None)
        worktree_release = (
            {"action": "PRESERVED_FOR_CONTINUATION"}
            if persistent_session
            else (
                cleanup(effective_session, user_id=user_id, result_id=result_id)
                if callable(cleanup)
                else {"action": "NOT_SUPPORTED_BY_SANDBOX_ADAPTER"}
            )
        )
        self.results.clear_staging(effective_session)
        return {
            **authoritative,
            "content": final_result,
            "result_id": result_id,
            "result_artifacts": delivery["artifacts"],
            "delivery_status": "PAGED" if truncated else "BOUNDED",
            "worktree_release": worktree_release,
        }
