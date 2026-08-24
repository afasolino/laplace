"""Bounded Qwen repository-agent orchestration for Zetsu.

Request identity and execution state are explicit per run. Model summaries are
semantic aids only; exact resumable state is checkpointed independently.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess  # nosec B404 - argv is allowlisted and shell=False
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping, Sequence, cast

from .agent_sandbox import AgentSandboxError, AgentSandboxManager, AgentSessionBinding, AgentToolPolicy
from .personal_corpus import CorpusError, PersonalCorpusStore
from .repository_authorization import RepositoryAuthorizationError, validate_workspace_path
from .service_tiers import ModelLane, ServiceTierError, TieredServingService
from .zetsu_context import compact_personal_results

JsonObject = dict[str, object]

_ALLOWED_VERIFY_EXECUTABLES = frozenset({"pytest", "ruff", "mypy"})
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
    ) -> None:
        self.tiered = tiered
        self.corpus = corpus
        self.checkpoints = checkpoint_store or AgentCheckpointStore(
            tiered.sandboxes.sandbox_root / "zetsu_agent_checkpoints"
        )
        self._session_locks_guard = threading.Lock()
        self._session_locks: dict[str, threading.Lock] = {}

    def _session_lock(self, session_id: str) -> threading.Lock:
        with self._session_locks_guard:
            return self._session_locks.setdefault(session_id, threading.Lock())

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are the bounded local Qwen repository agent subordinate to Codex. Return exactly "
            "one JSON object selecting one action. Allowed actions: "
            '{"action":"search","query":"literal","glob":"*.py"}; '
            '{"action":"read","path":"relative/path"}; '
            '{"action":"retrieve","query":"owner-authorized knowledge query"}; '
            '{"action":"edit","path":"relative/path","old_text":"exact once","new_text":"replacement"}; '
            '{"action":"create","path":"relative/path","content":"text"}; '
            '{"action":"verify","argv":["pytest","tests/test_x.py","-q"]}; '
            '{"action":"finish","result":"concise verified result"}. '
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
    def _run_git(worktree: Path, argv: Sequence[str], *, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
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
        if not isinstance(pattern, str) or not pattern or len(pattern) > 200 or ".." in Path(pattern).parts:
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

    def _read(self, worktree: Path, action: Mapping[str, object]) -> str:
        target = self._relative_target(worktree, action.get("path"))
        if not target.is_file() or target.stat().st_size > 2_000_000:
            raise ServiceTierError("zetsu_agent_read_target_unavailable")
        try:
            text = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ServiceTierError("zetsu_agent_read_target_not_text") from exc
        return text[:_MAX_READ_CHARS]

    def _retrieve(self, ctx: AgentRunContext, state: AgentExecutionState, action: Mapping[str, object]) -> str:
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

    @staticmethod
    def _require_edit_policy(policy: AgentToolPolicy) -> None:
        if "apply_patch" not in policy.allowed_tools:
            raise ServiceTierError("zetsu_agent_edit_not_allowed")

    def _edit(self, ctx: AgentRunContext, action: Mapping[str, object]) -> str:
        self._require_edit_policy(ctx.binding.tool_policy)
        target = self._relative_target(ctx.worktree, action.get("path"))
        old = action.get("old_text")
        new = action.get("new_text")
        if not isinstance(old, str) or not old or not isinstance(new, str):
            raise ServiceTierError("zetsu_agent_edit_invalid")
        self._check_text(old, maximum=_MAX_NEW_FILE_CHARS)
        self._check_text(new, maximum=_MAX_NEW_FILE_CHARS)
        if not target.is_file():
            raise ServiceTierError("zetsu_agent_edit_target_unavailable")
        try:
            original = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ServiceTierError("zetsu_agent_edit_target_not_text") from exc
        if original.count(old) != 1:
            raise ServiceTierError("zetsu_agent_edit_anchor_not_unique")
        target.write_text(original.replace(old, new, 1), encoding="utf-8")
        check = self._run_git(ctx.worktree, ["diff", "--check"])
        if check.returncode != 0:
            target.write_text(original, encoding="utf-8")
            raise ServiceTierError("zetsu_agent_edit_diff_check_failed", {"stderr": check.stderr[-2_000:]})
        return f"EDITED:{target.relative_to(ctx.worktree).as_posix()}"

    def _create(self, ctx: AgentRunContext, action: Mapping[str, object]) -> str:
        self._require_edit_policy(ctx.binding.tool_policy)
        target = self._relative_target(ctx.worktree, action.get("path"))
        content = action.get("content")
        if not isinstance(content, str):
            raise ServiceTierError("zetsu_agent_create_invalid")
        self._check_text(content, maximum=_MAX_NEW_FILE_CHARS)
        if target.exists() or not target.parent.is_dir():
            raise ServiceTierError("zetsu_agent_create_target_invalid")
        try:
            with target.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
        except OSError as exc:
            raise ServiceTierError("zetsu_agent_create_failed") from exc
        return f"CREATED:{target.relative_to(ctx.worktree).as_posix()}"

    @classmethod
    def _verify_argv(cls, worktree: Path, value: object) -> list[str]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
            raise ServiceTierError("zetsu_agent_verify_argv_invalid")
        argv = [str(item) for item in value]
        if (
            len(argv) > 64
            or any(not item or len(item) > 1_000 or "\x00" in item for item in argv)
        ):
            raise ServiceTierError("zetsu_agent_verify_argv_invalid")
        executable = Path(argv[0]).name
        if (
            argv[0] != executable
            or "/" in argv[0]
            or "\\" in argv[0]
            or executable not in _ALLOWED_VERIFY_EXECUTABLES
        ):
            raise ServiceTierError("zetsu_agent_verify_command_forbidden")

        lowered = [item.casefold() for item in argv[1:]]
        forbidden_options = {
            "-c",
            "-p",
            "-o",
            "--override-ini",
            "--basetemp",
            "--confcutdir",
            "--config",
            "--config-file",
            "--cache-dir",
            "--custom-typeshed-dir",
            "--python-executable",
            "--junit-xml",
            "--junitxml",
            "--output-file",
            "--pyargs",
            "--rootdir",
        }
        if any(
            item in forbidden_options
            or any(item.startswith(prefix + "=") for prefix in forbidden_options)
            for item in lowered
        ):
            raise ServiceTierError("zetsu_agent_verify_command_forbidden")
        if executable == "ruff" and (
            "--fix" in lowered
            or "--unsafe-fixes" in lowered
            or "format" in lowered
            or (argv[1:] and not argv[1].startswith("-") and argv[1] != "check")
        ):
            raise ServiceTierError("zetsu_agent_verify_command_forbidden")

        # Non-option arguments that denote repository targets must remain inside the
        # authorized worktree. Explicit node IDs such as tests/test_x.py::test_y are
        # checked using their path component. Known selector values (-k/-m) are not paths.
        skip_next_value = False
        for index, item in enumerate(argv[1:], start=1):
            if skip_next_value:
                skip_next_value = False
                continue
            lowered_item = item.casefold()
            if lowered_item in {"-k", "-m"} and executable == "pytest":
                skip_next_value = True
                continue
            if item.startswith("@"):
                raise ServiceTierError("zetsu_agent_verify_command_forbidden")
            candidate = item.split("::", 1)[0]
            if item.startswith("-"):
                if "=" in item:
                    option_value = item.split("=", 1)[1]
                    if Path(option_value).is_absolute() or ".." in Path(option_value).parts:
                        raise ServiceTierError("zetsu_agent_verify_path_forbidden")
                continue
            if executable == "ruff" and index == 1 and item == "check":
                continue
            path_value = Path(candidate)
            if path_value.is_absolute() or ".." in path_value.parts:
                raise ServiceTierError("zetsu_agent_verify_path_forbidden")
            # Verification targets are repository paths, not arbitrary installed modules.
            if candidate not in {"."} and not (
                "/" in candidate
                or "\\" in candidate
                or path_value.suffix in {".py", ".pyi"}
                or (worktree / candidate).exists()
            ):
                raise ServiceTierError("zetsu_agent_verify_target_invalid")
            cls._relative_target(worktree, candidate)
        if skip_next_value:
            raise ServiceTierError("zetsu_agent_verify_argv_invalid")
        return argv

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
    def _remove_resolved_verification_failures(
        state: AgentExecutionState, command_id: str
    ) -> None:
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
    def _read_file_tail(handle: object, *, limit: int = _MAX_VERIFY_CAPTURE_BYTES) -> str:
        file_handle = cast(object, handle)
        # TemporaryFile is seekable. Read only a bounded suffix to avoid loading noisy
        # verification output into coordinator memory.
        file_handle.seek(0, os.SEEK_END)  # type: ignore[attr-defined]
        size = file_handle.tell()  # type: ignore[attr-defined]
        file_handle.seek(max(0, size - limit), os.SEEK_SET)  # type: ignore[attr-defined]
        raw = file_handle.read()  # type: ignore[attr-defined]
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
            raise ServiceTierError(
                "zetsu_agent_verify_executable_missing", {"executable": argv[0]}
            )
        timeout = min(600.0, self._remaining_wall(ctx, state))
        deadline = time.monotonic() + timeout
        before_head, before_status, before_changed = self._worktree_state(ctx.worktree)
        returncode: int | None = None
        stdout = ""
        stderr = ""
        aborted_category: str | None = None
        with tempfile.TemporaryFile(mode="w+b") as stdout_file, tempfile.TemporaryFile(
            mode="w+b"
        ) as stderr_file:
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

        qualifies = self._verification_qualifies(argv, ctx.required_verification_argv) and not verification_mutated_worktree
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
            "validation_history": validation_summary,
            "unresolved_failures": state.unresolved_failures,
            "evidence_refs": state.evidence_refs,
            "next_state": state.next_state,
            "mutation_epoch": state.mutation_epoch,
            "last_verified_epoch": state.last_verified_epoch,
            "command_count": state.command_count,
        }
        return json.dumps(exact, sort_keys=True, ensure_ascii=False)

    def _messages(self, state: AgentExecutionState) -> tuple[dict[str, str], dict[str, str]]:
        prompt = (
            f"OBJECTIVE:\n{state.objective}\n\nEXACT STATE (authoritative):\n{self._state_digest(state)}\n\n"
            f"SEMANTIC SUMMARY:\n{state.summary or 'NONE'}\n\nRECENT OBSERVATIONS:\n"
            + ("\n\n".join(state.recent_observations[-_MAX_RECENT_OBSERVATIONS:]) if state.recent_observations else "NONE")
            + "\n\nChoose the next action."
        )
        return ({"role": "system", "content": self._system_prompt()}, {"role": "user", "content": prompt})

    @staticmethod
    def _approximate_active_tokens(messages: Sequence[Mapping[str, str]]) -> int:
        serialized = json.dumps(list(messages), sort_keys=True, ensure_ascii=False)
        return _rough_tokens(serialized)

    def _compact(self, ctx: AgentRunContext, state: AgentExecutionState, threshold: int) -> None:
        messages_before = self._messages(state)
        before = self._approximate_active_tokens(messages_before)
        exact_state = self._state_digest(state)
        self._ensure_active(ctx, state)
        result = self.tiered.chat(
            user_id=ctx.user_id,
            lane=ctx.lane,
            domain="json",
            session_id=ctx.session_id,
            messages=(
                {
                    "role": "system",
                    "content": (
                        "Return exactly one JSON object with string field summary. Compact only semantic "
                        "history. Preserve decisions, paths/symbols, unresolved reasoning and the next useful "
                        "action. Do not rewrite or infer the separately supplied exact state."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"OBJECTIVE:\n{state.objective}\n\nAUTHORITATIVE EXACT STATE:\n{exact_state}\n\n"
                        f"PRIOR SUMMARY:\n{state.summary}\n\nRECENT HISTORY:\n"
                        + "\n\n".join(state.recent_observations)
                    ),
                },
            ),
        )
        state.telemetry.add_usage(result, compaction=True)
        state.summary = self._summary_content(result)
        state.recent_observations = state.recent_observations[-3:]
        # The last reported prompt count described the pre-compaction/compaction prompt,
        # not the newly compacted active context. Do not reuse it as the next trigger.
        state.telemetry.last_model_reported_input_tokens = None
        after = self._approximate_active_tokens(self._messages(state))
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
            "consumed_wall_seconds": round(consumed, 6),
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
        if raw.get("repo_id") != ctx.repo_id or raw.get("base_revision") != ctx.binding.base_revision:
            raise ServiceTierError("zetsu_agent_checkpoint_repository_mismatch")
        if raw.get("objective") != instruction:
            raise ServiceTierError("zetsu_agent_resume_objective_mismatch")
        if (
            raw.get("step_limit") != ctx.max_steps
            or raw.get("max_chars") != ctx.max_chars
            or raw.get("compaction_ratio") != ctx.compaction_ratio
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

        def strings(name: str, *, maximum_items: int, maximum_chars: int) -> list[str]:
            value = raw.get(name)
            if (
                not isinstance(value, list)
                or len(value) > maximum_items
                or any(
                    not isinstance(item, str)
                    or len(item) > maximum_chars
                    or "\x00" in item
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
        unresolved_failures = strings(
            "unresolved_failures", maximum_items=16, maximum_chars=2_000
        )
        evidence_refs = strings("evidence_refs", maximum_items=64, maximum_chars=2_000)
        worktree_head = string("worktree_head", maximum=128, allow_empty=False)
        worktree_status_sha256 = string(
            "worktree_status_sha256", maximum=64, allow_empty=False
        )
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
        consumed_wall_seconds = number("consumed_wall_seconds")

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
            consumed_wall_seconds=consumed_wall_seconds,
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

    def _record_failure_best_effort(
        self, ctx: AgentRunContext, state: AgentExecutionState, category: str
    ) -> None:
        try:
            self.tiered.sandboxes.record_result(
                ctx.session_id,
                user_id=ctx.user_id,
                command_count=state.command_count,
                verification_summary=f"FAILED:{category}",
                failed=True,
            )
        except Exception:
            # Preserve the original failure/cancellation category.
            pass

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
    ) -> JsonObject:
        if session_id is None:
            return self._run_unlocked(
                user_id=user_id,
                repo_id=repo_id,
                instruction=instruction,
                lane=lane,
                session_id=None,
                max_steps=max_steps,
                max_chars=max_chars,
                compaction_ratio=compaction_ratio,
                verification_argv=verification_argv,
            )
        lock = self._session_lock(session_id)
        if not lock.acquire(blocking=False):
            raise ServiceTierError("zetsu_agent_session_busy")
        try:
            return self._run_unlocked(
                user_id=user_id,
                repo_id=repo_id,
                instruction=instruction,
                lane=lane,
                session_id=session_id,
                max_steps=max_steps,
                max_chars=max_chars,
                compaction_ratio=compaction_ratio,
                verification_argv=verification_argv,
            )
        finally:
            lock.release()

    def _run_unlocked(
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
    ) -> JsonObject:
        if lane not in {ModelLane.QUALITY, ModelLane.STANDARD}:
            raise ServiceTierError("zetsu_agent_lane_invalid")
        if not instruction.strip() or len(instruction) > 40_000:
            raise ServiceTierError("zetsu_agent_instruction_invalid")
        if not 1 <= max_steps <= 32 or not 512 <= max_chars <= 24_000:
            raise ServiceTierError("zetsu_agent_budget_invalid")
        if not 0.75 <= compaction_ratio <= 0.85:
            raise ServiceTierError("zetsu_agent_compaction_ratio_invalid")

        effective_session = session_id or f"zetsu-{uuid.uuid4().hex}"
        creating = session_id is None
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
            if isinstance(raw, Mapping) and isinstance(raw.get("consumed_wall_seconds"), (int, float)):
                prior_consumed = float(raw["consumed_wall_seconds"])
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
        )
        state = (
            AgentExecutionState(
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
            if creating
            else self._restore(ctx, instruction)
        )
        if creating:
            head, status_sha256, changed = self._worktree_state(worktree)
            state.worktree_head = head
            state.worktree_status_sha256 = status_sha256
            state.changed_paths = changed
        if state.step >= max_steps:
            raise ServiceTierError("zetsu_agent_step_budget_exhausted")

        self.tiered.sandboxes.start_task(
            effective_session,
            user_id=user_id,
            lane=lane.value,
            sanitized_model_name=route.model_id,
            instruction_digest=digest,
        )
        threshold = int(route.context_limit * compaction_ratio)
        status = "INCOMPLETE"
        final_result = ""
        failure_category: str | None = None
        try:
            for step in range(state.step + 1, max_steps + 1):
                state.step = step
                self._ensure_active(ctx, state)
                messages = self._messages(state)
                approximate = self._approximate_active_tokens(messages)
                reported = state.telemetry.last_model_reported_input_tokens or 0
                if max(approximate, reported) >= threshold:
                    self._compact(ctx, state, threshold)
                    self._checkpoint(ctx, state)
                    messages = self._messages(state)
                self._ensure_active(ctx, state)
                result = self.tiered.chat(
                    user_id=user_id,
                    lane=lane,
                    domain="json",
                    session_id=effective_session,
                    messages=messages,
                )
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
                        state.recent_observations.append(f"STEP {step} ACTION finish\n{observation}")
                        state.next_state = "verify_latest_mutation"
                        self._checkpoint(ctx, state)
                        continue
                    final_result = value.strip()
                    status = "SUCCESS"
                    state.next_state = "finished"
                    self._checkpoint(ctx, state)
                    break

                self._consume_tool_budget(ctx, state)
                if action_name == "search":
                    observation = self._search(ctx, state, action)
                elif action_name == "read":
                    observation = self._read(worktree, action)
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
            if status != "SUCCESS":
                failure_category = "max_steps_exhausted"
        except (AgentSandboxError, CorpusError) as exc:
            failure_category = getattr(exc, "category", type(exc).__name__)
            try:
                self._checkpoint(ctx, state)
            except Exception:
                pass
            self._record_failure_best_effort(ctx, state, str(failure_category))
            evidence = getattr(exc, "evidence", None)
            raise ServiceTierError(
                str(failure_category), evidence if isinstance(evidence, dict) else None
            ) from exc
        except (ServiceTierError, subprocess.TimeoutExpired) as exc:
            failure_category = getattr(exc, "category", "zetsu_agent_timeout")
            try:
                self._checkpoint(ctx, state)
            except Exception:
                pass
            self._record_failure_best_effort(ctx, state, str(failure_category))
            raise
        finally:
            elapsed = time.monotonic() - run_started
            state.consumed_wall_seconds += elapsed

        head, worktree_status_sha256, changed = self._worktree_state(worktree)
        state.worktree_head = head
        state.worktree_status_sha256 = worktree_status_sha256
        state.changed_paths = changed
        verification_summary = (
            f"PASSED:verified_epoch={state.last_verified_epoch}"
            if status == "SUCCESS"
            else f"FAILED:{failure_category or 'incomplete'}"
        )
        self.tiered.sandboxes.record_result(
            effective_session,
            user_id=user_id,
            command_count=state.command_count,
            verification_summary=verification_summary,
            failed=status != "SUCCESS",
        )
        if not final_result:
            final_result = "Maximum bounded agent steps reached before verified finish."
        truncated = len(final_result) > max_chars
        if truncated:
            final_result = final_result[: max_chars - 1].rstrip() + "…"
        return {
            "status": status,
            "session_id": effective_session,
            "repo_id": repo_id,
            "model_id": route.model_id,
            "effective_lane": lane.value,
            "content": final_result,
            "changed_paths": changed,
            "verification": state.validation_history[-1] if state.validation_history else None,
            "validation_history": state.validation_history,
            "unresolved_failures": state.unresolved_failures,
            "evidence_refs": state.evidence_refs,
            "checkpoint_path": str(self.checkpoints.path(effective_session)),
            "truncated": truncated,
            "max_chars": max_chars,
            "elapsed_seconds": round(time.monotonic() - run_started, 3),
            "telemetry": state.telemetry.as_dict(),
        }
