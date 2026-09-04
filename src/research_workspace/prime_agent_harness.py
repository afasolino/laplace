"""Prime Agent v0.9.1 harness for Laplace's selected local P8 model.

Prime supplies the mature outer agent loop (IPython, compaction, autonomous repair,
and bounded RLM recursion). Laplace remains authoritative for repository grants,
candidate observation, deterministic verification, evidence, and promotion.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess  # nosec B404 - fixed argv; no shell execution here
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence, TypeAlias, cast
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .agent_infrastructure.process import stop_process_tree
from .service_tiers import ModelRoute
import tempfile

JsonObject: TypeAlias = dict[str, object]
VerificationPlanLike = Sequence[tuple[str, Sequence[str]]]

PRIME_AGENT_REQUIRED_VERSION = (0, 9, 1)
PRIME_AGENT_RELEASE = "v0.9.1"
PRIME_AGENT_RELEASE_COMMIT = "81ae3cb34d27d38ee37f9e205a1e73694993b344"
PRIME_AGENT_PROVIDER_ID = "laplace-local-p8"
PRIME_AGENT_DIR_ENV = "PRIME_AGENT_CODING_AGENT_DIR"
PRIME_AGENT_EXECUTABLE_ENV = "LAPLACE_PRIME_AGENT_EXECUTABLE"
PRIME_AGENT_BACKEND_ENV = "LAPLACE_ZETSU_AGENT_BACKEND"
PRIME_AGENT_KERNEL_PYTHON_ENV = "LAPLACE_PRIME_AGENT_KERNEL_PYTHON"
_UPSTREAM_KERNEL_PYTHON_ENV = "PRIME_AGENT_KERNEL_PYTHON"
DEFAULT_ZETSU_ENDPOINT = "http://127.0.0.1:8765/mcp"
DEFAULT_ZETSU_TOKEN_ENV = "LAPLACE_ZETSU_TOKEN"
_BWRAP_ROOT = Path("/run/laplace-prime")
_BWRAP_WORKSPACE = _BWRAP_ROOT / "workspace"
_BWRAP_STATE = _BWRAP_ROOT / "state"
_BWRAP_KERNEL = _BWRAP_ROOT / "kernel"
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_VERSION_RE = re.compile(r"(?<!\d)(\d+)\.(\d+)\.(\d+)(?!\d)")
_MAX_EVENT_BYTES = 32 * 1024 * 1024
_MAX_EVENT_COUNT = 200_000
_MAX_STDERR_BYTES = 256 * 1024
_MAX_IPYTHON_CALLS = 100
_MAX_SESSION_ARTIFACT_BYTES = 8 * 1024 * 1024
_MAX_SESSION_FILES = 256
_SAFE_ENVIRONMENT_KEYS = (
    "PATH",
    "LANG",
    "LC_ALL",
    "TZ",
    "PYTHONUTF8",
    "PYTHONDONTWRITEBYTECODE",
    "PYTEST_ADDOPTS",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
)


class PrimeAgentHarnessError(RuntimeError):
    """Prime Agent failed a deterministic integration or confinement gate."""


@dataclass(frozen=True, slots=True)
class PrimeAgentProfile:
    provider_id: str
    model_id: str
    endpoint: str
    base_url: str
    context_window: int
    max_tokens: int


@dataclass(frozen=True, slots=True)
class PrimeToolExecution:
    tool_call_id: str
    tool_name: str
    args: object
    result: object
    is_error: bool

    @property
    def ipython_source(self) -> str | None:
        if self.tool_name != "ipython" or self.is_error or not isinstance(self.args, Mapping):
            return None
        for key in ("code", "cell", "source", "input"):
            value = self.args.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return None


@dataclass(frozen=True, slots=True)
class PrimeAgentRunResult:
    returncode: int
    final_text: str
    events: tuple[JsonObject, ...]
    tool_executions: tuple[PrimeToolExecution, ...]
    stdout: str
    stderr: str
    elapsed_seconds: float
    bwrap_used: bool
    rlm_child_usage_count: int = 0

    @property
    def successful_ipython_cells(self) -> tuple[str, ...]:
        return tuple(
            source
            for execution in self.tool_executions
            if (source := execution.ipython_source) is not None
        )

    @property
    def ipython_call_count(self) -> int:
        return sum(1 for item in self.tool_executions if item.tool_name == "ipython")

    @property
    def used_rlm(self) -> bool:
        admitted = any(
            "rlm(" in cell or "await rlm" in cell
            for cell in self.successful_ipython_cells
        )
        return admitted and self.rlm_child_usage_count > 0

    @property
    def used_zetsu_mcp(self) -> bool:
        return any(
            'mcp.call_tool("zetsu"' in cell
            or "mcp.call_tool('zetsu'" in cell
            for cell in self.successful_ipython_cells
        )

    @property
    def observed_models(self) -> tuple[str, ...]:
        return _assistant_identity(self.events, "model")

    @property
    def observed_providers(self) -> tuple[str, ...]:
        return _assistant_identity(self.events, "provider")

    @property
    def assistant_message_count(self) -> int:
        return sum(
            1
            for event in self.events
            if event.get("type") == "message_end"
            and isinstance(event.get("message"), Mapping)
            and cast(Mapping[str, object], event["message"]).get("role") == "assistant"
        )

    @property
    def usage_call_count(self) -> int:
        return sum(1 for _ in _message_usages(self.events))

    @property
    def turn_count(self) -> int:
        return sum(1 for event in self.events if event.get("type") == "turn_end")

    @property
    def compaction_count(self) -> int:
        return sum(
            1
            for event in self.events
            if event.get("type") == "compaction_end" and event.get("aborted") is not True
        )

    @property
    def usage(self) -> JsonObject:
        totals = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0}
        for usage in _message_usages(self.events):
            for key in totals:
                value = usage.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    totals[key] += value
        return cast(JsonObject, totals)

    def evidence(self) -> JsonObject:
        return {
            "release": PRIME_AGENT_RELEASE,
            "release_commit": PRIME_AGENT_RELEASE_COMMIT,
            "returncode": self.returncode,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "bwrap_used": self.bwrap_used,
            "event_count": len(self.events),
            "turn_count": self.turn_count,
            "assistant_message_count": self.assistant_message_count,
            "compaction_count": self.compaction_count,
            "ipython_call_count": self.ipython_call_count,
            "successful_ipython_cell_count": len(self.successful_ipython_cells),
            "rlm_observed": self.used_rlm,
            "rlm_child_usage_count": self.rlm_child_usage_count,
            "zetsu_mcp_observed": self.used_zetsu_mcp,
            "observed_models": list(self.observed_models),
            "observed_providers": list(self.observed_providers),
            "usage": self.usage,
            "final_text": self.final_text[-4_000:],
        }


def _assistant_identity(events: Sequence[Mapping[str, object]], key: str) -> tuple[str, ...]:
    values: list[str] = []
    for event in events:
        if event.get("type") != "message_end":
            continue
        message = event.get("message")
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        value = message.get(key)
        if isinstance(value, str) and value and value not in values:
            values.append(value)
    return tuple(values)


def _message_usages(events: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    values: list[Mapping[str, object]] = []
    for event in events:
        if event.get("type") != "message_end":
            continue
        message = event.get("message")
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        usage = message.get("usage")
        if isinstance(usage, Mapping):
            values.append(usage)
    return values


def _read_json_object(path: Path) -> JsonObject:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PrimeAgentHarnessError(f"invalid_json:{path}") from exc
    if not isinstance(value, dict):
        raise PrimeAgentHarnessError(f"json_object_required:{path}")
    return cast(JsonObject, value)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            os.chmod(temporary, 0o600)
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary, path)
        os.chmod(path, 0o600)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _require_loopback_http(url: str, *, label: str) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in _LOOPBACK_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise PrimeAgentHarnessError(f"{label}_loopback_http_required")
    return url.rstrip("/")


def _integer(value: object, *, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PrimeAgentHarnessError(f"invalid_{label}")
    return value


def load_selected_p8_profile(repository_root: Path) -> PrimeAgentProfile:
    """Resolve the selected quality route without duplicating model configuration."""

    root = repository_root.resolve()
    selected = _read_json_object(root / "configs/selected_serving_profiles.json")
    if selected.get("default_profile_id") != "P8_qwen38_w4a16_mtp":
        raise PrimeAgentHarnessError("selected_profile_is_not_p8")
    routes = selected.get("routes")
    quality = routes.get("quality") if isinstance(routes, Mapping) else None
    if not isinstance(quality, Mapping):
        raise PrimeAgentHarnessError("selected_quality_route_invalid")
    model_id = quality.get("model_id")
    endpoint = quality.get("endpoint")
    if model_id != "laplace-quality-qwen38-mtp8":
        raise PrimeAgentHarnessError("selected_quality_model_is_not_p8")
    if not isinstance(endpoint, str):
        raise PrimeAgentHarnessError("selected_quality_endpoint_invalid")
    endpoint = _require_loopback_http(endpoint, label="quality_endpoint")
    return PrimeAgentProfile(
        provider_id=PRIME_AGENT_PROVIDER_ID,
        model_id=cast(str, model_id),
        endpoint=endpoint,
        base_url=f"{endpoint}/v1",
        context_window=_integer(quality.get("context_limit"), label="context_limit"),
        max_tokens=_integer(quality.get("output_limit"), label="output_limit"),
    )


def profile_from_route(route: ModelRoute) -> PrimeAgentProfile:
    endpoint = _require_loopback_http(route.endpoint, label="quality_endpoint")
    if route.model_id != "laplace-quality-qwen38-mtp8":
        raise PrimeAgentHarnessError("prime_agent_route_is_not_p8")
    return PrimeAgentProfile(
        provider_id=PRIME_AGENT_PROVIDER_ID,
        model_id=route.model_id,
        endpoint=endpoint,
        base_url=f"{endpoint}/v1",
        context_window=route.context_limit,
        max_tokens=route.output_limit,
    )


def prime_models_payload(profile: PrimeAgentProfile) -> JsonObject:
    """Prime Agent v0.9.1 custom-provider configuration for local vLLM."""

    return {
        "providers": {
            profile.provider_id: {
                "baseUrl": profile.base_url,
                "api": "openai-completions",
                "apiKey": "local-only",
                "compat": {
                    "supportsDeveloperRole": False,
                    "supportsReasoningEffort": False,
                    "thinkingFormat": "qwen-chat-template",
                    "maxTokensField": "max_tokens",
                },
                "models": [
                    {
                        "id": profile.model_id,
                        "name": "Laplace Qwen3.8-27B P8",
                        "reasoning": True,
                        "input": ["text"],
                        "contextWindow": profile.context_window,
                        "maxTokens": profile.max_tokens,
                        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    }
                ],
            }
        }
    }


def prime_settings_payload(
    *,
    zetsu_endpoint: str = DEFAULT_ZETSU_ENDPOINT,
    token_env_var: str = DEFAULT_ZETSU_TOKEN_ENV,
    enable_zetsu_readonly: bool = True,
    rlm_max_depth: int = 1,
) -> JsonObject:
    """Create isolated Prime settings; recursive agent_task is never re-exported."""

    if not 0 <= rlm_max_depth <= 4:
        raise PrimeAgentHarnessError("prime_agent_rlm_depth_invalid")
    settings: JsonObject = {
        "rlmMaxDepth": rlm_max_depth,
        "autoRefine": {"enabled": False},
        "telemetry": {"enabled": False},
        "compaction": {"enabled": True, "agentCallable": True},
        "enableBuiltinSkills": False,
        "bundledSkills": {"websearch": False},
        "retry": {"enabled": True, "maxRetries": 2},
    }
    if enable_zetsu_readonly:
        endpoint = _require_loopback_http(zetsu_endpoint, label="zetsu_endpoint")
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{1,127}", token_env_var):
            raise PrimeAgentHarnessError("zetsu_token_env_invalid")
        settings["mcpServers"] = {
            "zetsu": {
                "type": "http",
                "url": endpoint,
                "bearerTokenEnvVar": token_env_var,
                "enabledTools": [
                    "search",
                    "get_evidence",
                    "project_context",
                    "experiment_context",
                ],
                "startupTimeoutMs": 20_000,
                "callTimeoutMs": 120_000,
            }
        }
    return settings


def prepare_prime_agent_state(
    repository_root: Path,
    state_root: Path,
    *,
    zetsu_endpoint: str = DEFAULT_ZETSU_ENDPOINT,
    token_env_var: str = DEFAULT_ZETSU_TOKEN_ENV,
    enable_zetsu_readonly: bool = True,
    rlm_max_depth: int = 1,
) -> PrimeAgentProfile:
    """Materialize qualification state below the canonical repository's .runtime."""

    repository = repository_root.resolve()
    state = state_root.resolve()
    runtime_root = (repository / ".runtime").resolve()
    try:
        state.relative_to(runtime_root)
    except ValueError as exc:
        raise PrimeAgentHarnessError("prime_state_must_be_under_repository_runtime") from exc
    profile = load_selected_p8_profile(repository)
    prepare_prime_runtime_state(
        profile,
        state,
        zetsu_endpoint=zetsu_endpoint,
        token_env_var=token_env_var,
        enable_zetsu_readonly=enable_zetsu_readonly,
        rlm_max_depth=rlm_max_depth,
    )
    return profile


def prepare_prime_runtime_state(
    profile: PrimeAgentProfile,
    state_root: Path,
    *,
    zetsu_endpoint: str = DEFAULT_ZETSU_ENDPOINT,
    token_env_var: str = DEFAULT_ZETSU_TOKEN_ENV,
    enable_zetsu_readonly: bool = False,
    rlm_max_depth: int = 1,
) -> None:
    """Create private state for a production Prime worker without touching ~/.prime."""

    state = state_root.resolve()
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(state, 0o700)
    for directory in ("agent", "sessions", "tmp", "home", "xdg-cache", "xdg-state", "xdg-runtime"):
        target = state / directory
        target.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(target, 0o700)
    _atomic_json(state / "agent/models.json", prime_models_payload(profile))
    _atomic_json(
        state / "agent/settings.json",
        prime_settings_payload(
            zetsu_endpoint=zetsu_endpoint,
            token_env_var=token_env_var,
            enable_zetsu_readonly=enable_zetsu_readonly,
            rlm_max_depth=rlm_max_depth,
        ),
    )


def parse_prime_version(text: str) -> tuple[int, int, int]:
    match = _VERSION_RE.search(text)
    if match is None:
        raise PrimeAgentHarnessError("prime_agent_version_unparseable")
    return cast(tuple[int, int, int], tuple(int(value) for value in match.groups()))


def resolve_prime_agent_executable(value: str | None = None) -> Path:
    requested = value or os.environ.get(PRIME_AGENT_EXECUTABLE_ENV, "prime-agent")
    candidate = Path(requested).expanduser()
    if candidate.is_absolute() or "/" in requested:
        absolute = Path(os.path.abspath(candidate))
        if not absolute.is_file() or not os.access(absolute, os.X_OK):
            raise PrimeAgentHarnessError("prime_agent_unavailable")
        return absolute
    located = shutil.which(requested)
    if located is None:
        raise PrimeAgentHarnessError("prime_agent_unavailable")
    absolute = Path(os.path.abspath(located))
    if not absolute.is_file() or not os.access(absolute, os.X_OK):
        raise PrimeAgentHarnessError("prime_agent_unavailable")
    return absolute

def resolve_prime_kernel_python(value: str | Path | None = None) -> Path:
    """Resolve an already-prepared upstream Prime kernel runtime, fail closed otherwise."""

    candidates: list[Path] = []
    if value is not None:
        candidates.append(Path(value).expanduser())
    else:
        configured = os.environ.get(PRIME_AGENT_KERNEL_PYTHON_ENV)
        upstream = os.environ.get(_UPSTREAM_KERNEL_PYTHON_ENV)
        if configured:
            candidates.append(Path(configured).expanduser())
        if upstream:
            candidates.append(Path(upstream).expanduser())
        candidates.append(Path.home() / ".prime" / "agent" / "kernel-venv" / "bin" / "python")
    for candidate in candidates:
        absolute = Path(os.path.abspath(candidate))
        if absolute.is_file() and os.access(absolute, os.X_OK):
            return absolute
    raise PrimeAgentHarnessError(
        "prime_agent_kernel_not_prepared; run Prime Agent once using its official bootstrap "
        "or set LAPLACE_PRIME_AGENT_KERNEL_PYTHON to a current prime-agent-runtime Python"
    )


def _prime_environment(
    base: Mapping[str, str] | None,
    *,
    enable_zetsu_readonly: bool,
    zetsu_token_env: str,
) -> dict[str, str]:
    """Build a secret-minimal process environment instead of inheriting the operator shell."""

    source = dict(base) if base is not None else os.environ
    env = {key: source[key] for key in _SAFE_ENVIRONMENT_KEYS if key in source}
    env.setdefault("PATH", os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"))
    env.setdefault("LANG", "C.UTF-8")
    env.setdefault("TZ", "UTC")
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    pytest_addopts = env.get("PYTEST_ADDOPTS", "").strip()
    if "no:cacheprovider" not in pytest_addopts:
        env["PYTEST_ADDOPTS"] = (pytest_addopts + " -p no:cacheprovider").strip()
    env["PI_OFFLINE"] = "1"
    env["PI_SKIP_VERSION_CHECK"] = "1"
    if enable_zetsu_readonly:
        token = os.environ.get(zetsu_token_env)
        if not token:
            raise PrimeAgentHarnessError(f"zetsu_token_missing:{zetsu_token_env}")
        env[zetsu_token_env] = token
    return env


def _count_child_usage_attributions(session_root: Path) -> int:
    """Count durable Prime child-usage records without trusting prompt/source text alone."""

    if not session_root.is_dir():
        return 0

    total_bytes = 0
    file_count = 0
    count = 0

    for path in sorted(session_root.rglob("*.jsonl")):
        if file_count >= _MAX_SESSION_FILES:
            raise PrimeAgentHarnessError(
                "prime_agent_session_file_count_exceeded"
            )

        try:
            size = path.stat().st_size
        except OSError as exc:
            raise PrimeAgentHarnessError(
                "prime_agent_session_artifact_unavailable"
            ) from exc

        total_bytes += size
        file_count += 1

        if total_bytes > _MAX_SESSION_ARTIFACT_BYTES:
            raise PrimeAgentHarnessError(
                "prime_agent_session_artifact_limit_exceeded"
            )

        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        value: object = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise PrimeAgentHarnessError(
                            "prime_agent_session_json_invalid"
                        ) from exc

                    if not isinstance(value, Mapping):
                        raise PrimeAgentHarnessError(
                            "prime_agent_session_entry_invalid"
                        )

                    entry_type = value.get("type")
                    if not isinstance(entry_type, str) or not entry_type:
                        raise PrimeAgentHarnessError(
                            "prime_agent_session_entry_invalid"
                        )

                    if entry_type != "child_usage_attributed":
                        continue

                    target_id = value.get("targetId")
                    child_usage = value.get("childUsage")
                    aggregate_usage = value.get("aggregateUsage")

                    if (
                        not isinstance(target_id, str)
                        or not target_id
                        or not isinstance(child_usage, Mapping)
                        or not isinstance(aggregate_usage, Mapping)
                    ):
                        raise PrimeAgentHarnessError(
                            "prime_agent_child_usage_attribution_invalid"
                        )

                    count += 1

        except UnicodeDecodeError as exc:
            raise PrimeAgentHarnessError(
                "prime_agent_session_json_invalid"
            ) from exc
        except OSError as exc:
            raise PrimeAgentHarnessError(
                "prime_agent_session_artifact_unavailable"
            ) from exc

    return count

def require_prime_agent_version(
    executable: Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> str:
    try:
        completed = subprocess.run(  # nosec B603
            [str(executable), "--version"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
            env=dict(environment) if environment is not None else None,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PrimeAgentHarnessError("prime_agent_unavailable") from exc
    rendered = (
        completed.stdout + b"\n" + completed.stderr
    ).decode("utf-8", errors="replace").strip()
    if completed.returncode != 0:
        raise PrimeAgentHarnessError("prime_agent_version_failed")
    if parse_prime_version(rendered) != PRIME_AGENT_REQUIRED_VERSION:
        raise PrimeAgentHarnessError("prime_agent_version_not_pinned_0_9_1")
    return rendered

def probe_local_model(profile: PrimeAgentProfile, *, timeout_seconds: float = 10.0) -> JsonObject:
    request = Request(f"{profile.base_url}/models", method="GET")
    try:
        # nosec B310 - profile construction admits loopback HTTP only.
        with urlopen(request, timeout=timeout_seconds) as response:
            payload: object = json.loads(response.read(1_000_000).decode("utf-8"))
    except Exception as exc:
        raise PrimeAgentHarnessError("prime_agent_p8_endpoint_unavailable") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise PrimeAgentHarnessError("prime_agent_p8_models_response_invalid")
    model_ids = {
        item.get("id")
        for item in cast(list[object], payload["data"])
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    if profile.model_id not in model_ids:
        raise PrimeAgentHarnessError("prime_agent_p8_model_not_served")
    return cast(JsonObject, payload)


def _extract_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        direct = value.get("text")
        if isinstance(direct, str):
            return direct
        return _extract_text(value.get("content"))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return "".join(_extract_text(item) for item in value)
    return ""


def parse_prime_events(
    stdout: str,
) -> tuple[tuple[JsonObject, ...], str, tuple[PrimeToolExecution, ...]]:
    """Parse v0.9.1 JSONL and count tools only after tool_execution_end."""

    events: list[JsonObject] = []
    final_text = ""
    started: dict[str, tuple[str, object]] = {}
    executions: list[PrimeToolExecution] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            value: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PrimeAgentHarnessError(
                "prime_agent_event_json_invalid"
            ) from exc
        if not isinstance(value, dict):
            raise PrimeAgentHarnessError(
                "prime_agent_event_object_required"
            )
        if len(events) >= _MAX_EVENT_COUNT:
            raise PrimeAgentHarnessError("prime_agent_event_count_exceeded")
        event = cast(JsonObject, value)
        events.append(event)
        event_type = event.get("type")
        if event_type == "message_end":
            message = event.get("message")
            if isinstance(message, Mapping) and message.get("role") == "assistant":
                text = _extract_text(message).strip()
                if text:
                    final_text = text
        elif event_type == "tool_execution_start":
            call_id = event.get("toolCallId")
            tool_name = event.get("toolName")
            if isinstance(call_id, str) and isinstance(tool_name, str):
                started[call_id] = (tool_name, event.get("args"))
        elif event_type == "tool_execution_end":
            call_id = event.get("toolCallId")
            tool_name = event.get("toolName")
            if not isinstance(call_id, str) or not isinstance(tool_name, str):
                continue
            prior = started.pop(call_id, None)
            args = prior[1] if prior is not None and prior[0] == tool_name else None
            executions.append(
                PrimeToolExecution(
                    tool_call_id=call_id,
                    tool_name=tool_name,
                    args=args,
                    result=event.get("result"),
                    is_error=event.get("isError") is True,
                )
            )
    return tuple(events), final_text, tuple(executions)


def _bounded_read(path: Path, maximum: int) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise PrimeAgentHarnessError("prime_agent_output_unavailable") from exc
    if size > maximum:
        raise PrimeAgentHarnessError("prime_agent_output_limit_exceeded")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise PrimeAgentHarnessError("prime_agent_output_unavailable") from exc


def _bwrap_argv(
    *,
    bwrap: Path,
    workspace: Path,
    state_root: Path,
    kernel_python: Path,
    command: Sequence[str],
    hide_paths: Sequence[Path],
) -> list[str]:
    """Confine writes with Bubblewrap while retaining the host loopback network."""

    kernel_root = kernel_python.parent.parent
    kernel_relative = kernel_python.relative_to(kernel_root)
    sandbox_kernel_python = str(_BWRAP_KERNEL / kernel_relative)

    project_prime = workspace / ".prime"
    if project_prime.is_symlink():
        raise PrimeAgentHarnessError(
            "prime_agent_project_configuration_unsafe"
        )
    if project_prime.exists() and not project_prime.is_dir():
        raise PrimeAgentHarnessError(
            "prime_agent_project_configuration_unsafe"
        )
    mask_project_prime = project_prime.is_dir()

    argv = [
        str(bwrap),
        "--die-with-parent",
        "--new-session",
        "--ro-bind",
        "/",
        "/",
        "--dev-bind",
        "/dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/run",
        "--dir",
        str(_BWRAP_ROOT),
        "--dir",
        str(_BWRAP_WORKSPACE),
        "--dir",
        str(_BWRAP_STATE),
        "--dir",
        str(_BWRAP_KERNEL),
        "--bind",
        str(workspace),
        str(_BWRAP_WORKSPACE),
        "--ro-bind",
        str(workspace / ".git"),
        str(_BWRAP_WORKSPACE / ".git"),
        "--bind",
        str(state_root),
        str(_BWRAP_STATE),
        "--ro-bind",
        str(kernel_root),
        str(_BWRAP_KERNEL),
    ]

    if mask_project_prime:
        argv.extend(
            [
                "--tmpfs",
                str(_BWRAP_WORKSPACE / ".prime"),
            ]
        )

    candidates: list[Path] = []
    seen: set[Path] = set()
    forbidden_masks = {
        Path("/"),
        Path("/dev"),
        Path("/proc"),
        Path("/run"),
    }
    for hidden in hide_paths:
        target = hidden.resolve()
        if target in forbidden_masks:
            raise PrimeAgentHarnessError("prime_agent_hide_path_unsafe")
        if target in seen or not target.is_dir() or target in {workspace, state_root}:
            continue
        seen.add(target)
        candidates.append(target)

    # If both a directory and one of its descendants were requested, masking
    # the parent is sufficient and avoids constructing a mount below an
    # already-hidden subtree.
    masked: list[Path] = []
    for target in sorted(candidates, key=lambda item: len(item.parts)):
        if any(parent == target or parent in target.parents for parent in masked):
            continue
        masked.append(target)
        argv.extend(["--tmpfs", str(target)])

    argv.extend(
        [
            "--chdir",
            str(_BWRAP_WORKSPACE),
            "--setenv",
            "HOME",
            str(_BWRAP_STATE / "home"),
            "--setenv",
            "XDG_CACHE_HOME",
            str(_BWRAP_STATE / "xdg-cache"),
            "--setenv",
            "XDG_STATE_HOME",
            str(_BWRAP_STATE / "xdg-state"),
            "--setenv",
            "XDG_RUNTIME_DIR",
            str(_BWRAP_STATE / "xdg-runtime"),
            "--setenv",
            "TMPDIR",
            str(_BWRAP_STATE / "tmp"),
            "--setenv",
            _UPSTREAM_KERNEL_PYTHON_ENV,
            sandbox_kernel_python,
            "--",
            *command,
        ]
    )
    return argv

def _gate_spec(
    *,
    state_root: Path,
    workspace: Path,
    plan: VerificationPlanLike,
    environment: Mapping[str, str],
) -> tuple[Path, str]:
    rendered_plan = [
        {"cwd": cwd, "argv": list(argv)}
        for cwd, argv in plan
    ]
    host_spec = state_root / "gate.json"
    _atomic_json(
        host_spec,
        {
            "workspace": str(_BWRAP_WORKSPACE),
            "scratch_root": str(_BWRAP_STATE / "tmp"),
            "plan": rendered_plan,
            "environment": dict(environment),
        },
    )
    gate_script = Path(__file__).with_name("prime_agent_gate.py").resolve()
    command = shlex.join(
        [
            sys.executable,
            str(gate_script),
            "--spec",
            str(_BWRAP_STATE / "gate.json"),
        ]
    )
    return host_spec, command

def run_prime_agent(
    *,
    executable: Path,
    profile: PrimeAgentProfile,
    state_root: Path,
    workspace: Path,
    prompt: str,
    timeout_seconds: float,
    thinking: str = "high",
    verification_plan: VerificationPlanLike | None = None,
    verification_environment: Mapping[str, str] | None = None,
    enable_zetsu_readonly: bool = False,
    zetsu_endpoint: str = DEFAULT_ZETSU_ENDPOINT,
    zetsu_token_env: str = DEFAULT_ZETSU_TOKEN_ENV,
    autonomous_max_turns: int = 12,
    require_bwrap: bool = True,
    hide_paths: Sequence[Path] = (),
    poll_guard: Callable[[], None] | None = None,
    kernel_python: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> PrimeAgentRunResult:
    """Run one bounded Prime task; no native-agent fallback is performed."""

    state = state_root.resolve()
    work = workspace.resolve()
    if not work.is_dir() or not (work / ".git").exists():
        raise PrimeAgentHarnessError("prime_agent_git_workspace_required")
    if not prompt.strip() or not 1 <= autonomous_max_turns <= 32:
        raise PrimeAgentHarnessError("prime_agent_request_invalid")
    resolved_kernel = (
        resolve_prime_kernel_python(kernel_python)
        if require_bwrap or kernel_python is not None
        else None
    )
    prepare_prime_runtime_state(
        profile,
        state,
        enable_zetsu_readonly=enable_zetsu_readonly,
        zetsu_endpoint=zetsu_endpoint,
        token_env_var=zetsu_token_env,
        rlm_max_depth=1,
    )
    command = [
        str(executable),
        "--mode",
        "json",
        "--provider",
        profile.provider_id,
        "--model",
        profile.model_id,
        "--thinking",
        thinking,
        "--session-dir",
        (
            str(_BWRAP_STATE / "sessions")
            if require_bwrap
            else str(state / "sessions")
        ),
        "--offline",
        "--autonomous",
        "--autonomous-gate-retries",
        "2",
        "--autonomous-max-continuations",
        "3",
        "--autonomous-max-turns",
        str(autonomous_max_turns),
        "--autonomous-max-tokens",
        "80000",
        "--autonomous-timeout-ms",
        str(max(1_000, int(timeout_seconds * 1000))),
        "--tools",
        "ipython",
        "--no-extensions",
        "--no-skills",
        "--no-prompt-templates",
        "--no-themes",
        "--no-context-files",
    ]
    if verification_plan is not None:
        if verification_environment is None:
            raise PrimeAgentHarnessError("prime_agent_gate_environment_required")
        _, gate_command = _gate_spec(
            state_root=state,
            workspace=work,
            plan=verification_plan,
            environment=verification_environment,
        )
        command.extend(["--autonomous-gate", gate_command])
    command.append(prompt)

    env = _prime_environment(
        environment,
        enable_zetsu_readonly=enable_zetsu_readonly,
        zetsu_token_env=zetsu_token_env,
    )

    prime_bin = str(executable.parent)
    existing_path = env.get("PATH", "")
    path_entries = existing_path.split(os.pathsep) if existing_path else []
    if prime_bin not in path_entries:
        env["PATH"] = (
            f"{prime_bin}{os.pathsep}{existing_path}"
            if existing_path
            else prime_bin
        )

    version_environment = dict(env)
    version_environment[PRIME_AGENT_DIR_ENV] = str(state / "agent")
    version_environment["HOME"] = str(state / "home")
    version_environment["XDG_CACHE_HOME"] = str(state / "xdg-cache")
    version_environment["XDG_STATE_HOME"] = str(state / "xdg-state")
    version_environment["XDG_RUNTIME_DIR"] = str(state / "xdg-runtime")
    version_environment["TMPDIR"] = str(state / "tmp")
    require_prime_agent_version(executable, environment=version_environment)
    if require_bwrap:
        env[PRIME_AGENT_DIR_ENV] = str(_BWRAP_STATE / "agent")
        env["HOME"] = str(_BWRAP_STATE / "home")
        env["XDG_CACHE_HOME"] = str(_BWRAP_STATE / "xdg-cache")
        env["XDG_STATE_HOME"] = str(_BWRAP_STATE / "xdg-state")
        env["XDG_RUNTIME_DIR"] = str(_BWRAP_STATE / "xdg-runtime")
        env["TMPDIR"] = str(_BWRAP_STATE / "tmp")
    else:
        env[PRIME_AGENT_DIR_ENV] = str(state / "agent")
        env["HOME"] = str(state / "home")
        env["XDG_CACHE_HOME"] = str(state / "xdg-cache")
        env["XDG_STATE_HOME"] = str(state / "xdg-state")
        env["XDG_RUNTIME_DIR"] = str(state / "xdg-runtime")
        env["TMPDIR"] = str(state / "tmp")
    if resolved_kernel is not None and not require_bwrap:
        env[_UPSTREAM_KERNEL_PYTHON_ENV] = str(resolved_kernel)

    bwrap_used = False
    argv = command
    cwd = work
    if require_bwrap:
        located = shutil.which("bwrap")
        if located is None:
            raise PrimeAgentHarnessError("prime_agent_bwrap_required")
        argv = _bwrap_argv(
            bwrap=Path(located).resolve(),
            workspace=work,
            state_root=state,
            kernel_python=cast(Path, resolved_kernel),
            command=command,
            hide_paths=hide_paths,
        )
        cwd = Path("/")
        bwrap_used = True

    stdout_path = state / "last-run.events.jsonl"
    stderr_path = state / "last-run.stderr.log"
    started = time.monotonic()
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        try:
            process = subprocess.Popen(  # nosec B603
                argv,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=True,
                close_fds=True,
            )
        except OSError as exc:
            raise PrimeAgentHarnessError("prime_agent_launch_failed") from exc
        deadline = started + timeout_seconds
        while process.poll() is None:
            if poll_guard is not None:
                try:
                    poll_guard()
                except BaseException:
                    stop_process_tree(process)
                    raise
            if time.monotonic() >= deadline:
                stop_process_tree(process)
                raise PrimeAgentHarnessError("prime_agent_task_timeout")
            try:
                stdout_size = stdout_path.stat().st_size
                stderr_size = stderr_path.stat().st_size
            except OSError:
                stdout_size = stderr_size = 0
            if stdout_size > _MAX_EVENT_BYTES or stderr_size > _MAX_STDERR_BYTES:
                stop_process_tree(process)
                raise PrimeAgentHarnessError("prime_agent_output_limit_exceeded")
            time.sleep(0.10)
        returncode = int(process.returncode or 0)

    stdout_raw = _bounded_read(stdout_path, _MAX_EVENT_BYTES)
    stderr_raw = _bounded_read(stderr_path, _MAX_STDERR_BYTES)
    stdout = stdout_raw.decode("utf-8", errors="replace")
    stderr = stderr_raw.decode("utf-8", errors="replace")
    events, final_text, executions = parse_prime_events(stdout)
    if sum(1 for item in executions if item.tool_name == "ipython") > _MAX_IPYTHON_CALLS:
        raise PrimeAgentHarnessError("prime_agent_ipython_budget_exceeded")
    result = PrimeAgentRunResult(
        returncode=returncode,
        final_text=final_text,
        events=events,
        tool_executions=executions,
        stdout=stdout,
        stderr=stderr,
        elapsed_seconds=max(0.0, time.monotonic() - started),
        bwrap_used=bwrap_used,
        rlm_child_usage_count=_count_child_usage_attributions(state / "sessions"),
    )
    if result.returncode != 0:
        lowered = result.stderr.casefold()
        if "first-time setup needs internet" in lowered or "prime-agent-runtime" in lowered:
            raise PrimeAgentHarnessError("prime_agent_kernel_not_prepared")
        return result
    if not result.observed_models or result.observed_models != (profile.model_id,):
        raise PrimeAgentHarnessError("prime_agent_model_identity_mismatch")
    if not result.observed_providers or result.observed_providers != (profile.provider_id,):
        raise PrimeAgentHarnessError("prime_agent_provider_identity_mismatch")
    return result


def repository_agent_backend(requested: str | None = None) -> str:
    value = (
        requested
        if requested is not None
        else os.environ.get(PRIME_AGENT_BACKEND_ENV, "native")
    ).strip().casefold()
    if value not in {"native", "prime"}:
        raise PrimeAgentHarnessError("prime_agent_backend_invalid")
    return value
