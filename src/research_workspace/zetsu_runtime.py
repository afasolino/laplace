"""Owned local lifecycle orchestration for the Zetsu production stack."""

from __future__ import annotations

import json
import hashlib
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping

from .zetsu_config import DEFAULT_TOKEN_ENV, ZetsuConfigError


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    argv: tuple[str, ...]
    environment: Mapping[str, str]
    log_path: Path
    probe_url: str
    expected_model: str | None


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    proc_start_ticks: int
    process_group_id: int
    session_id: int
    command_sha256: str


def default_state_root() -> Path:
    configured = os.environ.get("LAPLACE_STATE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    xdg_state = os.environ.get("XDG_STATE_HOME")
    base = Path(xdg_state).expanduser() if xdg_state else Path.home() / ".local/state"
    return (base / "laplace").resolve()


def bearer_token_file(state_root: Path) -> Path:
    return state_root.resolve() / "auth/bearer_tokens.json"


def load_local_plus_token(state_root: Path) -> str:
    """Read the local Plus credential without ever serializing it into a result."""

    path = bearer_token_file(state_root)
    try:
        metadata = path.stat()
    except OSError as exc:
        raise ZetsuConfigError("local_bearer_token_file_missing") from exc
    if metadata.st_mode & 0o077:
        raise ZetsuConfigError("local_bearer_token_file_permissions")
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ZetsuConfigError("local_bearer_token_file_invalid") from exc
    tokens = raw.get("tokens") if isinstance(raw, dict) else None
    if not isinstance(tokens, dict):
        raise ZetsuConfigError("local_bearer_token_file_invalid")
    matches = [
        token
        for token, binding in tokens.items()
        if isinstance(token, str)
        and isinstance(binding, dict)
        and binding.get("user_id") == "plus-local"
        and binding.get("capability_tier") == "plus"
    ]
    if len(matches) != 1:
        raise ZetsuConfigError("local_plus_token_missing_or_ambiguous")
    return matches[0]


def _load_routes(repository: Path) -> tuple[str, str]:
    path = repository / "configs/selected_serving_profiles.json"
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ZetsuConfigError("selected_serving_profiles_invalid") from exc
    routes = raw.get("routes") if isinstance(raw, dict) else None
    if not isinstance(routes, dict):
        raise ZetsuConfigError("selected_serving_profiles_invalid")
    quality = routes.get("quality")
    economy = routes.get("economy")
    if not isinstance(quality, dict) or not isinstance(economy, dict):
        raise ZetsuConfigError("selected_serving_profiles_invalid")
    quality_model = quality.get("model_id")
    economy_model = economy.get("model_id")
    if not isinstance(quality_model, str) or not isinstance(economy_model, str):
        raise ZetsuConfigError("selected_serving_profiles_invalid")
    return quality_model, economy_model


def build_service_specs(
    repository: Path,
    state_root: Path,
    *,
    python: Path,
    vllm: Path,
    ffmpeg_lib: Path,
    codev_enabled: bool = True,
) -> tuple[ServiceSpec, ...]:
    """Build the requested immutable runtime topology."""

    repo = repository.resolve()
    state = state_root.resolve()
    quality_model, economy_model = _load_routes(repo)
    common_environment = {
        "PYTHONPATH": str(repo / "src"),
        "LAPLACE_CONTROL_PLANE_PYTHON": str(python),
        "LAPLACE_VLLM_EXECUTABLE": str(vllm),
        "LAPLACE_FFMPEG_LIBRARY_PATH": str(ffmpeg_lib),
    }
    logs = state / "logs"
    token_path = bearer_token_file(state)
    codev = ServiceSpec(
            name="codev",
            argv=(
                str(python),
                str(repo / "scripts/run_codev_service.py"),
                "--repository-root",
                str(repo),
            ),
            environment=common_environment,
            log_path=logs / "zetsu-codev.log",
            probe_url="http://127.0.0.1:8103/v1/models",
            expected_model=economy_model,
        )
    qwen = ServiceSpec(
            name="qwen",
            argv=(
                str(python),
                str(repo / "scripts/run_selected_quality_service.py"),
                "--repository-root",
                str(repo),
                "--state-root",
                str(state / "tiered_serving/profile_runtime"),
                "--vllm",
                str(vllm),
                "--ffmpeg-lib",
                str(ffmpeg_lib),
            ),
            environment=common_environment,
            log_path=logs / "zetsu-qwen.log",
            probe_url="http://127.0.0.1:8207/v1/models",
            expected_model=quality_model,
        )
    operator_argv = [
        str(python),
        "-m",
        "research_workspace.operator_server",
        "--repository-root",
        str(repo),
        "--state-root",
        str(state),
        "--host",
        "127.0.0.1",
        "--port",
        "8765",
        "--deployment-mode",
        "local",
        "--enable-bearer-api",
        "--bearer-token-file",
        str(token_path),
    ]
    if not codev_enabled:
        operator_argv.append("--codev-disabled")
    operator = ServiceSpec(
            name="operator",
            argv=tuple(operator_argv),
            environment=common_environment,
            log_path=logs / "zetsu-operator.log",
            probe_url="http://127.0.0.1:8765/api/v1/health",
            expected_model=None,
        )
    return (codev, qwen, operator) if codev_enabled else (qwen, operator)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _probe(spec: ServiceSpec, *, timeout: float = 2.0) -> bool:
    request = urllib.request.Request(spec.probe_url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
            raw: object = json.loads(response.read(1_000_000))
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
    except json.JSONDecodeError as exc:
        raise ZetsuConfigError(f"{spec.name}_endpoint_incompatible") from exc
    if not isinstance(raw, dict):
        raise ZetsuConfigError(f"{spec.name}_endpoint_incompatible")
    if spec.expected_model is None:
        return True
    data = raw.get("data")
    models = {
        str(item.get("id"))
        for item in data
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    } if isinstance(data, list) else set()
    if spec.expected_model not in models:
        raise ZetsuConfigError(f"{spec.name}_endpoint_wrong_model")
    return True


def _log_tail(path: Path, maximum: int = 1_200) -> str:
    try:
        value = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return value[-maximum:]


def _probe_zetsu(token: str, *, timeout: float = 5.0) -> None:
    request = urllib.request.Request(
        "http://127.0.0.1:8765/api/v1/zetsu/status",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
            raw: object = json.loads(response.read(1_000_000))
    except urllib.error.HTTPError as exc:
        raise ZetsuConfigError(f"operator_zetsu_http_{exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ZetsuConfigError(
            f"operator_zetsu_connection_failed:{type(exc).__name__}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ZetsuConfigError("operator_zetsu_response_invalid") from exc
    if not isinstance(raw, dict):
        raise ZetsuConfigError("operator_zetsu_response_invalid")


def _spawn(spec: ServiceSpec) -> subprocess.Popen[bytes]:
    environment = dict(os.environ)
    environment.update(spec.environment)
    spec.log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        spec.log_path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )
    os.chmod(spec.log_path, 0o600)
    with os.fdopen(descriptor, "ab", buffering=0) as log:
        return subprocess.Popen(  # nosec B603
            list(spec.argv),
            cwd=Path(spec.environment["PYTHONPATH"]).parent,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )


def _wait_ready(
    spec: ServiceSpec,
    process: subprocess.Popen[bytes],
    *,
    deadline: float,
) -> None:
    delay = 0.5
    while time.monotonic() < deadline:
        if _probe(spec):
            return
        returncode = process.poll()
        if returncode is not None:
            detail = (
                f"inspect_log:{spec.log_path}"
                if spec.name == "operator"
                else _log_tail(spec.log_path).replace("\n", " ")
            )
            raise ZetsuConfigError(
                f"{spec.name}_startup_failed:exit_{returncode}:{detail}"
            )
        time.sleep(delay)
        delay = min(delay * 1.5, 5.0)
    raise ZetsuConfigError(f"{spec.name}_startup_timeout")


def _boot_id() -> str:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except OSError as exc:
        raise ZetsuConfigError("proc_boot_identity_unavailable") from exc
    if not value:
        raise ZetsuConfigError("proc_boot_identity_unavailable")
    return value


def _process_identity(pid: int) -> ProcessIdentity:
    try:
        raw_stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        closing = raw_stat.rfind(")")
        fields = raw_stat[closing + 2 :].split()
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError as exc:
        raise ZetsuConfigError("process_identity_unavailable") from exc
    if closing < 1 or len(fields) < 20:
        raise ZetsuConfigError("process_identity_invalid")
    try:
        return ProcessIdentity(
            pid=pid,
            proc_start_ticks=int(fields[19]),
            process_group_id=int(fields[2]),
            session_id=int(fields[3]),
            command_sha256=hashlib.sha256(command).hexdigest(),
        )
    except ValueError as exc:
        raise ZetsuConfigError("process_identity_invalid") from exc


def _process_environment(pid: int) -> dict[str, str]:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return {}
    result: dict[str, str] = {}
    for item in raw.split(b"\0"):
        key, separator, value = item.partition(b"=")
        if separator:
            result[key.decode("utf-8", errors="replace")] = value.decode(
                "utf-8", errors="replace"
            )
    return result


def _runtime_owned_processes(record: Mapping[str, object]) -> dict[int, ProcessIdentity]:
    runtime_id = record.get("runtime_id")
    state_root = record.get("state_root")
    repository = record.get("repository")
    boot_id = record.get("boot_id")
    if not all(isinstance(item, str) and item for item in (runtime_id, state_root, repository)):
        raise ZetsuConfigError("zetsu_runtime_record_invalid")
    if boot_id != _boot_id():
        return {}
    owned: dict[int, ProcessIdentity] = {}
    try:
        candidates = tuple(Path("/proc").iterdir())
    except OSError as exc:
        raise ZetsuConfigError("proc_inventory_unavailable") from exc
    for candidate in candidates:
        if not candidate.name.isdigit():
            continue
        pid = int(candidate.name)
        environment = _process_environment(pid)
        if (
            environment.get("LAPLACE_ZETSU_RUNTIME_ID") != runtime_id
            or environment.get("LAPLACE_ZETSU_STATE_ROOT") != state_root
            or environment.get("LAPLACE_ZETSU_REPOSITORY") != repository
        ):
            continue
        try:
            owned[pid] = _process_identity(pid)
        except ZetsuConfigError:
            continue
    return owned


def _identity_still_owned(
    record: Mapping[str, object], identity: ProcessIdentity
) -> bool:
    current = _runtime_owned_processes(record).get(identity.pid)
    return current == identity


def _signal_owned(
    record: Mapping[str, object], identities: Mapping[int, ProcessIdentity], signum: int
) -> list[int]:
    signalled: list[int] = []
    for pid, identity in sorted(identities.items(), reverse=True):
        if not _identity_still_owned(record, identity):
            continue
        try:
            os.kill(pid, signum)
            signalled.append(pid)
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            raise ZetsuConfigError(f"owned_process_signal_denied:{pid}") from exc
    return signalled


def _wait_for_owned_exit(record: Mapping[str, object], *, timeout: float) -> dict[int, ProcessIdentity]:
    deadline = time.monotonic() + timeout
    remaining = _runtime_owned_processes(record)
    while remaining and time.monotonic() < deadline:
        time.sleep(0.10)
        remaining = _runtime_owned_processes(record)
    return remaining


def _terminate_runtime_record(
    record: Mapping[str, object],
    *,
    graceful_timeout: float = 45.0,
    escalation_timeout: float = 15.0,
) -> dict[str, object]:
    """Terminate every process carrying this exact unguessable runtime identity."""

    initial = _runtime_owned_processes(record)
    services = record.get("services")
    supervisor_pids = {
        int(item["process"]["pid"])
        for item in services.values()
        if isinstance(services, dict)
        and isinstance(item, dict)
        and isinstance(item.get("process"), dict)
        and isinstance(item["process"].get("pid"), int)
    } if isinstance(services, dict) else set()
    supervisors = {pid: identity for pid, identity in initial.items() if pid in supervisor_pids}
    signalled_term = _signal_owned(record, supervisors, signal.SIGTERM)
    remaining = _wait_for_owned_exit(
        record,
        timeout=min(15.0, graceful_timeout),
    )
    if remaining:
        signalled_term.extend(_signal_owned(record, remaining, signal.SIGTERM))
        remaining = _wait_for_owned_exit(record, timeout=graceful_timeout)
    signalled_kill: list[int] = []
    if remaining:
        signalled_kill = _signal_owned(record, remaining, signal.SIGKILL)
        remaining = _wait_for_owned_exit(record, timeout=escalation_timeout)
    return {
        "initial_owned_pids": sorted(initial),
        "sigterm_pids": sorted(set(signalled_term)),
        "sigkill_pids": sorted(set(signalled_kill)),
        "survivors": [asdict(item) for item in remaining.values()],
    }


def _validate_paths(specs: tuple[ServiceSpec, ...], *, vllm: Path, ffmpeg: Path) -> None:
    if not vllm.is_file() or not os.access(vllm, os.X_OK):
        raise ZetsuConfigError("certified_vllm_executable_missing")
    if not ffmpeg.is_dir():
        raise ZetsuConfigError("ffmpeg_runtime_missing")
    for spec in specs:
        executable = Path(spec.argv[0])
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ZetsuConfigError(f"{spec.name}_python_missing")
        if spec.name != "operator" and not Path(spec.argv[1]).is_file():
            raise ZetsuConfigError(f"{spec.name}_supervisor_missing")


def start_local_runtime(
    repository: Path,
    state_root: Path,
    *,
    timeout: float = 1_800.0,
    dry_run: bool = False,
    python: Path | None = None,
    vllm: Path | None = None,
    ffmpeg_lib: Path | None = None,
    codev_enabled: bool = True,
) -> dict[str, object]:
    """Start the selected local stack with inherited process-tree ownership."""

    repo = repository.resolve()
    state = state_root.resolve()
    # Preserve the virtual-environment launcher path. Resolving its symlink would
    # invoke the base interpreter without the environment's installed packages.
    selected_python = (python or repo / ".venv/bin/python").expanduser().absolute()
    selected_vllm = (vllm or repo / ".venv-vllm-cu129/bin/vllm").resolve()
    selected_ffmpeg = (ffmpeg_lib or repo / ".runtime/ffmpeg7/lib").resolve()
    specs = build_service_specs(
        repo,
        state,
        python=selected_python,
        vllm=selected_vllm,
        ffmpeg_lib=selected_ffmpeg,
        codev_enabled=codev_enabled,
    )
    if dry_run:
        return {
            "status": "DRY_RUN",
            "service_order": [item.name for item in specs],
            "commands": {item.name: list(item.argv) for item in specs},
            "vllm": str(selected_vllm),
            "token_file": str(bearer_token_file(state)),
            "topology": "full" if codev_enabled else "nocodev",
            "codev": "required" if codev_enabled else "intentionally_disabled",
        }
    _validate_paths(specs, vllm=selected_vllm, ffmpeg=selected_ffmpeg)
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(state, 0o700)
    record_path = state / "run/zetsu_services.json"

    token = load_local_plus_token(state)
    requested_topology = "full" if codev_enabled else "nocodev"
    if record_path.is_file():
        try:
            existing: object = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ZetsuConfigError("zetsu_runtime_record_invalid") from exc
        if not isinstance(existing, dict):
            raise ZetsuConfigError("zetsu_runtime_record_invalid")
        if existing.get("schema_version") != 2:
            # A dormant v1 record can be replaced because no process is signalled.
            # Any responding endpoint remains protected as potentially unowned.
            if any(_probe(spec) for spec in specs):
                raise ZetsuConfigError(
                    "zetsu_runtime_legacy_record_with_live_endpoint_requires_operator_review"
                )
        else:
            existing_owned = _runtime_owned_processes(existing)
            same_topology = existing.get("topology") == requested_topology
            existing_services = existing.get("services")
            root_identities_valid = False
            if isinstance(existing_services, dict) and existing_services:
                roots: list[ProcessIdentity] = []
                try:
                    for item in existing_services.values():
                        if not isinstance(item, dict) or not isinstance(item.get("process"), dict):
                            raise ValueError
                        roots.append(ProcessIdentity(**item["process"]))
                except (TypeError, ValueError):
                    roots = []
                root_identities_valid = bool(roots) and all(
                    existing_owned.get(identity.pid) == identity for identity in roots
                )
            if same_topology and root_identities_valid and all(_probe(spec) for spec in specs):
                _probe_zetsu(token)
                return {
                    "status": "READY",
                    "runtime_state": "ALREADY_RUNNING",
                    "topology": requested_topology,
                    "codev": "healthy" if codev_enabled else "intentionally_disabled",
                    "service_order": [item.name for item in specs],
                    "services": existing_services,
                    "state_root": str(state),
                    "token_file": str(bearer_token_file(state)),
                    "token_env_var": DEFAULT_TOKEN_ENV,
                    "token_loaded_for_command": True,
                    "vllm": str(selected_vllm),
                }
            if existing_owned:
                recovered = _terminate_runtime_record(existing)
                if recovered["survivors"]:
                    _atomic_json(record_path, {**existing, "failed_recovery": recovered})
                    raise ZetsuConfigError("owned_runtime_recovery_failed")

    for spec in specs:
        if _probe(spec):
            raise ZetsuConfigError(f"{spec.name}_endpoint_in_use_by_unowned_process")

    runtime_id = uuid.uuid4().hex + uuid.uuid4().hex
    ownership_environment = {
        "LAPLACE_ZETSU_RUNTIME_ID": runtime_id,
        "LAPLACE_ZETSU_STATE_ROOT": str(state),
        "LAPLACE_ZETSU_REPOSITORY": str(repo),
        "LAPLACE_SERVER_OWNER_TOKEN": runtime_id,
    }
    specs = tuple(
        replace(spec, environment={**spec.environment, **ownership_environment})
        for spec in specs
    )

    started: list[tuple[ServiceSpec, subprocess.Popen[bytes]]] = []
    service_results: dict[str, object] = {}
    records: dict[str, object] = {}
    runtime_record: dict[str, object] = {
        "schema_version": 2,
        "runtime_id": runtime_id,
        "boot_id": _boot_id(),
        "repository": str(repo),
        "state_root": str(state),
        "topology": requested_topology,
        "codev": "required" if codev_enabled else "intentionally_disabled",
        "created_at_utc": time.time(),
        "services": records,
    }
    _atomic_json(record_path, runtime_record)
    try:
        for spec in specs:
            process = _spawn(spec)
            started.append((spec, process))
            identity = _process_identity(process.pid)
            records[spec.name] = {
                "process": asdict(identity),
                "log": str(spec.log_path),
                "probe_url": spec.probe_url,
                "expected_model": spec.expected_model,
            }
            _atomic_json(record_path, runtime_record)
            _wait_ready(spec, process, deadline=time.monotonic() + timeout)
            service_results[spec.name] = {
                "status": "STARTED_READY",
                "pid": process.pid,
                "endpoint": spec.probe_url,
                "log": str(spec.log_path),
            }
        _probe_zetsu(token)
        runtime_record["owned_processes_at_ready"] = [
            asdict(item) for item in _runtime_owned_processes(runtime_record).values()
        ]
        runtime_record["ready_at_utc"] = time.time()
        _atomic_json(record_path, runtime_record)
        os.environ.setdefault(DEFAULT_TOKEN_ENV, token)
        return {
            "status": "READY",
            "runtime_state": "STARTED",
            "topology": requested_topology,
            "codev": "healthy" if codev_enabled else "intentionally_disabled",
            "service_order": [item.name for item in specs],
            "services": service_results,
            "state_root": str(state),
            "token_file": str(bearer_token_file(state)),
            "token_env_var": DEFAULT_TOKEN_ENV,
            "token_loaded_for_command": True,
            "vllm": str(selected_vllm),
        }
    except BaseException:
        rollback = _terminate_runtime_record(
            runtime_record,
            graceful_timeout=15.0,
            escalation_timeout=10.0,
        )
        runtime_record["failed_start_rollback"] = rollback
        _atomic_json(record_path, runtime_record)
        raise


def stop_local_runtime(state_root: Path) -> dict[str, object]:
    """Stop the complete exact-identity process tree and report any survivor."""

    state = state_root.resolve()
    record_path = state / "run/zetsu_services.json"
    try:
        raw: object = json.loads(record_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"status": "NOT_RUNNING", "services": {}}
    except (OSError, json.JSONDecodeError) as exc:
        raise ZetsuConfigError("zetsu_runtime_record_invalid") from exc
    if not isinstance(raw, dict):
        raise ZetsuConfigError("zetsu_runtime_record_invalid")
    if raw.get("schema_version") != 2:
        return {
            "status": "FAILED",
            "failure": "legacy_runtime_record_lacks_pid_reuse_safe_identity",
            "services": raw.get("services"),
        }
    if raw.get("state_root") != str(state):
        raise ZetsuConfigError("zetsu_runtime_state_root_identity_mismatch")
    diagnostics = _terminate_runtime_record(raw)
    if diagnostics["survivors"]:
        updated = {**raw, "last_stop": diagnostics, "stop_status": "FAILED"}
        _atomic_json(record_path, updated)
        return {
            "status": "FAILED",
            "failure": "owned_processes_survived_stop",
            "topology": raw.get("topology"),
            "diagnostics": diagnostics,
        }
    stopped = {
        **raw,
        "services": {},
        "owned_processes_at_ready": [],
        "last_stop": diagnostics,
        "stop_status": "STOPPED",
    }
    _atomic_json(record_path, stopped)
    return {
        "status": "STOPPED",
        "topology": raw.get("topology"),
        "codev": raw.get("codev"),
        "diagnostics": diagnostics,
    }
