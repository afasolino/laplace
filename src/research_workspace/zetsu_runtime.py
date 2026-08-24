"""Owned local lifecycle orchestration for the Zetsu production stack."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
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
    ownership_markers: tuple[str, ...]


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
) -> tuple[ServiceSpec, ...]:
    """Build the immutable CodeV -> Qwen -> Operator startup order."""

    repo = repository.resolve()
    state = state_root.resolve()
    quality_model, economy_model = _load_routes(repo)
    common_environment = {
        "PYTHONPATH": str(repo / "src"),
        "LAPLACE_VLLM_EXECUTABLE": str(vllm),
        "LAPLACE_FFMPEG_LIBRARY_PATH": str(ffmpeg_lib),
    }
    logs = state / "logs"
    token_path = bearer_token_file(state)
    return (
        ServiceSpec(
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
            ownership_markers=("run_codev_service.py", str(repo)),
        ),
        ServiceSpec(
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
            ownership_markers=("run_selected_quality_service.py", str(repo)),
        ),
        ServiceSpec(
            name="operator",
            argv=(
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
            ),
            environment=common_environment,
            log_path=logs / "zetsu-operator.log",
            probe_url="http://127.0.0.1:8765/api/v1/health",
            expected_model=None,
            ownership_markers=("research_workspace.operator_server", str(repo)),
        ),
    )


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


def _process_matches(pid: int, markers: tuple[str, ...]) -> bool:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
    except OSError:
        return False
    return all(marker in command for marker in markers)


def _stop_owned(pid: int, markers: tuple[str, ...], *, timeout: float = 45.0) -> bool:
    if not _process_matches(pid, markers):
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not Path(f"/proc/{pid}").exists():
            return True
        time.sleep(0.25)
    return False


def _stop_started(
    process: subprocess.Popen[bytes],
    markers: tuple[str, ...],
    *,
    timeout: float = 150.0,
) -> bool:
    """Stop and reap a supervisor spawned by this invocation."""

    if not _process_matches(process.pid, markers):
        return False
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Every spawned supervisor owns a fresh session. Escalating to that exact
        # process group also prevents a model worker from becoming orphaned.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=15.0)
    return process.poll() is not None


def _recorded_legacy_operator_pid(state_root: Path) -> int | None:
    path = state_root / "run/operator_server.pid"
    try:
        value = path.read_text(encoding="ascii").strip()
    except OSError:
        return None
    return int(value) if value.isdigit() else None


def _replace_incompatible_owned_operator(
    spec: ServiceSpec,
    *,
    repository: Path,
    state_root: Path,
) -> int:
    """Stop a legacy Operator only when its private PID record and argv agree."""

    pid = _recorded_legacy_operator_pid(state_root)
    markers = (
        "research_workspace.operator_server",
        "--repository-root",
        str(repository),
        "--state-root",
        str(state_root),
        "--host",
        "127.0.0.1",
        "--port",
        "8765",
    )
    if pid is None or not _process_matches(pid, markers):
        raise ZetsuConfigError(
            "operator_endpoint_incompatible:port_8765_not_owned_by_this_runtime"
        )
    if not _stop_owned(pid, markers, timeout=75.0):
        raise ZetsuConfigError("operator_incompatible_owned_process_stop_failed")
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if not _probe(spec):
            return pid
        time.sleep(0.25)
    raise ZetsuConfigError("operator_incompatible_endpoint_remained_after_stop")


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
) -> dict[str, object]:
    """Start the complete local stack and roll back only newly-owned supervisors."""

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
    )
    _validate_paths(specs, vllm=selected_vllm, ffmpeg=selected_ffmpeg)
    if dry_run:
        return {
            "status": "DRY_RUN",
            "service_order": [item.name for item in specs],
            "commands": {item.name: list(item.argv) for item in specs},
            "vllm": str(selected_vllm),
            "token_file": str(bearer_token_file(state)),
        }
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(state, 0o700)
    record_path = state / "run/zetsu_services.json"

    token = load_local_plus_token(state)
    operator_spec = specs[-1]
    replaced_operator_pid: int | None = None
    if _probe(operator_spec):
        try:
            _probe_zetsu(token)
        except ZetsuConfigError:
            replaced_operator_pid = _replace_incompatible_owned_operator(
                operator_spec,
                repository=repo,
                state_root=state,
            )

    started: list[tuple[ServiceSpec, subprocess.Popen[bytes]]] = []
    service_results: dict[str, object] = {}
    records: dict[str, object] = {}
    try:
        for spec in specs:
            if _probe(spec):
                service_results[spec.name] = {
                    "status": "READY_EXISTING",
                    "endpoint": spec.probe_url,
                }
                continue
            process = _spawn(spec)
            started.append((spec, process))
            records[spec.name] = {
                "pid": process.pid,
                "markers": list(spec.ownership_markers),
                "log": str(spec.log_path),
            }
            _atomic_json(record_path, {"schema_version": 1, "services": records})
            _wait_ready(spec, process, deadline=time.monotonic() + timeout)
            service_results[spec.name] = {
                "status": "STARTED_READY",
                "pid": process.pid,
                "endpoint": spec.probe_url,
                "log": str(spec.log_path),
            }
        _probe_zetsu(token)
        os.environ.setdefault(DEFAULT_TOKEN_ENV, token)
        return {
            "status": "READY",
            "service_order": [item.name for item in specs],
            "services": service_results,
            "state_root": str(state),
            "token_file": str(bearer_token_file(state)),
            "token_env_var": DEFAULT_TOKEN_ENV,
            "token_loaded_for_command": True,
            "vllm": str(selected_vllm),
            "replaced_incompatible_operator_pid": replaced_operator_pid,
        }
    except BaseException:
        rollback: dict[str, bool] = {}
        for spec, process in reversed(started):
            rollback[spec.name] = _stop_started(process, spec.ownership_markers)
        if rollback:
            _atomic_json(
                record_path,
                {"schema_version": 1, "services": records, "failed_start_rollback": rollback},
            )
        raise


def stop_local_runtime(state_root: Path) -> dict[str, object]:
    """Stop only supervisors whose recorded PID still matches its exact command."""

    state = state_root.resolve()
    record_path = state / "run/zetsu_services.json"
    try:
        raw: object = json.loads(record_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"status": "NOT_RUNNING", "services": {}}
    except (OSError, json.JSONDecodeError) as exc:
        raise ZetsuConfigError("zetsu_runtime_record_invalid") from exc
    services = raw.get("services") if isinstance(raw, dict) else None
    if not isinstance(services, dict):
        raise ZetsuConfigError("zetsu_runtime_record_invalid")
    results: dict[str, object] = {}
    for name in ("operator", "qwen", "codev"):
        item = services.get(name)
        pid = item.get("pid") if isinstance(item, dict) else None
        markers = item.get("markers") if isinstance(item, dict) else None
        if not isinstance(pid, int) or not isinstance(markers, list) or any(
            not isinstance(marker, str) for marker in markers
        ):
            results[name] = "NOT_OWNED"
            continue
        results[name] = "STOPPED" if _stop_owned(pid, tuple(markers)) else "NOT_OWNED"
    _atomic_json(record_path, {"schema_version": 1, "services": {}, "last_stop": results})
    return {"status": "STOPPED", "services": results}
