"""Ownership-aware lifecycle for one resolved tiered-serving profile at a time."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess  # nosec B404
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, TextIO, TypeAlias

from .serving_profiles import (
    InstalledServingCapabilities,
    ResolvedServingProfile,
    endpoint_for,
    load_profiles,
    resolve_profile,
)

JsonObject: TypeAlias = dict[str, object]
Runner = Callable[..., subprocess.CompletedProcess[str]]


class ServingRuntimeError(RuntimeError):
    """A profile failed admission, launch, identity, or ownership validation."""

    def __init__(self, category: str, evidence: JsonObject) -> None:
        super().__init__(category)
        self.category = category
        self.evidence = evidence


@dataclass(frozen=True)
class GpuSnapshot:
    name: str
    total_mib: int
    used_mib: int
    free_mib: int
    utilization_percent: int
    power_watts: float
    compute_pids: tuple[int, ...]
    captured_at_utc: str


@dataclass(frozen=True)
class OwnedProfileProcess:
    profile_id: str
    pid: int
    process_group_id: int
    proc_start_ticks: int
    command_sha256: str
    resolution_sha256: str
    log_path: str
    started_at_utc: str
    launch_environment: dict[str, str] = field(default_factory=dict)


def _integer(value: str) -> int:
    return int(value.strip())


def observe_gpu(*, runner: Runner = subprocess.run) -> GpuSnapshot:
    gpu = runner(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    compute = runner(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    rows = [line for line in gpu.stdout.splitlines() if line.strip()]
    if gpu.returncode != 0 or len(rows) != 1:
        raise ServingRuntimeError(
            "gpu_observation_failed",
            {"returncode": gpu.returncode, "stderr": gpu.stderr[-2_000:]},
        )
    if compute.returncode != 0:
        raise ServingRuntimeError(
            "gpu_compute_observation_failed",
            {"returncode": compute.returncode, "stderr": compute.stderr[-2_000:]},
        )
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 6:
        raise ServingRuntimeError("gpu_observation_malformed", {"row": rows[0]})
    pids: list[int] = []
    for line in compute.stdout.splitlines():
        value = line.strip()
        if not value:
            continue
        if not value.isdigit():
            raise ServingRuntimeError(
                "gpu_compute_observation_malformed",
                {"row": value[:200]},
            )
        pids.append(int(value))
    try:
        snapshot = GpuSnapshot(
            name=fields[0],
            total_mib=_integer(fields[1]),
            used_mib=_integer(fields[2]),
            free_mib=_integer(fields[3]),
            utilization_percent=_integer(fields[4]),
            power_watts=float(fields[5]),
            compute_pids=tuple(sorted(set(pids))),
            captured_at_utc=datetime.now(UTC).isoformat(),
        )
    except ValueError as exc:
        raise ServingRuntimeError(
            "gpu_observation_malformed",
            {"row": rows[0]},
        ) from exc
    if (
        not snapshot.name
        or min(snapshot.total_mib, snapshot.used_mib, snapshot.free_mib) < 0
        or not 0 <= snapshot.utilization_percent <= 100
    ):
        raise ServingRuntimeError("gpu_observation_malformed", {"row": rows[0]})
    return snapshot


def _proc_start_ticks(pid: int) -> int:
    fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
    return int(fields[21])


def _command_sha256(command: tuple[str, ...]) -> str:
    return hashlib.sha256(
        json.dumps(list(command), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class ServingProfileRuntime:
    """Launch, identify, and release only processes created by this controller."""

    def __init__(
        self,
        state_root: Path,
        *,
        residual_free_mib: int = 2_048,
        ffmpeg_library_path: Path | None = None,
        launch_environment: dict[str, str] | None = None,
    ) -> None:
        self.state_root = state_root.resolve()
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.ownership_path = self.state_root / "owned_profile_process.json"
        self.residual_free_mib = residual_free_mib
        self.ffmpeg_library_path = ffmpeg_library_path
        self.launch_environment = dict(launch_environment or {})
        if any(
            not key or not value or "=" in key or "\n" in key or "\n" in value
            for key, value in self.launch_environment.items()
        ):
            raise ValueError("launch_environment must contain non-empty environment pairs")
        self._process: subprocess.Popen[str] | None = None
        self._log_handle: TextIO | None = None

    def admission(self, resolved: ResolvedServingProfile) -> JsonObject:
        snapshot = observe_gpu()
        required = (
            int(snapshot.total_mib * resolved.profile.gpu_memory_utilization)
            + self.residual_free_mib
        )
        admitted = snapshot.free_mib >= required
        return {
            "status": "ADMITTED" if admitted else "BLOCKED_UNRELATED_GPU_MEMORY",
            "required_free_mib": required,
            "residual_free_mib": self.residual_free_mib,
            "gpu": asdict(snapshot),
            "unowned_compute_pids": list(snapshot.compute_pids),
        }

    def start(self, resolved: ResolvedServingProfile) -> OwnedProfileProcess:
        if self.ownership_path.exists():
            raise ServingRuntimeError(
                "owned_profile_already_recorded",
                {"ownership_path": str(self.ownership_path)},
            )
        admission = self.admission(resolved)
        if admission["status"] != "ADMITTED":
            raise ServingRuntimeError("gpu_admission_blocked", admission)
        log_path = self.state_root / f"{resolved.profile.profile_id}.server.log"
        log_handle = log_path.open("a", encoding="utf-8", buffering=1)
        environment = {
            key: value
            for key, value in os.environ.items()
            if key
            in {
                "CUDA_HOME",
                "CUDA_VISIBLE_DEVICES",
                "HOME",
                "LANG",
                "LC_ALL",
                "PATH",
                "PYTHONPATH",
                "TZ",
                "VLLM_CACHE_ROOT",
                "VLLM_CONFIG_ROOT",
                "LAPLACE_ZETSU_RUNTIME_ID",
                "LAPLACE_ZETSU_STATE_ROOT",
                "LAPLACE_ZETSU_REPOSITORY",
            }
        }
        if self.ffmpeg_library_path is not None:
            environment["LD_LIBRARY_PATH"] = str(self.ffmpeg_library_path)
        environment.update(self.launch_environment)
        executable_directory = str(Path(resolved.command[0]).parent)
        environment["PATH"] = (
            executable_directory
            + os.pathsep
            + environment.get("PATH", "/usr/local/bin:/usr/bin:/bin")
        )
        process = subprocess.Popen(  # nosec B603 - argv is resolved and hashed
            list(resolved.command),
            cwd=self.state_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        self._process = process
        self._log_handle = log_handle
        owned = OwnedProfileProcess(
            profile_id=resolved.profile.profile_id,
            pid=process.pid,
            process_group_id=os.getpgid(process.pid),
            proc_start_ticks=_proc_start_ticks(process.pid),
            command_sha256=_command_sha256(resolved.command),
            resolution_sha256=resolved.resolution_sha256,
            log_path=str(log_path),
            launch_environment=self.launch_environment,
            started_at_utc=datetime.now(UTC).isoformat(),
        )
        self.ownership_path.write_text(
            json.dumps(asdict(owned), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(self.ownership_path, 0o600)
        return owned

    def wait_ready(self, resolved: ResolvedServingProfile) -> JsonObject:
        deadline = time.monotonic() + resolved.profile.startup_timeout
        endpoint = endpoint_for(resolved.profile)
        last_error = "not_probed"
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise ServingRuntimeError(
                    "profile_process_exited",
                    {
                        "returncode": self._process.returncode,
                        "log_path": str(
                            self.state_root
                            / f"{resolved.profile.profile_id}.server.log"
                        ),
                    },
                )
            try:
                with urllib.request.urlopen(  # nosec B310 - constructed localhost URL
                    endpoint + "/v1/models", timeout=5
                ) as response:
                    raw: object = json.loads(response.read())
                models = raw.get("data") if isinstance(raw, dict) else None
                identities = (
                    [item.get("id") for item in models if isinstance(item, dict)]
                    if isinstance(models, list)
                    else []
                )
                if resolved.profile.served_model_name in identities:
                    return {
                        "status": "READY_EXACT_MODEL",
                        "model_id": resolved.profile.served_model_name,
                        "endpoint": endpoint,
                    }
                last_error = f"identity_mismatch:{identities}"
            except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
                last_error = type(exc).__name__
            time.sleep(1)
        raise ServingRuntimeError(
            "profile_startup_timeout",
            {"profile_id": resolved.profile.profile_id, "last_error": last_error},
        )

    def _load_owned(self) -> OwnedProfileProcess:
        if not self.ownership_path.is_file():
            raise ServingRuntimeError("no_owned_profile", {})
        raw: object = json.loads(self.ownership_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ServingRuntimeError("ownership_record_malformed", {})
        try:
            return OwnedProfileProcess(**raw)
        except (TypeError, ValueError) as exc:
            raise ServingRuntimeError("ownership_record_malformed", {}) from exc

    def release_owned(self, *, timeout_seconds: float = 45) -> JsonObject:
        owned = self._load_owned()
        try:
            current_ticks = _proc_start_ticks(owned.pid)
        except (OSError, ValueError):
            self.ownership_path.unlink(missing_ok=True)
            return {"status": "OWNED_PROCESS_ALREADY_EXITED", "pid": owned.pid}
        if current_ticks != owned.proc_start_ticks:
            raise ServingRuntimeError(
                "pid_identity_changed",
                {"pid": owned.pid, "recorded_start_ticks": owned.proc_start_ticks},
            )
        if os.getpgid(owned.pid) != owned.process_group_id:
            raise ServingRuntimeError("process_group_identity_changed", {"pid": owned.pid})
        os.killpg(owned.process_group_id, signal.SIGTERM)
        if self._process is not None and self._process.pid == owned.pid:
            try:
                self._process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(owned.process_group_id, signal.SIGKILL)
                self._process.wait(timeout=10)
            self.ownership_path.unlink(missing_ok=True)
            if self._log_handle is not None:
                self._log_handle.close()
            return {
                "status": "RELEASED_OWNED_PROFILE",
                "profile_id": owned.profile_id,
                "pid": owned.pid,
            }
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline and Path(f"/proc/{owned.pid}").exists():
            time.sleep(0.25)
        if Path(f"/proc/{owned.pid}").exists():
            os.killpg(owned.process_group_id, signal.SIGKILL)
        self.ownership_path.unlink(missing_ok=True)
        log_handle = self._log_handle
        if log_handle is not None:
            log_handle.close()
        return {
            "status": "RELEASED_OWNED_PROFILE",
            "profile_id": owned.profile_id,
            "pid": owned.pid,
        }

    def status(self) -> JsonObject:
        gpu = asdict(observe_gpu())
        owned: JsonObject | None = None
        if self.ownership_path.is_file():
            record = self._load_owned()
            owned = asdict(record)
            owned["alive_exact_pid"] = (
                Path(f"/proc/{record.pid}").exists()
                and _proc_start_ticks(record.pid) == record.proc_start_ticks
            )
        return {"status": "OBSERVED", "gpu": gpu, "owned_profile": owned}


class ServingProfileOperator:
    """Lazy profile resolver and lifecycle adapter for the authenticated service layer."""

    def __init__(
        self,
        repository_root: Path,
        state_root: Path,
        executable: Path,
        ffmpeg_library_path: Path,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.executable = executable.resolve()
        self.ffmpeg_library_path = ffmpeg_library_path.resolve()
        self.runtime = ServingProfileRuntime(
            state_root,
            ffmpeg_library_path=self.ffmpeg_library_path,
        )

    def _capabilities(self) -> InstalledServingCapabilities:
        environment = dict(os.environ)
        environment["LD_LIBRARY_PATH"] = str(self.ffmpeg_library_path)
        version = subprocess.run(  # nosec B603
            [str(self.executable), "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
            env=environment,
        ).stdout
        help_text = subprocess.run(  # nosec B603
            [str(self.executable), "serve", "--help=all"],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
            env=environment,
        ).stdout
        return InstalledServingCapabilities.from_help(
            version=version,
            help_text=help_text,
        )

    def _resolve(self, profile_id: str) -> ResolvedServingProfile:
        profile = next(
            (
                item
                for item in load_profiles(
                    self.repository_root / "configs/serving_profiles"
                )
                if item.profile_id == profile_id
            ),
            None,
        )
        if profile is None:
            raise ServingRuntimeError("unknown_profile", {"profile_id": profile_id})
        return resolve_profile(
            profile,
            self._capabilities(),
            executable=self.executable,
        )

    def start(self, profile_id: str) -> JsonObject:
        resolved = self._resolve(profile_id)
        owned = self.runtime.start(resolved)
        ready = self.runtime.wait_ready(resolved)
        return {
            "status": "STARTED_READY",
            "profile_id": profile_id,
            "resolution_sha256": resolved.resolution_sha256,
            "owned_process": asdict(owned),
            "readiness": ready,
        }

    def stop(self) -> JsonObject:
        return self.runtime.release_owned()

    def status(self) -> JsonObject:
        return {
            **self.runtime.status(),
            "available_profiles": [
                profile.profile_id
                for profile in load_profiles(
                    self.repository_root / "configs/serving_profiles"
                )
            ],
        }
