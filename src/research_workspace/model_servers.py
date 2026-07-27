"""Fail-closed admission and ownership-aware lifecycle for local model servers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import subprocess  # nosec B404 - commands are fixed repository tools.
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Sequence, TypeAlias
from urllib.parse import urlsplit

JsonObject: TypeAlias = dict[str, object]
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]

_GPU_QUERY = (
    "name,memory.used,memory.free,memory.total,utilization.gpu,power.draw"
)
_GPU_COMMAND = (
    "nvidia-smi",
    f"--query-gpu={_GPU_QUERY}",
    "--format=csv,noheader,nounits",
)
_COMPUTE_COMMAND = (
    "nvidia-smi",
    "--query-compute-apps=pid,process_name,used_memory",
    "--format=csv,noheader,nounits",
)
_PROFILE_NAMES = ("phase2_main", "phase2_rtl_worker")


class ModelServerSafetyError(RuntimeError):
    """The model-server lifecycle cannot proceed safely."""

    def __init__(self, category: str, evidence: JsonObject) -> None:
        super().__init__(category)
        self.category = category
        self.evidence = evidence


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, value: object, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    data = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        mode,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_integer(value: str, *, label: str) -> int:
    text = value.strip()
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError(f"{label} is not an integer")
    return int(text)


@dataclass(frozen=True)
class ModelServerSpec:
    """One exact local OpenAI-compatible model-server identity."""

    profile: str
    endpoint: str
    port: int
    expected_model_id: str
    model_path: str

    def __post_init__(self) -> None:
        endpoint = urlsplit(self.endpoint)
        if endpoint.scheme != "http" or endpoint.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("Model server endpoint must be localhost HTTP")
        if endpoint.port != self.port or endpoint.path not in {"", "/"}:
            raise ValueError("Model server endpoint and port are inconsistent")
        if self.profile not in _PROFILE_NAMES:
            raise ValueError("Unsupported model-server profile")
        if not self.expected_model_id or "\n" in self.expected_model_id:
            raise ValueError("Invalid expected model identity")
        if not Path(self.model_path).is_absolute():
            raise ValueError("Model path must be absolute")

    def to_json(self) -> JsonObject:
        return asdict(self)


@dataclass(frozen=True)
class EmpiricalMemoryPolicy:
    """Admission threshold derived only from recorded measurements."""

    maximum_observed_used_mib: int
    required_residual_free_mib: int
    required_pre_start_free_mib: int
    evidence_paths: tuple[str, ...]
    evidence_sha256: tuple[str, ...]

    def to_json(self) -> JsonObject:
        return {
            "basis": "maximum_observed_dual_server_used_mib_plus_configured_residual",
            "maximum_observed_used_mib": self.maximum_observed_used_mib,
            "required_residual_free_mib": self.required_residual_free_mib,
            "required_pre_start_free_mib": self.required_pre_start_free_mib,
            "evidence_paths": list(self.evidence_paths),
            "evidence_sha256": list(self.evidence_sha256),
        }


def load_server_specs(repository_root: Path) -> tuple[ModelServerSpec, ...]:
    """Load the exact measured-arm model identities from the frozen artifact file."""

    path = (
        repository_root.resolve()
        / "codex_a6000/experiments/multilanguage_dual_model_ablation_v1"
        / "model_artifacts.json"
    )
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("artifacts"), list):
        raise ModelServerSafetyError(
            "model_server_configuration_error",
            {"path": str(path), "reason": "artifacts must be a list"},
        )
    by_profile = {
        item.get("artifact_id"): item
        for item in raw["artifacts"]
        if isinstance(item, dict) and isinstance(item.get("artifact_id"), str)
    }
    specs: list[ModelServerSpec] = []
    for profile in _PROFILE_NAMES:
        item = by_profile.get(profile)
        if not isinstance(item, dict) or not isinstance(item.get("serving"), dict):
            raise ModelServerSafetyError(
                "model_server_configuration_error",
                {"path": str(path), "reason": f"missing profile {profile}"},
            )
        serving = item["serving"]
        endpoint = str(serving["endpoint"])
        parsed = urlsplit(endpoint)
        if parsed.port is None:
            raise ModelServerSafetyError(
                "model_server_configuration_error",
                {"path": str(path), "reason": f"missing port for {profile}"},
            )
        specs.append(
            ModelServerSpec(
                profile=profile,
                endpoint=endpoint,
                port=parsed.port,
                expected_model_id=str(item["served_model_name"]),
                model_path=str(item["output_path"]),
            )
        )
    return tuple(specs)


def derive_empirical_memory_policy(repository_root: Path) -> EmpiricalMemoryPolicy:
    """Derive a conservative threshold from preserved successful-run GPU probes."""

    root = repository_root.resolve()
    config_path = root / "codex_a6000/PROJECT_CONFIG.json"
    config: object = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("hardware"), dict):
        raise ModelServerSafetyError(
            "model_server_configuration_error",
            {"path": str(config_path), "reason": "hardware configuration is absent"},
        )
    residual_gib = config["hardware"].get("minimum_free_vram_gib_after_main_server_start")
    if not isinstance(residual_gib, int) or residual_gib <= 0:
        raise ModelServerSafetyError(
            "model_server_configuration_error",
            {"path": str(config_path), "reason": "invalid residual VRAM requirement"},
        )
    candidates = sorted(root.glob("outputs/codev_*/project/host_logs/*_cuda_probe.json"))
    observations: list[tuple[int, Path, str]] = []
    for path in candidates:
        try:
            data = path.read_bytes()
            record: object = json.loads(data)
        except (OSError, json.JSONDecodeError):
            continue
        if (
            not isinstance(record, dict)
            or record.get("status") != "PASS"
            or not isinstance(record.get("stdout"), str)
        ):
            continue
        fields = [part.strip() for part in record["stdout"].strip().split(",")]
        if len(fields) < 4 or "A6000" not in fields[0]:
            continue
        try:
            used_mib = _parse_integer(fields[1], label="historical memory.used")
        except ValueError:
            continue
        # These preserved probes were captured while the measured dual-server
        # CodeV configuration was resident. Low-use probes are not treated as
        # evidence for that footprint.
        if used_mib < 30_000:
            continue
        observations.append(
            (used_mib, path, hashlib.sha256(data).hexdigest())
        )
    if not observations:
        raise ModelServerSafetyError(
            "missing_empirical_gpu_evidence",
            {
                "search_root": str(root / "outputs"),
                "reason": "no valid preserved A6000 dual-server probes",
            },
        )
    maximum = max(item[0] for item in observations)
    residual_mib = residual_gib * 1024
    return EmpiricalMemoryPolicy(
        maximum_observed_used_mib=maximum,
        required_residual_free_mib=residual_mib,
        required_pre_start_free_mib=maximum + residual_mib,
        evidence_paths=tuple(str(item[1].relative_to(root)) for item in observations),
        evidence_sha256=tuple(item[2] for item in observations),
    )


def _run_observation(
    command: Sequence[str],
    *,
    runner: CommandRunner = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    return runner(
        list(command),
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )


def observe_gpu(
    *,
    runner: CommandRunner = subprocess.run,
) -> JsonObject:
    """Observe the GPU and its compute processes with fixed read-only commands."""

    try:
        gpu_result = _run_observation(_GPU_COMMAND, runner=runner)
        compute_result = _run_observation(_COMPUTE_COMMAND, runner=runner)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "UNAVAILABLE", "error": type(exc).__name__}
    if gpu_result.returncode != 0:
        return {
            "status": "UNAVAILABLE",
            "returncode": gpu_result.returncode,
            "error": gpu_result.stderr.strip()[:1000],
        }
    rows = [row for row in gpu_result.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        return {
            "status": "UNSUPPORTED_GPU_TOPOLOGY",
            "gpu_count": len(rows),
            "reason": "the frozen configuration requires exactly one GPU",
        }
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 6:
        return {"status": "UNAVAILABLE", "reason": "malformed nvidia-smi GPU output"}
    try:
        gpu: JsonObject = {
            "name": fields[0],
            "memory_used_mib": _parse_integer(fields[1], label="memory.used"),
            "memory_free_mib": _parse_integer(fields[2], label="memory.free"),
            "memory_total_mib": _parse_integer(fields[3], label="memory.total"),
            "utilization_percent": _parse_integer(fields[4], label="utilization.gpu"),
            "power_draw_watts": float(fields[5]),
        }
    except ValueError as exc:
        return {"status": "UNAVAILABLE", "reason": str(exc)}
    processes: list[JsonObject] = []
    if compute_result.returncode == 0:
        for row in compute_result.stdout.splitlines():
            parts = [part.strip() for part in row.split(",")]
            if len(parts) != 3:
                continue
            try:
                processes.append(
                    {
                        "pid": _parse_integer(parts[0], label="compute PID"),
                        "process_name": parts[1][:500],
                        "used_memory_mib": _parse_integer(
                            parts[2], label="compute memory"
                        ),
                    }
                )
            except ValueError:
                continue
    return {
        "status": "OBSERVED",
        "gpu": gpu,
        "compute_processes": processes,
        "compute_process_probe_status": (
            "OBSERVED" if compute_result.returncode == 0 else "UNAVAILABLE"
        ),
    }


def _listening_socket_inodes(port: int) -> set[str]:
    inodes: set[str] = set()
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            lines = table.read_text(encoding="ascii").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":
                continue
            local = fields[1].rsplit(":", 1)
            if len(local) == 2 and int(local[1], 16) == port:
                inodes.add(fields[9])
    return inodes


def observe_port_owners(port: int) -> list[JsonObject]:
    """Resolve listening socket inodes to processes using read-only /proc data."""

    inodes = _listening_socket_inodes(port)
    if not inodes:
        return []
    owners: list[JsonObject] = []
    for process_dir in sorted(
        (path for path in Path("/proc").iterdir() if path.name.isdigit()),
        key=lambda item: int(item.name),
    ):
        try:
            file_descriptors = process_dir.joinpath("fd").iterdir()
            matched = any(
                descriptor.readlink().as_posix() in {f"socket:[{inode}]" for inode in inodes}
                for descriptor in file_descriptors
            )
        except (OSError, PermissionError):
            continue
        if not matched:
            continue
        try:
            command_line = (
                process_dir.joinpath("cmdline")
                .read_bytes()
                .replace(b"\x00", b" ")
                .decode("utf-8", errors="replace")
                .strip()
            )
            process_name = process_dir.joinpath("comm").read_text(encoding="utf-8").strip()
        except OSError:
            continue
        owners.append(
            {
                "pid": int(process_dir.name),
                "process_name": process_name[:256],
                "command_line": command_line[:2000],
            }
        )
    return owners


def probe_endpoint(spec: ModelServerSpec, *, timeout_seconds: float = 3.0) -> JsonObject:
    """Probe `/v1/models` and require the exact configured served-model identity."""

    request = urllib.request.Request(  # nosec B310 - endpoint is localhost-validated.
        f"{spec.endpoint}/v1/models",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # nosec B310
            if response.status != 200:
                return {"status": "UNAVAILABLE", "http_status": response.status}
            raw: object = json.loads(response.read())
    except (
        OSError,
        urllib.error.URLError,
        json.JSONDecodeError,
        TimeoutError,
    ) as exc:
        return {"status": "UNAVAILABLE", "error": type(exc).__name__}
    data = raw.get("data") if isinstance(raw, dict) else None
    identities = (
        sorted(
            item["id"]
            for item in data
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        )
        if isinstance(data, list)
        else []
    )
    return {
        "status": (
            "HEALTHY_EXACT_MODEL"
            if spec.expected_model_id in identities
            else "MODEL_IDENTITY_MISMATCH"
        ),
        "expected_model_id": spec.expected_model_id,
        "served_model_ids": identities,
    }


class ModelServerAdmission:
    """Create auditable preflight evidence and a fail-closed decision."""

    def __init__(
        self,
        repository_root: Path,
        *,
        gpu_observer: Callable[[], JsonObject] = observe_gpu,
        endpoint_probe: Callable[[ModelServerSpec], JsonObject] = probe_endpoint,
        port_observer: Callable[[int], list[JsonObject]] = observe_port_owners,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.gpu_observer = gpu_observer
        self.endpoint_probe = endpoint_probe
        self.port_observer = port_observer

    def preflight(
        self,
        *,
        output_path: Path,
        specs: Sequence[ModelServerSpec] | None = None,
        policy: EmpiricalMemoryPolicy | None = None,
        startup_timeout_seconds: int = 600,
    ) -> JsonObject:
        if startup_timeout_seconds < 30 or startup_timeout_seconds > 1800:
            raise ValueError("startup timeout must be between 30 and 1800 seconds")
        selected_specs = tuple(specs or load_server_specs(self.repository_root))
        selected_policy = policy or derive_empirical_memory_policy(self.repository_root)
        gpu_observation = self.gpu_observer()
        endpoints = {
            spec.profile: self.endpoint_probe(spec) for spec in selected_specs
        }
        ports = {
            str(spec.port): self.port_observer(spec.port) for spec in selected_specs
        }
        exact = all(
            result.get("status") == "HEALTHY_EXACT_MODEL"
            for result in endpoints.values()
        )
        mismatches = [
            profile
            for profile, result in endpoints.items()
            if result.get("status") == "MODEL_IDENTITY_MISMATCH"
        ]
        occupied_unhealthy = [
            spec.profile
            for spec in selected_specs
            if ports[str(spec.port)]
            and endpoints[spec.profile].get("status") != "HEALTHY_EXACT_MODEL"
        ]
        partial_exact = any(
            result.get("status") == "HEALTHY_EXACT_MODEL"
            for result in endpoints.values()
        ) and not exact

        decision = "ADMITTED_START_REQUIRED"
        failure_category: str | None = None
        if exact:
            decision = "REUSED_HEALTHY_SERVERS"
        elif mismatches or occupied_unhealthy or partial_exact:
            decision = "REFUSED"
            failure_category = "model_server_port_conflict"
        elif gpu_observation.get("status") != "OBSERVED":
            decision = "REFUSED"
            failure_category = "hardware_probe_failure"
        else:
            gpu = gpu_observation.get("gpu")
            if not isinstance(gpu, dict) or "A6000" not in str(gpu.get("name", "")):
                decision = "REFUSED"
                failure_category = "hardware_mismatch"
            elif (
                not isinstance(gpu.get("memory_free_mib"), int)
                or gpu["memory_free_mib"] < selected_policy.required_pre_start_free_mib
            ):
                decision = "REFUSED"
                failure_category = "resource_admission_failure"

        evidence: JsonObject = {
            "schema_version": 1,
            "timestamp_utc": _utc_now(),
            "decision": decision,
            "failure_category": failure_category,
            "startup_timeout_seconds": startup_timeout_seconds,
            "specifications": [spec.to_json() for spec in selected_specs],
            "memory_policy": selected_policy.to_json(),
            "gpu_observation": gpu_observation,
            "endpoint_observations": endpoints,
            "port_ownership": ports,
        }
        _atomic_json(output_path.resolve(), evidence)
        return evidence


class ModelServerController:
    """Bounded startup and safe release through the repository lifecycle script."""

    def __init__(
        self,
        repository_root: Path,
        state_directory: Path,
        *,
        admission: ModelServerAdmission | None = None,
        command_runner: CommandRunner = subprocess.run,
        sleeper: Callable[[float], None] = time.sleep,
        gpu_observer: Callable[[], JsonObject] = observe_gpu,
        endpoint_probe: Callable[[ModelServerSpec], JsonObject] = probe_endpoint,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.state_directory = state_directory.resolve()
        self.admission = admission or ModelServerAdmission(self.repository_root)
        self.command_runner = command_runner
        self.sleeper = sleeper
        self.gpu_observer = gpu_observer
        self.endpoint_probe = endpoint_probe
        self.specs = load_server_specs(self.repository_root)
        self.manager = self.repository_root / "scripts/manage_multilanguage_model_servers.sh"
        self.token_path = self.state_directory / ".lifecycle_owner.json"
        self.preflight_path = self.state_directory / "model_server_preflight.json"
        self.lifecycle_path = self.state_directory / "model_server_lifecycle.json"

    def _subprocess_environment(self, owner_token: str | None) -> dict[str, str]:
        allowed = (
            "PATH",
            "LANG",
            "LC_ALL",
            "LD_LIBRARY_PATH",
            "CUDA_VISIBLE_DEVICES",
            "LAPLACE_VLLM_EXECUTABLE",
            "LAPLACE_FFMPEG_LIBRARY_PATH",
        )
        environment = {
            key: value for key, value in os.environ.items() if key in allowed
        }
        if owner_token is not None:
            environment["LAPLACE_SERVER_OWNER_TOKEN"] = owner_token
        return environment

    def _run_manager(
        self,
        action: str,
        *,
        timeout_seconds: int,
        owner_token: str | None,
    ) -> subprocess.CompletedProcess[str]:
        if action not in {"start-phase3", "stop-phase3"}:
            raise ValueError("Unsupported lifecycle action")
        return self.command_runner(
            [str(self.manager), action],
            cwd=self.repository_root,
            env=self._subprocess_environment(owner_token),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )

    def start(self, *, startup_timeout_seconds: int = 600) -> JsonObject:
        evidence = self.admission.preflight(
            output_path=self.preflight_path,
            specs=self.specs,
            startup_timeout_seconds=startup_timeout_seconds,
        )
        if evidence["decision"] == "REUSED_HEALTHY_SERVERS":
            result: JsonObject = {
                "schema_version": 1,
                "status": "REUSED_HEALTHY_SERVERS",
                "timestamp_utc": _utc_now(),
                "preflight_path": str(self.preflight_path),
                "started_profiles": [],
                "owned_processes": self._owned_profiles(),
            }
            _atomic_json(self.lifecycle_path, result)
            return result
        if evidence["decision"] != "ADMITTED_START_REQUIRED":
            raise ModelServerSafetyError(str(evidence["failure_category"]), evidence)

        token = secrets.token_urlsafe(32)
        self.state_directory.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            self.token_path,
            {"schema_version": 1, "owner_token": token, "created_at_utc": _utc_now()},
            mode=0o600,
        )
        try:
            completed = self._run_manager(
                "start-phase3",
                timeout_seconds=min(startup_timeout_seconds, 120),
                owner_token=token,
            )
        except subprocess.TimeoutExpired as exc:
            raise ModelServerSafetyError(
                "model_server_startup_timeout",
                {"preflight_path": str(self.preflight_path), "timeout": exc.timeout},
            ) from exc
        if completed.returncode != 0:
            raise ModelServerSafetyError(
                "model_server_start_failure",
                {
                    "preflight_path": str(self.preflight_path),
                    "returncode": completed.returncode,
                    "stdout_tail": completed.stdout[-4000:],
                    "stderr_tail": completed.stderr[-4000:],
                },
            )

        deadline = time.monotonic() + startup_timeout_seconds
        last: dict[str, JsonObject] = {}
        while time.monotonic() < deadline:
            last = {spec.profile: self.endpoint_probe(spec) for spec in self.specs}
            if all(
                result.get("status") == "HEALTHY_EXACT_MODEL"
                for result in last.values()
            ):
                result = {
                    "schema_version": 1,
                    "status": "STARTED_HEALTHY_SERVERS",
                    "timestamp_utc": _utc_now(),
                    "preflight_path": str(self.preflight_path),
                    "started_profiles": [spec.profile for spec in self.specs],
                    "endpoint_observations": last,
                    "owned_processes": self._owned_profiles(),
                    "log_paths": self._profile_log_paths(),
                }
                _atomic_json(self.lifecycle_path, result)
                return result
            self.sleeper(1.0)
        raise ModelServerSafetyError(
            "model_server_startup_timeout",
            {
                "preflight_path": str(self.preflight_path),
                "startup_timeout_seconds": startup_timeout_seconds,
                "endpoint_observations": last,
            },
        )

    def _owned_profiles(self) -> list[JsonObject]:
        global_state = (
            self.repository_root / "outputs/a6000_agent_team/model_servers"
        )
        owned: list[JsonObject] = []
        for spec in self.specs:
            pid_path = global_state / f"{spec.profile}.pid"
            try:
                pid_text = pid_path.read_text(encoding="ascii").splitlines()[0]
                pid = _parse_integer(pid_text, label="PID")
                command_line = (
                    Path(f"/proc/{pid}/cmdline")
                    .read_bytes()
                    .replace(b"\x00", b" ")
                    .decode("utf-8", errors="replace")
                )
            except (OSError, IndexError, ValueError):
                continue
            if (
                spec.model_path in command_line
                and "--host 127.0.0.1" in command_line
                and f"--port {spec.port}" in command_line
            ):
                owned.append(
                    {"profile": spec.profile, "pid": pid, "pid_file": str(pid_path)}
                )
        return owned

    def _profile_log_paths(self) -> dict[str, str | None]:
        global_state = (
            self.repository_root / "outputs/a6000_agent_team/model_servers"
        )
        return {
            spec.profile: (
                str(matches[-1]) if matches else None
            )
            for spec in self.specs
            for matches in [sorted(global_state.glob(f"{spec.profile}_*.log"))]
        }

    def status(self) -> JsonObject:
        """Return machine-readable endpoint, ownership, port, and GPU status."""

        return {
            "schema_version": 1,
            "status": "OBSERVED",
            "timestamp_utc": _utc_now(),
            "gpu_observation": self.gpu_observer(),
            "servers": [
                {
                    **spec.to_json(),
                    "endpoint_observation": self.endpoint_probe(spec),
                    "port_ownership": observe_port_owners(spec.port),
                }
                for spec in self.specs
            ],
            "laplace_owned_processes": self._owned_profiles(),
            "log_paths": self._profile_log_paths(),
        }

    def release_owned(self, *, timeout_seconds: int = 75) -> JsonObject:
        before_gpu = self.gpu_observer()
        owned = self._owned_profiles()
        before_processes_raw = before_gpu.get("compute_processes")
        before_processes = (
            before_processes_raw if isinstance(before_processes_raw, list) else []
        )
        unrelated_before = [
            process
            for process in before_processes
            if isinstance(process, dict)
            and process.get("pid") not in {item["pid"] for item in owned}
        ]
        if not owned:
            return {
                "schema_version": 1,
                "status": "NO_LAPLACE_OWNED_SERVERS",
                "timestamp_utc": _utc_now(),
                "signalled_pids": [],
                "unrelated_compute_processes_preserved": unrelated_before,
            }
        token: str | None = None
        if self.token_path.is_file():
            raw: object = json.loads(self.token_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("owner_token"), str):
                token = raw["owner_token"]
        completed = self._run_manager(
            "stop-phase3",
            timeout_seconds=timeout_seconds,
            owner_token=token,
        )
        remaining = self._owned_profiles()
        after_gpu = self.gpu_observer()
        after_processes_raw = after_gpu.get("compute_processes")
        after_processes = (
            after_processes_raw if isinstance(after_processes_raw, list) else []
        )
        unrelated_after_pids = {
            process.get("pid")
            for process in after_processes
            if isinstance(process, dict)
        }
        preserved = all(
            process.get("pid") in unrelated_after_pids for process in unrelated_before
        )
        endpoint_observations = {
            spec.profile: self.endpoint_probe(spec) for spec in self.specs
        }
        endpoints_down = all(
            observation.get("status") == "UNAVAILABLE"
            for observation in endpoint_observations.values()
        )
        result: JsonObject = {
            "schema_version": 1,
            "status": (
                "RELEASED_LAPLACE_OWNED_SERVERS"
                if completed.returncode == 0
                and not remaining
                and preserved
                and endpoints_down
                else "MODEL_SERVER_RELEASE_INCOMPLETE"
            ),
            "timestamp_utc": _utc_now(),
            "signalled_pids": [item["pid"] for item in owned],
            "remaining_owned_processes": remaining,
            "unrelated_compute_processes_preserved": preserved,
            "endpoints_down": endpoints_down,
            "endpoint_observations": endpoint_observations,
            "manager_returncode": completed.returncode,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
            "gpu_after": after_gpu,
        }
        _atomic_json(self.state_directory / "model_server_release.json", result)
        if result["status"] != "RELEASED_LAPLACE_OWNED_SERVERS":
            raise ModelServerSafetyError("model_server_release_incomplete", result)
        return result
