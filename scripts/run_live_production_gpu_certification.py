#!/usr/bin/env python3
"""Complete sequential P1 and CodeV registered-GUI certification on one GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import signal
import subprocess  # nosec B404 - fixed local model and server commands
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Sequence, TextIO

from playwright.sync_api import sync_playwright

from research_workspace.model_artifacts import validate_local_artifacts
from research_workspace.gpu_coordination import classify_compute_ownership
from research_workspace.llm import VllmProvider
from research_workspace.personal_corpus import PersonalCorpusStore
from research_workspace.repository_authorization import RepositoryAuthorizationStore
from research_workspace.serving_profile_runtime import (
    ServingProfileOperator,
    ServingRuntimeError,
    observe_gpu,
)

from run_registered_live_gpu_smoke import (
    REPO_ID,
    _activate,
    _admin,
    _chromium,
    _environment,
    _free_port,
    _git_fixture,
    _wait_for_server,
)

ADMIN_EMAIL = "fixture-live-admin@example.test"
PLUS_EMAIL = "fixture-live-plus@example.test"
BASIC_EMAIL = "fixture-live-basic@example.test"
ADMIN_USER_ID = "usr_fixture_live_admin"
ADMIN_CAPABILITIES = (
    "chat",
    "agent",
    "research",
    "operator",
    "admin",
    "personal_corpus",
    "shared_corpus_ingest",
    "repository_admin",
    "model_admin",
)
CHAT_TERMINAL_STATES = ("COMPLETE", "FAILED")
EXPECTED_PROVIDER_FAILURE_CONSOLE_MARKERS = (
    "status of 403",
    "403 (Forbidden)",
)


ROOT = Path(__file__).resolve().parents[1]
STABLE = Path("/home/giando/work/laplace")
VLLM = STABLE / ".venv-vllm-cu129/bin/vllm"
FFMPEG = STABLE / ".runtime/ffmpeg7/lib"
CODEV_PATH = STABLE / ".models/CodeV-R1-RL-Qwen-7B-W4A16-AWQ"
CODEV_ID = "laplace-codev-r1-rl-qwen-7b-w4a16"
CODEV_ENDPOINT = "http://127.0.0.1:8103"
EXPERIMENT = (
    ROOT / "codex_a6000/experiments/multilanguage_dual_model_ablation_v1"
)


class GpuCoordinationBlocked(RuntimeError):
    """Abort after cleanup while preserving a structured coordination result."""

    def __init__(
        self,
        output: Path,
        status: str,
        evidence: dict[str, object],
    ) -> None:
        super().__init__(status)
        self.output = output
        self.status = status
        self.evidence = evidence


def _prepare_output(output: Path, *, resume: bool) -> None:
    resolved = output.resolve()
    stable = STABLE.resolve()
    if resolved == stable or stable in resolved.parents:
        raise RuntimeError("output_root_must_not_use_stable_checkout")
    if not resolved.exists():
        resolved.mkdir(parents=True, exist_ok=False)
        return
    if not resume or not resolved.is_dir():
        raise RuntimeError("output_root_exists_without_safe_resume")
    result_path = resolved / "live_production_gpu_results.json"
    try:
        raw: object = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("resume_record_missing_or_invalid") from exc
    safe_statuses = {
        "BLOCKED_BY_SPECDEC_ACTIVE",
        "BLOCKED_BY_UNCERTAIN_GPU_OWNERSHIP",
        "YIELDED_TO_SPECDEC",
    }
    if not isinstance(raw, dict) or raw.get("status") not in safe_statuses:
        raise RuntimeError("resume_record_not_retry_safe")
    if tuple(resolved.rglob("owned_profile_process.json")):
        raise RuntimeError("resume_ownership_record_present")


def _validate_static_preflight(
    *,
    stable_clean: bool,
    artifacts_available: bool,
    occupied_endpoints: tuple[str, ...],
    runtime_paths_available: bool,
) -> None:
    if not stable_clean:
        raise RuntimeError("stable_checkout_not_clean")
    if not artifacts_available:
        raise RuntimeError("local_model_artifact_verification_failed")
    if occupied_endpoints:
        raise RuntimeError("unrelated_target_endpoint_active")
    if not runtime_paths_available:
        raise RuntimeError("required_local_runtime_path_missing")


def _manifest(output: Path, *, status: str) -> dict[str, object]:
    files = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == "run_manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(output).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "schema_version": 1,
        "status": status,
        "repository_revision": subprocess.run(  # nosec B603 B607
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        ).stdout.strip(),
        "files": files,
        "production_state_modified": False,
    }


def _write_terminal_result(
    output: Path,
    *,
    status: str,
    evidence: dict[str, object],
) -> None:
    result = {
        "schema_version": 1,
        "status": status,
        "coordination": evidence,
        "model_servers_started": False,
        "production_state_modified": False,
        "unrelated_processes_preserved": True,
    }
    _write_json(output / "live_production_gpu_results.json", result)
    _write_json(output / "run_manifest.json", _manifest(output, status=status))


def _record_unexpected_failure(output: Path, exc: Exception) -> dict[str, object]:
    quality_down = _endpoint_down("http://127.0.0.1:8201/v1/models")
    codev_down = _endpoint_down(CODEV_ENDPOINT + "/v1/models")
    try:
        observed = observe_gpu()
        final_gpu: dict[str, object] = asdict(observed)
        final_coordination = classify_compute_ownership(observed.compute_pids)
    except (OSError, RuntimeError, ServingRuntimeError) as observation_error:
        final_gpu = {
            "status": "UNAVAILABLE",
            "error_type": type(observation_error).__name__,
        }
        final_coordination = {
            "status": "BLOCKED_BY_UNCERTAIN_GPU_OWNERSHIP",
            "reason": "post_failure_gpu_observation_unavailable",
        }
    result = {
        "schema_version": 1,
        "status": "FAIL",
        "failure_category": type(exc).__name__,
        "model_servers_started": any(output.rglob("*.server.log")),
        "production_state_modified": False,
        "unrelated_processes_preserved": True,
        "safe_shutdown": {
            "status": "PASS" if quality_down and codev_down else "FAIL",
            "quality_endpoint_down": quality_down,
            "codev_endpoint_down": codev_down,
            "final_gpu": final_gpu,
            "final_coordination": final_coordination,
        },
    }
    _write_json(output / "live_production_gpu_results.json", result)
    _write_json(output / "run_manifest.json", _manifest(output, status="FAIL"))
    return result


def _existing_safe_output(argv: Sequence[str] | None) -> Path | None:
    values = list(sys.argv[1:] if argv is None else argv)
    try:
        output = Path(values[values.index("--output-root") + 1]).resolve()
    except (ValueError, IndexError):
        return None
    stable = STABLE.resolve()
    if (
        not output.is_dir()
        or output == stable
        or stable in output.parents
    ):
        return None
    return output


class OwnedCodeV:
    """Own one exact CodeV process group and release only that identity."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.process: subprocess.Popen[str] | None = None
        self.log_handle: TextIO | None = None
        self.log_path = output / "codev.server.log"
        self.pid: int | None = None
        self.process_group_id: int | None = None
        self.start_ticks: int | None = None

    @staticmethod
    def _ticks(pid: int) -> int:
        return int(Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[21])

    def start(self) -> dict[str, object]:
        command = [
            str(VLLM),
            "serve",
            str(CODEV_PATH),
            "--host",
            "127.0.0.1",
            "--port",
            "8103",
            "--served-model-name",
            CODEV_ID,
            "--tensor-parallel-size",
            "1",
            "--max-model-len",
            "16384",
            "--max-num-seqs",
            "1",
            "--gpu-memory-utilization",
            "0.20",
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--enforce-eager",
        ]
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
            }
        }
        environment["LD_LIBRARY_PATH"] = str(FFMPEG)
        environment["PATH"] = (
            str(VLLM.parent)
            + os.pathsep
            + environment.get("PATH", "/usr/local/bin:/usr/bin:/bin")
        )
        self.log_handle = self.log_path.open("a", encoding="utf-8", buffering=1)
        self.process = subprocess.Popen(  # nosec B603 - fixed argv above
            command,
            cwd=self.output,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        self.pid = self.process.pid
        self.process_group_id = os.getpgid(self.pid)
        self.start_ticks = self._ticks(self.pid)
        return {
            "pid": self.pid,
            "process_group_id": self.process_group_id,
            "proc_start_ticks": self.start_ticks,
            "command_sha256": hashlib.sha256(
                json.dumps(command, separators=(",", ":")).encode()
            ).hexdigest(),
            "log_path": str(self.log_path),
        }

    def wait_ready(self, timeout: float = 300) -> dict[str, object]:
        deadline = time.monotonic() + timeout
        last_error = "not_probed"
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"CodeV exited before readiness: {self.process.returncode}"
                )
            try:
                with urllib.request.urlopen(  # nosec B310 - fixed loopback URL
                    CODEV_ENDPOINT + "/v1/models",
                    timeout=5,
                ) as response:
                    raw: object = json.loads(response.read())
                data = raw.get("data") if isinstance(raw, dict) else None
                served = [
                    item.get("id")
                    for item in data
                    if isinstance(item, dict)
                ] if isinstance(data, list) else []
                if CODEV_ID in served:
                    return {
                        "status": "READY_EXACT_MODEL",
                        "model_id": CODEV_ID,
                        "endpoint": CODEV_ENDPOINT,
                    }
                last_error = f"identity_mismatch:{served}"
            except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
                last_error = type(exc).__name__
            time.sleep(1)
        raise RuntimeError(f"CodeV readiness timeout: {last_error}")

    def stop(self) -> dict[str, object]:
        if (
            self.process is None
            or self.pid is None
            or self.process_group_id is None
            or self.start_ticks is None
        ):
            return {"status": "NOT_STARTED"}
        if self.process.poll() is not None:
            return {"status": "ALREADY_EXITED", "pid": self.pid}
        if (
            self._ticks(self.pid) != self.start_ticks
            or os.getpgid(self.pid) != self.process_group_id
        ):
            raise RuntimeError("CodeV PID identity changed; refusing to signal")
        os.killpg(self.process_group_id, signal.SIGTERM)
        try:
            self.process.wait(timeout=45)
            status = "RELEASED_OWNED_CODEV"
        except subprocess.TimeoutExpired:
            os.killpg(self.process_group_id, signal.SIGKILL)
            self.process.wait(timeout=10)
            status = "FORCE_RELEASED_OWNED_CODEV"
        if self.log_handle is not None:
            self.log_handle.close()
        return {"status": status, "pid": self.pid}


def _endpoint_down(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=2)  # nosec B310 - loopback caller only
    except (OSError, urllib.error.URLError):
        return True
    return False


def _wait_gpu_release(
    *,
    maximum_used_mib: int,
    timeout: float = 90,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    last = observe_gpu()
    while time.monotonic() < deadline:
        last = observe_gpu()
        if last.used_mib <= maximum_used_mib:
            return asdict(last)
        time.sleep(1)
    raise RuntimeError(
        f"GPU memory did not release below {maximum_used_mib} MiB; "
        f"last observation was {last.used_mib} MiB"
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _changed_worktree(
    state_root: Path,
    relative_path: str,
    expected_fragment: str,
) -> Path:
    candidates = []
    for path in state_root.rglob(relative_path):
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if "worktrees" in path.parts and expected_fragment in content:
            candidates.append(path.parents[len(Path(relative_path).parts) - 1])
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one changed worktree for {relative_path}, found {len(candidates)}"
        )
    return candidates[0]


def _run_verifier(
    command: Sequence[str],
    *,
    worktree: Path,
    timeout: int = 120,
) -> dict[str, object]:
    allowed_environment = {
        "COMSPEC",
        "LANG",
        "LC_ALL",
        "PATH",
        "PATHEXT",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "WINDIR",
    }
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in allowed_environment
    }
    environment.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    environment.setdefault("LANG", "C.UTF-8")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONUTF8"] = "1"
    completed = subprocess.run(  # nosec B603 - fixed verifier argv
        list(command),
        cwd=worktree,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    return {
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "returncode": completed.returncode,
        "output_tail": (completed.stdout + completed.stderr)[-2_000:],
    }


def _verify_python_worktree(state_root: Path) -> dict[str, object]:
    worktree = _changed_worktree(state_root, "python/value.py", "return 2")
    pytest_result = _run_verifier(
        [sys.executable, "-m", "pytest", "-q", "python/test_value.py"],
        worktree=worktree,
    )
    ruff_result = _run_verifier(
        [sys.executable, "-m", "ruff", "check", "python"],
        worktree=worktree,
    )
    return {
        "status": (
            "PASS"
            if pytest_result["status"] == "PASS"
            and ruff_result["status"] == "PASS"
            else "FAIL"
        ),
        "pytest": pytest_result,
        "ruff": ruff_result,
        "worktree_is_isolated": worktree.is_relative_to(state_root),
    }


def _verify_systemverilog_worktree(state_root: Path) -> dict[str, object]:
    worktree = _changed_worktree(state_root, "rtl/example.sv", "assign y = ~a")
    tools = {
        name: shutil.which(name)
        for name in ("verilator", "iverilog", "vvp", "yosys")
    }
    if any(path is None for path in tools.values()):
        return {
            "status": "FAIL",
            "category": "required_systemverilog_verifier_missing",
            "tools_available": {
                name: path is not None for name, path in tools.items()
            },
        }
    verilator_version = _run_verifier(
        [str(tools["verilator"]), "--version"],
        worktree=worktree,
    )
    if verilator_version["status"] != "PASS":
        return {
            "status": "FAIL",
            "category": "verilator_version_probe_failed",
            "verilator_version": verilator_version,
            "worktree_is_isolated": worktree.is_relative_to(state_root),
        }
    with tempfile.TemporaryDirectory(prefix="laplace-v8-live-sv-") as temporary:
        simulation = Path(temporary) / "tb.out"
        commands = {
            "verilator_lint": [
                str(tools["verilator"]),
                "--lint-only",
                "rtl/example.sv",
            ],
            "iverilog_compile": [
                str(tools["iverilog"]),
                "-g2012",
                "-s",
                "tb_example",
                "-o",
                str(simulation),
                "rtl/example.sv",
                "rtl/tb_example.sv",
            ],
            "simulation": [str(tools["vvp"]), str(simulation)],
            "yosys_synthesis": [
                str(tools["yosys"]),
                "-q",
                "-p",
                "read_verilog -sv rtl/example.sv; synth -top example; stat",
            ],
        }
        results: dict[str, dict[str, object]] = {}
        for name, command in commands.items():
            if name == "simulation" and results["iverilog_compile"]["status"] != "PASS":
                results[name] = {
                    "status": "FAIL",
                    "returncode": None,
                    "output_tail": "compile failed; simulation not started",
                }
                continue
            results[name] = _run_verifier(command, worktree=worktree)
    return {
        "status": (
            "PASS"
            if all(item["status"] == "PASS" for item in results.values())
            and "SYSTEMVERILOG_VERIFY_PASS"
            in str(results["simulation"]["output_tail"])
            else "FAIL"
        ),
        "results": results,
        "verilator_version": verilator_version,
        "verilator_lint_scope": "synthesizable_design",
        "worktree_is_isolated": worktree.is_relative_to(state_root),
    }


def _unexpected_console_errors(
    messages: Sequence[str],
    *,
    provider_failure_start: int,
) -> list[str]:
    if not 0 <= provider_failure_start <= len(messages):
        raise ValueError("provider_failure_console_boundary_invalid")
    unexpected = list(messages[:provider_failure_start])
    unexpected.extend(
        message
        for message in messages[provider_failure_start:]
        if not any(
            marker in message
            for marker in EXPECTED_PROVIDER_FAILURE_CONSOLE_MARKERS
        )
    )
    return unexpected


def _wait_for_account_tier(page: object, expected_tier: str) -> None:
    page.wait_for_function(  # type: ignore[attr-defined]
        "expected => document.querySelector('#account-tier')?.textContent"
        ".startsWith(`${expected} ·`)",
        arg=expected_tier,
        timeout=30_000,
    )


def _probe_liveness_and_readiness(endpoint: str, model_id: str) -> dict[str, object]:
    health_status: int | None = None
    models_status: int | None = None
    served: list[str] = []
    try:
        with urllib.request.urlopen(endpoint + "/health", timeout=10) as response:  # nosec B310
            health_status = response.status
        with urllib.request.urlopen(endpoint + "/v1/models", timeout=10) as response:  # nosec B310
            models_status = response.status
            raw: object = json.loads(response.read())
        data = raw.get("data") if isinstance(raw, dict) else None
        if isinstance(data, list):
            served = [
                str(item["id"])
                for item in data
                if isinstance(item, dict) and isinstance(item.get("id"), str)
            ]
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        pass
    return {
        "status": (
            "PASS"
            if health_status == 200
            and models_status == 200
            and model_id in served
            else "FAIL"
        ),
        "liveness_http_status": health_status,
        "readiness_http_status": models_status,
        "expected_model_id": model_id,
        "served_model_ids": served,
    }


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate ownership, ports, state and artifacts without starting a server.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only a terminal coordination-blocked output directory.",
    )
    arguments = parser.parse_args(argv)
    output = arguments.output_root.resolve()
    _prepare_output(output, resume=arguments.resume)
    server_logs = output / "server_logs"
    screenshots = output / "screenshots"
    server_logs.mkdir(exist_ok=True)
    screenshots.mkdir(exist_ok=True)

    try:
        initial_gpu = observe_gpu()
    except ServingRuntimeError as exc:
        raise GpuCoordinationBlocked(
            output,
            "BLOCKED_BY_UNCERTAIN_GPU_OWNERSHIP",
            {"category": exc.category, "evidence": exc.evidence},
        ) from exc
    initial_coordination = classify_compute_ownership(initial_gpu.compute_pids)
    if initial_coordination["status"] != "GPU_CLEAR":
        raise GpuCoordinationBlocked(
            output,
            str(initial_coordination["status"]),
            initial_coordination,
        )
    stable_status = subprocess.run(  # nosec B603 B607 - fixed read-only Git query
        ["git", "status", "--short"],
        cwd=STABLE,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    artifact_check = validate_local_artifacts(EXPERIMENT)
    occupied_endpoints = [
        endpoint
        for endpoint in (
            "http://127.0.0.1:8201/v1/models",
            CODEV_ENDPOINT + "/v1/models",
        )
        if not _endpoint_down(endpoint)
    ]
    _validate_static_preflight(
        stable_clean=stable_status.returncode == 0
        and not stable_status.stdout.strip(),
        artifacts_available=artifact_check.get("status")
        == "ALL_MODEL_ARTIFACTS_AVAILABLE",
        occupied_endpoints=tuple(occupied_endpoints),
        runtime_paths_available=VLLM.is_file()
        and CODEV_PATH.is_dir()
        and FFMPEG.is_dir(),
    )
    if arguments.preflight_only:
        preflight_result = {
            "schema_version": 1,
            "status": "PREFLIGHT_PASS",
            "initial_gpu": asdict(initial_gpu),
            "coordination": initial_coordination,
            "stable_checkout_clean": True,
            "local_artifacts_verified": True,
            "target_endpoints_free": True,
            "model_servers_started": False,
            "production_state_modified": False,
        }
        _write_json(output / "live_production_gpu_results.json", preflight_result)
        _write_json(
            output / "run_manifest.json",
            _manifest(output, status="PREFLIGHT_PASS"),
        )
        print(json.dumps(preflight_result, indent=2, sort_keys=True))
        return 0

    p1 = ServingProfileOperator(
        ROOT,
        output / "p1_runtime",
        VLLM,
        FFMPEG,
    )
    codev = OwnedCodeV(server_logs)
    operator_process: subprocess.Popen[str] | None = None
    operator_process_group: int | None = None
    operator_start_ticks: int | None = None
    operator_log: TextIO | None = None
    p1_started = False
    p1_release: dict[str, object] = {"status": "NOT_STARTED"}
    codev_release: dict[str, object] = {"status": "NOT_STARTED"}
    operator_stopped = False
    console_errors: list[str] = []
    page_errors: list[str] = []
    result: dict[str, object] = {}

    with tempfile.TemporaryDirectory(prefix="laplace-live-production-") as temporary:
        temporary_root = Path(temporary)
        state_root = temporary_root / "external-state"
        registry = state_root / "auth/registered_users.yaml"
        sessions = state_root / "auth/sessions.sqlite3"
        repository = temporary_root / "authorized-repository"
        revision = _git_fixture(repository)
        bootstrap, admin_code = _admin(
            state_root,
            "bootstrap",
            "--registry",
            str(registry),
            "--session-store",
            str(sessions),
            "--email",
            ADMIN_EMAIL,
            "--user-id",
            ADMIN_USER_ID,
            "--display-name",
            "Live Admin Fixture",
            "--capability-tier",
            "operator",
            "--role",
            "admin",
            "--default-lane",
            "quality",
            *(
                argument
                for capability in ADMIN_CAPABILITIES
                for argument in ("--capability", capability)
            ),
            expect_activation=True,
        )
        added, plus_code = _admin(
            state_root,
            "add",
            "--registry",
            str(registry),
            "--session-store",
            str(sessions),
            "--email",
            PLUS_EMAIL,
            "--user-id",
            "usr_live_plus",
            "--display-name",
            "Live Plus Fixture",
            "--capability-tier",
            "plus",
            "--role",
            "user",
            "--default-lane",
            "economy",
            expect_activation=True,
        )
        basic, basic_code = _admin(
            state_root,
            "add",
            "--registry",
            str(registry),
            "--session-store",
            str(sessions),
            "--email",
            BASIC_EMAIL,
            "--user-id",
            "usr_live_basic",
            "--display-name",
            "Live Basic Fixture",
            "--capability-tier",
            "basic",
            "--role",
            "user",
            "--default-lane",
            "standard",
            expect_activation=True,
        )
        _admin(
            state_root,
            "authorize-repo",
            "--registry",
            str(registry),
            "--email",
            ADMIN_EMAIL,
            "--repo-id",
            REPO_ID,
        )
        _admin(
            state_root,
            "authorize-repo",
            "--registry",
            str(registry),
            "--email",
            PLUS_EMAIL,
            "--repo-id",
            REPO_ID,
        )
        authorizations = RepositoryAuthorizationStore(
            state_root / "tiered_serving/repository_authorizations.sqlite3"
        )
        authorizations.register(REPO_ID, repository)
        authorizations.grant(ADMIN_USER_ID, REPO_ID, base_revision=revision)
        authorizations.grant("usr_live_plus", REPO_ID, base_revision=revision)
        corpus_store = PersonalCorpusStore(state_root)
        corpus_id = str(
            corpus_store.create_corpus(
                ADMIN_USER_ID,
                "Live private references",
            )["corpus_id"]
        )
        upload_id = str(
            corpus_store.create_upload(
                ADMIN_USER_ID,
                corpus_id,
                idempotency_key="upload:live-private-reference",
            )["upload_id"]
        )
        staged = corpus_store.stage_file(
            ADMIN_USER_ID,
            upload_id,
            logical_path="notes/live-evidence.md",
            content=(
                b"# Live evidence\n"
                b"The private calibration marker is LAP-V8-PRIVATE-2718.\n"
            ),
            client_mime="text/markdown",
        )
        indexed = corpus_store.index_upload(
            ADMIN_USER_ID,
            upload_id,
            idempotency_key="index:live-private-reference",
        )
        personal_search = corpus_store.search(
            ADMIN_USER_ID,
            "private calibration marker",
            corpus_id=corpus_id,
        )
        personal_results = personal_search.get("results")
        if (
            not isinstance(personal_results, list)
            or not personal_results
            or not isinstance(personal_results[0], dict)
            or not isinstance(personal_results[0].get("chunk_id"), str)
        ):
            raise RuntimeError("live personal corpus retrieval fixture was not indexed")
        personal_chunk_id = str(personal_results[0]["chunk_id"])
        admin_password = f"live-admin-{secrets.token_urlsafe(64)}"
        plus_password = f"live-plus-{secrets.token_urlsafe(64)}"
        basic_password = f"live-basic-{secrets.token_urlsafe(64)}"
        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"
        operator_log = (server_logs / "operator.server.log").open(
            "a",
            encoding="utf-8",
            buffering=1,
        )

        try:
            p1_start = p1.start("P1_fp8_kv")
            p1_started = True
            p1_ready_gpu = observe_gpu()
            owned_process = p1_start.get("owned_process")
            if not isinstance(owned_process, dict) or not isinstance(
                owned_process.get("pid"), int
            ):
                raise RuntimeError("quality owned process identity missing")
            p1_root_pid = int(owned_process["pid"])
            p1_ready_coordination = classify_compute_ownership(
                p1_ready_gpu.compute_pids,
                allowed_laplace_roots=(p1_root_pid,),
            )
            if p1_ready_coordination["status"] != "GPU_CLEAR_LAPLACE_OWNED_ONLY":
                status = str(p1_ready_coordination["status"])
                if status == "BLOCKED_BY_SPECDEC_ACTIVE":
                    status = "YIELDED_TO_SPECDEC"
                raise GpuCoordinationBlocked(
                    output,
                    status,
                    p1_ready_coordination,
                )
            p1_readiness = _probe_liveness_and_readiness(
                "http://127.0.0.1:8201",
                "laplace-quality-p1",
            )
            streamed = VllmProvider(
                "http://127.0.0.1:8201",
                "laplace-quality-p1",
                max_tokens=64,
                temperature=0,
                timeout_seconds=180,
            ).generate(
                "Reply with exactly the text LIVE_STREAM_PASS.",
                enable_thinking=False,
            )
            p1_streaming = {
                "status": (
                    "PASS"
                    if streamed.text.strip()
                    and streamed.ttft_seconds is not None
                    and streamed.completion_tokens is not None
                    else "FAIL"
                ),
                "transport": "vllm_sse",
                "time_to_first_token_observed": streamed.ttft_seconds is not None,
                "server_completion_tokens_reported": (
                    streamed.completion_tokens is not None
                ),
                "finish_reason": streamed.finish_reason,
            }
            operator_process = subprocess.Popen(  # nosec B603 - fixed local module
                [
                    sys.executable,
                    "-m",
                    "research_workspace.operator_server",
                    "--repository-root",
                    str(ROOT),
                    "--state-root",
                    str(state_root),
                    "--user-registry",
                    str(registry),
                    "--session-store",
                    str(sessions),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--allowed-origin",
                    base_url,
                    "--allowed-origin",
                    f"http://localhost:{port}",
                    "--allowed-host",
                    "127.0.0.1",
                    "--allowed-host",
                    "localhost",
                ],
                cwd=ROOT,
                env=_environment(),
                stdout=operator_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            operator_process_group = os.getpgid(operator_process.pid)
            operator_start_ticks = int(
                Path(f"/proc/{operator_process.pid}/stat")
                .read_text(encoding="utf-8")
                .split()[21]
            )
            _wait_for_server(base_url, operator_process)

            with sync_playwright() as runtime:
                browser = runtime.chromium.launch(
                    headless=True,
                    executable_path=str(_chromium()),
                )
                context = browser.new_context(
                    viewport={"width": 1440, "height": 1000},
                )
                context.grant_permissions(
                    ["clipboard-read", "clipboard-write"],
                    origin=base_url,
                )
                page = context.new_page()
                page.on(
                    "console",
                    lambda message: console_errors.append(message.text)
                    if message.type == "error"
                    else None,
                )
                page.on("pageerror", lambda error: page_errors.append(str(error)))
                page.goto(base_url, wait_until="networkidle")
                _activate(page, ADMIN_EMAIL, admin_code, admin_password)
                page.locator("#account-tier").get_by_text(
                    "operator · admin",
                    exact=True,
                ).wait_for()
                page.get_by_role("button", name="Knowledge", exact=True).click()
                page.get_by_text("Live private references", exact=True).wait_for()
                page.get_by_text("notes/live-evidence.md", exact=True).wait_for()
                page.screenshot(
                    path=screenshots / "live_personal_corpus.png",
                    full_page=True,
                )
                page.get_by_role("button", name="Chat", exact=True).click()
                page.locator("#chat-lane").select_option("quality")
                page.locator("#chat-domain").select_option("general")
                page.locator("#chat-retrieval").select_option("selected_personal")
                page.locator("#chat-message").fill(
                    "Reply concisely in Markdown with the heading 'Live Quality', "
                    "state the private calibration marker from the selected personal "
                    "corpus, cite the exact file and chunk identifier, and include a "
                    "fenced Python code block containing `result = \"quality-pass\"`."
                )
                page.get_by_role("button", name="Send", exact=True).click()
                page.wait_for_function(
                    "states => states.some((value) => "
                    "document.querySelector('#chat-state')?.textContent.startsWith(value))",
                    arg=CHAT_TERMINAL_STATES,
                    timeout=300_000,
                )
                if page.locator("#chat-state").inner_text().startswith("FAILED"):
                    raise RuntimeError("live Quality chat failed")
                quality_card = page.locator(".message-card.assistant")
                quality_card.wait_for(timeout=300_000)
                quality_text = quality_card.inner_text()
                quality_code_count = quality_card.locator(".code-block").count()
                quality_card.get_by_role("button", name="Copy code").click()
                quality_card.get_by_role("button", name="Copied").wait_for()
                page.get_by_text("Response details", exact=True).click()
                quality_details = page.locator(".metadata-grid").inner_text()
                page.screenshot(
                    path=screenshots / "live_quality_chat.png",
                    full_page=True,
                )

                page.get_by_role("button", name="Agent", exact=True).click()
                page.locator("#agent-repository").select_option(REPO_ID)
                page.locator("#agent-form select[name=lane]").select_option(
                    "quality"
                )
                page.locator("#agent-domain").select_option("python")
                page.locator("#agent-form textarea[name=instruction]").fill(
                    "The file python/value.py contains a function with `return 1`. "
                    "Modify only that exact text to `return 2`. Return the requested "
                    "strict JSON edit object with path python/value.py, exact old "
                    "text, and exact replacement text."
                )
                page.get_by_role(
                    "button",
                    name="Start isolated agent",
                ).click()
                page.wait_for_function(
                    "() => ['Complete', 'Failed'].includes("
                    "document.querySelector('#agent-state')?.textContent.trim())",
                    timeout=300_000,
                )
                if page.locator("#agent-state").inner_text() != "Complete":
                    raise RuntimeError("live Quality Python agent failed")
                python_diff_text = page.locator("#agent-diff").inner_text()
                python_tests_text = page.locator("#agent-tests").inner_text()
                python_plan_text = page.locator("#agent-plan").inner_text()
                python_verification = _verify_python_worktree(state_root)
                page.screenshot(
                    path=screenshots / "live_python_agent.png",
                    full_page=True,
                )

                p1_after_group_gpu = observe_gpu()
                p1_after_group_coordination = classify_compute_ownership(
                    p1_after_group_gpu.compute_pids,
                    allowed_laplace_roots=(p1_root_pid,),
                )
                if (
                    p1_after_group_coordination["status"]
                    != "GPU_CLEAR_LAPLACE_OWNED_ONLY"
                ):
                    status = str(p1_after_group_coordination["status"])
                    if status == "BLOCKED_BY_SPECDEC_ACTIVE":
                        status = "YIELDED_TO_SPECDEC"
                    raise GpuCoordinationBlocked(
                        output,
                        status,
                        p1_after_group_coordination,
                    )
                p1_release = p1.stop()
                p1_started = False
                after_p1_release = _wait_gpu_release(maximum_used_mib=4_000)
                after_p1_snapshot = observe_gpu()
                before_codev_coordination = classify_compute_ownership(
                    after_p1_snapshot.compute_pids
                )
                if before_codev_coordination["status"] != "GPU_CLEAR":
                    status = str(before_codev_coordination["status"])
                    if status == "BLOCKED_BY_SPECDEC_ACTIVE":
                        status = "YIELDED_TO_SPECDEC"
                    raise GpuCoordinationBlocked(
                        output,
                        status,
                        before_codev_coordination,
                    )
                p1_endpoint_down = _endpoint_down(
                    "http://127.0.0.1:8201/v1/models"
                )
                codev_owned = codev.start()
                codev_ready = codev.wait_ready()
                codev_ready_gpu = observe_gpu()
                codev_ready_coordination = classify_compute_ownership(
                    codev_ready_gpu.compute_pids,
                    allowed_laplace_roots=(int(codev_owned["pid"]),),
                )
                if (
                    codev_ready_coordination["status"]
                    != "GPU_CLEAR_LAPLACE_OWNED_ONLY"
                ):
                    status = str(codev_ready_coordination["status"])
                    if status == "BLOCKED_BY_SPECDEC_ACTIVE":
                        status = "YIELDED_TO_SPECDEC"
                    raise GpuCoordinationBlocked(
                        output,
                        status,
                        codev_ready_coordination,
                    )
                codev_readiness = _probe_liveness_and_readiness(
                    CODEV_ENDPOINT,
                    CODEV_ID,
                )

                page.get_by_role("button", name="Chat", exact=True).click()
                page.locator("#chat-lane").select_option("economy")
                page.locator("#chat-domain").select_option("systemverilog")
                page.locator("#chat-message").fill(
                    "Reply concisely in Markdown with heading 'Live CodeV' and a "
                    "fenced SystemVerilog module named pass that assigns y = ~a."
                )
                page.get_by_role("button", name="Send", exact=True).click()
                page.wait_for_function(
                    "states => states.some((value) => "
                    "document.querySelector('#chat-state')?.textContent.startsWith(value))",
                    arg=CHAT_TERMINAL_STATES,
                    timeout=300_000,
                )
                if page.locator("#chat-state").inner_text().startswith("FAILED"):
                    raise RuntimeError("live CodeV chat failed")
                page.locator(".message-card.assistant").last.wait_for(
                    timeout=300_000
                )
                codev_card = page.locator(".message-card.assistant").last
                codev_text = codev_card.inner_text()
                codev_code_count = codev_card.locator(".code-block").count()
                codev_card.get_by_role("button", name="Copy code").click()
                codev_card.get_by_role("button", name="Copied").wait_for()
                details = page.get_by_text("Response details", exact=True)
                details.last.click()
                codev_details = page.locator(".metadata-grid").last.inner_text()
                page.screenshot(
                    path=screenshots / "live_codev_chat.png",
                    full_page=True,
                )

                page.get_by_role("button", name="Sign out", exact=True).click()
                page.locator("#auth-dialog").wait_for(state="visible")
                _activate(page, PLUS_EMAIL, plus_code, plus_password)
                page.get_by_role("button", name="Knowledge", exact=True).click()
                page.get_by_text(
                    "Your personal corpus is empty",
                    exact=True,
                ).wait_for()
                cross_user_isolation = (
                    page.get_by_text(
                        "Live private references",
                        exact=True,
                    ).count()
                    == 0
                )
                page.get_by_role("button", name="Agent", exact=True).click()
                page.locator("#agent-repository").select_option(REPO_ID)
                page.locator("#agent-form select[name=lane]").select_option(
                    "economy"
                )
                page.locator("#agent-domain").select_option("systemverilog")
                page.locator("#agent-form textarea[name=instruction]").fill(
                    "The file rtl/example.sv contains `assign y = a;`. Modify only "
                    "that line to `assign y = ~a;`. Return the requested strict JSON "
                    "edit object with path rtl/example.sv, exact old text, and exact "
                    "replacement text."
                )
                page.get_by_role(
                    "button",
                    name="Start isolated agent",
                ).click()
                page.wait_for_function(
                    "() => ['Complete', 'Failed'].includes("
                    "document.querySelector('#agent-state')?.textContent.trim())",
                    timeout=300_000
                )
                if page.locator("#agent-state").inner_text() != "Complete":
                    raise RuntimeError("live Plus agent failed")
                diff_text = page.locator("#agent-diff").inner_text()
                tests_text = page.locator("#agent-tests").inner_text()
                codev_plan_text = page.locator("#agent-plan").inner_text()
                systemverilog_verification = _verify_systemverilog_worktree(
                    state_root
                )
                page.screenshot(
                    path=screenshots / "live_plus_agent.png",
                    full_page=True,
                )
                storage_entries = page.evaluate(
                    "() => Object.keys(localStorage).length + "
                    "Object.keys(sessionStorage).length"
                )
                session_cookie = next(
                    cookie
                    for cookie in context.cookies()
                    if cookie["name"] == "laplace_session"
                )
                codev_after_group_gpu = observe_gpu()
                codev_after_group_coordination = classify_compute_ownership(
                    codev_after_group_gpu.compute_pids,
                    allowed_laplace_roots=(int(codev_owned["pid"]),),
                )
                if (
                    codev_after_group_coordination["status"]
                    != "GPU_CLEAR_LAPLACE_OWNED_ONLY"
                ):
                    status = str(codev_after_group_coordination["status"])
                    if status == "BLOCKED_BY_SPECDEC_ACTIVE":
                        status = "YIELDED_TO_SPECDEC"
                    raise GpuCoordinationBlocked(
                        output,
                        status,
                        codev_after_group_coordination,
                    )
                page.get_by_role("button", name="Chat", exact=True).click()
                page.locator("#chat-lane").select_option("economy")
                page.locator("#chat-domain").select_option("systemverilog")
                page.locator("#chat-retrieval").select_option("none")
                page.locator("#chat-message").fill(
                    "Generate a detailed 1000-word explanation of ready/valid "
                    "handshakes so this bounded request can be cancelled."
                )
                page.get_by_role("button", name="Send", exact=True).click()
                page.locator("#stop-chat").wait_for(state="visible", timeout=10_000)
                page.locator("#stop-chat").click()
                page.wait_for_function(
                    "() => document.querySelector('#chat-state')?.textContent"
                    ".startsWith('CANCELLED')",
                    timeout=30_000,
                )
                cancellation_status = page.locator("#chat-state").inner_text()

                codev_release = codev.stop()
                if not _endpoint_down(CODEV_ENDPOINT + "/v1/models"):
                    raise RuntimeError("CodeV endpoint remained open after owned stop")
                provider_failure_console_start = len(console_errors)
                page.locator("#chat-message").fill(
                    "Reply with PROVIDER_FAILURE_SHOULD_NOT_SUCCEED."
                )
                page.get_by_role("button", name="Send", exact=True).click()
                page.wait_for_function(
                    "() => document.querySelector('#chat-state')?.textContent"
                    ".startsWith('FAILED')",
                    timeout=30_000,
                )
                provider_failure_status = page.locator("#chat-state").inner_text()
                unexpected_console_errors = _unexpected_console_errors(
                    console_errors,
                    provider_failure_start=provider_failure_console_start,
                )
                expected_provider_failure_console_error_count = sum(
                    any(
                        marker in message
                        for marker in EXPECTED_PROVIDER_FAILURE_CONSOLE_MARKERS
                    )
                    for message in console_errors[provider_failure_console_start:]
                )

                page.get_by_role("button", name="Sign out", exact=True).click()
                page.locator("#auth-dialog").wait_for(state="visible")
                _activate(page, BASIC_EMAIL, basic_code, basic_password)
                _wait_for_account_tier(page, "basic")
                basic_capability_enforcement = (
                    page.get_by_role("button", name="Agent", exact=True).count() == 0
                    and page.get_by_role(
                        "button",
                        name="Knowledge",
                        exact=True,
                    ).count()
                    == 0
                )
                browser.close()

            registry_text = registry.read_text(encoding="utf-8")
            audit_text = (
                state_root / "auth/authentication_audit.jsonl"
            ).read_text(encoding="utf-8")
            secrets_absent = all(
                secret not in registry_text and secret not in audit_text
                for secret in (
                    admin_code,
                    plus_code,
                    basic_code,
                    admin_password,
                    plus_password,
                    basic_password,
                )
            )
            checks = {
                "stable_checkout_clean": not stable_status.stdout.strip(),
                "local_artifacts_verified": True,
                "initial_gpu_has_no_compute_pids": not initial_gpu.compute_pids,
                "first_account_exact": bootstrap.get("email") == ADMIN_EMAIL
                and bootstrap.get("capability_tier") == "operator"
                and bootstrap.get("role") == "admin"
                and bootstrap.get("default_lane") == "quality",
                "quality_profile_exact": p1_start.get("profile_id") == "P1_fp8_kv"
                and p1_start.get("status") == "STARTED_READY",
                "quality_liveness_and_readiness_distinct": p1_readiness.get(
                    "status"
                )
                == "PASS",
                "quality_real_sse_streaming": p1_streaming.get("status") == "PASS",
                "quality_chat_readable": "Live Quality" in quality_text,
                "quality_code_block_and_copy": quality_code_count > 0,
                "quality_response_details": "laplace-quality-p1"
                in quality_details,
                "personal_corpus_indexed": staged.get("state") == "ACCEPTED"
                and indexed.get("status") == "INDEXED",
                "personal_corpus_retrieval_snapshot": personal_chunk_id
                in quality_details
                and "notes/live-evidence.md" in quality_details,
                "personal_corpus_answer_citations": "LAP-V8-PRIVATE-2718"
                in quality_text
                and "notes/live-evidence.md" in quality_text
                and personal_chunk_id in quality_text,
                "python_agent_structured_patch": "python/value.py"
                in python_diff_text
                and "return 2" in python_diff_text
                and "PASSED" in python_tests_text
                and "laplace-quality-p1" in python_plan_text,
                "python_verification": python_verification.get("status")
                == "PASS",
                "p1_released_before_codev": p1_release.get("status")
                == "RELEASED_OWNED_PROFILE"
                and p1_endpoint_down,
                "codev_exact_identity": codev_ready.get("model_id") == CODEV_ID,
                "codev_liveness_and_readiness_distinct": codev_readiness.get(
                    "status"
                )
                == "PASS",
                "codev_chat_readable": "Live CodeV" in codev_text,
                "codev_code_block_and_copy": codev_code_count > 0,
                "codev_response_details": CODEV_ID in codev_details,
                "plus_account_registered": added.get("capability_tier") == "plus",
                "basic_account_registered": basic.get("capability_tier") == "basic",
                "plus_diff_readable": "rtl/example.sv" in diff_text
                and "assign y = ~a" in diff_text,
                "plus_verification_readable": "PASSED" in tests_text,
                "codev_specialist_structured_routing": CODEV_ID
                in codev_plan_text,
                "systemverilog_verification": systemverilog_verification.get(
                    "status"
                )
                == "PASS",
                "cross_user_corpus_isolation": cross_user_isolation,
                "basic_capability_enforcement": basic_capability_enforcement,
                "queue_behavior_recorded": "Queue wait" in quality_details
                and "Queue position" in quality_details,
                "bounded_cancellation": cancellation_status.startswith("CANCELLED"),
                "provider_failure_handling": provider_failure_status.startswith(
                    "FAILED"
                ),
                "browser_credential_storage_empty": storage_entries == 0,
                "opaque_http_only_session": session_cookie["httpOnly"] is True
                and len(str(session_cookie["value"])) >= 22,
                "credentials_absent_from_registry_and_audit": secrets_absent,
                "no_browser_errors": not unexpected_console_errors
                and not page_errors,
            }
            result = {
                "schema_version": 1,
                "status": "PASS" if all(checks.values()) else "FAIL",
                "checks": checks,
                "profiles_run_sequentially": True,
                "specdec_coordination_status": (
                    "PASS_NO_PROTECTED_WORKLOAD_OBSERVED"
                ),
                "coordination": {
                    "initial": initial_coordination,
                    "quality_ready": p1_ready_coordination,
                    "quality_after_group": p1_after_group_coordination,
                    "before_codev": before_codev_coordination,
                    "codev_ready": codev_ready_coordination,
                    "codev_after_group": codev_after_group_coordination,
                },
                "reason_for_sequential_routes": (
                    "P1 reserves 90% GPU memory; CodeV is loaded on demand after "
                    "P1 release so only one generative route is resident."
                ),
                "initial_gpu": asdict(initial_gpu),
                "p1": {
                    "startup": p1_start,
                    "liveness_and_readiness": p1_readiness,
                    "real_streaming": p1_streaming,
                    "gpu_at_ready": asdict(p1_ready_gpu),
                    "coordination_at_ready": p1_ready_coordination,
                    "coordination_after_group": p1_after_group_coordination,
                    "release": p1_release,
                    "gpu_after_release": after_p1_release,
                    "python_verification": python_verification,
                    "personal_corpus": {
                        "status": "PASS",
                        "source": "notes/live-evidence.md",
                        "chunk_id": personal_chunk_id,
                        "snapshot_revision": indexed.get("snapshot_revision"),
                    },
                },
                "codev": {
                    "owned_process": codev_owned,
                    "readiness": codev_ready,
                    "liveness_and_readiness": codev_readiness,
                    "gpu_at_ready": asdict(codev_ready_gpu),
                    "coordination_at_ready": codev_ready_coordination,
                    "coordination_after_group": codev_after_group_coordination,
                    "systemverilog_verification": systemverilog_verification,
                    "cancellation": cancellation_status,
                    "provider_failure": provider_failure_status,
                },
                "registered_account": {
                    "email": ADMIN_EMAIL,
                    "capability_tier": "operator",
                    "role": "admin",
                    "default_lane": "quality",
                },
                "console_error_count": len(console_errors),
                "expected_provider_failure_console_error_count": (
                    expected_provider_failure_console_error_count
                ),
                "unexpected_console_error_count": len(unexpected_console_errors),
                "page_error_count": len(page_errors),
                "screenshots": [
                    "screenshots/live_personal_corpus.png",
                    "screenshots/live_quality_chat.png",
                    "screenshots/live_python_agent.png",
                    "screenshots/live_codev_chat.png",
                    "screenshots/live_plus_agent.png",
                ],
            }
            admin_code = plus_code = basic_code = ""
            admin_password = plus_password = basic_password = ""
        finally:
            if p1_started:
                try:
                    p1_release = p1.stop()
                except ServingRuntimeError as exc:
                    p1_release = {
                        "status": "RELEASE_FAILED",
                        "category": exc.category,
                    }
            if codev_release.get("status") == "NOT_STARTED":
                try:
                    codev_release = codev.stop()
                except (OSError, RuntimeError) as exc:
                    codev_release = {
                        "status": "RELEASE_FAILED",
                        "error_type": type(exc).__name__,
                    }
            if operator_process is not None and operator_process.poll() is None:
                if (
                    operator_process_group is not None
                    and operator_start_ticks is not None
                    and Path(f"/proc/{operator_process.pid}").exists()
                    and int(
                        Path(f"/proc/{operator_process.pid}/stat")
                        .read_text(encoding="utf-8")
                        .split()[21]
                    )
                    == operator_start_ticks
                    and os.getpgid(operator_process.pid)
                    == operator_process_group
                ):
                    os.killpg(operator_process_group, signal.SIGTERM)
                    try:
                        operator_process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(operator_process_group, signal.SIGKILL)
                        operator_process.wait(timeout=10)
            operator_stopped = (
                operator_process is None or operator_process.poll() is not None
            )
            if operator_log is not None:
                operator_log.close()

    final_gpu = _wait_gpu_release(maximum_used_mib=4_000)
    final_coordination = classify_compute_ownership(
        tuple(int(pid) for pid in final_gpu["compute_pids"])
    )
    codev_down = _endpoint_down(CODEV_ENDPOINT + "/v1/models")
    quality_down = _endpoint_down("http://127.0.0.1:8201/v1/models")
    safe_shutdown = {
        "status": (
            "PASS"
            if operator_stopped
            and codev_down
            and quality_down
            and codev_release.get("status") == "RELEASED_OWNED_CODEV"
            else "FAIL"
        ),
        "operator_stopped": operator_stopped,
        "quality_endpoint_down": quality_down,
        "codev_endpoint_down": codev_down,
        "p1_release": p1_release,
        "codev_release": codev_release,
        "final_gpu": final_gpu,
        "final_coordination": final_coordination,
        "unrelated_processes_preserved": True,
    }
    result["safe_shutdown"] = safe_shutdown
    result["status"] = (
        "PASS"
        if result.get("status") == "PASS" and safe_shutdown["status"] == "PASS"
        else "FAIL"
    )
    if final_coordination["status"] == "BLOCKED_BY_SPECDEC_ACTIVE":
        result["status"] = "YIELDED_TO_SPECDEC"
    elif final_coordination["status"] == "BLOCKED_BY_UNCERTAIN_GPU_OWNERSHIP":
        result["status"] = "BLOCKED_BY_UNCERTAIN_GPU_OWNERSHIP"
    _write_json(output / "live_production_gpu_results.json", result)
    _write_json(output / "run_manifest.json", _manifest(output, status=str(result["status"])))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _main(argv)
    except GpuCoordinationBlocked as exc:
        _write_terminal_result(
            exc.output,
            status=exc.status,
            evidence=exc.evidence,
        )
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": exc.status,
                    "coordination": exc.evidence,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 3
    except Exception as exc:
        output = _existing_safe_output(argv)
        if output is None:
            raise
        result = _record_unexpected_failure(output, exc)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
