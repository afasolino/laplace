#!/usr/bin/env python3
"""Run a pre-promotion co-resident Qwen3.8/CodeV production routing gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess  # nosec B404 - fixed local commands
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from research_workspace.production_model import verify_qwen38_artifact
from research_workspace.serving_profile_runtime import (
    ServingProfileRuntime,
    ServingRuntimeError,
    observe_gpu,
)
from research_workspace.serving_profiles import (
    InstalledServingCapabilities,
    ServingProfile,
    endpoint_for,
    resolve_profile,
)


ROOT = Path(__file__).resolve().parents[1]
VLLM = ROOT / ".venv-vllm-cu129/bin/vllm"
FFMPEG = ROOT / ".runtime/ffmpeg7/lib"
OPERATOR_PYTHON = ROOT / ".venv/bin/python"
PROFILE_IDS = ("P8_qwen38_w4a16_mtp",)
MANDATORY_PROFILE_GATES = (
    "model_identity",
    "normal_inference",
    "streaming",
    "reasoning",
    "tool_calling",
    "multi_turn",
    "cancellation",
    "context_window",
    "runtime_stability",
    "quantized_kernel",
    "gpu_headroom",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_id", choices=PROFILE_IDS)
    parser.add_argument("--profile-certification", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--vllm", type=Path, default=VLLM)
    parser.add_argument("--ffmpeg-lib", type=Path, default=FFMPEG)
    parser.add_argument("--operator-python", type=Path, default=OPERATOR_PYTHON)
    return parser


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path, error: str) -> dict[str, object]:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(error)
    return raw


def _profile(path: Path) -> ServingProfile:
    return ServingProfile.from_mapping(_load_object(path, "candidate_profile_malformed"))


def _capabilities(vllm: Path, ffmpeg: Path) -> InstalledServingCapabilities:
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = str(ffmpeg)
    version = subprocess.run(  # nosec B603
        [str(vllm), "--version"],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
        env=environment,
    ).stdout.strip()
    help_text = subprocess.run(  # nosec B603
        [str(vllm), "serve", "--help=all"],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
        env=environment,
    ).stdout
    return InstalledServingCapabilities.from_help(version=version, help_text=help_text)


def _git_revision(root: Path) -> str:
    return subprocess.run(  # nosec B603 B607
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    ).stdout.strip()


def validate_profile_certification(
    certification_path: Path,
    *,
    profile_id: str,
    profile_sha256: str,
    artifact_sha256: str,
    repository_revision: str,
) -> dict[str, object]:
    """Bind the co-resident gate to one complete, released profile run."""

    certification = _load_object(certification_path, "profile_certification_malformed")
    required = [*MANDATORY_PROFILE_GATES]
    if profile_id == "P8_qwen38_w4a16_mtp":
        required.append("mtp")
    gates = certification.get("gates")
    valid_gates = isinstance(gates, dict) and all(
        isinstance(gates.get(name), dict) and gates[name].get("status") == "PASS"
        for name in required
    )
    release = certification.get("release")
    if (
        certification.get("schema_version") != 1
        or certification.get("status") != "PASSED"
        or certification.get("profile_id") != profile_id
        or certification.get("profile_sha256") != profile_sha256
        or certification.get("artifact_sha256") != artifact_sha256
        or certification.get("repository_revision") != repository_revision
        or not valid_gates
        or not isinstance(release, dict)
        or release.get("status") != "RELEASED_OWNED_PROFILE"
        or certification.get("endpoint_down_after_release") is not True
        or certification.get("unrelated_processes_signalled") is not False
    ):
        raise RuntimeError("profile_certification_not_eligible_for_production_gate")
    return certification


def _selection_path(root: Path, profile_id: str) -> Path:
    if profile_id != "P8_qwen38_w4a16_mtp":
        raise RuntimeError("unsupported_production_profile")
    return root / "configs/selected_serving_profiles.json"


def _validate_staged_routes(
    selected: dict[str, object], profile: ServingProfile
) -> dict[str, dict[str, object]]:
    routes = selected.get("routes")
    if not isinstance(routes, dict):
        raise RuntimeError("staged_routes_invalid")
    quality = routes.get("quality")
    standard = routes.get("standard")
    economy = routes.get("economy")
    if (
        not isinstance(quality, dict)
        or not isinstance(standard, dict)
        or not isinstance(economy, dict)
    ):
        raise RuntimeError("staged_routes_invalid")
    endpoint = endpoint_for(profile)
    if (
        selected.get("schema_version") != 1
        or selected.get("default_profile_id") != profile.profile_id
        or selected.get("high_context_profile_id") != profile.profile_id
        or quality.get("model_id") != profile.served_model_name
        or standard.get("model_id") != profile.served_model_name
        or quality.get("endpoint") != endpoint
        or standard.get("endpoint") != endpoint
    ):
        raise RuntimeError("staged_routes_not_qwen38_quality")
    return {"quality": quality, "standard": standard, "economy": economy}


def _endpoint_down(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=2)  # nosec B310 - loopback URLs only
    except (OSError, urllib.error.URLError):
        return True
    return False


def _models(endpoint: str) -> list[str]:
    with urllib.request.urlopen(  # nosec B310 - checked loopback endpoints
        endpoint.rstrip("/") + "/v1/models", timeout=10
    ) as response:
        raw: object = json.loads(response.read())
    data = raw.get("data") if isinstance(raw, dict) else None
    return (
        [
            str(item["id"])
            for item in data
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]
        if isinstance(data, list)
        else []
    )


def _wait_model(endpoint: str, model: str, timeout: float) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    last_error = "not_probed"
    while time.monotonic() < deadline:
        try:
            identities = _models(endpoint)
            if identities == [model]:
                return {"status": "READY_EXACT_MODEL", "served": identities}
            last_error = f"identity_mismatch:{identities}"
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = type(exc).__name__
        time.sleep(1)
    raise RuntimeError(f"model_readiness_timeout:{model}:{last_error}")


def _chat(endpoint: str, model: str, marker: str, *, rtl: bool = False) -> dict[str, object]:
    prompt = (
        f"In SystemVerilog, the ready/valid transfer condition is ready && valid. "
        f"Acknowledge by replying exactly {marker}"
        if rtl
        else f"Reply exactly {marker}"
    )
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(  # nosec B310 - checked loopback endpoint
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:  # nosec B310
        value: object = json.loads(response.read())
    choices = value.get("choices") if isinstance(value, dict) else None
    message = choices[0].get("message") if isinstance(choices, list) and choices else None
    content = message.get("content") if isinstance(message, dict) else None
    output = content if isinstance(content, str) else ""
    return {
        "status": "PASS" if marker in output else "FAIL",
        "expected_marker": marker,
        "content_excerpt": output[:500],
        "usage": value.get("usage", {}) if isinstance(value, dict) else {},
    }


def _sample_gpu(stop: threading.Event, samples: list[dict[str, object]]) -> None:
    while not stop.is_set():
        try:
            samples.append(asdict(observe_gpu()))
        except ServingRuntimeError:
            pass
        stop.wait(0.25)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    root = arguments.repository_root.resolve()
    output = arguments.output_root.resolve()
    operator_python = arguments.operator_python.absolute()
    if not operator_python.is_file():
        raise RuntimeError("operator_python_missing")
    output.mkdir(parents=True, exist_ok=False)
    repository_revision = _git_revision(root)
    artifact = verify_qwen38_artifact(root)
    profile_path = root / "configs/serving_profiles" / f"{arguments.profile_id}.json"
    profile = _profile(profile_path)
    profile_model_path = Path(profile.model_path)
    profile_model_path = (
        profile_model_path.resolve()
        if profile_model_path.is_absolute()
        else (root / profile_model_path).resolve()
    )
    artifact_model_path = Path(str(artifact["artifact_path"]))
    artifact_model_path = (
        artifact_model_path.resolve()
        if artifact_model_path.is_absolute()
        else (root / artifact_model_path).resolve()
    )
    if profile_model_path != artifact_model_path:
        raise RuntimeError("candidate_profile_artifact_mismatch")
    profile_certification_path = arguments.profile_certification.resolve(strict=True)
    validate_profile_certification(
        profile_certification_path,
        profile_id=profile.profile_id,
        profile_sha256=_sha256(profile_path),
        artifact_sha256=str(artifact["artifact_sha256"]),
        repository_revision=repository_revision,
    )
    selected = _load_object(_selection_path(root, profile.profile_id), "staged_selection_invalid")
    routes = _validate_staged_routes(selected, profile)
    quality = routes["quality"]
    standard = routes["standard"]
    economy = routes["economy"]
    quality_endpoint = str(quality["endpoint"])
    quality_model = str(quality["model_id"])
    codev_endpoint = str(economy["endpoint"])
    codev_model = str(economy["model_id"])
    target_urls = [quality_endpoint + "/v1/models", codev_endpoint + "/v1/models"]
    if not all(_endpoint_down(url) for url in target_urls):
        raise RuntimeError("target_endpoint_already_owned_elsewhere")

    capabilities = _capabilities(arguments.vllm.resolve(), arguments.ffmpeg_lib.resolve())
    resolved = resolve_profile(
        profile,
        capabilities,
        executable=arguments.vllm.resolve(),
        require_model=True,
        repository_root=root,
    )
    _write_json(output / "resolved_profile.json", resolved.to_json())
    runtime = ServingProfileRuntime(
        output / "quality_runtime",
        residual_free_mib=2_048,
        ffmpeg_library_path=arguments.ffmpeg_lib.resolve(),
    )
    owner_token = "qwen38-production-gate-" + uuid.uuid4().hex
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(root / "src"),
            "LAPLACE_SERVER_OWNER_TOKEN": owner_token,
            "LAPLACE_VLLM_EXECUTABLE": str(arguments.vllm.resolve()),
            "LAPLACE_FFMPEG_LIBRARY_PATH": str(arguments.ffmpeg_lib.resolve()),
            "LD_LIBRARY_PATH": str(arguments.ffmpeg_lib.resolve()),
        }
    )
    initial_gpu = observe_gpu()
    samples: list[dict[str, object]] = [asdict(initial_gpu)]
    stop_sampler = threading.Event()
    sampler = threading.Thread(target=_sample_gpu, args=(stop_sampler, samples), daemon=True)
    sampler.start()
    codev_started = False
    results: dict[str, object] = {}
    quality_release: dict[str, object] = {"status": "NOT_STARTED"}
    codev_release: dict[str, object] = {"status": "NOT_STARTED"}
    try:
        start_codev = subprocess.run(  # nosec B603
            [str(root / "scripts/manage_multilanguage_model_servers.sh"), "start-phase3-worker"],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        results["codev_start"] = {
            "returncode": start_codev.returncode,
            "output_tail": (start_codev.stdout + start_codev.stderr)[-2_000:],
        }
        if start_codev.returncode != 0:
            raise RuntimeError("owned_codev_start_failed")
        codev_started = True
        results["codev_readiness"] = _wait_model(codev_endpoint, codev_model, 600)
        owned = runtime.start(resolved)
        results["quality_readiness"] = runtime.wait_ready(resolved)
        results["co_resident_models"] = {
            "quality": _models(quality_endpoint),
            "codev": _models(codev_endpoint),
        }
        results["quality_route"] = _chat(
            quality_endpoint, quality_model, "QWEN38_QUALITY_ROUTE_PASS"
        )
        results["standard_route"] = _chat(
            str(standard["endpoint"]),
            str(standard["model_id"]),
            "QWEN38_STANDARD_ROUTE_PASS",
        )
        results["economy_rtl_route"] = _chat(
            codev_endpoint, codev_model, "CODEV_ECONOMY_RTL_ROUTE_PASS", rtl=True
        )
        normal_output = output / "normal_laplace"
        normal = subprocess.run(  # nosec B603 - fixed local certification command
            [
                str(operator_python),
                str(root / "scripts/run_tiered_live_api_certification.py"),
                "--repository-root",
                str(root),
                "--output-root",
                str(normal_output),
                "--main-endpoint",
                quality_endpoint,
                "--main-model",
                quality_model,
                "--quality-context-limit",
                str(quality["context_limit"]),
                "--standard-context-limit",
                str(standard["context_limit"]),
                "--codev-endpoint",
                codev_endpoint,
                "--codev-model",
                codev_model,
                "--users",
                "5",
                "--requests",
                "5",
            ],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=3_600,
        )
        normal_report = normal_output / "parallel_api_smoke.json"
        (output / "normal_laplace.stdout.log").write_text(normal.stdout, encoding="utf-8")
        (output / "normal_laplace.stderr.log").write_text(normal.stderr, encoding="utf-8")
        results["normal_laplace"] = {
            "status": "PASS" if normal.returncode == 0 else "FAIL",
            "returncode": normal.returncode,
            "stdout_tail": normal.stdout[-2_000:],
            "stderr_tail": normal.stderr[-2_000:],
            "report_path": str(normal_report),
            "report_sha256": _sha256(normal_report) if normal_report.is_file() else None,
        }
        results["quality_owned_pid"] = owned.pid
    except Exception as exc:  # noqa: BLE001 - terminal production evidence
        results["failure"] = {
            "type": type(exc).__name__,
            "error": str(exc)[:2_000],
            "category": getattr(exc, "category", None),
            "evidence": getattr(exc, "evidence", None),
        }
    finally:
        if runtime.ownership_path.exists():
            try:
                quality_release = runtime.release_owned(timeout_seconds=90)
            except ServingRuntimeError as exc:
                quality_release = {
                    "status": "FAIL",
                    "category": exc.category,
                    "evidence": exc.evidence,
                }
        if codev_started:
            stopped = subprocess.run(  # nosec B603
                [
                    str(root / "scripts/manage_multilanguage_model_servers.sh"),
                    "stop-phase3-worker",
                ],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
            codev_release = {
                "status": "STOPPED_OWNED_CODEV" if stopped.returncode == 0 else "FAIL",
                "returncode": stopped.returncode,
                "output_tail": (stopped.stdout + stopped.stderr)[-2_000:],
            }
        stop_sampler.set()
        sampler.join(timeout=5)

    peak_used = max(int(sample["used_mib"]) for sample in samples)
    minimum_free = int(samples[0]["total_mib"]) - peak_used
    route_names = (
        "quality_route",
        "standard_route",
        "economy_rtl_route",
        "normal_laplace",
    )
    endpoints_down = all(_endpoint_down(url) for url in target_urls)
    passed = (
        all(
            isinstance(results.get(name), dict) and results[name].get("status") == "PASS"
            for name in route_names
        )
        and results.get("co_resident_models")
        == {"quality": [quality_model], "codev": [codev_model]}
        and minimum_free >= 2_048
        and quality_release.get("status") == "RELEASED_OWNED_PROFILE"
        and codev_release.get("status") == "STOPPED_OWNED_CODEV"
        and endpoints_down
    )
    result = {
        "schema_version": 1,
        "status": "PASSED" if passed else "FAILED",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "repository_revision": repository_revision,
        "selected_profile_id": selected["default_profile_id"],
        "staged_selection_sha256": _sha256(_selection_path(root, profile.profile_id)),
        "profile_sha256": _sha256(profile_path),
        "profile_certification_path": str(profile_certification_path),
        "profile_certification_sha256": _sha256(profile_certification_path),
        "artifact_sha256": artifact["artifact_sha256"],
        "routes": routes,
        "results": results,
        "gpu": {
            "initial_used_mib": initial_gpu.used_mib,
            "peak_used_mib": peak_used,
            "minimum_free_headroom_mib": minimum_free,
            "sample_count": len(samples),
            "sampling_interval_seconds": 0.25,
        },
        "release": {"quality": quality_release, "codev": codev_release},
        "target_endpoints_down_after_release": endpoints_down,
        "unrelated_processes_signalled": False,
    }
    _write_json(output / "gpu_samples.json", samples)
    _write_json(output / "production_gate.json", result)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
