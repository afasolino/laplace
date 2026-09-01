#!/usr/bin/env python3
"""Run one admitted profile, collect live quality/load/context evidence, and release it."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from research_workspace.service_tiers import LocalOpenAIChatBackend, ModelLane, ModelRoute
from research_workspace.serving_benchmark import (
    BenchmarkRequest,
    OpenAIStreamingProbe,
    context_probe_prompt,
    load_quality_manifest,
    run_concurrent,
    score_quality_output,
    summarize,
    tier_mix,
    write_measurements,
)
from research_workspace.serving_profile_runtime import (
    ServingProfileRuntime,
    ServingRuntimeError,
    observe_gpu,
)
from research_workspace.serving_profiles import (
    InstalledServingCapabilities,
    endpoint_for,
    load_profiles,
    resolve_profile,
)


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=root)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--vllm",
        type=Path,
        default=root / ".venv-vllm-cu129/bin/vllm",
    )
    parser.add_argument(
        "--ffmpeg-lib",
        type=Path,
        default=root / ".runtime/ffmpeg7/lib",
    )
    parser.add_argument("--smoke-only", action="store_true")
    return parser


def _capabilities(path: Path) -> InstalledServingCapabilities:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    installed = raw.get("installed") if isinstance(raw, dict) else None
    if not isinstance(installed, dict):
        raise RuntimeError("resolved profile manifest is malformed")
    flags = installed.get("flags")
    if not isinstance(flags, list) or not all(isinstance(flag, str) for flag in flags):
        raise RuntimeError("resolved profile flags are malformed")
    return InstalledServingCapabilities(
        version=str(installed["version"]),
        flags=frozenset(flags),
        help_sha256=str(installed["help_sha256"]),
    )


def _metrics(endpoint: str) -> str:
    try:
        with urllib.request.urlopen(endpoint.rstrip("/") + "/metrics", timeout=10) as response:
            return response.read().decode("utf-8", errors="replace")
    except (OSError, urllib.error.URLError):
        return ""


def _process_rss_mib(pid: int) -> float | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _gpu_sampler(stop: threading.Event, output: Path, owned_pid: int) -> None:
    fields = [*asdict(observe_gpu()), "owned_cpu_rss_mib"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        while not stop.is_set():
            try:
                writer.writerow(
                    {
                        **asdict(observe_gpu()),
                        "owned_cpu_rss_mib": _process_rss_mib(owned_pid),
                    }
                )
                handle.flush()
            except ServingRuntimeError:
                pass
            stop.wait(0.5)


def _quality(
    repository_root: Path,
    profile_id: str,
    endpoint: str,
    model_id: str,
    output_root: Path,
) -> list[dict[str, object]]:
    backend = LocalOpenAIChatBackend(timeout_seconds=600)
    route = ModelRoute(ModelLane.QUALITY, model_id, endpoint, 0)
    results: list[dict[str, object]] = []
    for case in load_quality_manifest(
        repository_root / "configs/serving_quality_manifest.json"
    ):
        response = backend.complete(
            messages=[{"role": "user", "content": case.prompt}],
            route=route,
            tools=(),
            request_id=f"quality-{profile_id}-{case.case_id}",
        )
        content = response.get("content")
        output = content if isinstance(content, str) else ""
        result = score_quality_output(profile_id, case, output)
        record = {
            **asdict(result),
            "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
            "output": output,
        }
        results.append(record)
    (output_root / "quality_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    repository_root = arguments.repository_root.resolve()
    output_root = arguments.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root.parents[1] / "resolved_profiles.json"
    capabilities = _capabilities(manifest_path)
    profile = next(
        (
            item
            for item in load_profiles(repository_root / "configs/serving_profiles")
            if item.profile_id == arguments.profile_id
        ),
        None,
    )
    if profile is None:
        raise SystemExit(f"unknown profile {arguments.profile_id}")
    resolved = resolve_profile(
        profile,
        capabilities,
        executable=arguments.vllm.resolve(),
    )
    resolved_record = {
        **resolved.to_json(),
        "environment_sha256": hashlib.sha256(
            json.dumps(
                {
                    "vllm": str(arguments.vllm.resolve()),
                    "ffmpeg_lib": str(arguments.ffmpeg_lib.resolve()),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
    }
    (output_root / "server_profile_resolved.json").write_text(
        json.dumps(resolved_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runtime = ServingProfileRuntime(
        output_root / "runtime",
        ffmpeg_library_path=arguments.ffmpeg_lib.resolve(),
    )
    before_pids = set(observe_gpu().compute_pids)
    release: dict[str, object] = {"status": "NOT_STARTED"}
    stop_sampler = threading.Event()
    sampler: threading.Thread | None = None
    try:
        startup_started = time.monotonic()
        owned = runtime.start(resolved)
        (output_root / "owned_process.json").write_text(
            json.dumps(asdict(owned), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        ready = {
            **runtime.wait_ready(resolved),
            "startup_seconds": time.monotonic() - startup_started,
            "cpu_rss_mib_at_ready": _process_rss_mib(owned.pid),
            "gpu_at_ready": asdict(observe_gpu()),
        }
        (output_root / "readiness.json").write_text(
            json.dumps(ready, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        endpoint = endpoint_for(profile)
        (output_root / "metrics_before.prom").write_text(
            _metrics(endpoint), encoding="utf-8"
        )
        stop_sampler.clear()
        sampler = threading.Thread(
            target=_gpu_sampler,
            args=(stop_sampler, output_root / "gpu_samples.csv", owned.pid),
            daemon=True,
        )
        sampler.start()
        quality = _quality(
            repository_root,
            profile.profile_id,
            endpoint,
            profile.served_model_name,
            output_root,
        )
        contexts = [2_048] if arguments.smoke_only else [2_048, 8_192, 16_384, 32_768, 65_536]
        contexts = [value for value in contexts if value <= profile.max_model_len]
        context_requests: list[BenchmarkRequest] = []
        for value in contexts:
            prompt, markers = context_probe_prompt(max(2_048, value - 128))
            context_requests.append(
                BenchmarkRequest(
                    request_id=f"context-{value}",
                    capability_tier="plus",
                    lane=ModelLane.QUALITY,
                    domain="context_retrieval",
                    prompt=prompt,
                    context_tokens=value,
                    max_output_tokens=96,
                    expected_markers=markers,
                )
            )
        probe = OpenAIStreamingProbe(
            endpoint,
            profile.served_model_name,
            timeout_seconds=profile.request_timeout,
        )
        measurements = list(
            run_concurrent(
                context_requests,
                profile_id=profile.profile_id,
                concurrency=1,
                probe=probe,
            )
        )
        concurrencies = [1] if arguments.smoke_only else [1, 2, 4, 8, 12]
        for concurrency in concurrencies:
            requests = [
                BenchmarkRequest(
                    request_id=f"load-{concurrency}-{index:02d}",
                    capability_tier="basic" if index % 2 == 0 else "plus",
                    lane=lane,
                    domain="systemverilog" if lane is ModelLane.ECONOMY else "general",
                    prompt=(
                        "Return exactly the token PASS."
                        if lane is not ModelLane.ECONOMY
                        else "State the ready/valid transfer condition in one sentence."
                    ),
                    context_tokens=0,
                    max_output_tokens=32,
                )
                for index, lane in enumerate(tier_mix(10))
            ]
            measurements.extend(
                run_concurrent(
                    requests,
                    profile_id=profile.profile_id,
                    concurrency=concurrency,
                    probe=probe,
                )
            )
        summary = summarize(profile.profile_id, measurements)
        write_measurements(output_root, measurements, summary)
        (output_root / "metrics_after.prom").write_text(
            _metrics(endpoint), encoding="utf-8"
        )
        topology = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        (output_root / "pcie_topology.txt").write_text(
            topology.stdout + topology.stderr,
            encoding="utf-8",
        )
        traffic = subprocess.run(
            ["nvidia-smi", "dmon", "-s", "t", "-c", "1"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        (output_root / "pcie_traffic_sample.txt").write_text(
            traffic.stdout + traffic.stderr,
            encoding="utf-8",
        )
        result = {
            "status": "COMPLETE",
            "profile_id": profile.profile_id,
            "resolution_sha256": resolved.resolution_sha256,
            "quality_score": sum(float(item["score"]) for item in quality) / len(quality),
            "hard_gate_pass": all(bool(item["passed_hard_gates"]) for item in quality),
            "benchmark": asdict(summary),
        }
        (output_root / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except ServingRuntimeError as exc:
        (output_root / "blocked_or_failed.json").write_text(
            json.dumps(
                {
                    "status": "BLOCKED_OR_FAILED",
                    "failure_category": exc.category,
                    "evidence": exc.evidence,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return 20
    finally:
        stop_sampler.set()
        if sampler is not None:
            sampler.join(timeout=5)
        if runtime.ownership_path.exists():
            try:
                release = runtime.release_owned()
            except ServingRuntimeError as exc:
                release = {
                    "status": "RELEASE_FAILED",
                    "failure_category": exc.category,
                    "evidence": exc.evidence,
                }
        after_pids = set(observe_gpu().compute_pids)
        (output_root / "safe_shutdown.json").write_text(
            json.dumps(
                {
                    **release,
                    "preexisting_compute_pids": sorted(before_pids),
                    "remaining_preexisting_compute_pids": sorted(before_pids & after_pids),
                    "unrelated_processes_signalled": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
