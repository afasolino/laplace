#!/usr/bin/env python3
"""Measure the smallest stable CodeV reservation for its production 8K route."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess  # nosec B404 - fixed local vLLM probes
import threading
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

import httpx

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
MODEL = ROOT / ".models/CodeV-R1-RL-Qwen-7B-W4A16-AWQ"
MODEL_ID = "laplace-codev-r1-rl-qwen-7b-w4a16"
PROMPT = (
    "Return only one synthesizable SystemVerilog module named ready_valid_gate. "
    "Inputs are logic ready and valid; output logic transfer must be assigned "
    "ready && valid. Do not include prose."
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-utils", type=float, nargs="+", default=(0.145, 0.13, 0.125, 0.12))
    parser.add_argument("--kv-cache-dtypes", choices=("auto", "fp8"), nargs="+", default=("auto",))
    parser.add_argument("--vllm", type=Path, default=VLLM)
    parser.add_argument("--ffmpeg-lib", type=Path, default=FFMPEG)
    return parser


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _sample_gpu(stop: threading.Event, samples: list[dict[str, object]]) -> None:
    while not stop.is_set():
        try:
            samples.append(asdict(observe_gpu()))
        except ServingRuntimeError:
            pass
        stop.wait(0.25)


def _chat(client: httpx.Client, endpoint: str) -> dict[str, object]:
    started = time.monotonic()
    response = client.post(
        endpoint + "/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0,
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    elapsed = time.monotonic() - started
    response.raise_for_status()
    value: object = response.json()
    choices = value.get("choices") if isinstance(value, dict) else None
    message = choices[0].get("message") if isinstance(choices, list) and choices else None
    content = message.get("content") if isinstance(message, dict) else None
    text = content if isinstance(content, str) else ""
    usage = value.get("usage") if isinstance(value, dict) else None
    return {
        "status": "PASS"
        if all(marker in text for marker in ("module ready_valid_gate", "ready && valid"))
        else "FAIL",
        "elapsed_seconds": elapsed,
        "usage": usage if isinstance(usage, dict) else {},
        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "content_excerpt": text[:500],
    }


def _workpoint(
    gpu_util: float,
    *,
    kv_cache_dtype: str,
    output: Path,
    capabilities: InstalledServingCapabilities,
    vllm: Path,
    ffmpeg: Path,
) -> dict[str, object]:
    label = str(gpu_util).replace(".", "p")
    profile = ServingProfile(
        profile_id=f"P8_codev_reservation_{label}_{kv_cache_dtype}",
        model_route="economy",
        model_path=str(MODEL),
        served_model_name=MODEL_ID,
        port=8103,
        max_model_len=8192,
        max_num_seqs=1,
        max_num_batched_tokens=4096,
        kv_cache_dtype=kv_cache_dtype,  # type: ignore[arg-type]
        kv_cache_memory_bytes=None,
        enable_prefix_caching=True,
        prefix_hash_algorithm="sha256",
        enable_chunked_prefill=True,
        scheduling_policy="fcfs",
        cpu_offload_gb=0.0,
        cpu_offload_params=(),
        offload_backend="auto",
        offload_group_size=0,
        offload_num_in_group=1,
        offload_prefetch_step=1,
        kv_offloading_size=None,
        kv_offloading_backend="native",
        gpu_memory_utilization=gpu_util,
        startup_timeout=600,
        request_timeout=300,
        extra_args=("--enforce-eager",),
    )
    resolved = resolve_profile(profile, capabilities, executable=vllm, require_model=True)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "resolved_profile.json", resolved.to_json())
    runtime = ServingProfileRuntime(output / "runtime", ffmpeg_library_path=ffmpeg)
    initial = observe_gpu()
    samples: list[dict[str, object]] = [asdict(initial)]
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_gpu, args=(stop, samples), daemon=True)
    result: dict[str, object] = {
        "gpu_memory_utilization": gpu_util,
        "kv_cache_dtype": kv_cache_dtype,
        "status": "FAILED",
    }
    release: dict[str, object] = {"status": "NOT_STARTED"}
    sampler.start()
    try:
        owned = runtime.start(resolved)
        readiness = runtime.wait_ready(resolved)
        ready_gpu = observe_gpu()
        with httpx.Client(timeout=httpx.Timeout(300)) as client:
            warmup = _chat(client, endpoint_for(profile))
            runs = [_chat(client, endpoint_for(profile)) for _ in range(3)]
        result.update(
            {
                "status": "PASS"
                if warmup["status"] == "PASS" and all(item["status"] == "PASS" for item in runs)
                else "FAILED",
                "owned_pid": owned.pid,
                "readiness": readiness,
                "ready_gpu": asdict(ready_gpu),
                "warmup": warmup,
                "runs": runs,
                "median_elapsed_seconds": statistics.median(
                    float(item["elapsed_seconds"]) for item in runs
                ),
            }
        )
    except Exception as exc:  # noqa: BLE001 - preserve unsupported/OOM evidence
        result["failure"] = {
            "type": type(exc).__name__,
            "error": str(exc)[:2_000],
            "evidence": getattr(exc, "evidence", None),
        }
    finally:
        stop.set()
        sampler.join(timeout=5)
        if runtime.ownership_path.exists():
            try:
                release = runtime.release_owned(timeout_seconds=90)
            except ServingRuntimeError as exc:
                release = {"status": "FAIL", "category": exc.category, "evidence": exc.evidence}
    residual = observe_gpu()
    peak = max(int(item["used_mib"]) for item in samples)
    result.update(
        {
            "gpu": {
                "initial_used_mib": initial.used_mib,
                "peak_used_mib": peak,
                "minimum_free_headroom_mib": initial.total_mib - peak,
                "configured_vllm_reservation_mib": math.floor(initial.total_mib * gpu_util),
                "residual_used_mib": residual.used_mib,
                "sample_count": len(samples),
            },
            "release": release,
            "endpoint_down_after_release": not residual.compute_pids,
        }
    )
    _write_json(output / "gpu_samples.json", samples)
    _write_json(output / "result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    utils = tuple(arguments.gpu_utils)
    if not utils or len(utils) != len(set(utils)) or any(not 0 < value < 1 for value in utils):
        raise RuntimeError("invalid_gpu_utilization_sweep")
    output = arguments.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    capabilities = _capabilities(
        arguments.vllm.resolve(strict=True), arguments.ffmpeg_lib.resolve(strict=True)
    )
    dtypes = tuple(arguments.kv_cache_dtypes)
    results = [
        _workpoint(
            value,
            kv_cache_dtype=dtype,
            output=output / f"{str(value).replace('.', 'p')}-{dtype}",
            capabilities=capabilities,
            vllm=arguments.vllm.resolve(),
            ffmpeg=arguments.ffmpeg_lib.resolve(),
        )
        for dtype in dtypes
        for value in utils
    ]
    report = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS" if any(item["status"] == "PASS" for item in results) else "FAILED",
        "context_tokens": 8192,
        "max_num_seqs": 1,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "runs_per_workpoint": 3,
        "results": results,
    }
    _write_json(output / "sweep.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
