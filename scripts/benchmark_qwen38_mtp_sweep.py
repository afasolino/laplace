#!/usr/bin/env python3
"""Benchmark native Qwen3.8 MTP workpoints with one frozen local workload."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess  # nosec B404 - fixed local vLLM capability probes
import threading
import time
from dataclasses import asdict, replace
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
BASE_PROFILE = ROOT / "configs/serving_profile_candidates/P7_qwen38_w4a16_mtp.json"
DEFAULT_KS = tuple(range(3, 11))
WORKLOAD = (
    "This is frozen local context for a deterministic speculative-decoding throughput "
    "measurement. Preserve the context but do not summarize it. "
) * 640 + (
    "\nWrite a long numbered list beginning at 1. Put one integer on each line and "
    "continue until the server output limit stops you. Do not add commentary."
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=BASE_PROFILE)
    parser.add_argument("--vllm", type=Path, default=VLLM)
    parser.add_argument("--ffmpeg-lib", type=Path, default=FFMPEG)
    parser.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    parser.add_argument("--timed-runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    return parser


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _profile(path: Path) -> ServingProfile:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("candidate_profile_malformed")
    return ServingProfile.from_mapping(raw)


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


def _metric_sum(metrics: str, name: str) -> float:
    total = 0.0
    for line in metrics.splitlines():
        if line.startswith("#") or not line.startswith(name):
            continue
        sample_name = line.split("{", 1)[0].split(None, 1)[0]
        if sample_name not in {name, name + "_total"}:
            continue
        try:
            total += float(line.rsplit(None, 1)[1])
        except (IndexError, ValueError):
            continue
    return total


def _mtp_counters(metrics: str) -> dict[str, float]:
    return {
        "drafts": _metric_sum(metrics, "vllm:spec_decode_num_drafts"),
        "draft_tokens": _metric_sum(metrics, "vllm:spec_decode_num_draft_tokens"),
        "accepted_tokens": _metric_sum(metrics, "vllm:spec_decode_num_accepted_tokens"),
    }


def _counter_delta(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {key: max(0.0, after[key] - before[key]) for key in before}


def _sample_gpu(stop: threading.Event, samples: list[dict[str, object]]) -> None:
    while not stop.is_set():
        try:
            samples.append(asdict(observe_gpu()))
        except ServingRuntimeError:
            pass
        stop.wait(0.25)


def _request(
    client: httpx.Client,
    endpoint: str,
    model: str,
    *,
    max_tokens: int,
) -> dict[str, object]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": WORKLOAD}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.monotonic()
    first_output_at: float | None = None
    output: list[str] = []
    usage: dict[str, object] = {}
    with client.stream("POST", endpoint + "/v1/chat/completions", json=body) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            value: object = json.loads(line[6:])
            if not isinstance(value, dict):
                continue
            if isinstance(value.get("usage"), dict):
                usage = dict(value["usage"])
            choices = value.get("choices")
            delta = choices[0].get("delta") if isinstance(choices, list) and choices else None
            fragment = delta.get("content") if isinstance(delta, dict) else None
            if isinstance(fragment, str) and fragment:
                if first_output_at is None:
                    first_output_at = time.monotonic()
                output.append(fragment)
    completed = time.monotonic()
    completion_tokens = usage.get("completion_tokens")
    decode_seconds = completed - first_output_at if first_output_at is not None else None
    output_rate = (
        (completion_tokens - 1) / decode_seconds
        if isinstance(completion_tokens, int)
        and completion_tokens > 1
        and isinstance(decode_seconds, float)
        and decode_seconds > 0
        else None
    )
    rendered = "".join(output)
    return {
        "status": "PASS" if output_rate is not None else "FAIL",
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "ttft_seconds": first_output_at - started if first_output_at is not None else None,
        "decode_seconds": decode_seconds,
        "elapsed_seconds": completed - started,
        "output_tok_s": output_rate,
        "output_sha256": _sha256_text(rendered),
    }


def _kv_cache_gib(log: str) -> float | None:
    matches = re.findall(r"Available KV cache memory:\s*([0-9.]+)\s*GiB", log)
    return float(matches[-1]) if matches else None


def _workpoint(
    base: ServingProfile,
    *,
    k: int,
    output: Path,
    capabilities: InstalledServingCapabilities,
    vllm: Path,
    ffmpeg: Path,
    timed_runs: int,
    max_tokens: int,
) -> dict[str, object]:
    base_extra = tuple(
        arg for arg in base.extra_args if not arg.startswith("--speculative-config=")
    )
    extra = (
        base_extra
        if k == 0
        else base_extra + (f'--speculative-config={{"method":"mtp","num_speculative_tokens":{k}}}',)
    )
    profile = replace(
        base,
        profile_id=("P6_qwen38_w4a16_sweep" if k == 0 else f"P7_qwen38_w4a16_mtp_k{k}"),
        served_model_name=(
            "laplace-quality-qwen38-sweep" if k == 0 else f"laplace-quality-qwen38-mtp-k{k}"
        ),
        extra_args=extra,
    )
    resolved = resolve_profile(profile, capabilities, executable=vllm, require_model=True)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "resolved_profile.json", resolved.to_json())
    runtime = ServingProfileRuntime(output / "runtime", ffmpeg_library_path=ffmpeg)
    initial = observe_gpu()
    samples: list[dict[str, object]] = [asdict(initial)]
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_gpu, args=(stop, samples), daemon=True)
    release: dict[str, object] = {"status": "NOT_STARTED"}
    result: dict[str, object] = {
        "k": k,
        "status": "FAILED",
        "profile": profile.to_json(),
    }
    sampler.start()
    try:
        owned = runtime.start(resolved)
        readiness = runtime.wait_ready(resolved)
        endpoint = endpoint_for(profile)
        with httpx.Client(timeout=httpx.Timeout(profile.request_timeout)) as client:
            tokenized = client.post(
                endpoint + "/tokenize",
                json={
                    "model": profile.served_model_name,
                    "messages": [{"role": "user", "content": WORKLOAD}],
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            tokenized.raise_for_status()
            token_value: object = tokenized.json()
            prompt_tokens = token_value.get("count") if isinstance(token_value, dict) else None
            metrics_before = _mtp_counters(client.get(endpoint + "/metrics").text)
            warmup = _request(
                client,
                endpoint,
                profile.served_model_name,
                max_tokens=max_tokens,
            )
            runs = [
                _request(
                    client,
                    endpoint,
                    profile.served_model_name,
                    max_tokens=max_tokens,
                )
                for _ in range(timed_runs)
            ]
            metrics_after = _mtp_counters(client.get(endpoint + "/metrics").text)
        counters = _counter_delta(metrics_after, metrics_before)
        rates = [item["output_tok_s"] for item in runs]
        ttfts = [item["ttft_seconds"] for item in runs]
        valid_rates = [value for value in rates if isinstance(value, float)]
        valid_ttfts = [value for value in ttfts if isinstance(value, float)]
        result.update(
            {
                "status": (
                    "PASS"
                    if warmup["status"] == "PASS"
                    and len(valid_rates) == timed_runs
                    and (k == 0 or counters["draft_tokens"] > 0 and counters["drafts"] > 0)
                    else "FAILED"
                ),
                "owned_pid": owned.pid,
                "readiness": readiness,
                "prompt_tokens": prompt_tokens,
                "warmup": warmup,
                "runs": runs,
                "median_output_tok_s": statistics.median(valid_rates) if valid_rates else None,
                "median_ttft_seconds": statistics.median(valid_ttfts) if valid_ttfts else None,
                "mtp": {
                    **counters,
                    "acceptance_rate": counters["accepted_tokens"] / counters["draft_tokens"]
                    if counters["draft_tokens"]
                    else None,
                    "committed_tokens_per_step": 1
                    + counters["accepted_tokens"] / counters["drafts"]
                    if counters["drafts"]
                    else None,
                },
            }
        )
    except Exception as exc:  # noqa: BLE001 - preserve terminal sweep evidence
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
                release = {
                    "status": "FAIL",
                    "category": exc.category,
                    "evidence": exc.evidence,
                }
    residual = observe_gpu()
    peak_used = max(int(item["used_mib"]) for item in samples)
    server_log = output / "runtime" / f"{profile.profile_id}.server.log"
    log_text = (
        server_log.read_text(encoding="utf-8", errors="replace") if server_log.is_file() else ""
    )
    result.update(
        {
            "gpu": {
                "initial_used_mib": initial.used_mib,
                "peak_used_mib": peak_used,
                "minimum_free_headroom_mib": initial.total_mib - peak_used,
                "configured_vllm_reservation_mib": math.floor(
                    initial.total_mib * profile.gpu_memory_utilization
                ),
                "configured_gpu_memory_utilization": profile.gpu_memory_utilization,
                "available_kv_cache_gib_from_runtime": _kv_cache_gib(log_text),
                "residual_used_mib": residual.used_mib,
                "sample_count": len(samples),
                "sampling_interval_seconds": 0.25,
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
    if arguments.timed_runs < 3:
        raise RuntimeError("timed_runs_must_be_at_least_three")
    ks = tuple(arguments.ks)
    if not ks or len(ks) != len(set(ks)) or any(k < 0 or k > 32 for k in ks):
        raise RuntimeError("invalid_speculative_token_sweep")
    output = arguments.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    base = _profile(arguments.profile.resolve(strict=True))
    if arguments.gpu_memory_utilization is not None:
        base = replace(base, gpu_memory_utilization=arguments.gpu_memory_utilization)
    capabilities = _capabilities(
        arguments.vllm.resolve(strict=True), arguments.ffmpeg_lib.resolve(strict=True)
    )
    results = [
        _workpoint(
            base,
            k=k,
            output=output / f"k{k}",
            capabilities=capabilities,
            vllm=arguments.vllm.resolve(),
            ffmpeg=arguments.ffmpeg_lib.resolve(),
            timed_runs=arguments.timed_runs,
            max_tokens=arguments.max_tokens,
        )
        for k in ks
    ]
    report = {
        "schema_version": 1,
        "status": "PASS" if all(item["status"] == "PASS" for item in results) else "PARTIAL",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "ks": list(ks),
        "timed_runs_per_workpoint": arguments.timed_runs,
        "sampling": {"temperature": 0, "concurrency": 1, "max_tokens": arguments.max_tokens},
        "workload_sha256": _sha256_text(WORKLOAD),
        "workload_utf8_bytes": len(WORKLOAD.encode("utf-8")),
        "base_profile_path": str(arguments.profile.resolve()),
        "base_profile_sha256": hashlib.sha256(arguments.profile.resolve().read_bytes()).hexdigest(),
        "results": results,
    }
    _write_json(output / "sweep.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
