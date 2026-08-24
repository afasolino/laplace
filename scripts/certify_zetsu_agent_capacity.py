#!/usr/bin/env python3
"""Bounded direct-inference stress for Zetsu agent-capacity certification.

This never calls ``agent_task`` and never lets a model mutate a repository. It
starts an owned runtime topology, sends deterministic direct Qwen inference
probes at bounded concurrency, records real latency/throughput/GPU evidence, and
uses the owned runtime stop path in ``finally``.
"""

from __future__ import annotations

import argparse
import json
import subprocess  # nosec B404 - fixed local diagnostic argv
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research_workspace.zetsu_runtime import (
    _atomic_json,
    _boot_id,
    _process_identity,
    _spawn,
    _wait_ready,
    build_service_specs,
    load_local_plus_token,
    start_local_runtime,
    stop_local_runtime,
)


def _gpu_snapshot() -> dict[str, object]:
    completed = subprocess.run(  # nosec B603
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    return {
        "returncode": completed.returncode,
        "csv": completed.stdout.strip(),
        "stderr": completed.stderr.strip()[:2_000],
    }


def _post_qwen(model_id: str, request_index: int) -> dict[str, object]:
    marker = f"CAPACITY-{request_index:03d}"
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Reply with a short JSON object containing marker {marker} and "
                    "the integer 17. Do not call tools."
                ),
            }
        ],
        "temperature": 0,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        "http://127.0.0.1:8207/v1/chat/completions",
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=180) as response:  # nosec B310
        body = response.read(1_000_001)
        status = response.status
    elapsed = time.monotonic() - started
    if len(body) > 1_000_000:
        raise RuntimeError("capacity_probe_response_too_large")
    raw: Any = json.loads(body)
    choices = raw.get("choices") if isinstance(raw, dict) else None
    choice = choices[0] if isinstance(choices, list) and choices else None
    message = choice.get("message") if isinstance(choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    usage = raw.get("usage") if isinstance(raw, dict) else None
    completion_tokens = (
        int(usage.get("completion_tokens", 0)) if isinstance(usage, dict) else 0
    )
    return {
        "http_status": status,
        "elapsed_seconds": round(elapsed, 6),
        "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
        "content_present": isinstance(content, str) and bool(content.strip()),
        "marker_present": isinstance(content, str) and marker in content,
        "completion_tokens": completion_tokens,
    }


def _model_id(repository: Path) -> str:
    raw: Any = json.loads(
        (repository / "configs/selected_serving_profiles.json").read_text(encoding="utf-8")
    )
    return str(raw["routes"]["quality"]["model_id"])


def _start_certification_runtime(
    *,
    repository: Path,
    artifact_repository: Path,
    state_root: Path,
    python: Path,
    vllm: Path,
    ffmpeg_lib: Path,
    codev_enabled: bool,
) -> dict[str, object]:
    """Start hotfix code with read-only artifacts from a separate checkout.

    This mode exists only for isolated certification worktrees. Runtime ownership
    remains rooted in ``repository``; only Qwen's verified model/profile lookup
    uses ``artifact_repository``. No file there is written.
    """

    specs = build_service_specs(
        repository,
        state_root,
        python=python,
        vllm=vllm,
        ffmpeg_lib=ffmpeg_lib,
        codev_enabled=codev_enabled,
    )
    adjusted = []
    for spec in specs:
        if spec.name != "qwen":
            adjusted.append(spec)
            continue
        argv = list(spec.argv)
        repository_index = argv.index("--repository-root") + 1
        argv[repository_index] = str(artifact_repository)
        adjusted.append(replace(spec, argv=tuple(argv)))
    runtime_id = uuid.uuid4().hex + uuid.uuid4().hex
    ownership = {
        "LAPLACE_ZETSU_RUNTIME_ID": runtime_id,
        "LAPLACE_ZETSU_STATE_ROOT": str(state_root),
        "LAPLACE_ZETSU_REPOSITORY": str(repository),
        "LAPLACE_SERVER_OWNER_TOKEN": runtime_id,
    }
    specs = tuple(
        replace(spec, environment={**spec.environment, **ownership}) for spec in adjusted
    )
    record: dict[str, object] = {
        "schema_version": 2,
        "runtime_id": runtime_id,
        "boot_id": _boot_id(),
        "repository": str(repository),
        "state_root": str(state_root),
        "topology": "full" if codev_enabled else "nocodev",
        "codev": "required" if codev_enabled else "intentionally_disabled",
        "services": {},
    }
    record_path = state_root / "run/zetsu_services.json"
    _atomic_json(record_path, record)
    services = record["services"]
    if not isinstance(services, dict):
        raise RuntimeError("certification_runtime_record_invalid")
    for spec in specs:
        process = _spawn(spec)
        services[spec.name] = {
            "process": asdict(_process_identity(process.pid)),
            "log": str(spec.log_path),
            "probe_url": spec.probe_url,
            "expected_model": spec.expected_model,
        }
        _atomic_json(record_path, record)
        _wait_ready(spec, process, deadline=time.monotonic() + 1_800)
    # Loading the token proves the isolated operator state has the exact local
    # credential shape without serializing the secret into evidence.
    load_local_plus_token(state_root)
    record["ready_at_utc"] = time.time()
    _atomic_json(record_path, record)
    return {
        "status": "READY",
        "runtime_state": "STARTED_CERTIFICATION",
        "topology": record["topology"],
        "service_order": [spec.name for spec in specs],
        "artifact_repository_read_only": str(artifact_repository),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--artifact-repository", type=Path)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--vllm", type=Path, required=True)
    parser.add_argument("--ffmpeg-lib", type=Path, required=True)
    parser.add_argument("--topology", choices=("full", "nocodev"), required=True)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.max_concurrency <= 8:
        raise SystemExit("max-concurrency must be in [1, 8]")

    repository = args.repository.resolve()
    state_root = args.state_root.resolve()
    record: dict[str, object] = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "topology": args.topology,
        "repository": str(repository),
        "bounded_concurrency_ceiling": args.max_concurrency,
        "gpu_before": _gpu_snapshot(),
        "batches": [],
    }
    exit_code = 1
    try:
        started = (
            _start_certification_runtime(
                repository=repository,
                artifact_repository=args.artifact_repository.resolve(),
                state_root=state_root,
                python=args.python.absolute(),
                vllm=args.vllm.resolve(),
                ffmpeg_lib=args.ffmpeg_lib.resolve(),
                codev_enabled=args.topology == "full",
            )
            if args.artifact_repository is not None
            else start_local_runtime(
                repository,
                state_root,
                timeout=1_800,
                python=args.python,
                vllm=args.vllm,
                ffmpeg_lib=args.ffmpeg_lib,
                codev_enabled=args.topology == "full",
            )
        )
        record["start"] = started
        model_id = _model_id(repository)
        batches: list[dict[str, object]] = []
        for concurrency in range(1, args.max_concurrency + 1):
            batch_started = time.monotonic()
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                results = list(
                    executor.map(
                        lambda index: _post_qwen(model_id, concurrency * 100 + index),
                        range(concurrency),
                    )
                )
            elapsed = time.monotonic() - batch_started
            tokens = sum(
                value
                for item in results
                if isinstance((value := item["completion_tokens"]), int)
            )
            passed = all(
                item["http_status"] == 200
                and item["content_present"] is True
                and item["finish_reason"] != "length"
                for item in results
            )
            batches.append(
                {
                    "concurrency": concurrency,
                    "passed": passed,
                    "batch_elapsed_seconds": round(elapsed, 6),
                    "completion_tokens": tokens,
                    "completion_tokens_per_second": (
                        round(tokens / elapsed, 6) if elapsed > 0 else None
                    ),
                    "responses": results,
                    "gpu_after_batch": _gpu_snapshot(),
                }
            )
            if not passed:
                break
        record["batches"] = batches
        stable_values = [
            value
            for item in batches
            if item["passed"] is True
            and isinstance((value := item["concurrency"]), int)
        ]
        record["maximum_stable_within_tested_bound"] = max(stable_values, default=0)
        exit_code = 0 if len(batches) == args.max_concurrency and all(
            item["passed"] is True for item in batches
        ) else 1
    except Exception as exc:
        record["error"] = {"type": type(exc).__name__, "message": str(exc)[:2_000]}
    finally:
        try:
            record["stop"] = stop_local_runtime(state_root)
        except Exception as exc:
            record["stop_error"] = {
                "type": type(exc).__name__,
                "message": str(exc)[:2_000],
            }
            exit_code = 1
        record["gpu_after_stop"] = _gpu_snapshot()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
