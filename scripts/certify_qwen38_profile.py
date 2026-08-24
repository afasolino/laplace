#!/usr/bin/env python3
"""Certify one fail-closed Qwen3.8 candidate on the local production GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess  # nosec B404 - fixed local executable and arguments
import threading
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import httpx

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
MANDATORY_GATES = (
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
    parser.add_argument(
        "profile_id",
        choices=("P6_qwen38_w4a16", "P7_qwen38_w4a16_mtp"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--vllm", type=Path, default=VLLM)
    parser.add_argument("--ffmpeg-lib", type=Path, default=FFMPEG)
    return parser


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _profile(path: Path) -> ServingProfile:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("candidate_profile_malformed")
    return ServingProfile.from_mapping(raw)


def _capabilities(vllm: Path, ffmpeg: Path) -> InstalledServingCapabilities:
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = str(ffmpeg)
    environment["PATH"] = (
        str(vllm.parent) + os.pathsep + environment.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    )
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


def _post(client: httpx.Client, endpoint: str, path: str, body: object) -> Any:
    response = client.post(endpoint + path, json=body)
    response.raise_for_status()
    return response.json()


def _message(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError("chat_response_not_object")
    choices = value.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError("chat_response_choices_missing")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise RuntimeError("chat_response_message_missing")
    return message


def _content(message: dict[str, object]) -> str:
    value = message.get("content")
    return value if isinstance(value, str) else ""


def _reasoning(message: dict[str, object]) -> str:
    for key in ("reasoning", "reasoning_content"):
        value = message.get(key)
        if isinstance(value, str):
            return value
    return ""


def _usage(value: object) -> dict[str, object]:
    if isinstance(value, dict) and isinstance(value.get("usage"), dict):
        return dict(value["usage"])
    return {}


def _compact_chat_evidence(value: object) -> dict[str, object]:
    message = _message(value)
    content = _content(message)
    reasoning = _reasoning(message)
    return {
        "model": value.get("model") if isinstance(value, dict) else None,
        "finish_reason": value["choices"][0].get("finish_reason"),
        "content_excerpt": content[:500],
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "reasoning_characters": len(reasoning),
        "reasoning_sha256": hashlib.sha256(reasoning.encode()).hexdigest(),
        "tool_calls": message.get("tool_calls"),
        "usage": _usage(value),
    }


def _base_body(model: str, messages: list[dict[str, str]], max_tokens: int) -> dict[str, object]:
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _normal(client: httpx.Client, endpoint: str, model: str) -> dict[str, object]:
    marker = "QWEN38_NORMAL_" + uuid.uuid4().hex[:12]
    value = _post(
        client,
        endpoint,
        "/v1/chat/completions",
        _base_body(model, [{"role": "user", "content": f"Reply exactly {marker}"}], 32),
    )
    evidence = _compact_chat_evidence(value)
    evidence["status"] = "PASS" if marker in _content(_message(value)) else "FAIL"
    evidence["expected_marker"] = marker
    return evidence


def _stream(client: httpx.Client, endpoint: str, model: str) -> dict[str, object]:
    marker = "QWEN38_STREAM_" + uuid.uuid4().hex[:12]
    body = _base_body(
        model,
        [{"role": "user", "content": f"Reply exactly {marker}"}],
        32,
    )
    body.update({"stream": True, "stream_options": {"include_usage": True}})
    content: list[str] = []
    reasoning: list[str] = []
    event_count = 0
    done = False
    usage: dict[str, object] = {}
    with client.stream("POST", endpoint + "/v1/chat/completions", json=body) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line == "data: [DONE]":
                done = True
                continue
            if not line.startswith("data: "):
                continue
            event_count += 1
            event: object = json.loads(line[6:])
            if isinstance(event, dict) and isinstance(event.get("usage"), dict):
                usage = dict(event["usage"])
            choices = event.get("choices") if isinstance(event, dict) else None
            delta = choices[0].get("delta") if isinstance(choices, list) and choices else None
            if isinstance(delta, dict):
                fragment = delta.get("content")
                if isinstance(fragment, str):
                    content.append(fragment)
                for key in ("reasoning", "reasoning_content"):
                    fragment = delta.get(key)
                    if isinstance(fragment, str):
                        reasoning.append(fragment)
                        break
    text = "".join(content)
    return {
        "status": "PASS" if marker in text and done and event_count > 1 else "FAIL",
        "event_count": event_count,
        "done_event": done,
        "expected_marker": marker,
        "content_excerpt": text[:500],
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "reasoning_characters": len("".join(reasoning)),
        "usage": usage,
    }


def _reasoning_gate(client: httpx.Client, endpoint: str, model: str) -> dict[str, object]:
    body = _base_body(
        model,
        [
            {
                "role": "user",
                "content": "Compute 19 times 23 carefully. Put only 437 in the final answer.",
            }
        ],
        768,
    )
    body.update(
        {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "chat_template_kwargs": {
                "enable_thinking": True,
                "preserve_thinking": True,
                "reasoning_effort": "medium",
            },
        }
    )
    value = _post(client, endpoint, "/v1/chat/completions", body)
    evidence = _compact_chat_evidence(value)
    message = _message(value)
    evidence["status"] = (
        "PASS" if "437" in _content(message) and bool(_reasoning(message).strip()) else "FAIL"
    )
    return evidence


def _tool_gate(client: httpx.Client, endpoint: str, model: str) -> dict[str, object]:
    body = _base_body(
        model,
        [
            {
                "role": "user",
                "content": "Use record_measurement to record exactly 37 degrees Celsius.",
            }
        ],
        256,
    )
    body["tools"] = [
        {
            "type": "function",
            "function": {
                "name": "record_measurement",
                "description": "Record one validated measurement.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "integer"},
                        "unit": {"type": "string", "enum": ["C"]},
                    },
                    "required": ["value", "unit"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    body["tool_choice"] = "required"
    value = _post(client, endpoint, "/v1/chat/completions", body)
    evidence = _compact_chat_evidence(value)
    calls = _message(value).get("tool_calls")
    valid = False
    parsed_arguments: object = None
    if isinstance(calls, list) and len(calls) == 1 and isinstance(calls[0], dict):
        function = calls[0].get("function")
        if isinstance(function, dict) and function.get("name") == "record_measurement":
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                parsed_arguments = json.loads(arguments)
                valid = parsed_arguments == {"value": 37, "unit": "C"}
    evidence["parsed_arguments"] = parsed_arguments
    evidence["status"] = "PASS" if valid else "FAIL"
    return evidence


def _multi_turn(client: httpx.Client, endpoint: str, model: str) -> dict[str, object]:
    marker = "QWEN38_MEMORY_" + uuid.uuid4().hex[:12]
    messages = [
        {
            "role": "user",
            "content": f"Remember this exact local continuity marker: {marker}. Reply READY.",
        }
    ]
    first = _post(client, endpoint, "/v1/chat/completions", _base_body(model, messages, 64))
    messages.extend(
        [
            {"role": "assistant", "content": _content(_message(first))},
            {"role": "user", "content": "Return the exact continuity marker."},
        ]
    )
    second = _post(client, endpoint, "/v1/chat/completions", _base_body(model, messages, 64))
    evidence = _compact_chat_evidence(second)
    evidence["first_turn"] = _compact_chat_evidence(first)
    evidence["expected_marker"] = marker
    evidence["status"] = "PASS" if marker in _content(_message(second)) else "FAIL"
    return evidence


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


def _metrics(client: httpx.Client, endpoint: str) -> str:
    response = client.get(endpoint + "/metrics")
    response.raise_for_status()
    return response.text


def _cancellation(client: httpx.Client, endpoint: str, model: str) -> dict[str, object]:
    request_id = "qwen38-cancel-" + uuid.uuid4().hex
    body = _base_body(
        model,
        [
            {
                "role": "user",
                "content": "Write an extremely long numbered sequence, one number per line, and do not stop early.",
            }
        ],
        4096,
    )
    body["stream"] = True
    event_count = 0
    with client.stream(
        "POST",
        endpoint + "/v1/chat/completions",
        json=body,
        headers={"X-Request-Id": request_id},
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                event_count += 1
                if event_count >= 2:
                    break
    deadline = time.monotonic() + 30
    running = -1.0
    waiting = -1.0
    while time.monotonic() < deadline:
        metrics = _metrics(client, endpoint)
        running = _metric_sum(metrics, "vllm:num_requests_running")
        waiting = _metric_sum(metrics, "vllm:num_requests_waiting")
        if running == 0 and waiting == 0:
            break
        time.sleep(0.25)
    return {
        "status": "PASS" if event_count >= 2 and running == 0 and waiting == 0 else "FAIL",
        "request_id": request_id,
        "events_before_disconnect": event_count,
        "running_after_disconnect": running,
        "waiting_after_disconnect": waiting,
    }


def _tokenize_chat(
    client: httpx.Client, endpoint: str, model: str, content: str
) -> tuple[int, int]:
    value = _post(
        client,
        endpoint,
        "/tokenize",
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    if not isinstance(value, dict) or not isinstance(value.get("count"), int):
        raise RuntimeError("tokenize_response_malformed")
    return value["count"], int(value.get("max_model_len", 0))


def _context_window(
    client: httpx.Client, endpoint: str, model: str, expected_limit: int
) -> dict[str, object]:
    first_marker = "QWEN38_FIRST_" + uuid.uuid4().hex[:12]
    last_marker = "QWEN38_LAST_" + uuid.uuid4().hex[:12]
    prefix = f"Remember the first code {first_marker}. "
    suffix = f" The last code is {last_marker}. Reply with both exact codes and no other text."
    low = 1
    high = expected_limit * 2
    selected_content = ""
    selected_count = 0
    model_limit = 0
    reserve = max(1_024, min(4_096, expected_limit // 64))
    target = max(2_048, expected_limit - reserve)
    while low <= high:
        midpoint = (low + high) // 2
        candidate = prefix + ("neutral " * midpoint) + suffix
        count, model_limit = _tokenize_chat(client, endpoint, model, candidate)
        if count <= target:
            selected_content = candidate
            selected_count = count
            low = midpoint + 1
        else:
            high = midpoint - 1
    body = _base_body(model, [{"role": "user", "content": selected_content}], 64)
    body.update({"stream": True, "stream_options": {"include_usage": True}})
    output_parts: list[str] = []
    usage: dict[str, object] = {}
    event_count = 0
    done = False
    started = time.monotonic()
    first_output_at: float | None = None
    with client.stream("POST", endpoint + "/v1/chat/completions", json=body) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            observed_at = time.monotonic()
            if line == "data: [DONE]":
                done = True
                continue
            if not line.startswith("data: "):
                continue
            event_count += 1
            event: object = json.loads(line[6:])
            if isinstance(event, dict) and isinstance(event.get("usage"), dict):
                usage = dict(event["usage"])
            choices = event.get("choices") if isinstance(event, dict) else None
            delta = choices[0].get("delta") if isinstance(choices, list) and choices else None
            fragment = delta.get("content") if isinstance(delta, dict) else None
            if isinstance(fragment, str) and fragment:
                if first_output_at is None:
                    first_output_at = observed_at
                output_parts.append(fragment)
    completed_at = time.monotonic()
    output = "".join(output_parts)
    reported_prompt = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    decode_seconds = completed_at - first_output_at if first_output_at is not None else None
    decode_throughput = (
        (completion_tokens - 1) / decode_seconds
        if isinstance(completion_tokens, int)
        and completion_tokens > 1
        and isinstance(decode_seconds, float)
        and decode_seconds > 0
        else None
    )
    passed = (
        target - 128 <= selected_count <= target
        and model_limit == expected_limit
        and done
        and first_marker in output
        and last_marker in output
        and isinstance(reported_prompt, int)
        and reported_prompt == selected_count
        and isinstance(decode_throughput, float)
        and decode_throughput > 0
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "tokenize_count": selected_count,
        "target_prompt_tokens": target,
        "reserved_prompt_tokens": reserve,
        "reported_prompt_tokens": reported_prompt,
        "reported_completion_tokens": completion_tokens,
        "reported_max_model_len": model_limit,
        "first_marker_recalled": first_marker in output,
        "last_marker_recalled": last_marker in output,
        "stream_event_count": event_count,
        "done_event": done,
        "content_excerpt": output[:500],
        "content_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "ttft_seconds": (first_output_at - started if first_output_at is not None else None),
        "decode_seconds": decode_seconds,
        "decode_throughput_tok_s": decode_throughput,
        "request_elapsed_seconds": completed_at - started,
        "usage": usage,
    }


def _identity(client: httpx.Client, endpoint: str, model: str) -> dict[str, object]:
    response = client.get(endpoint + "/v1/models")
    response.raise_for_status()
    value: object = response.json()
    data = value.get("data") if isinstance(value, dict) else None
    identities = (
        [
            item.get("id")
            for item in data
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]
        if isinstance(data, list)
        else []
    )
    return {
        "status": "PASS" if identities == [model] else "FAIL",
        "expected": model,
        "served": identities,
        "http_status": response.status_code,
    }


def _runtime_stability(
    client: httpx.Client, endpoint: str, model: str, process_pid: int
) -> dict[str, object]:
    health = client.get(endpoint + "/health")
    identity = _identity(client, endpoint, model)
    metrics = _metrics(client, endpoint)
    running = _metric_sum(metrics, "vllm:num_requests_running")
    waiting = _metric_sum(metrics, "vllm:num_requests_waiting")
    passed = (
        health.status_code == 200
        and identity["status"] == "PASS"
        and Path(f"/proc/{process_pid}").exists()
        and running == 0
        and waiting == 0
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "health_http_status": health.status_code,
        "identity": identity,
        "owned_process_alive": Path(f"/proc/{process_pid}").exists(),
        "running_requests": running,
        "waiting_requests": waiting,
    }


def _quantized_kernel(model_path: Path, log_path: Path) -> dict[str, object]:
    config: object = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    quantization = config.get("quantization_config") if isinstance(config, dict) else None
    log = log_path.read_text(encoding="utf-8", errors="replace")
    normalized = log.lower()
    format_ok = isinstance(quantization, dict) and quantization.get("quant_method") in {
        "compressed-tensors",
        "compressed_tensors",
    }
    kernel_markers = sorted(
        marker
        for marker in ("marlin", "compressedtensorsw4a16", "compressed_tensors")
        if marker in normalized
    )
    return {
        "status": "PASS" if format_ok and kernel_markers else "FAIL",
        "quantization_config": quantization,
        "runtime_log_markers": kernel_markers,
        "server_log_sha256": _sha256(log_path),
    }


def _mtp_tokens(profile: ServingProfile) -> int:
    for argument in profile.extra_args:
        prefix = "--speculative-config="
        if not argument.startswith(prefix):
            continue
        value: object = json.loads(argument.removeprefix(prefix))
        tokens = value.get("num_speculative_tokens") if isinstance(value, dict) else None
        method = value.get("method") if isinstance(value, dict) else None
        if method == "mtp" and isinstance(tokens, int) and tokens > 0:
            return tokens
    raise RuntimeError("mtp_profile_configuration_missing")


def _mtp(
    client: httpx.Client,
    endpoint: str,
    log_path: Path,
    expected_tokens: int,
) -> dict[str, object]:
    metrics = _metrics(client, endpoint)
    drafted = _metric_sum(metrics, "vllm:spec_decode_num_draft_tokens")
    accepted = _metric_sum(metrics, "vllm:spec_decode_num_accepted_tokens")
    drafts = _metric_sum(metrics, "vllm:spec_decode_num_drafts")
    log = log_path.read_text(encoding="utf-8", errors="replace").lower()
    configured = (
        f"num_speculative_tokens={expected_tokens}" in log
        or f"'num_speculative_tokens': {expected_tokens}" in log
        or f'"num_speculative_tokens": {expected_tokens}' in log
    ) and ("method='mtp'" in log or "'method': 'mtp'" in log or '"method": "mtp"' in log)
    return {
        "status": "PASS" if configured and drafted > 0 and drafts > 0 else "FAIL",
        "configured_mtp_tokens": expected_tokens if configured else None,
        "drafts": drafts,
        "draft_tokens": drafted,
        "accepted_tokens": accepted,
        "acceptance_rate": accepted / drafted if drafted else None,
        "committed_tokens_per_step": 1 + accepted / drafts if drafts else None,
    }


def _sample_gpu(stop: threading.Event, samples: list[dict[str, object]]) -> None:
    while not stop.is_set():
        try:
            samples.append(asdict(observe_gpu()))
        except ServingRuntimeError:
            pass
        stop.wait(0.25)


def _endpoint_down(endpoint: str) -> bool:
    try:
        response = httpx.get(endpoint + "/v1/models", timeout=2)
    except httpx.HTTPError:
        return True
    return response.status_code >= 500


def _run_gate(
    results: dict[str, dict[str, object]],
    name: str,
    operation: Callable[[], dict[str, object]],
) -> None:
    started = time.monotonic()
    try:
        value = operation()
    except Exception as exc:  # noqa: BLE001 - each gate must leave terminal evidence
        value = {
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc)[:2_000],
        }
    value["elapsed_seconds"] = round(time.monotonic() - started, 3)
    results[name] = value


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    root = arguments.repository_root.resolve()
    output = arguments.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    profile_path = root / "configs/serving_profile_candidates" / f"{arguments.profile_id}.json"
    profile = _profile(profile_path)
    artifact = verify_qwen38_artifact(root)
    if Path(profile.model_path).resolve() != Path(str(artifact["artifact_path"])).resolve():
        raise RuntimeError("candidate_profile_artifact_mismatch")
    capabilities = _capabilities(arguments.vllm.resolve(), arguments.ffmpeg_lib.resolve())
    resolved = resolve_profile(profile, capabilities, executable=arguments.vllm.resolve())
    _write_json(output / "resolved_profile.json", resolved.to_json())
    runtime = ServingProfileRuntime(
        output / "runtime", ffmpeg_library_path=arguments.ffmpeg_lib.resolve()
    )
    initial_gpu = observe_gpu()
    samples: list[dict[str, object]] = [asdict(initial_gpu)]
    stop_sampler = threading.Event()
    sampler: threading.Thread | None = None
    gates: dict[str, dict[str, object]] = {}
    release: dict[str, object] = {"status": "NOT_STARTED"}
    owned_pid = 0
    try:
        sampler = threading.Thread(target=_sample_gpu, args=(stop_sampler, samples), daemon=True)
        sampler.start()
        owned = runtime.start(resolved)
        owned_pid = owned.pid
        ready = runtime.wait_ready(resolved)
        _write_json(output / "readiness.json", ready)
        endpoint = endpoint_for(profile)
        with httpx.Client(timeout=httpx.Timeout(profile.request_timeout)) as client:
            _run_gate(
                gates,
                "model_identity",
                lambda: _identity(client, endpoint, profile.served_model_name),
            )
            _run_gate(
                gates,
                "normal_inference",
                lambda: _normal(client, endpoint, profile.served_model_name),
            )
            _run_gate(
                gates, "streaming", lambda: _stream(client, endpoint, profile.served_model_name)
            )
            _run_gate(
                gates,
                "reasoning",
                lambda: _reasoning_gate(client, endpoint, profile.served_model_name),
            )
            _run_gate(
                gates,
                "tool_calling",
                lambda: _tool_gate(client, endpoint, profile.served_model_name),
            )
            _run_gate(
                gates,
                "multi_turn",
                lambda: _multi_turn(client, endpoint, profile.served_model_name),
            )
            _run_gate(
                gates,
                "cancellation",
                lambda: _cancellation(client, endpoint, profile.served_model_name),
            )
            _run_gate(
                gates,
                "context_window",
                lambda: _context_window(
                    client, endpoint, profile.served_model_name, profile.max_model_len
                ),
            )
            _run_gate(
                gates,
                "runtime_stability",
                lambda: _runtime_stability(client, endpoint, profile.served_model_name, owned_pid),
            )
            _run_gate(
                gates,
                "quantized_kernel",
                lambda: _quantized_kernel(Path(profile.model_path), Path(owned.log_path)),
            )
            if arguments.profile_id == "P7_qwen38_w4a16_mtp":
                expected_mtp_tokens = _mtp_tokens(profile)
                _run_gate(
                    gates,
                    "mtp",
                    lambda: _mtp(client, endpoint, Path(owned.log_path), expected_mtp_tokens),
                )
    except Exception as exc:  # noqa: BLE001 - top-level terminal evidence
        gates["startup"] = {
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc)[:2_000],
        }
    finally:
        stop_sampler.set()
        if sampler is not None:
            sampler.join(timeout=5)
        if runtime.ownership_path.exists():
            try:
                release = runtime.release_owned()
            except ServingRuntimeError as exc:
                release = {
                    "status": "FAIL",
                    "category": exc.category,
                    "evidence": exc.evidence,
                }
    peak_used = max(int(sample["used_mib"]) for sample in samples)
    total_mib = int(samples[0]["total_mib"])
    headroom = total_mib - peak_used
    gates["gpu_headroom"] = {
        "status": "PASS" if headroom >= 4_096 else "FAIL",
        "initial_used_mib": initial_gpu.used_mib,
        "peak_used_mib": peak_used,
        "peak_profile_delta_mib": peak_used - initial_gpu.used_mib,
        "minimum_free_headroom_mib": headroom,
        "sample_count": len(samples),
        "sampling_interval_seconds": 0.25,
    }
    required = [*MANDATORY_GATES]
    if arguments.profile_id == "P7_qwen38_w4a16_mtp":
        required.append("mtp")
    endpoint_released = _endpoint_down(endpoint_for(profile))
    try:
        residual_gpu = observe_gpu()
        residual_evidence: dict[str, object] = {
            "status": "OBSERVED",
            "used_mib": residual_gpu.used_mib,
            "free_mib": residual_gpu.free_mib,
            "profile_delta_mib": residual_gpu.used_mib - initial_gpu.used_mib,
            "compute_pids": list(residual_gpu.compute_pids),
            "captured_at_utc": residual_gpu.captured_at_utc,
        }
    except ServingRuntimeError as exc:
        residual_evidence = {
            "status": "UNAVAILABLE",
            "category": exc.category,
            "evidence": exc.evidence,
        }
    passed = (
        all(gates.get(name, {}).get("status") == "PASS" for name in required)
        and release.get("status") == "RELEASED_OWNED_PROFILE"
        and residual_evidence.get("status") == "OBSERVED"
        and endpoint_released
    )
    result = {
        "schema_version": 1,
        "status": "PASSED" if passed else "FAILED",
        "profile_id": profile.profile_id,
        "profile_sha256": _sha256(profile_path),
        "model_id": profile.served_model_name,
        "artifact_sha256": artifact["artifact_sha256"],
        "base_revision": artifact["base_revision"],
        "repository_revision": subprocess.run(  # nosec B603 B607
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip(),
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "installed_vllm": capabilities.version,
        "installed_help_sha256": capabilities.help_sha256,
        "resolution_sha256": resolved.resolution_sha256,
        "gates": gates,
        "release": release,
        "residual_gpu_after_release": residual_evidence,
        "endpoint_down_after_release": endpoint_released,
        "unrelated_processes_signalled": False,
    }
    _write_json(output / "gpu_samples.json", samples)
    _write_json(output / "certification.json", result)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
