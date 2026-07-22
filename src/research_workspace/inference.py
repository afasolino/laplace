"""Measured local A6000 serving paths without CPU substitution."""

from __future__ import annotations

import concurrent.futures
import csv
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from .engineering import JsonObject, LocalToolRunner, collect_cuda_evidence
from .llm import LocalGenerationBackend, ModelRequired, SglangProvider, VllmProvider


Engine = Literal["vllm", "sglang"]


@dataclass(frozen=True)
class ServingCandidate:
    engine: Engine
    endpoint: str
    model: str
    revision: str
    quantization: str
    kernel: str
    prefix_caching: bool
    chunked_prefill: bool
    cuda_graph_mode: str
    scheduler: str
    model_path: str | None = None
    context_tokens: int = 8192
    max_output_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = 0
    request_timeout_seconds: int = 120
    context_safety_margin_tokens: int = 512
    minimum_completion_tokens: int = 256
    reviewer_max_output_tokens: int = 768
    structured_serialization_max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        parsed = urlparse(self.endpoint)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Serving endpoints must use loopback-only HTTP")
        if self.model_path is not None and not self.model_path.strip():
            raise ValueError("Serving model_path must be non-empty text or null")
        if self.context_tokens < 1024 or self.context_tokens > 262_144:
            raise ValueError("Serving context_tokens must be between 1024 and 262144")
        if self.max_output_tokens < 1 or self.max_output_tokens > 8192:
            raise ValueError("Serving max_output_tokens must be between 1 and 8192")
        structured_cap = self.structured_serialization_max_output_tokens
        if structured_cap is None:
            structured_cap = self.max_output_tokens
            object.__setattr__(self, "structured_serialization_max_output_tokens", structured_cap)
        if structured_cap < 1 or structured_cap > 8192:
            raise ValueError(
                "Structured serialization max output must be between 1 and 8192"
            )
        if self.temperature < 0.0 or self.temperature > 2.0:
            raise ValueError("Serving temperature must be between 0 and 2")
        if self.top_p <= 0.0 or self.top_p > 1.0:
            raise ValueError("Serving top_p must be greater than 0 and at most 1")
        if self.request_timeout_seconds < 1 or self.request_timeout_seconds > 1800:
            raise ValueError("Serving request timeout must be between 1 and 1800 seconds")
        if self.context_safety_margin_tokens < 64 or self.context_safety_margin_tokens > 4096:
            raise ValueError("Serving context safety margin must be between 64 and 4096 tokens")
        if self.minimum_completion_tokens < 64 or self.minimum_completion_tokens > 2048:
            raise ValueError("Serving minimum completion must be between 64 and 2048 tokens")
        if self.minimum_completion_tokens > self.max_output_tokens:
            raise ValueError("Serving minimum completion cannot exceed max_output_tokens")
        if self.reviewer_max_output_tokens < 64 or self.reviewer_max_output_tokens > 2048:
            raise ValueError("Reviewer max output must be between 64 and 2048 tokens")

    def to_json(self) -> JsonObject:
        return {
            "engine": self.engine,
            "endpoint": self.endpoint,
            "model": self.model,
            "model_path": self.model_path,
            "revision": self.revision,
            "quantization": self.quantization,
            "kernel": self.kernel,
            "prefix_caching": self.prefix_caching,
            "chunked_prefill": self.chunked_prefill,
            "cuda_graph_mode": self.cuda_graph_mode,
            "scheduler": self.scheduler,
            "context_tokens": self.context_tokens,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "request_timeout_seconds": self.request_timeout_seconds,
            "context_safety_margin_tokens": self.context_safety_margin_tokens,
            "minimum_completion_tokens": self.minimum_completion_tokens,
            "reviewer_max_output_tokens": self.reviewer_max_output_tokens,
            "structured_serialization_max_output_tokens": (
                self.structured_serialization_max_output_tokens
            ),
        }


def backend_for(candidate: ServingCandidate) -> LocalGenerationBackend:
    if candidate.engine == "vllm":
        return VllmProvider(
            candidate.endpoint,
            candidate.model,
            max_tokens=candidate.max_output_tokens,
            temperature=candidate.temperature,
            top_p=candidate.top_p,
            seed=candidate.seed,
            timeout_seconds=candidate.request_timeout_seconds,
        )
    return SglangProvider(
        candidate.endpoint,
        candidate.model,
        max_tokens=candidate.max_output_tokens,
        temperature=candidate.temperature,
        top_p=candidate.top_p,
        seed=candidate.seed,
        timeout_seconds=candidate.request_timeout_seconds,
    )


def _measure_once(backend: LocalGenerationBackend, prompt: str, context_tokens: int) -> JsonObject:
    started = time.monotonic()
    result = backend.generate(prompt, context_tokens=context_tokens)
    return {
        "status": result.status,
        "model": result.model,
        "wall_seconds": time.monotonic() - started,
        "ttft_seconds": result.ttft_seconds,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "input_tokens_per_second": result.input_tokens_per_second,
        "output_tokens_per_second": result.output_tokens_per_second,
        "output_characters": len(result.text),
    }


def _context_prompt(
    backend: LocalGenerationBackend, prompt: str, requested_context_tokens: int
) -> tuple[str, int | None]:
    """Fill vLLM prompts using its tokenizer, never an estimated word/token ratio."""
    if not isinstance(backend, VllmProvider):
        return prompt, None
    reserve = 512
    target = requested_context_tokens - reserve
    if target <= 0:
        raise ValueError("Requested context is too small for the benchmark response reserve")
    prefix = (
        prompt.rstrip() + "\n\nContext padding follows; ignore it and follow the final request.\n"
    )
    padding_unit = "context\n"
    prefix_count = backend.token_count(prefix)
    unit_count = backend.token_count(padding_unit)
    if unit_count < 1:
        raise ModelRequired("Local tokenizer returned a zero-token benchmark padding unit")
    maximum = max(0, (target - prefix_count) // unit_count)
    low, high = 0, maximum
    best = prefix
    best_count = prefix_count
    while low <= high:
        midpoint = (low + high) // 2
        candidate = prefix + padding_unit * midpoint + "\nReply with exactly the single word LOCAL."
        count = backend.token_count(candidate)
        if count <= target:
            best, best_count = candidate, count
            low = midpoint + 1
        else:
            high = midpoint - 1
    return best, best_count


def gpu_memory_snapshot(runner: LocalToolRunner) -> JsonObject:
    """Capture an observed host GPU memory point without estimating peak VRAM."""
    result = runner.run(
        "cuda_probe",
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout_seconds=30,
    )
    fields = [item.strip() for item in result.stdout.strip().split(",")]
    if result.returncode != 0 or len(fields) != 5:
        return {"status": "UNAVAILABLE", "command_log": result.log_path, "stderr": result.stderr}
    try:
        return {
            "status": "OBSERVED",
            "gpu_name": fields[0],
            "memory_used_mib": int(fields[1]),
            "memory_free_mib": int(fields[2]),
            "memory_total_mib": int(fields[3]),
            "gpu_utilization_percent": int(fields[4]),
            "command_log": result.log_path,
        }
    except ValueError:
        return {"status": "UNAVAILABLE", "command_log": result.log_path, "stdout": result.stdout}


def write_gpu_inference_reports(repository_root: Path, result: JsonObject) -> JsonObject:
    """Persist JSON and CSV measurements with one row per actual request."""
    root = repository_root.resolve() / "outputs" / "a6000_agent_team" / "benchmarks"
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "gpu_inference_results.json"
    csv_path = root / "gpu_inference_results.csv"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = [
        "requested_context_tokens",
        "concurrency",
        "run_index",
        "status",
        "wall_seconds",
        "ttft_seconds",
        "prompt_tokens",
        "completion_tokens",
        "input_tokens_per_second",
        "output_tokens_per_second",
        "output_characters",
        "observed_vram_before_mib",
        "observed_vram_after_mib",
    ]
    measurements = result.get("measurements")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        if isinstance(measurements, list):
            for measurement in measurements:
                if not isinstance(measurement, dict):
                    continue
                runs = measurement.get("runs")
                if not isinstance(runs, list):
                    continue
                before = measurement.get("gpu_before")
                after = measurement.get("gpu_after")
                for index, run in enumerate(runs):
                    if not isinstance(run, dict):
                        continue
                    writer.writerow(
                        {
                            "requested_context_tokens": measurement.get("context_tokens"),
                            "concurrency": measurement.get("concurrency"),
                            "run_index": index,
                            "status": run.get("status"),
                            "wall_seconds": run.get("wall_seconds"),
                            "ttft_seconds": run.get("ttft_seconds"),
                            "prompt_tokens": run.get("prompt_tokens"),
                            "completion_tokens": run.get("completion_tokens"),
                            "input_tokens_per_second": run.get("input_tokens_per_second"),
                            "output_tokens_per_second": run.get("output_tokens_per_second"),
                            "output_characters": run.get("output_characters"),
                            "observed_vram_before_mib": before.get("memory_used_mib")
                            if isinstance(before, dict)
                            else None,
                            "observed_vram_after_mib": after.get("memory_used_mib")
                            if isinstance(after, dict)
                            else None,
                        }
                    )
    return {"json": str(json_path), "csv": str(csv_path)}


def benchmark_local_candidate(
    repository_root: Path,
    candidate: ServingCandidate,
    *,
    prompt: str,
    contexts: tuple[int, ...] = (8192, 16384, 24576, 32768),
    concurrency: tuple[int, ...] = (1, 2, 4),
    timeout_seconds: int = 900,
) -> JsonObject:
    """Run a real local serving sweep only after CUDA/A6000 proof succeeds."""
    runner = LocalToolRunner(repository_root)
    cuda = collect_cuda_evidence(runner)
    if cuda["status"] != "CUDA_A6000_VERIFIED":
        return {
            "status": "BLOCKED_GPU",
            "candidate": candidate.to_json(),
            "cuda_evidence": cuda,
            "measurements": [],
            "reason": "No CPU benchmark is emitted for an A6000 serving candidate.",
        }
    backend = backend_for(candidate)
    health = backend.health()
    if health.get("status") != "AVAILABLE":
        return {
            "status": "MODEL_REQUIRED",
            "candidate": candidate.to_json(),
            "cuda_evidence": cuda,
            "health": health,
            "measurements": [],
        }
    if not prompt.strip():
        raise ValueError("Benchmark prompt must not be empty")
    if isinstance(backend, VllmProvider):
        backend = VllmProvider(
            candidate.endpoint,
            candidate.model,
            max_tokens=256,
            temperature=candidate.temperature,
            top_p=candidate.top_p,
            seed=candidate.seed,
            timeout_seconds=candidate.request_timeout_seconds,
        )
    measurements: list[JsonObject] = []
    for context_tokens in contexts:
        if context_tokens < 8192 or context_tokens > 32768:
            raise ValueError("Contexts must be between 8192 and 32768")
        context_prompt, tokenizer_prompt_tokens = _context_prompt(backend, prompt, context_tokens)
        for workers in concurrency:
            if workers not in {1, 2, 4}:
                raise ValueError("Concurrency must be one of 1, 2 or 4")
            gpu_before = gpu_memory_snapshot(runner)
            started = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(_measure_once, backend, context_prompt, context_tokens)
                    for _ in range(workers)
                ]
                runs = [future.result(timeout=timeout_seconds) for future in futures]
            wall = time.monotonic() - started
            gpu_after = gpu_memory_snapshot(runner)
            output_rates = [item["output_tokens_per_second"] for item in runs]
            numeric_rates = [float(item) for item in output_rates if isinstance(item, (int, float))]
            ttfts = [item["ttft_seconds"] for item in runs]
            numeric_ttfts = [float(item) for item in ttfts if isinstance(item, (int, float))]
            measurements.append(
                {
                    "context_tokens": context_tokens,
                    "tokenizer_prompt_tokens": tokenizer_prompt_tokens,
                    "concurrency": workers,
                    "wall_seconds": wall,
                    "runs": runs,
                    "gpu_before": gpu_before,
                    "gpu_after": gpu_after,
                    "mean_ttft_seconds": statistics.mean(numeric_ttfts) if numeric_ttfts else None,
                    "aggregate_output_tokens_per_second": sum(numeric_rates)
                    if numeric_rates
                    else None,
                    "mean_output_tokens_per_second": statistics.mean(numeric_rates)
                    if numeric_rates
                    else None,
                }
            )
    report: JsonObject = {
        "status": "MEASURED",
        "candidate": candidate.to_json(),
        "backend_identity": backend.model_identity(),
        "health": health,
        "cuda_evidence": cuda,
        "measurements": measurements,
        "measurement_warning": "All token rates derive from actual server usage fields and wall-clock timestamps. GPU memory is observed immediately before and after each request group; this report does not claim a sampled value is a peak.",
    }
    report["reports"] = write_gpu_inference_reports(repository_root, report)
    return report
