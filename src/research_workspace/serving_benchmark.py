"""Deterministic fixture and live workload runner for tiered local serving."""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, TypeAlias
from urllib.parse import urlsplit

from .service_tiers import ModelLane

JsonObject: TypeAlias = dict[str, object]


class ServingBenchmarkError(RuntimeError):
    """A benchmark configuration or live response is invalid."""


@dataclass(frozen=True)
class BenchmarkRequest:
    request_id: str
    capability_tier: Literal["basic", "plus"]
    lane: ModelLane
    domain: str
    prompt: str
    context_tokens: int
    max_output_tokens: int
    expected_markers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.request_id or not self.prompt:
            raise ValueError("request_id and prompt are required")
        if self.context_tokens < 0 or self.max_output_tokens < 1:
            raise ValueError("invalid token limits")


@dataclass(frozen=True)
class RequestMeasurement:
    profile_id: str
    request_id: str
    capability_tier: str
    lane: str
    domain: str
    context_tokens: int
    concurrency: int
    status: str
    queue_time_ms: float | None
    ttft_ms: float | None
    e2e_ms: float
    mean_inter_token_latency_ms: float | None
    input_tokens: int | None
    output_tokens: int | None
    output_tokens_per_second: float | None
    marker_recall: float | None
    error_category: str | None


@dataclass(frozen=True)
class BenchmarkSummary:
    profile_id: str
    request_count: int
    success_count: int
    error_count: int
    aggregate_output_tokens_per_second: float
    p50_ttft_ms: float | None
    p95_ttft_ms: float | None
    p50_e2e_ms: float | None
    p95_e2e_ms: float | None
    marker_recall: float | None


@dataclass(frozen=True)
class QualityCase:
    case_id: str
    domain: str
    prompt: str
    required_markers: tuple[str, ...]
    require_json_object: bool
    forbidden_markers: tuple[str, ...]
    source_policy: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> QualityCase:
        expected = {
            "case_id",
            "domain",
            "prompt",
            "required_markers",
            "require_json_object",
            "forbidden_markers",
            "source_policy",
        }
        if set(value) != expected:
            raise ServingBenchmarkError("invalid quality manifest fields")
        required = value["required_markers"]
        forbidden = value["forbidden_markers"]
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            raise ServingBenchmarkError("invalid required_markers")
        if not isinstance(forbidden, list) or not all(isinstance(item, str) for item in forbidden):
            raise ServingBenchmarkError("invalid forbidden_markers")
        if not all(
            isinstance(value[field], str)
            for field in ("case_id", "domain", "prompt", "source_policy")
        ) or not isinstance(value["require_json_object"], bool):
            raise ServingBenchmarkError("invalid quality case")
        return cls(
            case_id=str(value["case_id"]),
            domain=str(value["domain"]),
            prompt=str(value["prompt"]),
            required_markers=tuple(required),
            require_json_object=bool(value["require_json_object"]),
            forbidden_markers=tuple(forbidden),
            source_policy=str(value["source_policy"]),
        )


@dataclass(frozen=True)
class QualityResult:
    profile_id: str
    case_id: str
    domain: str
    score: float
    passed_hard_gates: bool
    missing_markers: tuple[str, ...]
    present_forbidden_markers: tuple[str, ...]
    json_object_valid: bool | None


def load_quality_manifest(path: Path) -> tuple[QualityCase, ...]:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    cases = raw.get("cases") if isinstance(raw, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ServingBenchmarkError("quality manifest has no cases")
    parsed = tuple(
        QualityCase.from_mapping(item)
        for item in cases
        if isinstance(item, dict)
    )
    if len(parsed) != len(cases) or len({item.case_id for item in parsed}) != len(parsed):
        raise ServingBenchmarkError("quality manifest cases are malformed or duplicated")
    return parsed


def score_quality_output(
    profile_id: str, case: QualityCase, output: str
) -> QualityResult:
    missing = tuple(marker for marker in case.required_markers if marker not in output)
    forbidden = tuple(marker for marker in case.forbidden_markers if marker in output)
    json_valid: bool | None = None
    if case.require_json_object:
        try:
            parsed: object = json.loads(output)
            json_valid = isinstance(parsed, dict)
        except json.JSONDecodeError:
            json_valid = False
    marker_score = (
        (len(case.required_markers) - len(missing)) / len(case.required_markers)
        if case.required_markers
        else 1.0
    )
    hard = not forbidden and json_valid is not False
    return QualityResult(
        profile_id=profile_id,
        case_id=case.case_id,
        domain=case.domain,
        score=marker_score if hard else 0.0,
        passed_hard_gates=hard,
        missing_markers=missing,
        present_forbidden_markers=forbidden,
        json_object_valid=json_valid,
    )


def percentile(values: Sequence[float], percentile_value: float) -> float | None:
    if not values:
        return None
    if not 0 <= percentile_value <= 100:
        raise ValueError("percentile must be between zero and 100")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def tier_mix(total: int) -> tuple[ModelLane, ...]:
    """Return an exact deterministic 20/60/20 mix when total is divisible by five."""

    if total < 5 or total % 5:
        raise ValueError("tier workload size must be a positive multiple of five")
    cycle = (
        ModelLane.STANDARD,
        ModelLane.QUALITY,
        ModelLane.STANDARD,
        ModelLane.ECONOMY,
        ModelLane.STANDARD,
    )
    return cycle * (total // 5)


def context_probe_prompt(approximate_tokens: int) -> tuple[str, tuple[str, ...]]:
    """Build beginning/middle/end retrieval markers without claiming tokenizer exactness."""

    if approximate_tokens < 2_048:
        raise ValueError("context probe must be at least 2k approximate tokens")
    markers = (
        f"LAPLACE_BEGIN_{approximate_tokens}",
        f"LAPLACE_MIDDLE_{approximate_tokens}",
        f"LAPLACE_END_{approximate_tokens}",
    )
    filler = "local deterministic context capacity probe "
    character_budget = approximate_tokens * 4
    segment_size = max(1, (character_budget - sum(map(len, markers))) // 2)
    first = (filler * (segment_size // len(filler) + 1))[:segment_size]
    second = (filler * (segment_size // len(filler) + 1))[:segment_size]
    prompt = (
        f"{markers[0]}\n{first}\n{markers[1]}\n{second}\n{markers[2]}\n"
        "Return the three LAPLACE markers exactly, in their original order."
    )
    return prompt, markers


class OpenAIStreamingProbe:
    """Measure a localhost OpenAI-compatible streaming response."""

    def __init__(self, endpoint: str, model_id: str, *, timeout_seconds: float) -> None:
        parsed = urlsplit(endpoint)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("benchmark endpoint must be localhost HTTP")
        self.endpoint = endpoint.rstrip("/")
        self.model_id = model_id
        self.timeout_seconds = timeout_seconds

    def __call__(
        self,
        request: BenchmarkRequest,
        *,
        profile_id: str,
        concurrency: int,
    ) -> RequestMeasurement:
        started = time.perf_counter()
        body = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": 0,
            "max_tokens": request.max_output_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
            "priority": {
                ModelLane.QUALITY: 0,
                ModelLane.STANDARD: 10,
                ModelLane.ECONOMY: 20,
            }[request.lane],
        }
        http_request = urllib.request.Request(  # nosec B310 - localhost checked
            self.endpoint + "/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Request-Id": request.request_id,
            },
            method="POST",
        )
        first_token_at: float | None = None
        token_arrivals: list[float] = []
        text_parts: list[str] = []
        usage: Mapping[str, object] = {}
        try:
            with urllib.request.urlopen(  # nosec B310 - localhost checked
                http_request, timeout=self.timeout_seconds
            ) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    payload: object = json.loads(line[6:])
                    if not isinstance(payload, dict):
                        continue
                    candidate_usage = payload.get("usage")
                    if isinstance(candidate_usage, dict):
                        usage = candidate_usage
                    choices = payload.get("choices")
                    if not isinstance(choices, list) or not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") if isinstance(choice, dict) else None
                    content = delta.get("content") if isinstance(delta, dict) else None
                    if isinstance(content, str) and content:
                        now = time.perf_counter()
                        first_token_at = first_token_at or now
                        token_arrivals.append(now)
                        text_parts.append(content)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            elapsed = (time.perf_counter() - started) * 1_000
            return RequestMeasurement(
                profile_id,
                request.request_id,
                request.capability_tier,
                request.lane.value,
                request.domain,
                request.context_tokens,
                concurrency,
                "ERROR",
                None,
                None,
                elapsed,
                None,
                None,
                None,
                None,
                None,
                type(exc).__name__,
            )
        finished = time.perf_counter()
        output_tokens_value = usage.get("completion_tokens")
        input_tokens_value = usage.get("prompt_tokens")
        output_tokens = output_tokens_value if isinstance(output_tokens_value, int) else None
        input_tokens = input_tokens_value if isinstance(input_tokens_value, int) else None
        generation_seconds = (
            finished - first_token_at if first_token_at is not None else finished - started
        )
        throughput = (
            output_tokens / generation_seconds
            if output_tokens is not None and generation_seconds > 0
            else None
        )
        inter_token = (
            statistics.fmean(
                (right - left) * 1_000
                for left, right in zip(token_arrivals, token_arrivals[1:])
            )
            if len(token_arrivals) > 1
            else None
        )
        output_text = "".join(text_parts)
        marker_recall = (
            sum(marker in output_text for marker in request.expected_markers)
            / len(request.expected_markers)
            if request.expected_markers
            else None
        )
        return RequestMeasurement(
            profile_id=profile_id,
            request_id=request.request_id,
            capability_tier=request.capability_tier,
            lane=request.lane.value,
            domain=request.domain,
            context_tokens=request.context_tokens,
            concurrency=concurrency,
            status="SUCCESS",
            queue_time_ms=None,
            ttft_ms=(first_token_at - started) * 1_000 if first_token_at else None,
            e2e_ms=(finished - started) * 1_000,
            mean_inter_token_latency_ms=inter_token,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output_tokens_per_second=throughput,
            marker_recall=marker_recall,
            error_category=None,
        )


Probe = Callable[..., RequestMeasurement]


def run_concurrent(
    requests: Sequence[BenchmarkRequest],
    *,
    profile_id: str,
    concurrency: int,
    probe: Probe,
) -> tuple[RequestMeasurement, ...]:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    measurements: list[RequestMeasurement] = []
    submitted_at = time.perf_counter()

    def measured_probe(request: BenchmarkRequest) -> RequestMeasurement:
        started_at = time.perf_counter()
        result = probe(
            request,
            profile_id=profile_id,
            concurrency=concurrency,
        )
        return replace(
            result,
            queue_time_ms=max(0.0, (started_at - submitted_at) * 1_000),
        )

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(measured_probe, request)
            for request in requests
        ]
        for future in as_completed(futures):
            measurements.append(future.result())
    return tuple(sorted(measurements, key=lambda item: item.request_id))


def summarize(
    profile_id: str, measurements: Sequence[RequestMeasurement]
) -> BenchmarkSummary:
    successes = [item for item in measurements if item.status == "SUCCESS"]
    ttft = [item.ttft_ms for item in successes if item.ttft_ms is not None]
    e2e = [item.e2e_ms for item in successes]
    load_successes = [item for item in successes if item.request_id.startswith("load-")]
    throughput_population = load_successes or successes
    highest_concurrency = max(
        (item.concurrency for item in throughput_population),
        default=0,
    )
    highest_batch = [
        item
        for item in throughput_population
        if item.concurrency == highest_concurrency
    ]
    highest_batch_tokens = sum(item.output_tokens or 0 for item in highest_batch)
    highest_batch_wall_ms = max(
        (
            (item.queue_time_ms or 0.0) + item.e2e_ms
            for item in highest_batch
        ),
        default=0.0,
    )
    aggregate_throughput = (
        highest_batch_tokens / (highest_batch_wall_ms / 1_000)
        if highest_batch_tokens and highest_batch_wall_ms > 0
        else 0.0
    )
    recalls = [
        item.marker_recall for item in successes if item.marker_recall is not None
    ]
    return BenchmarkSummary(
        profile_id=profile_id,
        request_count=len(measurements),
        success_count=len(successes),
        error_count=len(measurements) - len(successes),
        aggregate_output_tokens_per_second=aggregate_throughput,
        p50_ttft_ms=percentile(ttft, 50),
        p95_ttft_ms=percentile(ttft, 95),
        p50_e2e_ms=percentile(e2e, 50),
        p95_e2e_ms=percentile(e2e, 95),
        marker_recall=statistics.fmean(recalls) if recalls else None,
    )


def write_measurements(
    output_root: Path,
    measurements: Sequence[RequestMeasurement],
    summary: BenchmarkSummary,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "request_measurements.csv"
    fields = list(RequestMeasurement.__dataclass_fields__)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(item) for item in measurements)
    (output_root / "benchmark_summary.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_mix(requests: Iterable[BenchmarkRequest]) -> dict[str, int]:
    counts = Counter(request.lane.value for request in requests)
    return dict(sorted(counts.items()))
