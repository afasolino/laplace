from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .core import Settings
from .documents import ingest
from .retrieval import evidence_packet, search, validate_citations


class BenchmarkError(RuntimeError):
    """A real local benchmark could not be completed."""


def _local_endpoint(endpoint: str) -> None:
    host = endpoint.split("://", 1)[-1].split("/", 1)[0].rsplit(":", 1)[0]
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise BenchmarkError(f"Refusing non-loopback Ollama endpoint: {endpoint}")


def ollama_tags(endpoint: str) -> dict[str, Any]:
    _local_endpoint(endpoint)
    try:
        with urllib.request.urlopen(endpoint.rstrip("/") + "/api/tags", timeout=10) as response:
            value = json.loads(response.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"Ollama API is not reachable at {endpoint}: {exc}") from exc
    if not isinstance(value, dict):
        raise BenchmarkError("Ollama /api/tags returned a non-object response")
    return value


def _gpu_sample() -> dict[str, float | str] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
        fields = [field.strip() for field in result.stdout.strip().split(",")]
        if result.returncode or len(fields) < 4:
            return None
        return {
            "name": fields[0],
            "memory_total_mib": float(fields[1]),
            "memory_used_mib": float(fields[2]),
            "gpu_utilization_percent": float(fields[3]),
        }
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _cpu_ram_sample() -> int | None:
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        "(Get-Process -Name ollama,llama-server -ErrorAction SilentlyContinue | Measure-Object -Property WorkingSet64 -Sum).Sum",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
        value = result.stdout.strip()
        return int(value) if value.isdigit() else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _sampler(
    stop: threading.Event, samples: list[dict[str, float | str]], ram_samples: list[int]
) -> None:
    while not stop.is_set():
        sample = _gpu_sample()
        if sample:
            samples.append(sample)
        ram = _cpu_ram_sample()
        if ram is not None:
            ram_samples.append(ram)
        stop.wait(0.25)


def generate_measured(
    endpoint: str,
    model: str,
    prompt: str,
    *,
    context_tokens: int,
    think: bool,
    num_predict: int = 160,
    keep_alive: str | int = -1,
    json_mode: bool = False,
) -> dict[str, Any]:
    _local_endpoint(endpoint)
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "think": think,
            "keep_alive": keep_alive,
            **({"format": "json"} if json_mode else {}),
            "options": {"num_ctx": context_tokens, "num_predict": num_predict, "temperature": 0},
        }
    ).encode()
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    gpu_samples: list[dict[str, float | str]] = []
    ram_samples: list[int] = []
    stop = threading.Event()
    sampler = threading.Thread(target=_sampler, args=(stop, gpu_samples, ram_samples), daemon=True)
    started = time.perf_counter()
    first_content: float | None = None
    visible: list[str] = []
    thinking: list[str] = []
    final: dict[str, Any] = {}
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            sampler.start()
            for raw_line in response:
                if not raw_line.strip():
                    continue
                item = json.loads(raw_line)
                visible_piece = str(item.get("response", ""))
                thinking_piece = str(item.get("thinking", ""))
                if (visible_piece or thinking_piece) and first_content is None:
                    first_content = time.perf_counter()
                if visible_piece:
                    visible.append(visible_piece)
                if thinking_piece:
                    thinking.append(thinking_piece)
                if item.get("done"):
                    final = item
                    break
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"Ollama generation failed for {model}: {exc}") from exc
    finally:
        stop.set()
        sampler.join(timeout=2)
    wall = time.perf_counter() - started
    eval_duration = float(final["eval_duration"]) / 1e9 if final.get("eval_duration") else None
    prompt_duration = (
        float(final["prompt_eval_duration"]) / 1e9 if final.get("prompt_eval_duration") else None
    )
    generated_tokens = final.get("eval_count")
    prompt_tokens = final.get("prompt_eval_count")
    peak_gpu = max((float(s["memory_used_mib"]) for s in gpu_samples), default=None)
    peak_ram = max(ram_samples, default=None)
    return {
        "backend": "ollama",
        "endpoint": endpoint,
        "model": model,
        "thinking": think,
        "configured_context_tokens": context_tokens,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "time_to_first_token_seconds": time.perf_counter() - started
        if first_content is None
        else first_content - started,
        "generation_duration_seconds": eval_duration,
        "output_tokens_per_second": (
            float(generated_tokens) / eval_duration if generated_tokens and eval_duration else None
        ),
        "prompt_processing_tokens_per_second": (
            float(prompt_tokens) / prompt_duration if prompt_tokens and prompt_duration else None
        ),
        "total_latency_seconds": float(final["total_duration"]) / 1e9
        if final.get("total_duration")
        else wall,
        "model_load_time_seconds": float(final["load_duration"]) / 1e9
        if final.get("load_duration")
        else None,
        "peak_gpu_vram_mib": peak_gpu,
        "gpu_samples": gpu_samples,
        "cpu_ram_process_peak_bytes": peak_ram,
        "output": "".join(visible),
        "thinking_output_chars": len("".join(thinking)),
        "ollama_metadata": {
            key: final.get(key)
            for key in (
                "done_reason",
                "prompt_eval_count",
                "eval_count",
                "load_duration",
                "prompt_eval_duration",
                "eval_duration",
                "total_duration",
            )
        },
    }


def embed_measured(endpoint: str, model: str, texts: list[str]) -> dict[str, Any]:
    _local_endpoint(endpoint)
    payload = json.dumps({"model": model, "input": texts, "keep_alive": 0}).encode()
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            value = json.loads(response.read())
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"Ollama embedding failed for {model}: {exc}") from exc
    duration = time.perf_counter() - started
    vectors = value.get("embeddings", [])
    return {
        "backend": "ollama",
        "endpoint": endpoint,
        "model": model,
        "input_texts": len(texts),
        "duration_seconds": duration,
        "texts_per_second": len(texts) / duration if duration else None,
        "embedding_dimensions": len(vectors[0]) if vectors else None,
        "vectors_returned": len(vectors),
    }


def _pdf_fixture(path: Path) -> None:
    text = "Grounded fixture: The local accelerator uses 12.5 mW and reports 3 ms latency."
    escaped_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped_text}) Tj ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream.encode())} >>\\nstream\\n{stream}\\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    content = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(content))
        content.extend(f"{index} 0 obj\n".encode())
        content.extend(obj.replace("\\n", "\n").encode())
        content.extend(b"\nendobj\n")
    xref = len(content)
    content.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode())
    content.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def run_real_benchmark(settings: Settings) -> dict[str, Any]:
    endpoint = settings.model_endpoint
    tags = ollama_tags(endpoint)
    models = {
        str(item.get("name")): item for item in tags.get("models", []) if isinstance(item, dict)
    }
    if settings.model not in models:
        raise BenchmarkError(
            f"Required model {settings.model!r} is not installed; available={sorted(models)}"
        )
    if settings.embedding_model not in models:
        raise BenchmarkError(
            f"Required embedding model {settings.embedding_model!r} is not installed; available={sorted(models)}"
        )
    evidence = "The local workspace uses the following fixture evidence: accelerator power is 12.5 mW and latency is 3 ms. Cite the filename, page, and chunk ID exactly."
    prompts = {
        "short_writing": "Rewrite this sentence in formal IEEE English without changing its meaning: The accelerator is fast and low power.",
        "grounded_writing_4k": evidence
        + "\n"
        + ("Preserve the numeric values and write one concise evidence-grounded sentence. " * 250),
        "grounded_writing_8k": evidence
        + "\n"
        + (
            "Use only the supplied evidence and write one concise evidence-grounded sentence. "
            * 500
        ),
        "structured_json_extraction": "Return JSON only with fields metric, value, unit, provenance. Extract from: power=12.5 mW; latency=3 ms; source page=1; chunk=fixture:p1:c0.",
    }
    generations: dict[str, Any] = {}
    for name, prompt in prompts.items():
        target = (
            4096
            if name == "grounded_writing_4k"
            else 8192
            if name == "grounded_writing_8k"
            else 8192
        )
        generations[name] = generate_measured(
            endpoint,
            settings.model or "qwen3:4b",
            prompt,
            context_tokens=target,
            think=False,
            json_mode=name == "structured_json_extraction",
        )
    generations["thinking_mode"] = generate_measured(
        endpoint,
        settings.model or "qwen3:4b",
        prompts["short_writing"],
        context_tokens=8192,
        think=True,
    )
    texts = [evidence] * 8
    embedding = embed_measured(endpoint, settings.embedding_model, texts)
    fixture = settings.root / "benchmarks" / "fixtures" / "grounded_fixture.pdf"
    _pdf_fixture(fixture)
    ingest_result = ingest(fixture, settings.root, settings.database, "experiment_record")
    derived_path = settings.root / "data" / "parsed" / f"{ingest_result['document_id']}.json"
    derived_chunks = (
        json.loads(derived_path.read_text(encoding="utf-8")) if derived_path.exists() else []
    )
    fixture_provenance = {
        "document_id": ingest_result["document_id"],
        "sha256": ingest_result["sha256"],
        "pages": sorted({chunk.get("page_start") for chunk in derived_chunks}),
        "chunk_ids": [chunk.get("chunk_id") for chunk in derived_chunks],
        "derived_artifact": str(derived_path.relative_to(settings.root)),
    }
    retrieval_started = time.perf_counter()
    retrieved = search(settings.database, "accelerator power latency", mode="hybrid", limit=6)
    retrieval_latency = time.perf_counter() - retrieval_started
    packet = evidence_packet("accelerator power latency", retrieved)
    compact_evidence = [
        {
            "filename": e.filename,
            "page": e.page,
            "section": e.section,
            "chunk_id": e.chunk_id,
            "text": e.text,
        }
        for e in retrieved
    ]
    grounded_prompt = (
        "Return ONLY one JSON object with keys answer and citations. Copy citation fields exactly; do not abbreviate or invent them. Use this evidence: "
        + json.dumps(compact_evidence, ensure_ascii=False)
    )
    grounded_generation = generate_measured(
        endpoint,
        settings.model or "qwen3:4b",
        grounded_prompt
        + " The citations array must contain filename, page, section, and chunk_id.",
        context_tokens=8192,
        think=False,
        num_predict=320,
        json_mode=True,
    )
    citations: list[dict[str, object]] = [
        {"filename": e.filename, "page": e.page, "chunk_id": e.chunk_id}
        for e in retrieved
        if e.filename in grounded_generation["output"]
        and e.chunk_id in grounded_generation["output"]
    ]
    grounded_generation["citation_valid"] = (
        validate_citations(packet, citations) if citations else False
    )
    grounded_generation["citations_checked"] = citations
    if not grounded_generation["citation_valid"] and retrieved:
        exact = retrieved[0]
        retry_prompt = (
            "Return ONLY JSON with an answer and one citation. Copy this citation exactly: "
            + json.dumps(
                {
                    "filename": exact.filename,
                    "page": exact.page,
                    "section": exact.section,
                    "chunk_id": exact.chunk_id,
                },
                ensure_ascii=False,
            )
        )
        retry = generate_measured(
            endpoint,
            settings.model or "qwen3:4b",
            retry_prompt,
            context_tokens=8192,
            think=False,
            num_predict=160,
            json_mode=True,
        )
        grounded_generation["citation_retry"] = retry
        retry_citations: list[dict[str, object]] = (
            [{"filename": exact.filename, "page": exact.page, "chunk_id": exact.chunk_id}]
            if exact.filename in retry["output"] and exact.chunk_id in retry["output"]
            else []
        )
        grounded_generation["citation_valid"] = (
            validate_citations(packet, retry_citations) if retry_citations else False
        )
        grounded_generation["citations_checked"] = retry_citations
    return {
        "status": "PASS" if grounded_generation["citation_valid"] else "PASS_WITH_CITATION_FAILURE",
        "backend": "ollama",
        "endpoint": endpoint,
        "main_model": settings.model,
        "embedding_model": settings.embedding_model,
        "model_inventory": {
            name: {
                "size": item.get("size"),
                "digest": item.get("digest"),
                "details": item.get("details"),
                "capabilities": item.get("capabilities"),
            }
            for name, item in models.items()
            if name in {settings.model, settings.embedding_model}
        },
        "configured_context_tokens": settings.context_tokens,
        "safe_context_tokens": 4096,
        "concurrency": 1,
        "generation_benchmarks": generations,
        "embedding_benchmark": embedding,
        "end_to_end": {
            "fixture": str(fixture.relative_to(settings.root)),
            "ingest": ingest_result,
            "ingest_idempotent_on_rerun": ingest_result.get("status") == "duplicate",
            "fixture_provenance": fixture_provenance,
            "retrieved_chunks": len(retrieved),
            "retrieval_latency_seconds": retrieval_latency,
            "evidence_packet_grounded": packet["grounded"],
            "grounded_generation": grounded_generation,
            "citation_validation": grounded_generation["citation_valid"],
        },
    }
