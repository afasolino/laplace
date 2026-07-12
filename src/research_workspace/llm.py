from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class ModelRequired(RuntimeError):
    """No explicitly installed local model is available."""


@dataclass
class GenerationResult:
    text: str
    model: str
    ttft_seconds: float | None
    output_tokens_per_second: float | None
    status: str


class Provider:
    def generate(self, prompt: str, *, context_tokens: int = 8192) -> GenerationResult:
        raise NotImplementedError


class MockProvider(Provider):
    def generate(self, prompt: str, *, context_tokens: int = 8192) -> GenerationResult:
        return GenerationResult("[MOCK] " + prompt[:200], "mock", 0.0, None, "mock")


class OllamaProvider(Provider):
    def __init__(self, endpoint: str, model: str):
        self.endpoint, self.model = endpoint.rstrip("/"), model

    def generate(self, prompt: str, *, context_tokens: int = 8192) -> GenerationResult:
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_ctx": context_tokens, "num_predict": 512},
            }
        ).encode()
        request = urllib.request.Request(
            self.endpoint + "/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                value = json.loads(response.read())
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            raise ModelRequired(f"Local Ollama endpoint unavailable: {exc}") from exc
        elapsed = time.perf_counter() - started
        text = str(value.get("response", ""))
        tokens = value.get("eval_count")
        return GenerationResult(
            text,
            self.model,
            value.get("prompt_eval_duration", 0) / 1e9
            if value.get("prompt_eval_duration")
            else None,
            float(tokens) / elapsed if tokens and elapsed else None,
            "measured",
        )


class OpenAICompatibleProvider(Provider):
    """Client for a localhost OpenAI-compatible server (LM Studio/llama.cpp)."""

    def __init__(self, endpoint: str, model: str):
        self.endpoint, self.model = endpoint.rstrip("/"), model

    def generate(self, prompt: str, *, context_tokens: int = 8192) -> GenerationResult:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 512,
            }
        ).encode()
        request = urllib.request.Request(
            self.endpoint + "/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                value = json.loads(response.read())
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            raise ModelRequired(f"Local OpenAI-compatible endpoint unavailable: {exc}") from exc
        elapsed = time.perf_counter() - started
        choices = value.get("choices", [])
        text = str(choices[0].get("message", {}).get("content", "")) if choices else ""
        usage = value.get("usage", {})
        tokens = usage.get("completion_tokens")
        return GenerationResult(
            text,
            self.model,
            None,
            float(tokens) / elapsed if tokens and elapsed else None,
            "measured",
        )


class LlamaCppProvider(OpenAICompatibleProvider):
    """llama.cpp server compatibility through its OpenAI-compatible endpoint."""


def benchmark(provider: Provider, prompts: list[str], context_tokens: int = 8192) -> dict[str, Any]:
    results = []
    for prompt in prompts:
        try:
            result = provider.generate(prompt, context_tokens=context_tokens)
            results.append(result.__dict__)
        except ModelRequired as exc:
            results.append({"status": "MODEL_REQUIRED", "error": str(exc)})
    return {
        "context_tokens": context_tokens,
        "concurrency": 1,
        "results": results,
        "model_required": any(r.get("status") == "MODEL_REQUIRED" for r in results),
    }
