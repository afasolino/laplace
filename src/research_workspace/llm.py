from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse


class ModelRequired(RuntimeError):
    """No explicitly installed local model is available."""


class ModelInvocationError(ModelRequired):
    """A local model request failed with a machine-readable category."""

    def __init__(
        self,
        message: str,
        *,
        category: str,
        http_status: int | None = None,
        response_body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.http_status = http_status
        self.response_body = response_body


def _endpoint_failure_category(exc: BaseException) -> str:
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, TimeoutError):
        return "timeout"
    return "endpoint_unavailable"


@dataclass
class GenerationResult:
    text: str
    model: str
    ttft_seconds: float | None
    output_tokens_per_second: float | None
    status: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    input_tokens_per_second: float | None = None


class LocalGenerationBackend(Protocol):
    """Narrow local generation boundary shared by all Laplace backends."""

    def generate(
        self,
        prompt: str,
        *,
        context_tokens: int = 8192,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        """Generate one bounded response through a local endpoint."""

    def token_count(self, prompt: str) -> int:
        """Return the exact serialized prompt count when the backend supports it."""

    def health(self) -> dict[str, str]:
        """Return a local health record without exposing credentials."""

    def model_identity(self) -> dict[str, str]:
        """Return backend and configured model identity."""


class Provider:
    def generate(
        self,
        prompt: str,
        *,
        context_tokens: int = 8192,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        raise NotImplementedError

    def token_count(self, prompt: str) -> int:
        raise ModelRequired("The configured local backend has no tokenizer endpoint")

    def health(self) -> dict[str, str]:
        return {"status": "UNKNOWN", "backend": self.__class__.__name__}

    def model_identity(self) -> dict[str, str]:
        return {"backend": self.__class__.__name__, "model": "unknown"}


class MockProvider(Provider):
    def generate(
        self,
        prompt: str,
        *,
        context_tokens: int = 8192,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        del max_tokens
        return GenerationResult("[MOCK] " + prompt[:200], "mock", 0.0, None, "mock")

    def token_count(self, prompt: str) -> int:
        return max(1, (len(prompt.encode("utf-8")) + 2) // 3)

    def health(self) -> dict[str, str]:
        return {"status": "MOCK", "backend": "mock"}

    def model_identity(self) -> dict[str, str]:
        return {"backend": "mock", "model": "mock"}


class OllamaProvider(Provider):
    def __init__(self, endpoint: str, model: str):
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ModelRequired("Ollama endpoints must be loopback-only")
        self.endpoint, self.model = endpoint.rstrip("/"), model

    def generate(
        self,
        prompt: str,
        *,
        context_tokens: int = 8192,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        effective_max_tokens = 512 if max_tokens is None else max_tokens
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_ctx": context_tokens, "num_predict": effective_max_tokens},
            }
        ).encode()
        request = urllib.request.Request(
            self.endpoint + "/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:  # nosec B310
                value = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            body = exc.read(4000).decode("utf-8", errors="replace")
            raise ModelInvocationError(
                f"Local Ollama endpoint returned HTTP {exc.code}: {body}",
                category="http_error",
                http_status=exc.code,
                response_body=body,
            ) from exc
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            raise ModelInvocationError(
                f"Local Ollama endpoint unavailable: {exc}",
                category=_endpoint_failure_category(exc),
            ) from exc
        text = str(value.get("response", ""))
        prompt_tokens = value.get("prompt_eval_count")
        tokens = value.get("eval_count")
        prompt_duration = value.get("prompt_eval_duration")
        eval_duration = value.get("eval_duration")
        ttft = float(prompt_duration) / 1e9 if isinstance(prompt_duration, (int, float)) else None
        output_rate = (
            float(tokens) / (float(eval_duration) / 1e9)
            if isinstance(tokens, int)
            and isinstance(eval_duration, (int, float))
            and eval_duration > 0
            else None
        )
        return GenerationResult(
            text,
            self.model,
            ttft,
            output_rate,
            "measured",
            prompt_tokens if isinstance(prompt_tokens, int) else None,
            tokens if isinstance(tokens, int) else None,
            float(prompt_tokens) / ttft
            if isinstance(prompt_tokens, int) and ttft is not None and ttft > 0
            else None,
        )

    def health(self) -> dict[str, str]:
        try:
            with urllib.request.urlopen(  # nosec B310
                self.endpoint + "/api/tags", timeout=10
            ) as response:
                json.loads(response.read())
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            return {"status": "UNAVAILABLE", "backend": "ollama", "error": str(exc)}
        return {"status": "AVAILABLE", "backend": "ollama", "endpoint": self.endpoint}

    def model_identity(self) -> dict[str, str]:
        return {"backend": "ollama", "model": self.model, "endpoint": self.endpoint}


class OpenAICompatibleProvider(Provider):
    """Client for a localhost OpenAI-compatible server (LM Studio/llama.cpp)."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
        timeout_seconds: int = 120,
    ):
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ModelRequired("OpenAI-compatible endpoints must be loopback-only")
        if max_tokens < 1 or max_tokens > 8192:
            raise ModelRequired("OpenAI-compatible max_tokens must be between 1 and 8192")
        if temperature < 0.0 or temperature > 2.0:
            raise ModelRequired("OpenAI-compatible temperature must be between 0 and 2")
        if top_p <= 0.0 or top_p > 1.0:
            raise ModelRequired("OpenAI-compatible top_p must be greater than 0 and at most 1")
        if timeout_seconds < 1 or timeout_seconds > 1800:
            raise ModelRequired("OpenAI-compatible timeout must be between 1 and 1800 seconds")
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.timeout_seconds = timeout_seconds

    def _request_payload(
        self, prompt: str, *, stream: bool, max_tokens: int | None = None
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "stream": stream,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def generate(
        self,
        prompt: str,
        *,
        context_tokens: int = 8192,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        del context_tokens
        payload = json.dumps(
            self._request_payload(prompt, stream=False, max_tokens=max_tokens)
        ).encode()
        request = urllib.request.Request(
            self.endpoint + "/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(  # nosec B310
                request, timeout=self.timeout_seconds
            ) as response:
                value = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            body = exc.read(4000).decode("utf-8", errors="replace")
            raise ModelInvocationError(
                f"Local OpenAI-compatible endpoint returned HTTP {exc.code}: {body}",
                category="http_error",
                http_status=exc.code,
                response_body=body,
            ) from exc
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            raise ModelInvocationError(
                f"Local OpenAI-compatible endpoint unavailable: {exc}",
                category=_endpoint_failure_category(exc),
            ) from exc
        elapsed = time.perf_counter() - started
        choices = value.get("choices", [])
        text = str(choices[0].get("message", {}).get("content", "")) if choices else ""
        usage = value.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        return GenerationResult(
            text,
            self.model,
            None,
            float(tokens) / elapsed if tokens and elapsed else None,
            "measured",
            prompt_tokens if isinstance(prompt_tokens, int) else None,
            tokens if isinstance(tokens, int) else None,
            None,
        )

    def _generate_streaming(
        self, prompt: str, *, max_tokens: int | None = None
    ) -> GenerationResult:
        """Measure a loopback OpenAI stream using server-reported token counts."""
        payload = json.dumps(
            self._request_payload(prompt, stream=True, max_tokens=max_tokens)
        ).encode()
        request = urllib.request.Request(
            self.endpoint + "/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        started = time.perf_counter()
        first_token_at: float | None = None
        chunks: list[str] = []
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        try:
            with urllib.request.urlopen(  # nosec B310
                request, timeout=self.timeout_seconds
            ) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    event = line[6:]
                    if event == "[DONE]":
                        continue
                    value: object = json.loads(event)
                    if not isinstance(value, dict):
                        continue
                    choices = value.get("choices")
                    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                        delta = choices[0].get("delta")
                        if isinstance(delta, dict):
                            fragment = delta.get("content", delta.get("reasoning_content", ""))
                            if isinstance(fragment, str) and fragment:
                                if first_token_at is None:
                                    first_token_at = time.perf_counter()
                                chunks.append(fragment)
                    usage = value.get("usage")
                    if isinstance(usage, dict):
                        raw_prompt_tokens = usage.get("prompt_tokens")
                        raw_completion_tokens = usage.get("completion_tokens")
                        prompt_tokens = (
                            raw_prompt_tokens
                            if isinstance(raw_prompt_tokens, int)
                            else prompt_tokens
                        )
                        completion_tokens = (
                            raw_completion_tokens
                            if isinstance(raw_completion_tokens, int)
                            else completion_tokens
                        )
        except urllib.error.HTTPError as exc:
            body = exc.read(4000).decode("utf-8", errors="replace")
            raise ModelInvocationError(
                f"Local OpenAI-compatible endpoint returned HTTP {exc.code}: {body}",
                category="http_error",
                http_status=exc.code,
                response_body=body,
            ) from exc
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            raise ModelInvocationError(
                f"Local OpenAI-compatible endpoint unavailable: {exc}",
                category=_endpoint_failure_category(exc),
            ) from exc
        completed = time.perf_counter()
        ttft = first_token_at - started if first_token_at is not None else None
        decode_seconds = completed - first_token_at if first_token_at is not None else None
        return GenerationResult(
            "".join(chunks),
            self.model,
            ttft,
            float(completion_tokens) / decode_seconds
            if completion_tokens is not None and decode_seconds is not None and decode_seconds > 0
            else None,
            "measured",
            prompt_tokens,
            completion_tokens,
            float(prompt_tokens) / ttft
            if prompt_tokens is not None and ttft is not None and ttft > 0
            else None,
        )

    def token_count(self, prompt: str) -> int:
        """Ask a loopback serving engine to count the serialized chat request."""
        payload = json.dumps(
            {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
        ).encode()
        request = urllib.request.Request(
            self.endpoint + "/tokenize",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # nosec B310
                value: object = json.loads(response.read())
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            raise ModelRequired(f"Local tokenizer endpoint unavailable: {exc}") from exc
        count = value.get("count") if isinstance(value, dict) else None
        if not isinstance(count, int) or count < 0:
            raise ModelRequired("Local tokenizer endpoint did not return a valid token count")
        return count

    def health(self) -> dict[str, str]:
        request = urllib.request.Request(self.endpoint + "/v1/models", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:  # nosec B310
                value: object = json.loads(response.read())
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            return {"status": "UNAVAILABLE", "backend": "openai_compatible", "error": str(exc)}
        models: list[str] = []
        if isinstance(value, dict):
            data = value.get("data")
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and isinstance(item.get("id"), str):
                        models.append(item["id"])
        if self.model not in models:
            return {
                "status": "MODEL_MISMATCH",
                "backend": "openai_compatible",
                "endpoint": self.endpoint,
                "configured_model": self.model,
                "served_models": ",".join(models),
            }
        return {
            "status": "AVAILABLE",
            "backend": "openai_compatible",
            "endpoint": self.endpoint,
            "configured_model": self.model,
            "served_models": ",".join(models),
        }

    def model_identity(self) -> dict[str, str]:
        return {"backend": "openai_compatible", "model": self.model, "endpoint": self.endpoint}


class LlamaCppProvider(OpenAICompatibleProvider):
    """llama.cpp server compatibility through its OpenAI-compatible endpoint."""


class VllmProvider(OpenAICompatibleProvider):
    """vLLM's local OpenAI-compatible serving surface."""

    def generate(
        self,
        prompt: str,
        *,
        context_tokens: int = 8192,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        del context_tokens
        return self._generate_streaming(prompt, max_tokens=max_tokens)

    def model_identity(self) -> dict[str, str]:
        return {"backend": "vllm", "model": self.model, "endpoint": self.endpoint}


class SglangProvider(OpenAICompatibleProvider):
    """SGLang's local OpenAI-compatible serving surface."""

    def model_identity(self) -> dict[str, str]:
        return {"backend": "sglang", "model": self.model, "endpoint": self.endpoint}


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
