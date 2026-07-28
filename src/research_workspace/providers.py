"""Unified local provider adapters with deterministic fixture implementations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Literal, Protocol
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .contracts import (
    EmbeddingRequestV1,
    EmbeddingResponseV1,
    ModelRequestV1,
    ModelResponseV1,
    ProviderCapabilitiesV1,
    ProviderV1,
)

_MAX_PROVIDER_BODY = 8 * 1024 * 1024


class ProviderTransport(Protocol):
    def __call__(
        self,
        endpoint: str,
        path: str,
        *,
        method: Literal["GET", "POST"],
        payload: Mapping[str, object] | None,
        timeout_seconds: float,
    ) -> object: ...


class ProviderAdapterError(RuntimeError):
    """Sanitized provider failure safe for ordinary response details."""

    def __init__(self, category: str, public_reason: str) -> None:
        super().__init__(public_reason)
        self.category = category
        self.public_reason = public_reason


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> urllib.request.Request | None:
        del req, fp, code, msg, headers, newurl
        return None


class _ProviderPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)


class _OllamaModel(_ProviderPayload):
    name: str


class _OllamaTags(_ProviderPayload):
    models: list[_OllamaModel]


class _OllamaMessage(_ProviderPayload):
    content: str


class _OllamaGeneration(_ProviderPayload):
    model: str
    message: _OllamaMessage
    done_reason: str | None = None
    prompt_eval_count: int | None = Field(default=None, ge=0)
    eval_count: int | None = Field(default=None, ge=0)


class _OllamaEmbedding(_ProviderPayload):
    embeddings: list[list[float]]


class _VllmModel(_ProviderPayload):
    id: str


class _VllmModels(_ProviderPayload):
    data: list[_VllmModel]


class _VllmMessage(_ProviderPayload):
    content: str


class _VllmChoice(_ProviderPayload):
    message: _VllmMessage
    finish_reason: str | None = None


class _VllmUsage(_ProviderPayload):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)


class _VllmGeneration(_ProviderPayload):
    model: str
    choices: list[_VllmChoice] = Field(min_length=1)
    usage: _VllmUsage | None = None


class _VllmEmbeddingDatum(_ProviderPayload):
    embedding: list[float]


class _VllmEmbedding(_ProviderPayload):
    data: list[_VllmEmbeddingDatum]
    model: str


def _validate_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("provider endpoint must be a credential-free loopback HTTP origin")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("provider endpoint port is invalid") from exc
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("provider endpoint port is invalid")
    return endpoint.rstrip("/")


def _request_json(
    endpoint: str,
    path: str,
    *,
    method: Literal["GET", "POST"],
    payload: Mapping[str, object] | None,
    timeout_seconds: float,
) -> object:
    if not 0.1 <= timeout_seconds <= 30:
        raise ValueError("provider timeout must be 0.1..30 seconds")
    data = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if payload is not None
        else None
    )
    request = urllib.request.Request(
        endpoint + path,
        data=data,
        method=method,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "laplace-local-provider/1",
        },
    )
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        # The origin was restricted to credential-free loopback HTTP above.
        with opener.open(request, timeout=timeout_seconds) as response:  # nosec B310
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > _MAX_PROVIDER_BODY:
                raise ProviderAdapterError("response_too_large", "provider_response_too_large")
            body = response.read(_MAX_PROVIDER_BODY + 1)
    except ProviderAdapterError:
        raise
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, socket.timeout) as exc:
        category = "provider_timeout" if isinstance(exc, (TimeoutError, socket.timeout)) else (
            "provider_unavailable"
        )
        raise ProviderAdapterError(category, category) from exc
    if len(body) > _MAX_PROVIDER_BODY:
        raise ProviderAdapterError("response_too_large", "provider_response_too_large")
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise ProviderAdapterError("malformed_response", "provider_malformed_response") from exc


class FixtureModelProvider:
    """A deterministic provider whose outputs are explicitly marked as fixtures."""

    def __init__(
        self,
        responses: Mapping[str, str] | None = None,
        *,
        provider_id: str = "fixture",
        model_id: str = "fixture-model-v1",
    ) -> None:
        self._responses = dict(responses or {})
        self._model_id = model_id
        self._descriptor = ProviderV1(
            provider_id=provider_id,
            display_name="Deterministic fixture provider",
            provider_type="fixture",
            endpoint="fixture://in-memory",
            lifecycle="fixture",
            context_limit=8192,
            output_limit=4096,
            capabilities=ProviderCapabilitiesV1(
                streaming=False,
                tools=True,
                structured_output=True,
                embeddings=False,
                thinking_control=True,
                requires_gpu=False,
                supports_cpu=True,
                can_start=False,
                can_stop=False,
            ),
        )

    @property
    def descriptor(self) -> ProviderV1:
        return self._descriptor

    async def health(self) -> Mapping[str, object]:
        return {"status": "READY", "provider_id": self.descriptor.provider_id, "fixture": True}

    async def readiness(self) -> Mapping[str, object]:
        return await self.health()

    async def available_models(self) -> tuple[str, ...]:
        return (self._model_id,)

    async def generate(self, request: ModelRequestV1) -> ModelResponseV1:
        await asyncio.sleep(0)
        prompt = "\n".join(message.content for message in request.messages)
        default = "fixture:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:24]
        text = self._responses.get(request.request_id, default)
        return ModelResponseV1(
            request_id=request.request_id,
            provider_id=self.descriptor.provider_id,
            model_id=self._model_id,
            text=text,
            finish_reason="fixture",
            prompt_tokens=max(1, len(prompt.encode("utf-8")) // 4),
            completion_tokens=max(1, len(text.encode("utf-8")) // 4),
            fixture=True,
        )

    async def start(self) -> None:
        raise ProviderAdapterError("lifecycle_unsupported", "provider_start_not_supported")

    async def stop(self) -> None:
        raise ProviderAdapterError("lifecycle_unsupported", "provider_stop_not_supported")


class FixtureEmbeddingProvider:
    def __init__(
        self,
        *,
        provider_id: str = "fixture-embedding",
        model_id: str = "fixture-embedding-v1",
        dimensions: int = 16,
    ) -> None:
        if not 4 <= dimensions <= 256:
            raise ValueError("fixture dimensions must be 4..256")
        self._model_id = model_id
        self._dimensions = dimensions
        self._descriptor = ProviderV1(
            provider_id=provider_id,
            display_name="Deterministic fixture embeddings",
            provider_type="fixture",
            endpoint="fixture://in-memory",
            lifecycle="fixture",
            context_limit=8192,
            output_limit=1,
            capabilities=ProviderCapabilitiesV1(
                streaming=False,
                tools=False,
                structured_output=False,
                embeddings=True,
                thinking_control=False,
                requires_gpu=False,
                supports_cpu=True,
                can_start=False,
                can_stop=False,
            ),
        )

    @property
    def descriptor(self) -> ProviderV1:
        return self._descriptor

    async def embed(self, request: EmbeddingRequestV1) -> EmbeddingResponseV1:
        await asyncio.sleep(0)
        vectors: list[tuple[float, ...]] = []
        for text in request.texts:
            digest = hashlib.shake_256(text.encode("utf-8")).digest(self._dimensions)
            vectors.append(tuple(round((byte - 127.5) / 127.5, 8) for byte in digest))
        return EmbeddingResponseV1(
            request_id=request.request_id,
            provider_id=self.descriptor.provider_id,
            model_id=self._model_id,
            vectors=tuple(vectors),
            fixture=True,
        )


class OllamaProvider:
    """Ollama invocation adapter; lifecycle and model acquisition stay external."""

    def __init__(
        self,
        *,
        provider_id: str,
        endpoint: str,
        model_id: str,
        lifecycle: Literal["owned", "unowned"] = "unowned",
        timeout_seconds: float = 10,
        context_limit: int = 8192,
        output_limit: int = 4096,
        transport: ProviderTransport | None = None,
    ) -> None:
        self._endpoint = _validate_endpoint(endpoint)
        self._model_id = model_id
        self._timeout = timeout_seconds
        self._transport = transport or _request_json
        self._transport_in_worker = transport is None
        self._descriptor = ProviderV1(
            provider_id=provider_id,
            display_name="Ollama local provider",
            provider_type="ollama",
            endpoint=self._endpoint,
            lifecycle=lifecycle,
            context_limit=context_limit,
            output_limit=output_limit,
            capabilities=ProviderCapabilitiesV1(
                streaming=True,
                tools=False,
                structured_output=True,
                embeddings=True,
                thinking_control=True,
                requires_gpu=False,
                supports_cpu=True,
                can_start=False,
                can_stop=False,
            ),
        )

    @property
    def descriptor(self) -> ProviderV1:
        return self._descriptor

    async def _json(
        self,
        path: str,
        *,
        method: Literal["GET", "POST"] = "GET",
        payload: Mapping[str, object] | None = None,
    ) -> object:
        if self._transport_in_worker:
            return await asyncio.to_thread(
                self._transport,
                self._endpoint,
                path,
                method=method,
                payload=payload,
                timeout_seconds=self._timeout,
            )
        await asyncio.sleep(0)
        return self._transport(
            self._endpoint,
            path,
            method=method,
            payload=payload,
            timeout_seconds=self._timeout,
        )

    async def available_models(self) -> tuple[str, ...]:
        try:
            value = _OllamaTags.model_validate(await self._json("/api/tags"))
        except ValidationError as exc:
            raise ProviderAdapterError(
                "malformed_response", "provider_malformed_response"
            ) from exc
        return tuple(sorted({item.name for item in value.models}))

    async def health(self) -> Mapping[str, object]:
        try:
            models = await self.available_models()
        except ProviderAdapterError as exc:
            return {
                "status": "DEGRADED",
                "provider_id": self.descriptor.provider_id,
                "reason": exc.public_reason,
            }
        return {
            "status": "READY",
            "provider_id": self.descriptor.provider_id,
            "configured_model_available": self._model_id in models,
        }

    async def readiness(self) -> Mapping[str, object]:
        health = dict(await self.health())
        if not health.get("configured_model_available"):
            health["status"] = "DEGRADED"
            health["reason"] = "configured_model_unavailable"
        return health

    async def generate(self, request: ModelRequestV1) -> ModelResponseV1:
        payload: dict[str, object] = {
            "model": self._model_id,
            "stream": False,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
                if message.role in {"system", "user", "assistant"}
            ],
            "options": {"num_predict": request.max_output_tokens},
        }
        if request.structured_schema is not None:
            payload["format"] = request.structured_schema
        if request.thinking != "provider_default":
            payload["think"] = request.thinking == "enabled"
        try:
            value = _OllamaGeneration.model_validate(
                await self._json("/api/chat", method="POST", payload=payload)
            )
        except ValidationError as exc:
            raise ProviderAdapterError(
                "malformed_response", "provider_malformed_response"
            ) from exc
        return ModelResponseV1(
            request_id=request.request_id,
            provider_id=self.descriptor.provider_id,
            model_id=value.model,
            text=value.message.content,
            finish_reason="length" if value.done_reason == "length" else "stop",
            prompt_tokens=value.prompt_eval_count,
            completion_tokens=value.eval_count,
        )

    async def embed(self, request: EmbeddingRequestV1) -> EmbeddingResponseV1:
        try:
            value = _OllamaEmbedding.model_validate(
                await self._json(
                    "/api/embed",
                    method="POST",
                    payload={"model": request.model_id, "input": list(request.texts)},
                )
            )
        except ValidationError as exc:
            raise ProviderAdapterError(
                "malformed_response", "provider_malformed_response"
            ) from exc
        if len(value.embeddings) != len(request.texts):
            raise ProviderAdapterError("malformed_response", "provider_malformed_response")
        return EmbeddingResponseV1(
            request_id=request.request_id,
            provider_id=self.descriptor.provider_id,
            model_id=request.model_id,
            vectors=tuple(tuple(item) for item in value.embeddings),
        )

    async def start(self) -> None:
        raise ProviderAdapterError("lifecycle_unsupported", "provider_start_not_supported")

    async def stop(self) -> None:
        raise ProviderAdapterError("lifecycle_unsupported", "provider_stop_not_supported")


class VLLMProvider:
    """OpenAI-compatible vLLM adapter with no process-lifecycle authority."""

    def __init__(
        self,
        *,
        provider_id: str,
        endpoint: str,
        model_id: str,
        lifecycle: Literal["owned", "unowned"] = "unowned",
        timeout_seconds: float = 10,
        context_limit: int = 8192,
        output_limit: int = 4096,
        embedding_support: bool = False,
        transport: ProviderTransport | None = None,
    ) -> None:
        self._endpoint = _validate_endpoint(endpoint)
        self._model_id = model_id
        self._timeout = timeout_seconds
        self._transport = transport or _request_json
        self._transport_in_worker = transport is None
        self._descriptor = ProviderV1(
            provider_id=provider_id,
            display_name="vLLM local provider",
            provider_type="vllm",
            endpoint=self._endpoint,
            lifecycle=lifecycle,
            context_limit=context_limit,
            output_limit=output_limit,
            capabilities=ProviderCapabilitiesV1(
                streaming=True,
                tools=True,
                structured_output=True,
                embeddings=embedding_support,
                thinking_control=True,
                requires_gpu=True,
                supports_cpu=False,
                can_start=False,
                can_stop=False,
            ),
        )

    @property
    def descriptor(self) -> ProviderV1:
        return self._descriptor

    async def _json(
        self,
        path: str,
        *,
        method: Literal["GET", "POST"] = "GET",
        payload: Mapping[str, object] | None = None,
    ) -> object:
        if self._transport_in_worker:
            return await asyncio.to_thread(
                self._transport,
                self._endpoint,
                path,
                method=method,
                payload=payload,
                timeout_seconds=self._timeout,
            )
        await asyncio.sleep(0)
        return self._transport(
            self._endpoint,
            path,
            method=method,
            payload=payload,
            timeout_seconds=self._timeout,
        )

    async def available_models(self) -> tuple[str, ...]:
        try:
            value = _VllmModels.model_validate(await self._json("/v1/models"))
        except ValidationError as exc:
            raise ProviderAdapterError(
                "malformed_response", "provider_malformed_response"
            ) from exc
        return tuple(sorted({item.id for item in value.data}))

    async def health(self) -> Mapping[str, object]:
        try:
            models = await self.available_models()
        except ProviderAdapterError as exc:
            return {
                "status": "DEGRADED",
                "provider_id": self.descriptor.provider_id,
                "reason": exc.public_reason,
            }
        return {
            "status": "READY",
            "provider_id": self.descriptor.provider_id,
            "configured_model_available": self._model_id in models,
        }

    async def readiness(self) -> Mapping[str, object]:
        health = dict(await self.health())
        if not health.get("configured_model_available"):
            health["status"] = "DEGRADED"
            health["reason"] = "configured_model_unavailable"
        return health

    async def generate(self, request: ModelRequestV1) -> ModelResponseV1:
        payload: dict[str, object] = {
            "model": self._model_id,
            "stream": False,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
                if message.role in {"system", "user", "assistant"}
            ],
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
        }
        if request.structured_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "laplace_response", "schema": request.structured_schema},
            }
        if request.thinking != "provider_default":
            payload["chat_template_kwargs"] = {
                "enable_thinking": request.thinking == "enabled"
            }
        try:
            value = _VllmGeneration.model_validate(
                await self._json("/v1/chat/completions", method="POST", payload=payload)
            )
        except ValidationError as exc:
            raise ProviderAdapterError(
                "malformed_response", "provider_malformed_response"
            ) from exc
        choice = value.choices[0]
        usage = value.usage
        return ModelResponseV1(
            request_id=request.request_id,
            provider_id=self.descriptor.provider_id,
            model_id=value.model,
            text=choice.message.content,
            finish_reason="length" if choice.finish_reason == "length" else "stop",
            prompt_tokens=usage.prompt_tokens if usage is not None else None,
            completion_tokens=usage.completion_tokens if usage is not None else None,
        )

    async def embed(self, request: EmbeddingRequestV1) -> EmbeddingResponseV1:
        if not self.descriptor.capabilities.embeddings:
            raise ProviderAdapterError(
                "capability_unavailable", "provider_embedding_not_supported"
            )
        try:
            value = _VllmEmbedding.model_validate(
                await self._json(
                    "/v1/embeddings",
                    method="POST",
                    payload={"model": request.model_id, "input": list(request.texts)},
                )
            )
        except ValidationError as exc:
            raise ProviderAdapterError(
                "malformed_response", "provider_malformed_response"
            ) from exc
        if len(value.data) != len(request.texts):
            raise ProviderAdapterError("malformed_response", "provider_malformed_response")
        return EmbeddingResponseV1(
            request_id=request.request_id,
            provider_id=self.descriptor.provider_id,
            model_id=value.model,
            vectors=tuple(tuple(item.embedding) for item in value.data),
        )

    async def start(self) -> None:
        raise ProviderAdapterError("lifecycle_unsupported", "provider_start_not_supported")

    async def stop(self) -> None:
        raise ProviderAdapterError("lifecycle_unsupported", "provider_stop_not_supported")
