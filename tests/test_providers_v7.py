from __future__ import annotations

import asyncio

import pytest

from research_workspace.contracts import EmbeddingRequestV1, MessageV1, ModelRequestV1
from research_workspace.providers import (
    FixtureEmbeddingProvider,
    OllamaProvider,
    ProviderAdapterError,
    VLLMProvider,
)


def _request() -> ModelRequestV1:
    return ModelRequestV1(
        request_id="request-1",
        route_id="route-1",
        messages=(
            MessageV1(
                message_id="message-1",
                conversation_id="conversation-1",
                role="user",
                content="fixture only",
                created_at_utc="2026-07-28T00:00:00+00:00",
            ),
        ),
        max_output_tokens=128,
        temperature=0.0,
    )


def test_fixture_embeddings_are_repeatable() -> None:
    provider = FixtureEmbeddingProvider(dimensions=8)
    request = EmbeddingRequestV1(
        request_id="embedding-1",
        model_id="fixture-embedding-v1",
        texts=("alpha", "beta"),
    )
    first = asyncio.run(provider.embed(request))
    second = asyncio.run(provider.embed(request))
    assert first == second
    assert first.fixture is True
    assert len(first.vectors) == 2
    assert len(first.vectors[0]) == 8


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1:11434",
        "http://example.com:11434",
        "http://user:secret@127.0.0.1:11434",
        "http://127.0.0.1:11434/path",
        "http://127.0.0.1:11434?token=secret",
    ],
)
def test_live_provider_endpoints_fail_closed(endpoint: str) -> None:
    with pytest.raises(ValueError, match="credential-free loopback"):
        OllamaProvider(
            provider_id="ollama",
            endpoint=endpoint,
            model_id="configured-model",
        )


def test_ollama_adapter_strictly_parses_fixture_payload() -> None:
    calls: list[str] = []

    def fixture_request(
        endpoint: str,
        path: str,
        *,
        method: str,
        payload: object,
        timeout_seconds: float,
    ) -> object:
        del endpoint, method, payload, timeout_seconds
        calls.append(path)
        if path == "/api/tags":
            return {"models": [{"name": "configured-model"}]}
        return {
            "model": "configured-model",
            "message": {"content": "fixture response"},
            "done_reason": "stop",
            "prompt_eval_count": 3,
            "eval_count": 2,
        }

    provider = OllamaProvider(
        provider_id="ollama",
        endpoint="http://127.0.0.1:11434",
        model_id="configured-model",
        transport=fixture_request,  # type: ignore[arg-type]
    )
    assert asyncio.run(provider.readiness())["status"] == "READY"
    response = asyncio.run(provider.generate(_request()))
    assert response.text == "fixture response"
    assert response.fixture is False
    assert calls == ["/api/tags", "/api/chat"]
    with pytest.raises(ProviderAdapterError, match="provider_stop_not_supported"):
        asyncio.run(provider.stop())


def test_vllm_adapter_degrades_on_malformed_fixture_payload() -> None:
    def malformed(
        endpoint: str,
        path: str,
        *,
        method: str,
        payload: object,
        timeout_seconds: float,
    ) -> object:
        del endpoint, path, method, payload, timeout_seconds
        return {"data": "not-a-list"}

    provider = VLLMProvider(
        provider_id="vllm",
        endpoint="http://127.0.0.1:8201",
        model_id="configured-model",
        transport=malformed,  # type: ignore[arg-type]
    )
    result = asyncio.run(provider.health())
    assert result == {
        "status": "DEGRADED",
        "provider_id": "vllm",
        "reason": "provider_malformed_response",
    }


def test_vllm_descriptor_requires_gpu_but_adapter_never_controls_it() -> None:
    provider = VLLMProvider(
        provider_id="vllm",
        endpoint="http://localhost:8201",
        model_id="configured-model",
    )
    assert provider.descriptor.capabilities.requires_gpu is True
    assert provider.descriptor.capabilities.can_start is False
    assert provider.descriptor.capabilities.can_stop is False
    with pytest.raises(ProviderAdapterError, match="provider_start_not_supported"):
        asyncio.run(provider.start())
