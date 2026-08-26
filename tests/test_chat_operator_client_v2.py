from __future__ import annotations

import json
import os

import pytest

from research_workspace.chat_operator_client import (
    AGENT_CANCEL_PATH,
    AGENT_ASYNC_MESSAGES_PATH,
    AGENT_CREATE_PATH,
    AGENT_EVENTS_PATH,
    AGENT_MESSAGES_PATH,
    AGENT_STATUS_PATH,
    CHAT_PATH,
    CAPABILITIES_PATH,
    OPENAPI_PATH,
    SESSION_PATH,
    AGENT_SESSIONS_PATH,
    OperatorClient,
    OperatorClientError,
)


class FakeResponse:
    def __init__(self, payload):
        self._raw = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, _limit=-1):
        return self._raw


class SseResponse(FakeResponse):
    def __init__(self, payload: str):
        self._raw = payload.encode("utf-8")


class FakeOpener:
    def __init__(self, openapi, replies=None):
        self.openapi = openapi
        self.replies = list(replies or [])
        self.requests = []

    def open(self, request, timeout):
        self.requests.append(request)
        if request.full_url.endswith(OPENAPI_PATH):
            return FakeResponse(self.openapi)
        if request.full_url.endswith(SESSION_PATH):
            return FakeResponse({"csrf_token": "csrf-test-token"})
        if not self.replies:
            raise AssertionError(f"unexpected request {request.method} {request.full_url}")
        return FakeResponse(self.replies.pop(0))


def schema(required=(), properties=()):
    return {
        "type": "object",
        "required": list(required),
        "properties": {name: {"type": "string"} for name in properties},
    }


def openapi():
    chat_schema = {
        "type": "object",
        "required": ["lane", "messages"],
        "properties": {
            "lane": {"type": "string"},
            "messages": {"type": "array"},
            "domain": {"type": "string"},
            "session_id": {"type": "string"},
        },
    }
    create_schema = {
        "type": "object",
        "required": ["repo_id", "session_id"],
        "properties": {
            "repo_id": {"type": "string"},
            "session_id": {"type": "string"},
            "task_title": {"type": "string"},
            "idempotency_key": {"type": "string"},
        },
    }
    run_schema = {
        "type": "object",
        "required": ["lane", "instruction", "domain"],
        "properties": {
            "lane": {"type": "string"},
            "instruction": {"type": "string"},
            "domain": {"type": "string"},
            "retrieval_selection": {"type": "string", "default": "none"},
            "personal_corpus_id": {"type": ["string", "null"]},
        },
    }
    return {
        "openapi": "3.1.0",
        "info": {"title": "Laplace Operator", "version": "2"},
        "components": {
            "securitySchemes": {
                "Bearer": {"type": "http", "scheme": "bearer"}
            }
        },
        "paths": {
            SESSION_PATH: {"post": {}},
            CAPABILITIES_PATH: {"get": {}},
            AGENT_SESSIONS_PATH: {"get": {}},
            CHAT_PATH: {
                "post": {
                    "requestBody": {
                        "content": {"application/json": {"schema": chat_schema}}
                    }
                }
            },
            AGENT_CREATE_PATH: {
                "post": {
                    "requestBody": {
                        "content": {"application/json": {"schema": create_schema}}
                    }
                }
            },
            AGENT_MESSAGES_PATH: {
                "get": {},
                "post": {
                    "requestBody": {
                        "content": {"application/json": {"schema": run_schema}}
                    }
                }
            },
            AGENT_ASYNC_MESSAGES_PATH: {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    **run_schema,
                                    "required": [*run_schema["required"], "turn_id"],
                                    "properties": {
                                        **run_schema["properties"],
                                        "turn_id": {"type": "string"},
                                    },
                                }
                            }
                        }
                    }
                }
            },
            AGENT_EVENTS_PATH: {"get": {}},
            AGENT_STATUS_PATH: {"get": {}},
            AGENT_CANCEL_PATH: {"post": {}},
        },
    }


def test_contract_uses_existing_messages_route_and_bearer():
    opener = FakeOpener(openapi())
    client = OperatorClient(token="secret", opener=opener)
    contract = client.contract_check()
    assert contract.agent_turn_path == AGENT_MESSAGES_PATH
    assert contract.auth_kind == "http_bearer"
    assert opener.requests[0].full_url == f"http://127.0.0.1:8765{OPENAPI_PATH}"


def test_chat_payload_is_derived_from_openapi_not_extra_guesses():
    opener = FakeOpener(openapi(), replies=[{"content": "hello"}])
    client = OperatorClient(token="secret", opener=opener)
    result = client.chat(
        lane="quality",
        messages=[{"role": "user", "content": "hi"}],
        domain="general",
        session_id="chat-a",
    )
    assert result["content"] == "hello"
    session_request = opener.requests[-2]
    request = opener.requests[-1]
    assert session_request.full_url == f"http://127.0.0.1:8765{SESSION_PATH}"
    assert session_request.method == "POST"
    assert session_request.get_header("Authorization") == "Bearer secret"
    body = json.loads(request.data.decode())
    assert set(body) == {"lane", "messages", "domain", "session_id"}
    assert request.get_header("Authorization") == "Bearer secret"
    assert request.get_header("X-csrf-token") == "csrf-test-token"


def test_unknown_required_field_fails_closed():
    spec = openapi()
    spec["paths"][CHAT_PATH]["post"]["requestBody"]["content"]["application/json"]["schema"]["required"].append("mystery")
    spec["paths"][CHAT_PATH]["post"]["requestBody"]["content"]["application/json"]["schema"]["properties"]["mystery"] = {"type": "string"}
    client = OperatorClient(token="secret", opener=FakeOpener(spec))
    with pytest.raises(OperatorClientError, match="mystery"):
        client.chat(
            lane="quality",
            messages=[{"role": "user", "content": "hi"}],
            domain="general",
            session_id="chat-a",
        )


def test_non_loopback_refused():
    with pytest.raises(OperatorClientError, match="loopback"):
        OperatorClient(base_url="http://example.com:8765")


def test_agent_create_and_turn_use_openapi_contract():
    opener = FakeOpener(
        openapi(),
        replies=[
            {"status": "BOUND", "binding": {"session_id": "remote-a"}},
            {"status": "SUCCEEDED", "summary": "done"},
        ],
    )
    client = OperatorClient(token="secret", opener=opener)
    created = client.create_agent_session(
        repo_id="repo-a",
        requested_session_id="remote-a",
        task_title="Fix it",
    )
    assert created.session_id == "remote-a"
    result = client.run_agent_turn(
        session_id="remote-a",
        instruction="fix it",
        lane="quality",
        domain="software_engineering",
    )
    assert result["status"] == "SUCCEEDED"
    body = json.loads(opener.requests[-1].data.decode())
    assert body == {
        "lane": "quality",
        "instruction": "fix it",
        "domain": "software_engineering",
        "retrieval_selection": "none",
    }
    mutation_requests = [
        request
        for request in opener.requests
        if request.method == "POST" and not request.full_url.endswith(SESSION_PATH)
    ]
    assert len(mutation_requests) == 2
    assert all(r.get_header("X-csrf-token") == "csrf-test-token" for r in mutation_requests)


def test_async_agent_submission_uses_the_discovered_schema() -> None:
    opener = FakeOpener(
        openapi(),
        replies=[
            {
                "status": "SUBMITTED",
                "session_id": "remote-a",
                "turn_id": "turn-00000001",
                "event_cursor": 7,
            }
        ],
    )
    client = OperatorClient(token="secret", opener=opener)
    submitted = client.submit_agent_turn(
        session_id="remote-a",
        turn_id="turn-00000001",
        instruction="inspect it",
        lane="quality",
        domain="software_engineering",
    )
    assert submitted is not None
    assert submitted.event_cursor == 7
    request = opener.requests[-1]
    assert request.full_url.endswith("/remote-a/messages/async")
    assert json.loads(request.data.decode()) == {
        "turn_id": "turn-00000001",
        "instruction": "inspect it",
        "lane": "quality",
        "domain": "software_engineering",
        "retrieval_selection": "none",
    }


def test_agent_events_are_a_read_only_cursor_page() -> None:
    class EventOpener(FakeOpener):
        def open(self, request, timeout):
            if "/events?" in request.full_url:
                self.requests.append(request)
                return SseResponse(
                    "event: agent_event\n"
                    "id: 8\n"
                    'data: {"details":{"turn_id":"turn-00000001"},"event":"TURN_STARTED","sequence":8}\n\n'
                )
            return super().open(request, timeout)

    opener = EventOpener(openapi())
    client = OperatorClient(token="secret", opener=opener)
    events = client.agent_events("remote-a", after_sequence=7)
    assert events == [
        {
            "details": {"turn_id": "turn-00000001"},
            "event": "TURN_STARTED",
            "sequence": 8,
        }
    ]
    request = opener.requests[-1]
    assert request.method == "GET"
    assert request.get_header("X-csrf-token") is None
    assert request.get_header("Last-event-id") == "7"


def test_missing_csrf_token_fails_closed_before_mutation():
    class MissingCsrfOpener(FakeOpener):
        def open(self, request, timeout):
            self.requests.append(request)
            if request.full_url.endswith(OPENAPI_PATH):
                return FakeResponse(self.openapi)
            if request.full_url.endswith(SESSION_PATH):
                return FakeResponse({})
            raise AssertionError("mutation must not be sent without CSRF")

    opener = MissingCsrfOpener(openapi())
    client = OperatorClient(token="secret", opener=opener)
    with pytest.raises(OperatorClientError, match="missing_valid_csrf"):
        client.chat(
            lane="quality",
            messages=[{"role": "user", "content": "hi"}],
            domain="general",
            session_id="chat-a",
        )
    assert not any(
        request.method == "POST" and not request.full_url.endswith(SESSION_PATH)
        for request in opener.requests
    )


def test_status_get_does_not_fetch_or_send_csrf():
    opener = FakeOpener(openapi(), replies=[{"session_id": "remote-a", "status": "RUNNING"}])
    client = OperatorClient(token="secret", opener=opener)
    result = client.agent_status("remote-a")
    assert result["status"] == "RUNNING"
    assert len(opener.requests) == 2
    assert opener.requests[1].get_header("X-csrf-token") is None


def test_token_file_must_be_private_regular_and_not_symlink(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("LAPLACE_CHAT_TOKEN", raising=False)
    monkeypatch.delenv("LAPLACE_ZETSU_TOKEN", raising=False)
    token = tmp_path / "token"
    token.write_text("private-token", encoding="utf-8")
    os.chmod(token, 0o644)
    monkeypatch.setenv("LAPLACE_CHAT_TOKEN_FILE", str(token))
    with pytest.raises(OperatorClientError, match="permissions_too_open"):
        OperatorClient(opener=FakeOpener(openapi()))
    os.chmod(token, 0o600)
    link = tmp_path / "token-link"
    link.symlink_to(token)
    monkeypatch.setenv("LAPLACE_CHAT_TOKEN_FILE", str(link))
    with pytest.raises(OperatorClientError, match="symlink_refused"):
        OperatorClient(opener=FakeOpener(openapi()))


def test_header_control_characters_are_rejected():
    with pytest.raises(OperatorClientError, match="chat_token_invalid"):
        OperatorClient(token="secret\nredirected")
