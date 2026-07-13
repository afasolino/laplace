from __future__ import annotations

import json
from pathlib import Path
from threading import Event

import pytest
from fastapi.testclient import TestClient

from research_workspace import laplace_cli
from research_workspace.api import create_app
from research_workspace.chat import (
    ChatEngine,
    ChatProject,
    ChatResponse,
    ConversationStore,
    _prompt,
    normalize_revisions,
    normalize_response,
)
from research_workspace.retrieval import Evidence


def _evidence() -> list[Evidence]:
    return [
        Evidence(
            filename="paper.pdf",
            page=2,
            section=None,
            chunk_id="document:p2:c0",
            text="The accelerator reduces latency to 3 ms.",
            score=0.91,
            document_class="user_work",
            source_path="C:/Library/paper.pdf",
            title="Paper title",
            availability="COMPLETE_LOCAL_PDF",
        )
    ]


def _project(tmp_path: Path) -> tuple[ChatProject, ConversationStore]:
    project = ChatProject(tmp_path, tmp_path / "laplace.db", "test-project", "test", "qwen3:4b")
    return project, ConversationStore(project)


def test_nested_json_and_fenced_json_normalize_to_readable_content(tmp_path: Path) -> None:
    project, _ = _project(tmp_path)
    evidence = _evidence()
    nested = json.dumps(
        {"answer": json.dumps({"content": "The latency is 3 ms. [1]", "citations": [1]})}
    )
    response, _ = normalize_response(
        nested,
        conversation_id="c",
        query="latency",
        evidence=evidence,
        model=project.model,
        collections=["MyWorks"],
        candidate_count=1,
    )
    assert response.status == "GROUNDED"
    assert response.content == "The latency is 3 ms. [1]"
    assert response.citations[0].chunk_id == "document:p2:c0"
    fenced, _ = normalize_response(
        '```json\n{"content":"3 ms [1]","citations":[1]}\n```',
        conversation_id="c",
        query="latency",
        evidence=evidence,
        model=project.model,
        collections=["MyWorks"],
        candidate_count=1,
    )
    assert fenced.content == "3 ms [1]"


def test_invalid_or_empty_citations_use_extractable_fallback(tmp_path: Path) -> None:
    project, _ = _project(tmp_path)
    response, audit = normalize_response(
        '{"content":"Unsupported claim","citations":[99]}',
        conversation_id="c",
        query="q",
        evidence=_evidence(),
        model=project.model,
        collections=["MyWorks"],
        candidate_count=1,
    )
    assert response.status == "GROUNDED_EXTRACTIVE_FALLBACK"
    assert response.fallback_used is True
    assert response.citations and response.citations[0].citation_id == 1
    assert audit["model_citation_valid"] is False


def test_rejected_draft_and_fallback_are_distinct_revisions(tmp_path: Path) -> None:
    project, store = _project(tmp_path)
    conversation = store.create("Revisions")
    evidence = _evidence()
    candidate, fallback, audit = normalize_revisions(
        '{"content":"Model draft with an unsupported claim","citations":[99]}',
        conversation_id=conversation.conversation_id,
        query="latency",
        evidence=evidence,
        model=project.model,
        collections=["MyWorks"],
        candidate_count=1,
    )
    assert fallback is not None
    assert candidate.message_id != fallback.message_id
    assert candidate.state == "CITATION_REJECTED"
    assert candidate.content == "Model draft with an unsupported claim"
    assert fallback.fallback_of_message_id == candidate.message_id
    assert len(fallback.content) <= 1100
    assert fallback.content.startswith("The retrieved evidence indicates:")
    store.append_assistant(candidate, audit)
    store.append_assistant(fallback, audit)
    messages = store.detail(conversation.conversation_id).messages
    assert [item["state"] for item in messages] == ["CITATION_REJECTED", "GROUNDED_FALLBACK"]


def test_compact_evidence_ids_and_marker_recovery(tmp_path: Path) -> None:
    project, _ = _project(tmp_path)
    evidence = _evidence()
    prompt = _prompt("latency", evidence, [])
    assert '"evidence_id": "E1"' in prompt
    response, _ = normalize_response(
        '{"content":"The latency is 3 ms [E1]","citations":[]}',
        conversation_id="c",
        query="latency",
        evidence=evidence,
        model=project.model,
        collections=["MyWorks"],
        candidate_count=1,
    )
    assert response.status == "GROUNDED"
    assert response.citations[0].evidence_id == "E1"


def test_stream_preserves_rejected_draft_then_emits_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, store = _project(tmp_path)
    conversation = store.create("Stream")
    engine = ChatEngine(project, store)
    engine._retrieve = lambda query, selected: _evidence()  # type: ignore[method-assign]

    def fake_generation(*args: object, **kwargs: object):
        yield {"response": '{"content":"Draft","citations":[99]}'}

    monkeypatch.setattr("research_workspace.chat.generate_stream", fake_generation)
    events = list(engine.stream(conversation.conversation_id, "latency"))
    kinds = [event["type"] for event in events]
    assert (
        "message_rejected" in kinds
        and "fallback_started" in kinds
        and "fallback_completed" in kinds
    )
    assert (
        kinds.index("message_rejected")
        < kinds.index("fallback_started")
        < kinds.index("fallback_completed")
    )
    messages = store.detail(conversation.conversation_id).messages
    assert len(messages) == 3
    assert messages[-2]["state"] == "CITATION_REJECTED"
    assert messages[-1]["state"] == "GROUNDED_FALLBACK"
    assert messages[-1]["fallback_of_message_id"] == messages[-2]["message_id"]


def test_conversation_store_persists_and_isolates_project(tmp_path: Path) -> None:
    project, store = _project(tmp_path)
    conversation = store.create("Persistent", ["MyWorks"])
    store.append_user(conversation.conversation_id, "Question")
    assert store.detail(conversation.conversation_id).messages[0]["content"] == "Question"
    other = ConversationStore(ChatProject(tmp_path, tmp_path / "other.db", "other", "other"))
    with pytest.raises(KeyError):
        other.detail(conversation.conversation_id)
    store.rename(conversation.conversation_id, "Renamed")
    assert store.summary(conversation.conversation_id).title == "Renamed"
    store.archive(conversation.conversation_id)
    assert store.list() == []


def test_stop_only_marks_active_generation(tmp_path: Path) -> None:
    project, store = _project(tmp_path)
    engine = ChatEngine(project, store)
    event = Event()
    engine._active["conversation"] = event
    assert engine.stop("conversation") is True
    assert event.is_set()
    assert engine.stop("missing") is False


def test_chat_api_contract_stream_and_attachment_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = TestClient(create_app(tmp_path, tmp_path / "db.sqlite"))
    created = client.post("/api/chat/conversations", json={"title": "API chat"})
    assert created.status_code == 200
    conversation_id = created.json()["conversation_id"]

    response = ChatEngine

    def fake_answer(
        self: ChatEngine, conversation_id: str, query: str, **kwargs: object
    ) -> ChatResponse:
        return ChatResponse(
            message_id="m",
            conversation_id=conversation_id,
            content="Readable answer [1]",
            status="GROUNDED",
            created_at="now",
            model="qwen3:4b",
            retrieval={
                "query": query,
                "collections": ["MyWorks"],
                "candidate_count": 1,
                "evidence_count": 1,
                "mode": "hybrid",
            },
        )

    monkeypatch.setattr(response, "answer", fake_answer)
    reply = client.post(
        f"/api/chat/conversations/{conversation_id}/messages", json={"content": "Question"}
    )
    assert reply.status_code == 200
    assert reply.json()["content"] == "Readable answer [1]"
    assert client.get("/chat").status_code == 200
    page = client.get("/chat").text
    assert "Evidence" in page
    assert "message_rejected" in page and "fallback_started" in page and "activeGenerations" in page
    assert client.get("/library").status_code == 200
    assert client.get("/research").status_code == 200
    assert client.get("/downloads").status_code == 200
    assert client.get("/settings").status_code == 200
    assert client.get("/api/project/settings").json()["secrets_included"] is False
    changed = client.patch(
        "/api/project/settings/retrieval.yaml", json={"content": "mode: semantic\n"}
    )
    assert changed.status_code == 200
    assert (
        client.patch(
            "/api/project/settings/retrieval.yaml", json={"content": "not: [valid"}
        ).status_code
        == 400
    )
    assert (
        client.post(
            f"/api/chat/conversations/{conversation_id}/attachments",
            files={"file": ("../bad.exe", b"x")},
        ).status_code
        == 415
    )
    assert client.delete(f"/api/chat/conversations/{conversation_id}").status_code == 400

    def fake_stream(self: ChatEngine, conversation_id: str, query: str, **kwargs: object):
        yield {"type": "message_started", "conversation_id": conversation_id}
        yield {"type": "token", "text": "hello"}
        yield {
            "type": "message_completed",
            "message": {"content": "hello", "status": "GROUNDED", "citations": []},
        }

    monkeypatch.setattr(response, "stream", fake_stream)
    with client.stream(
        "POST",
        f"/api/chat/conversations/{conversation_id}/messages?stream=true",
        json={"content": "Stream"},
    ) as stream:
        text = stream.read().decode()
    assert "message_started" in text and "token" in text and "message_completed" in text


def test_laplace_ask_terminal_output_and_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(laplace_cli, "APP_HOME", tmp_path / "home")
    monkeypatch.setattr(laplace_cli, "REGISTRY_PATH", tmp_path / "home" / "projects.json")
    monkeypatch.setattr(laplace_cli, "CONFIG_PATH", tmp_path / "home" / "config.yaml")
    root = Path(laplace_cli.init_laplace("ChatProject", cwd=tmp_path)["project"])
    result = {
        "response": {
            "content": "Readable answer",
            "status": "GROUNDED",
            "model": "qwen3:4b",
            "citations": [{"citation_id": 1, "title": "Paper", "filename": "paper.pdf", "page": 1}],
        },
        "answer_path": str(root / "Outputs" / "Conversations" / "response.json"),
    }
    monkeypatch.setattr(laplace_cli, "_ask", lambda paths, query: result)
    assert laplace_cli.main(["--project", str(root), "--ask", "question"]) == 0
    assert "Readable answer" in capsys.readouterr().out
    assert laplace_cli.main(["--project", str(root), "--ask", "question", "--json"]) == 0
    assert '"response"' in capsys.readouterr().out
