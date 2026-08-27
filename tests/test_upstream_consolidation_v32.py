import json
import subprocess
from pathlib import Path

import pytest

import research_workspace.zetsu_codex as codex_module
from research_workspace.chat_operator_client import AgentTurnResult
from research_workspace.laplace_web import LaplaceWebError, WebController
from research_workspace.zetsu_codex import status


class WebClient:
    def __init__(self) -> None:
        self.last_agent_kwargs = None
        self.fail_chat = False

    def contract_check(self):
        return object()

    def capabilities(self):
        return {"agent_enabled": True, "authorized_repositories": [{"repo_id": "repo"}]}

    def create_agent_session(self, **_kwargs):
        return AgentTurnResult(session_id="remote-1", payload={})

    def run_agent_turn(self, **kwargs):
        self.last_agent_kwargs = kwargs
        return {"content": "agent answer"}

    def chat(self, **_kwargs):
        if self.fail_chat:
            raise LaplaceWebError("fake_backend_failure")
        return {"content": "chat answer"}

    def cancel_agent(self, _session_id):
        return {"status": "CANCELLED"}


def controller(tmp_path: Path, client: WebClient) -> WebController:
    repo = tmp_path / "repo"
    repo.mkdir()
    return WebController(
        repository_root=repo,
        repo_id="repo",
        state_root=tmp_path / "state",
        operator_url="http://127.0.0.1:8765",
        client_factory=lambda _url: client,
    )


def test_web_read_preserves_verifier_but_disables_mutation(tmp_path: Path) -> None:
    client = WebClient()
    web = controller(tmp_path, client)
    history, session_id, _, _ = web.respond(
        "hello",
        [],
        "",
        "chat",
        "read",
        "quality",
        "pytest tests/test_task_labels.py",
    )
    web.respond("inspect", history, session_id, "agent", "read", "quality", "")
    assert client.last_agent_kwargs["verification_argv"] == (
        "pytest",
        "tests/test_task_labels.py",
    )
    assert client.last_agent_kwargs["allow_mutation"] is False


def test_web_write_requires_verifier_and_enables_mutation(tmp_path: Path) -> None:
    client = WebClient()
    web = controller(tmp_path, client)
    with pytest.raises(LaplaceWebError, match="write_access_requires_verification"):
        web.respond("edit", [], "", "agent", "write", "quality", "")
    web.respond(
        "edit",
        [],
        "",
        "agent",
        "write",
        "quality",
        "pytest tests/test_task_labels.py",
    )
    assert client.last_agent_kwargs["allow_mutation"] is True


def test_web_failed_turn_is_not_persisted(tmp_path: Path) -> None:
    client = WebClient()
    web = controller(tmp_path, client)
    history, session_id, _, _ = web.respond(
        "first", [], "", "chat", "read", "quality", ""
    )
    client.fail_chat = True
    with pytest.raises(LaplaceWebError, match="fake_backend_failure"):
        web.respond("failed", history, session_id, "chat", "read", "quality", "")
    session = web.store.load(
        session_id,
        repo_id="repo",
        repository_root=str(web.repository_root),
    )
    assert [(message.role, message.content) for message in session.messages] == [
        ("user", "first"),
        ("assistant", "chat answer"),
    ]


def completed(argv):
    return subprocess.CompletedProcess(
        argv,
        0,
        '{"name":"zetsu","transport":"stdio"}',
        "",
    )


def test_codex_status_accepts_repo(monkeypatch, capsys, tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    value = status(
        repository=repository,
        runner=completed,
        codex_executable="/usr/bin/codex",
    )
    assert value["repository"] == str(repository.resolve())

    observed = {}

    def fake_status(*, repository=None, **_kwargs):
        observed["repository"] = repository
        return {
            "status": "CONFIGURED",
            "repository": str(repository.resolve()),
            "codex": '{"name":"zetsu","transport":"stdio"}',
        }

    monkeypatch.setattr(codex_module, "status", fake_status)
    assert codex_module.main(["status", "--repo", str(repository)]) == 0
    assert observed["repository"] == repository
    assert json.loads(capsys.readouterr().out)["repository"] == str(
        repository.resolve()
    )
