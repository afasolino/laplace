from __future__ import annotations

from pathlib import Path

from research_workspace.chat_cli import ChatShell
from research_workspace.chat_session import ChatSessionStore
from research_workspace.laplace_web import WebController


class FakeClient:
    def __init__(self) -> None:
        self.chat_calls: list[dict[str, object]] = []

    def contract_check(self) -> object:
        return object()

    def chat(self, **kwargs: object) -> dict[str, object]:
        self.chat_calls.append(dict(kwargs))
        return {"content": "ok"}

    def capabilities(self) -> dict[str, object]:
        return {
            "agent_enabled": True,
            "authorized_repositories": [{"repo_id": "repo-a"}],
        }


def _shell(tmp_path: Path, client: FakeClient) -> ChatShell:
    repo = tmp_path / "repo"
    repo.mkdir()
    store = ChatSessionStore(tmp_path / "state")
    session = store.create(
        repo_id="repo-a",
        repository_root=str(repo),
        lane="quality",
        domain="software_engineering",
        interaction_mode="agent",
        access_mode="read",
    )
    return ChatShell(
        client=client,  # type: ignore[arg-type]
        store=store,
        session=session,
        max_steps=None,
        max_chars=None,
        wait_timeout_seconds=None,
        verification_argv=None,
        route_mode="auto",
    )


def test_cli_chat_and_retrieval_do_not_forward_legacy_agent_domain(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    shell = _shell(tmp_path, client)

    shell._dispatch_turn("What is 2 + 2?")
    shell._dispatch_turn("Search my corpus for HBM4")

    assert [call["domain"] for call in client.chat_calls] == ["general", "general"]
    assert client.chat_calls[0].get("retrieval_selection") is None
    assert client.chat_calls[1]["retrieval_selection"] == "personal"


def test_web_chat_and_retrieval_do_not_forward_legacy_agent_domain(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    repo = tmp_path / "repo-web"
    repo.mkdir()
    controller = WebController(
        repository_root=repo,
        repo_id="repo-a",
        state_root=tmp_path / "web-state",
        operator_url="http://127.0.0.1:8765",
        client_factory=lambda _url: client,  # type: ignore[arg-type]
    )

    history, session_id, _, _ = controller.respond(
        "What is 2 + 2?",
        [],
        "",
        "auto",
        "read",
        "quality",
        "",
    )
    controller.respond(
        "Search my corpus for HBM4",
        history,
        session_id,
        "auto",
        "read",
        "quality",
        "",
    )

    assert [call["domain"] for call in client.chat_calls] == ["general", "general"]
    assert client.chat_calls[0].get("retrieval_selection") is None
    assert client.chat_calls[1]["retrieval_selection"] == "personal"
