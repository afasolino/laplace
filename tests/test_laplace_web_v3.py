from pathlib import Path

import pytest

from research_workspace.chat_operator_client import AgentTurnResult
from research_workspace.laplace_web import LaplaceWebError, WebController, _verification


class FakeClient:
    def __init__(self) -> None:
        self.agent_calls = 0
        self.chat_calls = 0
        self.cancel_calls = 0

    def contract_check(self):  # type: ignore[no-untyped-def]
        return object()

    def capabilities(self):  # type: ignore[no-untyped-def]
        return {
            "agent_enabled": True,
            "capability_tier": "owner",
            "capabilities": ["agent", "chat"],
            "model_lanes": ["quality"],
            "authorized_repositories": [{"repo_id": "repo"}],
        }

    def create_agent_session(self, **_kwargs):  # type: ignore[no-untyped-def]
        return AgentTurnResult(session_id="remote-1", payload={})

    def run_agent_turn(self, **_kwargs):  # type: ignore[no-untyped-def]
        self.agent_calls += 1
        return {"content": "agent answer"}

    def chat(self, **_kwargs):  # type: ignore[no-untyped-def]
        self.chat_calls += 1
        return {"content": "chat answer"}

    def cancel_agent(self, _session_id: str):  # type: ignore[no-untyped-def]
        self.cancel_calls += 1
        return {"status": "CANCELLED"}


def controller(tmp_path: Path, client: FakeClient) -> WebController:
    repo = tmp_path / "repo"
    repo.mkdir()
    return WebController(
        repository_root=repo,
        repo_id="repo",
        state_root=tmp_path / "state",
        operator_url="http://127.0.0.1:8765",
        client_factory=lambda _url: client,  # type: ignore[arg-type]
    )


def test_web_chat_and_agent_share_operator_client(tmp_path: Path) -> None:
    fake = FakeClient()
    web = controller(tmp_path, fake)
    history, session_id, cleared, status = web.respond(
        "hello",
        [],
        "",
        "chat",
        "read",
        "quality",
        "",
    )
    assert history[-1] == {"role": "assistant", "content": "chat answer"}
    assert cleared == ""
    assert session_id
    assert "mode=chat" in status
    assert fake.chat_calls == 1

    history, session_id, _, _ = web.respond(
        "inspect only",
        history,
        session_id,
        "agent",
        "read",
        "quality",
        "",
    )
    assert history[-1]["content"] == "agent answer"
    assert fake.agent_calls == 1


def test_web_write_requires_deterministic_verifier(tmp_path: Path) -> None:
    web = controller(tmp_path, FakeClient())
    with pytest.raises(LaplaceWebError, match="write_access_requires_verification"):
        web.respond("edit", [], "", "agent", "write", "quality", "")


def test_web_verifier_is_shlex_argv_not_shell() -> None:
    assert _verification("pytest 'tests/test one.py'") == ("pytest", "tests/test one.py")


def test_gradio6_app_constructs_without_launching(tmp_path: Path) -> None:
    from research_workspace.laplace_web import build_app

    web = controller(tmp_path, FakeClient())
    demo = build_app(web)
    assert hasattr(demo, "launch")
    assert hasattr(demo, "queue")


def test_web_main_handles_keyboard_interrupt_without_traceback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import research_workspace.laplace_web as web_module

    repo = tmp_path / "repo"
    repo.mkdir()

    class FakeController:
        def _client(self) -> object:
            return object()

    class FakeApp:
        def queue(self, **_kwargs: object) -> "FakeApp":
            return self

        def launch(self, **_kwargs: object) -> None:
            raise KeyboardInterrupt

    monkeypatch.setattr(web_module, "_git_root", lambda _cwd: repo)
    monkeypatch.setattr(web_module, "WebController", lambda **_kwargs: FakeController())
    monkeypatch.setattr(web_module, "build_app", lambda _controller: FakeApp())

    assert web_module.main(["--repo-id", "repo", "--no-browser"]) == 130
