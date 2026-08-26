from pathlib import Path

import pytest

from research_workspace.chat_cli import ChatCLIError, ChatShell, sanitize_terminal
from research_workspace.chat_session import ChatSessionStore


class Client:
    def __init__(self):
        self.model_calls = 0
        self.status_calls = 0
        self.cancel_calls = 0
        self.create_calls = 0
        self.turn_sessions = []

    def contract_check(self):
        class C:
            openapi_version = "2"
            base_url = "http://127.0.0.1:8765"
            agent_turn_path = "/api/v1/agent/sessions/{session_id}/messages"
        return C()

    def agent_status(self, session_id):
        self.status_calls += 1
        return {
            "session_id": session_id,
            "status": "SUCCEEDED",
            "diff": "D",
            "verification": {"status": "PASS"},
        }

    def status_view(self, payload, key):
        if key == "diff":
            return payload["diff"]
        if key == "tests":
            return payload["verification"]
        return payload

    def cancel_agent(self, session_id):
        self.cancel_calls += 1
        return {"session_id": session_id, "status": "CANCELLED"}

    def capabilities(self):
        return {
            "agent_enabled": True,
            "authorized_repositories": [{"repo_id": "repo-a"}],
        }

    def create_agent_session(self, *, repo_id, requested_session_id, task_title):
        del repo_id, task_title
        self.create_calls += 1

        class Created:
            session_id = requested_session_id

        return Created()

    def run_agent_turn(self, *, session_id, instruction, **_kwargs):
        self.model_calls += 1
        self.turn_sessions.append(session_id)
        return {"session_id": session_id, "content": f"done: {instruction}"}

    def agent_messages(self, session_id):
        return {
            "conversation": {
                "agent_session_id": session_id,
                "repo_id": "repo-a",
                "messages": [],
            }
        }


def shell(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    store = ChatSessionStore(tmp_path / "state")
    session = store.create(
        repo_id="repo-a",
        repository_root=str(repo),
        lane="quality",
        domain="software_engineering",
        interaction_mode="agent",
        access_mode="confirm",
    )
    session = store.update(session, remote_agent_session_id="remote-a")
    client = Client()
    return ChatShell(
        client=client,  # type: ignore[arg-type]
        store=store,
        session=session,
        max_steps=None,
        max_chars=None,
        wait_timeout_seconds=None,
        verification_argv=None,
    ), client


def test_monitoring_commands_make_no_model_calls(tmp_path: Path, capsys):
    chat, client = shell(tmp_path)
    assert chat.command("/status")
    assert chat.command("/diff")
    assert chat.command("/tests")
    assert client.status_calls == 3
    assert client.model_calls == 0


def test_terminal_escape_sequences_are_removed():
    assert sanitize_terminal("\x1b[31mBAD\x1b[0m\x07") == "BAD"


def test_natural_language_turns_reuse_one_remote_agent_session(tmp_path: Path):
    chat, client = shell(tmp_path)
    chat.session = chat.store.update(chat.session, remote_agent_session_id=None)
    chat._agent_turn("inspect component")
    chat._agent_turn("follow up on that component")
    assert client.create_calls == 1
    assert len(set(client.turn_sessions)) == 1
    assert client.turn_sessions == [
        chat.session.remote_agent_session_id,
        chat.session.remote_agent_session_id,
    ]


def test_mutating_turn_confirmation_fails_closed_before_remote_call(
    tmp_path: Path, monkeypatch
):
    chat, client = shell(tmp_path)
    chat.verification_argv = ("pytest", "-q")
    monkeypatch.setattr("builtins.input", lambda _prompt: "no")
    with pytest.raises(ChatCLIError, match="agent_turn_cancelled"):
        chat._agent_turn("change component")
    assert client.create_calls == 0
    assert client.model_calls == 0


def test_async_agent_turn_uses_events_instead_of_waiting_for_sync_http(tmp_path: Path):
    class AsyncClient(Client):
        def submit_agent_turn(self, *, session_id, turn_id, **_kwargs):
            self.turn_sessions.append(session_id)

            class Submitted:
                event_cursor = 0

            return Submitted()

        def agent_events(self, session_id, *, after_sequence):
            assert session_id == "remote-a"
            assert after_sequence == 0
            return [
                {
                    "sequence": 1,
                    "event": "TURN_COMPLETED",
                    "details": {"turn_id": self.turn_id},
                }
            ]

        def agent_messages(self, session_id):
            return {
                "conversation": {
                    "agent_session_id": session_id,
                    "repo_id": "repo-a",
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "completed without a long HTTP wait",
                            "metadata": {"turn_id": self.turn_id},
                        }
                    ],
                }
            }

        def run_agent_turn(self, **_kwargs):
            raise AssertionError("synchronous /messages fallback must not run")

    chat, _client = shell(tmp_path)
    client = AsyncClient()
    chat.client = client  # type: ignore[assignment]
    original = client.submit_agent_turn

    def submit_with_turn_id(*, turn_id, **kwargs):
        client.turn_id = turn_id
        return original(turn_id=turn_id, **kwargs)

    client.submit_agent_turn = submit_with_turn_id  # type: ignore[method-assign]
    chat._agent_turn("inspect component")
    assert client.model_calls == 0
    assert chat.session.active_turn_id is None
    assert chat.session.messages[-1].content == "completed without a long HTTP wait"
