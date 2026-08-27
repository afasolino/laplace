from __future__ import annotations

import os
from pathlib import Path

import pytest

from research_workspace.chat_cli import ChatCLIError, ChatShell
from research_workspace.chat_discovery import command_names, render_capabilities, render_skills
from research_workspace.chat_input import _history_path, build_chat_input
from research_workspace.chat_session import ChatSessionStore
from research_workspace.chat_verification import (
    ChatVerificationError,
    ChatVerificationStore,
    resolve_verification,
)


class _Client:
    def __init__(self) -> None:
        self.capability_calls = 0
        self.model_calls = 0

    def contract_check(self) -> object:
        class _Contract:
            openapi_version = "2"
            base_url = "http://127.0.0.1:8765"
            agent_turn_path = "/api/v1/agent/sessions/{session_id}/messages"

        return _Contract()

    def capabilities(self) -> dict[str, object]:
        self.capability_calls += 1
        return {
            "agent_enabled": True,
            "capability_tier": "plus",
            "capabilities": ["agent", "chat", "personal_corpus"],
            "model_lanes": ["quality", "standard", "economy"],
            "authorized_repositories": [{"repo_id": "repo-a"}],
        }

    def agent_messages(self, session_id: str) -> dict[str, object]:
        return {
            "conversation": {
                "agent_session_id": session_id,
                "repo_id": "repo-a",
                "messages": [],
            }
        }


@pytest.fixture
def sessions(tmp_path: Path) -> tuple[ChatSessionStore, object]:
    store = ChatSessionStore(tmp_path / "sessions")
    session = store.create(
        repo_id="repo-a",
        repository_root=str(tmp_path),
        lane="quality",
        domain="software_engineering",
        interaction_mode="agent",
        access_mode="confirm",
    )
    return store, session


def _shell(
    tmp_path: Path,
    store: ChatSessionStore,
    session: object,
    *,
    verification: tuple[str, ...] | None = None,
) -> ChatShell:
    return ChatShell(
        client=_Client(),  # type: ignore[arg-type]
        store=store,
        session=session,  # type: ignore[arg-type]
        max_steps=None,
        max_chars=None,
        wait_timeout_seconds=None,
        verification_argv=verification,
        verification_store=ChatVerificationStore(tmp_path / "verifiers"),
        input_reader=lambda: "",
    )


def test_verification_contract_roundtrip_restore_and_conflict(tmp_path: Path) -> None:
    store = ChatVerificationStore(tmp_path / "verifiers")
    argv = ("pytest", "tests/test_task_labels.py")

    assert resolve_verification(store, "chat-a", argv) == argv
    assert resolve_verification(store, "chat-a", None) == argv
    with pytest.raises(ChatVerificationError, match="resume_verification_conflict"):
        resolve_verification(store, "chat-a", ("pytest", "tests/test_other.py"))

    mode = store._path("chat-a").stat().st_mode & 0o777
    assert mode == 0o600


def test_help_completion_and_capability_discovery_are_deterministic() -> None:
    commands = command_names()
    assert "/help" in commands
    assert "/skills" in commands
    assert "/capabilities" in commands
    assert "/verification" in commands

    packet = {
        "capability_tier": "plus",
        "capabilities": ["chat", "agent", "personal_corpus"],
        "model_lanes": ["quality", "standard"],
        "authorized_repositories": [{"repo_id": "repo-a"}],
    }
    rendered = render_capabilities(packet)
    assert "capabilities=agent, chat, personal_corpus" in rendered
    assert "authorized_repositories=repo-a" in rendered
    skills = render_skills(packet)
    assert "repository-agent" in skills
    assert "retrieval" in skills
    assert "model-admin" not in skills


def test_prompt_toolkit_history_is_private_and_reader_is_constructible(tmp_path: Path) -> None:
    history = _history_path(tmp_path, "repo/a")
    assert history.parent.name == "chat_history_v3"
    assert history.name == "repo_a.history"
    if os.name != "nt":
        assert history.stat().st_mode & 0o777 == 0o600
        assert history.parent.stat().st_mode & 0o777 == 0o700
    assert callable(build_chat_input(tmp_path, "repo/a"))



def test_multiline_reader_value_is_one_logical_turn(
    tmp_path: Path, sessions: tuple[ChatSessionStore, object]
) -> None:
    store, session = sessions
    values = iter(("first line\nsecond line\nthird line", "/exit"))
    shell = ChatShell(
        client=_Client(),  # type: ignore[arg-type]
        store=store,
        session=session,  # type: ignore[arg-type]
        max_steps=None,
        max_chars=None,
        wait_timeout_seconds=None,
        verification_argv=None,
        verification_store=ChatVerificationStore(tmp_path / "verifiers"),
        input_reader=lambda: next(values),
    )
    observed: list[str] = []
    shell._agent_turn = observed.append  # type: ignore[method-assign]

    assert shell.run() == 0
    assert observed == ["first line\nsecond line\nthird line"]


def test_capabilities_and_verification_commands_are_control_plane_only(
    tmp_path: Path, sessions: tuple[ChatSessionStore, object], capsys: pytest.CaptureFixture[str]
) -> None:
    store, session = sessions
    shell = _shell(
        tmp_path,
        store,
        session,
        verification=("pytest", "tests/test_task_labels.py"),
    )
    client = shell.client

    assert shell.command("/capabilities") is True
    assert shell.command("/verification") is True
    output = capsys.readouterr().out
    assert "capabilities=agent, chat, personal_corpus" in output
    assert "verification=pytest tests/test_task_labels.py" in output
    assert client.capability_calls == 1
    assert client.model_calls == 0

def test_write_access_fails_early_without_verifier(
    tmp_path: Path, sessions: tuple[ChatSessionStore, object]
) -> None:
    store, session = sessions
    shell = _shell(tmp_path, store, session)

    with pytest.raises(ChatCLIError, match="write_access_requires_verification"):
        shell.command("/access write")
    assert shell.session.access_mode == "confirm"


def test_skills_is_control_plane_only(
    tmp_path: Path, sessions: tuple[ChatSessionStore, object], capsys: pytest.CaptureFixture[str]
) -> None:
    store, session = sessions
    shell = _shell(tmp_path, store, session)
    client = shell.client

    assert shell.command("/skills") is True
    output = capsys.readouterr().out
    assert "repository-agent" in output
    assert client.capability_calls == 1
    assert client.model_calls == 0


def test_in_shell_resume_restores_persisted_verifier(
    tmp_path: Path, sessions: tuple[ChatSessionStore, object]
) -> None:
    store, current = sessions
    target = store.create(
        repo_id="repo-a",
        repository_root=str(tmp_path),
        lane="quality",
        domain="software_engineering",
        interaction_mode="agent",
        access_mode="write",
    )
    verifier_store = ChatVerificationStore(tmp_path / "verifiers")
    verifier_store.save(target.session_id, ("pytest", "tests/test_task_labels.py"))
    shell = ChatShell(
        client=_Client(),  # type: ignore[arg-type]
        store=store,
        session=current,  # type: ignore[arg-type]
        max_steps=None,
        max_chars=None,
        wait_timeout_seconds=None,
        verification_argv=None,
        verification_store=verifier_store,
        input_reader=lambda: "",
    )

    assert shell.command(f"/resume {target.session_id}") is True
    assert shell.session.session_id == target.session_id
    assert shell.verification_argv == ("pytest", "tests/test_task_labels.py")
