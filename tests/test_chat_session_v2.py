from pathlib import Path

import pytest

from research_workspace.chat_session import ChatSessionError, ChatSessionStore


def test_session_resume_is_repository_bound(tmp_path: Path):
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
    loaded = store.load(
        session.session_id,
        repo_id="repo-a",
        repository_root=str(repo),
    )
    assert loaded.remote_agent_session_id == "remote-a"
    with pytest.raises(ChatSessionError, match="repository_mismatch"):
        store.load(
            session.session_id,
            repo_id="repo-b",
            repository_root=str(repo),
        )


def test_chat_history_persists(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    store = ChatSessionStore(tmp_path / "state")
    session = store.create(
        repo_id="repo-a",
        repository_root=str(repo),
        lane="quality",
        domain="general",
        interaction_mode="chat",
        access_mode="read",
    )
    session = store.append_message(session, role="user", content="hello")
    session = store.append_message(session, role="assistant", content="world")
    resumed = store.last(repo_id="repo-a", repository_root=str(repo))
    assert resumed is not None
    assert [(m.role, m.content) for m in resumed.messages] == [
        ("user", "hello"),
        ("assistant", "world"),
    ]


def test_session_updates_validate_before_persistence(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    store = ChatSessionStore(tmp_path / "state")
    session = store.create(
        repo_id="repo-a",
        repository_root=str(repo),
        lane="quality",
        domain="general",
        interaction_mode="chat",
        access_mode="read",
    )
    with pytest.raises(ChatSessionError, match="lane_invalid"):
        store.update(session, lane="unknown")
    assert store.load(
        session.session_id,
        repo_id="repo-a",
        repository_root=str(repo),
    ).lane == "quality"
