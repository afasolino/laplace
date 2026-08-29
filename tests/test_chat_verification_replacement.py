from research_workspace.chat_verification import ChatVerificationStore, resolve_verification


def test_explicit_verifier_can_replace_persisted_verifier(tmp_path) -> None:
    store = ChatVerificationStore(tmp_path)
    assert resolve_verification(store, "session-1", ("pytest", "-q")) == ("pytest", "-q")
    assert resolve_verification(store, "session-1", ("ruff", "check", "src")) == ("ruff", "check", "src")
    assert store.load("session-1") == ("ruff", "check", "src")
