from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from research_workspace import zetsu_cli


def test_zetsu_cli_offline_lifecycle(tmp_path: Path, capsys, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))

    assert (
        zetsu_cli.main(
            ["configure", "--repo", str(repo), "--endpoint", "https://laplace.example/mcp", "--json"]
        )
        == 0
    )
    configured = json.loads(capsys.readouterr().out)
    assert configured["configured"] is True

    assert zetsu_cli.main(["test", "--repo", str(repo), "--offline", "--json"]) == 0
    tested = json.loads(capsys.readouterr().out)
    assert tested["ok"] is True
    assert tested["detail"] == "offline"

    assert zetsu_cli.main(["remove", "--repo", str(repo), "--json"]) == 0
    removed = json.loads(capsys.readouterr().out)
    assert removed["configured"] is False


def test_bare_zetsu_configures_then_runs_status_and_test(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setenv("LAPLACE_ZETSU_TOKEN", "local-test-token")
    retrieval_calls: list[bool] = []

    def online_probe(
        endpoint: str,
        token_env_var: str,
        timeout: float,
        *,
        retrieval: bool,
    ) -> dict[str, object]:
        assert endpoint == "http://127.0.0.1:8765/mcp"
        assert token_env_var == "LAPLACE_ZETSU_TOKEN"
        assert timeout == 10.0
        retrieval_calls.append(retrieval)
        return {"reachable": True, "retrieval_check": retrieval}

    monkeypatch.setattr(zetsu_cli, "_online_probe", online_probe)
    monkeypatch.setattr(zetsu_cli, "_laplace_status", lambda *_: {"status": "READY"})
    monkeypatch.setattr(zetsu_cli, "_laplace_readiness", lambda *_: {"status": "READY"})
    monkeypatch.setattr(
        zetsu_cli,
        "_codex_recognition",
        lambda _: {"recognized": True},
    )

    assert zetsu_cli.main([]) == 2
    output = capsys.readouterr().out
    assert "action: ensure" in output
    assert "configured_now: True" in output
    assert "repository:authenticated_principal_unavailable" in output
    assert retrieval_calls == [False, True]
    assert (repo / ".agents/skills/zetsu/SKILL.md").is_file()


def test_start_configures_and_delegates_to_owned_runtime(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    state = tmp_path / "state"
    repo.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    observed: dict[str, object] = {}

    def start_runtime(
        repository: Path,
        state_root: Path,
        *,
        timeout: float,
        dry_run: bool,
        vllm: Path | None,
        ffmpeg_lib: Path | None,
    ) -> dict[str, object]:
        observed.update(
            {
                "repository": repository,
                "state_root": state_root,
                "timeout": timeout,
                "dry_run": dry_run,
                "vllm": vllm,
                "ffmpeg_lib": ffmpeg_lib,
            }
        )
        return {"status": "DRY_RUN"}

    monkeypatch.setattr(zetsu_cli, "start_local_runtime", start_runtime)
    assert (
        zetsu_cli.main(
            [
                "start",
                "--repo",
                str(repo),
                "--state-root",
                str(state),
                "--dry-run",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["runtime"]["status"] == "DRY_RUN"
    assert observed == {
        "repository": repo.resolve(),
        "state_root": state,
        "timeout": 1_800.0,
        "dry_run": True,
        "vllm": None,
        "ffmpeg_lib": None,
    }


def test_start_nocodev_passes_persisted_topology_choice(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    state = tmp_path / "state"
    repo.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    observed: dict[str, object] = {}

    def start_runtime(repository: Path, state_root: Path, **kwargs: object) -> dict[str, object]:
        observed.update(repository=repository, state_root=state_root, **kwargs)
        return {
            "status": "DRY_RUN",
            "topology": "nocodev",
            "codev": "intentionally_disabled",
        }

    monkeypatch.setattr(zetsu_cli, "start_local_runtime", start_runtime)
    assert zetsu_cli.main(
        [
            "start",
            "--repo",
            str(repo),
            "--state-root",
            str(state),
            "--nocodev",
            "--dry-run",
            "--json",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["runtime"]["topology"] == "nocodev"
    assert observed["codev_enabled"] is False


def test_worktrees_and_gc_cli_are_bounded_and_dry_run_by_default_only_when_requested(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    class Manager:
        def operator_inventory(self) -> list[dict[str, object]]:
            return [{"session_id": "terminal", "physical_state": "PRESENT"}]

        def collect_garbage(self, *, dry_run: bool, limit: int) -> dict[str, object]:
            return {
                "status": "DRY_RUN" if dry_run else "COMPLETE",
                "examined": 1,
                "released": 0 if dry_run else 1,
                "protected": 0,
                "items": [
                    {
                        "session_id": "terminal",
                        "action": "WOULD_RELEASE" if dry_run else "RELEASED",
                    }
                ],
                "limit": limit,
            }

    monkeypatch.setattr(zetsu_cli, "_worktree_manager", lambda _state: Manager())
    assert zetsu_cli.main(["worktrees", "--state-root", str(tmp_path), "--json"]) == 0
    inventory = json.loads(capsys.readouterr().out)
    assert inventory["worktrees"][0]["session_id"] == "terminal"
    assert zetsu_cli.main(
        ["gc", "--state-root", str(tmp_path), "--dry-run", "--limit", "7", "--json"]
    ) == 0
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run["status"] == "DRY_RUN"
    assert dry_run["items"][0]["action"] == "WOULD_RELEASE"
    assert zetsu_cli.main(["gc", "--state-root", str(tmp_path), "--json"]) == 0
    collected = json.loads(capsys.readouterr().out)
    assert collected["status"] == "COMPLETE"
    assert collected["items"][0]["action"] == "RELEASED"


def test_status_fails_closed_on_degraded_operator_readiness(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setenv("LAPLACE_ZETSU_TOKEN", "local-test-token")
    assert zetsu_cli.main(["configure", "--repo", str(repo), "--json"]) == 0
    capsys.readouterr()
    monkeypatch.setattr(
        zetsu_cli,
        "_online_probe",
        lambda *_, **__: {"reachable": True},
    )
    monkeypatch.setattr(zetsu_cli, "_laplace_status", lambda *_: {"status": "READY"})
    monkeypatch.setattr(
        zetsu_cli,
        "_laplace_readiness",
        lambda *_: {
            "status": "DEGRADED",
            "reasons": ["model_endpoint_unavailable:economy"],
        },
    )
    monkeypatch.setattr(zetsu_cli, "_codex_recognition", lambda _: {"recognized": True})

    assert zetsu_cli.main(["status", "--repo", str(repo), "--json"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["detail"] == (
        "laplace_readiness_degraded:model_endpoint_unavailable:economy"
    )


def test_codex_launch_injects_protected_token_only_into_child(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    state = tmp_path / "state"
    repo.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    token_path = state / "auth/bearer_tokens.json"
    token_path.parent.mkdir(parents=True)
    token_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "child-only-secret": {
                        "role": "read",
                        "user_id": "plus-local",
                        "capability_tier": "plus",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    os.chmod(token_path, 0o600)
    monkeypatch.setattr(
        zetsu_cli,
        "_ensure_payload",
        lambda *_, **__: {"ok": True},
    )
    monkeypatch.setattr(zetsu_cli.shutil, "which", lambda _: "/usr/bin/codex")

    class Launched(Exception):
        pass

    observed: dict[str, object] = {}

    def execvpe(executable: str, argv: list[str], environment: dict[str, str]) -> None:
        observed.update(executable=executable, argv=argv, environment=environment)
        raise Launched

    monkeypatch.setattr(zetsu_cli.os, "execvpe", execvpe)
    with pytest.raises(Launched):
        zetsu_cli.main(
            [
                "codex",
                "--repo",
                str(repo),
                "--state-root",
                str(state),
                "--",
                "--version",
            ]
        )

    assert observed["executable"] == "/usr/bin/codex"
    assert observed["argv"] == ["/usr/bin/codex", "--version"]
    environment = observed["environment"]
    assert isinstance(environment, dict)
    assert environment["LAPLACE_ZETSU_TOKEN"] == "child-only-secret"
    assert "child-only-secret" not in capsys.readouterr().out
