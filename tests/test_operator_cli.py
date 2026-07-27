from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_workspace.operator_cli import main


ROOT = Path(__file__).resolve().parents[1]


def test_operator_status_is_machine_readable_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "--repository-root",
                str(ROOT),
                "--state-root",
                str(tmp_path),
                "status",
                "--json",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "OK"
    assert output["local_only"] is True
    assert output["event_count"] == 0


def test_operator_cli_returns_structured_authorization_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "run.json"
    config.write_text(
        json.dumps(
            {
                "task_id": "rtl",
                "arm_id": "arm_c",
                "model_route": "main+codev",
                "corpus_snapshot_sha256": "1" * 64,
                "skills_lock_sha256": "2" * 64,
                "smoke_profile": "fixture",
                "request_sha256": "3" * 64,
                "gpu_required": False,
            }
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "--repository-root",
                str(ROOT),
                "--state-root",
                str(tmp_path / "state"),
                "run",
                "prepare",
                "--config",
                str(config),
                "--json",
            ]
        )
        == 2
    )
    output = json.loads(capsys.readouterr().out)
    assert output["failure_category"] == "authorization_failure"
